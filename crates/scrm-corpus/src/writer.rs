use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, DictionaryArray, Int32Array, Int32Builder, ListBuilder, RecordBatch, StringArray, UInt8Builder,
    UInt16Builder, UInt32Builder,
};
use arrow::datatypes::{Int32Type, SchemaRef};
use arrow_ipc::writer::{DictionaryHandling, FileWriter, IpcWriteOptions};
use scrm_core::MoleculeContigs;
use scrm_core::stamp::Declared;
use sha2::{Digest, Sha256};

use crate::{
    CellRun, CorpusError, META_CELL_ROWS, META_N_ROWS, META_N_TOKENS, META_TOKENS_SHA256, Result, check_row,
    corpus_schema, recount_scratch, token_item_field,
};

pub const BATCH_ROWS: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusSummary {
    pub n_rows: usize,
    pub n_tokens: u64,
    pub tokens_sha256: String,
    pub n_cells: usize,
    pub first_cell: String,
    pub last_cell: String,
}

struct CurrentCell {
    name: String,
    packed: u64,
    key: i32,
}

pub struct CorpusWriter<W: Write + Seek> {
    inner: FileWriter<W>,
    schema: SchemaRef,

    cell_dict: Vec<String>,

    cell_keys: Int32Builder,
    n_reads: UInt32Builder,
    n_contigs: UInt16Builder,
    tokens: ListBuilder<UInt8Builder>,
    rows_in_batch: usize,

    n_rows: usize,
    n_tokens: u64,
    hasher: Sha256,

    runs: Vec<CellRun>,
    current: Option<CurrentCell>,
    seen_cells: HashSet<String>,
    seen_packed: HashMap<u64, String>,
    scratch: MoleculeContigs,
}

impl CorpusWriter<BufWriter<File>> {
    pub fn create(path: &Path, declared: &Declared) -> Result<Self> {
        let file = File::create(path)?;
        Self::new(BufWriter::new(file), declared)
    }
}

impl<W: Write + Seek> CorpusWriter<W> {
    pub fn new(sink: W, declared: &Declared) -> Result<Self> {
        let declared_json = declared.to_json().map_err(|source| CorpusError::Json {
            key: crate::META_DECLARED,
            source,
        })?;
        let schema: SchemaRef = Arc::new(corpus_schema(declared_json));
        let options = IpcWriteOptions::default().with_dictionary_handling(DictionaryHandling::Delta);
        let inner = FileWriter::try_new_with_options(sink, &schema, options)?;
        Ok(Self {
            inner,
            schema,
            cell_dict: Vec::new(),
            cell_keys: Int32Builder::new(),
            n_reads: UInt32Builder::new(),
            n_contigs: UInt16Builder::new(),
            tokens: ListBuilder::new(UInt8Builder::new()).with_field(Arc::new(token_item_field())),
            rows_in_batch: 0,
            n_rows: 0,
            n_tokens: 0,
            hasher: Sha256::new(),
            runs: Vec::new(),
            current: None,
            seen_cells: HashSet::new(),
            seen_packed: HashMap::new(),
            scratch: recount_scratch(),
        })
    }

    pub fn push(&mut self, cell_id: &str, mc: &MoleculeContigs) -> Result<()> {
        self.push_row(cell_id, mc.cell.0, mc.n_reads, mc.n_contigs, &mc.tokens)
    }

    pub fn push_row(&mut self, cell_id: &str, packed: u64, n_reads: u32, n_contigs: u16, tokens: &[u8]) -> Result<()> {
        check_row(tokens, n_contigs, self.n_rows, &mut self.scratch)?;

        let cell_key = self.cell_key(cell_id, packed)?;

        self.cell_keys.append_value(cell_key);
        self.n_reads.append_value(n_reads);
        self.n_contigs.append_value(n_contigs);
        self.tokens.values().append_slice(tokens);
        self.tokens.append(true);

        self.hasher.update(tokens);
        self.n_tokens += tokens.len() as u64;
        self.n_rows += 1;
        self.rows_in_batch += 1;
        if let Some(run) = self.runs.last_mut() {
            run.len += 1;
        }

        if self.rows_in_batch >= BATCH_ROWS {
            self.flush_batch()?;
        }
        Ok(())
    }

    fn cell_key(&mut self, cell_id: &str, packed: u64) -> Result<i32> {
        if let Some(cur) = &self.current {
            if cur.name == cell_id {
                if cur.packed != packed {
                    return Err(CorpusError::CellIdAliased {
                        first_cell: cell_id.to_string(),
                        first_packed: cur.packed,
                        second_cell: cell_id.to_string(),
                        second_packed: packed,
                    });
                }
                return Ok(cur.key);
            }
            if self.seen_cells.contains(cell_id) {
                return Err(scrm_core::Error::CellIdNotContiguous {
                    cell: cell_id.to_string(),
                    runs: 2,
                }
                .into());
            }
            if cell_id <= cur.name.as_str() {
                return Err(CorpusError::CellIdUnsorted {
                    prev: cur.name.clone(),
                    next: cell_id.to_string(),
                });
            }
        }
        if let Some(first_cell) = self.seen_packed.get(&packed) {
            return Err(CorpusError::CellIdAliased {
                first_cell: first_cell.clone(),
                first_packed: packed,
                second_cell: cell_id.to_string(),
                second_packed: packed,
            });
        }
        self.seen_packed.insert(packed, cell_id.to_string());
        let key = self.cell_dict.len() as i32;
        self.cell_dict.push(cell_id.to_string());
        self.seen_cells.insert(cell_id.to_string());
        self.runs.push(CellRun {
            cell: cell_id.to_string(),
            start: self.n_rows,
            len: 0,
        });
        self.current = Some(CurrentCell {
            name: cell_id.to_string(),
            packed,
            key,
        });
        Ok(key)
    }

    fn flush_batch(&mut self) -> Result<()> {
        if self.rows_in_batch == 0 {
            return Ok(());
        }
        let cells = dict_array(self.cell_keys.finish(), &self.cell_dict)?;
        let columns: Vec<ArrayRef> = vec![
            cells,
            Arc::new(self.n_reads.finish()),
            Arc::new(self.n_contigs.finish()),
            Arc::new(self.tokens.finish()),
        ];
        let batch = RecordBatch::try_new(self.schema.clone(), columns)?;
        self.inner.write(&batch)?;
        self.rows_in_batch = 0;
        Ok(())
    }

    pub fn finish(mut self) -> Result<CorpusSummary> {
        self.flush_batch()?;
        let sha = format!("{:x}", self.hasher.finalize_reset());
        let cell_rows = serde_json::to_string(&self.runs).map_err(|source| CorpusError::Json {
            key: META_CELL_ROWS,
            source,
        })?;
        self.inner.write_metadata(META_N_ROWS, self.n_rows.to_string());
        self.inner.write_metadata(META_N_TOKENS, self.n_tokens.to_string());
        self.inner.write_metadata(META_TOKENS_SHA256, &sha);
        self.inner.write_metadata(META_CELL_ROWS, cell_rows);
        self.inner.finish()?;
        self.inner.get_mut().flush()?;
        Ok(CorpusSummary {
            n_rows: self.n_rows,
            n_tokens: self.n_tokens,
            tokens_sha256: sha,
            n_cells: self.runs.len(),
            first_cell: self.runs.first().map(|r| r.cell.clone()).unwrap_or_default(),
            last_cell: self.runs.last().map(|r| r.cell.clone()).unwrap_or_default(),
        })
    }
}

fn dict_array(keys: Int32Array, values: &[String]) -> Result<ArrayRef> {
    let values = StringArray::from_iter_values(values.iter());
    let dict = DictionaryArray::<Int32Type>::try_new(keys, Arc::new(values))?;
    Ok(Arc::new(dict))
}
