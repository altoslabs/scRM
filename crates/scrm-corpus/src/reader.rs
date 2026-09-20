use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::ops::Range;
use std::path::Path;

use arrow::array::{Array, Int32Array, ListArray, RecordBatch, StringArray, UInt8Array, UInt16Array, UInt32Array};
use arrow::datatypes::Int32Type;
use arrow_ipc::reader::FileReader;
use scrm_core::MoleculeContigs;
use scrm_core::stamp::Declared;

use crate::{
    CellRun, CorpusError, META_CELL_ROWS, META_DECLARED, META_N_ROWS, META_N_TOKENS, META_TOKENS_SHA256, Result,
    check_row, corpus_fields, meta_str, meta_u64, recount_scratch,
};

pub struct CorpusReader {
    inner: FileReader<BufReader<File>>,
    declared: Declared,
    n_rows: usize,
    n_tokens: u64,
    tokens_sha256: String,
    runs: Vec<CellRun>,
    next_row: usize,
    scratch: MoleculeContigs,
}

impl CorpusReader {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let inner = FileReader::try_new_buffered(file, None)?;

        let schema = inner.schema();
        if schema.fields() != &corpus_fields() {
            return Err(CorpusError::SchemaMismatch {
                detail: format!("fields are {:?}", schema.fields()),
            });
        }
        let declared =
            Declared::from_json(meta_str(schema.metadata(), META_DECLARED)?).map_err(|source| CorpusError::Json {
                key: META_DECLARED,
                source,
            })?;

        let footer: &HashMap<String, String> = inner.custom_metadata();
        let n_rows = meta_u64(footer, META_N_ROWS)? as usize;
        let n_tokens = meta_u64(footer, META_N_TOKENS)?;
        let tokens_sha256 = meta_str(footer, META_TOKENS_SHA256)?.to_string();
        let runs: Vec<CellRun> =
            serde_json::from_str(meta_str(footer, META_CELL_ROWS)?).map_err(|source| CorpusError::Json {
                key: META_CELL_ROWS,
                source,
            })?;
        check_cell_index(&runs, n_rows)?;

        Ok(Self {
            inner,
            declared,
            n_rows,
            n_tokens,
            tokens_sha256,
            runs,
            next_row: 0,
            scratch: recount_scratch(),
        })
    }

    pub fn declared(&self) -> &Declared {
        &self.declared
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn n_tokens(&self) -> u64 {
        self.n_tokens
    }

    pub fn tokens_sha256(&self) -> &str {
        &self.tokens_sha256
    }

    pub fn cells(&self) -> &[CellRun] {
        &self.runs
    }

    pub fn cell_range(&self, cell_id: &str) -> Option<Range<usize>> {
        self.runs
            .binary_search_by(|run| run.cell.as_str().cmp(cell_id))
            .ok()
            .map(|i| self.runs[i].range())
    }

    pub fn next_batch(&mut self) -> Option<Result<CorpusBatch>> {
        let batch = self.inner.next()?;
        Some(self.decode(batch))
    }

    fn decode(&mut self, batch: std::result::Result<RecordBatch, arrow::error::ArrowError>) -> Result<CorpusBatch> {
        let batch = CorpusBatch::new(batch?, self.next_row, &mut self.scratch)?;
        self.next_row += batch.len();
        Ok(batch)
    }
}

fn check_cell_index(runs: &[CellRun], n_rows: usize) -> Result<()> {
    let mut at = 0usize;
    for (i, run) in runs.iter().enumerate() {
        if run.start != at {
            return Err(CorpusError::CellIndexMismatch {
                detail: format!("cell {:?} starts at {} but row {} is next", run.cell, run.start, at),
            });
        }
        if run.len == 0 {
            return Err(CorpusError::CellIndexMismatch {
                detail: format!("cell {:?} has an empty row range", run.cell),
            });
        }
        if i > 0 && runs[i - 1].cell >= run.cell {
            return Err(CorpusError::CellIdUnsorted {
                prev: runs[i - 1].cell.clone(),
                next: run.cell.clone(),
            });
        }
        at += run.len;
    }
    if at != n_rows {
        return Err(CorpusError::TallyMismatch {
            what: "cell index row",
            expected: n_rows as u64,
            found: at as u64,
        });
    }
    Ok(())
}

#[derive(Debug)]
pub struct CorpusBatch {
    cell_keys: Int32Array,
    cell_values: StringArray,
    n_reads: UInt32Array,
    n_contigs: UInt16Array,
    token_offsets: Vec<i32>,
    token_values: UInt8Array,
    first_row: usize,
}

impl CorpusBatch {
    fn new(batch: RecordBatch, first_row: usize, scratch: &mut MoleculeContigs) -> Result<Self> {
        let (cell_keys, cell_values) = dict_column(&batch, 0)?;
        let n_reads = primitive::<UInt32Array>(&batch, 1)?;
        let n_contigs = primitive::<UInt16Array>(&batch, 2)?;
        let tokens =
            batch
                .column(3)
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| CorpusError::SchemaMismatch {
                    detail: "column 3 (tokens) is not a List".to_string(),
                })?;
        let token_values = tokens
            .values()
            .as_any()
            .downcast_ref::<UInt8Array>()
            .ok_or_else(|| CorpusError::SchemaMismatch {
                detail: "tokens values are not UInt8".to_string(),
            })?
            .clone();
        let out = Self {
            cell_keys,
            cell_values,
            n_reads,
            n_contigs,
            token_offsets: tokens.value_offsets().to_vec(),
            token_values,
            first_row,
        };
        for i in 0..out.len() {
            let row = out.row(i);
            check_row(row.tokens, row.n_contigs, first_row + i, scratch)?;
        }
        Ok(out)
    }

    pub fn len(&self) -> usize {
        self.n_reads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn first_row(&self) -> usize {
        self.first_row
    }

    pub fn row(&self, i: usize) -> CorpusRow<'_> {
        let start = self.token_offsets[i] as usize;
        let end = self.token_offsets[i + 1] as usize;
        CorpusRow {
            cell_id: self.cell_values.value(self.cell_keys.value(i) as usize),
            n_reads: self.n_reads.value(i),
            n_contigs: self.n_contigs.value(i),
            tokens: &self.token_values.values()[start..end],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CorpusRow<'a> {
    pub cell_id: &'a str,
    pub n_reads: u32,
    pub n_contigs: u16,
    pub tokens: &'a [u8],
}

fn dict_column(batch: &RecordBatch, i: usize) -> Result<(Int32Array, StringArray)> {
    let dict = batch
        .column(i)
        .as_any()
        .downcast_ref::<arrow::array::DictionaryArray<Int32Type>>()
        .ok_or_else(|| CorpusError::SchemaMismatch {
            detail: format!("column {i} is not Dictionary<Int32, Utf8>"),
        })?;
    let values = dict
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| CorpusError::SchemaMismatch {
            detail: format!("column {i} dictionary values are not Utf8"),
        })?
        .clone();
    Ok((dict.keys().clone(), values))
}

fn primitive<T: Array + Clone + 'static>(batch: &RecordBatch, i: usize) -> Result<T> {
    batch
        .column(i)
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| CorpusError::SchemaMismatch {
            detail: format!("column {i} has type {:?}", batch.column(i).data_type()),
        })
        .cloned()
}
