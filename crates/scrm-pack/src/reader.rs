use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, ListArray, UInt8Array};
use arrow::buffer::Buffer;
use arrow_ipc::convert::fb_to_schema;
use arrow_ipc::reader::{FileDecoder, read_footer_length};
use arrow_ipc::root_as_footer;
use memmap2::Mmap;

use crate::{PackError, RowSource};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellLoc {
    pub name: String,
    pub shard: u32,
    pub start_row: u32,
    pub n_rows: u32,
}

struct BatchTokens {
    first_row: u32,
    list: ListArray,
    values: UInt8Array,
}

struct Shard {
    _mmap: Arc<Mmap>,
    batches: Vec<BatchTokens>,
}

pub struct Corpus {
    shards: Vec<Shard>,
    cells: Vec<CellLoc>,
    compat_key: OwnedCompatKey,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnedCompatKey {
    pub alphabet_size: u8,
    pub k: usize,
    pub dedup_rule: String,
    pub bubble_resolution: String,
    pub min_umi_len: usize,
    pub max_umi_len: usize,
    pub min_cell_molecules: usize,
}

impl OwnedCompatKey {
    fn of(c: &scrm_core::stamp::Config) -> Self {
        let (alphabet_size, k, dedup_rule, bubble_resolution, min_umi_len, max_umi_len, min_cell_molecules) =
            c.compat_key();
        Self {
            alphabet_size,
            k,
            dedup_rule: dedup_rule.to_string(),
            bubble_resolution: bubble_resolution.to_string(),
            min_umi_len,
            max_umi_len,
            min_cell_molecules,
        }
    }
}

impl Corpus {
    pub fn open(dir: &Path) -> Result<Self, PackError> {
        Self::open_many(std::slice::from_ref(&dir))
    }

    pub fn open_many(dirs: &[&Path]) -> Result<Self, PackError> {
        if dirs.is_empty() {
            return Err(PackError::Corpus("no corpus directories given".into()));
        }
        let mut shards = Vec::new();
        let mut cells = Vec::new();
        let mut agreed: Option<(OwnedCompatKey, &Path)> = None;

        for dir in dirs {
            let manifest = scrm_corpus::Manifest::read(dir).map_err(|e| PackError::Corpus(e.to_string()))?;
            let key = OwnedCompatKey::of(&manifest.declared.config);
            match &agreed {
                None => agreed = Some((key, dir)),
                Some((first, first_dir)) if *first != key => {
                    return Err(PackError::CompatKeyMismatch {
                        a: format!("{}: {first:?}", first_dir.display()),
                        b: format!("{}: {key:?}", dir.display()),
                    });
                }
                Some(_) => {}
            }
            Self::index_one(dir, &manifest, &mut shards, &mut cells)?;
        }
        Ok(Self {
            shards,
            cells,
            compat_key: agreed.expect("dirs is non-empty").0,
        })
    }

    pub fn compat_key(&self) -> &OwnedCompatKey {
        &self.compat_key
    }

    fn index_one(
        dir: &Path,
        manifest: &scrm_corpus::Manifest,
        shards: &mut Vec<Shard>,
        cells: &mut Vec<CellLoc>,
    ) -> Result<(), PackError> {
        let base = shards.len();
        for (shard_i, path) in manifest.shard_paths(dir).into_iter().enumerate() {
            let shard_i = base + shard_i;
            let file = File::open(&path).map_err(|e| PackError::Corpus(format!("{}: {e}", path.display())))?;
            let mmap = Arc::new(unsafe { Mmap::map(&file) }.map_err(|e| PackError::Corpus(e.to_string()))?);

            let buffer = unsafe {
                Buffer::from_custom_allocation(
                    std::ptr::NonNull::new(mmap.as_ptr() as *mut u8)
                        .ok_or_else(|| PackError::Corpus(format!("{} mapped to a null pointer", path.display())))?,
                    mmap.len(),
                    Arc::clone(&mmap) as Arc<dyn arrow_buffer::alloc::Allocation>,
                )
            };

            let trailer = buffer
                .len()
                .checked_sub(10)
                .ok_or_else(|| PackError::Corpus(format!("{} is too short to be Arrow IPC", path.display())))?;
            let footer_len = read_footer_length(buffer[trailer..].try_into().unwrap())
                .map_err(|e| PackError::Corpus(e.to_string()))?;
            let footer = root_as_footer(&buffer[trailer - footer_len..trailer])
                .map_err(|e| PackError::Corpus(format!("{}: {e}", path.display())))?;
            let schema = Arc::new(fb_to_schema(
                footer
                    .schema()
                    .ok_or_else(|| PackError::Corpus("footer has no schema".into()))?,
            ));

            let tokens_col = schema
                .index_of("tokens")
                .map_err(|e| PackError::Corpus(format!("{}: {e}", path.display())))?;
            let mut decoder = FileDecoder::new(schema, footer.version()).with_projection(vec![tokens_col]);
            for block in footer.dictionaries().iter().flatten() {
                let len = block.bodyLength() as usize + block.metaDataLength() as usize;
                let data = buffer.slice_with_length(block.offset() as _, len);
                decoder
                    .read_dictionary(block, &data)
                    .map_err(|e| PackError::Corpus(e.to_string()))?;
            }

            let mut batches = Vec::new();
            let mut row = 0u32;
            for block in footer.recordBatches().iter().flatten() {
                let len = block.bodyLength() as usize + block.metaDataLength() as usize;
                let data = buffer.slice_with_length(block.offset() as _, len);
                let batch = decoder
                    .read_record_batch(block, &data)
                    .map_err(|e| PackError::Corpus(e.to_string()))?
                    .ok_or_else(|| PackError::Corpus("projected batch was empty".into()))?;
                let list = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<ListArray>()
                    .ok_or_else(|| PackError::Corpus("tokens is not a List".into()))?
                    .clone();
                let values = list
                    .values()
                    .as_any()
                    .downcast_ref::<UInt8Array>()
                    .ok_or_else(|| PackError::Corpus("tokens values are not UInt8".into()))?
                    .clone();
                let n = list.len() as u32;
                batches.push(BatchTokens {
                    first_row: row,
                    list,
                    values,
                });
                row += n;
            }

            let reader = scrm_corpus::CorpusReader::open(&path).map_err(|e| PackError::Corpus(e.to_string()))?;
            for run in reader.cells() {
                cells.push(CellLoc {
                    name: run.cell.clone(),
                    shard: shard_i as u32,
                    start_row: run.start as u32,
                    n_rows: run.len as u32,
                });
            }
            shards.push(Shard { _mmap: mmap, batches });
        }
        Ok(())
    }

    pub fn cells(&self) -> &[CellLoc] {
        &self.cells
    }

    pub fn cell_lengths(&self, cell: &CellLoc, out: &mut Vec<u32>) -> Result<(), PackError> {
        out.clear();
        out.reserve(cell.n_rows as usize);
        for seg in self.segments(cell)? {
            let offs = seg.list.value_offsets();
            for i in 0..seg.n_rows as usize {
                let at = seg.start_in_batch as usize + i;
                out.push((offs[at + 1] - offs[at]) as u32);
            }
        }
        Ok(())
    }

    pub fn cell_rows(&self, cell: &CellLoc) -> Result<CellRows<'_>, PackError> {
        Ok(CellRows {
            segs: self.segments(cell)?,
        })
    }

    fn segments(&self, cell: &CellLoc) -> Result<Vec<Segment<'_>>, PackError> {
        let shard = self.shards.get(cell.shard as usize).ok_or(PackError::RowOutOfRange {
            row: cell.shard,
            n: self.shards.len(),
        })?;
        let mut segs = Vec::with_capacity(2);
        let mut taken = 0u32;
        for b in &shard.batches {
            let b_len = b.list.len() as u32;
            let b_end = b.first_row + b_len;
            let want_start = cell.start_row + taken;
            if want_start >= b_end {
                continue;
            }
            if want_start < b.first_row {
                break;
            }
            let start_in_batch = want_start - b.first_row;
            let n = (b_len - start_in_batch).min(cell.n_rows - taken);
            segs.push(Segment {
                list: &b.list,
                values: b.values.values(),
                first_row: taken,
                start_in_batch,
                n_rows: n,
            });
            taken += n;
            if taken == cell.n_rows {
                break;
            }
        }
        if taken != cell.n_rows {
            return Err(PackError::Corpus(format!(
                "cell {:?} claims {} rows from shard {} row {} but only {taken} were found",
                cell.name, cell.n_rows, cell.shard, cell.start_row
            )));
        }
        Ok(segs)
    }
}

struct Segment<'a> {
    list: &'a ListArray,
    values: &'a [u8],
    first_row: u32,
    start_in_batch: u32,
    n_rows: u32,
}

pub struct CellRows<'a> {
    segs: Vec<Segment<'a>>,
}

impl RowSource for CellRows<'_> {
    fn row(&self, i: u32) -> Result<&[u8], PackError> {
        for seg in &self.segs {
            if i >= seg.first_row && i < seg.first_row + seg.n_rows {
                let at = seg.start_in_batch as usize + (i - seg.first_row) as usize;
                let offs = seg.list.value_offsets();
                return Ok(&seg.values[offs[at] as usize..offs[at + 1] as usize]);
            }
        }
        Err(PackError::RowOutOfRange {
            row: i,
            n: self.n_rows(),
        })
    }

    fn n_rows(&self) -> usize {
        self.segs.iter().map(|s| s.n_rows as usize).sum()
    }
}
