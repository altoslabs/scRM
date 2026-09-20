use std::collections::HashSet;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::{CellRun, CorpusError, CorpusReader, Result};

pub fn verify(path: &Path) -> Result<()> {
    let mut reader = CorpusReader::open(path)?;

    let mut hasher = Sha256::new();
    let mut rows = 0usize;
    let mut tokens = 0u64;
    let mut runs: Vec<CellRun> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    while let Some(batch) = reader.next_batch() {
        let batch = batch?;
        for i in 0..batch.len() {
            let row = batch.row(i);
            hasher.update(row.tokens);
            tokens += row.tokens.len() as u64;

            match runs.last_mut() {
                Some(run) if run.cell == row.cell_id => run.len += 1,
                last => {
                    if let Some(prev) = last {
                        if seen.contains(row.cell_id) {
                            return Err(scrm_core::Error::CellIdNotContiguous {
                                cell: row.cell_id.to_string(),
                                runs: 2,
                            }
                            .into());
                        }
                        if row.cell_id <= prev.cell.as_str() {
                            return Err(CorpusError::CellIdUnsorted {
                                prev: prev.cell.clone(),
                                next: row.cell_id.to_string(),
                            });
                        }
                    }
                    seen.insert(row.cell_id.to_string());
                    runs.push(CellRun {
                        cell: row.cell_id.to_string(),
                        start: rows,
                        len: 1,
                    });
                }
            }
            rows += 1;
        }
    }

    if rows as u64 != reader.n_rows() as u64 {
        return Err(CorpusError::TallyMismatch {
            what: "row count",
            expected: reader.n_rows() as u64,
            found: rows as u64,
        });
    }
    if tokens != reader.n_tokens() {
        return Err(CorpusError::TallyMismatch {
            what: "token count",
            expected: reader.n_tokens(),
            found: tokens,
        });
    }
    let sha = format!("{:x}", hasher.finalize());
    if sha != reader.tokens_sha256() {
        return Err(CorpusError::ShaMismatch {
            expected: reader.tokens_sha256().to_string(),
            found: sha,
        });
    }
    if runs.as_slice() != reader.cells() {
        return Err(CorpusError::CellIndexMismatch {
            detail: format!(
                "recomputed {} runs, metadata records {}",
                runs.len(),
                reader.cells().len()
            ),
        });
    }
    Ok(())
}
