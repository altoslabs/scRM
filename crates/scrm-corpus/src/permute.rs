use std::path::Path;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::{CorpusError, CorpusReader, CorpusWriter, Manifest, Result, ShardEntry, verify_corpus};

pub fn cross_cell_permutation(cell_of_row: &[u32], seed: u64) -> Result<Vec<usize>> {
    let n = cell_of_row.len();
    let max_run = {
        let mut counts = std::collections::HashMap::new();
        for &c in cell_of_row {
            *counts.entry(c).or_insert(0usize) += 1;
        }
        counts.values().copied().max().unwrap_or(0)
    };
    if n == 0 || 2 * max_run > n {
        return Err(CorpusError::CellIndexMismatch {
            detail: format!("no cross-cell derangement exists: {n} rows and the largest cell holds {max_run}"),
        });
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut perm: Vec<usize> = (0..n).collect();
    for j in (1..n).rev() {
        perm.swap(j, rng.random_range(0..=j));
    }
    loop {
        let bad: Vec<usize> = (0..n).filter(|&j| cell_of_row[perm[j]] == cell_of_row[j]).collect();
        if bad.is_empty() {
            break;
        }
        for &j in &bad {
            loop {
                let k = rng.random_range(0..n);
                if k != j && cell_of_row[perm[k]] != cell_of_row[j] && cell_of_row[perm[j]] != cell_of_row[k] {
                    perm.swap(j, k);
                    break;
                }
            }
        }
    }
    Ok(perm)
}

pub fn permute_corpus(src: &Path, dst: &Path, seed: u64) -> Result<Manifest> {
    let manifest = Manifest::read(src)?;

    let mut cell_names: Vec<String> = Vec::new();
    let mut cell_packed: Vec<u64> = Vec::new();
    let mut cell_of_row: Vec<u32> = Vec::new();
    let mut spans: Vec<(u64, u32, u32, u16)> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut shard_rows: Vec<(String, usize)> = Vec::new();
    for entry in &manifest.shards {
        let mut reader = CorpusReader::open(&src.join(&entry.file))?;
        let mut rows_here = 0usize;
        while let Some(batch) = reader.next_batch() {
            let batch = batch?;
            for i in 0..batch.len() {
                let row = batch.row(i);
                if cell_names.last().map(String::as_str) != Some(row.cell_id) {
                    let packed =
                        scrm_core::pack_acgt(row.cell_id.as_bytes()).ok_or_else(|| CorpusError::CellIndexMismatch {
                            detail: format!("cell_id {:?} is not packable ACGT", row.cell_id),
                        })?;
                    cell_names.push(row.cell_id.to_string());
                    cell_packed.push(packed);
                }
                cell_of_row.push((cell_names.len() - 1) as u32);
                spans.push((buf.len() as u64, row.tokens.len() as u32, row.n_reads, row.n_contigs));
                buf.extend_from_slice(row.tokens);
                rows_here += 1;
            }
        }
        shard_rows.push((entry.file.clone(), rows_here));
    }

    let perm = cross_cell_permutation(&cell_of_row, seed)?;

    std::fs::create_dir_all(dst)?;
    let mut entries: Vec<ShardEntry> = Vec::new();
    let mut j = 0usize;
    for (file, rows_here) in &shard_rows {
        let mut writer = CorpusWriter::create(&dst.join(file), &manifest.declared)?;
        for _ in 0..*rows_here {
            let cell = cell_of_row[j] as usize;
            let (off, len, n_reads, n_contigs) = spans[perm[j]];
            let tokens = &buf[off as usize..off as usize + len as usize];
            writer.push_row(&cell_names[cell], cell_packed[cell], n_reads, n_contigs, tokens)?;
            j += 1;
        }
        let summary = writer.finish()?;
        entries.push(ShardEntry {
            file: file.clone(),
            n_rows: summary.n_rows,
            n_tokens: summary.n_tokens,
            tokens_sha256: summary.tokens_sha256,
            first_cell: summary.first_cell,
            last_cell: summary.last_cell,
        });
    }

    let out = Manifest::new(manifest.declared.clone(), manifest.measured.clone(), entries);
    out.write(dst)?;
    let stats = verify_corpus(dst)?;
    if stats.n_rows != manifest.n_rows || stats.n_tokens != manifest.n_tokens {
        return Err(CorpusError::TallyMismatch {
            what: "permuted corpus rows/tokens vs source",
            expected: manifest.n_tokens,
            found: stats.n_tokens,
        });
    }

    let sidecar = serde_json::json!({
        "transformation": "cross_cell_molecule_permutation",
        "source": src.to_string_lossy(),
        "seed": seed,
        "rng": "ChaCha8Rng::seed_from_u64",
        "n_rows": manifest.n_rows,
        "n_cells": cell_names.len(),
        "constraint": "every row's payload comes from a different cell; per-cell molecule counts and cell order are unchanged",
    });
    std::fs::write(
        dst.join("permutation.json"),
        serde_json::to_string_pretty(&sidecar).map_err(|source| CorpusError::Json {
            key: "permutation.json",
            source,
        })?,
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_moves_every_row_out_of_its_cell_and_preserves_depth() {
        let cells: Vec<u32> = [0u32, 0, 0, 1, 1, 2, 2, 2, 3, 4].to_vec();
        let perm = cross_cell_permutation(&cells, 7).unwrap();
        let mut seen = perm.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..cells.len()).collect::<Vec<_>>(), "must be a bijection");
        for (j, &p) in perm.iter().enumerate() {
            assert_ne!(cells[p], cells[j], "row {j} kept a payload from its own cell");
        }
    }

    #[test]
    fn permutation_is_reproducible_from_its_seed_and_moves_with_it() {
        let cells: Vec<u32> = (0..500).map(|i| i / 5).collect();
        let a = cross_cell_permutation(&cells, 20260901).unwrap();
        let b = cross_cell_permutation(&cells, 20260901).unwrap();
        let c = cross_cell_permutation(&cells, 20260902).unwrap();
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn an_infeasible_layout_is_refused_not_looped() {
        let cells: Vec<u32> = [0u32, 0, 0, 1].to_vec();
        assert!(cross_cell_permutation(&cells, 1).is_err());
        assert!(cross_cell_permutation(&[], 1).is_err());
    }

    #[test]
    fn a_layout_at_the_feasibility_bound_is_accepted_and_deranged() {
        for cells in [
            vec![0u32, 0, 1, 1],
            vec![0, 0, 0, 1, 1, 1],
            vec![0, 0, 1, 1, 2, 2],
            vec![0, 0, 0, 0, 1, 1, 1, 1],
        ] {
            let n = cells.len();
            let perm = cross_cell_permutation(&cells, 3)
                .unwrap_or_else(|e| panic!("2 * max_run == n is feasible, not refusable: {cells:?}: {e}"));
            let mut seen = perm.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..n).collect::<Vec<_>>(), "{cells:?} must stay a bijection");
            for (j, &pj) in perm.iter().enumerate() {
                assert_ne!(cells[pj], cells[j], "{cells:?} row {j} kept its own cell's payload");
            }
        }
    }
}
