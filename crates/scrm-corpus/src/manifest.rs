use std::path::{Path, PathBuf};

use rayon::prelude::*;
use scrm_core::stamp::{Declared, Measured};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{CorpusError, CorpusReader, Result};

pub const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShardEntry {
    pub file: String,
    pub n_rows: usize,
    pub n_tokens: u64,
    pub tokens_sha256: String,
    pub first_cell: String,
    pub last_cell: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub declared: Declared,
    pub measured: Measured,
    pub shards: Vec<ShardEntry>,
    pub n_rows: usize,
    pub n_tokens: u64,
    pub tokens_sha256_root: String,
}

impl Manifest {
    pub fn new(declared: Declared, measured: Measured, shards: Vec<ShardEntry>) -> Self {
        let n_rows = shards.iter().map(|s| s.n_rows).sum();
        let n_tokens = shards.iter().map(|s| s.n_tokens).sum();
        Self {
            declared,
            measured,
            tokens_sha256_root: root_digest(&shards),
            shards,
            n_rows,
            n_tokens,
        }
    }

    pub fn write(&self, dir: &Path) -> Result<()> {
        use std::io::Write;

        let json = serde_json::to_string_pretty(self).map_err(|source| CorpusError::Json {
            key: MANIFEST_FILE,
            source,
        })?;
        let tmp = dir.join(format!(".{MANIFEST_FILE}.{}.tmp", std::process::id()));
        let staged = (|| -> std::io::Result<()> {
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(json.as_bytes())?;
            f.sync_all()
        })();
        if let Err(source) = staged {
            let _ = std::fs::remove_file(&tmp);
            return Err(source.into());
        }
        std::fs::rename(&tmp, dir.join(MANIFEST_FILE))?;
        std::fs::File::open(dir)?.sync_all()?;
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join(MANIFEST_FILE))?;
        serde_json::from_str(&raw).map_err(|source| CorpusError::Json {
            key: MANIFEST_FILE,
            source,
        })
    }

    pub fn shard_paths(&self, dir: &Path) -> Vec<PathBuf> {
        self.shards.iter().map(|s| dir.join(&s.file)).collect()
    }
}

fn root_digest(shards: &[ShardEntry]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((shards.len() as u64).to_le_bytes());
    for s in shards {
        for field in [&s.file, &s.tokens_sha256, &s.first_cell, &s.last_cell] {
            hasher.update((field.len() as u64).to_le_bytes());
            hasher.update(field.as_bytes());
        }
        hasher.update((s.n_rows as u64).to_le_bytes());
        hasher.update(s.n_tokens.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorpusStats {
    pub root: String,
    pub n_rows: usize,
    pub n_tokens: u64,
    pub n_shards: usize,
}

pub fn verify_corpus(dir: &Path) -> Result<CorpusStats> {
    let manifest = Manifest::read(dir)?;
    if manifest.shards.is_empty() {
        return Err(CorpusError::SchemaMismatch {
            detail: "manifest lists no shards".into(),
        });
    }

    manifest
        .shards
        .par_iter()
        .enumerate()
        .try_for_each(|(k, entry)| verify_shard(dir, k, entry, &manifest.declared))?;

    let mut prev_last: Option<&str> = None;
    for entry in &manifest.shards {
        if prev_last.is_some_and(|prev| prev >= entry.first_cell.as_str()) {
            return Err(scrm_core::Error::CellIdNotContiguous {
                cell: entry.first_cell.clone(),
                runs: 2,
            }
            .into());
        }
        prev_last = Some(&entry.last_cell);
    }

    let root = root_digest(&manifest.shards);
    if root != manifest.tokens_sha256_root {
        return Err(CorpusError::ShaMismatch {
            expected: manifest.tokens_sha256_root.clone(),
            found: root,
        });
    }
    let n_rows: usize = manifest.shards.iter().map(|s| s.n_rows).sum();
    let n_tokens: u64 = manifest.shards.iter().map(|s| s.n_tokens).sum();
    if n_rows != manifest.n_rows || n_tokens != manifest.n_tokens {
        return Err(CorpusError::SchemaMismatch {
            detail: format!(
                "manifest totals {} rows / {} tokens, shards sum to {n_rows} / {n_tokens}",
                manifest.n_rows, manifest.n_tokens
            ),
        });
    }

    Ok(CorpusStats {
        root,
        n_rows,
        n_tokens,
        n_shards: manifest.shards.len(),
    })
}

fn verify_shard(dir: &Path, k: usize, entry: &ShardEntry, declared: &scrm_core::stamp::Declared) -> Result<()> {
    let path = dir.join(&entry.file);
    crate::verify(&path)?;
    let reader = CorpusReader::open(&path)?;

    let (shard_key, manifest_key) = (reader.declared().config.compat_key(), declared.config.compat_key());
    if shard_key != manifest_key {
        return Err(CorpusError::SchemaMismatch {
            detail: format!(
                "shard {k} ({}) declares the molecule definition {shard_key:?} but the manifest \
                 declares {manifest_key:?} -- two definitions of what a molecule is, in one corpus",
                entry.file
            ),
        });
    }

    if reader.n_rows() != entry.n_rows || reader.n_tokens() != entry.n_tokens {
        return Err(CorpusError::SchemaMismatch {
            detail: format!(
                "manifest says shard {k} has {} rows / {} tokens, the file has {} / {}",
                entry.n_rows,
                entry.n_tokens,
                reader.n_rows(),
                reader.n_tokens()
            ),
        });
    }
    if reader.tokens_sha256() != entry.tokens_sha256 {
        return Err(CorpusError::ShaMismatch {
            expected: entry.tokens_sha256.clone(),
            found: reader.tokens_sha256().to_string(),
        });
    }

    let cells = reader.cells();
    let (Some(first), Some(last)) = (cells.first(), cells.last()) else {
        return Err(CorpusError::SchemaMismatch {
            detail: format!("shard {k} holds no cells"),
        });
    };
    if first.cell != entry.first_cell || last.cell != entry.last_cell {
        return Err(CorpusError::SchemaMismatch {
            detail: format!(
                "manifest says shard {k} spans {}..{}, the file spans {}..{}",
                entry.first_cell, entry.last_cell, first.cell, last.cell
            ),
        });
    }
    Ok(())
}

pub fn stream_digest(dir: &Path) -> Result<String> {
    let manifest = Manifest::read(dir)?;
    let mut hasher = Sha256::new();
    for path in manifest.shard_paths(dir) {
        let mut reader = CorpusReader::open(&path)?;
        while let Some(batch) = reader.next_batch() {
            let batch = batch?;
            for i in 0..batch.len() {
                hasher.update(batch.row(i).tokens);
            }
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}
