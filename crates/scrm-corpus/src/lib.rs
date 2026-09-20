use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Fields, Schema};
use arrow::error::ArrowError;
use scrm_core::record::Umi;
use scrm_core::{ALPHABET_SIZE, CellId, MoleculeContigs, Token};
use serde::{Deserialize, Serialize};

mod manifest;
mod permute;
mod reader;
mod verify;
mod writer;

pub use manifest::{CorpusStats, MANIFEST_FILE, Manifest, ShardEntry, stream_digest, verify_corpus};
pub use permute::{cross_cell_permutation, permute_corpus};
pub use reader::{CorpusBatch, CorpusReader, CorpusRow};
pub use verify::verify;
pub use writer::{BATCH_ROWS, CorpusSummary, CorpusWriter};

pub const META_DECLARED: &str = "scrm_declared";
pub const META_N_ROWS: &str = "scrm_n_rows";
pub const META_N_TOKENS: &str = "scrm_n_tokens";
pub const META_TOKENS_SHA256: &str = "scrm_tokens_sha256";
pub const META_CELL_ROWS: &str = "scrm_cell_rows";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CellRun {
    pub cell: String,
    pub start: usize,
    pub len: usize,
}

impl CellRun {
    pub fn range(&self) -> Range<usize> {
        self.start..self.start + self.len
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CorpusError {
    #[error(transparent)]
    Core(#[from] scrm_core::Error),

    #[error(transparent)]
    Arrow(#[from] ArrowError),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("corpus metadata {key:?} is not valid JSON: {source}")]
    Json {
        key: &'static str,
        #[source]
        source: serde_json::Error,
    },

    #[error("cell_id {next:?} follows {prev:?}: corpus rows must ascend by cell_id")]
    CellIdUnsorted { prev: String, next: String },

    #[error("cell identity is aliased: {first_cell:?}/{first_packed} vs {second_cell:?}/{second_packed}")]
    CellIdAliased {
        first_cell: String,
        first_packed: u64,
        second_cell: String,
        second_packed: u64,
    },

    #[error("row {row}: token {token} is {what}")]
    Separator { row: usize, token: u8, what: &'static str },

    #[error("row {row}: n_contigs is {declared} but tokens recount to {recounted}")]
    ContigCountMismatch {
        row: usize,
        declared: u16,
        recounted: usize,
    },

    #[error("corpus metadata is missing {key:?}")]
    MissingMetadata { key: &'static str },

    #[error("row {row}: no tokens -- a molecule with no sequence is not a molecule")]
    EmptyRow { row: usize },

    #[error(
        "{dir} already holds a corpus (its manifest.json is there). Delete it or build elsewhere: \
         building into a finished corpus's directory publishes new shards under the old manifest, \
         which is what every reader and the embed-time pin gate read"
    )]
    AlreadyACorpus { dir: String },

    #[error("corpus metadata {key:?} is not a decimal integer: {value:?}")]
    MalformedMetadata { key: &'static str, value: String },

    #[error("{what} mismatch: metadata says {expected}, corpus contains {found}")]
    TallyMismatch {
        what: &'static str,
        expected: u64,
        found: u64,
    },

    #[error("token sha256 mismatch: metadata says {expected}, corpus hashes to {found}")]
    ShaMismatch { expected: String, found: String },

    #[error("cell index disagrees with the data: {detail}")]
    CellIndexMismatch { detail: String },

    #[error("corpus schema is not the scRM corpus schema: {detail}")]
    SchemaMismatch { detail: String },
}

pub type Result<T> = std::result::Result<T, CorpusError>;

pub fn refuse_if_corpus(dir: &Path) -> Result<()> {
    if dir.join(MANIFEST_FILE).exists() {
        return Err(CorpusError::AlreadyACorpus {
            dir: dir.display().to_string(),
        });
    }
    Ok(())
}

pub fn corpus_fields() -> Fields {
    let dict = || DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
    Fields::from(vec![
        Field::new("cell_id", dict(), false),
        Field::new("n_reads", DataType::UInt32, false),
        Field::new("n_contigs", DataType::UInt16, false),
        Field::new_list("tokens", Arc::new(token_item_field()), false),
    ])
}

pub(crate) fn token_item_field() -> Field {
    Field::new("item", DataType::UInt8, false)
}

pub fn corpus_schema(declared_json: String) -> Schema {
    Schema::new(corpus_fields()).with_metadata([(META_DECLARED.to_string(), declared_json)].into())
}

pub(crate) fn check_row(tokens: &[u8], n_contigs: u16, row: usize, scratch: &mut MoleculeContigs) -> Result<()> {
    const TAB: u8 = Token::Tab as u8;
    const NEWLINE: u8 = Token::Newline as u8;

    if tokens.is_empty() {
        return Err(CorpusError::EmptyRow { row });
    }

    for &t in tokens {
        if t >= ALPHABET_SIZE {
            return Err(scrm_core::Error::TokenOutOfRange { token: t }.into());
        }
        if t == NEWLINE {
            return Err(CorpusError::Separator {
                row,
                token: t,
                what: "Token::Newline, which is never stored in the corpus",
            });
        }
    }
    if tokens.first() == Some(&TAB) {
        return Err(CorpusError::Separator {
            row,
            token: TAB,
            what: "a leading separator",
        });
    }
    if tokens.last() == Some(&TAB) {
        return Err(CorpusError::Separator {
            row,
            token: TAB,
            what: "a trailing separator",
        });
    }
    if tokens.windows(2).any(|w| w[0] == TAB && w[1] == TAB) {
        return Err(CorpusError::Separator {
            row,
            token: TAB,
            what: "a doubled separator, i.e. an empty contig",
        });
    }

    scratch.tokens.clear();
    scratch.tokens.extend_from_slice(tokens);
    let recounted = scratch.recount_contigs();
    if recounted != n_contigs as usize {
        return Err(CorpusError::ContigCountMismatch {
            row,
            declared: n_contigs,
            recounted,
        });
    }
    Ok(())
}

pub(crate) fn recount_scratch() -> MoleculeContigs {
    MoleculeContigs {
        cell: CellId(0),
        umi: Umi(0),
        n_reads: 0,
        tokens: Vec::new(),
        n_contigs: 0,
    }
}

pub(crate) fn meta_str<'a>(meta: &'a std::collections::HashMap<String, String>, key: &'static str) -> Result<&'a str> {
    meta.get(key)
        .map(String::as_str)
        .ok_or(CorpusError::MissingMetadata { key })
}

pub(crate) fn meta_u64(meta: &std::collections::HashMap<String, String>, key: &'static str) -> Result<u64> {
    let raw = meta_str(meta, key)?;
    raw.parse().map_err(|_| CorpusError::MalformedMetadata {
        key,
        value: raw.to_string(),
    })
}
