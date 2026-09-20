pub mod alphabet;
pub mod kmer;
pub mod pack;
pub mod record;
pub mod stamp;

pub use alphabet::{ALPHABET_SIZE, Token};
pub use kmer::{
    K, K_DEFAULT, K_VALUES, K9, K11, K13, K15, K17, K19, K21, K23, K25, K27, K29, K31, Kmer9, Kmer11, Kmer13, Kmer15,
    Kmer17, Kmer19, Kmer21, Kmer23, Kmer25, Kmer27, Kmer29, Kmer31, MIN_TRIM_LEN, min_trim_len_for, validate_k,
};
pub use pack::{MAX_PACK_LEN, hamming_packed, pack_acgt, unpack_acgt};
pub use record::{
    CellId, LenFilter, MAX_UMI_LEN_DEFAULT, MIN_CELL_MOLECULES_DEFAULT, MIN_UMI_LEN_DEFAULT, Molecule, MoleculeContigs,
    MoleculeRead, ParsedPair, Umi,
};
pub use stamp::Stamp;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("read pair {index} has mismatched ids: R1 {r1:?} vs R2 {r2:?}")]
    PairIdMismatch { index: u64, r1: String, r2: String },

    #[error("token {token} is outside the alphabet (size {ALPHABET_SIZE})")]
    TokenOutOfRange { token: u8 },

    #[error("cell id {cell:?} appears in {runs} separate runs; corpus must be sorted by cell_id")]
    CellIdNotContiguous { cell: String, runs: usize },

    #[error("patch at offset {offset} spans umi_segment_id {from} -> {to}")]
    PatchSpansUmi { offset: usize, from: u32, to: u32 },

    #[error("molecule length bounds are unusable: min {min}, max {max} (need 1 <= min <= max)")]
    MoleculeBoundsInvalid { min: usize, max: usize },
}

pub type Result<T> = std::result::Result<T, Error>;
