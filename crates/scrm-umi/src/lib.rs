//! UMI deduplication, per cell, with NO gene partition: the corpus is reference-free, so there
//! are no genes to partition by.
//!
//! Two published methods are owed credit here, and neither is code taken from its source.
//! `flow` is CellRanger's `1MM_CR_flow` (10x Genomics), reimplemented from its description; see
//! `flow.rs` for the rule and what this tree pins down that the description leaves open.
//! `pairs::hamming1_pairs` enumerates the Hamming-1 candidate pairs with STARsolo's sort/swap
//! kernel (alexdobin/STAR): sort the 2-bit packed UMIs and scan forward while the xor stays
//! inside the low half, then swap the two halves, sort and scan again, so every 1-mismatch pair
//! is adjacent in one of the two orders. That is where the `MIN_UMI_LEN`..=`MAX_UMI_LEN` bound
//! below comes from. Both notices are in THIRD_PARTY_NOTICES.md.

pub mod flow;
pub mod molecule;
pub mod pairs;
pub mod stats;

pub use flow::{Flow, UmiGroup, flow};
pub use molecule::{DedupedCell, dedup_cell};
pub use pairs::{MAX_UMI_LEN, MIN_UMI_LEN, hamming1_pairs};
pub use scrm_core::record::{CellId, Molecule, ParsedPair, Umi};
pub use stats::{
    CellUmiStats, OBSERVED_MAX_CHAIN_DEPTH, OBSERVED_MAX_GROUP_UMIS, PERCOLATION_DEGREE, UmiStatsAccum,
    percolation_warning,
};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("umi_len {0} out of range: the sort/swap kernel needs {MIN_UMI_LEN}..={MAX_UMI_LEN} bases")]
    UmiLenOutOfRange(u8),

    #[error("umi 0x{umi:016x} has bits set above the {umi_len}-base packed width")]
    UmiOutOfRange { umi: u64, umi_len: u8 },

    #[error("umi 0x{0:016x} appears twice; sum read counts per distinct umi before calling")]
    DuplicateUmi(u64),

    #[error("read pair carries cell {found:?} but this call is for cell {expected:?}")]
    ForeignCell { expected: CellId, found: CellId },

    #[error("token {0} is not a base (A=0, C=1, G=2, T=3)")]
    NotABase(u8),
}

pub type Result<T> = std::result::Result<T, Error>;

pub fn pack_umi(bases: &[u8]) -> Result<Umi> {
    if bases.len() < MIN_UMI_LEN as usize || bases.len() > MAX_UMI_LEN as usize {
        return Err(Error::UmiLenOutOfRange(bases.len().min(u8::MAX as usize) as u8));
    }
    let mut v = 0u64;
    for &b in bases {
        if b > 3 {
            return Err(Error::NotABase(b));
        }
        v = (v << 2) | b as u64;
    }
    Ok(Umi(v))
}

pub fn unpack_umi(umi: Umi, umi_len: u8) -> Vec<u8> {
    (0..umi_len as u32)
        .rev()
        .map(|i| ((umi.0 >> (2 * i)) & 3) as u8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use scrm_core::Token;

    #[test]
    fn pack_umi_matches_token_ids() {
        let umi = pack_umi(&[Token::A as u8, Token::C as u8, Token::G as u8, Token::T as u8]).unwrap();
        assert_eq!(umi, Umi(0b00_01_10_11));
        assert_eq!(unpack_umi(umi, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn pack_umi_rejects_non_bases() {
        assert_eq!(
            pack_umi(&[0, 1, Token::N as u8, 3]),
            Err(Error::NotABase(Token::N as u8))
        );
    }

    proptest! {
        #[test]
        fn packed_order_is_lexicographic(
            a in prop::collection::vec(0u8..4, 6),
            b in prop::collection::vec(0u8..4, 6),
        ) {
            let (pa, pb) = (pack_umi(&a).unwrap(), pack_umi(&b).unwrap());
            let ascii = |v: &[u8]| -> Vec<u8> { v.iter().map(|&t| b"ACGT"[t as usize]).collect() };
            prop_assert_eq!(pa.cmp(&pb), ascii(&a).cmp(&ascii(&b)));
        }
    }
}
