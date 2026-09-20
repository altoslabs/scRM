mod batch;
mod pack;
mod reader;
mod tile;

pub use batch::{Batch, BatchSpec, build_batch, cells_below_v, derive_seed, kept_molecules, windows_for_cell};
pub use pack::{CellTokens, RowSource, WindowOut, WindowStats, max_patches, pack};
pub use reader::{CellLoc, CellRows, Corpus, OwnedCompatKey};
pub use tile::{NEWLINE, Window, tile, tile_first, window_occupancy};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PackError {
    #[error("molecule of {len} tokens cannot fit a window of {seq_len}")]
    MoleculeTooLong { len: usize, seq_len: usize },

    #[error("row index {row} is outside this cell's {n} rows")]
    RowOutOfRange { row: u32, n: usize },

    #[error("contig permutation has {got} entries for {want} contigs")]
    BadPermutation { got: usize, want: usize },

    #[error("window overflow: {len} tokens at cursor {cursor} exceeds seq_len {seq_len}")]
    WindowOverflow { cursor: usize, len: usize, seq_len: usize },

    #[error("patch {n} exceeds the buffer sized by max_patches()")]
    TooManyPatches { n: usize },

    #[error("corpora cannot be mixed -- compat_key differs.\n  {a}\n  {b}")]
    CompatKeyMismatch { a: String, b: String },

    #[error("corpus: {0}")]
    Corpus(String),

    #[error("geometry is unusable: max_umi_len {max_umi_len} against seq_len {seq_len}: {detail}")]
    Geometry {
        max_umi_len: usize,
        seq_len: usize,
        detail: &'static str,
    },
}

pub const MIN_MOLECULES_PER_WINDOW: usize = 10;

pub fn assert_geometry(max_umi_len: usize, seq_len: usize) -> Result<(), PackError> {
    if max_umi_len == 0 || seq_len == 0 {
        return Err(PackError::Geometry {
            max_umi_len,
            seq_len,
            detail: "neither may be zero",
        });
    }
    if seq_len / 2 > u16::MAX as usize {
        return Err(PackError::Geometry {
            max_umi_len,
            seq_len,
            detail: "seq_len / 2 must fit u16, or a window's contig ids can alias",
        });
    }
    if max_umi_len * MIN_MOLECULES_PER_WINDOW > seq_len {
        return Err(PackError::Geometry {
            max_umi_len,
            seq_len,
            detail: "max_umi_len must be at most a tenth of seq_len, or a window can hold too few \
                     molecules to be a bag of them",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shipped_geometry_holds() {
        assert!(assert_geometry(scrm_core::MAX_UMI_LEN_DEFAULT, 16384).is_ok());
    }

    #[test]
    fn a_window_that_could_hold_one_molecule_is_refused() {
        let err = assert_geometry(16384, 16384).unwrap_err();
        assert!(matches!(err, PackError::Geometry { .. }), "{err:?}");
        assert!(assert_geometry(1638, 16384).is_ok(), "exactly a tenth is fine");
        assert!(assert_geometry(1639, 16384).is_err(), "just over a tenth is not");
    }

    #[test]
    fn a_seq_len_whose_contig_ids_could_alias_u16_is_refused() {
        assert!(assert_geometry(1600, 16384).is_ok());
        assert!(assert_geometry(13106, 131_070).is_ok(), "exactly at the u16 bound");
        assert!(assert_geometry(13107, 131_072).is_err(), "one over aliases");
    }

    #[test]
    fn zero_is_refused_rather_than_dividing() {
        assert!(assert_geometry(0, 16384).is_err());
        assert!(assert_geometry(1600, 0).is_err());
    }
}
