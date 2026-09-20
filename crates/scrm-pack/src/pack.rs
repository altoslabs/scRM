use crate::{NEWLINE, PackError};

const TAB: u8 = scrm_core::Token::Tab as u8;

pub const fn max_patches(seq_len: usize) -> usize {
    (seq_len + 1) / (scrm_core::MIN_UMI_LEN_DEFAULT + 1)
}

pub trait RowSource {
    fn row(&self, i: u32) -> Result<&[u8], PackError>;
    fn n_rows(&self) -> usize;
}

#[derive(Clone, Copy)]
pub struct CellTokens<'a> {
    pub values: &'a [u8],
    pub offsets: &'a [u32],
}

impl RowSource for CellTokens<'_> {
    fn row(&self, i: u32) -> Result<&[u8], PackError> {
        let n = self.n_rows();
        if i as usize >= n {
            return Err(PackError::RowOutOfRange { row: i, n });
        }
        let (a, b) = (self.offsets[i as usize] as usize, self.offsets[i as usize + 1] as usize);
        Ok(&self.values[a..b])
    }

    fn n_rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

pub struct WindowOut<'a> {
    pub tokens: &'a mut [u8],
    pub valid: &'a mut [bool],
    pub contig_seg: &'a mut [u16],
    pub umi_seg: &'a mut [u16],
    pub patch_starts: &'a mut [u32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowStats {
    pub used: usize,
    pub n_molecules: usize,
    pub n_contigs: usize,
    pub n_patches: usize,
}

pub fn pack(
    window: &[u32],
    cell: &impl RowSource,
    mut contig_perm: impl FnMut(u32, usize, &mut Vec<u32>),
    out: &mut WindowOut<'_>,
    seq_len: usize,
) -> Result<WindowStats, PackError> {
    debug_assert_eq!(out.tokens.len(), seq_len);
    out.tokens.fill(NEWLINE);
    out.contig_seg.fill(0);
    out.umi_seg.fill(0);

    let mut cursor = 0usize;
    let mut n_contigs_total = 0usize;
    let mut n_patches = 0usize;
    let mut perm: Vec<u32> = Vec::new();
    let mut pieces: Vec<(usize, usize)> = Vec::new();

    let last_mol = window.len().saturating_sub(1);
    for (mol_i, &row) in window.iter().enumerate() {
        let tokens = cell.row(row)?;

        pieces.clear();
        if let Some(first_tab) = tokens.iter().position(|&b| b == TAB) {
            let mut start = 0usize;
            let mut at = first_tab;
            loop {
                pieces.push((start, at));
                start = at + 1;
                match tokens[start..].iter().position(|&b| b == TAB) {
                    Some(off) => at = start + off,
                    None => {
                        pieces.push((start, tokens.len()));
                        break;
                    }
                }
            }
        } else {
            pieces.push((0, tokens.len()));
        }

        perm.clear();
        contig_perm(row, pieces.len(), &mut perm);
        if perm.len() != pieces.len() {
            return Err(PackError::BadPermutation {
                got: perm.len(),
                want: pieces.len(),
            });
        }

        *out.patch_starts
            .get_mut(n_patches)
            .ok_or(PackError::TooManyPatches { n: n_patches })? = cursor as u32;
        n_patches += 1;

        let last_contig = perm.len() - 1;
        for (k, &pi) in perm.iter().enumerate() {
            let (a, b) = *pieces.get(pi as usize).ok_or(PackError::BadPermutation {
                got: pi as usize,
                want: pieces.len(),
            })?;
            let len = b - a;

            let sep = if k < last_contig {
                Some(TAB)
            } else if mol_i < last_mol {
                Some(NEWLINE)
            } else {
                None
            };
            let span = len + usize::from(sep.is_some());
            if cursor + span > seq_len {
                return Err(PackError::WindowOverflow {
                    cursor,
                    len: span,
                    seq_len,
                });
            }

            out.tokens[cursor..cursor + len].copy_from_slice(&tokens[a..b]);
            if let Some(byte) = sep {
                out.tokens[cursor + len] = byte;
            }
            let seg = n_contigs_total as u16;
            out.contig_seg[cursor..cursor + span].fill(seg);
            out.umi_seg[cursor..cursor + span].fill(mol_i as u16);

            cursor += span;
            n_contigs_total += 1;
        }
    }

    out.valid[..cursor].fill(true);
    out.valid[cursor..].fill(false);
    Ok(WindowStats {
        used: cursor,
        n_molecules: window.len(),
        n_contigs: n_contigs_total,
        n_patches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: usize = 64;

    pub(super) struct Buf {
        pub(super) tokens: Vec<u8>,
        pub(super) valid: Vec<bool>,
        pub(super) contig: Vec<u16>,
        pub(super) umi: Vec<u16>,
        pub(super) patches: Vec<u32>,
    }

    impl Buf {
        pub(super) fn new(seq: usize) -> Self {
            Self {
                tokens: vec![0; seq],
                valid: vec![false; seq],
                contig: vec![0; seq],
                umi: vec![0; seq],
                patches: vec![0; seq],
            }
        }
        pub(super) fn out(&mut self) -> WindowOut<'_> {
            WindowOut {
                tokens: &mut self.tokens,
                valid: &mut self.valid,
                contig_seg: &mut self.contig,
                umi_seg: &mut self.umi,
                patch_starts: &mut self.patches,
            }
        }
    }

    fn assert_patches_are_molecule_starts(b: &Buf, st: &WindowStats) {
        assert_eq!(
            st.n_patches, st.n_molecules,
            "a patch is a UMI, so the counts must be equal"
        );
        for i in 0..st.n_patches {
            let start = b.patches[i] as usize;
            let end = if i + 1 < st.n_patches {
                b.patches[i + 1] as usize
            } else {
                st.used
            };
            assert!(end > start, "patch {i} at {start} is empty");
            assert!(
                b.umi[start..end].iter().all(|&x| x as usize == i),
                "patch {i} at {start}..{end} does not span exactly molecule {i}'s umi ids: {:?}",
                &b.umi[start..end]
            );
            assert!(
                start == 0 || b.umi[start - 1] as usize == i - 1,
                "patch {i}'s start at {start} is not the first byte of its molecule"
            );
        }
    }

    pub(super) fn identity(_row: u32, n: usize, out: &mut Vec<u32>) {
        out.extend(0..n as u32);
    }

    fn cell(rows: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>) {
        let mut values = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            values.extend_from_slice(r);
            offsets.push(values.len() as u32);
        }
        (values, offsets)
    }

    #[test]
    fn two_molecules_are_joined_by_one_newline_and_nothing_else() {
        let (v, o) = cell(&[vec![0, 1, 2], vec![3, 3]]);
        let mut b = Buf::new(SEQ);
        let st = pack(
            &[0, 1],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap();
        assert_eq!(st.used, 6, "3 + 1 separator + 2");
        assert_eq!(&b.tokens[..6], &[0, 1, 2, NEWLINE, 3, 3]);
        assert!(b.valid[..6].iter().all(|&x| x));
        assert!(!b.valid[6], "valid stops at used");
        assert_eq!(&b.umi[..6], &[0, 0, 0, 0, 1, 1]);
        assert_eq!(&b.patches[..st.n_patches], &[0, 4]);
        assert_patches_are_molecule_starts(&b, &st);
    }

    #[test]
    fn contigs_within_a_molecule_get_distinct_contig_ids_and_one_umi_id() {
        let (v, o) = cell(&[vec![0, 1, TAB, 2, 3]]);
        let mut b = Buf::new(SEQ);
        let st = pack(
            &[0],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap();
        assert_eq!(st.n_contigs, 2);
        assert_eq!(st.n_molecules, 1);
        assert_eq!(&b.tokens[..5], &[0, 1, TAB, 2, 3]);
        assert_eq!(&b.contig[..5], &[0, 0, 0, 1, 1], "the tab terminates contig 0");
        assert_eq!(&b.umi[..5], &[0, 0, 0, 0, 0], "but it is all one molecule");
    }

    #[test]
    fn a_patch_is_a_umi_spanning_all_its_contigs() {
        let (v, o) = cell(&[
            vec![0; 6].into_iter().chain([TAB]).chain(vec![1; 6]).collect(),
            vec![2, 2, 2],
        ]);
        let mut b = Buf::new(SEQ);
        let st = pack(
            &[0, 1],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap();
        assert_eq!(st.n_contigs, 3, "two contigs in molecule 0, one in molecule 1");
        assert_eq!(&b.patches[..st.n_patches], &[0, 14]);
        assert_patches_are_molecule_starts(&b, &st);
    }

    #[test]
    fn the_contig_permutation_reorders_bytes_without_changing_the_id_structure() {
        let (v, o) = cell(&[vec![0, 0, TAB, 1, 1, 1]]);
        let mut b = Buf::new(SEQ);
        let st = pack(
            &[0],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            |_r, n, out| out.extend((0..n as u32).rev()),
            &mut b.out(),
            SEQ,
        )
        .unwrap();
        assert_eq!(&b.tokens[..st.used], &[1, 1, 1, TAB, 0, 0], "contig order reversed");
        assert_eq!(st.n_contigs, 2);
        assert_eq!(&b.contig[..st.used], &[0, 0, 0, 0, 1, 1]);
        assert_eq!(&b.umi[..st.used], &[0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn a_wrong_length_permutation_is_rejected_rather_than_silently_truncating() {
        let (v, o) = cell(&[vec![0, 0, TAB, 1, 1]]);
        let mut b = Buf::new(SEQ);
        let err = pack(
            &[0],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            |_r, _n, out| out.push(0),
            &mut b.out(),
            SEQ,
        )
        .unwrap_err();
        assert!(matches!(err, PackError::BadPermutation { got: 1, want: 2 }), "{err:?}");
    }

    #[test]
    fn the_single_contig_fast_path_agrees_with_the_general_one() {
        let (v, o) = cell(&[vec![0, 1, 2, 3]]);
        let mut b = Buf::new(SEQ);
        let st = pack(
            &[0],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap();
        assert_eq!(st.n_contigs, 1);
        assert_eq!(&b.tokens[..4], &[0, 1, 2, 3]);
        assert_eq!(&b.contig[..4], &[0, 0, 0, 0]);
        assert_eq!(&b.patches[..st.n_patches], &[0]);
    }

    #[test]
    fn every_position_up_to_used_has_a_segment_id() {
        let (v, o) = cell(&[vec![0, TAB, 1], vec![2, 2], vec![3, TAB, 4, TAB, 0]]);
        let mut b = Buf::new(SEQ);
        let st = pack(
            &[0, 1, 2],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap();
        assert_eq!(st.n_molecules, 3);
        assert_eq!(st.n_contigs, 6, "2 + 1 + 3");
        let umi = &b.umi[..st.used];
        assert!(
            umi.windows(2).all(|w| w[0] <= w[1]),
            "umi ids must not go backwards: {umi:?}"
        );
        assert_eq!(*umi.last().unwrap(), 2);
        let contig = &b.contig[..st.used];
        assert!(contig.windows(2).all(|w| w[0] <= w[1]), "{contig:?}");
        assert_eq!(*contig.last().unwrap(), 5);
        assert_patches_are_molecule_starts(&b, &st);
    }

    #[test]
    fn overflowing_the_window_is_an_error_not_a_silent_truncation() {
        let (v, o) = cell(&[vec![7; 40], vec![7; 40]]);
        let mut b = Buf::new(SEQ);
        let err = pack(
            &[0, 1],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap_err();
        assert!(matches!(err, PackError::WindowOverflow { .. }), "{err:?}");
    }

    #[test]
    fn an_out_of_range_row_is_rejected() {
        let (v, o) = cell(&[vec![0]]);
        let mut b = Buf::new(SEQ);
        let err = pack(
            &[3],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap_err();
        assert!(matches!(err, PackError::RowOutOfRange { row: 3, n: 1 }), "{err:?}");
    }

    #[test]
    fn the_patch_bound_holds_for_the_worst_case_it_was_derived_from() {
        const TRAIN_SEQ: usize = 16384;
        let mp = max_patches(TRAIN_SEQ);
        assert_eq!(mp, 268, "the design's stated value at seq_len 16384 (13.6)");

        let min = scrm_core::MIN_UMI_LEN_DEFAULT;
        let rows: Vec<Vec<u8>> = (0..mp + 1).map(|_| vec![0u8; min]).collect();
        let (v, o) = cell(&rows);
        let ct = CellTokens {
            values: &v,
            offsets: &o,
        };

        let window: Vec<u32> = (0..mp as u32).collect();
        let mut b = Buf::new(TRAIN_SEQ);
        let st = pack(&window, &ct, identity, &mut b.out(), TRAIN_SEQ).unwrap();
        assert_eq!(
            st.n_patches, mp,
            "the bound must be attained by the case it was derived from"
        );
        assert_eq!(st.used, mp * min + mp - 1);
        assert_patches_are_molecule_starts(&b, &st);

        let window: Vec<u32> = (0..mp as u32 + 1).collect();
        let err = pack(&window, &ct, identity, &mut b.out(), TRAIN_SEQ).unwrap_err();
        assert!(matches!(err, PackError::WindowOverflow { .. }), "{err:?}");
    }

    #[test]
    fn a_filter_violating_input_hits_the_patch_buffer_bound_as_an_error_not_a_panic() {
        let (v, o) = cell(&[vec![0], vec![1], vec![2]]);
        let mut b = Buf::new(SEQ);
        b.patches.truncate(max_patches(SEQ));
        let err = pack(
            &[0, 1, 2],
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            identity,
            &mut b.out(),
            SEQ,
        )
        .unwrap_err();
        assert!(matches!(err, PackError::TooManyPatches { .. }), "{err:?}");
    }
}

#[cfg(test)]
mod multi_batch_tests {
    use super::*;

    struct TwoBatches<'a> {
        first: CellTokens<'a>,
        second: CellTokens<'a>,
    }

    impl RowSource for TwoBatches<'_> {
        fn row(&self, i: u32) -> Result<&[u8], PackError> {
            let split = self.first.n_rows() as u32;
            if i < split {
                self.first.row(i)
            } else {
                self.second.row(i - split)
            }
        }
        fn n_rows(&self) -> usize {
            self.first.n_rows() + self.second.n_rows()
        }
    }

    fn cat(rows: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>) {
        let mut values = Vec::new();
        let mut offsets = vec![0u32];
        for r in rows {
            values.extend_from_slice(r);
            offsets.push(values.len() as u32);
        }
        (values, offsets)
    }

    #[test]
    fn a_cell_split_across_two_buffers_packs_identically_to_one() {
        let rows = vec![
            vec![0, 1, 2],
            vec![3, TAB, 0, 1],
            vec![2, 2, 2],
            vec![1, TAB, 2, TAB, 3],
            vec![0],
        ];
        let seq = 64;
        let window: Vec<u32> = (0..rows.len() as u32).collect();

        let (v, o) = cat(&rows);
        let mut one = super::tests::Buf::new(seq);
        let a = pack(
            &window,
            &CellTokens {
                values: &v,
                offsets: &o,
            },
            super::tests::identity,
            &mut one.out(),
            seq,
        )
        .unwrap();

        let (v1, o1) = cat(&rows[..2]);
        let (v2, o2) = cat(&rows[2..]);
        let split = TwoBatches {
            first: CellTokens {
                values: &v1,
                offsets: &o1,
            },
            second: CellTokens {
                values: &v2,
                offsets: &o2,
            },
        };
        let mut two = super::tests::Buf::new(seq);
        let b = pack(&window, &split, super::tests::identity, &mut two.out(), seq).unwrap();

        assert_eq!(a, b, "stats differ between one buffer and two");
        assert_eq!(one.tokens, two.tokens, "tokens differ");
        assert_eq!(one.contig, two.contig, "contig ids differ");
        assert_eq!(one.umi, two.umi, "umi ids differ");
        assert_eq!(one.valid, two.valid, "valid mask differs");
        assert_eq!(
            one.patches[..a.n_patches],
            two.patches[..b.n_patches],
            "patch starts differ"
        );
    }

    #[test]
    fn a_row_past_the_end_of_both_buffers_is_rejected() {
        let (v1, o1) = cat(&[vec![0]]);
        let (v2, o2) = cat(&[vec![1]]);
        let split = TwoBatches {
            first: CellTokens {
                values: &v1,
                offsets: &o1,
            },
            second: CellTokens {
                values: &v2,
                offsets: &o2,
            },
        };
        assert_eq!(split.n_rows(), 2);
        assert!(split.row(0).is_ok() && split.row(1).is_ok());
        assert!(split.row(2).is_err(), "index past both buffers must not silently wrap");
    }
}
