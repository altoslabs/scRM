use proptest::prelude::*;
use scrm_pack::{CellTokens, RowSource, WindowOut, pack, tile, tile_first, window_occupancy};

const TAB: u8 = 5;
const NEWLINE: u8 = 6;

fn cell_strategy(max_mol: usize, max_len: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    prop::collection::vec(
        prop::collection::vec(0u8..=4, 1..=max_len).prop_flat_map(|bases| {
            let n = bases.len();
            prop::collection::btree_set(1usize..n.max(2), 0..=2usize.min(n.saturating_sub(1))).prop_map(move |cuts| {
                let mut out = Vec::with_capacity(bases.len() + cuts.len());
                let mut prev = 0usize;
                for &c in &cuts {
                    if c <= prev || c >= bases.len() {
                        continue;
                    }
                    out.extend_from_slice(&bases[prev..c]);
                    out.push(TAB);
                    prev = c;
                }
                out.extend_from_slice(&bases[prev..]);
                out
            })
        }),
        1..=max_mol,
    )
}

fn flat(rows: &[Vec<u8>]) -> (Vec<u8>, Vec<u32>) {
    let mut values = Vec::new();
    let mut offsets = vec![0u32];
    for r in rows {
        values.extend_from_slice(r);
        offsets.push(values.len() as u32);
    }
    (values, offsets)
}

proptest! {
    #[test]
    fn tile_partitions_exactly_once(
        lengths in prop::collection::vec(1u32..200, 1..80),
        seq_len in 200usize..2000,
    ) {
        let order: Vec<u32> = (0..lengths.len() as u32).collect();
        let windows = tile(&lengths, &order, seq_len).unwrap();
        let mut seen: Vec<u32> = windows.iter().flatten().copied().collect();
        seen.sort_unstable();
        prop_assert_eq!(seen, order);
        for w in &windows {
            prop_assert!(window_occupancy(&lengths, w) <= seq_len);
            prop_assert!(!w.is_empty(), "an empty window carries no molecule and must not exist");
        }
    }

    #[test]
    fn tile_first_is_a_prefix_of_tile(
        lengths in prop::collection::vec(1u32..200, 1..80),
        seq_len in 200usize..2000,
        want in 1usize..6,
    ) {
        let order: Vec<u32> = (0..lengths.len() as u32).collect();
        let all = tile(&lengths, &order, seq_len).unwrap();
        let first = tile_first(&lengths, &order, seq_len, want).unwrap();
        prop_assert_eq!(first.len(), all.len().min(want));
        prop_assert_eq!(&first[..], &all[..first.len()]);
    }

    #[test]
    fn any_order_still_partitions_exactly_once(
        lengths in prop::collection::vec(1u32..200, 1..60),
        seq_len in 200usize..2000,
        shift in 0usize..60,
    ) {
        let n = lengths.len();
        let order: Vec<u32> = (0..n).map(|i| ((i + shift) % n) as u32).collect();
        let windows = tile(&lengths, &order, seq_len).unwrap();
        let mut seen: Vec<u32> = windows.iter().flatten().copied().collect();
        seen.sort_unstable();
        prop_assert_eq!(seen, (0..n as u32).collect::<Vec<_>>());
    }

    #[test]
    fn pack_invariants_hold(rows in cell_strategy(12, 40), seq_len in 400usize..1200) {
        let (values, offsets) = flat(&rows);
        let cell = CellTokens { values: &values, offsets: &offsets };
        let lengths: Vec<u32> = (0..rows.len() as u32).map(|i| cell.row(i).unwrap().len() as u32).collect();
        let order: Vec<u32> = (0..rows.len() as u32).collect();
        let windows = tile(&lengths, &order, seq_len).unwrap();

        let mut tok = vec![0u8; seq_len];
        let mut val = vec![false; seq_len];
        let mut cseg = vec![0u16; seq_len];
        let mut useg = vec![0u16; seq_len];
        let mut ps = vec![0u32; seq_len];

        for w in &windows {
            let mut out = WindowOut {
                tokens: &mut tok,
                valid: &mut val,
                contig_seg: &mut cseg,
                umi_seg: &mut useg,
                patch_starts: &mut ps,
            };
            let st = pack(w, &cell, |_r, n, buf| buf.extend(0..n as u32), &mut out, seq_len).unwrap();
            let used = st.used;
            prop_assert!(used > 0 && used <= seq_len);

            prop_assert!(val[..used].iter().all(|&v| v));
            prop_assert!(!val[used..].iter().any(|&v| v));
            prop_assert!(tok[..used].iter().all(|&t| t < 7));
            prop_assert!(cseg[..used].windows(2).all(|p| p[0] <= p[1]));
            prop_assert!(useg[..used].windows(2).all(|p| p[0] <= p[1]));
            prop_assert_eq!(st.n_patches, w.len(), "a patch is a UMI, so one per molecule");
            prop_assert_eq!(ps[0], 0, "the first patch must start at 0");
            for i in 0..st.n_patches {
                let s = ps[i] as usize;
                let e = if i + 1 < st.n_patches { ps[i + 1] as usize } else { used };
                prop_assert!(e > s, "patch {} at {} is empty", i, s);
                prop_assert!(
                    useg[s..e].iter().all(|&x| x as usize == i),
                    "patch {} at {}..{} does not span exactly molecule {}'s umi ids", i, s, e, i
                );
            }
            let got: Vec<u8> = tok[..used].iter().copied().filter(|&b| b != TAB && b != NEWLINE).collect();
            let want: Vec<u8> = w
                .iter()
                .flat_map(|&i| cell.row(i).unwrap().iter().copied().filter(|&b| b != TAB))
                .collect();
            prop_assert_eq!(got, want, "packing lost or invented bases");
        }
    }

    #[test]
    fn a_contig_permutation_preserves_the_multiset_of_bases(rows in cell_strategy(8, 30), seq_len in 400usize..1200) {
        let (values, offsets) = flat(&rows);
        let cell = CellTokens { values: &values, offsets: &offsets };
        let window: Vec<u32> = (0..rows.len() as u32).collect();
        let mut a = (vec![0u8; seq_len], vec![false; seq_len], vec![0u16; seq_len], vec![0u16; seq_len], vec![0u32; seq_len]);
        let mut b = a.clone();

        let sa = pack(&window, &cell, |_r, n, buf| buf.extend(0..n as u32), &mut WindowOut {
            tokens: &mut a.0, valid: &mut a.1, contig_seg: &mut a.2, umi_seg: &mut a.3, patch_starts: &mut a.4,
        }, seq_len);
        let sb = pack(&window, &cell, |_r, n, buf| buf.extend((0..n as u32).rev()), &mut WindowOut {
            tokens: &mut b.0, valid: &mut b.1, contig_seg: &mut b.2, umi_seg: &mut b.3, patch_starts: &mut b.4,
        }, seq_len);

        prop_assert_eq!(sa.is_ok(), sb.is_ok());
        if let (Ok(sa), Ok(sb)) = (sa, sb) {
            prop_assert_eq!(sa.used, sb.used, "a reshuffle changed the occupancy");
            prop_assert_eq!(sa.n_contigs, sb.n_contigs);
            prop_assert_eq!(sa.n_patches, sb.n_patches);
            let mut x: Vec<u8> = a.0[..sa.used].to_vec();
            let mut y: Vec<u8> = b.0[..sb.used].to_vec();
            x.sort_unstable();
            y.sort_unstable();
            prop_assert_eq!(x, y, "a reshuffle changed which bytes are present");
            prop_assert_eq!(&a.3[..sa.used], &b.3[..sb.used], "umi id structure changed");
            prop_assert_eq!(&a.4[..sa.n_patches], &b.4[..sb.n_patches], "patch starts changed");

            let n_runs = |ids: &[u16]| -> usize {
                ids.windows(2).filter(|p| p[0] != p[1]).count() + usize::from(!ids.is_empty())
            };
            prop_assert_eq!(
                n_runs(&a.2[..sa.used]),
                sa.n_contigs,
                "a contig is split across more than one run, so it would be two attention blocks"
            );
            prop_assert_eq!(n_runs(&b.2[..sb.used]), sb.n_contigs);
        }
    }
}
