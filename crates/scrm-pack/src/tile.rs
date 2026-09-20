use crate::PackError;

pub const NEWLINE: u8 = scrm_core::Token::Newline as u8;

pub type Window = Vec<u32>;

pub fn tile(lengths: &[u32], order: &[u32], seq_len: usize) -> Result<Vec<Window>, PackError> {
    tile_first(lengths, order, seq_len, usize::MAX)
}

pub fn tile_first(lengths: &[u32], order: &[u32], seq_len: usize, want: usize) -> Result<Vec<Window>, PackError> {
    let mut windows = Vec::new();
    let mut current: Window = Vec::new();
    let mut used = 0usize;

    for &idx in order {
        let len = *lengths.get(idx as usize).ok_or(PackError::RowOutOfRange {
            row: idx,
            n: lengths.len(),
        })? as usize;
        if len > seq_len {
            return Err(PackError::MoleculeTooLong { len, seq_len });
        }
        let cost = if current.is_empty() { len } else { len + 1 };
        if used + cost > seq_len {
            windows.push(std::mem::take(&mut current));
            if windows.len() == want {
                return Ok(windows);
            }
            used = len;
            current.push(idx);
        } else {
            used += cost;
            current.push(idx);
        }
    }
    if !current.is_empty() && windows.len() < want {
        windows.push(current);
    }
    Ok(windows)
}

pub fn window_occupancy(lengths: &[u32], window: &[u32]) -> usize {
    if window.is_empty() {
        return 0;
    }
    window.iter().map(|&i| lengths[i as usize] as usize).sum::<usize>() + window.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQ: usize = 64;

    fn ids(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn the_partition_is_disjoint_and_covering() {
        let lengths: Vec<u32> = (1..=40).map(|i| (i * 7) % 23 + 3).collect();
        let windows = tile(&lengths, &ids(40), SEQ).unwrap();
        let mut seen: Vec<u32> = windows.iter().flatten().copied().collect();
        seen.sort_unstable();
        assert_eq!(seen, ids(40), "every molecule exactly once, in some window");
        for w in &windows {
            assert!(
                window_occupancy(&lengths, w) <= SEQ,
                "window of {} tokens exceeds seq_len {SEQ}",
                window_occupancy(&lengths, w)
            );
        }
    }

    #[test]
    fn separators_are_counted_between_molecules_and_not_after_the_last() {
        let lengths = vec![20, 20, 20];
        assert_eq!(tile(&lengths, &ids(3), 64).unwrap().len(), 1);
        assert_eq!(window_occupancy(&lengths, &[0, 1, 2]), 62);

        let lengths = vec![21, 21, 21];
        let w = tile(&lengths, &ids(3), 64).unwrap();
        assert_eq!(w.len(), 2, "65 tokens cannot fit 64");
        assert_eq!(w[0], vec![0, 1]);
        assert_eq!(w[1], vec![2]);
        assert_eq!(window_occupancy(&lengths, &w[0]), 43);
    }

    #[test]
    fn a_molecule_filling_a_window_exactly_is_not_split() {
        let lengths = vec![64, 10];
        let w = tile(&lengths, &ids(2), 64).unwrap();
        assert_eq!(w, vec![vec![0], vec![1]], "a full window is full, not overfull");
    }

    #[test]
    fn order_decides_the_partition_and_is_respected_verbatim() {
        let lengths = vec![30, 30, 30, 30];
        let a = tile(&lengths, &[0, 1, 2, 3], 64).unwrap();
        let b = tile(&lengths, &[3, 1, 0, 2], 64).unwrap();
        assert_eq!(a, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(b, vec![vec![3, 1], vec![0, 2]], "order is a permutation, not a hint");
        assert_ne!(a, b);
    }

    #[test]
    fn a_molecule_longer_than_a_window_is_an_error_not_a_loop() {
        let err = tile(&[100], &[0], 64).unwrap_err();
        assert!(
            matches!(err, PackError::MoleculeTooLong { len: 100, seq_len: 64 }),
            "{err:?}"
        );
    }

    #[test]
    fn an_out_of_range_order_index_is_rejected() {
        let err = tile(&[10, 10], &[0, 5], 64).unwrap_err();
        assert!(matches!(err, PackError::RowOutOfRange { row: 5, n: 2 }), "{err:?}");
    }

    #[test]
    fn a_cell_with_no_molecules_yields_no_windows() {
        assert!(tile(&[], &[], SEQ).unwrap().is_empty());
    }

    #[test]
    fn windows_per_cell_matches_the_measured_corpus_geometry() {
        for (mean_len, want_lo, want_hi) in [(104u32, 150, 160), (109, 145, 155), (325, 48, 53)] {
            let lengths = vec![mean_len; 4000];
            let windows = tile(&lengths, &ids(4000), 16384).unwrap();
            let per = windows[0].len();
            assert!(
                (want_lo..=want_hi).contains(&per),
                "at mean length {mean_len} a window held {per} molecules, expected {want_lo}..={want_hi}"
            );
        }
    }
}
