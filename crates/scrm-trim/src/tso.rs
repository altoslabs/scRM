pub const TSO: &[u8] = b"AAGCAGTGGTATCAACGCAGAGTACATGGG";

pub const MIN_SCORE: usize = 20;

const MAX_ERROR_PCT: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TsoMatch {
    pub start: usize,
    pub end: usize,
    pub matches: usize,
    pub mismatches: usize,
}

impl TsoMatch {
    fn score(&self) -> usize {
        self.matches - self.mismatches
    }
}

pub fn find(read: &[u8], adapter: &[u8], min_score: usize) -> Option<TsoMatch> {
    let n = read.len() as isize;
    let m = adapter.len() as isize;
    let min = min_score as isize;
    if n < min || m < min {
        return None;
    }

    let mut best: Option<TsoMatch> = None;
    for offset in -(m - min)..=(n - min) {
        let lo = offset.max(0);
        let hi = (offset + m).min(n);
        if hi - lo < min {
            continue;
        }
        if let Some(window) = best_window(read, adapter, offset, lo, hi, min_score) {
            let better = best.is_none_or(|prev| window.score() > prev.score());
            if better {
                best = Some(window);
            }
        }
    }
    best
}

fn best_window(read: &[u8], adapter: &[u8], offset: isize, lo: isize, hi: isize, min_score: usize) -> Option<TsoMatch> {
    let mut run_score: isize = 0;
    let mut run_start = lo;
    let mut run_matches = 0usize;
    let mut run_mismatches = 0usize;

    let mut best: Option<TsoMatch> = None;

    // Compared as two SLICES rather than by index. The bodies are identical; indexing each byte
    // with `(pos - offset) as usize` cost a cast and two bounds checks per base, and this function
    // is 85% of all trimming time -- 81 alignments over a 90-base read is ~2,400 comparisons.
    let r = &read[lo as usize..hi as usize];
    let a = &adapter[(lo - offset) as usize..(hi - offset) as usize];

    for (i, (&rb, &ab)) in r.iter().zip(a.iter()).enumerate() {
        let pos = lo + i as isize;
        // EXACT BOUND. Every remaining position can add at most 1, and a fresh run can start no
        // earlier than the next position, so if neither the current run nor a new one can still
        // reach `min_score`, nothing later in this alignment can either.
        let remaining = hi - pos;
        if run_score.max(0) + remaining < min_score as isize {
            break;
        }
        let hit = ab == rb;
        if run_score < 0 {
            run_score = 0;
            run_start = pos;
            run_matches = 0;
            run_mismatches = 0;
        }
        if hit {
            run_score += 1;
            run_matches += 1;
        } else {
            run_score -= 1;
            run_mismatches += 1;
        }
        if run_score >= min_score as isize {
            let candidate = TsoMatch {
                start: run_start as usize,
                end: pos as usize + 1,
                matches: run_matches,
                mismatches: run_mismatches,
            };
            let overlap = candidate.end - candidate.start;
            if candidate.mismatches * 100 <= overlap * MAX_ERROR_PCT {
                let better = best.is_none_or(|prev| candidate.score() > prev.score());
                if better {
                    best = Some(candidate);
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const CDNA: &[u8] = b"GCTAGCTAGGCATCGATCGGATCCAAGCTTGGATCCTAGCTAGCTAGGCATCGATCGGATCC";

    fn read_with_prefix(prefix: &[u8]) -> Vec<u8> {
        let mut v = prefix.to_vec();
        v.extend_from_slice(CDNA);
        v
    }

    #[test]
    fn truncated_tso_at_the_five_prime_end_is_found() {
        let read = read_with_prefix(&TSO[..22]);
        let m = find(&read, TSO, MIN_SCORE).unwrap();
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 22);
        assert_eq!(m.matches, 22);
        assert_eq!(m.mismatches, 0);
    }

    #[test]
    fn every_measured_truncation_length_is_found() {
        for len in 20..=29 {
            let read = read_with_prefix(&TSO[..len]);
            let m = find(&read, TSO, MIN_SCORE).unwrap_or_else(|| panic!("TSO[..{len}] at the 5' end was missed"));
            let want_end = len + usize::from(TSO[len] == CDNA[0]);
            assert_eq!((m.start, m.end), (0, want_end), "len {len}");
        }
    }

    #[test]
    fn a_window_above_the_mismatch_cap_is_refused_however_high_it_scores() {
        let mut p = TSO.to_vec();
        for i in [6usize, 13, 20, 27] {
            p[i] = if p[i] == b'A' { b'C' } else { b'A' };
        }
        assert!(
            find(&read_with_prefix(&p), TSO, MIN_SCORE).is_none(),
            "13.3% mismatch is above the {MAX_ERROR_PCT}% cap and must not be called an oligo"
        );

        let mut p = TSO.to_vec();
        for i in [7usize, 15, 23] {
            p[i] = if p[i] == b'A' { b'C' } else { b'A' };
        }
        let m = find(&read_with_prefix(&p), TSO, MIN_SCORE).expect("exactly 10% mismatch is at the cap, not above it");
        assert_eq!((m.start, m.end, m.mismatches), (0, 30, 3));
    }

    #[test]
    fn nineteen_oligo_bases_score_below_the_threshold() {
        let cdna = b"TCTAGCTAGGCATCGATCGGATCCAAGCTTGG";
        assert_ne!(cdna[0], TSO[19], "fixture must not extend the window");
        let mut read = TSO[..19].to_vec();
        read.extend_from_slice(cdna);
        assert!(find(&read, TSO, MIN_SCORE).is_none());
    }

    #[test]
    fn tso_suffix_hanging_off_the_five_prime_edge_is_found() {
        let read = read_with_prefix(&TSO[8..]);
        let m = find(&read, TSO, MIN_SCORE).unwrap();
        assert_eq!(m.start, 0);
        assert_eq!(m.end, 22);
    }

    #[test]
    fn substitutions_within_ten_percent_are_tolerated() {
        let mut prefix = TSO[..25].to_vec();
        prefix[5] = if prefix[5] == b'A' { b'C' } else { b'A' };
        prefix[15] = if prefix[15] == b'A' { b'C' } else { b'A' };
        let read = read_with_prefix(&prefix);
        let m = find(&read, TSO, MIN_SCORE).unwrap();
        assert_eq!(m.end, 25);
        assert_eq!(m.mismatches, 2);
    }

    #[test]
    fn a_read_with_no_tso_yields_none() {
        assert!(find(CDNA, TSO, MIN_SCORE).is_none());
    }

    #[test]
    fn full_length_tso_is_found_even_though_it_is_never_observed() {
        let read = read_with_prefix(TSO);
        let m = find(&read, TSO, MIN_SCORE).unwrap();
        assert_eq!(m.end, 30);
        assert_eq!(m.matches, 30);
    }

    #[test]
    fn a_read_shorter_than_the_threshold_yields_none() {
        assert!(find(&TSO[..15], TSO, MIN_SCORE).is_none());
        assert!(find(b"", TSO, MIN_SCORE).is_none());
        assert!(find(&TSO[..MIN_SCORE - 1], TSO, MIN_SCORE).is_none(), "n = min - 1");
        let m = find(&TSO[..MIN_SCORE], TSO, MIN_SCORE).expect("n == min_score must still match");
        assert_eq!((m.start, m.end, m.mismatches), (0, MIN_SCORE, 0));
        assert!(
            find(&read_with_prefix(TSO), &TSO[..MIN_SCORE - 1], MIN_SCORE).is_none(),
            "an adapter shorter than min_score cannot be matched"
        );
    }

    #[test]
    fn internal_truncated_occurrence_is_found() {
        let mut read = b"GCTAGCTAGG".to_vec();
        read.extend_from_slice(&TSO[..24]);
        read.extend_from_slice(CDNA);
        let m = find(&read, TSO, MIN_SCORE).unwrap();
        assert_eq!(m.start, 10);
        assert_eq!(m.end, 34);
    }
}
