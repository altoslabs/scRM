//! 3' adapter clipping.
//!
//! A read-through adapter appears as a SUFFIX of the read: the adapter's prefix aligns with the
//! read's tail, possibly truncated at the read's end. So the match is suffix-anchored, and
//! `stringency` is the minimum overlap AT THE END, not a score anywhere in the read. A local
//! matcher fires on a chance 5-base similarity mid-read and then discards everything after it --
//! measured at a 0.36 adapter rate against an expected 0.0011.
//!
//! The leftmost qualifying position is taken, which removes the most, and a match is allowed
//! `floor(error_rate * overlap)` mismatches. Written from cutadapt's documented 3' adapter rule.

/// Index to cut at: `read[..three_prime_cut(..)]` survives. `read.len()` means no adapter.
pub fn three_prime_cut(read: &[u8], adapter: &[u8], min_overlap: usize, error_rate: f64) -> usize {
    let n = read.len();
    let m = adapter.len();
    if n == 0 || m == 0 || min_overlap == 0 {
        return n;
    }
    for start in 0..n {
        let overlap = (n - start).min(m);
        if overlap < min_overlap {
            break; // every later start is shorter still
        }
        // STOP AS SOON AS THE BOUND IS BLOWN. Almost every read carries no adapter, so this is
        // the hot path: counting all `overlap` mismatches before testing the bound did the full
        // comparison at every start position to answer "no". At an error rate of 0.1 a 13-base
        // overlap allows one mismatch, and random sequence blows that within a few bases.
        let allowed = (error_rate * overlap as f64).floor() as usize;
        let mut mismatches = 0usize;
        let mut ok = true;
        for k in 0..overlap {
            if read[start + k] != adapter[k] {
                mismatches += 1;
                if mismatches > allowed {
                    ok = false;
                    break;
                }
            }
        }
        if ok {
            return start;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    const AD: &[u8] = b"AGATCGGAAGAGC";
    const ER: f64 = 0.1;

    #[test]
    fn a_full_adapter_at_the_end_is_cut() {
        let r = [b"ACGTACGTACGTACGT".as_slice(), AD].concat();
        assert_eq!(three_prime_cut(&r, AD, 5, ER), 16);
    }

    #[test]
    fn a_clean_read_is_untouched() {
        let r = b"ACGTACGTACGTACGTACGTACGT";
        assert_eq!(three_prime_cut(r, AD, 5, ER), r.len());
    }

    /// A 3' adapter takes EVERYTHING AFTER IT, wherever it is found. That is cutadapt's `-a`
    /// rule: read-through means the insert ended, so the adapter and the rest of the read go.
    #[test]
    fn an_adapter_takes_the_sequence_after_it_too() {
        let r = [b"ACGTACGT".as_slice(), AD, b"TTTTGGGGCCCCAAAA".as_slice()].concat();
        assert_eq!(three_prime_cut(&r, AD, 5, ER), 8);
    }

    #[test]
    fn a_partial_adapter_at_the_end_is_cut_when_it_reaches_the_overlap() {
        let r = [b"ACGTACGTACGTACGT".as_slice(), &AD[..6]].concat();
        assert_eq!(three_prime_cut(&r, AD, 5, ER), 16, "6 bases clears an overlap of 5");
    }

    #[test]
    fn a_partial_adapter_below_the_overlap_is_left_alone() {
        let r = [b"ACGTACGTACGTACGT".as_slice(), &AD[..3]].concat();
        assert_eq!(three_prime_cut(&r, AD, 5, ER), r.len(), "3 bases is below 5");
    }

    /// One mismatch in a 13-base overlap is 0.077, under the 0.1 rate, so it still matches.
    #[test]
    fn one_mismatch_inside_the_error_rate_still_matches() {
        let mut ad = AD.to_vec();
        ad[6] = if ad[6] == b'A' { b'C' } else { b'A' };
        let r = [b"ACGTACGTACGTACGT".as_slice(), &ad].concat();
        assert_eq!(three_prime_cut(&r, AD, 5, ER), 16);
    }

    /// Two mismatches in 13 is 0.154, over the rate, so the full-length match is refused. The
    /// overlap is pinned at 13 so no shorter partial can match by chance and mask the point.
    #[test]
    fn two_mismatches_outside_the_error_rate_do_not() {
        let mut ad = AD.to_vec();
        ad[0] = b'T'; // A -> T
        ad[4] = b'T'; // C -> T, so two REAL mismatches
        let r = [b"ACGTACGTACGTACGT".as_slice(), &ad].concat();
        assert_eq!(three_prime_cut(&r, AD, AD.len(), ER), r.len());
    }

    #[test]
    fn a_read_that_is_entirely_adapter_cuts_to_nothing() {
        assert_eq!(three_prime_cut(AD, AD, 5, ER), 0);
    }
}
