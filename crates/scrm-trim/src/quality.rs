//! 3' quality trimming.
//!
//! The rule is BWA's, as cutadapt documents and implements it: subtract the cutoff from every
//! quality, take partial sums from the 3' end, and cut where that sum is smallest. It keeps a
//! stretch of good bases beyond a dip instead of stopping at the first base under the cutoff,
//! which a plain threshold scan would do.
//!
//! Written from that description.

/// Index to cut the 3' end at: `qual[..quality_trim_end(..)]` survives.
///
/// `phred_offset` is the ASCII base, 33 for Sanger. A quality byte below the offset is treated as
/// zero rather than wrapping, because a malformed record must not produce a negative quality that
/// reads as excellent.
pub fn quality_trim_end(qual: &[u8], cutoff: u8, phred_offset: u8) -> usize {
    let mut running: i32 = 0;
    let mut best: i32 = 0;
    let mut cut = qual.len();
    for i in (0..qual.len()).rev() {
        let q = qual[i].saturating_sub(phred_offset) as i32;
        running += cutoff as i32 - q;
        if running < 0 {
            break;
        }
        if running > best {
            best = running;
            cut = i;
        }
    }
    cut
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFF: u8 = 33;
    fn q(vals: &[u8]) -> Vec<u8> {
        vals.iter().map(|v| v + OFF).collect()
    }

    #[test]
    fn a_read_entirely_above_the_cutoff_is_untouched() {
        let s = q(&[40; 20]);
        assert_eq!(quality_trim_end(&s, 20, OFF), 20);
    }

    #[test]
    fn a_tail_below_the_cutoff_is_removed() {
        let mut s = q(&[40; 10]);
        s.extend(q(&[2; 5]));
        assert_eq!(quality_trim_end(&s, 20, OFF), 10, "the five bad bases go");
    }

    /// THE REASON THE RULE IS A PARTIAL SUM AND NOT A THRESHOLD SCAN. A single dip followed by
    /// good bases must NOT end the read: the partial sum recovers, so the cut stays at the end.
    #[test]
    fn a_dip_followed_by_good_bases_does_not_end_the_read() {
        let mut s = q(&[40; 10]);
        s.extend(q(&[2]));
        s.extend(q(&[40; 10]));
        assert_eq!(quality_trim_end(&s, 20, OFF), 21, "one bad base cannot cut a good tail");
    }

    #[test]
    fn everything_below_the_cutoff_cuts_to_nothing() {
        let s = q(&[2; 12]);
        assert_eq!(quality_trim_end(&s, 20, OFF), 0);
    }

    #[test]
    fn an_empty_read_has_nothing_to_cut() {
        assert_eq!(quality_trim_end(&[], 20, OFF), 0);
    }

    /// A quality byte below the phred offset is malformed. It must read as the WORST quality, not
    /// wrap to a huge one: saturating_sub is load-bearing.
    #[test]
    fn a_byte_below_the_offset_reads_as_the_worst_quality() {
        let s = vec![0u8; 8];
        assert_eq!(quality_trim_end(&s, 20, OFF), 0);
    }

    /// THE CUT IS GLOBAL, NOT LOCAL, AND THAT IS THE WHOLE POINT. At cutoff 20 the partial sum
    /// over 30,30,5,5,30,30,1,1,1 is deepest at index 2, not at index 6: the mid-read 5,5 patch
    /// plus the 1,1,1 tail together outweigh the 30,30 between them, so the good pair in the
    /// middle does not save the read. A threshold scan would have cut at 6.
    #[test]
    fn the_cut_is_the_deepest_minimum_of_the_partial_sum() {
        let s = q(&[30, 30, 5, 5, 30, 30, 1, 1, 1]);
        assert_eq!(quality_trim_end(&s, 20, OFF), 2);
    }
}
