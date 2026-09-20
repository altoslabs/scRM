use crate::polyrun;

use crate::artifacts::{ReadFlags, RunClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AnchoredScan {
    pub target_3prime: u8,
    pub revcomp: bool,
}

pub const ANCHORED_SCANS: [AnchoredScan; 4] = [
    AnchoredScan {
        target_3prime: b'A',
        revcomp: false,
    },
    AnchoredScan {
        target_3prime: b'A',
        revcomp: true,
    },
    AnchoredScan {
        target_3prime: b'G',
        revcomp: false,
    },
    AnchoredScan {
        target_3prime: b'G',
        revcomp: true,
    },
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut {
    pub start: usize,
    pub end: usize,
}

impl Cut {
    fn whole(len: usize) -> Self {
        Cut { start: 0, end: len }
    }

    fn removed(&self, len: usize) -> usize {
        len - (self.end - self.start)
    }
}

pub fn anchored_cut(seq: &[u8], scan: AnchoredScan, min_run: usize) -> Cut {
    let n = seq.len();
    if n == 0 {
        return Cut::whole(0);
    }
    let idx = if scan.target_3prime == b'A' {
        polyrun::poly_a_trim_index(seq, scan.revcomp)
    } else {
        polyrun::homopolymer_trim_index(seq, scan.target_3prime, scan.revcomp)
    };
    if scan.revcomp {
        if idx >= min_run {
            Cut { start: idx, end: n }
        } else {
            Cut::whole(n)
        }
    } else {
        if n - idx >= min_run {
            Cut { start: 0, end: idx }
        } else {
            Cut::whole(n)
        }
    }
}

pub fn trim_anchored(seq: &[u8], min_run: usize) -> Cut {
    let mut span = Cut::whole(seq.len());
    for scan in ANCHORED_SCANS {
        let sub = &seq[span.start..span.end];
        let cut = anchored_cut(sub, scan, min_run);
        span = Cut {
            start: span.start + cut.start,
            end: span.start + cut.end,
        };
        if span.start >= span.end {
            return Cut {
                start: span.start,
                end: span.start,
            };
        }
    }
    span
}

pub fn longest_internal_at_run(seq: &[u8], min_run: usize) -> Option<(usize, usize)> {
    let n = seq.len();
    let mut best: Option<(usize, usize)> = None;
    for (base, start, end) in maximal_runs(seq) {
        let internal = start > 0 && end < n;
        let is_at = base == b'A' || base == b'T';
        if internal && is_at && end - start >= min_run {
            let better = best.is_none_or(|(bs, be)| end - start > be - bs);
            if better {
                best = Some((start, end));
            }
        }
    }
    best
}

pub fn measure(seq: &[u8], flags: &mut ReadFlags) {
    let n = seq.len();
    for (base, start, end) in maximal_runs(seq) {
        let len = end - start;
        if end == n {
            flags.note_run(base, RunClass::ThreePrime, len);
        }
        if start == 0 {
            flags.note_run(base, RunClass::FivePrime, len);
        }
        if start > 0 && end < n {
            flags.note_run(base, RunClass::Internal, len);
        }
    }
}

fn maximal_runs(seq: &[u8]) -> impl Iterator<Item = (u8, usize, usize)> + '_ {
    let mut i = 0;
    std::iter::from_fn(move || {
        if i >= seq.len() {
            return None;
        }
        let base = seq[i];
        let start = i;
        while i < seq.len() && seq[i] == base {
            i += 1;
        }
        Some((base, start, i))
    })
}

pub fn removed(seq: &[u8], min_run: usize) -> usize {
    trim_anchored(seq, min_run).removed(seq.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN_RUN: usize = 10;

    #[test]
    fn three_prime_poly_a_is_trimmed() {
        let mut seq = b"ACGTACGTACGTACGTACGTACGTACGT".to_vec();
        seq.extend_from_slice(&[b'A'; 25]);
        let cut = trim_anchored(&seq, MIN_RUN);
        assert_eq!(cut, Cut { start: 0, end: 28 });
    }

    #[test]
    fn five_prime_poly_t_is_trimmed() {
        let mut seq = vec![b'T'; 25];
        seq.extend_from_slice(b"ACGTACGTACGTACGTACGTACGTACGT");
        let cut = trim_anchored(&seq, MIN_RUN);
        assert_eq!(cut, Cut { start: 25, end: 53 });
    }

    #[test]
    fn both_ends_are_trimmed_in_one_pass() {
        let mut seq = vec![b'T'; 20];
        seq.extend_from_slice(b"ACGTACGTACGTACGTACGTACGTACGT");
        seq.extend_from_slice(&[b'A'; 20]);
        let cut = trim_anchored(&seq, MIN_RUN);
        assert_eq!(cut, Cut { start: 20, end: 48 });
    }

    #[test]
    fn poly_g_and_poly_c_are_trimmed_in_both_orientations() {
        let mut g = b"ACGTACGTACGTACGTACGTACGT".to_vec();
        g.extend_from_slice(&[b'G'; 15]);
        assert_eq!(trim_anchored(&g, MIN_RUN), Cut { start: 0, end: 24 });

        let mut c = vec![b'C'; 15];
        c.extend_from_slice(b"ACGTACGTACGTACGTACGTACGT");
        assert_eq!(trim_anchored(&c, MIN_RUN), Cut { start: 15, end: 39 });
    }

    #[test]
    fn runs_at_the_threshold_are_trimmed_and_runs_below_it_are_left_alone() {
        let flank = b"ACGTACGTACGTACGTACGTACGT";
        let run = |n: usize| {
            let mut seq = flank.to_vec();
            seq.extend_from_slice(&vec![b'A'; n]);
            seq
        };
        let seq = run(MIN_RUN - 1);
        let n = seq.len();
        assert_eq!(trim_anchored(&seq, MIN_RUN), Cut { start: 0, end: n });

        let seq = run(MIN_RUN);
        assert_eq!(
            trim_anchored(&seq, MIN_RUN),
            Cut {
                start: 0,
                end: flank.len()
            },
            "a run of exactly min_run = {MIN_RUN} must be trimmed"
        );

        let head = |n: usize| {
            let mut seq = vec![b'T'; n];
            seq.extend_from_slice(flank);
            seq
        };
        let seq = head(MIN_RUN - 1);
        let n = seq.len();
        assert_eq!(trim_anchored(&seq, MIN_RUN), Cut { start: 0, end: n }, "5' below");
        let seq = head(MIN_RUN);
        assert_eq!(
            trim_anchored(&seq, MIN_RUN),
            Cut {
                start: MIN_RUN,
                end: MIN_RUN + flank.len()
            },
            "5' at the threshold"
        );
    }

    #[test]
    fn an_internal_run_is_not_trimmed_whichever_flank_is_longer() {
        let mut seq = b"ACGTACGTAC".to_vec();
        seq.extend_from_slice(&[b'T'; 25]);
        seq.extend_from_slice(b"ACGTACGTACGTACGTACGTACGT");
        let n = seq.len();
        assert_eq!(longest_internal_at_run(&seq, MIN_RUN), Some((10, 35)));
        assert_eq!(trim_anchored(&seq, MIN_RUN), Cut { start: 0, end: n });
        assert_eq!(removed(&seq, MIN_RUN), 0);

        let mut seq = b"ACGTACGTACGTACGTACGTACGT".to_vec();
        seq.extend_from_slice(&[b'T'; 25]);
        seq.extend_from_slice(b"ACGTACGTAC");
        let n = seq.len();
        assert_eq!(longest_internal_at_run(&seq, MIN_RUN), Some((23, 49)));
        assert_eq!(trim_anchored(&seq, MIN_RUN), Cut { start: 0, end: n });
        assert_eq!(removed(&seq, MIN_RUN), 0);
    }

    #[test]
    fn an_internal_run_past_k_is_still_not_trimmed() {
        let mut seq = b"ACGTACGTACGTACGTACGT".to_vec();
        seq.extend_from_slice(&[b'A'; 30]);
        seq.extend_from_slice(b"CGTACGTACGTACGTACGTC");
        let n = seq.len();
        const { assert!(30 > scrm_core::K) };
        assert_eq!(trim_anchored(&seq, MIN_RUN), Cut { start: 0, end: n });
    }

    #[test]
    fn trimming_an_anchored_run_does_not_reach_the_interior() {
        let mut seq = b"ACGTACGTAC".to_vec();
        seq.extend_from_slice(&[b'A'; 25]);
        seq.extend_from_slice(b"CGTACGTACGTACGTACGTC");
        seq.extend_from_slice(&[b'A'; 20]);
        let cut = trim_anchored(&seq, MIN_RUN);
        let kept = &seq[cut.start..cut.end];
        assert_eq!(cut.end, 55, "the 3' tail is trimmed");
        assert!(
            kept.windows(25).any(|w| w == [b'A'; 25]),
            "the internal run must survive: {}",
            String::from_utf8_lossy(kept)
        );
    }

    #[test]
    fn measurement_classifies_every_position() {
        let mut seq = vec![b'T'; 12];
        seq.extend_from_slice(b"ACGT");
        seq.extend_from_slice(&[b'A'; 21]);
        seq.extend_from_slice(b"ACGT");
        seq.extend_from_slice(&[b'G'; 31]);
        let mut flags = ReadFlags::default();
        measure(&seq, &mut flags);
        assert!(flags.has(b'T', RunClass::FivePrime, 10));
        assert!(!flags.has(b'T', RunClass::FivePrime, 20));
        assert!(flags.has(b'A', RunClass::Internal, 20));
        assert!(!flags.has(b'A', RunClass::Internal, 30));
        assert!(flags.has(b'G', RunClass::ThreePrime, 30));
        assert!(!flags.has(b'C', RunClass::Internal, 10));
    }

    #[test]
    fn an_internal_run_is_counted_but_not_cut() {
        let mut seq = b"ACGTACGTACGTACGTACGT".to_vec();
        seq.extend_from_slice(&[b'T'; 25]);
        seq.extend_from_slice(b"ACGTACGTACGTACGTACGT");
        let before = seq.clone();
        let n = seq.len();

        let mut flags = ReadFlags::default();
        measure(&seq, &mut flags);

        assert!(flags.has(b'T', RunClass::Internal, 10));
        assert!(flags.has(b'T', RunClass::Internal, 20));
        assert!(!flags.has(b'T', RunClass::Internal, 30));
        assert!(!flags.has(b'T', RunClass::FivePrime, 10));
        assert!(!flags.has(b'T', RunClass::ThreePrime, 10));
        assert_eq!(trim_anchored(&seq, MIN_RUN), Cut { start: 0, end: n });
        assert_eq!(removed(&seq, MIN_RUN), 0);
        assert_eq!(seq, before);
    }

    #[test]
    fn measurement_never_mutates_the_read() {
        let seq = b"TTTTTTTTTTTTACGTAAAAAAAAAAAA".to_vec();
        let before = seq.clone();
        let mut flags = ReadFlags::default();
        measure(&seq, &mut flags);
        assert_eq!(seq, before);
    }

    #[test]
    fn an_all_a_read_is_consumed_entirely() {
        let seq = vec![b'A'; 40];
        let cut = trim_anchored(&seq, MIN_RUN);
        assert_eq!(cut.end - cut.start, 0);
        assert_eq!(removed(&seq, MIN_RUN), 40);
    }
}
