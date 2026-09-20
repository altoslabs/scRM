const MAX_ERROR_RATE: f64 = 0.20;

const MATCH: i64 = 1;
const MISMATCH: i64 = -2;

const MIN_RUN: usize = 3;

pub fn homopolymer_trim_index(seq: &[u8], target_3prime: u8, revcomp: bool) -> usize {
    let n = seq.len();
    if n == 0 {
        return 0;
    }
    let target = if revcomp {
        complement(target_3prime)
    } else {
        target_3prime
    };

    let mut score: i64 = 0;
    let mut best_score: i64 = 0;
    let mut best_len: usize = 0;
    let mut matches: usize = 0;
    let mut best_matches: usize = 0;

    for i in 0..n {
        let base = if revcomp { seq[i] } else { seq[n - 1 - i] };
        if base == target {
            score += MATCH;
            matches += 1;
        } else {
            score += MISMATCH;
        }
        let cand = i + 1;
        let errors = cand - matches;
        let within_cap = (errors as f64) <= MAX_ERROR_RATE * (cand as f64);
        if score > best_score && within_cap {
            best_score = score;
            best_len = cand;
            best_matches = matches;
        }
    }
    let _ = best_matches;

    if best_len < MIN_RUN {
        best_len = 0;
    }

    if revcomp { best_len } else { n - best_len }
}

pub fn poly_a_trim_index(seq: &[u8], revcomp: bool) -> usize {
    homopolymer_trim_index(seq, b'A', revcomp)
}

fn complement(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pure_tail_is_fully_removed() {
        let seq = b"CGCGCGCGCGAAAAAAAAAA";
        assert_eq!(homopolymer_trim_index(seq, b'A', false), 10);
    }

    #[test]
    fn a_read_with_no_run_keeps_everything() {
        let seq = b"CGCGCGCGCGCGCGCGCGCG";
        assert_eq!(homopolymer_trim_index(seq, b'A', false), seq.len());
    }

    #[test]
    fn one_mismatch_inside_a_long_run_is_paid_for() {
        let seq = b"CGCGCGCGCGAAAAACAAAAA";
        let keep = homopolymer_trim_index(seq, b'A', false);
        assert_eq!(keep, 10, "the whole 11-base run should go, mismatch included");
    }

    #[test]
    fn a_single_stray_base_does_not_extend_the_run() {
        let seq = b"CGCGCGCGCGACGCG";
        let keep = homopolymer_trim_index(seq, b'A', false);
        assert!(keep >= seq.len() - 1, "keep={keep} cut too much on a stray base");
    }

    #[test]
    fn the_five_prime_orientation_scans_the_complement_and_returns_a_clip() {
        let seq = b"TTTTTTTTTTCGCGCGCGCG";
        assert_eq!(homopolymer_trim_index(seq, b'A', true), 10, "clip 10 from the head");
    }

    #[test]
    fn an_empty_read_is_handled_in_both_orientations() {
        assert_eq!(homopolymer_trim_index(b"", b'A', false), 0);
        assert_eq!(homopolymer_trim_index(b"", b'A', true), 0);
    }

    #[test]
    fn the_error_cap_rejects_a_mostly_wrong_candidate() {
        let seq = b"CGCGCGCGCGACACA";
        let keep = homopolymer_trim_index(seq, b'A', false);
        assert!(keep > 10, "keep={keep}: a 40%-error candidate must be rejected");
    }

    #[test]
    fn lower_case_is_not_a_match_and_that_is_deliberate() {
        assert_eq!(
            homopolymer_trim_index(b"cgcgcgcgcgaaaaaaaaaa", b'A', false),
            20,
            "no match"
        );
        assert_eq!(
            homopolymer_trim_index(b"CGCGCGCGCGAAAAAAAAAA", b'A', false),
            10,
            "matches"
        );
    }
}
