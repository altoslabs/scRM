//! Quality-weighted consensus calling at a disputed column.
//!
//! The model is fgbio's `ConsensusCaller` / `CallMolecularConsensusReads`
//! (fulcrumgenomics/fgbio, MIT), reimplemented here: the two-trial error composition and the
//! parameter vocabulary below -- `MIN_CONSENSUS_Q`, `OMEGA_LOG10`, `ERROR_RATE_PRE`,
//! `ERROR_RATE_POST` -- are fgbio's by name. The notice is in THIRD_PARTY_NOTICES.md.
//!
//! Each disputed column is called by quality-weighted likelihood rather than by counting reads,
//! which is why the three retired `BubbleResolution` policies are gone: they decided confidence
//! without confidence data.

use scrm_core::Token;

pub const MIN_CONSENSUS_Q: u8 = 10;

pub const OMEGA_LOG10: u8 = 7;

pub const ERROR_RATE_PRE: u8 = 45;

pub const ERROR_RATE_POST: u8 = 40;

const MAX_PHRED: usize = 93;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Call {
    Base(u8, u8),
    Tie,
    LowConfidence(u8),
}

impl Call {
    #[inline]
    pub fn token(self) -> u8 {
        match self {
            Call::Base(b, _) => b,
            Call::Tie | Call::LowConfidence(_) => Token::N as u8,
        }
    }

    #[inline]
    pub fn is_no_call(self) -> bool {
        !matches!(self, Call::Base(..))
    }
}

#[derive(Debug, Clone)]
pub struct ConsensusModel {
    ln_truth: Vec<f64>,
    ln_third: Vec<f64>,
    p_err: Vec<f64>,
    p_pre: f64,
    min_consensus_q: u8,
    omega_log10: u8,
}

#[inline]
fn two_trials(e1: f64, e2: f64) -> f64 {
    e1 + e2 - 2.0 * e1 * e2
}

#[inline]
fn phred_to_p(q: u8) -> f64 {
    10f64.powf(-(q as f64) / 10.0)
}

impl Default for ConsensusModel {
    fn default() -> Self {
        Self::new(ERROR_RATE_PRE, ERROR_RATE_POST, MIN_CONSENSUS_Q)
    }
}

impl ConsensusModel {
    pub fn with_omega(error_rate_pre: u8, error_rate_post: u8, min_consensus_q: u8, omega_log10: u8) -> Self {
        let mut m = Self::new(error_rate_pre, error_rate_post, min_consensus_q);
        m.omega_log10 = omega_log10;
        m
    }

    pub fn new(error_rate_pre: u8, error_rate_post: u8, min_consensus_q: u8) -> Self {
        let e_post = phred_to_p(error_rate_post);
        let mut ln_truth = Vec::with_capacity(MAX_PHRED + 1);
        let mut ln_third = Vec::with_capacity(MAX_PHRED + 1);
        let mut p_err_tab = Vec::with_capacity(MAX_PHRED + 1);
        for q in 0..=MAX_PHRED {
            let p_err = two_trials(e_post, phred_to_p(q as u8)).clamp(1e-12, 0.75);
            ln_truth.push((1.0 - p_err).ln());
            ln_third.push((p_err / 3.0).ln());
            p_err_tab.push(p_err);
        }
        ConsensusModel {
            ln_truth,
            ln_third,
            p_err: p_err_tab,
            p_pre: phred_to_p(error_rate_pre),
            min_consensus_q,
            omega_log10: OMEGA_LOG10,
        }
    }

    #[inline]
    pub fn expected_disagreement(&self, obs: &[(u8, u8)]) -> f64 {
        let mut cnt = [0u32; MAX_PHRED + 1];
        for &(_, q) in obs {
            cnt[(q as usize).min(MAX_PHRED)] += 1;
        }
        let s: f64 = (0..=MAX_PHRED)
            .filter(|&q| cnt[q] > 0)
            .map(|q| self.p_err[q] * cnt[q] as f64)
            .sum();
        s.min(1.0)
    }

    pub fn origin_surprise(&self, k: u64, lambda: f64) -> f64 {
        if k == 0 {
            return 0.0;
        }
        if lambda <= 0.0 {
            return f64::INFINITY;
        }
        let mut term = (-lambda).exp();
        for i in 0..k {
            term *= lambda / (i as f64 + 1.0);
        }
        let mut tail = 0.0f64;
        let mut t = term;
        for j in 0..4096u64 {
            tail += t;
            t *= lambda / (k + j + 1) as f64;
            if t < 1e-18 * tail.max(f64::MIN_POSITIVE) {
                break;
            }
        }
        if tail <= 0.0 {
            return f64::INFINITY;
        }
        -tail.log10()
    }

    #[inline]
    pub fn rejects_same_origin(&self, k: u64, lambda: f64) -> bool {
        self.origin_surprise(k, lambda) > self.omega_log10 as f64
    }

    #[inline]
    pub fn omega_log10(&self) -> u8 {
        self.omega_log10
    }

    #[inline]
    pub fn min_consensus_q(&self) -> u8 {
        self.min_consensus_q
    }

    pub fn call_column(&self, obs: &[(u8, u8)]) -> Call {
        if obs.is_empty() {
            return Call::Tie;
        }
        let mut cnt = [[0u32; 4]; MAX_PHRED + 1];
        let (mut lo, mut hi) = (MAX_PHRED, 0usize);
        for &(base, q) in obs {
            debug_assert!(base < 4, "a column observation must be a base, got token {base}");
            let q = (q as usize).min(MAX_PHRED);
            cnt[q][base as usize] += 1;
            lo = lo.min(q);
            hi = hi.max(q);
        }
        let mut ll = [0f64; 4];
        let mut comp = [0f64; 4];
        for (q, per_base) in cnt.iter().enumerate().take(hi + 1).skip(lo) {
            let n: u32 = per_base.iter().sum();
            if n == 0 {
                continue;
            }
            let (truth, third) = (self.ln_truth[q], self.ln_third[q]);
            for ((ll_b, comp_b), &c) in ll.iter_mut().zip(comp.iter_mut()).zip(per_base) {
                let term = truth * c as f64 + third * (n - c) as f64;
                let adjusted = term - *comp_b;
                let sum = *ll_b + adjusted;
                *comp_b = (sum - *ll_b) - adjusted;
                *ll_b = sum;
            }
        }

        let mut best = 0usize;
        for b in 1..4 {
            if ll[b] > ll[best] {
                best = b;
            }
        }
        if (0..4).any(|b| b != best && ll[b] == ll[best]) {
            return Call::Tie;
        }

        let m = ll[best];
        let denom: f64 = ll.iter().map(|&x| (x - m).exp()).sum();
        let p_consensus_err = (1.0 - 1.0 / denom).clamp(0.0, 1.0);
        let p = two_trials(self.p_pre, p_consensus_err).clamp(1e-12, 1.0);
        let phred = (-10.0 * p.log10()).round().clamp(0.0, 60.0) as u8;
        if phred < self.min_consensus_q {
            return Call::LowConfidence(phred);
        }
        Call::Base(best as u8, phred)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: u8 = 0;
    const C: u8 = 1;

    #[test]
    fn a_single_observation_matches_the_arithmetic_computed_by_hand() {
        let e_post = 1e-4_f64;
        let p_q30 = 1e-3_f64;
        let p_err = e_post + p_q30 - 2.0 * e_post * p_q30;
        let ll_a = (1.0 - p_err).ln();
        let ll_other = (p_err / 3.0).ln();
        let denom = 1.0 + 3.0 * (ll_other - ll_a).exp();
        let p_consensus_err = 1.0 - 1.0 / denom;
        let e_pre = 10f64.powf(-4.5);
        let p = e_pre + p_consensus_err - 2.0 * e_pre * p_consensus_err;
        let want = (-10.0 * p.log10()).round() as u8;
        assert_eq!(
            want, 29,
            "hand arithmetic itself changed; check the derivation, not the code"
        );

        let m = ConsensusModel::default();
        assert_eq!(m.call_column(&[(A, 30)]), Call::Base(A, 29));
    }

    #[test]
    fn equal_qualities_disagreeing_is_a_tie_and_a_tie_is_never_broken_by_order() {
        let m = ConsensusModel::default();
        assert_eq!(m.call_column(&[(A, 30), (C, 30)]), Call::Tie);
        assert_eq!(m.call_column(&[(C, 30), (A, 30)]), Call::Tie);
        assert_eq!(m.call_column(&[(2, 30), (3, 30)]), Call::Tie);
    }

    #[test]
    fn a_clearly_better_read_wins_where_weight_counting_would_have_tied() {
        let m = ConsensusModel::default();
        assert_eq!(m.call_column(&[(A, 40), (C, 10)]), Call::Base(A, 27));
        assert_eq!(m.call_column(&[(A, 40), (C, 8)]), Call::Base(A, 29));
    }

    #[test]
    fn three_weak_agreeing_reads_outweigh_one_strong_dissenter_but_not_confidently() {
        let no_floor = ConsensusModel::new(ERROR_RATE_PRE, ERROR_RATE_POST, 0);
        assert_eq!(
            no_floor.call_column(&[(A, 10), (A, 10), (A, 10), (C, 40)]),
            Call::Base(A, 4)
        );
        assert_eq!(
            ConsensusModel::default().call_column(&[(A, 10), (A, 10), (A, 10), (C, 40)]),
            Call::LowConfidence(4)
        );
    }

    #[test]
    fn a_near_tie_is_not_a_tie_and_the_floor_is_what_catches_it() {
        let no_floor = ConsensusModel::new(ERROR_RATE_PRE, ERROR_RATE_POST, 0);
        assert_eq!(no_floor.call_column(&[(A, 31), (C, 30)]), Call::Base(A, 3));
        assert_eq!(
            ConsensusModel::default().call_column(&[(A, 31), (C, 30)]),
            Call::LowConfidence(3)
        );
    }

    #[test]
    fn a_zero_quality_observation_is_a_null_vote_but_q2_still_decides() {
        let m = ConsensusModel::default();
        let alone = m.call_column(&[(A, 30)]);
        assert_eq!(
            m.call_column(&[(A, 30), (C, 0)]),
            alone,
            "a Q0 dissent must not move the call"
        );
        assert_eq!(m.call_column(&[(A, 30), (C, 1)]), alone, "nor a Q1 dissent");

        assert_eq!(m.call_column(&[(A, 30), (C, 2)]), alone);

        assert_eq!(m.call_column(&[(A, 30), (C, 30)]), Call::Tie);
        let no_floor = ConsensusModel::new(ERROR_RATE_PRE, ERROR_RATE_POST, 0);
        assert_eq!(no_floor.call_column(&[(A, 30), (C, 30), (A, 2)]), Call::Base(A, 4));
    }

    #[test]
    fn depth_saturates_confidence_at_the_pre_umi_ceiling_by_two_reads() {
        let m = ConsensusModel::default();
        let q = |n: usize| match m.call_column(&vec![(A, 30); n]) {
            Call::Base(_, q) => q,
            other => panic!("unanimous column should call a base, got {other:?}"),
        };
        assert_eq!(q(1), 29);
        assert!(q(1) < q(2), "a second agreeing read must raise confidence");
        for n in [2, 3, 5, 10, 50, 500] {
            assert_eq!(q(n), ERROR_RATE_PRE, "the ceiling is error_rate_pre, not the Q60 cap");
        }
    }

    #[test]
    fn a_unanimous_low_quality_column_still_calls_its_base() {
        let m = ConsensusModel::default();
        assert_eq!(m.call_column(&[(A, 2), (A, 2), (A, 2)]), Call::LowConfidence(4));
        let no_floor = ConsensusModel::new(ERROR_RATE_PRE, ERROR_RATE_POST, 0);
        assert_eq!(no_floor.call_column(&[(A, 2), (A, 2), (A, 2)]), Call::Base(A, 4));
    }

    #[test]
    fn an_empty_column_is_a_no_call_rather_than_a_panic() {
        assert_eq!(ConsensusModel::default().call_column(&[]), Call::Tie);
        assert_eq!(Call::Tie.token(), Token::N as u8);
        assert!(Call::Tie.is_no_call());
        assert!(!Call::Base(A, 30).is_no_call());
    }

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed | 1;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    #[test]
    fn the_call_is_invariant_to_observation_order() {
        let m = ConsensusModel::default();

        let counterexample = [(C, 22), (A, 28), (A, 22), (C, 28)];
        for perm in permutations(&counterexample) {
            assert_eq!(
                m.call_column(&perm),
                Call::Tie,
                "the review's counterexample must tie in every order, got {perm:?}"
            );
        }

        let mut next = rng(0x5DEE_CE66_D3AD_BEEF);
        for set in 0..20_000u64 {
            let n = 3 + (next() % 4) as usize;
            let obs: Vec<(u8, u8)> = (0..n).map(|_| ((next() % 4) as u8, 2 + (next() % 40) as u8)).collect();
            let want = m.call_column(&obs);
            let want_lambda = m.expected_disagreement(&obs);
            let mut shuffled = obs.clone();
            for _ in 0..8 {
                for i in (1..shuffled.len()).rev() {
                    shuffled.swap(i, (next() % (i as u64 + 1)) as usize);
                }
                assert_eq!(
                    m.call_column(&shuffled),
                    want,
                    "set {set} changed call under permutation: {obs:?} vs {shuffled:?}"
                );
                assert_eq!(
                    m.expected_disagreement(&shuffled).to_bits(),
                    want_lambda.to_bits(),
                    "set {set} changed lambda under permutation: {obs:?} vs {shuffled:?}"
                );
            }
        }
    }

    #[test]
    fn an_exactly_symmetric_column_is_reported_as_a_tie_at_every_quality() {
        let m = ConsensusModel::default();
        for q1 in 2..=41u8 {
            for q2 in 2..=41u8 {
                assert_eq!(
                    m.call_column(&[(A, q1), (C, q1), (A, q2), (C, q2)]),
                    Call::Tie,
                    "symmetric 4-observation set at q=({q1}, {q2})"
                );
                for q3 in 2..=41u8 {
                    assert_eq!(
                        m.call_column(&[(A, q1), (C, q1), (A, q2), (C, q2), (A, q3), (C, q3)]),
                        Call::Tie,
                        "symmetric 6-observation set at q=({q1}, {q2}, {q3})"
                    );
                }
            }
        }
    }

    fn permutations(obs: &[(u8, u8)]) -> Vec<Vec<(u8, u8)>> {
        if obs.len() <= 1 {
            return vec![obs.to_vec()];
        }
        let mut out = Vec::new();
        for i in 0..obs.len() {
            let mut rest = obs.to_vec();
            let head = rest.remove(i);
            for mut tail in permutations(&rest) {
                tail.insert(0, head);
                out.push(tail);
            }
        }
        out
    }

    #[test]
    fn a_quality_above_the_table_is_clamped_to_the_table_ceiling() {
        let m = ConsensusModel::default();
        assert_eq!(m.call_column(&[(A, 93)]), Call::Base(A, 39));
        for q in [94u8, 200, 255] {
            assert_eq!(
                m.call_column(&[(A, q)]),
                Call::Base(A, 39),
                "Q{q} must be the Q93 row, not a lower one"
            );
        }
        assert_eq!(m.call_column(&[(A, 41)]), Call::Base(A, 37));
    }

    #[test]
    fn expected_disagreement_sums_the_observations_error_rates_and_clamps_at_one() {
        let m = ConsensusModel::default();
        let two_trials_at = |q: u8| {
            let (e1, e2) = (1e-4_f64, 10f64.powf(-(q as f64) / 10.0));
            e1 + e2 - 2.0 * e1 * e2
        };
        assert_eq!(m.expected_disagreement(&[(A, 30)]), two_trials_at(30));
        assert_eq!(m.expected_disagreement(&[(A, 30), (C, 30)]), 2.0 * two_trials_at(30));
        assert_eq!(m.expected_disagreement(&[(A, 40), (C, 40)]), 2.0 * two_trials_at(40));
        assert_eq!(m.expected_disagreement(&[]), 0.0, "no observation, nothing expected");
        assert_eq!(m.expected_disagreement(&[(A, 2), (C, 2)]), 1.0, "the per-column clamp");
        assert_eq!(
            m.expected_disagreement(&[(A, 2); 10]),
            1.0,
            "and it does not grow past 1"
        );
    }

    #[test]
    fn the_origin_test_rejects_at_the_measured_number_of_disagreeing_columns() {
        let m = ConsensusModel::default();
        assert_eq!(m.omega_log10(), OMEGA_LOG10);
        for (q, want_k) in [(40u8, 4u64), (30, 6), (8, 57)] {
            let lambda = 80.0 * m.expected_disagreement(&[(A, q), (A, q)]);
            assert!(
                m.rejects_same_origin(want_k, lambda),
                "two Q{q} reads must be rejected at {want_k} disagreements in 80 columns \
                 (lambda {lambda}, surprise {})",
                m.origin_surprise(want_k, lambda)
            );
            assert!(
                !m.rejects_same_origin(want_k - 1, lambda),
                "...and NOT at {} (lambda {lambda}, surprise {})",
                want_k - 1,
                m.origin_surprise(want_k - 1, lambda)
            );
        }
        for lambda in [1e-9f64, 0.176, 25.4] {
            assert!(!m.rejects_same_origin(0, lambda));
        }
        let lambda = 80.0 * m.expected_disagreement(&[(A, 30), (A, 30)]);
        let s: Vec<f64> = (0..12).map(|k| m.origin_surprise(k, lambda)).collect();
        assert!(
            s.windows(2).all(|w| w[1] > w[0]),
            "surprise must be monotone in k: {s:?}"
        );
    }
}
