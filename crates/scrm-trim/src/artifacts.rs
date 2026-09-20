use std::collections::BTreeMap;

use scrm_core::stamp::ArtifactRates;

pub const RUN_THRESHOLDS: [usize; 3] = [10, 20, 30];

pub const BASES: [u8; 4] = *b"ACGT";

const CLASSES: [&str; 3] = ["fwd_anchored", "rev_anchored", "fwd_internal"];

const N_FLAGS: usize = BASES.len() * CLASSES.len() * RUN_THRESHOLDS.len();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunClass {
    ThreePrime,
    FivePrime,
    Internal,
}

impl RunClass {
    fn index(self) -> usize {
        match self {
            RunClass::ThreePrime => 0,
            RunClass::FivePrime => 1,
            RunClass::Internal => 2,
        }
    }
}

fn flag_index(base: u8, class: RunClass, thr_idx: usize) -> Option<usize> {
    let base_idx = BASES.iter().position(|&b| b == base)?;
    Some((base_idx * CLASSES.len() + class.index()) * RUN_THRESHOLDS.len() + thr_idx)
}

fn flag_key(flag: usize) -> String {
    let thr_idx = flag % RUN_THRESHOLDS.len();
    let rest = flag / RUN_THRESHOLDS.len();
    let class_idx = rest % CLASSES.len();
    let base_idx = rest / CLASSES.len();
    format!(
        "{}_{}_ge{}",
        BASES[base_idx] as char, CLASSES[class_idx], RUN_THRESHOLDS[thr_idx]
    )
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ReadFlags {
    bits: u64,
}

impl ReadFlags {
    pub fn note_run(&mut self, base: u8, class: RunClass, len: usize) {
        for (thr_idx, &thr) in RUN_THRESHOLDS.iter().enumerate() {
            if len >= thr
                && let Some(flag) = flag_index(base, class, thr_idx)
            {
                self.bits |= 1u64 << flag;
            }
        }
    }

    pub fn has(&self, base: u8, class: RunClass, thr: usize) -> bool {
        let Some(thr_idx) = RUN_THRESHOLDS.iter().position(|&t| t == thr) else {
            return false;
        };
        flag_index(base, class, thr_idx).is_some_and(|f| self.bits & (1u64 << f) != 0)
    }
}

#[derive(Debug, Clone)]
pub struct ArtifactCounters {
    pub reads: u64,
    pub adapter: u64,
    pub tso: u64,
    pub dropped_short: u64,
    homopolymer: [u64; N_FLAGS],
}

impl Default for ArtifactCounters {
    fn default() -> Self {
        Self {
            reads: 0,
            adapter: 0,
            tso: 0,
            dropped_short: 0,
            homopolymer: [0; N_FLAGS],
        }
    }
}

impl ArtifactCounters {
    pub fn note_read(&mut self, flags: ReadFlags, had_adapter: bool, had_tso: bool) {
        self.reads += 1;
        if had_adapter {
            self.adapter += 1;
        }
        if had_tso {
            self.tso += 1;
        }
        for flag in 0..N_FLAGS {
            if flags.bits & (1u64 << flag) != 0 {
                self.homopolymer[flag] += 1;
            }
        }
    }

    pub fn merge(&mut self, other: &ArtifactCounters) {
        self.reads += other.reads;
        self.adapter += other.adapter;
        self.tso += other.tso;
        self.dropped_short += other.dropped_short;
        for (a, b) in self.homopolymer.iter_mut().zip(other.homopolymer.iter()) {
            *a += b;
        }
    }

    pub fn homopolymer_count(&self, base: u8, class: RunClass, thr: usize) -> u64 {
        let Some(thr_idx) = RUN_THRESHOLDS.iter().position(|&t| t == thr) else {
            return 0;
        };
        flag_index(base, class, thr_idx).map_or(0, |f| self.homopolymer[f])
    }

    pub fn rates(&self) -> ArtifactRates {
        let denom = if self.reads == 0 { 1.0 } else { self.reads as f64 };
        let mut homopolymer = BTreeMap::new();
        for flag in 0..N_FLAGS {
            homopolymer.insert(flag_key(flag), self.homopolymer[flag] as f64 / denom);
        }
        ArtifactRates {
            adapter_fraction: self.adapter as f64 / denom,
            tso_fraction: self.tso as f64 / denom,
            homopolymer,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_format_matches_the_stamp_doc_example() {
        let mut flags = ReadFlags::default();
        flags.note_run(b'T', RunClass::Internal, 25);
        let mut counters = ArtifactCounters::default();
        counters.note_read(flags, false, false);
        let rates = counters.rates();
        assert_eq!(rates.homopolymer["T_fwd_internal_ge20"], 1.0);
        assert_eq!(rates.homopolymer["T_fwd_internal_ge10"], 1.0);
        assert_eq!(rates.homopolymer["T_fwd_internal_ge30"], 0.0);
        assert_eq!(rates.homopolymer.len(), N_FLAGS);
    }

    #[test]
    fn a_run_counts_once_per_read_per_key() {
        let mut flags = ReadFlags::default();
        flags.note_run(b'A', RunClass::ThreePrime, 12);
        flags.note_run(b'A', RunClass::ThreePrime, 40);
        let mut counters = ArtifactCounters::default();
        counters.note_read(flags, true, true);
        assert_eq!(counters.homopolymer_count(b'A', RunClass::ThreePrime, 10), 1);
        assert_eq!(counters.rates().adapter_fraction, 1.0);
        assert_eq!(counters.rates().tso_fraction, 1.0);
    }

    #[test]
    fn merge_sums_every_field() {
        let mut flags = ReadFlags::default();
        flags.note_run(b'G', RunClass::ThreePrime, 30);
        let mut a = ArtifactCounters::default();
        a.note_read(flags, true, false);
        let mut b = ArtifactCounters::default();
        b.note_read(flags, false, true);
        b.dropped_short = 3;
        a.merge(&b);
        assert_eq!(a.reads, 2);
        assert_eq!(a.adapter, 1);
        assert_eq!(a.tso, 1);
        assert_eq!(a.dropped_short, 3);
        assert_eq!(a.homopolymer_count(b'G', RunClass::ThreePrime, 30), 2);
        assert_eq!(a.rates().homopolymer["G_fwd_anchored_ge30"], 1.0);
    }

    #[test]
    fn empty_counters_report_zero_not_nan() {
        let rates = ArtifactCounters::default().rates();
        assert_eq!(rates.adapter_fraction, 0.0);
        assert!(rates.homopolymer.values().all(|v| *v == 0.0));
    }
}
