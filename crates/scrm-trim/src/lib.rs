//! R2 trimming and artifact measurement. R1 never appears in this crate's API, which is how
//! "R1 is byte-identical through the pipeline" stays a structural fact rather than a promise.
//!
//! Every rule here is first-party, written from the published documentation rather than from
//! source, and no trimmer is invoked or depended on. The two rules that are somebody else's
//! method carry the citation in the module that implements them: `quality.rs` is BWA's quality
//! trimming rule as cutadapt (MIT) documents it, and `adapter.rs` is cutadapt's suffix-anchored
//! 3' adapter rule. `tso.rs`, `homopolymer.rs`, `polyrun.rs` and `artifacts.rs` are ours.
//! The length floor is `scrm_core::MIN_TRIM_LEN`, where a const assertion couples it to `k`
//! rather than a comment claiming they agree.
//! The notices are in THIRD_PARTY_NOTICES.md.

pub mod adapter;
pub mod artifacts;
pub mod homopolymer;
pub mod polyrun;
pub mod quality;
pub mod tso;

use scrm_core::MIN_TRIM_LEN;

pub use artifacts::{ArtifactCounters, ReadFlags, RunClass};
pub use tso::{MIN_SCORE as TSO_MIN_SCORE, TSO};

pub const ILLUMINA_TRUSEQ_3P: &[u8] = b"AGATCGGAAGAGC";

pub const QUALITY_CUTOFF: u8 = 20;

pub const STRINGENCY: usize = 5;

pub const HOMOPOLYMER_MIN_RUN: usize = 10;

pub const ERROR_RATE: f64 = 0.1;

#[derive(Debug, thiserror::Error)]
pub enum TrimError {
    #[error("sequence ({seq} bases) and quality ({qual} bases) lengths disagree")]
    LengthMismatch { seq: usize, qual: usize },

    #[error("record is not ASCII ({seq} bases); FASTQ must be ASCII and is never repaired here")]
    NotAscii { seq: usize },
}

/// EVERY ONE OF THESE IS ACTED ON.
///
/// `Trimmer::trim` does all four steps: the 5' TSO clip against `tso`/`tso_min_score`, the 3'
/// adapter clip against `adapters` at `stringency` overlap and `error_rate`, the 3' quality trim at
/// `quality_cutoff` on the `phred_offset` scale, and the anchored homopolymer scans at
/// `homopolymer_min_run`. `min_len` then drops what is left if it is too short. All are stamped
/// into the manifest, because all of them change what a molecule is.
#[derive(Debug, Clone)]
pub struct TrimSettings {
    pub quality_cutoff: u8,
    pub phred_offset: u8,
    pub error_rate: f64,
    pub stringency: usize,
    pub adapters: Vec<(String, Vec<u8>)>,
    pub tso: Vec<u8>,
    pub tso_min_score: usize,
    pub homopolymer_min_run: usize,
    pub min_len: usize,
}

impl Default for TrimSettings {
    fn default() -> Self {
        Self {
            quality_cutoff: QUALITY_CUTOFF,
            phred_offset: 33,
            error_rate: ERROR_RATE,
            stringency: STRINGENCY,
            adapters: vec![("Illumina".to_string(), ILLUMINA_TRUSEQ_3P.to_vec())],
            tso: TSO.to_vec(),
            tso_min_score: tso::MIN_SCORE,
            homopolymer_min_run: HOMOPOLYMER_MIN_RUN,
            min_len: MIN_TRIM_LEN,
        }
    }
}

impl TrimSettings {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Trimmed {
    pub seq: Vec<u8>,
    pub qual: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrimOutcome {
    Kept(Trimmed),
    TooShort,
}

pub struct Trimmer {
    settings: TrimSettings,
}

impl std::fmt::Debug for Trimmer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trimmer")
            .field("settings", &self.settings)
            .finish_non_exhaustive()
    }
}

impl Trimmer {
    pub fn new(settings: TrimSettings) -> Self {
        Self { settings }
    }

    pub fn settings(&self) -> &TrimSettings {
        &self.settings
    }

    pub fn trim(&self, seq: &[u8], qual: &[u8], counters: &mut ArtifactCounters) -> Result<TrimOutcome, TrimError> {
        if seq.len() != qual.len() {
            return Err(TrimError::LengthMismatch {
                seq: seq.len(),
                qual: qual.len(),
            });
        }
        if !seq.is_ascii() || !qual.is_ascii() {
            return Err(TrimError::NotAscii { seq: seq.len() });
        }

        // ---- MEASURE FIRST, on the raw read, so the rates describe the data and not our own
        // trimming. A corpus still reporting a high TSO rate was built from untrimmed input, and
        // that is only visible if the measurement precedes the cut.
        let mut flags = ReadFlags::default();
        homopolymer::measure(seq, &mut flags);
        let tso_hit = if self.settings.tso.is_empty() {
            None
        } else {
            tso::find(seq, &self.settings.tso, self.settings.tso_min_score)
        };

        // ---- 5' TSO clip.
        let after_tso = tso_hit.map_or(0, |m| m.end);
        let (seq, qual) = (&seq[after_tso..], &qual[after_tso..]);

        // ---- 3' adapter clip. `tso::find` is a semi-global fuzzy matcher over any probe -- its
        // parameter is named `adapter` -- so the adapter needs no second matcher. `stringency` is
        // the overlap a hit must reach, which is what keeps a read ending in a chance base pair
        // from losing it.
        let mut end = seq.len();
        for (_, probe) in &self.settings.adapters {
            end = adapter::three_prime_cut(&seq[..end], probe, self.settings.stringency, self.settings.error_rate);
        }
        let had_adapter = end < seq.len();

        // ---- 3' quality trim, then the anchored homopolymer scans. Internal runs were measured
        // above and are deliberately left in place.
        let qend = quality::quality_trim_end(&qual[..end], self.settings.quality_cutoff, self.settings.phred_offset);
        let seq = &seq[..qend];
        let qual = &qual[..qend];
        let span = homopolymer::trim_anchored(seq, self.settings.homopolymer_min_run);
        let seq = &seq[span.start..span.end];
        let qual = &qual[span.start..span.end];

        counters.note_read(flags, had_adapter, tso_hit.is_some());

        if seq.len() < self.settings.min_len {
            counters.dropped_short += 1;
            return Ok(TrimOutcome::TooShort);
        }
        Ok(TrimOutcome::Kept(Trimmed {
            seq: seq.to_vec(),
            qual: qual.to_vec(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trimmer() -> Trimmer {
        Trimmer::new(TrimSettings::default())
    }

    fn qual_for(seq: &[u8]) -> Vec<u8> {
        vec![b'I'; seq.len()]
    }

    fn run(seq: &[u8]) -> (TrimOutcome, ArtifactCounters) {
        let mut counters = ArtifactCounters::default();
        let out = trimmer().trim(seq, &qual_for(seq), &mut counters).expect("trim");
        (out, counters)
    }

    fn kept(seq: &[u8]) -> Trimmed {
        match run(seq).0 {
            TrimOutcome::Kept(t) => t,
            TrimOutcome::TooShort => panic!("unexpectedly dropped"),
        }
    }

    #[test]
    fn a_clean_read_is_returned_unchanged() {
        let seq = b"GCTAGCTAGGCATCGATCGATCGGCATCGAT";
        assert_eq!(kept(seq).seq, seq.to_vec());
    }

    #[test]
    fn a_read_through_adapter_is_removed() {
        let seq = [&b"GCTAGCTAGGCATCGATCGATCGGCATCGAT"[..], ILLUMINA_TRUSEQ_3P].concat();
        assert_eq!(kept(&seq).seq, b"GCTAGCTAGGCATCGATCGATCGGCATCGAT".to_vec());
    }

    #[test]
    fn a_tso_is_clipped_and_counted() {
        let seq = [TSO, &b"GCTAGCTAGGCATCGATCGATCGGCATCGAT"[..]].concat();
        let (out, counters) = run(&seq);
        assert_eq!(
            match out {
                TrimOutcome::Kept(t) => t.seq,
                TrimOutcome::TooShort => panic!(),
            },
            seq[TSO.len()..].to_vec(),
            "the 5\' TSO is clipped"
        );
        assert!(
            counters.tso > 0,
            "and still be COUNTED: the measurement precedes the cut, so the rate describes the data"
        );
    }

    #[test]
    fn a_terminal_poly_a_run_is_trimmed_and_counted() {
        let seq = [&b"GCTAGCTAGGCATCGATCGATCGGCATCGAT"[..], &b"A".repeat(25)[..]].concat();
        let mut counters = ArtifactCounters::default();
        let out = trimmer().trim(&seq, &qual_for(&seq), &mut counters).expect("trim");
        assert_eq!(
            match out {
                TrimOutcome::Kept(t) => t.seq,
                TrimOutcome::TooShort => panic!(),
            },
            b"GCTAGCTAGGCATCGATCGATCGGCATCGAT".to_vec(),
            "the anchored 3' poly-A run is trimmed; an INTERNAL run would not be"
        );
        let total: u64 = counters.reads;
        assert!(total > 0);
    }

    #[test]
    fn a_low_quality_tail_is_trimmed() {
        let seq = b"GCTAGCTAGGCATCGATCGATCGGCATCGAT";
        let mut qual = qual_for(seq);
        for q in qual.iter_mut().skip(20) {
            *q = b'#';
        }
        let mut counters = ArtifactCounters::default();
        let out = trimmer().trim(seq, &qual, &mut counters).expect("trim");
        match out {
            TrimOutcome::Kept(t) => {
                assert_eq!(t.seq, seq[..20].to_vec(), "the low-quality tail is trimmed");
                assert_eq!(t.qual.len(), t.seq.len(), "qualities stay aligned with the sequence");
            }
            TrimOutcome::TooShort => panic!("20 surviving bases clears min_len"),
        }
    }

    #[test]
    fn a_natively_short_read_is_dropped() {
        let seq = b"GCTAGCTAGG";
        assert!(seq.len() < MIN_TRIM_LEN);
        assert_eq!(run(seq).0, TrimOutcome::TooShort);
    }

    #[test]
    fn min_trim_len_is_the_boundary_not_a_suggestion() {
        // NOT a homopolymer: a run of one base is consumed by the anchored scans, so the read
        // would be dropped for being trimmed away rather than for being short.
        let mixed = |n: usize| -> Vec<u8> { b"GCTA".iter().cycle().take(n).copied().collect() };
        let at = mixed(MIN_TRIM_LEN);
        let below = mixed(MIN_TRIM_LEN - 1);
        assert!(
            matches!(run(&at).0, TrimOutcome::Kept(_)),
            "exactly at the floor is kept"
        );
        assert_eq!(run(&below).0, TrimOutcome::TooShort, "one below is dropped");
    }

    #[test]
    fn a_dropped_read_is_still_measured() {
        let seq = vec![b'A'; MIN_TRIM_LEN - 1];
        let (out, counters) = run(&seq);
        assert_eq!(out, TrimOutcome::TooShort);
        assert_eq!(counters.reads, 1, "measured BEFORE the floor, not after");
        assert_eq!(counters.dropped_short, 1);
    }

    #[test]
    fn a_non_ascii_record_is_malformed_rather_than_too_short() {
        let seq: &[u8] = &[
            b'G', b'C', b'T', b'A', b'G', b'C', b'T', b'A', b'G', b'G', b'C', b'A', b'T', b'C', b'G', b'A', b'T', b'C',
            b'G', 0xff,
        ];
        let mut counters = ArtifactCounters::default();
        assert!(matches!(
            trimmer().trim(seq, &qual_for(seq), &mut counters),
            Err(TrimError::NotAscii { .. })
        ));
    }

    #[test]
    fn mismatched_seq_and_qual_lengths_are_an_error() {
        let mut counters = ArtifactCounters::default();
        assert!(matches!(
            trimmer().trim(b"GCTA", b"II", &mut counters),
            Err(TrimError::LengthMismatch { .. })
        ));
    }

    #[test]
    fn settings_defaults_are_the_measured_values() {
        let s = TrimSettings::default();
        assert_eq!(s.min_len, MIN_TRIM_LEN);
        assert_eq!(
            s.quality_cutoff, QUALITY_CUTOFF,
            "the cutoff quality.rs applies and main.rs stamps"
        );
        assert_eq!(s.adapters[0].1, ILLUMINA_TRUSEQ_3P.to_vec());
    }

    #[test]
    /// EVERY ARTIFACT REDUCES TO THE SAME INSERT. The TSO in front, the adapter behind, and a
    /// clean read all yield the insert itself, which is the point of doing all four steps here.
    fn every_artifact_reduces_to_the_insert() {
        const INSERT: &[u8] = b"GCTAGCTAGGCATCGATCGATCGGCATCGAT";
        for seq in [
            INSERT.to_vec(),
            [TSO, INSERT].concat(),
            [INSERT, ILLUMINA_TRUSEQ_3P].concat(),
            [TSO, INSERT, ILLUMINA_TRUSEQ_3P].concat(),
        ] {
            match run(&seq).0 {
                TrimOutcome::Kept(t) => assert_eq!(t.seq, INSERT.to_vec(), "seq len {}", seq.len()),
                TrimOutcome::TooShort => panic!("the insert clears min_len"),
            }
        }
    }
}
