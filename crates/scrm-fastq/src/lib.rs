pub mod decode;
pub mod ids;
pub mod stream;

use scrm_chem::{ChemError, Chemistry};
use scrm_core::{CellId, Error, MIN_TRIM_LEN, ParsedPair, Token, Umi, pack_acgt};
use scrm_trim::{ArtifactCounters, TrimOutcome, TrimSettings, Trimmer};

pub use stream::{StreamConfig, StreamError, stream_pairs};

#[derive(Debug, Clone, Copy)]
pub struct RawRead<'a> {
    pub id: &'a [u8],
    pub seq: &'a [u8],
    pub qual: &'a [u8],
}

impl<'a> RawRead<'a> {
    pub fn new(id: &'a [u8], seq: &'a [u8], qual: &'a [u8]) -> Self {
        Self { id, seq, qual }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropReason {
    Gated,
    Geometry,
    AmbiguousBase,
    TooShort,
    Malformed,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    Kept(ParsedPair),
    Dropped(DropReason),
}

impl Outcome {
    pub fn dropped(&self) -> Option<DropReason> {
        match self {
            Outcome::Kept(_) => None,
            Outcome::Dropped(reason) => Some(*reason),
        }
    }

    pub fn kept(self) -> Option<ParsedPair> {
        match self {
            Outcome::Kept(pair) => Some(pair),
            Outcome::Dropped(_) => None,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Counters {
    pub pairs_in: u64,
    pub dropped_malformed_record: u64,
    pub dropped_geometry: u64,
    pub dropped_n_cbumi: u64,
    pub dropped_gated: u64,
    pub dropped_short: u64,
    pub dropped_malformed_trim: u64,
    pub dropped_n_read: u64,
    pub kept: u64,
    pub artifacts: ArtifactCounters,
}

impl Counters {
    pub fn merge(&mut self, other: &Counters) {
        self.pairs_in += other.pairs_in;
        self.dropped_malformed_record += other.dropped_malformed_record;
        self.dropped_geometry += other.dropped_geometry;
        self.dropped_n_cbumi += other.dropped_n_cbumi;
        self.dropped_gated += other.dropped_gated;
        self.dropped_short += other.dropped_short;
        self.dropped_malformed_trim += other.dropped_malformed_trim;
        self.dropped_n_read += other.dropped_n_read;
        self.kept += other.kept;
        self.artifacts.merge(&other.artifacts);
    }

    pub fn survival(&self) -> f64 {
        if self.pairs_in == 0 {
            return 0.0;
        }
        self.kept as f64 / self.pairs_in as f64
    }

    pub fn dropped_n(&self) -> u64 {
        self.dropped_n_cbumi + self.dropped_n_read
    }

    pub fn dropped_malformed(&self) -> u64 {
        self.dropped_malformed_record + self.dropped_malformed_trim
    }

    pub fn geometry_accepted(&self) -> u64 {
        self.pairs_in - self.dropped_malformed_record - self.dropped_geometry
    }

    pub fn gate_offered(&self) -> u64 {
        self.geometry_accepted() - self.dropped_n_cbumi
    }

    pub fn trimmed(&self) -> u64 {
        self.gate_offered() - self.dropped_gated
    }

    pub fn balanced(&self) -> bool {
        self.kept
            + self.dropped_malformed_record
            + self.dropped_geometry
            + self.dropped_n_cbumi
            + self.dropped_gated
            + self.dropped_short
            + self.dropped_malformed_trim
            + self.dropped_n_read
            == self.pairs_in
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error(transparent)]
    Chemistry(#[from] ChemError),
}

impl std::fmt::Debug for Parser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parser")
            .field("chemistry", &self.chemistry)
            .field("gate", &"<fn>")
            .finish_non_exhaustive()
    }
}

pub struct Parser {
    gate: Box<dyn Fn(u64) -> Option<CellId> + Send + Sync>,
    chemistry: Chemistry,
    trimmer: Trimmer,
}

impl Parser {
    pub fn new(
        geometry: &str,
        settings: TrimSettings,
        gate: impl Fn(u64) -> Option<CellId> + Send + Sync + 'static,
    ) -> Result<Self, ParseError> {
        Ok(Self {
            gate: Box::new(gate),
            chemistry: Chemistry::new(geometry)?,
            trimmer: Trimmer::new(settings),
        })
    }

    pub fn chemistry(&self) -> &Chemistry {
        &self.chemistry
    }

    pub fn trimmer(&self) -> &Trimmer {
        &self.trimmer
    }

    pub fn parse_pair(
        &self,
        index: u64,
        r1: RawRead<'_>,
        r2: RawRead<'_>,
        counters: &mut Counters,
    ) -> Result<Outcome, Error> {
        let gate = |cb: u64| (self.gate)(cb);
        counters.pairs_in += 1;

        if !ids::matches(r1.id, r2.id) {
            return Err(Error::PairIdMismatch {
                index,
                r1: ids::display(r1.id),
                r2: ids::display(r2.id),
            });
        }

        if r2.seq.len() != r2.qual.len() {
            counters.dropped_malformed_record += 1;
            return Ok(Outcome::Dropped(DropReason::Malformed));
        }

        let Some(seg) = self.chemistry.extract(r1.seq, r2.seq) else {
            counters.dropped_geometry += 1;
            return Ok(Outcome::Dropped(DropReason::Geometry));
        };
        let qual = &r2.qual[seg.read_offset..seg.read_offset + seg.read.len()];

        let (Some(cb_packed), Some(umi)) = (pack_acgt(seg.cb), pack_acgt(seg.umi)) else {
            counters.dropped_n_cbumi += 1;
            return Ok(Outcome::Dropped(DropReason::AmbiguousBase));
        };
        let Some(cell) = gate(cb_packed) else {
            counters.dropped_gated += 1;
            return Ok(Outcome::Dropped(DropReason::Gated));
        };

        let trimmed = match self.trimmer.trim(seg.read, qual, &mut counters.artifacts) {
            Ok(TrimOutcome::Kept(t)) => t,
            Ok(TrimOutcome::TooShort) => {
                counters.dropped_short += 1;
                return Ok(Outcome::Dropped(DropReason::TooShort));
            }
            Err(_) => {
                counters.dropped_malformed_trim += 1;
                return Ok(Outcome::Dropped(DropReason::Malformed));
            }
        };

        let mut tokens = Vec::with_capacity(trimmed.seq.len());
        for &b in &trimmed.seq {
            match Token::from_acgt(b) {
                Some(token) => tokens.push(token as u8),
                None => {
                    counters.dropped_n_read += 1;
                    return Ok(Outcome::Dropped(DropReason::AmbiguousBase));
                }
            }
        }

        debug_assert!(tokens.len() >= MIN_TRIM_LEN);
        debug_assert!(tokens.iter().all(|&t| t <= Token::T as u8));
        counters.kept += 1;
        let offset = self.trimmer.settings().phred_offset;
        let qual: Vec<u8> = trimmed.qual.iter().map(|&q| q.saturating_sub(offset)).collect();
        debug_assert_eq!(qual.len(), tokens.len(), "quality must match the trimmed sequence");
        Ok(Outcome::Kept(ParsedPair {
            cb: cell,
            umi: Umi(umi),
            r2: tokens,
            qual,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEOMETRY: &str = "1{b[16]u[12]}2{r:}";
    const CB: &[u8] = b"ACGTACGTACGTACGT";
    const UMI: &[u8] = b"TTTTGGGGCCCC";
    const CDNA: &[u8] = b"GCTAGCTAGGCATCGATCGGATCCAAGCTTGGATCCTAGCTAGCTAGGCATCGATCGGAT";

    fn parser() -> Parser {
        Parser::new(GEOMETRY, TrimSettings::default(), |cb| Some(scrm_core::CellId(cb))).unwrap()
    }

    fn r1_bytes() -> Vec<u8> {
        let mut v = CB.to_vec();
        v.extend_from_slice(UMI);
        v
    }

    fn qual(n: usize) -> Vec<u8> {
        vec![b'I'; n]
    }

    fn parse(r1_id: &[u8], r1_seq: &[u8], r2_id: &[u8], r2_seq: &[u8]) -> (Result<Outcome, Error>, Counters) {
        let r2q = qual(r2_seq.len());
        parse_with_qual(r1_id, r1_seq, r2_id, r2_seq, &r2q)
    }

    fn parse_with_qual(
        r1_id: &[u8],
        r1_seq: &[u8],
        r2_id: &[u8],
        r2_seq: &[u8],
        r2q: &[u8],
    ) -> (Result<Outcome, Error>, Counters) {
        let mut counters = Counters::default();
        let r1q = qual(r1_seq.len());
        let out = parser().parse_pair(
            7,
            RawRead::new(r1_id, r1_seq, &r1q),
            RawRead::new(r2_id, r2_seq, r2q),
            &mut counters,
        );
        (out, counters)
    }

    fn kept(r2_seq: &[u8]) -> ParsedPair {
        match parse(b"R", &r1_bytes(), b"R", r2_seq).0.unwrap() {
            Outcome::Kept(p) => p,
            Outcome::Dropped(r) => panic!("expected the pair to survive, dropped: {r:?}"),
        }
    }

    #[test]
    fn a_clean_pair_is_kept_with_cb_umi_and_tokens() {
        let pair = kept(CDNA);
        assert_eq!(pair.cb, CellId(pack_acgt(CB).unwrap()));
        assert_eq!(pair.umi, Umi(pack_acgt(UMI).unwrap()));
        assert_eq!(pair.r2.len(), CDNA.len());
        let expect: Vec<u8> = CDNA.iter().map(|&b| Token::from_acgt(b).unwrap() as u8).collect();
        assert_eq!(pair.r2, expect);
        assert!(pair.r2.iter().all(|&t| t <= 3));
        assert_eq!(
            pair.qual,
            vec![40u8; CDNA.len()],
            "qual must be numeric Phred, not ASCII"
        );
    }

    #[test]
    fn r2_qualities_arrive_as_numeric_phred_at_their_own_positions() {
        let ascii: Vec<u8> = (0..CDNA.len()).map(|i| 33 + i as u8).collect();
        let pair = match parse_with_qual(b"R", &r1_bytes(), b"R", CDNA, &ascii).0.unwrap() {
            Outcome::Kept(p) => p,
            Outcome::Dropped(r) => panic!("a Q0-and-up read is not dropped by trimming: {r:?}"),
        };
        let want: Vec<u8> = ascii.iter().map(|&q| q - 33).collect();
        assert_eq!(pair.qual.len(), pair.r2.len(), "one Phred per retained token");
        assert_eq!(pair.qual, want[..pair.qual.len()], "offset 33, per position");
        assert_eq!(pair.qual[0], 0, "ASCII 33 is Q0, not Q33 and not saturated away");

        let low = vec![32u8; CDNA.len()];
        if let Outcome::Kept(p) = parse_with_qual(b"R", &r1_bytes(), b"R", CDNA, &low).0.unwrap() {
            assert!(p.qual.iter().all(|&q| q == 0), "sub-offset bytes saturate at 0");
        }
    }

    #[test]
    fn mismatched_ids_return_pair_id_mismatch() {
        let err = parse(b"READ001", &r1_bytes(), b"READ002", CDNA).0.unwrap_err();
        match err {
            Error::PairIdMismatch { index, r1, r2 } => {
                assert_eq!(index, 7);
                assert_eq!(r1, "READ001");
                assert_eq!(r2, "READ002");
            }
            other => panic!("expected PairIdMismatch, got {other:?}"),
        }
    }

    #[test]
    fn mate_suffixes_are_not_a_mismatch() {
        let out = parse(b"READ001/1", &r1_bytes(), b"READ001/2", CDNA).0;
        assert!(matches!(out, Ok(Outcome::Kept(_))), "{out:?}");
    }

    #[test]
    fn comment_fields_are_not_a_mismatch() {
        let out = parse(
            b"A00519:1:HK:1:1101:1:1 1:N:0:AGGTTCGA",
            &r1_bytes(),
            b"A00519:1:HK:1:1101:1:1 2:N:0:AGGTTCGA",
            CDNA,
        )
        .0;
        assert!(matches!(out, Ok(Outcome::Kept(_))), "{out:?}");
    }

    #[test]
    fn mismatched_ids_with_comments_still_error() {
        let err = parse(
            b"A00519:1:HK:1:1101:1:1 1:N:0:AGGTTCGA",
            &r1_bytes(),
            b"A00519:1:HK:1:1101:9:9 2:N:0:AGGTTCGA",
            CDNA,
        )
        .0
        .unwrap_err();
        assert!(matches!(err, Error::PairIdMismatch { .. }), "{err:?}");
    }

    #[test]
    fn mismatched_ids_with_mate_suffixes_still_error() {
        let err = parse(b"READ001/1", &r1_bytes(), b"READ002/2", CDNA).0.unwrap_err();
        assert!(matches!(err, Error::PairIdMismatch { .. }), "{err:?}");
    }

    #[test]
    fn the_id_check_precedes_every_other_check() {
        let err = parse(b"A", b"", b"B", b"").0.unwrap_err();
        assert!(matches!(err, Error::PairIdMismatch { .. }), "{err:?}");
    }

    #[test]
    fn r1_is_byte_identical_after_processing() {
        let r1 = r1_bytes();
        let original = r1.clone();
        let r1q = qual(r1.len());
        let original_qual = r1q.clone();
        let r2q = qual(CDNA.len());
        let mut counters = Counters::default();
        let out = parser()
            .parse_pair(
                0,
                RawRead::new(b"R", &r1, &r1q),
                RawRead::new(b"R", CDNA, &r2q),
                &mut counters,
            )
            .unwrap();
        assert!(matches!(out, Outcome::Kept(_)));
        assert_eq!(r1, original, "R1 sequence bytes changed");
        assert_eq!(r1q, original_qual, "R1 quality bytes changed");
    }

    #[test]
    fn r1_is_untouched_even_when_it_looks_trimmable() {
        let mut r1 = CB.to_vec();
        r1.extend_from_slice(UMI);
        r1.extend_from_slice(b"TTTCTTATATGGG");
        r1.extend_from_slice(&[b'A'; 40]);
        r1.extend_from_slice(b"AGATCGGAAGAGC");
        let original = r1.clone();
        let r1q = qual(r1.len());
        let r2q = qual(CDNA.len());
        let mut counters = Counters::default();
        parser()
            .parse_pair(
                0,
                RawRead::new(b"R", &r1, &r1q),
                RawRead::new(b"R", CDNA, &r2q),
                &mut counters,
            )
            .unwrap();
        assert_eq!(r1, original);
        assert_eq!(counters.kept, 1);
        assert_eq!(counters.artifacts.adapter, 0);
    }

    #[test]
    fn n_in_the_cell_barcode_discards_the_pair() {
        let mut r1 = r1_bytes();
        r1[3] = b'N';
        let (out, counters) = parse(b"R", &r1, b"R", CDNA);
        assert_eq!(
            out.unwrap().dropped(),
            Some(DropReason::AmbiguousBase),
            "N in CB must discard"
        );
        assert_eq!((counters.dropped_n_cbumi, counters.dropped_n_read), (1, 0));
        assert_eq!(counters.kept, 0);
        assert_eq!(r1[3], b'N');
    }

    #[test]
    fn n_in_the_umi_discards_the_pair() {
        let mut r1 = r1_bytes();
        r1[20] = b'N';
        let (out, counters) = parse(b"R", &r1, b"R", CDNA);
        assert_eq!(out.unwrap().dropped(), Some(DropReason::AmbiguousBase));
        assert_eq!((counters.dropped_n_cbumi, counters.dropped_n_read), (1, 0));
    }

    #[test]
    fn n_in_the_retained_r2_discards_the_pair() {
        let mut r2 = CDNA.to_vec();
        r2[30] = b'N';
        let (out, counters) = parse(b"R", &r1_bytes(), b"R", &r2);
        assert_eq!(out.unwrap().dropped(), Some(DropReason::AmbiguousBase));
        assert_eq!(
            (counters.dropped_n_cbumi, counters.dropped_n_read),
            (0, 1),
            "an N in R2 is a POST-gate drop and must not be counted with the pre-gate ones"
        );
        assert_eq!(r2[30], b'N', "input must not be rewritten");
    }

    #[test]
    fn no_position_in_a_retained_region_substitutes_a_base() {
        for pos in 0..CDNA.len() {
            let mut r2 = CDNA.to_vec();
            r2[pos] = b'N';
            let (out, _) = parse(b"R", &r1_bytes(), b"R", &r2);
            assert_eq!(
                out.unwrap().dropped(),
                Some(DropReason::AmbiguousBase),
                "position {pos} was not discarded"
            );
        }
        for pos in 0..(CB.len() + UMI.len()) {
            let mut r1 = r1_bytes();
            r1[pos] = b'N';
            let (out, _) = parse(b"R", &r1, b"R", CDNA);
            assert_eq!(
                out.unwrap().dropped(),
                Some(DropReason::AmbiguousBase),
                "R1 position {pos} was not discarded"
            );
        }
    }

    /// AN N IN THE INSERT DISCARDS THE PAIR. The base is unknown and the assembly cannot use it.
    #[test]
    fn an_n_in_the_retained_insert_discards() {
        let mut r2 = CDNA.to_vec();
        r2[10] = b'N';
        let (out, counters) = parse(b"R", &r1_bytes(), b"R", &r2);
        assert_eq!(
            out.unwrap().dropped(),
            Some(DropReason::AmbiguousBase),
            "an N the trimming keeps must discard the pair"
        );
        assert_eq!(counters.dropped_n(), 1);
    }

    /// AN N THAT TRIMMING REMOVES DOES NOT. It never reaches the assembly, so discarding the pair
    /// would throw away a usable insert -- the same principle as
    /// `n_in_the_discarded_part_of_r1_does_not_discard`.
    #[test]
    fn an_n_in_the_trimmed_tail_of_r2_does_not_discard() {
        let mut r2 = CDNA.to_vec();
        let mut tail = vec![b'A'; 30];
        tail[10] = b'N';
        r2.extend_from_slice(&tail);
        let (out, _) = parse(b"R", &r1_bytes(), b"R", &r2);
        assert_eq!(
            out.unwrap().dropped(),
            None,
            "the poly-A tail carrying the N is trimmed away"
        );
    }

    #[test]
    fn n_in_the_discarded_part_of_r1_does_not_discard() {
        let mut r1 = r1_bytes();
        r1.extend_from_slice(b"NNNNNNNNNNNNN");
        let (out, _) = parse(b"R", &r1, b"R", CDNA);
        assert!(matches!(out.unwrap(), Outcome::Kept(_)));
    }

    #[test]
    fn a_natively_short_read_is_dropped() {
        let r2 = CDNA[..15].to_vec();
        let (out, counters) = parse(b"R", &r1_bytes(), b"R", &r2);
        assert_eq!(out.unwrap().dropped(), Some(DropReason::TooShort));
        assert_eq!(counters.dropped_short, 1);
        assert_eq!(counters.artifacts.dropped_short, 1);
        assert!(counters.balanced());
    }

    #[test]
    fn an_r2_shorter_than_min_trim_len_is_dropped_untrimmed() {
        let (out, counters) = parse(b"R", &r1_bytes(), b"R", &CDNA[..10]);
        assert_eq!(out.unwrap().dropped(), Some(DropReason::TooShort));
        assert_eq!(counters.dropped_short, 1);
    }

    #[test]
    fn kept_r2_always_reaches_min_trim_len() {
        let pair = kept(&CDNA[..MIN_TRIM_LEN]);
        assert_eq!(pair.r2.len(), MIN_TRIM_LEN);
    }

    #[test]
    fn an_r1_too_short_for_the_geometry_is_dropped() {
        let (out, counters) = parse(b"R", &CB[..10], b"R", CDNA);
        assert_eq!(out.unwrap().dropped(), Some(DropReason::Geometry));
        assert_eq!(counters.dropped_geometry, 1);
        assert!(counters.balanced());
    }

    #[test]
    fn a_umi_length_change_moves_the_umi_and_nothing_else() {
        let r1 = r1_bytes();
        let r1q = qual(r1.len());
        let r2q = qual(CDNA.len());
        let mut kept = Vec::new();
        for geometry in ["1{b[16]u[10]}2{r:}", "1{b[16]u[12]}2{r:}"] {
            let parser = Parser::new(geometry, TrimSettings::default(), |cb| Some(scrm_core::CellId(cb))).unwrap();
            let mut counters = Counters::default();
            let out = parser
                .parse_pair(
                    0,
                    RawRead::new(b"R", &r1, &r1q),
                    RawRead::new(b"R", CDNA, &r2q),
                    &mut counters,
                )
                .unwrap();
            match out {
                Outcome::Kept(p) => kept.push(p),
                other => panic!("{geometry}: {other:?}"),
            }
        }
        assert_eq!(kept[0].cb, kept[1].cb, "CB must not move");
        assert_eq!(kept[0].r2, kept[1].r2, "R2 must not move");
        assert_ne!(kept[0].umi, kept[1].umi, "the UMI must");
        assert_eq!(kept[1].umi, Umi(pack_acgt(UMI).unwrap()));
        assert_eq!(kept[0].umi, Umi(pack_acgt(&UMI[..10]).unwrap()));
    }

    #[test]
    fn malformed_r2_quality_length_is_dropped_not_guessed() {
        let mut counters = Counters::default();
        let r1 = r1_bytes();
        let out = parser()
            .parse_pair(
                0,
                RawRead::new(b"R", &r1, &qual(r1.len())),
                RawRead::new(b"R", CDNA, b"III"),
                &mut counters,
            )
            .unwrap();
        assert_eq!(out.dropped(), Some(DropReason::Malformed));
        assert_eq!(
            (counters.dropped_malformed_record, counters.dropped_malformed_trim),
            (1, 0)
        );
        assert!(counters.balanced());
    }

    #[test]
    fn a_non_ascii_r2_is_counted_as_malformed_not_as_short() {
        let mut r2 = CDNA.to_vec();
        r2[10] = 0xFF;
        let (out, c) = parse(b"R", &r1_bytes(), b"R", &r2);
        assert_eq!(out.unwrap().dropped(), Some(DropReason::Malformed));
        assert_eq!(
            (c.dropped_malformed_trim, c.dropped_short),
            (1, 0),
            "corruption is malformed, and it was found after the gate"
        );
        assert!(c.balanced(), "{c:?}");
    }

    #[test]
    fn artifacts_are_measured_on_pairs_that_are_then_dropped() {
        let r2 = vec![b'T'; 15];
        assert!(r2.len() < scrm_core::MIN_TRIM_LEN);
        let (out, counters) = parse(b"R", &r1_bytes(), b"R", &r2);
        assert_eq!(out.unwrap().dropped(), Some(DropReason::TooShort));
        assert_eq!(counters.artifacts.reads, 1, "measured before the floor");
        assert_eq!(
            counters
                .artifacts
                .homopolymer_count(b'T', scrm_trim::RunClass::FivePrime, 10),
            1,
            "the run on a DROPPED read is still counted: the rate describes the data, not the survivors"
        );
    }

    #[test]
    fn each_drop_counter_is_counted_over_the_population_that_reached_its_stage() {
        let admit = pack_acgt(CB).unwrap();
        let parser = Parser::new(GEOMETRY, TrimSettings::default(), move |cb| {
            (cb == admit).then_some(CellId(cb))
        })
        .unwrap();
        let mut c = Counters::default();
        let feed = |r1: &[u8], r2: &[u8], counters: &mut Counters| {
            let (r1q, r2q) = (qual(r1.len()), qual(r2.len()));
            parser
                .parse_pair(0, RawRead::new(b"R", r1, &r1q), RawRead::new(b"R", r2, &r2q), counters)
                .unwrap()
                .dropped()
        };

        assert_eq!(feed(&CB[..4], CDNA, &mut c), Some(DropReason::Geometry));
        let mut n_cb = r1_bytes();
        n_cb[3] = b'N';
        assert_eq!(feed(&n_cb, CDNA, &mut c), Some(DropReason::AmbiguousBase));
        let mut other = r1_bytes();
        other[0] = if CB[0] == b'A' { b'C' } else { b'A' };
        assert_eq!(feed(&other, CDNA, &mut c), Some(DropReason::Gated));
        let mut n_r2 = CDNA.to_vec();
        n_r2[30] = b'N';
        assert_eq!(feed(&r1_bytes(), &n_r2, &mut c), Some(DropReason::AmbiguousBase));
        assert_eq!(feed(&r1_bytes(), CDNA, &mut c), None);

        assert_eq!(
            (c.dropped_n_cbumi, c.dropped_n_read),
            (1, 1),
            "the two `N` drops are different populations and must be counted apart"
        );
        assert_eq!(c.dropped_n(), 2, "and the stamp's single field is their sum");
        assert_eq!((c.pairs_in, c.kept, c.dropped_geometry, c.dropped_gated), (5, 1, 1, 1));

        assert_eq!(c.geometry_accepted(), 4);
        assert_eq!(
            c.gate_offered(),
            3,
            "the CB-N pair never reached the gate, so it is not in dropped_cb's denominator"
        );
        assert_eq!(c.trimmed(), 2);
        assert!(c.balanced(), "{c:?}");
    }

    #[test]
    fn the_pair_ledger_balances_when_the_gate_rejects_most_pairs() {
        let admit = pack_acgt(CB).unwrap();
        let parser = Parser::new(GEOMETRY, TrimSettings::default(), move |cb| {
            (cb == admit).then_some(CellId(cb))
        })
        .unwrap();
        let mut c = Counters::default();
        for i in 0..20u8 {
            let mut r1 = r1_bytes();
            if i % 3 != 0 {
                r1[0] = if CB[0] == b'A' { b'C' } else { b'A' };
                r1[1] = b"ACGT"[(i % 4) as usize];
            }
            let (r1q, r2q) = (qual(r1.len()), qual(CDNA.len()));
            parser
                .parse_pair(
                    i as u64,
                    RawRead::new(b"R", &r1, &r1q),
                    RawRead::new(b"R", CDNA, &r2q),
                    &mut c,
                )
                .unwrap();
        }
        assert!(
            c.dropped_gated > 0,
            "the gate must actually reject, or this proves nothing"
        );
        assert!(c.balanced(), "{c:?}");
        assert_eq!(c.dropped_gated + c.kept, c.gate_offered());
    }

    #[test]
    fn counters_merge_and_stay_balanced() {
        let (_, a) = parse(b"R", &r1_bytes(), b"R", CDNA);
        let (_, b) = parse(b"R", &CB[..4], b"R", CDNA);
        let mut merged = a.clone();
        merged.merge(&b);
        assert_eq!(merged.pairs_in, 2);
        assert_eq!(merged.kept, 1);
        assert_eq!(merged.dropped_geometry, 1);
        assert!(merged.balanced());
        assert_eq!(merged.survival(), 0.5);
    }

    #[test]
    fn accounting_holds_for_arbitrary_input() {
        use proptest::prelude::*;
        let base = prop_oneof![Just(b'A'), Just(b'C'), Just(b'G'), Just(b'T'), Just(b'N')];
        proptest!(|(r1 in proptest::collection::vec(base.clone(), 0..40usize),
                    r2 in proptest::collection::vec(base, 0..120usize))| {
            let mut counters = Counters::default();
            let out = parser().parse_pair(
                0,
                RawRead::new(b"R", &r1, &qual(r1.len())),
                RawRead::new(b"R", &r2, &qual(r2.len())),
                &mut counters,
            ).unwrap();
            prop_assert!(counters.balanced());
            if let Outcome::Kept(p) = out {
                prop_assert!(p.r2.len() >= MIN_TRIM_LEN);
                prop_assert!(p.r2.iter().all(|&t| t <= 3));
            }
        });
    }
}
