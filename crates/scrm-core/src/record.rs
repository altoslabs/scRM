use crate::alphabet::Token;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CellId(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Umi(pub u64);

#[derive(Debug, Clone)]
pub struct ParsedPair {
    pub cb: CellId,
    pub umi: Umi,
    pub r2: Vec<u8>,
    pub qual: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Molecule {
    pub cell: CellId,
    pub umi: Umi,
    pub n_reads: u32,
    pub reads: Vec<MoleculeRead>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoleculeRead {
    pub seq: Vec<u8>,
    pub quals: Vec<Vec<u8>>,
}

impl MoleculeRead {
    #[inline]
    pub fn weight(&self) -> u32 {
        self.quals.len().max(1) as u32
    }

    pub fn unqualified(seq: Vec<u8>, weight: u32) -> Self {
        Self {
            seq,
            quals: vec![Vec::new(); weight as usize],
        }
    }
}

impl Molecule {
    #[inline]
    pub fn is_pass_through(&self) -> bool {
        self.reads.len() <= 1
    }
}

#[derive(Debug, Clone)]
pub struct MoleculeContigs {
    pub cell: CellId,
    pub umi: Umi,
    pub n_reads: u32,
    pub tokens: Vec<u8>,
    pub n_contigs: u16,
}

impl MoleculeContigs {
    pub fn recount_contigs(&self) -> usize {
        if self.tokens.is_empty() {
            return 0;
        }
        1 + self.tokens.iter().filter(|&&t| t == Token::Tab as u8).count()
    }
}

pub const MIN_UMI_LEN_DEFAULT: usize = 60;

pub const MAX_UMI_LEN_DEFAULT: usize = 1600;

pub const MIN_CELL_MOLECULES_DEFAULT: usize = 500;

const SEQ_LEN: usize = 16384;

const _: () = assert!(
    MAX_UMI_LEN_DEFAULT * 10 <= SEQ_LEN,
    "max_umi_len must stay an order of magnitude under seq_len, or a window can hold one molecule"
);
const _: () = assert!(MIN_UMI_LEN_DEFAULT < MAX_UMI_LEN_DEFAULT);

const HASH_OLIGO_TRIMMED_LEN: usize = 11;

const _: () = assert!(
    MIN_UMI_LEN_DEFAULT >= HASH_OLIGO_TRIMMED_LEN * 5,
    "min_umi_len must stay well clear of a trimmed hash oligo (11 tokens, measured on sci-Plex \
     SRR10401658), or a feature-barcode library's perturbation LABEL enters the corpus and any \
     perturbation task becomes label-reading rather than response prediction"
);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LenFilter {
    min: usize,
    max: usize,
    pub n_short: u64,
    pub tokens_short: u64,
    pub n_long: u64,
    pub tokens_long: u64,
}

impl LenFilter {
    pub fn new(min: usize, max: usize) -> crate::Result<Self> {
        if min == 0 || min > max {
            return Err(crate::Error::MoleculeBoundsInvalid { min, max });
        }
        Ok(Self {
            min,
            max,
            n_short: 0,
            tokens_short: 0,
            n_long: 0,
            tokens_long: 0,
        })
    }

    pub fn min(&self) -> usize {
        self.min
    }

    pub fn max(&self) -> usize {
        self.max
    }

    pub fn keep(&mut self, tokens_len: usize) -> bool {
        if tokens_len < self.min {
            self.n_short += 1;
            self.tokens_short += tokens_len as u64;
            false
        } else if tokens_len > self.max {
            self.n_long += 1;
            self.tokens_long += tokens_len as u64;
            false
        } else {
            true
        }
    }

    pub fn n_dropped(&self) -> u64 {
        self.n_short + self.n_long
    }

    pub fn tokens_dropped(&self) -> u64 {
        self.tokens_short + self.tokens_long
    }

    pub fn merge(&mut self, other: &Self) {
        assert_eq!(
            (self.min, self.max),
            (other.min, other.max),
            "shards filtered under different molecule bounds"
        );
        self.n_short += other.n_short;
        self.tokens_short += other.tokens_short;
        self.n_long += other.n_long;
        self.tokens_long += other.tokens_long;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pass_through_covers_single_sequence_molecules() {
        let m = Molecule {
            cell: CellId(0),
            umi: Umi(0),
            n_reads: 7,
            reads: vec![MoleculeRead::unqualified(vec![0, 1, 2, 3], 7)],
        };
        assert!(m.is_pass_through());
        assert_eq!(m.n_reads, 7);
    }

    #[test]
    fn recount_contigs_matches_separator_count() {
        let t = Token::Tab as u8;
        let mc = MoleculeContigs {
            cell: CellId(0),
            umi: Umi(0),
            n_reads: 1,
            tokens: vec![0, 1, t, 2, 3, t, 0],
            n_contigs: 3,
        };
        assert_eq!(mc.recount_contigs(), mc.n_contigs as usize);
    }

    #[test]
    fn len_filter_keeps_the_closed_interval_and_tallies_each_end() {
        let mut f = LenFilter::new(60, 1600).unwrap();
        assert!(f.keep(60), "min is inclusive");
        assert!(f.keep(1600), "max is inclusive");
        assert!(f.keep(90), "the modal molecule length must survive");
        assert!(!f.keep(59));
        assert!(!f.keep(1601));
        assert_eq!((f.n_short, f.tokens_short), (1, 59));
        assert_eq!((f.n_long, f.tokens_long), (1, 1601));
        assert_eq!(f.n_dropped(), 2);
        assert_eq!(f.tokens_dropped(), 59 + 1601);
    }

    #[test]
    fn len_filter_rejects_bounds_that_would_empty_a_corpus() {
        assert!(
            LenFilter::new(0, 1600).is_err(),
            "a zero floor keeps nothing meaningful"
        );
        assert!(LenFilter::new(1600, 60).is_err(), "an inverted range keeps nothing");
        assert!(
            LenFilter::new(60, 60).is_ok(),
            "a single admissible length is degenerate but valid"
        );
    }

    #[test]
    fn len_filter_merges_per_shard_tallies() {
        let mut a = LenFilter::new(60, 1600).unwrap();
        let mut b = LenFilter::new(60, 1600).unwrap();
        a.keep(10);
        b.keep(20);
        b.keep(9999);
        a.merge(&b);
        assert_eq!((a.n_short, a.tokens_short), (2, 30));
        assert_eq!((a.n_long, a.tokens_long), (1, 9999));
    }

    #[test]
    #[should_panic(expected = "different molecule bounds")]
    fn len_filter_refuses_to_merge_across_different_bounds() {
        let mut a = LenFilter::new(60, 1600).unwrap();
        let b = LenFilter::new(60, 1601).unwrap();
        a.merge(&b);
    }

    #[test]
    fn len_filter_admits_every_molecule_the_defaults_were_derived_from() {
        let mut f = LenFilter::new(MIN_UMI_LEN_DEFAULT, MAX_UMI_LEN_DEFAULT).unwrap();
        assert!(f.keep(673), "the longest single contig observed must survive");
        assert!(f.keep(1180), "the longest <=3-contig molecule observed must survive");
        assert!(
            !f.keep(1976),
            "the longest <=7-contig molecule is above the cap by design"
        );
        assert!(!f.keep(16385), "nothing may exceed seq_len");
    }

    #[test]
    fn newline_never_appears_in_stored_tokens() {
        let mc = MoleculeContigs {
            cell: CellId(0),
            umi: Umi(0),
            n_reads: 1,
            tokens: vec![0, 1, 2, 3],
            n_contigs: 1,
        };
        assert!(!mc.tokens.contains(&(Token::Newline as u8)));
    }
}
