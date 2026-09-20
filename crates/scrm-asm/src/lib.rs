//! Per-(CB, UMI) local assembly: one UMI is one captured mRNA, so a molecule has exactly one true
//! sequence and every read disagreement is an error to be resolved rather than a variant to keep.
//!
//! The graph is built with the `debruijn` crate (10XGenomics/rust-debruijn, MIT) at k = 19 --
//! `graph.rs` calls its `filter_kmers` and `compress_kmers_with_hash`, with this crate's
//! `WeightSummarizer` as the k-mer filter. `columns.rs` carries each read's Phred to the branch
//! point and `consensus.rs` calls every disputed column by quality-weighted likelihood, which is
//! fgbio's `ConsensusCaller` reimplemented -- that module's own header carries the citation. The
//! notices are in THIRD_PARTY_NOTICES.md.

pub mod columns;
pub mod consensus;
mod graph;
mod stats;
mod summarize;

pub use graph::{MoleculeGraph, STRANDED, WALK_RULE, build as build_graph, has_self_loop};
pub use stats::{Retention, StatsCollector};
pub use summarize::{MIN_KMER_OBS, WeightSummarizer};

use scrm_core::{Molecule, MoleculeContigs, Token};

pub fn default_model() -> &'static consensus::ConsensusModel {
    static MODEL: std::sync::LazyLock<consensus::ConsensusModel> =
        std::sync::LazyLock::new(consensus::ConsensusModel::default);
    &MODEL
}

pub fn resolution_name() -> String {
    format!(
        "likelihood-q{}-pre{}-post{}-omega{}-{}",
        consensus::MIN_CONSENSUS_Q,
        consensus::ERROR_RATE_PRE,
        consensus::ERROR_RATE_POST,
        consensus::OMEGA_LOG10,
        graph::WALK_RULE
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Diagnostics {
    pub graph_built: bool,
    pub self_loop: bool,
    pub bubble_n_tokens: u64,
    pub n_tie: u64,
    pub n_low_confidence: u64,
    pub n_indel_interior: u64,
    pub n_ambiguous_reads: u64,
    pub n_branch_decisions: u64,
    pub n_origin_splits: u64,
    pub origin_surprise_max: u8,
    pub min_called_q: u8,
    pub origin_k_total: u64,
    pub origin_lambda_total: f64,
}

impl Default for Diagnostics {
    fn default() -> Self {
        Diagnostics {
            graph_built: false,
            self_loop: false,
            bubble_n_tokens: 0,
            n_tie: 0,
            n_low_confidence: 0,
            n_indel_interior: 0,
            n_ambiguous_reads: 0,
            n_branch_decisions: 0,
            n_origin_splits: 0,
            origin_surprise_max: 0,
            min_called_q: u8::MAX,
            origin_k_total: 0,
            origin_lambda_total: 0.0,
        }
    }
}

pub fn assemble(molecule: &Molecule, model: &consensus::ConsensusModel, k: usize) -> (MoleculeContigs, Diagnostics) {
    if molecule.is_pass_through() {
        let tokens = molecule.reads.first().map(|r| r.seq.clone()).unwrap_or_default();
        debug_assert!(tokens.iter().all(|&t| t < 4), "reads carry base token ids 0..=3 only");
        let n_contigs = u16::from(!tokens.is_empty());
        return (emit(molecule, tokens, n_contigs), Diagnostics::default());
    }

    scrm_core::dispatch_k!(k, |KS| { assemble_at::<KS>(molecule, model) })
}

fn assemble_at<KS: debruijn::kmer::KmerSize>(
    molecule: &Molecule,
    model: &consensus::ConsensusModel,
) -> (MoleculeContigs, Diagnostics) {
    let graph = graph::build::<KS>(&molecule.reads);
    let self_loop = graph::has_self_loop(&graph);

    let mut walk = graph::Walk::new(&graph, &molecule.reads, model);
    let contigs = walk.contigs();
    let diagnostics = Diagnostics {
        graph_built: true,
        self_loop,
        bubble_n_tokens: walk.bubble_n_tokens,
        n_tie: walk.n_tie,
        n_low_confidence: walk.n_low_confidence,
        n_indel_interior: walk.n_indel_interior,
        n_ambiguous_reads: walk.n_ambiguous_reads,
        n_branch_decisions: walk.n_branch_decisions,
        n_origin_splits: walk.n_origin_splits,
        origin_surprise_max: walk.origin_surprise_max,
        min_called_q: walk.min_called_q,
        origin_k_total: walk.origin_k_total,
        origin_lambda_total: walk.origin_lambda_total,
    };

    let mut tokens = Vec::with_capacity(contigs.iter().map(|c| c.len() + 1).sum());
    for (i, contig) in contigs.iter().enumerate() {
        if i > 0 {
            tokens.push(Token::Tab as u8);
        }
        tokens.extend_from_slice(contig);
    }
    let n_contigs = u16::try_from(contigs.len()).expect("molecule produced more than u16::MAX contigs");

    (emit(molecule, tokens, n_contigs), diagnostics)
}

fn emit(molecule: &Molecule, tokens: Vec<u8>, n_contigs: u16) -> MoleculeContigs {
    debug_assert!(
        !tokens.contains(&(Token::Newline as u8)),
        "Newline is inserted at packing time and never stored"
    );
    MoleculeContigs {
        cell: molecule.cell,
        umi: molecule.umi,
        n_reads: molecule.n_reads,
        tokens,
        n_contigs,
    }
}
