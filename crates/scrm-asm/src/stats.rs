use scrm_core::stamp::AssemblyStats;
use scrm_core::{Molecule, MoleculeContigs, Token};

use crate::Diagnostics;

#[derive(Debug, Clone, Default)]
pub struct StatsCollector {
    molecules: u64,
    pass_through: u64,
    self_loop_molecules: u64,
    bubble_n_tokens: u64,
    bubble_n_tie: u64,
    bubble_n_low_confidence: u64,
    bubble_n_indel_interior: u64,
    ambiguous_read_placements: u64,
    branch_decisions: u64,
    origin_splits: u64,
    origin_surprise_hist: Vec<u64>,
    called_q_hist: Vec<u64>,
    origin_k_total: u64,
    origin_lambda_total: f64,
    reads_per_molecule: Vec<u64>,
    distinct_seqs_per_molecule: Vec<u64>,
    contigs_per_molecule: Vec<u64>,
    contig_length: Vec<u64>,
}

fn bump(hist: &mut Vec<u64>, index: usize) {
    if hist.len() <= index {
        hist.resize(index + 1, 0);
    }
    hist[index] += 1;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    Kept,
    LengthFiltered,
}

impl StatsCollector {
    pub fn record(
        &mut self,
        molecule: &Molecule,
        out: &MoleculeContigs,
        diagnostics: &Diagnostics,
        retention: Retention,
    ) {
        assert_eq!(
            out.n_contigs as usize,
            out.recount_contigs(),
            "n_contigs disagrees with the tokens"
        );

        if diagnostics.self_loop {
            self.self_loop_molecules += 1;
        }
        self.ambiguous_read_placements += diagnostics.n_ambiguous_reads;
        self.branch_decisions += diagnostics.n_branch_decisions;
        self.origin_splits += diagnostics.n_origin_splits;
        self.origin_k_total += diagnostics.origin_k_total;
        self.origin_lambda_total += diagnostics.origin_lambda_total;
        if diagnostics.n_branch_decisions > 0 {
            bump(&mut self.origin_surprise_hist, diagnostics.origin_surprise_max as usize);
        }
        if diagnostics.min_called_q != u8::MAX {
            bump(&mut self.called_q_hist, diagnostics.min_called_q as usize);
        }
        debug_assert_eq!(
            diagnostics.bubble_n_tokens,
            diagnostics.n_tie + diagnostics.n_low_confidence + diagnostics.n_indel_interior,
            "an N reached the stats without a recorded cause"
        );
        if retention == Retention::LengthFiltered {
            return;
        }

        self.molecules += 1;
        if !diagnostics.graph_built {
            self.pass_through += 1;
        }
        self.bubble_n_tokens += diagnostics.bubble_n_tokens;
        self.bubble_n_tie += diagnostics.n_tie;
        self.bubble_n_low_confidence += diagnostics.n_low_confidence;
        self.bubble_n_indel_interior += diagnostics.n_indel_interior;

        bump(&mut self.reads_per_molecule, molecule.n_reads as usize);
        bump(&mut self.distinct_seqs_per_molecule, molecule.reads.len());
        bump(&mut self.contigs_per_molecule, out.n_contigs as usize);
        if !out.tokens.is_empty() {
            for contig in out.tokens.split(|&t| t == Token::Tab as u8) {
                bump(&mut self.contig_length, contig.len());
            }
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.molecules += other.molecules;
        self.pass_through += other.pass_through;
        self.self_loop_molecules += other.self_loop_molecules;
        self.bubble_n_tokens += other.bubble_n_tokens;
        self.bubble_n_tie += other.bubble_n_tie;
        self.bubble_n_low_confidence += other.bubble_n_low_confidence;
        self.bubble_n_indel_interior += other.bubble_n_indel_interior;
        self.ambiguous_read_placements += other.ambiguous_read_placements;
        self.branch_decisions += other.branch_decisions;
        self.origin_splits += other.origin_splits;
        self.origin_k_total += other.origin_k_total;
        self.origin_lambda_total += other.origin_lambda_total;
        for (dst, src) in [
            (&mut self.reads_per_molecule, &other.reads_per_molecule),
            (&mut self.distinct_seqs_per_molecule, &other.distinct_seqs_per_molecule),
            (&mut self.contigs_per_molecule, &other.contigs_per_molecule),
            (&mut self.contig_length, &other.contig_length),
            (&mut self.origin_surprise_hist, &other.origin_surprise_hist),
            (&mut self.called_q_hist, &other.called_q_hist),
        ] {
            if dst.len() < src.len() {
                dst.resize(src.len(), 0);
            }
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += s;
            }
        }
    }

    pub fn molecules(&self) -> u64 {
        self.molecules
    }

    pub fn self_loop_molecules(&self) -> u64 {
        self.self_loop_molecules
    }

    pub fn bubble_n_tokens(&self) -> u64 {
        self.bubble_n_tokens
    }

    pub fn origin(&self) -> (u64, u64) {
        (self.branch_decisions, self.origin_splits)
    }

    pub fn bubble_n_causes(&self) -> (u64, u64, u64) {
        (
            self.bubble_n_tie,
            self.bubble_n_low_confidence,
            self.bubble_n_indel_interior,
        )
    }

    pub fn finish(&self) -> AssemblyStats {
        AssemblyStats {
            reads_per_molecule_hist: self.reads_per_molecule.clone(),
            distinct_seqs_per_molecule_hist: self.distinct_seqs_per_molecule.clone(),
            pass_through_fraction: if self.molecules == 0 {
                0.0
            } else {
                self.pass_through as f64 / self.molecules as f64
            },
            contigs_per_molecule_hist: self.contigs_per_molecule.clone(),
            contig_length_hist: self.contig_length.clone(),
            self_loop_molecules: self.self_loop_molecules,
            bubble_n_tokens: self.bubble_n_tokens,
            bubble_n_tie: self.bubble_n_tie,
            bubble_n_low_confidence: self.bubble_n_low_confidence,
            bubble_n_indel_interior: self.bubble_n_indel_interior,
            ambiguous_read_placements: self.ambiguous_read_placements,
            branch_decisions: self.branch_decisions,
            origin_splits: self.origin_splits,
            origin_surprise_hist: self.origin_surprise_hist.clone(),
            called_q_hist: self.called_q_hist.clone(),
            origin_k_total: self.origin_k_total,
            origin_lambda_total: self.origin_lambda_total,
        }
    }
}
