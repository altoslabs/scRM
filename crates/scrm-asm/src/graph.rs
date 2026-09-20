pub const WALK_RULE: &str = "walk2-consume-tip-anchored-rejoin-flank-seeded";

use debruijn::compression::{SimpleCompress, compress_kmers_with_hash};
use debruijn::filter::filter_kmers;
use debruijn::graph::DebruijnGraph;
use debruijn::kmer::{KmerSize, VarIntKmer};
use debruijn::{Dir, DnaBytes, Exts};
use scrm_core::Token;

use crate::columns::{self, Placement, ReadIndex};
use crate::consensus::{Call, ConsensusModel};
use crate::summarize::WeightSummarizer;

pub const STRANDED: bool = true;
const _: () = assert!(STRANDED, "rc-canonicalisation would scramble strand orientation");

const FILTER_MEMORY_GB: usize = 1;

pub type MoleculeGraph<KS> = DebruijnGraph<VarIntKmer<u64, KS>, u32>;

pub fn build<KS: KmerSize>(reads: &[scrm_core::MoleculeRead]) -> MoleculeGraph<KS> {
    let k = KS::K();
    let seqs: Vec<(DnaBytes, Exts, u32)> = reads
        .iter()
        .map(|r| {
            let (seq, weight) = (&r.seq, r.weight());
            assert!(
                seq.len() >= k,
                "read of length {} is shorter than k={k}; trimming should guarantee >= {}",
                seq.len(),
                scrm_core::min_trim_len_for(k)
            );
            debug_assert!(seq.iter().all(|&b| b < 4), "reads carry base token ids 0..=3 only");
            debug_assert!(
                r.quals.iter().all(|q| q.is_empty() || q.len() == seq.len()),
                "a quality vector must match its sequence length or be absent"
            );
            (DnaBytes(seq.clone()), Exts::empty(), weight)
        })
        .collect();

    let (index, _all_kmers) = filter_kmers::<VarIntKmer<u64, KS>, DnaBytes, u32, u32, WeightSummarizer>(
        &seqs,
        &Box::new(WeightSummarizer),
        STRANDED,
        false,
        FILTER_MEMORY_GB,
    );

    let spec = SimpleCompress::new(|a: u32, b: &u32| a.min(*b));
    let graph = compress_kmers_with_hash(STRANDED, &spec, &index).finish_serial();

    assert!(
        graph.base.stranded,
        "graph is not stranded; k-mers were rc-canonicalised and orientation is lost"
    );
    graph
}

pub fn has_self_loop<KS: KmerSize>(graph: &MoleculeGraph<KS>) -> bool {
    (0..graph.len()).any(|id| {
        let node = graph.get_node(id);
        node.l_edges().iter().any(|&(t, _, _)| t == id) || node.r_edges().iter().any(|&(t, _, _)| t == id)
    })
}

struct Step {
    bases: Vec<u8>,
    next: Option<usize>,
}

pub struct Walk<'g, KS: KmerSize> {
    graph: &'g MoleculeGraph<KS>,
    reads: &'g [scrm_core::MoleculeRead],
    model: &'g ConsensusModel,
    overlap: usize,
    index: Option<ReadIndex>,
    used: Vec<bool>,
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

impl<'g, KS: KmerSize> Walk<'g, KS> {
    pub fn new(graph: &'g MoleculeGraph<KS>, reads: &'g [scrm_core::MoleculeRead], model: &'g ConsensusModel) -> Self {
        Walk {
            graph,
            reads,
            model,
            overlap: KS::K() - 1,
            index: None,
            used: vec![false; graph.len()],
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

    pub fn contigs(&mut self) -> Vec<Vec<u8>> {
        for id in 0..self.graph.len() {
            if self.successors(id, Dir::Right).contains(&id) || self.successors(id, Dir::Left).contains(&id) {
                self.used[id] = true;
            }
        }

        let mut seeds: Vec<usize> = (0..self.graph.len()).collect();
        seeds.sort_by_key(|&id| (self.is_branch_member(id), std::cmp::Reverse(self.weight(id)), id));

        let mut contigs = Vec::new();
        for seed in seeds {
            if !self.used[seed] {
                contigs.push(self.walk_from(seed));
            }
        }
        debug_assert_eq!(
            self.bubble_n_tokens,
            self.n_tie + self.n_low_confidence + self.n_indel_interior,
            "an N was emitted without a recorded cause"
        );
        contigs
    }

    fn weight(&self, id: usize) -> u32 {
        *self.graph.get_node(id).data()
    }

    fn bases(&self, id: usize) -> Vec<u8> {
        self.graph.get_node(id).sequence().bytes()
    }

    fn base_len(&self, id: usize) -> usize {
        self.bases(id).len()
    }

    fn is_branch_member(&self, id: usize) -> bool {
        [Dir::Left, Dir::Right].iter().any(|&from| {
            let toward = match from {
                Dir::Left => Dir::Right,
                Dir::Right => Dir::Left,
            };
            self.successors(id, from)
                .into_iter()
                .any(|p| self.successors(p, toward).len() >= 2)
        })
    }

    fn successors(&self, id: usize, dir: Dir) -> Vec<usize> {
        self.graph
            .get_node(id)
            .edges(dir)
            .iter()
            .map(|&(target, _, rc)| {
                assert!(
                    !rc,
                    "reverse-complement edge in a stranded graph: STRANDED was not honoured"
                );
                target
            })
            .collect()
    }

    fn successor_set(&self, id: usize, dir: Dir) -> Vec<usize> {
        let mut s = self.successors(id, dir);
        s.sort_unstable();
        s
    }

    fn walk_from(&mut self, seed: usize) -> Vec<u8> {
        self.used[seed] = true;

        let mut right = self.bases(seed);
        let mut current = seed;
        while let Some(step) = self.step(current, Dir::Right) {
            debug_assert!(
                !step.bases[..self.overlap].contains(&(Token::N as u8)),
                "tie N inside the shared overlap"
            );
            right.extend_from_slice(&step.bases[self.overlap..]);
            match step.next {
                Some(next) => current = next,
                None => break,
            }
        }

        let mut left: Vec<u8> = Vec::new();
        current = seed;
        while let Some(step) = self.step(current, Dir::Left) {
            let novel = step.bases.len() - self.overlap;
            debug_assert!(
                !step.bases[novel..].contains(&(Token::N as u8)),
                "tie N inside the shared overlap"
            );
            left.extend(step.bases[..novel].iter().rev());
            match step.next {
                Some(next) => current = next,
                None => break,
            }
        }
        left.reverse();

        left.extend_from_slice(&right);
        left
    }

    fn step(&mut self, current: usize, dir: Dir) -> Option<Step> {
        let candidates: Vec<usize> = self
            .successors(current, dir)
            .into_iter()
            .filter(|&t| !self.used[t])
            .collect();
        if candidates.is_empty() {
            return None;
        }

        let bases = match candidates[..] {
            [only] => self.bases(only),
            _ => {
                let branches: Vec<Vec<u8>> = candidates.iter().map(|&t| self.bases(t)).collect();
                let equal_len = branches.iter().all(|b| b.len() == branches[0].len());
                let agreed_prefix = common_prefix_len(&branches);
                let agreed_suffix = common_suffix_len(&branches);
                let far_anchor = match dir {
                    Dir::Right => agreed_suffix,
                    Dir::Left => agreed_prefix,
                };
                let rejoins = far_anchor >= self.overlap;
                if !equal_len && rejoins {
                    let (segment, mid) = indel_rejoining(&branches, dir, self.overlap, agreed_prefix, agreed_suffix)?;
                    if mid > 1 {
                        return None;
                    }
                    self.n_indel_interior += mid;
                    self.bubble_n_tokens += mid;
                    segment
                } else {
                    self.call_columns(&branches, dir)?
                }
            }
        };

        let joined = self.successor_set(candidates[0], dir);
        let rejoins = candidates.iter().all(|&t| self.successor_set(t, dir) == joined);
        for &t in &candidates {
            self.used[t] = true;
        }

        let next = if rejoins {
            Some(candidates[0])
        } else {
            self.continuation(&candidates, dir)
        };
        Some(Step { bases, next })
    }

    fn continuation(&self, candidates: &[usize], dir: Dir) -> Option<usize> {
        let longest = candidates.iter().map(|&t| self.base_len(t)).max()?;
        candidates
            .iter()
            .copied()
            .filter(|&t| self.base_len(t) == longest && !self.successor_set(t, dir).is_empty())
            .max_by_key(|&t| (self.weight(t), std::cmp::Reverse(t)))
    }

    fn call_columns(&mut self, branches: &[Vec<u8>], dir: Dir) -> Option<Vec<u8>> {
        let max_len = branches.iter().map(Vec::len).max()?;
        if max_len <= self.overlap {
            return None;
        }
        if self.index.is_none() {
            self.index = Some(ReadIndex::build(self.reads, self.overlap + 1));
        }

        let origin = |len: usize| match dir {
            Dir::Right => 0,
            Dir::Left => max_len - len,
        };

        let mut placed: Vec<Placement> = Vec::new();
        let ambiguous = {
            let index = self.index.as_ref().expect("built immediately above");
            index.place(branches, origin, &mut placed)
        };
        self.n_ambiguous_reads += u64::from(ambiguous);
        if placed.is_empty() {
            return None;
        }

        let (flank, called) = match dir {
            Dir::Right => (0..self.overlap, self.overlap..max_len),
            Dir::Left => (max_len - self.overlap..max_len, 0..max_len - self.overlap),
        };

        let mut obs: Vec<(u8, u8)> = Vec::new();
        let mut disagreeing: u64 = 0;
        let mut lambda = 0.0f64;
        for pos in called.clone() {
            let n_reads = columns::observations(self.reads, &placed, pos, &mut obs);
            if n_reads < 2 {
                continue;
            }
            let first = obs[0].0;
            if obs.iter().any(|&(b, _)| b != first) {
                disagreeing += 1;
            }
            lambda += self.model.expected_disagreement(&obs);
        }

        self.n_branch_decisions += 1;
        self.origin_k_total += disagreeing;
        self.origin_lambda_total += lambda;
        let surprise = self.model.origin_surprise(disagreeing, lambda);
        let bucket = if surprise.is_finite() {
            surprise.round().clamp(0.0, 60.0) as u8
        } else {
            60
        };
        self.origin_surprise_max = self.origin_surprise_max.max(bucket);
        if self.model.rejects_same_origin(disagreeing, lambda) {
            self.n_origin_splits += 1;
            return None;
        }

        let mut out = vec![Token::N as u8; max_len];
        let longest = branches.iter().max_by_key(|b| b.len()).expect("non-empty");
        out[flank.clone()].copy_from_slice(&longest[flank]);

        let (mut ties, mut low) = (0u64, 0u64);
        let (mut run, mut longest_run) = (0usize, 0usize);
        let mut min_q = self.min_called_q;
        for pos in called {
            columns::observations(self.reads, &placed, pos, &mut obs);
            let call = self.model.call_column(&obs);
            out[pos] = call.token();
            match call {
                Call::Base(_, q) => {
                    run = 0;
                    min_q = min_q.min(q);
                }
                Call::Tie => {
                    ties += 1;
                    run += 1;
                }
                Call::LowConfidence(_) => {
                    low += 1;
                    run += 1;
                }
            }
            longest_run = longest_run.max(run);
        }

        if longest_run >= 2 {
            return None;
        }
        self.n_tie += ties;
        self.n_low_confidence += low;
        self.bubble_n_tokens += ties + low;
        self.min_called_q = min_q;
        Some(out)
    }
}

#[inline]
fn common_prefix_len(branches: &[Vec<u8>]) -> usize {
    let first = &branches[0];
    (0..first.len())
        .take_while(|&i| branches[1..].iter().all(|b| b.get(i) == Some(&first[i])))
        .count()
}

#[inline]
fn common_suffix_len(branches: &[Vec<u8>]) -> usize {
    let first = &branches[0];
    (1..=first.len())
        .take_while(|&i| {
            branches[1..]
                .iter()
                .all(|b| b.len() >= i && b[b.len() - i] == first[first.len() - i])
        })
        .count()
}

fn indel_rejoining(
    branches: &[Vec<u8>],
    dir: Dir,
    overlap: usize,
    agreed_prefix: usize,
    agreed_suffix: usize,
) -> Option<(Vec<u8>, u64)> {
    let max_len = branches.iter().map(Vec::len).max()?;
    let template = branches.iter().max_by_key(|b| b.len())?;
    let mid = branches
        .iter()
        .map(|b| b.len().saturating_sub(agreed_prefix + agreed_suffix))
        .max()?
        .max(1);
    let (prefix, suffix) = match dir {
        Dir::Right => (agreed_prefix, max_len.checked_sub(agreed_prefix + mid)?),
        Dir::Left => (max_len.checked_sub(mid + agreed_suffix)?, agreed_suffix),
    };
    if dir.pick(suffix, prefix) < overlap {
        return None;
    }
    let mut out = Vec::with_capacity(max_len);
    out.extend_from_slice(&template[..prefix]);
    out.extend(std::iter::repeat_n(Token::N as u8, mid));
    out.extend_from_slice(&template[max_len - suffix..]);
    debug_assert_eq!(out.len(), max_len, "indel segment must be max(L_i) long");
    Some((out, mid as u64))
}
