use scrm_core::MoleculeRead;

pub const NOMINAL_Q: u8 = 30;

type Entry = (u64, u32, u32);

pub type Placement = (u32, i32);

pub struct ReadIndex {
    entries: Vec<Entry>,
    k: usize,
}

impl ReadIndex {
    pub fn build(reads: &[MoleculeRead], k: usize) -> Self {
        assert!(
            (1..=32).contains(&k),
            "k={k} does not pack into a u64 at 2 bits per base"
        );
        let mask = if k == 32 { u64::MAX } else { (1u64 << (2 * k)) - 1 };
        let total: usize = reads.iter().map(|r| r.seq.len().saturating_sub(k - 1)).sum();
        let mut entries: Vec<Entry> = Vec::with_capacity(total);
        for (ri, r) in reads.iter().enumerate() {
            let mut code = 0u64;
            for (i, &b) in r.seq.iter().enumerate() {
                debug_assert!(b < 4, "reads carry base token ids 0..=3 only");
                code = ((code << 2) | b as u64) & mask;
                if i + 1 >= k {
                    entries.push((code, ri as u32, (i + 1 - k) as u32));
                }
            }
        }
        entries.sort_unstable_by_key(|e| e.0);
        ReadIndex { entries, k }
    }

    #[inline]
    fn occurrences(&self, code: u64) -> &[Entry] {
        let lo = self.entries.partition_point(|e| e.0 < code);
        let hi = self.entries[lo..].partition_point(|e| e.0 == code);
        &self.entries[lo..lo + hi]
    }

    pub fn place(&self, candidates: &[Vec<u8>], origin: impl Fn(usize) -> usize, out: &mut Vec<Placement>) -> u32 {
        out.clear();
        let mut conflicted: Vec<u32> = Vec::new();
        for cand in candidates {
            if cand.len() < self.k {
                continue;
            }
            let mask = if self.k == 32 {
                u64::MAX
            } else {
                (1u64 << (2 * self.k)) - 1
            };
            let mut code = 0u64;
            for (i, &b) in cand.iter().enumerate() {
                code = ((code << 2) | b as u64) & mask;
                if i + 1 < self.k {
                    continue;
                }
                let cand_pos = origin(cand.len()) + i + 1 - self.k;
                for &(_, read, off) in self.occurrences(code) {
                    let delta = off as i32 - cand_pos as i32;
                    match out.iter_mut().find(|(r, _)| *r == read) {
                        Some((_, d)) if *d == delta => {}
                        Some((_, _)) => {
                            if !conflicted.contains(&read) {
                                conflicted.push(read);
                            }
                        }
                        None => out.push((read, delta)),
                    }
                }
            }
        }
        out.retain(|(r, _)| !conflicted.contains(r));
        conflicted.len() as u32
    }

    pub fn k(&self) -> usize {
        self.k
    }
}

pub fn observations(reads: &[MoleculeRead], placed: &[Placement], pos: usize, out: &mut Vec<(u8, u8)>) -> usize {
    out.clear();
    let mut n_reads = 0usize;
    for &(ri, delta) in placed {
        let read = &reads[ri as usize];
        let rp = pos as i32 + delta;
        if rp < 0 || rp as usize >= read.seq.len() {
            continue;
        }
        let rp = rp as usize;
        let base = read.seq[rp];
        n_reads += 1;
        if read.quals.is_empty() {
            out.push((base, NOMINAL_Q));
            continue;
        }
        for q in &read.quals {
            out.push((base, q.get(rp).copied().unwrap_or(NOMINAL_Q)));
        }
    }
    n_reads
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consensus::{Call, ConsensusModel};

    fn read(seq: &[u8], quals: Vec<Vec<u8>>) -> MoleculeRead {
        MoleculeRead {
            seq: seq.to_vec(),
            quals,
        }
    }

    fn seq(seed: u64, len: usize) -> Vec<u8> {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) % 4) as u8
            })
            .collect()
    }

    #[test]
    fn a_read_is_placed_at_the_offset_its_kmers_imply() {
        let base = seq(1, 40);
        let r = read(&base[7..], vec![]);
        let idx = ReadIndex::build(std::slice::from_ref(&r), 5);
        let mut placed = Vec::new();
        assert_eq!(idx.place(std::slice::from_ref(&base), |_| 0, &mut placed), 0);
        assert_eq!(placed, vec![(0, -7)], "read_pos = candidate_pos - 7");
    }

    #[test]
    fn a_read_starting_inside_the_candidate_is_still_placed_and_votes() {
        let flank = seq(2, 30);
        let mut cand = flank.clone();
        cand.extend(seq(3, 30));
        let inside = read(&cand[35..], vec![vec![40; cand.len() - 35]]);
        let idx = ReadIndex::build(std::slice::from_ref(&inside), 19);
        let mut placed = Vec::new();
        assert_eq!(idx.place(std::slice::from_ref(&cand), |_| 0, &mut placed), 0);
        assert_eq!(placed, vec![(0, -35)]);

        let mut obs = Vec::new();
        let n_reads = observations(std::slice::from_ref(&inside), &placed, 40, &mut obs);
        assert_eq!(
            obs,
            vec![(cand[40], 40)],
            "it must vote at column 40 with its real Phred"
        );
        assert_eq!(n_reads, 1);
    }

    #[test]
    fn a_read_that_does_not_reach_a_column_simply_does_not_vote() {
        let cand = seq(4, 60);
        let short = read(&cand[..30], vec![vec![37; 30]]);
        let idx = ReadIndex::build(std::slice::from_ref(&short), 19);
        let mut placed = Vec::new();
        idx.place(std::slice::from_ref(&cand), |_| 0, &mut placed);
        let mut obs = Vec::new();
        let n_reads = observations(std::slice::from_ref(&short), &placed, 10, &mut obs);
        assert_eq!(obs, vec![(cand[10], 37)], "inside its span it votes its own base");
        assert_eq!(n_reads, 1, "one distinct read reached column 10");

        let n_reads = observations(std::slice::from_ref(&short), &placed, 45, &mut obs);
        assert!(
            obs.is_empty(),
            "past its end it makes no claim, rather than voting a wrong base"
        );
        assert_eq!(
            n_reads, 0,
            "a read that did not vote is not a read that reached the column"
        );
    }

    #[test]
    fn each_observation_takes_the_quality_at_its_own_position() {
        let cand = seq(9, 40);
        let ramp: Vec<u8> = (0..40).map(|i| 2 + i as u8).collect();
        let r = read(&cand, vec![ramp.clone()]);
        let idx = ReadIndex::build(std::slice::from_ref(&r), 19);
        let mut placed = Vec::new();
        idx.place(std::slice::from_ref(&cand), |_| 0, &mut placed);
        let mut obs = Vec::new();
        for pos in [0usize, 1, 20, 39] {
            observations(std::slice::from_ref(&r), &placed, pos, &mut obs);
            assert_eq!(obs, vec![(cand[pos], ramp[pos])], "column {pos}");
        }
        let tail = read(&cand[7..], vec![ramp[7..].to_vec()]);
        let idx = ReadIndex::build(std::slice::from_ref(&tail), 19);
        idx.place(std::slice::from_ref(&cand), |_| 0, &mut placed);
        assert_eq!(placed, vec![(0, -7)]);
        observations(std::slice::from_ref(&tail), &placed, 20, &mut obs);
        assert_eq!(
            obs,
            vec![(cand[20], ramp[20])],
            "read_pos = 20 - 7 = 13, whose Phred is the candidate column's"
        );
    }

    #[test]
    fn a_right_aligned_candidate_group_places_reads_through_its_origin() {
        let long = seq(10, 40);
        let short = long[10..].to_vec();
        let r = read(&long, vec![vec![40; 40]]);
        let idx = ReadIndex::build(std::slice::from_ref(&r), 19);
        let mut placed = Vec::new();
        let skipped = idx.place(&[long.clone(), short.clone()], |len| 40 - len, &mut placed);
        assert_eq!(
            skipped, 0,
            "both candidates imply the same delta under the right origin"
        );
        assert_eq!(placed, vec![(0, 0)]);

        let skipped = idx.place(&[long, short], |_| 0, &mut placed);
        assert_eq!(skipped, 1, "left-aligning a right-aligned group must lose the read");
        assert!(placed.is_empty());
    }

    #[test]
    fn a_read_whose_kmers_imply_two_offsets_is_skipped_and_counted() {
        let unit = seq(5, 12);
        let repeated: Vec<u8> = unit.iter().chain(unit.iter()).copied().collect();
        let r = read(&unit, vec![]);
        let idx = ReadIndex::build(std::slice::from_ref(&r), 5);
        let mut placed = Vec::new();
        let skipped = idx.place(&[repeated], |_| 0, &mut placed);
        assert_eq!(skipped, 1);
        assert!(placed.is_empty(), "an ambiguous read must not vote at all");
    }

    #[test]
    fn every_observation_votes_with_its_own_quality() {
        let cand = seq(6, 40);
        let r = read(&cand, vec![vec![40; 40], vec![12; 40], vec![30; 40]]);
        let idx = ReadIndex::build(std::slice::from_ref(&r), 19);
        let mut placed = Vec::new();
        idx.place(std::slice::from_ref(&cand), |_| 0, &mut placed);
        let mut obs = Vec::new();
        let n_reads = observations(std::slice::from_ref(&r), &placed, 20, &mut obs);
        assert_eq!(obs, vec![(cand[20], 40), (cand[20], 12), (cand[20], 30)]);
        assert_eq!(n_reads, 1, "one read, however many observations it carries");
    }

    #[test]
    fn weight_counting_is_the_equal_quality_special_case() {
        let m = ConsensusModel::default();
        let cand = seq(7, 40);
        let mut variant = cand.clone();
        variant[20] = (variant[20] + 1) % 4;

        let gather = |a: u32, b: u32| -> Call {
            let reads = vec![
                MoleculeRead::unqualified(cand.clone(), a),
                MoleculeRead::unqualified(variant.clone(), b),
            ];
            let idx = ReadIndex::build(&reads, 19);
            let mut placed = Vec::new();
            idx.place(&[cand.clone(), variant.clone()], |_| 0, &mut placed);
            let mut obs = Vec::new();
            let n_reads = observations(&reads, &placed, 20, &mut obs);
            assert_eq!(
                n_reads, 2,
                "two distinct reads reached the column, whatever their weights"
            );
            assert_eq!(obs.len(), (a + b) as usize, "one observation per read observation");
            m.call_column(&obs)
        };
        assert_eq!(gather(1, 1), Call::Tie, "1:1 is a tie, hence N -- the old rule exactly");
        assert_eq!(gather(2, 1), Call::Base(cand[20], 34), "2:1 gives the majority base");
        assert_eq!(gather(1, 2), Call::Base(variant[20], 34), "and symmetrically");
        assert_eq!(gather(5, 5), Call::Tie, "any exact tie, at any depth");
    }

    #[test]
    fn an_index_refuses_a_k_that_would_not_pack() {
        let r = read(&seq(8, 40), vec![]);
        let msg = |k: usize| -> String {
            let prev = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {}));
            let out = std::panic::catch_unwind(|| ReadIndex::build(std::slice::from_ref(&r), k));
            std::panic::set_hook(prev);
            let payload = out.err().unwrap_or_else(|| panic!("k={k} must be refused"));
            payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        };
        for k in [0usize, 33, 64] {
            let m = msg(k);
            assert!(
                m.contains("does not pack into a u64"),
                "k={k} panicked with {m:?}, not the k-range assert"
            );
        }
        assert_eq!(ReadIndex::build(std::slice::from_ref(&r), 1).k(), 1);
        assert_eq!(ReadIndex::build(std::slice::from_ref(&r), 32).k(), 32);
    }
}
