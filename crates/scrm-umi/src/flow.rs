//! `1MM_CR_flow`: the count-directed Hamming-1 chain rule.
//!
//! The method is CellRanger's `1MM_CR_flow` (10x Genomics), reimplemented here from its published
//! description; no CellRanger code is in this tree. The notice is in THIRD_PARTY_NOTICES.md.
//!
//! Each UMI points at the Hamming-1 neighbour with the highest read count, and the inequality is
//! STRICT: equal counts yield no edge at all, which is what makes the relation acyclic by
//! construction rather than by a visited set. Among neighbours tied on count, the higher packed
//! UMI wins, so the choice is a function of the data and not of iteration order. Chains are then
//! followed to their local maximum, and the whole group collapses onto that representative.

use crate::pairs::hamming1_pairs;
use crate::{Result, Umi};

const NO_TARGET: u32 = u32::MAX;

#[derive(Debug, Clone)]
pub struct Flow {
    umis: Vec<Umi>,
    counts: Vec<u32>,
    target: Vec<u32>,
    root: Vec<u32>,
    depth: Vec<u32>,
    n_pairs: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UmiGroup {
    pub representative: Umi,
    pub members: Vec<Umi>,
    pub n_reads: u32,
    pub chain_depth: u32,
}

pub fn flow(counts: &[(Umi, u32)], umi_len: u8) -> Result<Flow> {
    let umis: Vec<Umi> = counts.iter().map(|&(u, _)| u).collect();
    let read_counts: Vec<u32> = counts.iter().map(|&(_, c)| c).collect();
    let pairs = hamming1_pairs(&umis, umi_len)?;

    let n = umis.len();
    let mut target = vec![NO_TARGET; n];
    let mut best = vec![(0u32, Umi(0)); n];
    for &(i, j) in &pairs {
        for (from, to) in [(i as usize, j as usize), (j as usize, i as usize)] {
            if read_counts[to] <= read_counts[from] {
                continue;
            }
            let cand = (read_counts[to], umis[to]);
            if target[from] == NO_TARGET || cand > best[from] {
                target[from] = to as u32;
                best[from] = cand;
            }
        }
    }

    let mut root = vec![NO_TARGET; n];
    let mut depth = vec![0u32; n];
    let mut path: Vec<u32> = Vec::new();
    for start in 0..n {
        if root[start] != NO_TARGET {
            continue;
        }
        path.clear();
        let mut cur = start as u32;
        while root[cur as usize] == NO_TARGET {
            let next = target[cur as usize];
            if next == NO_TARGET {
                root[cur as usize] = cur;
                break;
            }
            debug_assert!(
                read_counts[next as usize] > read_counts[cur as usize],
                "an edge must strictly increase the count, or the relation is not acyclic"
            );
            path.push(cur);
            cur = next;
        }
        let (r, mut d) = (root[cur as usize], depth[cur as usize]);
        for &node in path.iter().rev() {
            d += 1;
            root[node as usize] = r;
            depth[node as usize] = d;
        }
    }

    Ok(Flow {
        umis,
        counts: read_counts,
        target,
        root,
        depth,
        n_pairs: pairs.len(),
    })
}

impl Flow {
    pub fn n_umis(&self) -> usize {
        self.umis.len()
    }

    pub fn n_hamming1_pairs(&self) -> usize {
        self.n_pairs
    }

    pub fn umi(&self, i: usize) -> Umi {
        self.umis[i]
    }

    pub fn count(&self, i: usize) -> u32 {
        self.counts[i]
    }

    pub fn target(&self, i: usize) -> Option<usize> {
        (self.target[i] != NO_TARGET).then(|| self.target[i] as usize)
    }

    pub fn root(&self, i: usize) -> usize {
        self.root[i] as usize
    }

    pub fn depth(&self, i: usize) -> u32 {
        self.depth[i]
    }

    pub fn groups(&self) -> Vec<UmiGroup> {
        let mut by_root: Vec<(Umi, Umi, usize)> = (0..self.n_umis())
            .map(|i| (self.umis[self.root(i)], self.umis[i], i))
            .collect();
        by_root.sort_unstable();

        let mut out: Vec<UmiGroup> = Vec::new();
        for (rep, umi, i) in by_root {
            match out.last_mut() {
                Some(g) if g.representative == rep => {
                    g.members.push(umi);
                    g.n_reads += self.counts[i];
                    g.chain_depth = g.chain_depth.max(self.depth[i]);
                }
                _ => out.push(UmiGroup {
                    representative: rep,
                    members: vec![umi],
                    n_reads: self.counts[i],
                    chain_depth: self.depth[i],
                }),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pairs::tests::{clustered_umis, small_space_umis};
    use crate::{Error, pack_umi};
    use proptest::prelude::*;
    use std::collections::{BTreeMap, BTreeSet};

    fn cellranger_target(counts: &BTreeMap<u64, u32>, umi: u64, umi_len: u8) -> u64 {
        let orig = counts[&umi];
        let (mut best_umi, mut best_count) = (umi, orig);
        for pos in 0..umi_len as u32 {
            let shift = 2 * pos;
            let cur = (umi >> shift) & 3;
            for base in 0..4u64 {
                if base == cur {
                    continue;
                }
                let test = (umi & !(3 << shift)) | (base << shift);
                let count = counts.get(&test).copied().unwrap_or(0);
                if count > best_count || (count == best_count && test > best_umi) {
                    best_umi = test;
                    best_count = count;
                }
            }
        }
        best_umi
    }

    fn strict_target(counts: &BTreeMap<u64, u32>, umi: u64, umi_len: u8) -> u64 {
        let orig = counts[&umi];
        let (mut best_umi, mut best_count) = (umi, orig);
        for pos in 0..umi_len as u32 {
            let shift = 2 * pos;
            let cur = (umi >> shift) & 3;
            for base in 0..4u64 {
                if base == cur {
                    continue;
                }
                let test = (umi & !(3 << shift)) | (base << shift);
                let count = counts.get(&test).copied().unwrap_or(0);
                if count <= orig {
                    continue;
                }
                if count > best_count || (count == best_count && test > best_umi) {
                    best_umi = test;
                    best_count = count;
                }
            }
        }
        best_umi
    }

    fn partition(classes: impl IntoIterator<Item = Vec<u64>>) -> BTreeSet<Vec<u64>> {
        classes
            .into_iter()
            .map(|mut c| {
                c.sort_unstable();
                c
            })
            .collect()
    }

    fn group_by_target(counts: &BTreeMap<u64, u32>, target: impl Fn(u64) -> u64) -> BTreeSet<Vec<u64>> {
        let mut classes: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for &umi in counts.keys() {
            classes.entry(target(umi)).or_default().push(umi);
        }
        partition(classes.into_values())
    }

    fn group_by_chain(counts: &BTreeMap<u64, u32>, target: impl Fn(u64) -> u64) -> BTreeSet<Vec<u64>> {
        let mut classes: BTreeMap<u64, Vec<u64>> = BTreeMap::new();
        for &umi in counts.keys() {
            let mut cur = umi;
            let mut hops = 0;
            while target(cur) != cur {
                cur = target(cur);
                hops += 1;
                assert!(hops <= counts.len(), "reference chain did not terminate");
            }
            classes.entry(cur).or_default().push(umi);
        }
        partition(classes.into_values())
    }

    fn flow_partition(f: &Flow) -> BTreeSet<Vec<u64>> {
        partition(
            f.groups()
                .into_iter()
                .map(|g| g.members.into_iter().map(|u| u.0).collect()),
        )
    }

    fn no_equal_count_neighbour(f: &Flow, pairs: &[(u32, u32)]) -> bool {
        pairs.iter().all(|&(i, j)| f.count(i as usize) != f.count(j as usize))
    }

    fn no_chain_over_one_hop(f: &Flow) -> bool {
        (0..f.n_umis()).all(|i| f.depth(i) <= 1)
    }

    fn has_edge(f: &Flow) -> bool {
        (0..f.n_umis()).any(|i| f.target(i).is_some())
    }

    fn counted(strategy: impl Strategy<Value = (u8, Vec<Umi>)>) -> impl Strategy<Value = (u8, Vec<(Umi, u32)>)> {
        strategy.prop_flat_map(|(umi_len, umis)| {
            let n = umis.len();
            (Just(umi_len), Just(umis), prop::collection::vec(1u32..=4, n))
                .prop_map(|(l, u, c)| (l, u.into_iter().zip(c).collect()))
        })
    }

    fn as_map(counts: &[(Umi, u32)]) -> BTreeMap<u64, u32> {
        counts.iter().map(|&(u, c)| (u.0, c)).collect()
    }

    fn abc_chain() -> (u8, Vec<(Umi, u32)>, Umi, Umi, Umi) {
        let a = pack_umi(&[0, 0, 0, 0, 0, 0]).unwrap();
        let b = pack_umi(&[1, 0, 0, 0, 0, 0]).unwrap();
        let c = pack_umi(&[1, 1, 0, 0, 0, 0]).unwrap();
        (6, vec![(a, 1), (b, 2), (c, 5)], a, b, c)
    }

    #[test]
    fn flow_does_not_orphan_a_in_the_abc_chain() {
        let (umi_len, counts, a, b, c) = abc_chain();
        let f = flow(&counts, umi_len).unwrap();
        let groups = f.groups();

        assert_eq!(groups.len(), 1, "flow must emit one molecule, not two");
        assert_eq!(groups[0].representative, c, "the local maximum is C");
        assert_eq!(groups[0].members, vec![a, b, c], "A must not be orphaned");
        assert_eq!(groups[0].n_reads, 8, "1 + 2 + 5 reads");
        assert_eq!(groups[0].chain_depth, 2, "A -> B -> C is two hops");
    }

    #[test]
    fn one_hop_cellranger_orphans_a_in_the_abc_chain() {
        let (umi_len, counts, a, b, c) = abc_chain();
        let map = as_map(&counts);
        let cr = group_by_target(&map, |u| cellranger_target(&map, u, umi_len));
        assert_eq!(
            cr,
            partition([vec![a.0], vec![b.0, c.0]]),
            "CellRanger fragments the chain"
        );
    }

    #[test]
    fn chain_defeats_the_no_equal_count_condition_alone() {
        let (umi_len, counts, ..) = abc_chain();
        let f = flow(&counts, umi_len).unwrap();
        let pairs = crate::hamming1_pairs(&counts.iter().map(|&(u, _)| u).collect::<Vec<_>>(), umi_len).unwrap();
        assert!(no_equal_count_neighbour(&f, &pairs), "the condition holds");
        assert!(!no_chain_over_one_hop(&f), "but a two-hop chain exists");

        let map = as_map(&counts);
        assert_ne!(
            flow_partition(&f),
            group_by_target(&map, |u| cellranger_target(&map, u, umi_len)),
            "so parity with CellRanger cannot follow from that condition alone"
        );
    }

    #[test]
    fn equal_counts_produce_no_edge() {
        let a = pack_umi(&[0, 0, 0, 0]).unwrap();
        let b = pack_umi(&[1, 0, 0, 0]).unwrap();
        let f = flow(&[(a, 3), (b, 3)], 4).unwrap();
        assert_eq!(f.target(0), None);
        assert_eq!(f.target(1), None);
        assert_eq!(f.groups().len(), 2, "a tie is two molecules, not a coin-flip merge");
    }

    #[test]
    fn target_is_the_highest_count_neighbour() {
        let a = pack_umi(&[0, 0, 0, 0]).unwrap();
        let lo = pack_umi(&[1, 0, 0, 0]).unwrap();
        let hi = pack_umi(&[2, 0, 0, 0]).unwrap();
        let f = flow(&[(a, 1), (lo, 3), (hi, 9)], 4).unwrap();
        assert_eq!(f.umi(f.target(0).unwrap()), hi);
        assert_eq!(f.groups()[0].members, vec![a, lo, hi]);
    }

    #[test]
    fn equal_count_targets_break_ties_lexicographically_like_cellranger() {
        let a = pack_umi(&[0, 0, 0, 0]).unwrap();
        let c = pack_umi(&[1, 0, 0, 0]).unwrap();
        let g = pack_umi(&[2, 0, 0, 0]).unwrap();
        let counts = vec![(a, 1), (c, 5), (g, 5)];
        let f = flow(&counts, 4).unwrap();
        assert_eq!(f.umi(f.target(0).unwrap()), g, "G > C lexicographically");
        let map = as_map(&counts);
        assert_eq!(
            cellranger_target(&map, a.0, 4),
            g.0,
            "which is what CellRanger does too"
        );
        assert_eq!(
            f.groups(),
            vec![
                UmiGroup {
                    representative: c,
                    members: vec![c],
                    n_reads: 5,
                    chain_depth: 0,
                },
                UmiGroup {
                    representative: g,
                    members: vec![a, g],
                    n_reads: 6,
                    chain_depth: 1,
                },
            ],
            "a tie splits the component; single linkage would merge all three"
        );
    }

    #[test]
    fn groups_are_ordered_by_representative_and_independent_of_input_order() {
        let a = pack_umi(&[0, 0, 0, 0]).unwrap();
        let c = pack_umi(&[1, 0, 0, 0]).unwrap();
        let g = pack_umi(&[2, 0, 0, 0]).unwrap();
        let counts = vec![(a, 1), (c, 5), (g, 5)];

        let want = flow(&counts, 4).unwrap().groups();
        assert!(
            want.windows(2).all(|w| w[0].representative < w[1].representative),
            "groups must be ascending by representative: {want:?}"
        );
        for gr in &want {
            assert!(
                gr.members.windows(2).all(|w| w[0] < w[1]),
                "members must be ascending: {gr:?}"
            );
            assert!(gr.members.contains(&gr.representative));
        }

        let perms = [
            vec![(a, 1), (g, 5), (c, 5)],
            vec![(c, 5), (a, 1), (g, 5)],
            vec![(c, 5), (g, 5), (a, 1)],
            vec![(g, 5), (a, 1), (c, 5)],
            vec![(g, 5), (c, 5), (a, 1)],
        ];
        for perm in perms {
            let got = flow(&perm, 4).unwrap();
            assert_eq!(got.groups(), want, "input order {perm:?} changed the grouping");
            let a_at = (0..got.n_umis()).find(|&i| got.umi(i) == a).unwrap();
            assert_eq!(
                got.umi(got.target(a_at).unwrap()),
                g,
                "the lexicographic tie-break must not depend on which neighbour was seen last"
            );
        }
    }

    #[test]
    fn a_lone_umi_is_its_own_molecule() {
        let a = pack_umi(&[0, 1, 2, 3]).unwrap();
        let f = flow(&[(a, 7)], 4).unwrap();
        assert_eq!(
            f.groups(),
            vec![UmiGroup {
                representative: a,
                members: vec![a],
                n_reads: 7,
                chain_depth: 0
            }]
        );
    }

    #[test]
    fn duplicate_umi_is_rejected() {
        let a = pack_umi(&[0, 0, 0, 0]).unwrap();
        assert_eq!(flow(&[(a, 1), (a, 2)], 4).unwrap_err(), Error::DuplicateUmi(a.0));
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

        #[test]
        fn acyclicity_holds_on_random_inputs((umi_len, counts) in counted(small_space_umis())) {
            let f = flow(&counts, umi_len).unwrap();
            for i in 0..f.n_umis() {
                if let Some(t) = f.target(i) {
                    prop_assert!(f.count(t) > f.count(i), "edge must strictly increase the count");
                    prop_assert_ne!(t, i, "no self-edge");
                }
                let (mut cur, mut hops) = (i, 0u32);
                while let Some(t) = f.target(cur) {
                    cur = t;
                    hops += 1;
                    prop_assert!(hops as usize <= f.n_umis(), "walk did not terminate: cycle");
                }
                prop_assert_eq!(cur, f.root(i), "the walk ends at the recorded local maximum");
                prop_assert_eq!(hops, f.depth(i), "depth is the hop count");
                prop_assert!(f.target(f.root(i)).is_none(), "a root is a local maximum");
            }
        }

        #[test]
        fn groups_partition_the_umis_and_conserve_reads((umi_len, counts) in counted(small_space_umis())) {
            let f = flow(&counts, umi_len).unwrap();
            let groups = f.groups();
            let members: Vec<u64> = groups.iter().flat_map(|g| g.members.iter().map(|u| u.0)).collect();
            let expected: BTreeSet<u64> = counts.iter().map(|&(u, _)| u.0).collect();
            prop_assert_eq!(members.len(), expected.len(), "every umi in exactly one group");
            prop_assert_eq!(members.iter().copied().collect::<BTreeSet<u64>>(), expected);
            let total: u32 = counts.iter().map(|&(_, c)| c).sum();
            prop_assert_eq!(groups.iter().map(|g| g.n_reads).sum::<u32>(), total);
            for g in &groups {
                prop_assert!(g.members.contains(&g.representative), "a group contains its local maximum");
                prop_assert!(g.members.windows(2).all(|w| w[0] < w[1]), "members ascending and unique");
            }
        }

    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, max_global_rejects: 20_000, ..ProptestConfig::default() })]

        #[test]
        fn deviation_1_parity_no_equal_count_neighbour((umi_len, counts) in counted(small_space_umis())) {
            let f = flow(&counts, umi_len).unwrap();
            let umis: Vec<Umi> = counts.iter().map(|&(u, _)| u).collect();
            let pairs = crate::hamming1_pairs(&umis, umi_len).unwrap();
            prop_assume!(no_equal_count_neighbour(&f, &pairs) && has_edge(&f));
            let map = as_map(&counts);
            prop_assert_eq!(flow_partition(&f), group_by_chain(&map, |u| cellranger_target(&map, u, umi_len)));
        }

        #[test]
        fn deviation_2_parity_no_chain_over_one_hop((umi_len, counts) in counted(small_space_umis())) {
            let f = flow(&counts, umi_len).unwrap();
            prop_assume!(no_chain_over_one_hop(&f) && has_edge(&f));
            let map = as_map(&counts);
            prop_assert_eq!(flow_partition(&f), group_by_target(&map, |u| strict_target(&map, u, umi_len)));
        }

        #[test]
        fn full_cellranger_parity_under_both_conditions((umi_len, counts) in counted(small_space_umis())) {
            let f = flow(&counts, umi_len).unwrap();
            let umis: Vec<Umi> = counts.iter().map(|&(u, _)| u).collect();
            let pairs = crate::hamming1_pairs(&umis, umi_len).unwrap();
            prop_assume!(no_equal_count_neighbour(&f, &pairs) && no_chain_over_one_hop(&f) && has_edge(&f));
            let map = as_map(&counts);
            prop_assert_eq!(flow_partition(&f), group_by_target(&map, |u| cellranger_target(&map, u, umi_len)));
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

        #[test]
        fn parity_at_real_umi_lengths((umi_len, counts) in counted(clustered_umis())) {
            let f = flow(&counts, umi_len).unwrap();
            let umis: Vec<Umi> = counts.iter().map(|&(u, _)| u).collect();
            let pairs = crate::hamming1_pairs(&umis, umi_len).unwrap();
            let map = as_map(&counts);
            if no_equal_count_neighbour(&f, &pairs) {
                prop_assert_eq!(flow_partition(&f), group_by_chain(&map, |u| cellranger_target(&map, u, umi_len)));
            }
            if no_chain_over_one_hop(&f) {
                prop_assert_eq!(flow_partition(&f), group_by_target(&map, |u| strict_target(&map, u, umi_len)));
            }
        }
    }
}
