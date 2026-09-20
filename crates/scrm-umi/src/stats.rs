use crate::Flow;
use scrm_core::stamp::UmiStats;

pub const PERCOLATION_DEGREE: f64 = 1.0;
pub const OBSERVED_MAX_CHAIN_DEPTH: u32 = 5;
pub const OBSERVED_MAX_GROUP_UMIS: u32 = 24;

pub fn percolation_warning(s: &UmiStats) -> Option<String> {
    (s.mean_hamming1_degree >= PERCOLATION_DEGREE).then(|| {
        format!(
            "mean Hamming-1 degree {:.4} >= {PERCOLATION_DEGREE:.1}: the merge graph percolates, so \
             transitive single-linkage merging is no longer a local correction. Largest group {} \
             UMIs, chain depth {}.",
            s.mean_hamming1_degree, s.max_group_umis, s.max_chain_depth
        )
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CellUmiStats {
    pub n_umis: u64,
    pub n_hamming1_pairs: u64,
    pub n_molecules: u64,
    pub n_single_umi_molecules: u64,
    pub max_group_umis: u32,
    pub max_chain_depth: u32,
}

impl CellUmiStats {
    pub fn from_flow(f: &Flow) -> Self {
        let n = f.n_umis();
        let mut group_size = vec![0u32; n];
        let mut max_chain_depth = 0;
        for i in 0..n {
            group_size[f.root(i)] += 1;
            max_chain_depth = max_chain_depth.max(f.depth(i));
        }
        Self {
            n_umis: n as u64,
            n_hamming1_pairs: f.n_hamming1_pairs() as u64,
            n_molecules: group_size.iter().filter(|&&s| s > 0).count() as u64,
            n_single_umi_molecules: group_size.iter().filter(|&&s| s == 1).count() as u64,
            max_group_umis: group_size.iter().copied().max().unwrap_or(0),
            max_chain_depth,
        }
    }

    pub fn max_group_fraction(&self) -> f64 {
        if self.n_umis == 0 {
            0.0
        } else {
            self.max_group_umis as f64 / self.n_umis as f64
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct UmiStatsAccum {
    n_umis: u64,
    n_pairs: u64,
    n_molecules: u64,
    n_single: u64,
    max_group_umis: u32,
    max_group_fraction: f64,
    max_group_fraction_umis: u64,
    max_chain_depth: u32,
}

impl UmiStatsAccum {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_cell(&mut self, c: &CellUmiStats) {
        self.n_umis += c.n_umis;
        self.n_pairs += c.n_hamming1_pairs;
        self.n_molecules += c.n_molecules;
        self.n_single += c.n_single_umi_molecules;
        self.max_group_umis = self.max_group_umis.max(c.max_group_umis);
        self.max_chain_depth = self.max_chain_depth.max(c.max_chain_depth);
        if (c.max_group_fraction(), c.n_umis) > (self.max_group_fraction, self.max_group_fraction_umis) {
            self.max_group_fraction = c.max_group_fraction();
            self.max_group_fraction_umis = c.n_umis;
        }
    }

    pub fn merge(&mut self, other: &Self) {
        self.n_umis += other.n_umis;
        self.n_pairs += other.n_pairs;
        self.n_molecules += other.n_molecules;
        self.n_single += other.n_single;
        self.max_group_umis = self.max_group_umis.max(other.max_group_umis);
        self.max_chain_depth = self.max_chain_depth.max(other.max_chain_depth);
        if (other.max_group_fraction, other.max_group_fraction_umis)
            > (self.max_group_fraction, self.max_group_fraction_umis)
        {
            self.max_group_fraction = other.max_group_fraction;
            self.max_group_fraction_umis = other.max_group_fraction_umis;
        }
    }

    pub fn finish(&self) -> UmiStats {
        let ratio = |num: f64, den: u64| if den == 0 { 0.0 } else { num / den as f64 };
        UmiStats {
            mean_hamming1_degree: ratio(2.0 * self.n_pairs as f64, self.n_umis),
            singleton_fraction: ratio(self.n_single as f64, self.n_molecules),
            merge_rate: ratio((self.n_umis - self.n_molecules) as f64, self.n_umis),
            max_group_fraction: self.max_group_fraction,
            max_group_umis: self.max_group_umis,
            max_chain_depth: self.max_chain_depth,
            max_group_fraction_umis: self.max_group_fraction_umis,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{flow, pack_umi};

    #[test]
    fn percolation_warns_only_when_the_graph_percolates() {
        let stats = |degree: f64, group: u32, depth: u32| UmiStats {
            mean_hamming1_degree: degree,
            singleton_fraction: 0.9,
            merge_rate: 0.1,
            max_group_fraction: 0.01,
            max_group_fraction_umis: 5000,
            max_group_umis: group,
            max_chain_depth: depth,
        };

        let w = percolation_warning(&stats(1.1662, 256, 8)).expect("1.1662 must warn");
        assert!(w.contains("percolates"), "{w}");
        assert!(
            w.contains("256") && w.contains("8"),
            "the warning carries the corroborating numbers: {w}"
        );

        assert!(percolation_warning(&stats(0.9269, 128, 7)).is_none());
        assert!(percolation_warning(&stats(0.0970, 34, 4)).is_none());
        let mouse_brain_group: u32 = 34;
        assert!(mouse_brain_group > OBSERVED_MAX_GROUP_UMIS);

        assert!(percolation_warning(&stats(PERCOLATION_DEGREE, 1, 1)).is_some());
        assert!(percolation_warning(&stats(PERCOLATION_DEGREE - 1e-9, 1, 1)).is_none());
    }

    #[test]
    fn a_one_umi_cell_reports_its_denominator_not_a_hidden_alarm() {
        let shallow = CellUmiStats {
            n_umis: 1,
            n_molecules: 1,
            n_single_umi_molecules: 1,
            max_group_umis: 1,
            ..Default::default()
        };
        let mut acc = UmiStatsAccum::new();
        acc.add_cell(&shallow);
        let stats = acc.finish();
        assert_eq!(stats.max_group_fraction, 1.0);
        assert_eq!(
            stats.max_group_fraction_umis, 1,
            "1.0 over 1 UMI is visibly uninformative"
        );
        assert_eq!(stats.max_group_umis, 1);
    }

    #[test]
    fn equal_fractions_report_the_deeper_cell() {
        let cell = |n_umis: u64, group: u32| CellUmiStats {
            n_umis,
            n_molecules: n_umis - group as u64 + 1,
            n_single_umi_molecules: 0,
            max_group_umis: group,
            max_chain_depth: 1,
            n_hamming1_pairs: 0,
        };
        let mut acc = UmiStatsAccum::new();
        acc.add_cell(&cell(10, 1));
        acc.add_cell(&cell(5000, 500));
        let stats = acc.finish();
        assert!((stats.max_group_fraction - 0.1).abs() < 1e-12);
        assert_eq!(stats.max_group_fraction_umis, 5000);
    }

    #[test]
    fn the_guard_is_invariant_to_how_cells_are_partitioned() {
        let cells: Vec<CellUmiStats> = (0..6)
            .map(|i| CellUmiStats {
                n_umis: 500 + i * 100,
                n_molecules: 500,
                n_single_umi_molecules: 10,
                n_hamming1_pairs: i,
                max_group_umis: 3 + i as u32,
                max_chain_depth: 1 + i as u32,
            })
            .collect();

        let mut whole = UmiStatsAccum::new();
        for c in &cells {
            whole.add_cell(c);
        }

        let mut a = UmiStatsAccum::new();
        let mut b = UmiStatsAccum::new();
        for (i, c) in cells.iter().enumerate() {
            if i % 2 == 0 { a.add_cell(c) } else { b.add_cell(c) }
        }
        a.merge(&b);

        assert_eq!(a.finish(), whole.finish(), "partitioning changed the stamped guards");
    }

    fn worked_example() -> CellUmiStats {
        let a = pack_umi(&[0, 0, 0, 0, 0, 0]).unwrap();
        let b = pack_umi(&[1, 0, 0, 0, 0, 0]).unwrap();
        let c = pack_umi(&[1, 1, 0, 0, 0, 0]).unwrap();
        let d = pack_umi(&[3, 3, 3, 3, 3, 3]).unwrap();
        let e = pack_umi(&[2, 3, 3, 3, 3, 3]).unwrap();
        let f = pack_umi(&[0, 1, 2, 3, 0, 1]).unwrap();
        let counts = vec![(a, 1), (b, 2), (c, 5), (d, 1), (e, 4), (f, 9)];
        CellUmiStats::from_flow(&flow(&counts, 6).unwrap())
    }

    #[test]
    fn cell_stats_match_a_hand_computed_example() {
        let s = worked_example();
        assert_eq!(s.n_umis, 6);
        assert_eq!(s.n_hamming1_pairs, 3);
        assert_eq!(s.n_molecules, 3, "{{A,B,C}}, {{D,E}}, {{F}}");
        assert_eq!(s.n_single_umi_molecules, 1, "only F");
        assert_eq!(s.max_group_umis, 3);
        assert_eq!(s.max_chain_depth, 2, "A -> B -> C");
        assert!((s.max_group_fraction() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn corpus_stats_average_over_umis_and_maximise_over_cells() {
        let mut acc = UmiStatsAccum::new();
        acc.add_cell(&worked_example());
        acc.add_cell(&CellUmiStats {
            n_umis: 4,
            n_hamming1_pairs: 0,
            n_molecules: 4,
            n_single_umi_molecules: 4,
            max_group_umis: 1,
            max_chain_depth: 0,
        });
        let s = acc.finish();
        assert!(
            (s.mean_hamming1_degree - 6.0 / 10.0).abs() < 1e-12,
            "2*3 pairs over 10 umis"
        );
        assert!(
            (s.singleton_fraction - 5.0 / 7.0).abs() < 1e-12,
            "1 + 4 singletons of 7 molecules"
        );
        assert!((s.merge_rate - 3.0 / 10.0).abs() < 1e-12, "10 umis -> 7 molecules");
        assert!(
            (s.max_group_fraction - 0.5).abs() < 1e-12,
            "per-cell max, not corpus-wide"
        );
        assert_eq!(s.max_group_fraction_umis, 6, "and it names the cell that attained it");
        assert_eq!(s.max_group_umis, 3);
        assert_eq!(s.max_chain_depth, 2);
    }

    #[test]
    fn an_empty_corpus_produces_zeros_not_nans() {
        let s = UmiStatsAccum::new().finish();
        assert_eq!(s.mean_hamming1_degree, 0.0);
        assert_eq!(s.singleton_fraction, 0.0);
        assert_eq!(s.merge_rate, 0.0);
        assert_eq!(s.max_group_fraction, 0.0);
        assert_eq!(s.max_group_umis, 0);
        assert_eq!(s.max_chain_depth, 0);
    }
}
