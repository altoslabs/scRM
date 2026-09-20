use crate::stats::CellUmiStats;
use crate::{CellId, Error, Molecule, ParsedPair, Result, flow};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct DedupedCell {
    pub molecules: Vec<Molecule>,
    pub stats: CellUmiStats,
}

type PerUmi = BTreeMap<u64, (u32, BTreeMap<Vec<u8>, Vec<Vec<u8>>>)>;

pub fn dedup_cell(cell: CellId, pairs: Vec<ParsedPair>, umi_len: u8) -> Result<DedupedCell> {
    let mut per_umi: PerUmi = BTreeMap::new();
    for p in pairs {
        if p.cb != cell {
            return Err(Error::ForeignCell {
                expected: cell,
                found: p.cb,
            });
        }
        let entry = per_umi.entry(p.umi.0).or_default();
        entry.0 += 1;
        entry.1.entry(p.r2.clone()).or_default().push(p.qual.clone());
    }

    let counts: Vec<(crate::Umi, u32)> = per_umi.iter().map(|(&u, &(n, _))| (crate::Umi(u), n)).collect();
    let resolved = flow(&counts, umi_len)?;
    let stats = CellUmiStats::from_flow(&resolved);

    let mut molecules = Vec::with_capacity(resolved.n_umis());
    for group in resolved.groups() {
        let mut reads: BTreeMap<Vec<u8>, Vec<Vec<u8>>> = BTreeMap::new();
        for member in &group.members {
            let (_, member_reads) = per_umi.remove(&member.0).expect("every group member is an input umi");
            for (seq, quals) in member_reads {
                reads.entry(seq).or_default().extend(quals);
            }
        }
        debug_assert_eq!(
            reads.values().map(|q| q.len() as u32).sum::<u32>(),
            group.n_reads,
            "the group's read count and its read multiset must be the same reads"
        );
        molecules.push(Molecule {
            cell,
            umi: group.representative,
            n_reads: group.n_reads,
            reads: reads
                .into_iter()
                .map(|(seq, quals)| scrm_core::MoleculeRead { seq, quals })
                .collect(),
        });
    }
    Ok(DedupedCell { molecules, stats })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Umi, pack_umi};

    fn pair(cell: CellId, umi: Umi, r2: &[u8]) -> ParsedPair {
        ParsedPair {
            cb: cell,
            umi,
            r2: r2.to_vec(),
            qual: vec![30; r2.len()],
        }
    }

    fn seq_weights(m: &Molecule) -> Vec<(Vec<u8>, u32)> {
        m.reads.iter().map(|r| (r.seq.clone(), r.weight())).collect()
    }

    #[test]
    fn a_chain_becomes_one_molecule_with_unioned_reads_and_kept_multiplicity() {
        let cell = CellId(42);
        let a = pack_umi(&[0, 0, 0, 0, 0, 0]).unwrap();
        let b = pack_umi(&[1, 0, 0, 0, 0, 0]).unwrap();
        let c = pack_umi(&[1, 1, 0, 0, 0, 0]).unwrap();
        let seq = [0u8, 1, 2, 3];
        let variant = [0u8, 1, 2, 2];
        let pairs = vec![
            pair(cell, a, &seq),
            pair(cell, b, &seq),
            pair(cell, b, &variant),
            pair(cell, c, &seq),
            pair(cell, c, &seq),
            pair(cell, c, &seq),
            pair(cell, c, &variant),
            pair(cell, c, &variant),
        ];
        let out = dedup_cell(cell, pairs, 6).unwrap();

        assert_eq!(out.molecules.len(), 1);
        let m = &out.molecules[0];
        assert_eq!(m.umi, c, "the molecule is keyed by the local maximum");
        assert_eq!(m.n_reads, 8, "1 + 2 + 5 reads");
        assert_eq!(seq_weights(m), vec![(variant.to_vec(), 3), (seq.to_vec(), 5)]);
        assert!(!m.is_pass_through(), "two distinct sequences means the graph is needed");
        assert_eq!(out.stats.n_molecules, 1);
        assert_eq!(out.stats.max_chain_depth, 2);
    }

    #[test]
    fn identical_reads_collapse_to_one_pass_through_sequence() {
        let cell = CellId(1);
        let u = pack_umi(&[0, 1, 2, 3, 0, 1]).unwrap();
        let seq = [3u8, 3, 2, 1];
        let out = dedup_cell(
            cell,
            vec![pair(cell, u, &seq), pair(cell, u, &seq), pair(cell, u, &seq)],
            6,
        )
        .unwrap();
        let m = &out.molecules[0];
        assert_eq!(seq_weights(m), vec![(seq.to_vec(), 3)]);
        assert_eq!(m.n_reads, 3);
        assert!(
            m.is_pass_through(),
            "one distinct sequence: assembly is skipped entirely"
        );
    }

    #[test]
    fn molecules_come_out_ordered_by_umi() {
        let cell = CellId(7);
        let hi = pack_umi(&[3, 3, 3, 3, 3, 3]).unwrap();
        let lo = pack_umi(&[0, 0, 0, 0, 0, 0]).unwrap();
        let mid = pack_umi(&[1, 1, 1, 1, 1, 1]).unwrap();
        let pairs = vec![pair(cell, hi, &[0]), pair(cell, lo, &[0]), pair(cell, mid, &[0])];
        let out = dedup_cell(cell, pairs, 6).unwrap();
        let umis: Vec<Umi> = out.molecules.iter().map(|m| m.umi).collect();
        assert_eq!(umis, vec![lo, mid, hi]);
    }

    #[test]
    fn a_pair_from_another_cell_is_an_error() {
        let cell = CellId(1);
        let other = CellId(2);
        let u = pack_umi(&[0, 1, 2, 3, 0, 1]).unwrap();
        let err = dedup_cell(cell, vec![pair(cell, u, &[0]), pair(other, u, &[0])], 6).unwrap_err();
        assert_eq!(
            err,
            Error::ForeignCell {
                expected: cell,
                found: other
            }
        );
    }

    #[test]
    fn an_empty_cell_yields_no_molecules() {
        let out = dedup_cell(CellId(0), vec![], 10).unwrap();
        assert!(out.molecules.is_empty());
        assert_eq!(out.stats, CellUmiStats::default());
    }
}
