use crate::{Error, Result};
use scrm_core::record::Umi;

pub const MIN_UMI_LEN: u8 = 2;

pub const MAX_UMI_LEN: u8 = 32;

#[inline]
const fn mask(bits: u32) -> u64 {
    if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
}

#[inline]
const fn half_bits(umi_len: u8) -> (u32, u32) {
    let low_bases = (umi_len / 2) as u32;
    (2 * low_bases, 2 * (umi_len as u32 - low_bases))
}

#[inline]
pub(crate) const fn swap_halves(v: u64, low_bits: u32, high_bits: u32) -> u64 {
    ((v & mask(low_bits)) << high_bits) | (v >> low_bits)
}

pub fn hamming1_pairs(umis: &[Umi], umi_len: u8) -> Result<Vec<(u32, u32)>> {
    if !(MIN_UMI_LEN..=MAX_UMI_LEN).contains(&umi_len) {
        return Err(Error::UmiLenOutOfRange(umi_len));
    }
    let width = 2 * umi_len as u32;
    let (low_bits, high_bits) = half_bits(umi_len);

    let mut buf: Vec<(u64, u32)> = Vec::with_capacity(umis.len());
    for (i, umi) in umis.iter().enumerate() {
        if width < 64 && umi.0 > mask(width) {
            return Err(Error::UmiOutOfRange { umi: umi.0, umi_len });
        }
        buf.push((umi.0, i as u32));
    }

    let mut pairs = Vec::new();
    buf.sort_unstable();
    if let Some(w) = buf.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(Error::DuplicateUmi(w[0].0));
    }
    scan_low_half(&buf, mask(low_bits), &mut pairs);

    for e in &mut buf {
        e.0 = swap_halves(e.0, low_bits, high_bits);
    }
    buf.sort_unstable();
    scan_low_half(&buf, mask(high_bits), &mut pairs);

    pairs.sort_unstable();
    Ok(pairs)
}

fn scan_low_half(sorted: &[(u64, u32)], low_mask: u64, out: &mut Vec<(u32, u32)>) {
    for (pos, &(vi, idx_i)) in sorted.iter().enumerate() {
        for &(vj, idx_j) in &sorted[pos + 1..] {
            let x = vi ^ vj;
            if x > low_mask {
                break;
            }
            if x >> (x.trailing_zeros() / 2 * 2) > 3 {
                continue;
            }
            out.push((idx_i.min(idx_j), idx_i.max(idx_j)));
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::collections::{BTreeSet, HashMap};

    pub(crate) fn hamming1_pairs_masking(umis: &[Umi], umi_len: u8) -> Vec<(u32, u32)> {
        let mut buckets: HashMap<(u32, u64), Vec<u32>> = HashMap::new();
        for (i, umi) in umis.iter().enumerate() {
            for pos in 0..umi_len as u32 {
                let key = umi.0 & !(0b11 << (2 * pos));
                buckets.entry((pos, key)).or_default().push(i as u32);
            }
        }
        let mut pairs = BTreeSet::new();
        for ids in buckets.values() {
            for (a, &i) in ids.iter().enumerate() {
                for &j in &ids[a + 1..] {
                    if umis[i as usize] != umis[j as usize] {
                        pairs.insert((i.min(j), i.max(j)));
                    }
                }
            }
        }
        pairs.into_iter().collect()
    }

    pub(crate) fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    pub(crate) fn distinct_umis(raw: &[u64], umi_len: u8) -> Vec<Umi> {
        let m = mask(2 * umi_len as u32);
        raw.iter()
            .map(|v| v & m)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .map(Umi)
            .collect()
    }

    pub(crate) fn small_space_umis() -> impl Strategy<Value = (u8, Vec<Umi>)> {
        (MIN_UMI_LEN..=6u8).prop_flat_map(|umi_len| {
            let hi = 1u64 << (2 * umi_len as u32);
            (Just(umi_len), prop::collection::btree_set(0u64..hi, 1..=48))
                .prop_map(|(l, s)| (l, s.into_iter().map(Umi).collect()))
        })
    }

    pub(crate) fn clustered_umis_of_len(umi_len: u8) -> impl Strategy<Value = Vec<Umi>> {
        let hi = 1u64 << (2 * umi_len as u32);
        (
            prop::collection::vec(0u64..hi, 1..=8),
            prop::collection::vec(
                (0u32..umi_len as u32, 1u64..4, 0usize..8, 0u32..umi_len as u32, 0u64..4),
                0..40,
            ),
        )
            .prop_map(move |(seeds, muts)| {
                let mut out: BTreeSet<u64> = seeds.iter().copied().collect();
                for (p1, d1, seed_ix, p2, d2) in muts {
                    let base = seeds[seed_ix % seeds.len()];
                    let mut v = base ^ (d1 << (2 * p1));
                    v ^= d2 << (2 * p2);
                    out.insert(v & mask(2 * umi_len as u32));
                }
                out.into_iter().map(Umi).collect()
            })
    }

    pub(crate) fn clustered_umis() -> impl Strategy<Value = (u8, Vec<Umi>)> {
        prop::sample::select(vec![10u8, 11, 12])
            .prop_flat_map(|umi_len| (Just(umi_len), clustered_umis_of_len(umi_len)))
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

        #[test]
        fn kernel_matches_masking_reference((umi_len, umis) in small_space_umis()) {
            let kernel = hamming1_pairs(&umis, umi_len).unwrap();
            let reference = hamming1_pairs_masking(&umis, umi_len);
            prop_assert_eq!(&kernel, &reference);
            prop_assert!(kernel.windows(2).all(|w| w[0] < w[1]), "pairs must be sorted and unique");
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

        #[test]
        fn kernel_matches_masking_at_real_umi_lengths((umi_len, umis) in clustered_umis()) {
            let kernel = hamming1_pairs(&umis, umi_len).unwrap();
            prop_assert_eq!(kernel, hamming1_pairs_masking(&umis, umi_len));
        }
    }

    #[test]
    fn kernel_matches_masking_on_the_full_space() {
        let umis: Vec<Umi> = (0..64).map(Umi).collect();
        let kernel = hamming1_pairs(&umis, 3).unwrap();
        assert_eq!(kernel, hamming1_pairs_masking(&umis, 3));
        assert_eq!(kernel.len(), 64 * 3 * 3 / 2);
    }

    #[test]
    fn kernel_matches_masking_at_scale() {
        let mut state = 0x2026_0814_u64;
        let raw: Vec<u64> = (0..20_000).map(|_| xorshift(&mut state)).collect();
        let umis = distinct_umis(&raw, 10);
        let kernel = hamming1_pairs(&umis, 10).unwrap();
        assert_eq!(kernel, hamming1_pairs_masking(&umis, 10));
        let n = umis.len() as f64;
        let predicted = (n - 1.0) * 30.0 / 1_048_576.0;
        let observed = 2.0 * kernel.len() as f64 / n;
        assert!(
            (observed - predicted).abs() < 0.2,
            "observed degree {observed} vs predicted {predicted}"
        );
    }

    #[test]
    fn swap_halves_round_trips() {
        for umi_len in MIN_UMI_LEN..=MAX_UMI_LEN {
            let (low, high) = half_bits(umi_len);
            let v = 0xDEAD_BEEF_CAFE_F00D & mask(2 * umi_len as u32);
            assert_eq!(
                swap_halves(swap_halves(v, low, high), high, low),
                v,
                "umi_len {umi_len}"
            );
        }
    }

    #[test]
    fn odd_umi_length_splits_on_a_base_boundary() {
        assert_eq!(half_bits(5), (4, 6));
        let a = crate::pack_umi(&[0, 0, 0, 0, 0]).unwrap();
        let b = crate::pack_umi(&[0, 0, 2, 0, 0]).unwrap();
        assert_eq!(hamming1_pairs(&[a, b], 5).unwrap(), vec![(0, 1)]);
    }

    #[test]
    fn input_contract_violations_are_errors() {
        let u = |v| Umi(v);
        assert_eq!(hamming1_pairs(&[u(0), u(1)], 1), Err(Error::UmiLenOutOfRange(1)));
        assert_eq!(hamming1_pairs(&[u(0), u(1)], 33), Err(Error::UmiLenOutOfRange(33)));
        assert_eq!(hamming1_pairs(&[u(0), u(0)], 4), Err(Error::DuplicateUmi(0)));
        assert_eq!(
            hamming1_pairs(&[u(1 << 8)], 4),
            Err(Error::UmiOutOfRange {
                umi: 1 << 8,
                umi_len: 4
            })
        );
    }
}
