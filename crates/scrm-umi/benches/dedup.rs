use criterion::{Criterion, criterion_group, criterion_main};
use scrm_umi::{Umi, flow, hamming1_pairs};
use std::collections::BTreeSet;
use std::hint::black_box;

fn cell(n: usize, umi_len: u8, seed: u64) -> Vec<(Umi, u32)> {
    let mut state = seed;
    let mask = (1u64 << (2 * umi_len as u32)) - 1;
    let mut set = BTreeSet::new();
    while set.len() < n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        set.insert(state & mask);
    }
    set.into_iter()
        .enumerate()
        .map(|(i, u)| (Umi(u), (i % 8 + 1) as u32))
        .collect()
}

fn bench(c: &mut Criterion) {
    for &(umi_len, n) in &[(10u8, 22_268usize), (12, 9_560)] {
        let counts = cell(n, umi_len, 0x2026_0814);
        let umis: Vec<Umi> = counts.iter().map(|&(u, _)| u).collect();
        c.bench_function(&format!("hamming1_pairs/{umi_len}bp/{n}"), |b| {
            b.iter(|| hamming1_pairs(black_box(&umis), umi_len).unwrap())
        });
        c.bench_function(&format!("flow/{umi_len}bp/{n}"), |b| {
            b.iter(|| flow(black_box(&counts), umi_len).unwrap().groups())
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
