use criterion::{Criterion, criterion_group, criterion_main};
use scrm_asm::{assemble, default_model};
use scrm_core::MoleculeRead;
use scrm_core::record::Umi;
use scrm_core::{CellId, K, K_DEFAULT, Molecule};
use std::hint::black_box;

const READ_LEN: usize = 90;

fn read(seed: u64, len: usize) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state & 3) as u8
        })
        .collect()
}

fn molecule(reads: Vec<(Vec<u8>, u32)>) -> Molecule {
    Molecule {
        cell: CellId(0x0123_4567_89ab_cdef),
        umi: Umi(0xfeed_face),
        n_reads: reads.iter().map(|(_, w)| w).sum(),
        reads: reads
            .into_iter()
            .map(|(seq, w)| MoleculeRead::unqualified(seq, w))
            .collect(),
    }
}

fn bench(c: &mut Criterion) {
    let base = read(0x2026_0815, READ_LEN);

    let pass_through = molecule(vec![(base.clone(), 1)]);

    let identical = molecule(vec![(base.clone(), 7)]);

    let overlap = {
        let long = read(0xdead_beef, READ_LEN * 2 - (K - 1));
        molecule(vec![
            (long[..READ_LEN].to_vec(), 3),
            (long[READ_LEN - (K - 1)..].to_vec(), 2),
        ])
    };

    let bubble = {
        let mut variant = base.clone();
        variant[READ_LEN / 2] = (variant[READ_LEN / 2] + 1) % 4;
        molecule(vec![(base.clone(), 3), (variant, 1)])
    };

    let deep = molecule(
        (0..32)
            .map(|i| (read(0x1000 + i as u64, READ_LEN), 1 + i % 5))
            .collect(),
    );

    for (name, m) in [
        ("pass_through", &pass_through),
        ("identical_reads", &identical),
        ("overlap", &overlap),
        ("bubble", &bubble),
        ("deep_32_seqs", &deep),
    ] {
        c.bench_function(&format!("assemble/{name}"), |b| {
            b.iter(|| assemble(black_box(m), default_model(), K_DEFAULT))
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
