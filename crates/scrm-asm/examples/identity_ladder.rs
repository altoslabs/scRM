use scrm_asm::{assemble, default_model};
use scrm_core::record::Umi;
use scrm_core::{CellId, K_DEFAULT, Molecule, MoleculeRead, Token};
use std::collections::HashSet;

fn seq(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 4) as u8
        })
        .collect()
}

fn drift(base: &[u8], frac: f64, seed: u64) -> (Vec<u8>, usize) {
    let mut v = base.to_vec();
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut n = 0;
    for b in v.iter_mut().skip(40) {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        if ((s >> 11) as f64 / (1u64 << 53) as f64) < frac {
            *b = (*b + 1 + ((s >> 5) % 3) as u8) % 4;
            n += 1;
        }
    }
    (v, n)
}

fn kmers(s: &[u8]) -> HashSet<&[u8]> {
    s.windows(K_DEFAULT).collect()
}

fn main() {
    let base = seq(7, 120);
    println!("k = {K_DEFAULT}, reads 120 bp, first 40 bp shared flank, both reads ONE observation each\n");
    println!(
        "{:>8} {:>7} {:>8}  {:>38}  {:>38}",
        "mismatch", "n_mism", "shared", "Q30/Q30", "Q40/Q40"
    );
    println!(
        "{:>8} {:>7} {:>8}  {:>38}  {:>38}",
        "rate", "", "19-mers", "contigs/N/decisions/splits/surprise", "same"
    );
    for &frac in &[0.0, 0.005, 0.02, 0.05, 0.10, 0.20, 0.30] {
        let (var, n_mism) = drift(&base, frac, 99);
        let shared = kmers(&base).intersection(&kmers(&var)).count();
        print!("{:>7.1}% {:>7} {:>8}  ", frac * 100.0, n_mism, shared);
        for (qa, qb) in [(30u8, 30u8), (40u8, 40u8)] {
            let m = Molecule {
                cell: CellId(1),
                umi: Umi(1),
                n_reads: 2,
                reads: vec![
                    MoleculeRead {
                        seq: base.clone(),
                        quals: vec![vec![qa; base.len()]],
                    },
                    MoleculeRead {
                        seq: var.clone(),
                        quals: vec![vec![qb; var.len()]],
                    },
                ],
            };
            let (out, d) = assemble(&m, default_model(), K_DEFAULT);
            let ns = out.tokens.iter().filter(|&&t| t == Token::N as u8).count();
            print!(
                "{:>3}c {:>3}N dec={:<3} split={:<3} surp={:<3} ",
                out.n_contigs, ns, d.n_branch_decisions, d.n_origin_splits, d.origin_surprise_max
            );
        }
        println!();
    }
    println!("\n'shared 19-mers' counts windows present in BOTH reads -- the only way the graph can rejoin.");
    println!("A 120bp read has 102 windows; 40bp of shared flank gives 22 of them.");
}
