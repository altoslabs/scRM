use scrm_asm::{assemble, default_model};
use scrm_core::MoleculeRead;
use scrm_core::record::Umi;
use scrm_core::{CellId, K_DEFAULT, Molecule, Token};

fn seq(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 4) as u8
        })
        .collect()
}

fn molecule(a: &[u8], qa: u8, b: &[u8], qb: u8) -> Molecule {
    Molecule {
        cell: CellId(1),
        umi: Umi(1),
        n_reads: 2,
        reads: vec![
            MoleculeRead {
                seq: a.to_vec(),
                quals: vec![vec![qa; a.len()]],
            },
            MoleculeRead {
                seq: b.to_vec(),
                quals: vec![vec![qb; b.len()]],
            },
        ],
    }
}

fn show(label: &str, a: &[u8], b: &[u8]) {
    let n_tok = Token::N as u8;
    print!("{label:<44}");
    for (tag, qa, qb) in [("Q30/Q30", 30, 30), ("Q40/Q08", 40, 8)] {
        let m = molecule(a, qa, b, qb);
        let (out, d) = assemble(&m, default_model(), K_DEFAULT);
        let ns = out.tokens.iter().filter(|&&t| t == n_tok).count();
        print!(
            "  {tag}: {}c {ns:>2}N (tie={} low={} indel={})",
            out.n_contigs, d.n_tie, d.n_low_confidence, d.n_indel_interior
        );
    }
    println!("  [|a|={} |b|={}]", a.len(), b.len());
}

fn main() {
    let flank_l = seq(11, 40);
    let flank_r = seq(22, 40);
    let cat = |parts: &[&[u8]]| -> Vec<u8> { parts.concat() };

    println!("k = {K_DEFAULT}\n");

    let a1 = cat(&[&flank_l, &[0], &flank_r]);
    let b1 = cat(&[&flank_l, &[1], &flank_r]);
    show("1 rejoin, equal, 1 sub", &a1, &b1);

    let mid = seq(33, 10);
    let a2 = cat(&[&flank_l, &[0], &mid, &[0], &flank_r]);
    let b2 = cat(&[&flank_l, &[1], &mid, &[1], &flank_r]);
    show("2 rejoin, equal, 2 scattered subs", &a2, &b2);

    let a3 = cat(&[&flank_l, &[0, 0], &flank_r]);
    let b3 = cat(&[&flank_l, &[1, 1], &flank_r]);
    show("3 rejoin, equal, run of 2", &a3, &b3);

    let a4 = cat(&[&flank_l, &[0], &flank_r]);
    let b4 = cat(&[&flank_l, &[0, 1], &flank_r]);
    show("4 rejoin, UNEQUAL (true indel)", &a4, &b4);

    let long6 = cat(&[&flank_l, &[0], &seq(44, 46)]);
    let mut tip6 = long6[..flank_l.len() + 1 + 6].to_vec();
    let f = flank_l.len();
    tip6[f] = (tip6[f] + 1) % 4;
    show("6 TIP, unequal, 1 sub", &long6, &tip6);

    let long7 = cat(&[&flank_l, &seq(55, 46)]);
    let tip7 = long7[..flank_l.len() + 6].to_vec();
    show("7 TIP, unequal, 0 subs (pure tip)", &long7, &tip7);

    let a8 = seq(66, 90);
    let b8 = seq(77, 90);
    show("8 disjoint, no shared k-mer", &a8, &b8);
}
