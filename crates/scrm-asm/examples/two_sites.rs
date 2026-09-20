use scrm_asm::{assemble, default_model};
use scrm_core::record::Umi;
use scrm_core::{CellId, K_DEFAULT, Molecule, MoleculeRead, Token};

fn seq(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) % 4) as u8
        })
        .collect()
}

fn read(seq: &[u8], q: u8) -> MoleculeRead {
    MoleculeRead {
        seq: seq.to_vec(),
        quals: vec![vec![q; seq.len()]],
    }
}

fn show(label: &str, reads: Vec<MoleculeRead>) {
    let m = Molecule {
        cell: CellId(1),
        umi: Umi(1),
        n_reads: reads.len() as u32,
        reads,
    };
    let (out, d) = assemble(&m, default_model(), K_DEFAULT);
    let tab = Token::Tab as u8;
    let lens: Vec<usize> = out.tokens.split(|&t| t == tab).map(|c| c.len()).collect();
    let ns = out.tokens.iter().filter(|&&t| t == Token::N as u8).count();
    println!(
        "{label:<34} {}c  {ns}N (tie={} low={} indel={})  contig lengths {:?}",
        out.n_contigs, d.n_tie, d.n_low_confidence, d.n_indel_interior, lens
    );
}

fn main() {
    let fl = seq(11, 40);
    let mid = seq(33, 40);
    let fr = seq(22, 40);
    let cat = |p: &[&[u8]]| -> Vec<u8> { p.concat() };

    let a = cat(&[&fl, &[0], &mid, &[0], &fr]);
    let b = cat(&[&fl, &[0], &mid, &[1], &fr]);
    let tip_5 = cat(&[&fl, &[1], &mid[..6]]);

    println!("k = {K_DEFAULT}\n");
    println!("-- one site only, for reference (this is bubble_cases 1 and 6) --");
    show("bubble alone (a,b)", vec![read(&a, 30), read(&b, 30)]);
    show("tip alone (a,tip)", vec![read(&a, 30), read(&tip_5, 30)]);

    println!("\n-- two sites: a 5' tip BEFORE a bubble --");
    show(
        "tip@5' + bubble  Q30/Q30",
        vec![read(&a, 30), read(&b, 30), read(&tip_5, 30)],
    );
    show(
        "tip@5' + bubble  Q40/Q08",
        vec![read(&a, 40), read(&b, 8), read(&tip_5, 30)],
    );

    let tip_3 = cat(&[&fr[..0], &fl, &[0], &mid, &[0], &fr[..20], &[(fr[20] + 1) % 4]]);
    println!("\n-- two sites: a bubble BEFORE a 3' tip --");
    show(
        "bubble + tip@3'  Q30/Q30",
        vec![read(&a, 30), read(&b, 30), read(&tip_3, 30)],
    );
}
