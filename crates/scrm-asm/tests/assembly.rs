use std::collections::HashSet;

use scrm_asm::{Diagnostics, Retention, StatsCollector, assemble, default_model};
use scrm_core::K_DEFAULT;

fn assemble_weighted(m: &scrm_core::Molecule) -> (scrm_core::MoleculeContigs, Diagnostics) {
    assemble(m, default_model(), K_DEFAULT)
}

fn contigs(m: &scrm_core::Molecule) -> scrm_core::MoleculeContigs {
    assemble_weighted(m).0
}
use scrm_core::MoleculeRead;
use scrm_core::record::Umi;
use scrm_core::{ALPHABET_SIZE, CellId, K, MIN_TRIM_LEN, Molecule, MoleculeContigs, Token};

fn seq(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 33) as u8 & 3
        })
        .collect()
}

fn molecule(reads: Vec<(Vec<u8>, u32)>) -> Molecule {
    let n_reads = reads.iter().map(|(_, w)| *w).sum();
    Molecule {
        cell: CellId(42),
        umi: Umi(7),
        n_reads,
        reads: reads
            .into_iter()
            .map(|(seq, w)| MoleculeRead::unqualified(seq, w))
            .collect(),
    }
}

fn molecule_q(reads: Vec<(Vec<u8>, u8)>) -> Molecule {
    Molecule {
        cell: CellId(42),
        umi: Umi(7),
        n_reads: reads.len() as u32,
        reads: reads
            .into_iter()
            .map(|(seq, q)| MoleculeRead {
                quals: vec![vec![q; seq.len()]],
                seq,
            })
            .collect(),
    }
}

fn tab() -> u8 {
    Token::Tab as u8
}

fn n_token() -> u8 {
    Token::N as u8
}

fn kmers(s: &[u8]) -> HashSet<&[u8]> {
    s.windows(K).collect()
}

fn with_variant_at(s: &[u8], pos: usize) -> Vec<u8> {
    let mut v = s.to_vec();
    v[pos] = (v[pos] + 1) % 4;
    v
}

fn contigs_of(out: &MoleculeContigs) -> Vec<Vec<u8>> {
    out.tokens.split(|&t| t == tab()).map(<[u8]>::to_vec).collect()
}

#[test]
fn single_read_is_emitted_byte_for_byte_with_no_graph() {
    let read = seq(1, 90);
    let (out, diag) = assemble_weighted(&molecule(vec![(read.clone(), 1)]));

    assert_eq!(out.tokens, read);
    assert_eq!(out.n_contigs, 1);
    assert_eq!(out.recount_contigs(), 1);
    assert_eq!(diag, Diagnostics::default());
    assert!(!diag.graph_built);
    assert_eq!(diag.bubble_n_tokens, 0);
}

#[test]
fn identical_reads_collapse_to_one_sequence_but_keep_n_reads() {
    let read = seq(2, 90);
    let one = contigs(&molecule(vec![(read.clone(), 1)]));
    let (seven, diag) = assemble_weighted(&molecule(vec![(read.clone(), 7)]));

    assert_eq!(seven.tokens, one.tokens, "seven identical reads must assemble like one");
    assert_eq!(seven.tokens, read);
    assert_eq!(seven.n_contigs, 1);
    assert_eq!(seven.n_reads, 7);
    assert_eq!(one.n_reads, 1);
    assert!(!diag.graph_built, "one distinct sequence still needs no graph");
}

#[test]
fn a_pass_through_molecule_keeps_its_cell_and_umi() {
    let out = contigs(&molecule(vec![(seq(3, 60), 4)]));
    assert_eq!(out.cell, CellId(42));
    assert_eq!(out.umi, Umi(7));
}

#[test]
fn two_reads_overlapping_by_thirty_bases_merge_into_one_contig() {
    let full = seq(4, 120);
    let left = full[..75].to_vec();
    let right = full[45..].to_vec();
    assert_eq!(left.len() + right.len() - full.len(), 30, "reads overlap by 30 bases");

    let (out, diag) = assemble_weighted(&molecule(vec![(left, 1), (right, 1)]));

    assert!(diag.graph_built);
    assert_eq!(out.n_contigs, 1);
    assert!(!out.tokens.contains(&tab()));
    assert_eq!(out.tokens.len(), 120, "merged length is L1 + L2 - overlap");
    assert_eq!(out.tokens, full);
    assert_eq!(diag.bubble_n_tokens, 0);
    assert!(!diag.self_loop);
}

#[test]
fn two_reads_without_overlap_emit_two_contigs_separated_by_one_tab() {
    let a = seq(5, 60);
    let b = seq(6, 60);
    assert!(kmers(&a).is_disjoint(&kmers(&b)), "test fixture must share no k-mer");

    let (out, _) = assemble_weighted(&molecule(vec![(a.clone(), 1), (b.clone(), 1)]));

    assert_eq!(out.n_contigs, 2);
    assert_eq!(out.recount_contigs(), 2, "n_contigs must survive recomputation");
    assert_eq!(out.tokens.iter().filter(|&&t| t == tab()).count(), 1, "exactly one Tab");

    let mut got = contigs_of(&out);
    got.sort();
    let mut want = vec![a, b];
    want.sort();
    assert_eq!(got, want, "each contig is one of the input reads, unaltered");
}

#[test]
fn a_bubble_tie_emits_n_at_the_disagreeing_position() {
    let base = seq(7, 80);
    let variant = with_variant_at(&base, 40);
    let (out, diag) = assemble_weighted(&molecule(vec![(base.clone(), 1), (variant, 1)]));

    assert_eq!(out.n_contigs, 1, "a bubble is one contig, not two");
    assert_eq!(out.tokens.len(), 80);
    assert_eq!(
        out.tokens.iter().filter(|&&t| t == n_token()).count(),
        1,
        "exactly one N: got {:?}",
        out.tokens
    );
    assert_eq!(out.tokens[40], n_token(), "the N sits at the variant position");
    assert_eq!(diag.bubble_n_tokens, 1);

    for (i, (&got, &want)) in out.tokens.iter().zip(base.iter()).enumerate() {
        if i != 40 {
            assert_eq!(got, want, "position {i} changed");
        }
    }
}

#[test]
fn a_weighted_majority_emits_the_majority_base_and_never_n() {
    let base = seq(8, 80);
    let variant = with_variant_at(&base, 40);
    let (out, diag) = assemble_weighted(&molecule(vec![(base.clone(), 2), (variant, 1)]));

    assert!(
        !out.tokens.contains(&n_token()),
        "majority 2:1 must not emit N: {:?}",
        out.tokens
    );
    assert_eq!(diag.bubble_n_tokens, 0);
    assert_eq!(out.n_contigs, 1, "the losing branch is consumed by the resolution");
    assert_eq!(out.tokens, base, "the majority sequence is emitted in full");
    assert_eq!(out.n_reads, 3);
}

#[test]
fn the_minority_read_can_win_when_its_weight_is_higher() {
    let base = seq(9, 80);
    let variant = with_variant_at(&base, 40);
    let (out, _) = assemble_weighted(&molecule(vec![(base, 1), (variant.clone(), 3)]));
    assert_eq!(out.tokens, variant);
}

fn flanked(variable: &[u8]) -> Vec<u8> {
    let mut v = seq(30, 25);
    v.extend_from_slice(variable);
    v.extend(seq(31, 25));
    v
}

#[test]
fn an_indel_tie_emits_n_over_the_divergent_region() {
    let a = flanked(&[Token::A as u8, Token::C as u8, Token::G as u8, Token::T as u8]);
    let b = flanked(&[
        Token::A as u8,
        Token::C as u8,
        Token::G as u8,
        Token::G as u8,
        Token::T as u8,
    ]);
    assert_eq!((a.len(), b.len()), (54, 55));

    {
        let policy = default_model();
        let (out, diag) = assemble(&molecule(vec![(a.clone(), 1), (b.clone(), 1)]), policy, K_DEFAULT);

        assert_eq!(
            out.n_contigs, 1,
            "the branches rejoin, so this is one contig ({policy:?})"
        );
        assert_eq!(
            out.tokens.len(),
            b.len(),
            "emitted length is max(|a|, |b|) ({policy:?})"
        );

        assert_eq!(out.tokens[..25], a[..25], "prefix flank ({policy:?})");
        assert_eq!(
            out.tokens[out.tokens.len() - 25..],
            a[a.len() - 25..],
            "suffix flank ({policy:?})"
        );

        let window = &out.tokens[25..out.tokens.len() - 25];
        assert_eq!(window.len(), 5, "the divergent window is max(4, 5) tokens wide");
        assert_eq!(
            window.iter().filter(|&&t| t == n_token()).count(),
            1,
            "one N ({policy:?})"
        );
        assert_eq!(diag.bubble_n_tokens, 1);
        let called: Vec<u8> = window.iter().copied().filter(|&t| t != n_token()).collect();
        assert_eq!(
            called,
            vec![Token::A as u8, Token::C as u8, Token::G as u8, Token::T as u8]
        );
    }
}

#[test]
fn a_run_of_unresolved_columns_splits_instead_of_emitting_an_n_run() {
    let a = flanked(&[Token::A as u8, Token::A as u8, Token::A as u8, Token::A as u8]);
    let b = flanked(&[Token::C as u8, Token::C as u8, Token::C as u8, Token::C as u8]);
    let reads = vec![(a.clone(), 1), (b.clone(), 1)];

    let (split, sd) = assemble(&molecule(reads), default_model(), K_DEFAULT);
    assert_eq!(sd.bubble_n_tokens, 0, "a split invents nothing and erases nothing");
    assert!(!split.tokens.contains(&n_token()));
    assert_eq!(
        split.n_contigs,
        4,
        "the two flanks and the two branches, and no more: got lengths {:?}",
        contigs_of(&split).iter().map(Vec::len).collect::<Vec<_>>()
    );
    let split_contigs = contigs_of(&split);
    let emitted: HashSet<&[u8]> = split_contigs.iter().flat_map(|c| c.windows(K_DEFAULT)).collect();
    for (read, name) in [(&a, "a"), (&b, "b")] {
        for km in read.windows(K_DEFAULT) {
            assert!(emitted.contains(km), "read {name} lost a k-mer");
        }
    }
}

#[test]
fn adjacent_disagreements_backed_by_a_majority_resolve_rather_than_split() {
    let a = flanked(&[Token::A as u8, Token::A as u8, Token::A as u8, Token::A as u8]);
    let b = flanked(&[Token::C as u8, Token::C as u8, Token::A as u8, Token::A as u8]);
    let (out, d) = assemble(&molecule(vec![(a.clone(), 3), (b, 1)]), default_model(), K_DEFAULT);
    assert_eq!(out.n_contigs, 1, "a decidable pair of columns must not split");
    assert_eq!(d.bubble_n_tokens, 0, "the 3:1 majority decides both");
    assert_eq!(out.tokens, a, "and the majority sequence is what survives");
}

#[test]
fn scattered_single_ties_still_emit_single_ns_under_weighted_split() {
    let base = seq(42, 80);
    let mut variant = with_variant_at(&base, 40);
    variant[45] = (variant[45] + 2) % 4;
    let (out, diag) = assemble(&molecule(vec![(base, 1), (variant, 1)]), default_model(), K_DEFAULT);
    assert_eq!(out.n_contigs, 1, "scattered point ties must not split");
    assert_eq!(diag.bubble_n_tokens, 2);
    assert!(!out.tokens.windows(2).any(|w| w[0] == n_token() && w[1] == n_token()));
}

#[test]
fn a_homopolymer_self_loop_is_discarded_and_the_flanks_survive() {
    let prefix = seq(50, 30);
    let run = vec![Token::A as u8; 30];
    let suffix = seq(51, 30);
    let mut read1 = prefix.clone();
    read1.extend_from_slice(&run);
    read1.extend_from_slice(&suffix);
    let mut read2 = suffix.clone();
    read2.extend_from_slice(&seq(52, 30));
    let reads = vec![(read1.clone(), 1), (read2.clone(), 1)];

    let longest_a = |tokens: &[u8]| {
        let mut best = 0usize;
        let mut cur = 0usize;
        for &t in tokens {
            cur = if t == Token::A as u8 { cur + 1 } else { 0 };
            best = best.max(cur);
        }
        best
    };

    let (out, d) = assemble(&molecule(reads), default_model(), K_DEFAULT);
    assert!(d.self_loop, "the count survives: discarded, still stamped");
    assert!(
        longest_a(&out.tokens) < K_DEFAULT,
        "the collapsed, length-wrong run must be gone"
    );
    assert_eq!(
        out.n_contigs, 2,
        "exactly the two flanks survive: `>= 2` would also accept a walk that cut them further"
    );
    let out_contigs = contigs_of(&out);
    let emitted: HashSet<&[u8]> = out_contigs.iter().flat_map(|c| c.windows(K_DEFAULT)).collect();
    for region in [&prefix, &suffix] {
        for km in region.windows(K_DEFAULT) {
            assert!(emitted.contains(km), "flank lost a k-mer");
        }
    }
}

fn drift(base: &[u8], n_mismatches: usize, flank: usize) -> Vec<u8> {
    let mut v = base.to_vec();
    let span = v.len() - flank;
    for i in 0..n_mismatches {
        let pos = flank + (i * span) / n_mismatches + 1;
        v[pos] = (v[pos] + 1) % 4;
    }
    v
}

#[test]
fn confident_branches_at_ten_percent_divergence_are_reported_as_two_molecules() {
    let base = seq(77, 120);
    let variant = drift(&base, 7, 40);
    let (out, d) = assemble(&molecule_q(vec![(base, 40), (variant, 40)]), default_model(), K_DEFAULT);
    assert!(
        d.n_branch_decisions > 0,
        "the fixture must actually reach a branch point"
    );
    assert!(
        d.n_origin_splits > 0,
        "7 disagreements between two Q40 reads must reject same-origin, got surprise {} against \
         OMEGA {}",
        d.origin_surprise_max,
        scrm_asm::consensus::OMEGA_LOG10
    );
    assert_eq!(
        d.n_origin_splits, 1,
        "exactly one origin split -- one branch point, rejected once"
    );
    assert_eq!(
        out.n_contigs,
        3,
        "and the branches must surface separately: the shared flank plus one contig per molecule, \
         got lengths {:?}",
        contigs_of(&out).iter().map(Vec::len).collect::<Vec<_>>()
    );
}

#[test]
fn a_low_quality_dissent_cannot_testify_against_same_origin() {
    let base = seq(77, 120);
    let variant = drift(&base, 7, 40);
    let (_, d) = assemble(&molecule_q(vec![(base, 40), (variant, 8)]), default_model(), K_DEFAULT);
    assert!(d.n_branch_decisions > 0);
    assert_eq!(
        d.n_origin_splits, 0,
        "disagreements a Q8 read fully explains must not reject same-origin"
    );
}

#[test]
fn one_mismatch_is_explained_by_error_and_never_rejects_same_origin() {
    let base = seq(78, 120);
    let variant = drift(&base, 1, 40);
    let (out, d) = assemble(&molecule_q(vec![(base, 40), (variant, 40)]), default_model(), K_DEFAULT);
    assert_eq!(d.n_origin_splits, 0, "one mismatch is not evidence of two molecules");
    assert_eq!(out.n_contigs, 1, "so the branches merge");
    assert_eq!(d.n_tie, 1, "and the disagreement is resolved as an exact tie -> N");
    assert_eq!(
        contigs_of(&out)[0].iter().filter(|&&t| t == n_token()).count(),
        1,
        "exactly one N, at the disagreeing position"
    );
}

#[test]
fn the_origin_surprise_matches_an_independent_poisson_tail() {
    let m = scrm_asm::consensus::ConsensusModel::default();
    assert!(
        (m.origin_surprise(1, 1.0) - 0.199_8).abs() < 1e-3,
        "{}",
        m.origin_surprise(1, 1.0)
    );
    assert!(
        (m.origin_surprise(2, 0.1) - 2.329_8).abs() < 1e-3,
        "{}",
        m.origin_surprise(2, 0.1)
    );
    let s: Vec<f64> = (1..8).map(|k| m.origin_surprise(k, 0.2)).collect();
    assert!(
        s.windows(2).all(|w| w[1] > w[0]),
        "more disagreements must be more surprising: {s:?}"
    );
    assert_eq!(m.origin_surprise(0, 1e-9), 0.0);
}

#[test]
fn a_long_unequal_indel_interior_splits_instead_of_emitting_an_n_run() {
    let a = flanked(&[Token::A as u8, Token::C as u8, Token::G as u8, Token::T as u8]);
    let b = flanked(&[
        Token::T as u8,
        Token::G as u8,
        Token::C as u8,
        Token::A as u8,
        Token::G as u8,
        Token::G as u8,
        Token::A as u8,
    ]);
    let reads = vec![(a.clone(), 1), (b.clone(), 1)];

    let (split, sd) = assemble(&molecule(reads), default_model(), K_DEFAULT);
    assert_eq!(sd.bubble_n_tokens, 0, "a split invents nothing and erases nothing");
    assert!(!split.tokens.contains(&n_token()), "no N under split");
    assert_eq!(
        split.n_contigs,
        4,
        "the interior must split into exactly the two flanks and the two interiors, got lengths {:?}",
        contigs_of(&split).iter().map(Vec::len).collect::<Vec<_>>()
    );
    let split_contigs = contigs_of(&split);
    let emitted: HashSet<&[u8]> = split_contigs.iter().flat_map(|c| c.windows(K_DEFAULT)).collect();
    for (read, name) in [(&a, "a"), (&b, "b")] {
        for km in read.windows(K_DEFAULT) {
            assert!(emitted.contains(km), "read {name} lost a k-mer under split");
        }
    }
}

#[test]
fn quality_decides_a_column_where_weight_counting_is_forced_to_n() {
    let base = seq(40, 80);
    let variant = with_variant_at(&base, 40);

    let (majority, md) = assemble(
        &molecule(vec![(base.clone(), 5), (variant.clone(), 1)]),
        default_model(),
        K_DEFAULT,
    );
    assert_eq!(majority.tokens, base, "a 5-vs-1 majority still calls the majority base");
    assert_eq!(md.bubble_n_tokens, 0);

    let (tie, td) = assemble(
        &molecule_q(vec![(base.clone(), 30), (variant.clone(), 30)]),
        default_model(),
        K_DEFAULT,
    );
    assert_eq!(tie.tokens[40], n_token(), "equal confidence disagreeing is a tie");
    assert_eq!(td.n_tie, 1);

    let (decided, dd) = assemble(
        &molecule_q(vec![(base.clone(), 40), (variant, 8)]),
        default_model(),
        K_DEFAULT,
    );
    assert_eq!(decided.tokens, base, "Phred 40 beats Phred 8 at the same weight");
    assert_eq!(dd.bubble_n_tokens, 0, "and it is a real call, not an N");
    assert_eq!(
        tie.tokens.len(),
        decided.tokens.len(),
        "quality must not change the length"
    );
}

#[test]
fn an_exact_tie_is_n_and_is_recorded_as_a_tie() {
    let base = seq(41, 80);
    let variant = with_variant_at(&base, 40);
    let reads = vec![(base, 1), (variant, 1)];
    let (out, diag) = assemble(&molecule(reads), default_model(), K_DEFAULT);
    assert_eq!(out.tokens[40], n_token());
    assert_eq!(diag.bubble_n_tokens, 1);
    assert_eq!(diag.n_tie, 1, "an exact tie, not a low-confidence call");
    assert_eq!(diag.n_low_confidence, 0);
    assert_eq!(diag.n_indel_interior, 0);
}

#[test]
fn an_equal_length_tie_keeps_the_per_position_consensus() {
    let base = seq(42, 80);
    let mut variant = with_variant_at(&base, 40);
    variant[45] = (variant[45] + 2) % 4;
    let (out, diag) = assemble(
        &molecule(vec![(base.clone(), 1), (variant, 1)]),
        default_model(),
        K_DEFAULT,
    );

    assert_eq!(out.tokens.len(), 80);
    assert_eq!(diag.bubble_n_tokens, 2);
    let n_at: Vec<usize> = out
        .tokens
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == n_token())
        .map(|(i, _)| i)
        .collect();
    assert_eq!(n_at, vec![40, 45], "only the disagreeing positions");
    assert_eq!(
        out.tokens[41..45],
        base[41..45],
        "the agreed bases between them survive"
    );
}

#[test]
fn molecules_without_a_branch_come_through_as_their_own_sequence() {
    let full = seq(43, 120);
    let pass_through = seq(44, 90);
    let (a, b) = (seq(45, 60), seq(46, 60));
    assert!(
        kmers(&a).is_disjoint(&kmers(&b)),
        "the disjoint case must share no k-mer"
    );

    let cases: Vec<(Molecule, Vec<Vec<u8>>)> = vec![
        (molecule(vec![(pass_through.clone(), 3)]), vec![pass_through]),
        (
            molecule(vec![(full[..75].to_vec(), 1), (full[45..].to_vec(), 1)]),
            vec![full.clone()],
        ),
        (molecule(vec![(a.clone(), 1), (b.clone(), 2)]), vec![a, b]),
    ];
    for (m, want) in cases {
        let (out, d) = assemble_weighted(&m);
        let mut got = contigs_of(&out);
        got.sort();
        let mut want_sorted = want.clone();
        want_sorted.sort();
        assert_eq!(got, want_sorted, "a branchless molecule was altered");
        assert_eq!(out.n_contigs as usize, want.len());
        assert_eq!(out.recount_contigs(), want.len());
        assert!(!out.tokens.contains(&n_token()), "no N may appear without a branch");
        assert_eq!(
            (
                d.bubble_n_tokens,
                d.n_tie,
                d.n_low_confidence,
                d.n_indel_interior,
                d.n_branch_decisions
            ),
            (0, 0, 0, 0, 0),
            "a branchless molecule must reach no branch decision at all"
        );
    }
}

fn with_homopolymer(seed: u64, run: usize, base: u8) -> Vec<u8> {
    let mut s = seq(seed, 40);
    s[39] = (base + 1) % 4;
    s.extend(std::iter::repeat_n(base, run));
    let mut tail = seq(seed + 1000, 40);
    tail[0] = (base + 2) % 4;
    s.extend(tail);
    s
}

#[test]
fn a_homopolymer_run_of_twenty_five_increments_self_loop_molecules() {
    let full = with_homopolymer(10, 25, 0);
    let left = full[..70].to_vec();
    let right = full[35..].to_vec();
    let m = molecule(vec![(left, 1), (right, 1)]);

    let (out, diag) = assemble_weighted(&m);
    assert!(diag.graph_built);
    assert!(diag.self_loop, "a 25-base run at k=19 must register as a self-loop");

    let mut stats = StatsCollector::default();
    stats.record(&m, &out, &diag, Retention::Kept);
    assert_eq!(stats.self_loop_molecules(), 1);
    assert_eq!(stats.finish().self_loop_molecules, 1);

    assert_eq!(
        out.n_contigs, 2,
        "the molecule splits around the discarded run into exactly its two flanks -- a bound of \
         `>= 2` here would also accept a walk that cut the flanks as well"
    );
    let longest_run = out.tokens.split(|&t| t != 0).map(<[u8]>::len).max().unwrap_or(0);
    assert!(
        longest_run < K,
        "no length-wrong run may reach the corpus, got {longest_run}"
    );
}

#[test]
fn a_run_shorter_than_k_does_not_register_a_self_loop() {
    let full = with_homopolymer(11, K - 1, 3);
    let left = full[..70].to_vec();
    let right = full[35..].to_vec();
    let m = molecule(vec![(left, 1), (right, 1)]);

    let (out, diag) = assemble_weighted(&m);
    assert!(!diag.self_loop, "an 18-base run fits inside a k-mer and cannot loop");
    assert_eq!(out.tokens, full, "and the run keeps its length");

    let mut stats = StatsCollector::default();
    stats.record(&m, &out, &diag, Retention::Kept);
    assert_eq!(stats.self_loop_molecules(), 0);
}

#[test]
fn every_token_is_inside_the_alphabet_and_newline_is_absent() {
    let base = seq(12, 80);
    let cases = vec![
        molecule(vec![(base.clone(), 1)]),
        molecule(vec![(base.clone(), 1), (with_variant_at(&base, 40), 1)]),
        molecule(vec![(seq(13, 60), 1), (seq(14, 60), 1)]),
    ];
    for m in cases {
        let out = contigs(&m);
        assert!(!out.tokens.contains(&(Token::Newline as u8)), "Newline is never stored");
        for &t in &out.tokens {
            assert!(t < ALPHABET_SIZE, "token {t} outside the alphabet");
        }
    }
}

#[test]
fn stats_report_the_pass_through_fraction_and_check_n_contigs() {
    let mut stats = StatsCollector::default();
    for seed in 0..3 {
        let m = molecule(vec![(seq(100 + seed, 60), 2)]);
        let (out, diag) = assemble_weighted(&m);
        stats.record(&m, &out, &diag, Retention::Kept);
    }
    let full = seq(200, 120);
    let m = molecule(vec![(full[..75].to_vec(), 1), (full[45..].to_vec(), 1)]);
    let (out, diag) = assemble_weighted(&m);
    stats.record(&m, &out, &diag, Retention::Kept);

    let s = stats.finish();
    assert_eq!(stats.molecules(), 4);
    assert!((s.pass_through_fraction - 0.75).abs() < 1e-12);
    assert_eq!(s.contigs_per_molecule_hist[1], 4);
    assert_eq!(s.contig_length_hist[60], 3);
    assert_eq!(s.contig_length_hist[120], 1);
    assert_eq!(s.distinct_seqs_per_molecule_hist[1], 3);
    assert_eq!(s.distinct_seqs_per_molecule_hist[2], 1);
    assert_eq!(s.reads_per_molecule_hist[2], 4, "three at depth 2, plus the 1+1 pair");
    assert_eq!(s.self_loop_molecules, 0);
    assert_eq!(s.bubble_n_tokens, 0);
}

#[test]
fn a_filtered_molecule_still_counts_toward_the_derivation_instruments() {
    let m = molecule(vec![(vec![0u8; 40], 1), (vec![0u8; 41], 1)]);
    let (out, diag) = assemble_weighted(&m);
    assert!(diag.self_loop, "a 40-base run at k=19 must register as a self-loop");
    assert_eq!(
        out.tokens.len(),
        0,
        "and the molecule is nothing else, so no contig survives"
    );

    let mut stats = StatsCollector::default();
    stats.record(&m, &out, &diag, Retention::LengthFiltered);
    assert_eq!(
        stats.self_loop_molecules(),
        1,
        "the discard must stay a stamped measurement even when it empties the molecule"
    );
    assert_eq!(
        stats.molecules(),
        0,
        "but the molecule is not in the corpus and is not counted as such"
    );
    assert!(
        stats.finish().contigs_per_molecule_hist.is_empty(),
        "no content histogram may describe a molecule the corpus does not contain"
    );

    let base = seq(300, 80);
    let m2 = molecule(vec![(base.clone(), 1), (with_variant_at(&base, 40), 1)]);
    let (out2, d2) = assemble_weighted(&m2);
    assert!(
        d2.n_branch_decisions > 0 && d2.bubble_n_tokens > 0,
        "fixture must branch"
    );

    let mut st = StatsCollector::default();
    st.record(&m2, &out2, &d2, Retention::LengthFiltered);
    let f = st.finish();
    assert_eq!(
        (f.branch_decisions, f.origin_k_total),
        (d2.n_branch_decisions, d2.origin_k_total),
        "the origin test's k and its denominator describe every molecule assembled"
    );
    assert!(
        f.origin_lambda_total > 0.0,
        "and so does the lambda it is compared against"
    );
    assert_eq!(
        f.origin_surprise_hist.iter().sum::<u64>(),
        1,
        "the surprise distribution OMEGA is read off must include filtered molecules"
    );
    assert_eq!(
        f.bubble_n_tokens, 0,
        "an N in a filtered molecule reaches no corpus, so it is not in the corpus's N budget"
    );
}

#[test]
fn stats_merge_is_additive() {
    let base = seq(300, 80);
    let m = molecule(vec![(base.clone(), 1), (with_variant_at(&base, 40), 1)]);
    let (out, diag) = assemble_weighted(&m);

    let mut a = StatsCollector::default();
    a.record(&m, &out, &diag, Retention::Kept);
    let mut b = StatsCollector::default();
    b.record(&m, &out, &diag, Retention::Kept);
    b.record(&m, &out, &diag, Retention::Kept);

    a.merge(&b);
    assert_eq!(a.molecules(), 3);
    assert_eq!(a.bubble_n_tokens(), 3, "one tie N per molecule");
    assert_eq!(a.finish().contigs_per_molecule_hist[1], 3);
}

mod properties {
    use super::*;
    use proptest::prelude::*;

    prop_compose! {
        fn read()(bases in prop::collection::vec(0u8..4, MIN_TRIM_LEN..90)) -> Vec<u8> {
            bases
        }
    }

    prop_compose! {
        fn random_molecule()(reads in prop::collection::vec((read(), 1u32..5), 1..5)) -> Molecule {
            let mut seen = HashSet::new();
            let reads: Vec<_> = reads.into_iter().filter(|(s, _)| seen.insert(s.clone())).collect();
            molecule(reads)
        }
    }

    prop_compose! {
        fn fragment_molecule()(
            fragment in prop::collection::vec(0u8..4, 200..400),
            picks in prop::collection::vec((0usize..200, 30usize..90, 1u32..4), 2..6),
            errors in prop::collection::vec((0usize..1000, 1u8..4), 0..6),
        ) -> Molecule {
            let mut seen = HashSet::new();
            let mut reads = Vec::new();
            for (i, (start, len, weight)) in picks.into_iter().enumerate() {
                let start = start.min(fragment.len() - MIN_TRIM_LEN);
                let end = (start + len).min(fragment.len());
                let mut read = fragment[start..end].to_vec();
                for &(pos, delta) in errors.iter().skip(i).take(2) {
                    if !read.is_empty() {
                        let pos = pos % read.len();
                        read[pos] = (read[pos] + delta) % 4;
                    }
                }
                if read.len() >= MIN_TRIM_LEN && seen.insert(read.clone()) {
                    reads.push((read, weight));
                }
            }
            molecule(reads)
        }
    }

    proptest! {
        #[test]
        fn overlapping_reads_with_errors_assemble_without_panicking(
            m in fragment_molecule(),
        ) {
            let (out, diag) = assemble(&m, default_model(), K_DEFAULT);
            prop_assert_eq!(out.n_contigs as usize, out.recount_contigs());
            prop_assert!(!out.tokens.contains(&(Token::Newline as u8)));
            for &t in &out.tokens {
                prop_assert!(t < ALPHABET_SIZE, "token {} outside the alphabet", t);
            }
            let n_count = out.tokens.iter().filter(|&&t| t == n_token()).count() as u64;
            prop_assert_eq!(n_count, diag.bubble_n_tokens);
            prop_assert_eq!(
                diag.bubble_n_tokens,
                diag.n_tie + diag.n_low_confidence + diag.n_indel_interior
            );
            for contig in contigs_of(&out) {
                prop_assert!(contig.len() >= K || out.tokens.is_empty());
            }
        }

        #[test]
        fn newline_never_appears_and_tokens_stay_in_the_alphabet(m in random_molecule()) {
            let (out, _) = assemble_weighted(&m);
            prop_assert!(!out.tokens.contains(&(Token::Newline as u8)));
            for &t in &out.tokens {
                prop_assert!(t < ALPHABET_SIZE, "token {} outside the alphabet", t);
            }
        }

        #[test]
        fn n_contigs_always_survives_recomputation(m in random_molecule()) {
            let out = contigs(&m);
            prop_assert_eq!(out.n_contigs as usize, out.recount_contigs());
        }

        #[test]
        fn identity_and_depth_are_carried_through(m in random_molecule()) {
            let out = contigs(&m);
            prop_assert_eq!(out.cell, m.cell);
            prop_assert_eq!(out.umi, m.umi);
            prop_assert_eq!(out.n_reads, m.n_reads);
        }

        #[test]
        fn no_contig_is_empty_and_tabs_only_separate(m in random_molecule()) {
            let out = contigs(&m);
            for contig in contigs_of(&out) {
                prop_assert!(!contig.is_empty());
            }
            prop_assert_ne!(out.tokens.first(), Some(&tab()));
            prop_assert_ne!(out.tokens.last(), Some(&tab()));
        }

        #[test]
        fn assembly_is_deterministic(m in random_molecule()) {
            prop_assert_eq!(contigs(&m).tokens, contigs(&m).tokens);
        }

        #[test]
        fn a_single_read_is_always_returned_verbatim(bases in read()) {
            let out = contigs(&molecule(vec![(bases.clone(), 3)]));
            prop_assert_eq!(out.tokens, bases);
        }
    }
}

#[test]
fn an_error_near_a_read_end_is_corrected_rather_than_duplicated() {
    let full = seq(20, 80);
    let variant = with_variant_at(&full, 75);

    let (out, diag) = assemble_weighted(&molecule(vec![(full.clone(), 2), (variant.clone(), 1)]));
    assert!(!out.tokens.contains(&n_token()), "a 2:1 majority decides the base");
    assert_eq!(diag.bubble_n_tokens, 0);
    assert_eq!(
        out.n_contigs, 1,
        "the corrected read must not reappear as a near-duplicate"
    );
    assert_eq!(out.tokens, full, "and the majority sequence is what survives");

    let (tied, td) = assemble_weighted(&molecule(vec![(full.clone(), 1), (variant.clone(), 1)]));
    assert_eq!(td.n_tie + td.n_low_confidence + td.n_indel_interior, td.bubble_n_tokens);
    assert_eq!(
        (td.n_tie, td.n_low_confidence, td.n_indel_interior, td.bubble_n_tokens),
        (1, 0, 0, 1),
        "a 1:1 dissent at equal quality is exactly one tie and exactly one N"
    );
    assert_eq!(
        tied.n_contigs, 1,
        "and it stays ONE contig -- the tip is not re-emitted"
    );
    assert_eq!(tied.tokens.len(), full.len(), "the contig spans the whole molecule");
    assert_eq!(tied.tokens[75], n_token(), "the N sits at the disagreeing position");
    for (i, (&got, &want)) in tied.tokens.iter().zip(full.iter()).enumerate() {
        if i != 75 {
            assert_eq!(got, want, "position {i} changed");
        }
    }
}

#[test]
fn the_resolution_name_is_the_string_the_stamp_records() {
    let name = scrm_asm::resolution_name();
    assert_eq!(
        name,
        format!(
            "likelihood-q{}-pre{}-post{}-omega{}-{}",
            scrm_asm::consensus::MIN_CONSENSUS_Q,
            scrm_asm::consensus::ERROR_RATE_PRE,
            scrm_asm::consensus::ERROR_RATE_POST,
            scrm_asm::consensus::OMEGA_LOG10,
            scrm_asm::WALK_RULE
        )
    );
    assert!(!scrm_asm::WALK_RULE.is_empty(), "the walk rule must name the walk");
    assert!(name.starts_with("likelihood-"), "{name}");
}

#[test]
fn a_single_base_disagreement_on_a_tip_becomes_one_n() {
    let mut coincidences = 0;
    for far in 0..4u8 {
        let mut long_read = seq(41, 106);
        long_read[105] = far;
        let mut tip = long_read[..66].to_vec();
        tip[55] = (tip[55] + 1) % 4;
        if *tip.last().unwrap() == far {
            coincidences += 1;
        }

        let (out, d) = assemble(
            &molecule(vec![(long_read.clone(), 1), (tip, 1)]),
            default_model(),
            K_DEFAULT,
        );
        assert_eq!(
            out.n_contigs,
            1,
            "a tip must be consumed into one contig, whatever the long read's last base is; \
             got lengths {:?} for far={far}",
            contigs_of(&out).iter().map(Vec::len).collect::<Vec<_>>()
        );
        assert_eq!(out.tokens.len(), long_read.len(), "the contig spans the whole molecule");
        assert_eq!(
            (d.bubble_n_tokens, d.n_tie),
            (1, 1),
            "one disputed column at equal quality is one tie and one N (far={far})"
        );
        assert_eq!(
            d.n_branch_decisions, 1,
            "the column must be CALLED, not stepped around (far={far})"
        );
    }
    assert_eq!(
        coincidences, 1,
        "the loop must cover the coincidence case exactly once, or it is not testing it"
    );

    let mut coincidences = 0;
    for far in 0..4u8 {
        let mut long_read = seq(41, 106);
        long_read[0] = far;
        let mut tip = long_read[40..].to_vec();
        tip[10] = (tip[10] + 1) % 4;
        if tip[0] == far {
            coincidences += 1;
        }

        let (out, d) = assemble(
            &molecule(vec![(long_read.clone(), 1), (tip, 1)]),
            default_model(),
            K_DEFAULT,
        );
        assert_eq!(
            out.n_contigs,
            1,
            "the 5' mirror must behave the same; got lengths {:?} for far={far}",
            contigs_of(&out).iter().map(Vec::len).collect::<Vec<_>>()
        );
        assert_eq!(out.tokens.len(), long_read.len(), "the contig spans the whole molecule");
        assert_eq!(
            (d.bubble_n_tokens, d.n_tie),
            (1, 1),
            "one disputed column at equal quality is one tie and one N (mirror, far={far})"
        );
        assert_eq!(
            d.n_branch_decisions, 1,
            "the column must be CALLED, not stepped around (mirror, far={far})"
        );
    }
    assert_eq!(coincidences, 1, "the mirror loop must cover the coincidence case too");
}
#[test]
fn a_second_site_beyond_a_non_rejoining_branch_does_not_cut_the_molecule() {
    let cat = |p: &[&[u8]]| -> Vec<u8> { p.concat() };
    let fl = seq(11, 40);
    let mid = seq(33, 40);
    let fr = seq(22, 40);

    let a = cat(&[&fl, &[0], &mid, &[0], &fr]);
    let b = cat(&[&fl, &[0], &mid, &[1], &fr]);
    let tip_5 = cat(&[&fl, &[1], &mid[..6]]);

    let five = contigs(&molecule_q(vec![(a.clone(), 30), (b.clone(), 30), (tip_5, 30)]));
    assert_eq!(
        five.n_contigs,
        1,
        "a tip before a bubble must not cut the molecule at the bubble; got lengths {:?}",
        five.tokens
            .split(|&t| t == Token::Tab as u8)
            .map(<[u8]>::len)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        five.tokens.len(),
        a.len(),
        "the single contig must span the whole molecule"
    );

    let tip_3 = cat(&[&fl, &[0], &mid, &[0], &fr[..20], &[(fr[20] + 1) % 4]]);
    let three = contigs(&molecule_q(vec![(a.clone(), 30), (b, 30), (tip_3, 30)]));
    assert_eq!(
        five.n_contigs, three.n_contigs,
        "which end the tip sits on must not change the decomposition"
    );
    assert_eq!(
        three.tokens.len(),
        a.len(),
        "and the mirror's single contig must span the whole molecule too -- the contig COUNT alone \
         would not notice a mirror that came back short"
    );
}

#[test]
fn a_read_confined_to_one_branch_cannot_silence_the_bubble() {
    let cat = |p: &[&[u8]]| -> Vec<u8> { p.concat() };
    let fl = seq(11, 40);
    let fr = seq(22, 40);
    let a = cat(&[&fl, &[0], &fr]);
    let b = cat(&[&fl, &[1], &fr]);

    let branch_only = a[fl.len() + 1 - K..fl.len() + K].to_vec();
    assert_eq!(
        branch_only.len(),
        2 * K - 1,
        "the read must span exactly the branch node"
    );

    let (out, d) = assemble_weighted(&molecule(vec![(a.clone(), 1), (b, 1), (branch_only, 2)]));
    assert_eq!(
        out.n_contigs,
        1,
        "the bubble must still be resolved; got lengths {:?}",
        out.tokens
            .split(|&t| t == Token::Tab as u8)
            .map(<[u8]>::len)
            .collect::<Vec<_>>()
    );
    assert_eq!(out.tokens.len(), a.len(), "the contig must span the whole molecule");
    assert_eq!(
        out.tokens[fl.len()],
        0,
        "the disputed column must be called for the majority base"
    );
    assert_ne!(
        out.tokens[fl.len()],
        Token::N as u8,
        "the disputed column must not be N"
    );
    assert_eq!(d.n_tie, 0, "2 observations against 1 is not a tie");
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn walk_battery() -> Vec<Molecule> {
    let mut out = Vec::new();
    for seed in [1u64, 2, 3] {
        let base = seq(seed, 120);
        out.push(molecule_q(vec![(base[..90].to_vec(), 30), (base[30..].to_vec(), 30)]));
        out.push(molecule_q(vec![(base.clone(), 30), (with_variant_at(&base, 60), 30)]));
        out.push(molecule_q(vec![(base.clone(), 35), (with_variant_at(&base, 60), 12)]));
        let two = with_variant_at(&with_variant_at(&base, 40), 80);
        out.push(molecule_q(vec![(base.clone(), 30), (two, 30)]));
        let var = with_variant_at(&base, 60);
        out.push(molecule_q(vec![
            (base.clone(), 30),
            (var.clone(), 30),
            (var[45..75].to_vec(), 30),
        ]));
        let mut tip = base[..70].to_vec();
        let n = tip.len();
        tip[n - 1] = (tip[n - 1] + 2) % 4;
        out.push(molecule_q(vec![(base.clone(), 30), (tip, 30)]));
    }
    for far in 0..4u8 {
        let mut long_read = seq(41, 106);
        long_read[105] = far;
        let mut tip = long_read[..66].to_vec();
        tip[55] = (tip[55] + 1) % 4;
        out.push(molecule(vec![(long_read, 1), (tip, 1)]));
    }
    out
}

#[test]
fn the_walk_rule_names_the_walk_it_produces() {
    const PINNED_RULE: &str = "walk2-consume-tip-anchored-rejoin-flank-seeded";

    assert_eq!(
        scrm_asm::WALK_RULE,
        PINNED_RULE,
        "WALK_RULE is pinned to the value above. If the walk moved, update WALK_RULE and the \
         digest below in the same commit."
    );

    let mut bytes = Vec::new();
    for m in walk_battery() {
        let out = contigs(&m);
        bytes.extend_from_slice(&out.tokens);
        bytes.push(0xFF);
    }
    let got = fnv1a(&bytes);

    const PINNED_DIGEST: u64 = 0x73ac_8fcc_e738_4bbe;
    assert_eq!(
        got, PINNED_DIGEST,
        "the walk's output over the battery moved while WALK_RULE stayed {PINNED_RULE:?}, so two \
         corpora built either side of this change would carry the same compat_key. Bump WALK_RULE \
         and this digest together."
    );
}
