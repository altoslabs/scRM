use std::path::{Path, PathBuf};
use std::process::Command;

use scrm_core::Token;

const TAB: u8 = Token::Tab as u8;

const CELL_A: &str = "ACGTACGTACGTACGT";
const CELL_B: &str = "TTTTGGGGCCCCAAAA";
const CELL_C: &str = "GATTACAGATTACAGA";

const CDNA: &str = "ACGTTGCATTGACCGATCAGTTGCAATCGGATCAGTTACGGATCCAGTTAACGGTTCAGC";

const NEW_TAIL: &str = "TGACCAGTTGCAATCGGATCCAGTTACGGATCAGT";

const OVERLAP_BP: usize = 25;

const _: () = assert!(
    OVERLAP_BP >= scrm_core::K_DEFAULT,
    "the overlap fixture must share at least one whole k-mer or it tests nothing"
);

fn cdna_right() -> String {
    format!("{}{}", &CDNA[CDNA.len() - OVERLAP_BP..], NEW_TAIL)
}

fn base_to_token(b: u8) -> u8 {
    match b {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        other => panic!("fixture base {} is not ACGT", other as char),
    }
}

fn tokens(seq: &str) -> Vec<u8> {
    seq.bytes().map(base_to_token).collect()
}

#[derive(Clone)]
struct Pair {
    cb: String,
    umi: String,
    r2: String,
    r2_qual: Option<String>,
}

fn pair(cb: &str, umi: &str, r2: &str) -> Pair {
    assert_eq!(cb.len(), 16, "cb must be 16 bp");
    assert_eq!(umi.len(), 12, "umi must be 12 bp");
    Pair {
        cb: cb.to_string(),
        umi: umi.to_string(),
        r2: r2.to_string(),
        r2_qual: None,
    }
}

fn pair_low_q_at(cb: &str, umi: &str, r2: &str, pos: usize, phred: u8) -> Pair {
    let mut q: Vec<u8> = vec![b'I'; r2.len()];
    q[pos] = 33 + phred;
    Pair {
        r2_qual: Some(String::from_utf8(q).unwrap()),
        ..pair(cb, umi, r2)
    }
}

fn with_sub_at(r2: &str, pos: usize) -> String {
    let mut b = r2.as_bytes().to_vec();
    b[pos] = match b[pos] {
        b'A' => b'C',
        b'C' => b'G',
        b'G' => b'T',
        _ => b'A',
    };
    String::from_utf8(b).unwrap()
}

fn write_fastqs(dir: &Path, pairs: &[Pair]) -> (PathBuf, PathBuf) {
    let (mut r1, mut r2) = (String::new(), String::new());
    for (i, p) in pairs.iter().enumerate() {
        let seq1 = format!("{}{}", p.cb, p.umi);
        r1.push_str(&format!("@READ{i} 1:N:0:AAAA\n{seq1}\n+\n{}\n", "I".repeat(seq1.len())));
        let q2 = p.r2_qual.clone().unwrap_or_else(|| "I".repeat(p.r2.len()));
        assert_eq!(q2.len(), p.r2.len(), "quality must match the read length");
        r2.push_str(&format!("@READ{i} 2:N:0:AAAA\n{}\n+\n{q2}\n", p.r2));
    }
    let (p1, p2) = (dir.join("r1.fastq"), dir.join("r2.fastq"));
    std::fs::write(&p1, r1).unwrap();
    std::fs::write(&p2, r2).unwrap();
    (p1, p2)
}

fn build_corpus(dir: &Path, pairs: &[Pair], cells: &[&str]) -> PathBuf {
    build_corpus_with(dir, pairs, cells, &["--min-umi-len", "1", "--min-cell-molecules", "1"])
}

fn build_corpus_with(dir: &Path, pairs: &[Pair], cells: &[&str], extra: &[&str]) -> PathBuf {
    let out = dir.join("corpus");
    let o = try_build(dir, &out, pairs, cells, &["--threads", "2"], extra);
    assert!(
        o.status.success(),
        "the pipeline exited with {}\n--- stderr ---\n{}",
        o.status,
        String::from_utf8_lossy(&o.stderr)
    );
    out
}

fn try_build(
    dir: &Path,
    out: &Path,
    pairs: &[Pair],
    cells: &[&str],
    fixed: &[&str],
    extra: &[&str],
) -> std::process::Output {
    let (r1, r2) = write_fastqs(dir, pairs);
    let bc = dir.join("barcodes.tsv");
    std::fs::write(&bc, cells.iter().map(|c| format!("{c}-1\n")).collect::<String>()).unwrap();

    let sheet = sheet_for(out, &r1, &r2, &bc, "1{b[16]u[12]}2{r:}");
    Command::new(env!("CARGO_BIN_EXE_scrm-corpus"))
        .args(["--samplesheet", sheet.to_str().unwrap()])
        .args(["--out", out.parent().unwrap().to_str().unwrap()])
        .args(fixed)
        .args(extra)
        .output()
        .expect("failed to run scrm-corpus")
}

fn sheet_for(out: &Path, r1: &Path, r2: &Path, bc: &Path, geometry: &str) -> PathBuf {
    let name = out.file_name().unwrap().to_str().unwrap();
    let root = out.parent().unwrap();
    std::fs::create_dir_all(root).unwrap();
    let path = root.join(format!("samplesheet-{name}.csv"));
    std::fs::write(
        &path,
        format!(
            "experiment,r1,r2,barcodes,geometry\n{name},{},{},{},{geometry}\n",
            r1.display(),
            r2.display(),
            bc.display()
        ),
    )
    .unwrap();
    path
}

fn write_fastqs_10x(dir: &Path, pairs: &[Pair], sample: &str) -> (PathBuf, PathBuf) {
    let (generic1, generic2) = write_fastqs(dir, pairs);
    let r1 = dir.join(format!("{sample}_S7_L001_R1_001.fastq"));
    let r2 = dir.join(format!("{sample}_S7_L001_R2_001.fastq"));
    std::fs::rename(&generic1, &r1).unwrap();
    std::fs::rename(&generic2, &r2).unwrap();
    (r1, r2)
}

fn tmp(name: &str) -> PathBuf {
    // PID-UNIQUE, like the four sibling test files. A fixed path here is what made two
    // concurrent `cargo test --workspace` runs collide: one builds a corpus into
    // e2e-determinism/corpus-t4 and the builder then REFUSES to overwrite it, so the second
    // run fails on the first run's scratch. A comment in a launcher cannot prevent that.
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("e2e-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn read_rows(corpus: &Path) -> Vec<(String, u32, Vec<u8>)> {
    let manifest = scrm_corpus::Manifest::read(corpus).expect("manifest");
    let mut out = Vec::new();
    for path in manifest.shard_paths(corpus) {
        let mut r = scrm_corpus::CorpusReader::open(&path).expect("shard");
        while let Some(batch) = r.next_batch() {
            let batch = batch.expect("batch");
            for i in 0..batch.len() {
                let row = batch.row(i);
                out.push((row.cell_id.to_string(), row.n_reads, row.tokens.to_vec()));
            }
        }
    }
    out
}

#[test]
fn the_pipeline_produces_the_expected_corpus() {
    let dir = tmp("full");

    const U1: &str = "AAAAAAAAAAAA";
    const U2: &str = "CCCCCCCCCCCC";
    const U3: &str = "GGGGGGGGGGGG";
    const U4: &str = "TTTTTTTTTTTT";
    const U1_NEIGHBOUR: &str = "AAAAAAAAAAAC";
    const CELL_A_1MM: &str = "ACGTACGTACGTACGA";
    const NOT_A_CELL: &str = "CCCCAAAATTTTGGGG";

    let mut pairs = vec![
        pair(CELL_A, U1, CDNA),
        pair(CELL_A, U1, CDNA),
        pair(CELL_A_1MM, U1, CDNA),
        pair(CELL_A, U1_NEIGHBOUR, CDNA),
        pair(CELL_A, U2, CDNA),
        pair(CELL_A, U2, &cdna_right()),
    ];

    let mut variant: Vec<u8> = CDNA.bytes().collect();
    variant[30] = if variant[30] == b'A' { b'C' } else { b'A' };
    let variant = String::from_utf8(variant).unwrap();
    for _ in 0..3 {
        pairs.push(pair(CELL_B, U3, CDNA));
    }
    pairs.push(pair(CELL_B, U3, &variant));

    pairs.push(pair(CELL_C, U1, CDNA));
    pairs.push(pair(CELL_C, U2, &format!("{}N{}", &CDNA[..30], &CDNA[31..])));
    pairs.push(pair(CELL_C, U3, &CDNA[..19]));
    pairs.push(pair(NOT_A_CELL, U4, CDNA));

    let corpus = build_corpus(&dir, &pairs, &[CELL_A, CELL_B, CELL_C]);
    scrm_corpus::verify_corpus(&corpus).expect("the corpus must verify");
    let rows = read_rows(&corpus);

    let mut cells: Vec<&str> = rows.iter().map(|(c, ..)| c.as_str()).collect();
    cells.dedup();
    let mut want_cells = vec![CELL_A, CELL_B, CELL_C];
    want_cells.sort_unstable();
    assert_eq!(cells, want_cells, "wrong cells, or not in ascending order");

    let count = |cell: &str| rows.iter().filter(|(c, ..)| c == cell).count();
    assert_eq!(count(CELL_A), 2, "A should hold U1 (with its merged neighbour) and U2");
    assert_eq!(count(CELL_B), 1, "B should hold one molecule");
    assert_eq!(count(CELL_C), 1, "C's other three pairs are all dropped");
    assert_eq!(rows.len(), 4);

    let find = |cell: &str, n_reads: u32| -> Vec<u8> {
        let hits: Vec<&(String, u32, Vec<u8>)> = rows.iter().filter(|(c, r, _)| c == cell && *r == n_reads).collect();
        assert_eq!(
            hits.len(),
            1,
            "expected one molecule in {cell} with {n_reads} reads: {hits:?}"
        );
        hits[0].2.clone()
    };

    let u1 = find(CELL_A, 4);
    assert_eq!(
        u1,
        tokens(CDNA),
        "identical-read collapse or the UMI merge changed the sequence"
    );

    let u2 = find(CELL_A, 2);
    let joined: String = format!("{CDNA}{NEW_TAIL}");
    assert_eq!(
        u2,
        tokens(&joined),
        "overlapping reads did not assemble into one contig ({} tokens, {} contigs)",
        u2.len(),
        1 + u2.iter().filter(|&&t| t == TAB).count()
    );
    assert!(!u2.contains(&TAB), "a joined molecule must be a single contig");

    let b = find(CELL_B, 4);
    assert_eq!(b, tokens(CDNA), "the 3:1 bubble did not resolve to the majority base");
    assert!(!b.contains(&(Token::N as u8)), "a 3:1 majority must not emit N");

    assert_eq!(find(CELL_C, 1), tokens(CDNA));

    let m = scrm_corpus::Manifest::read(&corpus).unwrap();
    let c = &m.measured.counts;
    assert_eq!(c.pairs_in, pairs.len() as u64);
    assert_eq!(c.dropped_n, 1, "the N-bearing read");
    assert_eq!(c.dropped_short, 1, "the 19 bp read");
    assert_eq!(c.dropped_cb, 1, "the uncorrectable barcode");
    assert_eq!(c.cells_called, 3);
    assert_eq!(c.molecules_out, 4);
    assert_eq!(c.tokens_out, rows.iter().map(|(_, _, t)| t.len() as u64).sum::<u64>());

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn read_qualities_reach_bubble_resolution_and_change_the_call() {
    const U: &str = "AAAAAAAAAAAA";
    const POS: usize = 30;
    let mut variant: Vec<u8> = CDNA.bytes().collect();
    variant[POS] = if variant[POS] == b'A' { b'C' } else { b'A' };
    let variant = String::from_utf8(variant).unwrap();

    let build = |name: &str, pairs: Vec<Pair>| -> Vec<u8> {
        let dir = tmp(name);
        let (r1, r2) = write_fastqs(&dir, &pairs);
        let bc = dir.join("barcodes.tsv");
        std::fs::write(&bc, format!("{CELL_A}-1\n")).unwrap();
        let corpus = dir.join("corpus");
        let status = Command::new(env!("CARGO_BIN_EXE_scrm-corpus"))
            .args([
                "--samplesheet",
                sheet_for(&corpus, &r1, &r2, &bc, "1{b[16]u[12]}2{r:}")
                    .to_str()
                    .unwrap(),
            ])
            .args(["--out", corpus.parent().unwrap().to_str().unwrap()])
            .args(["--threads", "2", "--min-umi-len", "1", "--min-cell-molecules", "1"])
            .status()
            .unwrap();
        assert!(status.success(), "{name} exited with {status}");
        let rows = read_rows(&corpus);
        assert_eq!(rows.len(), 1, "{name}");
        let tokens = rows[0].2.clone();
        std::fs::remove_dir_all(&dir).ok();
        tokens
    };

    let tie = build("qual-tie", vec![pair(CELL_A, U, CDNA), pair(CELL_A, U, &variant)]);
    let decided = build(
        "qual-decided",
        vec![pair(CELL_A, U, CDNA), pair_low_q_at(CELL_A, U, &variant, POS, 2)],
    );

    let n = Token::N as u8;
    assert!(
        tie.contains(&n),
        "two equally confident reads disagreeing must tie, hence N -- got {} tokens, no N",
        tie.len()
    );
    assert!(
        !decided.contains(&n),
        "a Phred 2 dissent must lose to a Phred 40 base rather than force an N"
    );
    assert_ne!(
        tie, decided,
        "identical bases, one Phred apart, produced identical tokens: quality is being dropped \
         somewhere between the FASTQ and the walk"
    );
    assert_eq!(tie.len(), decided.len(), "the call must not change the contig length");
}

#[test]
fn the_trimming_rules_reach_the_corpus() {
    const CELL_D: &str = "CACACACAGTGTGTGT";
    const CELL_E: &str = "TGTGTGTGCACACACA";
    const ILLUMINA: &str = "AGATCGGAAGAGC";
    const TSO: &str = "AAGCAGTGGTATCAACGCAGAGTACATGGG";
    const POLY_A: &str = "AAAAAAAAAAAAAAA";

    let clean = &CDNA[..40];

    let dir = tmp("trimrules");
    let pairs = vec![
        pair(CELL_A, "AAAAAAAAAAAA", &format!("{clean}{ILLUMINA}TTTTGGGG")),
        pair(CELL_B, "CCCCCCCCCCCC", &format!("{TSO}{clean}")),
        pair(CELL_C, "GGGGGGGGGGGG", &format!("{clean}{POLY_A}")),
        pair(
            CELL_D,
            "TTTTTTTTTTTT",
            &format!("{}{POLY_A}{}", &CDNA[..20], &CDNA[20..45]),
        ),
        pair(CELL_E, "ACACACACACAC", clean),
    ];

    let corpus = build_corpus(&dir, &pairs, &[CELL_A, CELL_B, CELL_C, CELL_D, CELL_E]);
    scrm_corpus::verify_corpus(&corpus).expect("must verify");
    let rows = read_rows(&corpus);
    let got = |cell: &str| -> Vec<u8> {
        let hits: Vec<&(String, u32, Vec<u8>)> = rows.iter().filter(|(c, ..)| c == cell).collect();
        assert_eq!(hits.len(), 1, "expected exactly one molecule in {cell}");
        hits[0].2.clone()
    };

    assert_eq!(
        got(CELL_A),
        tokens(clean),
        "the 3' adapter and everything after it go: read-through means the insert ended"
    );
    assert_eq!(got(CELL_B), tokens(clean), "the 5' TSO is clipped");
    assert_eq!(got(CELL_C), tokens(clean), "the anchored 3' poly-A run is trimmed");
    assert_eq!(
        got(CELL_D),
        tokens(&format!("{}{POLY_A}{}", &CDNA[..20], &CDNA[20..45])),
        "an INTERNAL run was never cut and still is not"
    );
    assert_eq!(got(CELL_E), tokens(clean), "a clean read survives untouched");

    let m = scrm_corpus::Manifest::read(&corpus).unwrap();
    let a = &m.measured.artifact_rates;
    assert!(
        a.adapter_fraction > 0.0,
        "adapter_fraction must be NONZERO: one fixture read carries a read-through adapter, so a \
         zero rate would mean the adapter scan stopped running"
    );
    assert!(
        a.tso_fraction > 0.0,
        "the TSO read must still be COUNTED even though it is not cut"
    );
    let internal: f64 = a
        .homopolymer
        .iter()
        .filter(|(key, _)| key.contains("internal"))
        .map(|(_, v)| *v)
        .sum();
    assert!(
        internal > 0.0,
        "the internal run must be MEASURED even though it is not cut"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_length_filter_defines_which_molecules_exist() {
    const CELL_A: &str = "AAAACCCCGGGGTTTT";
    const CELL_B: &str = "CCCCGGGGTTTTAAAA";
    const CELL_C: &str = "GGGGTTTTAAAACCCC";
    const CELL_D: &str = "TTTTAAAACCCCGGGG";

    let short = &CDNA[..30];
    let keep = &CDNA[..45];
    let long = CDNA;

    let dir = tmp("lenfilter");
    let pairs = vec![
        pair(CELL_A, "AAAAAAAAAAAA", short),
        pair(CELL_B, "CCCCCCCCCCCC", keep),
        pair(CELL_C, "GGGGGGGGGGGG", long),
        pair(CELL_D, "TTTTTTTTTTTT", keep),
        pair(CELL_D, "TTTTTTTTTTTA", long),
    ];

    let corpus = build_corpus_with(
        &dir,
        &pairs,
        &[CELL_A, CELL_B, CELL_C, CELL_D],
        &[
            "--min-umi-len",
            "40",
            "--max-umi-len",
            "50",
            "--min-cell-molecules",
            "1",
        ],
    );
    scrm_corpus::verify_corpus(&corpus).expect("a filtered corpus must still verify");

    let rows = read_rows(&corpus);
    let cells: Vec<&str> = rows.iter().map(|(c, ..)| c.as_str()).collect();
    assert_eq!(
        cells,
        vec![CELL_B, CELL_D],
        "only the admissible molecules survive, and a cell whose every molecule was filtered \
         must be ABSENT rather than present with zero rows -- the format has no encoding for \
         the latter and CorpusReader::open rejects it"
    );
    for (cell, _n_reads, tokens) in &rows {
        assert_eq!(
            tokens.len(),
            45,
            "cell {cell} kept a molecule of {} tokens, outside [40, 50]",
            tokens.len()
        );
    }

    let m = scrm_corpus::Manifest::read(&corpus).unwrap();
    assert_eq!((m.declared.config.min_umi_len, m.declared.config.max_umi_len), (40, 50));

    let c = &m.measured.counts;
    assert_eq!((c.dropped_umi_short, c.dropped_umi_short_tokens), (1, 30));
    assert_eq!((c.dropped_umi_long, c.dropped_umi_long_tokens), (2, 120));
    assert_eq!(c.molecules_out, 2);
    assert_eq!(
        c.molecules_out + c.dropped_umi_short + c.dropped_umi_long,
        5,
        "every assembled molecule must be either written or counted as dropped"
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_build_into_a_finished_corpus_is_refused_before_anything_is_read() {
    let dir = tmp("occupied");
    let pairs = vec![pair(CELL_A, "AAAAAAAAAAAA", CDNA), pair(CELL_A, "CCCCCCCCCCCC", CDNA)];
    let corpus = build_corpus(&dir, &pairs, &[CELL_A]);
    let before = std::fs::read(corpus.join("manifest.json")).unwrap();

    let out = try_build(
        &dir,
        &corpus,
        &pairs,
        &[CELL_A],
        &["--threads", "2"],
        &["--min-umi-len", "1", "--min-cell-molecules", "1"],
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "the second build succeeded\n--- stderr ---\n{err}"
    );
    assert!(
        err.contains("already holds a corpus"),
        "refused for another reason:\n{err}"
    );
    assert_eq!(
        std::fs::read(corpus.join("manifest.json")).unwrap(),
        before,
        "the refused build still changed the corpus"
    );
}

#[test]
fn the_corpus_is_a_pure_function_of_its_input_not_of_thread_count() {
    let dir = tmp("determinism");
    fn acgt(mut i: usize, len: usize) -> String {
        let mut s = vec![b'A'; len];
        let mut at = len;
        while at > 0 {
            at -= 1;
            s[at] = b"ACGT"[i % 4];
            i /= 4;
        }
        String::from_utf8(s).unwrap()
    }

    let mut pairs = Vec::new();
    let cells: Vec<String> = (0..12).map(|i| acgt(i, 16)).collect();
    for (i, cell) in cells.iter().enumerate() {
        let len = 40 + (i % 8);
        let plain = CDNA[..len].to_string();
        let dissent = with_sub_at(&plain, 30);
        for (seq, q) in [(&plain, 22u8), (&plain, 33), (&dissent, 22), (&dissent, 33)] {
            pairs.push(pair_low_q_at(cell, &acgt(i, 12), seq, 30, q));
        }
        pairs.push(pair(cell, &acgt(1000 + i, 12), &CDNA[..45]));
    }
    let cell_refs: Vec<&str> = cells.iter().map(String::as_str).collect();

    let reversed: Vec<Pair> = pairs.iter().rev().cloned().collect();

    let mut digests = Vec::new();
    let mut stamped = Vec::new();
    for (tag, threads, input) in [
        ("t1", "1", &pairs),
        ("t4", "4", &pairs),
        ("t4again", "4", &pairs),
        ("reversed", "4", &reversed),
    ] {
        let out = dir.join(format!("corpus-{tag}"));
        let o = try_build(
            &dir,
            &out,
            input,
            &cell_refs,
            &["--threads", threads],
            &["--min-umi-len", "1", "--min-cell-molecules", "1"],
        );
        assert!(
            o.status.success(),
            "build at threads={threads} failed\n{}",
            String::from_utf8_lossy(&o.stderr)
        );
        scrm_corpus::verify_corpus(&out).expect("must verify");
        digests.push((tag, scrm_corpus::stream_digest(&out).expect("stream digest")));
        let a = scrm_corpus::Manifest::read(&out).unwrap().measured.assembly_stats;
        stamped.push((
            tag,
            (
                a.bubble_n_tie,
                a.bubble_n_low_confidence,
                a.origin_k_total,
                a.branch_decisions,
                a.origin_lambda_total.to_bits(),
            ),
        ));
    }

    assert_eq!(
        digests[0].1, digests[1].1,
        "thread count changed the corpus: 1 thread gave {} and 4 gave {}. Thread count is a \
         throughput parameter, so this is a race in collate or the shard reduction, not a \
         tuning difference",
        digests[0].1, digests[1].1
    );
    assert_eq!(
        digests[1].1, digests[2].1,
        "two runs at the same thread count disagreed -- the pipeline is nondeterministic"
    );
    assert_eq!(
        digests[0].1, digests[3].1,
        "presenting the same reads in the opposite order changed the corpus"
    );

    let (tie, low, k_total, decisions, lambda) = stamped[0].1;
    assert!(
        decisions > 0 && k_total > 0 && lambda > 0,
        "the fixture must actually reach a branch decision with a disagreeing column: \
         tie={tie} low={low} k={k_total} decisions={decisions}"
    );
    for w in stamped.windows(2) {
        assert_eq!(
            w[0].1, w[1].1,
            "the stamped consensus numbers are not a pure function of the input: {} gave {:?} and \
             {} gave {:?} (bubble_n_tie, bubble_n_low_confidence, origin_k_total, \
             branch_decisions, origin_lambda_total bits)",
            w[0].0, w[0].1, w[1].0, w[1].1
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_stamped_tool_versions_are_read_from_the_lockfile_not_typed() {
    let dir = tmp("provenance");
    let pairs = vec![pair("ACGTACGTACGTACGT", "AAAAAAAAAAAA", &CDNA[..40])];
    let out = build_corpus(&dir, &pairs, &["ACGTACGTACGTACGT"]);
    let m = scrm_corpus::Manifest::read(&out).unwrap();

    let rustc = &m.declared.tools.rustc;
    assert!(
        rustc.starts_with("rustc "),
        "the rustc field must name the COMPILER, got {rustc:?}"
    );
    assert_ne!(
        rustc,
        env!("CARGO_PKG_VERSION"),
        "the rustc field is the crate version again"
    );
    let built = &m.declared.build.built_at_utc;
    assert!(
        built.starts_with("20") && built.ends_with('Z') && built.len() == 20,
        "built_at_utc must be an ISO-8601 UTC instant, got {built:?}"
    );

    let lock = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock"))
        .expect("workspace Cargo.lock");
    let t = &m.declared.tools;
    for (lock_name, stamped) in [
        ("paraseq-temp", &t.paraseq),
        ("rapidgzip-core", &t.rapidgzip_core),
        ("libradicl", &t.libradicl),
        ("seq_geom_parser", &t.seq_geom_parser),
        ("debruijn", &t.debruijn),
        ("arrow", &t.arrow),
    ] {
        let (version, git) = match stamped.split_once(" (git ") {
            Some((v, g)) => (v, Some(g.trim_end_matches(')'))),
            None => (stamped.as_str(), None),
        };
        assert!(!version.is_empty(), "{lock_name} has no stamped version");
        assert!(
            lock.contains(&format!("name = \"{lock_name}\"\nversion = \"{version}\"")),
            "the stamp says {lock_name} {version}, which is not what Cargo.lock locks"
        );
        if let Some(rev) = git {
            assert!(
                lock.contains(rev),
                "{lock_name}'s stamped git rev {rev} is not in Cargo.lock"
            );
        }
    }
    assert!(
        t.libradicl.contains("(git "),
        "libradicl is pinned to a revision in Cargo.toml, so the stamp must name it: {:?}",
        t.libradicl
    );

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_molecule_dropped_by_the_length_filter_is_still_counted_as_assembled() {
    let dir = tmp("filtered-still-counted");
    let run = format!("{}{}{}", &CDNA[..25], "A".repeat(20), &CDNA[35..]);
    let mut pairs = vec![
        pair("ACGTACGTACGTACGT", "AAAAAAAAAAAA", &run),
        pair("ACGTACGTACGTACGT", "AAAAAAAAAAAA", &run[10..]),
    ];
    let survivors = ["CCGTACGTACGTACGT", "GCGTACGTACGTACGT"];
    let long = CDNA.repeat(4);
    for (i, cell) in survivors.iter().enumerate() {
        pairs.push(pair(
            cell,
            &format!("CCCCCCCCCCC{}", "ACGT".as_bytes()[i] as char),
            &long,
        ));
    }
    let cells = ["ACGTACGTACGTACGT", survivors[0], survivors[1]];

    let out = build_corpus_with(
        &dir,
        &pairs,
        &cells,
        &["--min-umi-len", "200", "--min-cell-molecules", "1"],
    );
    let m = scrm_corpus::Manifest::read(&out).unwrap();
    assert_eq!(
        m.measured.counts.dropped_umi_short, 1,
        "the fixture must actually be removed by the floor"
    );
    assert_eq!(
        m.measured.assembly_stats.self_loop_molecules, 1,
        "the homopolymer discard must stay a stamped measurement for a molecule the floor removed"
    );
    assert_eq!(m.measured.counts.molecules_out, 2, "and only the survivors are written");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn unusable_bounds_are_refused_before_any_work() {
    let dir = tmp("badbounds");
    let pairs = vec![pair("ACGTACGTACGTACGT", "AAAAAAAAAAAA", &CDNA[..40])];
    let cells = ["ACGTACGTACGTACGT"];

    for (label, args) in [
        ("a zero floor", vec!["--min-umi-len", "0", "--min-cell-molecules", "1"]),
        (
            "an inverted range",
            vec![
                "--min-umi-len",
                "1600",
                "--max-umi-len",
                "60",
                "--min-cell-molecules",
                "1",
            ],
        ),
    ] {
        let out = dir.join("corpus-bad");
        let o = try_build(&dir, &out, &pairs, &cells, &["--threads", "2"], &args);
        assert!(!o.status.success(), "{label} was accepted");
        let stderr = String::from_utf8_lossy(&o.stderr);
        assert!(
            stderr.contains("molecule length bounds"),
            "{label} failed without naming the bounds, so the user must guess. stderr was:\n{stderr}"
        );
        assert!(!out.exists(), "{label} was rejected but left {} behind", out.display());
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_build_with_no_admissible_molecules_fails_naming_the_filter() {
    let dir = tmp("allfiltered");
    let pairs = vec![
        pair("ACGTACGTACGTACGT", "AAAAAAAAAAAA", &CDNA[..40]),
        pair("TTTTGGGGCCCCAAAA", "CCCCCCCCCCCC", &CDNA[..45]),
    ];
    let cells = ["ACGTACGTACGTACGT", "TTTTGGGGCCCCAAAA"];
    let out = dir.join("corpus-empty");
    let o = try_build(
        &dir,
        &out,
        &pairs,
        &cells,
        &["--threads", "2"],
        &[
            "--min-umi-len",
            "10000",
            "--max-umi-len",
            "20000",
            "--min-cell-molecules",
            "1",
        ],
    );
    let stderr = String::from_utf8_lossy(&o.stderr);
    assert!(
        !o.status.success(),
        "a corpus with no molecules was written as if it were fine\n{stderr}"
    );
    assert!(
        stderr.contains("molecule bounds") && stderr.contains("10000"),
        "the failure must name the molecule bounds as the cause. stderr was:\n{stderr}"
    );
    assert!(
        !stderr.contains("min-cell-molecules"),
        "the cell floor is 1 here and is NOT the cause; blaming it misattributes a molecule-bound \
         failure to a cell filter. stderr was:\n{stderr}"
    );
    assert!(
        !out.join("manifest.json").exists(),
        "a build that failed published a manifest, so a directory of shards it never finished now \
         reads as a finished corpus to every launcher and to the pin gate"
    );
    if out.exists() {
        scrm_corpus::refuse_if_corpus(&out).expect("a manifest-less shard directory is a build target");
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn output_files_are_prefixed_with_the_experiment_name_from_the_samplesheet() {
    const EXPERIMENT: &str = "sod1_spinal_cord_specimen_1";
    const MISLEADING_FASTQ_SAMPLE: &str = "20k_Mouse_PBMC_Splenocyte_5p_gem-x_multiplex_GEX";
    let dir = tmp("naming");
    let pairs = vec![
        pair(CELL_A, "AAAAAAAAAAAA", &CDNA[..40]),
        pair(CELL_B, "CCCCCCCCCCCC", &CDNA[..45]),
    ];
    let (r1, r2) = write_fastqs_10x(&dir, &pairs, MISLEADING_FASTQ_SAMPLE);
    let bc = dir.join("barcodes.tsv");
    std::fs::write(&bc, format!("{CELL_A}-1\n{CELL_B}-1\n")).unwrap();
    let root = dir.join("root");
    let out = root.join(EXPERIMENT);
    let sheet = sheet_for(&out, &r1, &r2, &bc, "1{b[16]u[12]}2{r:}");

    let status = Command::new(env!("CARGO_BIN_EXE_scrm-corpus"))
        .args(["--samplesheet", sheet.to_str().unwrap()])
        .args(["--out", root.to_str().unwrap()])
        .args(["--threads", "2", "--min-umi-len", "1", "--min-cell-molecules", "1"])
        .status()
        .expect("failed to run scrm-corpus");
    assert!(status.success(), "the pipeline exited with {status}");
    scrm_corpus::verify_corpus(&out).expect("must verify");

    let mut shards: Vec<String> = std::fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".arrow"))
        .collect();
    shards.sort();
    assert!(!shards.is_empty(), "no shards were written");
    for name in &shards {
        assert!(
            name.starts_with(EXPERIMENT) && name.contains("-shard-"),
            "shard {name:?} does not carry the experiment name {EXPERIMENT:?}"
        );
        assert!(
            !name.contains("PBMC") && !name.contains("_S7_") && !name.contains("_L001_"),
            "shard {name:?} took its name from the FASTQ instead of the samplesheet"
        );
    }

    let m = scrm_corpus::Manifest::read(&out).unwrap();
    assert!(
        out.join("manifest.json").exists(),
        "the entry point must keep its fixed name"
    );
    for entry in &m.shards {
        assert!(
            entry.file.starts_with(EXPERIMENT),
            "manifest records {:?}, which is not prefixed with the experiment name",
            entry.file
        );
        assert!(
            out.join(&entry.file).exists(),
            "manifest names {:?} but no such file exists -- the two writes disagree",
            entry.file
        );
    }

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_cell_floor_decides_which_cells_exist() {
    const DEEP: &str = "ACGTACGTACGTACGT";
    const SHALLOW: &str = "TTTTGGGGCCCCAAAA";
    let dir = tmp("cellfloor");

    let mut pairs = Vec::new();
    for i in 0..4 {
        pairs.push(pair(
            DEEP,
            &format!("AAAAAAAAAAA{}", "ACGT".as_bytes()[i] as char),
            &CDNA[..40],
        ));
    }
    for i in 0..2 {
        pairs.push(pair(
            SHALLOW,
            &format!("CCCCCCCCCCC{}", "ACGT".as_bytes()[i] as char),
            &CDNA[..45],
        ));
    }

    let out = dir.join("corpus");
    let o = try_build(
        &dir,
        &out,
        &pairs,
        &[DEEP, SHALLOW],
        &["--threads", "2"],
        &["--min-umi-len", "1", "--min-cell-molecules", "3"],
    );
    assert!(
        o.status.success(),
        "build failed\n{}",
        String::from_utf8_lossy(&o.stderr)
    );
    scrm_corpus::verify_corpus(&out).expect("a cell-filtered corpus must still verify");

    let rows = read_rows(&out);
    let cells: Vec<&str> = rows.iter().map(|(c, ..)| c.as_str()).collect();
    assert!(
        cells.iter().all(|c| *c == DEEP),
        "the shallow cell must be absent, not merely small: got {cells:?}"
    );
    assert_eq!(rows.len(), 4, "the deep cell keeps all four molecules");

    let m = scrm_corpus::Manifest::read(&out).unwrap();
    assert_eq!(m.declared.config.min_cell_molecules, 3, "the floor is stamped");
    let c = &m.measured.counts;
    assert_eq!(
        (
            c.dropped_cells_shallow,
            c.dropped_cells_molecules,
            c.dropped_cells_emptied
        ),
        (1, 2, 0),
        "one cell dropped holding two molecules, and none emptied upstream -- a silent cell filter \
         is how a corpus ends up with fewer cells than cells_called claims, and `emptied` is the \
         second way a cell can vanish, which went uncounted until the books were closed"
    );
    assert_eq!(
        c.cells_called - c.dropped_cells_shallow - c.dropped_cells_emptied,
        1,
        "cells_called minus both drop paths must equal the cells actually written"
    );
    assert_eq!(
        c.cells_called, 2,
        "cells_called counts called barcodes, not written cells"
    );
    assert_eq!(c.molecules_out, 4);

    std::fs::remove_dir_all(&dir).ok();
}
