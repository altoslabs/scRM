use std::path::{Path, PathBuf};

use scrm_core::record::Umi;
use scrm_core::{CellId, MoleculeContigs, Token};
use scrm_corpus::{CorpusWriter, Manifest, ShardEntry};
use scrm_pack::{BatchSpec, Corpus, PackError, build_batch, derive_seed};

const TAB: u8 = Token::Tab as u8;

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("scrm-pack-mix-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn declared(max_umi_len: usize) -> scrm_core::stamp::Declared {
    use scrm_core::stamp::*;
    Declared {
        build: Build {
            scrm_sha: "abc1234".into(),
            dirty: false,
            built_at_utc: "2026-09-03T00:00:00Z".into(),
        },
        config: Config {
            geometry: "1{b[16]u[12]}2{r:}".into(),
            cb_len: 16,
            umi_len: 12,
            whitelist_sha256: "0".repeat(64),
            alphabet_size: scrm_core::ALPHABET_SIZE,
            dedup_rule: "1MM_CR_flow".into(),
            k: scrm_core::K,
            min_trim_len: scrm_core::MIN_TRIM_LEN,
            trim: serde_json::json!({}),
            read_policy: "R1: CB+UMI only, byte-identical".into(),
            bubble_resolution: "likelihood-q10-pre45-post40-omega7".into(),
            max_windows_per_cell: 0,
            min_umi_len: 1,
            max_umi_len,
            min_cell_molecules: 1,
        },
        inputs: vec![],
        tools: Tools {
            rustc: "1.95.0".into(),
            paraseq: "0.5.0-pre.2".into(),
            rapidgzip_core: "0.3.1".into(),
            thread_broker: "unused".into(),
            libradicl: "0.17.0".into(),
            seq_geom_parser: "1.1.0".into(),
            debruijn: "0.3.4".into(),
            arrow: "56".into(),
        },
    }
}

fn measured() -> scrm_core::stamp::Measured {
    use scrm_core::stamp::*;
    Measured {
        counts: Counts {
            pairs_in: 10,
            dropped_n: 0,
            dropped_short: 0,
            dropped_cb: 0,
            cells_called: 2,
            dropped_umi_short: 0,
            dropped_umi_short_tokens: 0,
            dropped_umi_long: 0,
            dropped_umi_long_tokens: 0,
            dropped_cells_shallow: 0,
            dropped_cells_molecules: 0,
            dropped_cells_emptied: 0,
            molecules_out: 0,
            tokens_out: 0,
        },
        umi_stats: UmiStats {
            mean_hamming1_degree: 0.05,
            singleton_fraction: 0.9,
            merge_rate: 0.01,
            max_group_fraction: 0.0006,
            max_group_fraction_umis: 3300,
            max_group_umis: 2,
            max_chain_depth: 1,
        },
        assembly_stats: AssemblyStats {
            reads_per_molecule_hist: vec![],
            distinct_seqs_per_molecule_hist: vec![],
            pass_through_fraction: 0.83,
            contigs_per_molecule_hist: vec![],
            contig_length_hist: vec![],
            self_loop_molecules: 0,
            bubble_n_tokens: 0,
            bubble_n_tie: 0,
            bubble_n_low_confidence: 0,
            bubble_n_indel_interior: 0,
            ambiguous_read_placements: 0,
            branch_decisions: 0,
            origin_splits: 0,
            origin_surprise_hist: vec![],
            called_q_hist: vec![],
            origin_k_total: 9,
            origin_lambda_total: 0.5,
        },
        artifact_rates: ArtifactRates {
            adapter_fraction: 0.01,
            tso_fraction: 0.02,
            homopolymer: Default::default(),
        },
        throughput: Throughput {
            read_pairs_per_sec: 1.0,
            molecules_per_sec: 1.0,
            secs_stream: 1.0,
            secs_assemble_write: 1.0,
            n_threads: 4,
        },
    }
}

fn mol(cell: u64, umi: u64, tokens: Vec<u8>) -> MoleculeContigs {
    let mut m = MoleculeContigs {
        cell: CellId(cell),
        umi: Umi(umi),
        n_reads: 1,
        tokens,
        n_contigs: 0,
    };
    m.n_contigs = m.recount_contigs() as u16;
    m
}

fn corpus_declaring(root: &Path, name: &str, cells: &[(&str, u64)], max_umi_len: usize) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let decl = declared(max_umi_len);
    let file = "shard-000.arrow";
    let mut w = CorpusWriter::create(&dir.join(file), &decl).unwrap();
    for (cell, packed) in cells {
        w.push(cell, &mol(*packed, 0, vec![1, TAB, 2, 3])).unwrap();
    }
    let s = w.finish().unwrap();
    let entry = ShardEntry {
        file: file.into(),
        n_rows: s.n_rows,
        n_tokens: s.n_tokens,
        tokens_sha256: s.tokens_sha256,
        first_cell: s.first_cell,
        last_cell: s.last_cell,
    };
    Manifest::new(decl, measured(), vec![entry]).write(&dir).unwrap();
    dir
}

#[test]
fn two_corpora_under_one_molecule_definition_open_as_one_population() {
    let root = tmp_dir("agreeing");
    let a = corpus_declaring(&root, "a", &[("CELL0001", 1)], 1600);
    let b = corpus_declaring(&root, "b", &[("CELL0002", 2)], 1600);

    let corpus = Corpus::open_many(&[a.as_path(), b.as_path()]).expect("one definition must open");
    let names: Vec<&str> = corpus.cells().iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["CELL0001", "CELL0002"]);
    assert_eq!(corpus.compat_key().max_umi_len, 1600);
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn two_corpora_declaring_different_molecule_definitions_are_refused() {
    let root = tmp_dir("disagreeing");
    let a = corpus_declaring(&root, "a", &[("CELL0001", 1)], 1600);
    let b = corpus_declaring(&root, "b", &[("CELL0002", 2)], 2400);

    match Corpus::open_many(&[a.as_path(), b.as_path()]) {
        Err(PackError::CompatKeyMismatch { a: sa, b: sb }) => {
            assert!(sa.contains("1600") && sa.contains(a.to_str().unwrap()), "{sa}");
            assert!(sb.contains("2400") && sb.contains(b.to_str().unwrap()), "{sb}");
        }
        Err(other) => panic!("refused for the wrong reason: {other:?}"),
        Ok(c) => panic!(
            "two molecule definitions opened as one corpus of {} cells, compat_key {:?}",
            c.cells().len(),
            c.compat_key()
        ),
    }

    assert!(Corpus::open_many(&[b.as_path(), a.as_path()]).is_err());
    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn an_empty_batch_spec_is_refused_without_real_data() {
    let root = tmp_dir("empty-spec");
    let dir = corpus_declaring(&root, "a", &[("CELL0001", 1)], 1600);
    let corpus = Corpus::open(&dir).expect("the fixture corpus must open");

    let spec = |cells: Vec<u32>, v: usize| BatchSpec {
        cells,
        v,
        seq_len: 16384,
        epoch_seed: 1,
        frac: 1.0,
        frac_seed: 0,
    };
    for (label, s) in [("no cells", spec(vec![], 2)), ("v = 0", spec(vec![0], 0))] {
        match build_batch(&corpus, &s) {
            Err(PackError::Corpus(msg)) => assert!(
                msg.contains("at least one cell and one view"),
                "{label} refused for another reason: {msg}"
            ),
            Err(other) => panic!("{label}: wrong error variant: {other:?}"),
            Ok(b) => panic!("{label}: accepted, producing {} windows", b.n_windows),
        }
    }

    let ok = build_batch(&corpus, &spec(vec![0], 1)).expect("a well-formed spec must be accepted");
    assert_eq!(ok.n_windows, 1);
    assert_eq!(ok.cell_index, vec![0]);

    assert_eq!(derive_seed(1, 2, 3), derive_seed(1, 2, 3));
    assert_ne!(derive_seed(1, 2, 3), derive_seed(1, 2, 4));

    std::fs::remove_dir_all(&root).ok();
}
