use std::path::{Path, PathBuf};

use scrm_core::record::Umi;
use scrm_core::{CellId, MoleculeContigs, Token};
use scrm_corpus::{CorpusWriter, Manifest, ShardEntry, stream_digest, verify_corpus};

const TAB: u8 = Token::Tab as u8;

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("scrm-manifest-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn declared() -> scrm_core::stamp::Declared {
    use scrm_core::stamp::*;
    Declared {
        build: Build {
            scrm_sha: "abc1234".into(),
            dirty: false,
            built_at_utc: "2026-08-15T00:00:00Z".into(),
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
            bubble_resolution: "weighted".into(),
            max_windows_per_cell: 0,
            min_umi_len: scrm_core::MIN_UMI_LEN_DEFAULT,
            max_umi_len: scrm_core::MAX_UMI_LEN_DEFAULT,
            min_cell_molecules: scrm_core::MIN_CELL_MOLECULES_DEFAULT,
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
            cells_called: 4,
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

fn packed(cell_id: &str) -> u64 {
    cell_id
        .trim_start_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .expect("fixture cell ids end in digits")
}

fn write_shard(dir: &Path, k: usize, cells: &[(&str, usize)]) -> ShardEntry {
    write_shard_declaring(dir, k, cells, &declared())
}

fn write_shard_declaring(
    dir: &Path,
    k: usize,
    cells: &[(&str, usize)],
    declared: &scrm_core::stamp::Declared,
) -> ShardEntry {
    let file = format!("shard-{k:03}.arrow");
    let mut w = CorpusWriter::create(&dir.join(&file), declared).unwrap();
    for (c, n) in cells {
        for u in 0..*n {
            w.push(c, &mol(packed(c), u as u64, vec![1, TAB, 2, 3])).unwrap();
        }
    }
    let s = w.finish().unwrap();
    ShardEntry {
        file,
        n_rows: s.n_rows,
        n_tokens: s.n_tokens,
        tokens_sha256: s.tokens_sha256,
        first_cell: s.first_cell,
        last_cell: s.last_cell,
    }
}

#[test]
fn an_ordered_set_of_shards_verifies() {
    let dir = tmp_dir("ok");
    let shards = vec![
        write_shard(&dir, 0, &[("CELL0001", 2), ("CELL0002", 1)]),
        write_shard(&dir, 1, &[("CELL0003", 3)]),
        write_shard(&dir, 2, &[("CELL0004", 1), ("CELL0005", 2)]),
    ];
    Manifest::new(declared(), measured(), shards).write(&dir).unwrap();

    let d = verify_corpus(&dir).unwrap();
    assert_eq!(d.n_rows, 9);
    assert_eq!(d.n_tokens, 9 * 4);
    assert_eq!(d.n_shards, 3);
    assert_ne!(d.root, stream_digest(&dir).unwrap());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_cell_spanning_two_shards_is_rejected() {
    let dir = tmp_dir("straddle");
    let shards = vec![
        write_shard(&dir, 0, &[("CELL0001", 2), ("CELL0002", 1)]),
        write_shard(&dir, 1, &[("CELL0002", 1), ("CELL0003", 1)]),
    ];
    Manifest::new(declared(), measured(), shards).write(&dir).unwrap();

    let err = verify_corpus(&dir).unwrap_err().to_string();
    assert!(err.contains("CELL0002"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn shards_listed_out_of_cell_order_are_rejected() {
    let dir = tmp_dir("misordered");
    let a = write_shard(&dir, 0, &[("CELL0001", 1)]);
    let b = write_shard(&dir, 1, &[("CELL0009", 1)]);
    Manifest::new(declared(), measured(), vec![b, a]).write(&dir).unwrap();

    let err = verify_corpus(&dir).unwrap_err().to_string();
    assert!(err.contains("CELL0001"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_manifest_tally_that_disagrees_with_its_shard_is_rejected() {
    let dir = tmp_dir("tally");
    let mut shard = write_shard(&dir, 0, &[("CELL0001", 2)]);
    shard.n_rows += 1;
    Manifest::new(declared(), measured(), vec![shard]).write(&dir).unwrap();

    let err = verify_corpus(&dir).unwrap_err().to_string();
    assert!(err.contains("rows"), "{err}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_tampered_shard_digest_is_rejected() {
    let dir = tmp_dir("digest");
    let mut shard = write_shard(&dir, 0, &[("CELL0001", 2)]);
    shard.tokens_sha256 = "0".repeat(64);
    Manifest::new(declared(), measured(), vec![shard]).write(&dir).unwrap();

    assert!(verify_corpus(&dir).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_root_changes_when_shards_are_reordered() {
    let dir = tmp_dir("root");
    let a = write_shard(&dir, 0, &[("CELL0001", 1)]);
    let b = write_shard(&dir, 1, &[("CELL0009", 1)]);
    let forward = Manifest::new(declared(), measured(), vec![a.clone(), b.clone()]);
    let backward = Manifest::new(declared(), measured(), vec![b, a]);
    assert_ne!(forward.tokens_sha256_root, backward.tokens_sha256_root);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn the_stream_digest_survives_re_sharding() {
    let cells: [(&str, usize); 4] = [("CELL0001", 2), ("CELL0002", 1), ("CELL0003", 2), ("CELL0004", 1)];

    let one = tmp_dir("shard-1");
    let s = write_shard(&one, 0, &cells);
    Manifest::new(declared(), measured(), vec![s]).write(&one).unwrap();
    let d1 = verify_corpus(&one).unwrap();
    let s1 = stream_digest(&one).unwrap();

    let four = tmp_dir("shard-4");
    let shards: Vec<ShardEntry> = cells
        .iter()
        .enumerate()
        .map(|(k, c)| write_shard(&four, k, std::slice::from_ref(c)))
        .collect();
    Manifest::new(declared(), measured(), shards).write(&four).unwrap();
    let d4 = verify_corpus(&four).unwrap();
    let s4 = stream_digest(&four).unwrap();

    assert_eq!(s1, s4, "the same rows hashed differently");
    assert_eq!(d1.n_rows, d4.n_rows);
    assert_ne!(d1.root, d4.root, "different shardings should have different roots");

    std::fs::remove_dir_all(&one).ok();
    std::fs::remove_dir_all(&four).ok();
}

#[test]
fn a_manifest_with_no_shards_is_rejected() {
    let dir = tmp_dir("empty");
    Manifest::new(declared(), measured(), vec![]).write(&dir).unwrap();
    assert!(verify_corpus(&dir).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_missing_shard_file_is_rejected() {
    let dir = tmp_dir("missing");
    let shard = write_shard(&dir, 0, &[("CELL0001", 1)]);
    std::fs::remove_file(dir.join(&shard.file)).unwrap();
    Manifest::new(declared(), measured(), vec![shard]).write(&dir).unwrap();
    assert!(verify_corpus(&dir).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_shard_declaring_another_molecule_definition_is_rejected() {
    let dir = tmp_dir("mixed-definitions");
    let ours = write_shard(&dir, 0, &[("CELL0001", 2)]);

    let mut other = declared();
    other.config.bubble_resolution = "weighted_split".into();
    let theirs = write_shard_declaring(&dir, 1, &[("CELL0002", 2)], &other);

    Manifest::new(declared(), measured(), vec![ours.clone(), theirs])
        .write(&dir)
        .unwrap();
    let err = verify_corpus(&dir).expect_err("a corpus holding two molecule definitions must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("weighted_split") && msg.contains("two definitions"),
        "the refusal does not name the disagreement: {msg}"
    );

    std::fs::remove_file(dir.join("shard-001.arrow")).unwrap();
    let second = write_shard(&dir, 1, &[("CELL0002", 2)]);
    Manifest::new(declared(), measured(), vec![ours, second])
        .write(&dir)
        .unwrap();
    verify_corpus(&dir).expect("one definition throughout must verify");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn a_directory_holding_a_manifest_is_refused_as_a_build_target_and_one_without_is_not() {
    let dir = tmp_dir("occupied");
    let shard = write_shard(&dir, 0, &[("CELL0001", 1)]);
    scrm_corpus::refuse_if_corpus(&dir).expect("shards with no manifest are an unfinished build");

    Manifest::new(declared(), measured(), vec![shard]).write(&dir).unwrap();
    let err = scrm_corpus::refuse_if_corpus(&dir).expect_err("a finished corpus must be refused");
    assert!(
        err.to_string().contains("already holds a corpus"),
        "unhelpful refusal: {err}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn publishing_a_manifest_never_shows_a_reader_a_partial_one() {
    use std::io::Read;

    let dir = tmp_dir("atomic-manifest");
    let shard = write_shard(&dir, 0, &[("CELL0001", 1)]);

    let first = Manifest::new(declared(), measured(), vec![shard.clone()]);
    first.write(&dir).unwrap();
    let before = std::fs::read_to_string(dir.join("manifest.json")).unwrap();
    let mut open_before = std::fs::File::open(dir.join("manifest.json")).unwrap();

    let mut second = Manifest::new(declared(), measured(), vec![shard]);
    second.declared.build.scrm_sha = "z".repeat(4096);
    second.write(&dir).unwrap();

    let mut through_old_handle = String::new();
    open_before.read_to_string(&mut through_old_handle).unwrap();
    assert_eq!(
        through_old_handle, before,
        "the manifest was rewritten in place: a reader holding it open saw the publish happen, so a \
         killed publish would leave a manifest no reader can parse"
    );

    let back = Manifest::read(&dir).unwrap();
    assert_eq!(back.declared.build.scrm_sha.len(), 4096);
    let leftovers: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains("tmp"))
        .collect();
    assert!(leftovers.is_empty(), "staging files left behind: {leftovers:?}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn permute_corpus_moves_every_payload_out_of_its_cell_and_keeps_depth() {
    use scrm_corpus::{CorpusReader, permute_corpus};
    let src = tmp_dir("perm-src");
    let cells: &[(&str, usize)] = &[("AAAA", 3), ("AAAC", 1), ("AAAG", 4), ("AAAT", 2), ("AACA", 2)];
    let file = "shard-000.arrow".to_string();
    let mut w = CorpusWriter::create(&src.join(&file), &declared()).unwrap();
    let mut row = 0u64;
    for (c, n) in cells {
        let packed = scrm_core::pack_acgt(c.as_bytes()).unwrap();
        for _ in 0..*n {
            let tokens: Vec<u8> = (0..6).map(|d| ((row >> (2 * d)) & 3) as u8).collect();
            w.push(c, &mol(packed, row, tokens)).unwrap();
            row += 1;
        }
    }
    let s = w.finish().unwrap();
    let entry = ShardEntry {
        file: file.clone(),
        n_rows: s.n_rows,
        n_tokens: s.n_tokens,
        tokens_sha256: s.tokens_sha256,
        first_cell: s.first_cell,
        last_cell: s.last_cell,
    };
    Manifest::new(declared(), measured(), vec![entry]).write(&src).unwrap();

    let dst = tmp_dir("perm-dst");
    permute_corpus(&src, &dst, 20260901).unwrap();

    let cell_of_row: Vec<&str> = cells.iter().flat_map(|(c, n)| std::iter::repeat_n(*c, *n)).collect();
    let mut reader = CorpusReader::open(&dst.join(&file)).unwrap();
    let mut j = 0usize;
    let mut seen = vec![false; cell_of_row.len()];
    while let Some(batch) = reader.next_batch() {
        let batch = batch.unwrap();
        for i in 0..batch.len() {
            let r = batch.row(i);
            assert_eq!(r.cell_id, cell_of_row[j], "the cell column must be untouched");
            let origin = r
                .tokens
                .iter()
                .enumerate()
                .fold(0usize, |acc, (d, &t)| acc | ((t as usize) << (2 * d)));
            assert!(!seen[origin], "payload {origin} appears twice");
            seen[origin] = true;
            assert_ne!(
                cell_of_row[origin], cell_of_row[j],
                "row {j} kept its own cell's molecule"
            );
            j += 1;
        }
    }
    assert_eq!(j, cell_of_row.len());
    assert!(seen.iter().all(|&x| x), "the payload set must be exactly the source's");
    assert_ne!(stream_digest(&src).unwrap(), stream_digest(&dst).unwrap());
    let dst2 = tmp_dir("perm-dst2");
    permute_corpus(&src, &dst2, 20260901).unwrap();
    assert_eq!(stream_digest(&dst).unwrap(), stream_digest(&dst2).unwrap());
    for d in [&src, &dst, &dst2] {
        std::fs::remove_dir_all(d).ok();
    }
}
