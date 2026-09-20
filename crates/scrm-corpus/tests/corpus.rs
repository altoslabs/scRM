use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, DictionaryArray, Int32Array, ListBuilder, RecordBatch, StringArray, UInt8Builder, UInt16Array,
    UInt32Array,
};
use arrow::datatypes::{Int32Type, Schema};
use arrow_ipc::writer::{DictionaryHandling, FileWriter, IpcWriteOptions};
use proptest::prelude::*;
use scrm_core::record::Umi;
use scrm_core::{ALPHABET_SIZE, CellId, MoleculeContigs, Stamp, Token};
use scrm_corpus::{
    CorpusError, CorpusReader, CorpusWriter, META_CELL_ROWS, META_DECLARED, META_N_ROWS, META_N_TOKENS,
    META_TOKENS_SHA256, corpus_fields, verify,
};
use sha2::{Digest, Sha256};

const TAB: u8 = Token::Tab as u8;
const NEWLINE: u8 = Token::Newline as u8;

fn stamp() -> Stamp {
    Stamp {
        declared: declared(),
        measured: measured(),
    }
}

fn declared() -> scrm_core::stamp::Declared {
    use scrm_core::stamp::*;
    Declared {
        build: Build {
            scrm_sha: "d99e4df".into(),
            dirty: false,
            built_at_utc: "2026-08-15T00:00:00Z".into(),
        },
        config: Config {
            geometry: "1{b[16]u[12]}2{r:}".into(),
            cb_len: 16,
            umi_len: 12,
            whitelist_sha256: "0".repeat(64),
            alphabet_size: ALPHABET_SIZE,
            dedup_rule: "1MM_CR_flow".into(),
            k: scrm_core::K,
            min_trim_len: scrm_core::MIN_TRIM_LEN,
            trim: serde_json::json!({"length": 20, "poly_a": true}),
            read_policy: "R1: CB+UMI only, byte-identical".into(),
            bubble_resolution: "weighted".into(),
            max_windows_per_cell: 0,
            min_umi_len: scrm_core::MIN_UMI_LEN_DEFAULT,
            max_umi_len: scrm_core::MAX_UMI_LEN_DEFAULT,
            min_cell_molecules: scrm_core::MIN_CELL_MOLECULES_DEFAULT,
        },
        inputs: vec![InputFile {
            path: "/dev/null".into(),
            bytes: 0,
            sha256: "1".repeat(64),
        }],
        tools: Tools {
            rustc: "1.95.0".into(),
            paraseq: "0.5.0-pre.2".into(),
            rapidgzip_core: "0.3.1".into(),
            thread_broker: "0.1.0".into(),
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
            molecules_out: 3,
            tokens_out: 30,
        },
        umi_stats: UmiStats {
            mean_hamming1_degree: 0.051,
            singleton_fraction: 0.9,
            merge_rate: 0.01,
            max_group_fraction: 0.0006,
            max_group_fraction_umis: 3300,
            max_group_umis: 2,
            max_chain_depth: 1,
        },
        assembly_stats: AssemblyStats {
            reads_per_molecule_hist: vec![0, 3],
            distinct_seqs_per_molecule_hist: vec![0, 3],
            pass_through_fraction: 0.8325,
            contigs_per_molecule_hist: vec![0, 3],
            contig_length_hist: vec![0, 3],
            self_loop_molecules: 0,
            bubble_n_tokens: 3,
            bubble_n_tie: 2,
            bubble_n_low_confidence: 1,
            bubble_n_indel_interior: 0,
            ambiguous_read_placements: 4,
            branch_decisions: 5,
            origin_splits: 2,
            origin_surprise_hist: vec![0, 3, 1, 0, 1],
            called_q_hist: vec![0, 0, 2, 4],
            origin_k_total: 9,
            origin_lambda_total: 0.5,
        },
        artifact_rates: ArtifactRates {
            adapter_fraction: 0.01,
            tso_fraction: 0.02,
            homopolymer: [("A_fwd_internal_ge20".to_string(), 0.00484)].into_iter().collect(),
        },
        throughput: Throughput {
            read_pairs_per_sec: 312_000.0,
            molecules_per_sec: 1000.0,
            secs_stream: 1.0,
            secs_assemble_write: 2.0,
            n_threads: 4,
        },
    }
}

fn mol(cell: u64, umi: u64, n_reads: u32, tokens: Vec<u8>) -> MoleculeContigs {
    let mut m = MoleculeContigs {
        cell: CellId(cell),
        umi: Umi(umi),
        n_reads,
        tokens,
        n_contigs: 0,
    };
    m.n_contigs = m.recount_contigs() as u16;
    m
}

fn tmp(name: &str) -> PathBuf {
    // PID-unique for the same reason as scrm-cli/tests/end_to_end.rs: a fixed
    // target/tmp/pyarrow_fixture.arrow made two concurrent workspace runs fight over one file.
    let dir = &Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("p{}", std::process::id()));
    std::fs::create_dir_all(dir).ok();
    std::fs::create_dir_all(dir).unwrap();
    dir.join(name)
}

fn sample_rows() -> Vec<(&'static str, MoleculeContigs)> {
    vec![
        ("AAACCCAAGAAACACT", mol(1, 11, 3, vec![0, 1, 2, 3, 0])),
        ("AAACCCAAGAAACACT", mol(1, 12, 1, vec![3, 3, 3, TAB, 0, 0, 4])),
        ("AAACCCAAGAAACACT", mol(1, 13, 9, vec![4])),
        ("AAACCCACAGAGGTAT", mol(2, 21, 2, vec![2, 2, TAB, 1, 1, TAB, 0, 0])),
        ("TTTGTTGCATTAGCCA", mol(3, 31, 7, vec![0])),
        ("TTTGTTGCATTAGCCA", mol(3, 32, 7, vec![1, 2, 3, 4, 0, 1, 2, 3])),
    ]
}

fn write_corpus(path: &Path, rows: &[(&str, MoleculeContigs)]) -> scrm_corpus::Result<()> {
    let mut w = CorpusWriter::create(path, &declared())?;
    for (cell, m) in rows {
        w.push(cell, m)?;
    }
    w.finish().map(|_| ())
}

#[test]
fn rows_spanning_a_batch_boundary_round_trip() {
    use scrm_corpus::BATCH_ROWS;
    const FIRST: &str = "AAACCCAAGGCATCAG";
    const SECOND: &str = "CCCTTTGGGAAACCCA";
    const THIRD: &str = "TTTGTTGCATTAGCCA";

    let mut rows: Vec<(String, MoleculeContigs)> = Vec::new();
    for u in 0..BATCH_ROWS + 8 {
        rows.push((FIRST.to_string(), mol(0, u as u64, 1, vec![u as u8 % 5])));
    }
    for u in 0..4 {
        rows.push((SECOND.to_string(), mol(1, u, 1, vec![1, TAB, 2])));
    }
    rows.push((THIRD.to_string(), mol(2, 0, 1, vec![3])));

    let path = tmp("batch_boundary.arrow");
    let borrowed: Vec<(&str, MoleculeContigs)> = rows.iter().map(|(c, m)| (c.as_str(), m.clone())).collect();
    write_corpus(&path, &borrowed).unwrap();
    verify(&path).unwrap();

    let mut r = CorpusReader::open(&path).unwrap();
    assert_eq!(r.n_rows(), rows.len());
    assert_eq!(r.cells().len(), 3, "cell index lost a cell across the boundary");

    let mut seen = 0usize;
    let mut batches = 0usize;
    while let Some(batch) = r.next_batch() {
        let batch = batch.unwrap();
        batches += 1;
        for i in 0..batch.len() {
            let got = batch.row(i);
            let (cell, want) = &rows[seen];
            assert_eq!(got.cell_id, cell.as_str(), "row {seen} cell_id");
            assert_eq!(got.tokens, want.tokens.as_slice(), "row {seen} tokens");
            seen += 1;
        }
    }
    assert_eq!(seen, rows.len());
    assert!(batches >= 2, "only {batches} batch(es): the boundary was never crossed");
    std::fs::remove_file(&path).ok();
}

#[test]
fn round_trip_is_byte_identical() {
    let path = tmp("round_trip.arrow");
    let rows = sample_rows();
    write_corpus(&path, &rows).unwrap();

    let mut r = CorpusReader::open(&path).unwrap();
    assert_eq!(r.n_rows(), rows.len());
    assert_eq!(
        r.n_tokens(),
        rows.iter().map(|(_, m)| m.tokens.len() as u64).sum::<u64>()
    );

    let mut seen = 0usize;
    while let Some(batch) = r.next_batch() {
        let batch = batch.unwrap();
        for i in 0..batch.len() {
            let got = batch.row(i);
            let (cell, want) = &rows[seen];
            assert_eq!(got.cell_id, *cell, "row {seen} cell_id");
            assert_eq!(got.n_reads, want.n_reads, "row {seen} n_reads");
            assert_eq!(got.n_contigs, want.n_contigs, "row {seen} n_contigs");
            assert_eq!(got.tokens, want.tokens.as_slice(), "row {seen} tokens");
            seen += 1;
        }
    }
    assert_eq!(seen, rows.len());
    verify(&path).unwrap();
}

#[test]
fn cell_range_is_the_one_contiguous_slice_of_a_cell() {
    let path = tmp("cell_range.arrow");
    write_corpus(&path, &sample_rows()).unwrap();
    let r = CorpusReader::open(&path).unwrap();
    assert_eq!(r.cell_range("AAACCCAAGAAACACT"), Some(0..3));
    assert_eq!(r.cell_range("AAACCCACAGAGGTAT"), Some(3..4));
    assert_eq!(r.cell_range("TTTGTTGCATTAGCCA"), Some(4..6));
    assert_eq!(r.cell_range("NOTACELL"), None);
    assert_eq!(r.cells().len(), 3);
}

#[test]
fn many_cells_over_many_batches_round_trip() {
    let path = tmp("many_batches.arrow");
    let mut rows: Vec<(String, MoleculeContigs)> = Vec::new();
    for c in 0..400u64 {
        for u in 0..3u64 {
            let tokens = vec![(c % 4) as u8, ((c + u) % 4) as u8, TAB, (u % 5) as u8];
            rows.push((format!("CELL{c:05}"), mol(c, u, 1 + u as u32, tokens)));
        }
    }
    let borrowed: Vec<(&str, MoleculeContigs)> = rows.iter().map(|(c, m)| (c.as_str(), m.clone())).collect();
    write_corpus(&path, &borrowed).unwrap();
    verify(&path).unwrap();

    let r = CorpusReader::open(&path).unwrap();
    assert_eq!(r.n_rows(), 1200);
    assert_eq!(r.cells().len(), 400);
    assert_eq!(r.cell_range("CELL00399"), Some(1197..1200));
}

#[test]
fn a_declaration_carrying_an_unknown_field_is_refused() {
    let json = declared().to_json().unwrap();

    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["config"]["min_reads_per_molecule"] = serde_json::json!(2);
    let err = scrm_core::stamp::Declared::from_json(&serde_json::to_string(&v).unwrap())
        .expect_err("an unknown config field must be refused");
    assert!(
        err.to_string().contains("min_reads_per_molecule"),
        "the refusal must name the field: {err}"
    );

    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["provenance"] = serde_json::json!({"who": "a newer builder"});
    assert!(scrm_core::stamp::Declared::from_json(&serde_json::to_string(&v).unwrap()).is_err());
    scrm_core::stamp::Declared::from_json(&json).expect("the real declaration must still load");
}

#[test]
fn a_shard_carries_the_declared_stamp_verbatim() {
    let path = tmp("stamp.arrow");
    write_corpus(&path, &sample_rows()).unwrap();
    let r = CorpusReader::open(&path).unwrap();
    let want = declared();

    assert_eq!(r.declared().to_json().unwrap(), want.to_json().unwrap());

    assert_eq!(r.declared().config.alphabet_size, want.config.alphabet_size);
    assert_eq!(r.declared().config.k, want.config.k);
    assert_eq!(r.declared().config.umi_len, want.config.umi_len);
    assert_eq!(r.declared().config.dedup_rule, want.config.dedup_rule);
    assert_eq!(r.declared().config.bubble_resolution, want.config.bubble_resolution);
    assert_eq!(r.declared().build.scrm_sha, want.build.scrm_sha);
    assert_eq!(r.declared().config.geometry, want.config.geometry);
    assert_eq!(r.declared().tools.debruijn, want.tools.debruijn);

    std::fs::remove_file(&path).ok();
}

#[test]
fn compat_key_reads_through_both_halves() {
    let want = stamp();
    assert_eq!(
        want.compat_key(),
        (
            want.declared.config.alphabet_size,
            want.declared.config.k,
            want.declared.config.dedup_rule.as_str(),
            want.declared.config.bubble_resolution.as_str(),
            want.declared.config.min_umi_len,
            want.declared.config.max_umi_len,
            want.declared.config.min_cell_molecules,
        )
    );
    assert_eq!(
        want.measured.artifact_rates.homopolymer.get("A_fwd_internal_ge20"),
        Some(&0.00484)
    );
    let back = scrm_core::Stamp::from_json(&want.to_json().unwrap()).unwrap();
    assert_eq!(back.compat_key(), want.compat_key());
}

#[test]
fn compat_key_separates_corpora_filtered_under_different_bounds() {
    let base = stamp();
    for (label, mutate) in [
        (
            "min",
            (|c: &mut scrm_core::stamp::Config| c.min_umi_len += 1) as fn(&mut _),
        ),
        ("max", |c: &mut scrm_core::stamp::Config| c.max_umi_len += 1),
        ("min_cell_molecules", |c: &mut scrm_core::stamp::Config| {
            c.min_cell_molecules += 1
        }),
    ] {
        let mut other = stamp();
        mutate(&mut other.declared.config);
        assert_ne!(
            base.compat_key(),
            other.compat_key(),
            "{label} must gate compatibility: a corpus filtered differently holds a different \
             molecule or cell set, and mixing the two must hard-fail on load"
        );
    }
    let mut same = stamp();
    same.measured.counts.pairs_in += 1;
    assert_eq!(base.compat_key(), same.compat_key());
}

#[test]
fn compat_key_ignores_umi_length() {
    let base = stamp();
    let mut v2 = stamp();
    v2.declared.config.umi_len = 10;
    assert_ne!(
        base.declared.config.umi_len, v2.declared.config.umi_len,
        "the fixture must actually differ or this test asserts nothing"
    );
    assert_eq!(
        base.compat_key(),
        v2.compat_key(),
        "a 10 bp UMI is Chromium V2, not an incompatibility: the model reads cDNA tokens, \
         nothing it sees depends on UMI length, and this project already mixes 3' and 5' \
         chemistries that differ far more"
    );
}

#[test]
fn out_of_order_cell_id_is_rejected() {
    let path = tmp("unsorted.arrow");
    let mut w = CorpusWriter::create(&path, &declared()).unwrap();
    w.push("CCCC", &mol(1, 1, 1, vec![0])).unwrap();
    let err = w.push("BBBB", &mol(2, 2, 1, vec![1])).unwrap_err();
    assert!(
        matches!(err, CorpusError::CellIdUnsorted { .. }),
        "expected CellIdUnsorted, got {err:?}"
    );
}

#[test]
fn cell_id_in_two_runs_is_rejected() {
    let path = tmp("noncontig.arrow");
    let mut w = CorpusWriter::create(&path, &declared()).unwrap();
    w.push("AAAA", &mol(1, 1, 1, vec![0])).unwrap();
    w.push("BBBB", &mol(2, 2, 1, vec![1])).unwrap();
    let err = w.push("AAAA", &mol(1, 3, 1, vec![2])).unwrap_err();
    assert!(
        matches!(
            err,
            CorpusError::Core(scrm_core::Error::CellIdNotContiguous { runs: 2, .. })
        ),
        "expected CellIdNotContiguous, got {err:?}"
    );
}

#[test]
fn one_cell_id_string_under_two_packed_barcodes_is_rejected() {
    let path = tmp("aliased.arrow");
    let mut w = CorpusWriter::create(&path, &declared()).unwrap();
    w.push("AAAA", &mol(1, 1, 1, vec![0])).unwrap();
    let err = w.push("AAAA", &mol(999, 2, 1, vec![1])).unwrap_err();
    assert!(
        matches!(err, CorpusError::CellIdAliased { .. }),
        "expected CellIdAliased, got {err:?}"
    );
}

#[test]
fn token_at_or_above_alphabet_size_is_rejected_on_write() {
    let path = tmp("bad_token.arrow");
    let mut w = CorpusWriter::create(&path, &declared()).unwrap();
    let err = w.push("AAAA", &mol(1, 1, 1, vec![0, ALPHABET_SIZE])).unwrap_err();
    assert!(
        matches!(
            err,
            CorpusError::Core(scrm_core::Error::TokenOutOfRange { token }) if token == ALPHABET_SIZE
        ),
        "expected TokenOutOfRange, got {err:?}"
    );
}

#[test]
fn a_row_with_no_tokens_is_rejected_on_write() {
    let path = tmp("empty-row.arrow");
    let mut w = CorpusWriter::create(&path, &declared()).unwrap();
    let err = w.push("AAAA", &mol(1, 1, 1, vec![])).unwrap_err();
    assert!(
        matches!(err, CorpusError::EmptyRow { row } if row == 0),
        "expected EmptyRow, got {err:?}"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn newline_token_is_rejected_on_write() {
    let path = tmp("newline.arrow");
    let mut w = CorpusWriter::create(&path, &declared()).unwrap();
    let err = w.push("AAAA", &mol(1, 1, 1, vec![0, NEWLINE, 1])).unwrap_err();
    assert!(
        matches!(err, CorpusError::Separator { token, .. } if token == NEWLINE),
        "expected a Newline separator error, got {err:?}"
    );
}

#[test]
fn malformed_tab_structure_is_rejected_on_write() {
    for (name, tokens) in [
        ("leading", vec![TAB, 0, 1]),
        ("trailing", vec![0, 1, TAB]),
        ("doubled", vec![0, TAB, TAB, 1]),
        ("only", vec![TAB]),
    ] {
        let path = tmp(&format!("tab_{name}.arrow"));
        let mut w = CorpusWriter::create(&path, &declared()).unwrap();
        let err = w.push("AAAA", &mol(1, 1, 1, tokens)).unwrap_err();
        assert!(
            matches!(err, CorpusError::Separator { token, .. } if token == TAB),
            "{name}: expected a Tab separator error, got {err:?}"
        );
    }
}

#[test]
fn n_contigs_disagreeing_with_the_recount_is_rejected_on_write() {
    let path = tmp("n_contigs.arrow");
    let mut w = CorpusWriter::create(&path, &declared()).unwrap();
    let mut m = mol(1, 1, 1, vec![0, 1, TAB, 2, 3]);
    m.n_contigs = 1;
    let err = w.push("AAAA", &m).unwrap_err();
    assert!(
        matches!(
            err,
            CorpusError::ContigCountMismatch {
                declared: 1,
                recounted: 2,
                ..
            }
        ),
        "expected ContigCountMismatch, got {err:?}"
    );
}

#[test]
fn truncation_is_detected() {
    let path = tmp("truncated.arrow");
    write_corpus(&path, &sample_rows()).unwrap();
    let full = std::fs::metadata(&path).unwrap().len();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(full - 64)
        .unwrap();

    let err = verify(&path).unwrap_err();
    assert!(
        matches!(err, CorpusError::Arrow(_) | CorpusError::Io(_)),
        "expected the Arrow footer check to fail, got {err:?}"
    );
    assert!(CorpusReader::open(&path).is_err());
}

#[test]
fn a_flipped_token_byte_is_detected_by_sha256() {
    let path = tmp("flipped.arrow");
    let needle: Vec<u8> = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1];
    let rows = vec![("AAAA", mol(1, 1, 1, needle.clone()))];
    write_corpus(&path, &rows).unwrap();
    verify(&path).unwrap();

    let mut bytes = Vec::new();
    File::open(&path).unwrap().read_to_end(&mut bytes).unwrap();
    let hits: Vec<usize> = bytes
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == &needle[..])
        .map(|(i, _)| i)
        .collect();
    assert_eq!(hits.len(), 1, "token pattern must be unique in the file");
    bytes[hits[0]] = 3;
    std::fs::write(&path, &bytes).unwrap();

    let err = verify(&path).unwrap_err();
    assert!(
        matches!(err, CorpusError::ShaMismatch { .. }),
        "expected ShaMismatch, got {err:?}"
    );
}

#[test]
fn a_short_write_that_closed_cleanly_is_detected() {
    let path = tmp("short_write.arrow");
    let rows = raw_rows();
    write_raw(&path, &rows, |meta, honest_rows, honest_tokens, sha| {
        meta.insert(META_N_ROWS.into(), (honest_rows + 1).to_string());
        meta.insert(META_N_TOKENS.into(), honest_tokens.to_string());
        meta.insert(META_TOKENS_SHA256.into(), sha.into());
        meta.insert(
            META_CELL_ROWS.into(),
            r#"[{"cell":"AAACCCAAGAAACACT","start":0,"len":3},
                {"cell":"AAACCCACAGAGGTAT","start":3,"len":1},
                {"cell":"TTTGTTGCATTAGCCA","start":4,"len":3}]"#
                .into(),
        );
    });
    let err = verify(&path).unwrap_err();
    assert!(
        matches!(err, CorpusError::TallyMismatch { what: "row count", .. }),
        "expected a row-count TallyMismatch, got {err:?}"
    );

    let path = tmp("short_tokens.arrow");
    write_raw(&path, &rows, |meta, honest_rows, honest_tokens, sha| {
        meta.insert(META_N_ROWS.into(), honest_rows.to_string());
        meta.insert(META_N_TOKENS.into(), (honest_tokens + 1).to_string());
        meta.insert(META_TOKENS_SHA256.into(), sha.into());
    });
    let err = verify(&path).unwrap_err();
    assert!(
        matches!(
            err,
            CorpusError::TallyMismatch {
                what: "token count",
                ..
            }
        ),
        "expected a token-count TallyMismatch, got {err:?}"
    );
}

#[test]
fn a_foreign_writers_out_of_range_token_is_rejected_on_first_read() {
    let path = tmp("foreign_token.arrow");
    let rows = vec![RawRow {
        cell: "AAAA".into(),
        n_reads: 1,
        n_contigs: 1,
        tokens: vec![0, 1, ALPHABET_SIZE + 2],
    }];
    write_raw(&path, &rows, |meta, r, t, sha| {
        meta.insert(META_N_ROWS.into(), r.to_string());
        meta.insert(META_N_TOKENS.into(), t.to_string());
        meta.insert(META_TOKENS_SHA256.into(), sha.into());
    });
    let mut reader = CorpusReader::open(&path).unwrap();
    let err = reader.next_batch().unwrap().unwrap_err();
    assert!(
        matches!(err, CorpusError::Core(scrm_core::Error::TokenOutOfRange { .. })),
        "expected TokenOutOfRange, got {err:?}"
    );
}

#[test]
fn a_foreign_writers_unsorted_cell_ids_are_rejected_by_verify() {
    let path = tmp("foreign_unsorted.arrow");
    let rows = ["AAAA", "BBBB", "AAAA"]
        .iter()
        .enumerate()
        .map(|(i, cell)| RawRow {
            cell: (*cell).into(),
            n_reads: 1,
            n_contigs: 1,
            tokens: vec![i as u8],
        })
        .collect::<Vec<_>>();
    write_raw(&path, &rows, |meta, r, t, sha| {
        meta.insert(META_N_ROWS.into(), r.to_string());
        meta.insert(META_N_TOKENS.into(), t.to_string());
        meta.insert(META_TOKENS_SHA256.into(), sha.into());
        meta.insert(
            META_CELL_ROWS.into(),
            r#"[{"cell":"AAAA","start":0,"len":1},{"cell":"BBBB","start":1,"len":1},{"cell":"CCCC","start":2,"len":1}]"#
                .into(),
        );
    });
    let err = verify(&path).unwrap_err();
    assert!(
        matches!(err, CorpusError::Core(scrm_core::Error::CellIdNotContiguous { .. })),
        "expected CellIdNotContiguous, got {err:?}"
    );
}

#[test]
fn write_pyarrow_fixture() {
    use scrm_corpus::BATCH_ROWS;

    const CELLS: usize = 400;
    let per_cell = BATCH_ROWS / 100;
    let mut rows: Vec<(String, MoleculeContigs)> = Vec::with_capacity(CELLS * per_cell);
    for c in 0..CELLS {
        let cell_id = format!("CELL{c:05}");
        for u in 0..per_cell {
            let tokens = vec![(c % 4) as u8, TAB, (u % 4) as u8];
            rows.push((cell_id.clone(), mol(c as u64, u as u64, 1 + (u as u32 % 7), tokens)));
        }
    }
    assert!(
        rows.len() > BATCH_ROWS * 2,
        "{} rows is not enough to cross a {BATCH_ROWS}-row batch boundary twice",
        rows.len()
    );

    let path = tmp("pyarrow_fixture.arrow");
    let borrowed: Vec<(&str, MoleculeContigs)> = rows.iter().map(|(c, m)| (c.as_str(), m.clone())).collect();
    write_corpus(&path, &borrowed).unwrap();
    verify(&path).unwrap();

    let r = CorpusReader::open(&path).unwrap();
    assert_eq!(r.n_rows(), rows.len());
    assert_eq!(r.cells().len(), CELLS);
    println!(
        "pyarrow fixture: {} ({} rows, {CELLS} cells, >{} batches)",
        path.display(),
        rows.len(),
        rows.len() / BATCH_ROWS
    );
}

fn corpus_strategy() -> impl Strategy<Value = Vec<(String, MoleculeContigs)>> {
    let contig = prop::collection::vec(0u8..=4, 1..6);
    let molecule = prop::collection::vec(contig, 1..4);
    let cell = prop::collection::vec((molecule, 1u32..100), 1..4);
    prop::collection::vec(cell, 1..5).prop_map(|cells| {
        let mut rows = Vec::new();
        for (c, mols) in cells.into_iter().enumerate() {
            let cell_id = format!("CELL{c:04}");
            for (u, (contigs, n_reads)) in mols.into_iter().enumerate() {
                let mut tokens = Vec::new();
                for (i, contig) in contigs.iter().enumerate() {
                    if i > 0 {
                        tokens.push(TAB);
                    }
                    tokens.extend_from_slice(contig);
                }
                rows.push((cell_id.clone(), mol(c as u64, u as u64, n_reads, tokens)));
            }
        }
        rows
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_valid_corpora_round_trip_and_verify(rows in corpus_strategy()) {
        let path = tmp("proptest.arrow");
        let borrowed: Vec<(&str, MoleculeContigs)> =
            rows.iter().map(|(c, m)| (c.as_str(), m.clone())).collect();
        write_corpus(&path, &borrowed).unwrap();
        verify(&path).unwrap();

        let mut r = CorpusReader::open(&path).unwrap();
        prop_assert_eq!(r.n_rows(), rows.len());
        let mut seen = 0usize;
        while let Some(batch) = r.next_batch() {
            let batch = batch.unwrap();
            for i in 0..batch.len() {
                let got = batch.row(i);
                let (cell, want) = &rows[seen];
                prop_assert_eq!(got.cell_id, cell.as_str());
                prop_assert_eq!(got.n_reads, want.n_reads);
                prop_assert_eq!(got.n_contigs, want.n_contigs);
                prop_assert_eq!(got.tokens, want.tokens.as_slice());
                seen += 1;
            }
        }
        prop_assert_eq!(seen, rows.len());

        let mut at = 0usize;
        for run in r.cells() {
            let range = r.cell_range(&run.cell).unwrap();
            prop_assert_eq!(range.start, at);
            at = range.end;
        }
        prop_assert_eq!(at, rows.len());
    }
}

struct RawRow {
    cell: String,
    n_reads: u32,
    n_contigs: u16,
    tokens: Vec<u8>,
}

fn raw_rows() -> Vec<RawRow> {
    sample_rows()
        .into_iter()
        .map(|(cell, m)| RawRow {
            cell: cell.into(),
            n_reads: m.n_reads,
            n_contigs: m.n_contigs,
            tokens: m.tokens,
        })
        .collect()
}

fn write_raw(path: &Path, rows: &[RawRow], patch: impl Fn(&mut HashMap<String, String>, u64, u64, &str)) {
    let mut cell_dict: Vec<String> = Vec::new();
    let mut cell_keys = Vec::new();
    let mut tokens = ListBuilder::new(UInt8Builder::new()).with_field(Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::UInt8,
        false,
    )));
    let mut hasher = Sha256::new();
    let mut n_tokens = 0u64;
    for row in rows {
        let key = |dict: &mut Vec<String>, v: &str| match dict.iter().position(|d| d == v) {
            Some(i) => i as i32,
            None => {
                dict.push(v.to_string());
                (dict.len() - 1) as i32
            }
        };
        cell_keys.push(key(&mut cell_dict, &row.cell));
        tokens.values().append_slice(&row.tokens);
        tokens.append(true);
        hasher.update(&row.tokens);
        n_tokens += row.tokens.len() as u64;
    }
    let dict = |keys: Vec<i32>, values: &[String]| -> ArrayRef {
        Arc::new(
            DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(keys),
                Arc::new(StringArray::from_iter_values(values.iter())),
            )
            .unwrap(),
        )
    };
    let schema = Arc::new(
        Schema::new(corpus_fields()).with_metadata([(META_DECLARED.to_string(), declared().to_json().unwrap())].into()),
    );
    let columns: Vec<ArrayRef> = vec![
        dict(cell_keys, &cell_dict),
        Arc::new(UInt32Array::from(rows.iter().map(|r| r.n_reads).collect::<Vec<_>>())),
        Arc::new(UInt16Array::from(rows.iter().map(|r| r.n_contigs).collect::<Vec<_>>())),
        Arc::new(tokens.finish()),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).unwrap();

    let options = IpcWriteOptions::default().with_dictionary_handling(DictionaryHandling::Delta);
    let mut w =
        FileWriter::try_new_with_options(BufWriter::new(File::create(path).unwrap()), &schema, options).unwrap();
    w.write(&batch).unwrap();

    let sha = format!("{:x}", hasher.finalize());
    let mut runs: Vec<serde_json::Value> = Vec::new();
    for (start, row) in rows.iter().enumerate() {
        match runs.last_mut() {
            Some(last) if last["cell"] == row.cell.as_str() => {
                let len = last["len"].as_u64().unwrap() + 1;
                last["len"] = serde_json::json!(len);
            }
            _ => runs.push(serde_json::json!({"cell": row.cell, "start": start, "len": 1})),
        }
    }
    let mut meta: HashMap<String, String> = [(META_CELL_ROWS.to_string(), serde_json::to_string(&runs).unwrap())]
        .into_iter()
        .collect();
    patch(&mut meta, rows.len() as u64, n_tokens, &sha);
    for (k, v) in meta {
        w.write_metadata(k, v);
    }
    w.finish().unwrap();
    w.get_mut().flush().unwrap();
}
