use std::path::{Path, PathBuf};

use scrm_pack::{Corpus, RowSource};

const ROOT_ENV: [&str; 3] = ["SCRM_CORPUS_ROOT", "CORPORA_ROOT", "CORPUS_ROOT"];
const PROJ_DEFAULT: &str = "";

fn corpus_root() -> PathBuf {
    for name in ROOT_ENV {
        if let Some(v) = std::env::var_os(name).filter(|v| !v.is_empty()) {
            return PathBuf::from(v);
        }
    }
    PathBuf::from(
        std::env::var_os("SCRM_PROJ")
            .filter(|v| !v.is_empty())
            .unwrap_or(PROJ_DEFAULT.into()),
    )
    .join("corpus")
}

fn corpus_dir() -> Option<PathBuf> {
    let root = corpus_root();
    let found = smallest_corpus_in(&root).unwrap_or_else(|e| panic!("{e}"));
    if found.is_none() && root.parent().is_some_and(Path::is_dir) {
        panic!(
            "{} holds no corpus with a manifest.json, but its parent directory exists -- so this is \
             a wrong or stale root, not a machine without the data. Set one of {ROOT_ENV:?} (or \
             SCRM_PROJ) to the tree that holds the corpora; skipping here would report `ok`.",
            root.display()
        );
    }
    found
}

fn smallest_corpus_in(root: &Path) -> Result<Option<PathBuf>, String> {
    let Ok(entries) = std::fs::read_dir(root) else {
        return Ok(None);
    };
    let mut best: Option<(u64, PathBuf)> = None;
    for entry in entries.flatten() {
        let dir = entry.path();
        let manifest = dir.join("manifest.json");
        if !manifest.exists() {
            continue;
        }
        let named = |what: String| format!("{}: {what}", manifest.display());
        let text = std::fs::read_to_string(&manifest).map_err(|e| named(format!("unreadable ({e})")))?;
        let json = serde_json::from_str::<serde_json::Value>(&text)
            .map_err(|e| named(format!("not JSON ({e}) -- a truncated manifest looks like this")))?;
        let rows = json
            .get("n_rows")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| named("no integer n_rows -- present, but not a manifest".into()))?;
        if best.as_ref().is_none_or(|(b, _)| rows < *b) {
            best = Some((rows, dir));
        }
    }
    Ok(best.map(|(_, d)| d))
}

fn scratch_root(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("scrm-pack-scan-{}-{name}", std::process::id()));
    std::fs::remove_dir_all(&d).ok();
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn fake_corpus(root: &Path, name: &str, manifest: &str) {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("manifest.json"), manifest).unwrap();
}

#[test]
fn a_manifest_that_is_not_a_manifest_fails_the_scan_instead_of_emptying_it() {
    let root = scratch_root("bad-neighbour");
    fake_corpus(&root, "good_small", r#"{"n_rows": 10}"#);
    fake_corpus(&root, "good_large", r#"{"n_rows": 99}"#);
    assert_eq!(
        smallest_corpus_in(&root).unwrap(),
        Some(root.join("good_small")),
        "with every manifest readable the scan must pick the smallest corpus"
    );

    fake_corpus(&root, "torn", r#"{"n_rows": 10, "shar"#);
    let err = smallest_corpus_in(&root).expect_err("a torn manifest must be reported, not skipped");
    assert!(
        err.contains("torn") && err.contains("not JSON"),
        "unhelpful error: {err}"
    );

    std::fs::write(root.join("torn").join("manifest.json"), r#"{"n_rows": "many"}"#).unwrap();
    let err = smallest_corpus_in(&root).expect_err("a non-numeric n_rows must be reported");
    assert!(err.contains("no integer n_rows"), "unhelpful error: {err}");

    std::fs::remove_dir_all(&root).ok();
}

#[test]
fn an_absent_corpus_root_is_the_only_reason_to_skip() {
    let root = scratch_root("absent");
    std::fs::remove_dir_all(&root).unwrap();
    assert_eq!(smallest_corpus_in(&root).unwrap(), None);
}

#[test]
fn the_training_reader_returns_exactly_what_the_pipeline_reader_returns() {
    let Some(dir) = corpus_dir() else {
        eprintln!("SKIP: no corpus with a manifest.json under {}", corpus_root().display());
        return;
    };
    let corpus = Corpus::open(&dir).expect("open");
    let cells = corpus.cells();
    let declared_cells = corpus.cells().len();
    assert!(declared_cells > 0, "the corpus holds no cells");

    assert!(
        cells.windows(2).all(|w| w[0].name < w[1].name),
        "cells are not in ascending order"
    );

    let manifest = scrm_corpus::Manifest::read(&dir).unwrap();
    let shard0 = dir.join(&manifest.shards[0].file);
    let mut want: Vec<Vec<u8>> = Vec::new();
    let mut r = scrm_corpus::CorpusReader::open(&shard0).unwrap();
    while let Some(batch) = r.next_batch() {
        let batch = batch.unwrap();
        for i in 0..batch.len() {
            want.push(batch.row(i).tokens.to_vec());
        }
    }
    assert!(!want.is_empty());

    let mut got: Vec<Vec<u8>> = Vec::new();
    let mut lengths = Vec::new();
    let mut spanning = 0usize;
    for cell in cells.iter().filter(|c| c.shard == 0) {
        let rows = corpus.cell_rows(cell).expect("cell_rows");
        assert_eq!(rows.n_rows(), cell.n_rows as usize);
        corpus.cell_lengths(cell, &mut lengths).expect("cell_lengths");
        assert_eq!(lengths.len(), cell.n_rows as usize);
        if (cell.start_row / 65_536) != ((cell.start_row + cell.n_rows - 1) / 65_536) {
            spanning += 1;
        }
        for i in 0..cell.n_rows {
            let row = rows.row(i).expect("row");
            assert_eq!(row.len() as u32, lengths[i as usize], "length disagrees with the row");
            got.push(row.to_vec());
        }
    }

    assert_eq!(got.len(), want.len(), "row count differs from the pipeline reader");
    assert!(got == want, "token bytes differ from the pipeline reader");
    assert!(
        spanning > 0,
        "no cell straddled a batch boundary, so the multi-batch path was never exercised -- \
         the test would pass with a single-batch-only reader"
    );
    eprintln!(
        "checked {} rows, {spanning} cells straddling a batch boundary",
        got.len()
    );
}

#[test]
fn opening_a_directory_that_is_not_a_corpus_fails_rather_than_panics() {
    match Corpus::open(Path::new("/tmp")) {
        Err(scrm_pack::PackError::Corpus(msg)) => {
            assert!(!msg.is_empty(), "the error must say what went wrong");
        }
        Err(other) => panic!("wrong error variant: {other:?}"),
        Ok(_) => panic!("/tmp is not a corpus and must not open as one"),
    }
}

use scrm_pack::{BatchSpec, build_batch, derive_seed};

fn spec(cells: Vec<u32>, seed: u64) -> BatchSpec {
    BatchSpec {
        cells,
        v: 2,
        seq_len: 16384,
        epoch_seed: seed,
        frac: 1.0,
        frac_seed: 0,
    }
}

#[test]
fn a_batch_is_cell_grouped_and_every_window_is_well_formed() {
    let Some(dir) = corpus_dir() else {
        eprintln!("SKIP: no corpus under {}", corpus_root().display());
        return;
    };
    let corpus = Corpus::open(&dir).unwrap();
    let b = 16usize;
    let batch = build_batch(&corpus, &spec((0..b as u32).collect(), 7)).unwrap();

    assert_eq!(batch.n_windows, b * 2);
    assert_eq!(
        batch.short_cells, 0,
        "the 500-molecule floor guarantees >= 2 windows per cell"
    );

    for (w, &ci) in batch.cell_index.iter().enumerate() {
        assert_eq!(ci as usize, w / 2, "window {w} is not in its cell's group");
    }

    let sl = batch.seq_len;
    let mp = scrm_pack::max_patches(sl);
    let mut total_used = 0usize;
    for w in 0..batch.n_windows {
        let tok = &batch.tokens[w * sl..(w + 1) * sl];
        let val = &batch.valid[w * sl..(w + 1) * sl];
        let cseg = &batch.contig_seg[w * sl..(w + 1) * sl];
        let useg = &batch.umi_seg[w * sl..(w + 1) * sl];
        let used = val.iter().filter(|&&v| v).count();
        total_used += used;

        assert!(used > 0, "window {w} is empty");
        assert!(val[..used].iter().all(|&v| v), "valid must be a prefix");
        assert!(!val[used..].iter().any(|&v| v), "valid must be a prefix");
        assert!(
            tok[..used].iter().all(|&t| t < 7),
            "window {w} holds a token outside the alphabet"
        );
        assert!(
            useg[..used].windows(2).all(|p| p[0] <= p[1]),
            "umi ids go backwards in {w}"
        );
        assert!(
            cseg[..used].windows(2).all(|p| p[0] <= p[1]),
            "contig ids go backwards in {w}"
        );

        let np = batch.n_patches[w] as usize;
        assert!(np > 0 && np <= mp, "window {w}: {np} patches against max_patches {mp}");
        let ps = &batch.patch_starts[w * mp..w * mp + np];
        assert_eq!(
            useg[used - 1] as usize + 1,
            np,
            "window {w}: n_patches disagrees with the number of distinct umi ids"
        );
        for i in 0..np {
            let s = ps[i] as usize;
            let e = if i + 1 < np { ps[i + 1] as usize } else { used };
            assert!(e > s, "patch {i} of window {w} is empty");
            assert!(
                useg[s..e].iter().all(|&x| x as usize == i),
                "patch {i} of window {w} does not span exactly molecule {i}'s umi ids"
            );
        }
    }
    let occupancy = total_used as f64 / (batch.n_windows * sl) as f64;
    assert!(occupancy > 0.95, "windows only {:.1}% full", occupancy * 100.0);
    eprintln!(
        "B={b} V=2  {} windows  {} tokens  {:.2}% occupancy",
        batch.n_windows,
        total_used,
        occupancy * 100.0
    );
}

#[test]
fn a_batch_is_reproducible_from_its_seed_and_rotating_it_changes_the_views() {
    let Some(dir) = corpus_dir() else {
        eprintln!("SKIP: no corpus under {}", corpus_root().display());
        return;
    };
    let corpus = Corpus::open(&dir).unwrap();
    let cells: Vec<u32> = (0..8).collect();

    let a = build_batch(&corpus, &spec(cells.clone(), 42)).unwrap();
    let b = build_batch(&corpus, &spec(cells.clone(), 42)).unwrap();
    assert_eq!(a.tokens, b.tokens, "the same seed must give the same batch");
    assert_eq!(a.contig_seg, b.contig_seg);
    assert_eq!(a.umi_seg, b.umi_seg);
    assert_eq!(a.n_patches, b.n_patches);

    let c = build_batch(&corpus, &spec(cells, 43)).unwrap();
    assert_ne!(
        a.tokens, c.tokens,
        "rotating the seed must change the views, or coverage never accumulates and every step \
         draws the same pair -- which would look like slow learning, not like a bug"
    );

    let drawn = |b: &scrm_pack::Batch, cell: usize| -> std::collections::HashSet<Vec<u8>> {
        let sl = b.seq_len;
        let mut out = std::collections::HashSet::new();
        for w in [cell * 2, cell * 2 + 1] {
            let tok = &b.tokens[w * sl..(w + 1) * sl];
            let useg = &b.umi_seg[w * sl..(w + 1) * sl];
            let used = b.valid[w * sl..(w + 1) * sl].iter().filter(|&&v| v).count();
            let mut at = 0usize;
            while at < used {
                let id = useg[at];
                let mut end = at;
                while end < used && useg[end] == id {
                    end += 1;
                }
                let mut span: &[u8] = &tok[at..end];
                if span.last() == Some(&6) {
                    span = &span[..span.len() - 1];
                }
                let mut parts: Vec<Vec<u8>> = span.split(|&x| x == 5).map(|p| p.to_vec()).collect();
                parts.sort();
                out.insert(parts.concat());
                at = end;
            }
        }
        out
    };

    let mut rotated = 0usize;
    for cell in 0..8 {
        let (x, y) = (drawn(&a, cell), drawn(&c, cell));
        assert!(!x.is_empty());
        let shared = x.intersection(&y).count();
        let overlap = shared as f64 / x.len() as f64;
        if x != y {
            rotated += 1;
        }
        assert!(
            overlap < 0.5,
            "cell {cell}: two seeds drew {:.0}% of the same molecules -- the molecule draw is not \
             rotating, so a step's views barely change and coverage never accumulates",
            overlap * 100.0
        );
    }
    assert_eq!(
        rotated, 8,
        "every cell's molecule set must change when the seed rotates"
    );
}

#[test]
fn a_cells_two_views_draw_no_molecule_twice() {
    let Some(dir) = corpus_dir() else {
        eprintln!("SKIP: no corpus under {}", corpus_root().display());
        return;
    };
    let corpus = Corpus::open(&dir).unwrap();
    let n_cells = 8usize;
    let batch = build_batch(&corpus, &spec((0..n_cells as u32).collect(), 11)).unwrap();
    let sl = batch.seq_len;
    let all = corpus.cells();

    let spans = |w: usize| -> Vec<Vec<u8>> {
        let tok = &batch.tokens[w * sl..(w + 1) * sl];
        let useg = &batch.umi_seg[w * sl..(w + 1) * sl];
        let used = batch.valid[w * sl..(w + 1) * sl].iter().filter(|&&v| v).count();
        let mut out = Vec::new();
        let mut at = 0usize;
        while at < used {
            let id = useg[at];
            let mut end = at;
            while end < used && useg[end] == id {
                end += 1;
            }
            let mut span: &[u8] = &tok[at..end];
            if span.last() == Some(&6) {
                span = &span[..span.len() - 1];
            }
            out.push(span.to_vec());
            at = end;
        }
        out
    };

    let canon = |m: &[u8]| -> Vec<Vec<u8>> {
        let mut parts: Vec<Vec<u8>> = m.split(|&b| b == 5).map(|p| p.to_vec()).collect();
        parts.sort();
        parts
    };

    for (c, loc) in all.iter().enumerate().take(n_cells) {
        let rows = corpus.cell_rows(loc).unwrap();
        let mut have: std::collections::HashMap<Vec<Vec<u8>>, usize> = std::collections::HashMap::new();
        for i in 0..loc.n_rows {
            *have.entry(canon(rows.row(i).unwrap())).or_default() += 1;
        }

        let mut drawn: std::collections::HashMap<Vec<Vec<u8>>, usize> = std::collections::HashMap::new();
        for w in [c * 2, c * 2 + 1] {
            for m in spans(w) {
                *drawn.entry(canon(&m)).or_default() += 1;
            }
        }
        assert!(!drawn.is_empty());
        for (m, &n) in &drawn {
            let available = have.get(m).copied().unwrap_or(0);
            assert!(
                available > 0,
                "cell {c}: a window holds a molecule the cell does not, so packing invented or \
                 corrupted a sequence"
            );
            assert!(
                n <= available,
                "cell {c}: a sequence appears {n} times across the two views but only {available} \
                 times in the cell, so a molecule was drawn twice -- the draw is not without \
                 replacement, and two views sharing a molecule makes the invariance target \
                 trivially easier for that cell"
            );
        }
    }
}

#[test]
fn an_empty_batch_spec_is_refused() {
    let Some(dir) = corpus_dir() else {
        eprintln!(
            "SKIP: no corpus under {} (see mixing_definitions.rs for the data-free version)",
            corpus_root().display()
        );
        return;
    };
    let corpus = Corpus::open(&dir).unwrap();
    assert!(build_batch(&corpus, &spec(vec![], 1)).is_err(), "no cells");
    assert!(
        build_batch(
            &corpus,
            &BatchSpec {
                cells: vec![0],
                v: 0,
                seq_len: 16384,
                epoch_seed: 1,
                frac: 1.0,
                frac_seed: 0
            }
        )
        .is_err(),
        "v = 0"
    );
    assert_eq!(derive_seed(1, 2, 3), derive_seed(1, 2, 3));
}
