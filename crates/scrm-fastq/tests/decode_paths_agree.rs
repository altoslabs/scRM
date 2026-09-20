use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use scrm_core::ParsedPair;
use scrm_fastq::{Parser, StreamConfig, stream_pairs};
use scrm_trim::TrimSettings;

const GEOMETRY: &str = "1{b[16]u[12]}2{r:}";
const CB: &[u8] = b"ACGTACGTACGTACGT";
const CDNA: &[u8] = b"GCTAGCTAGGCATCGATCGGATCCAAGCTTGGATCCTAGCTAGCTAGGCATCGATCGGAT";
const N_PAIRS: usize = 300;
const MEMBERS: usize = 4;

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("scrm-agree-{}-{name}", std::process::id()));
    p
}

fn record(i: usize, read1: bool) -> Vec<u8> {
    let bases = *b"ACGT";
    if read1 {
        let mut v = CB.to_vec();
        for k in 0..12 {
            v.push(bases[(i + k) % 4]);
        }
        v
    } else {
        CDNA.to_vec()
    }
}

fn phred(i: usize, k: usize) -> u8 {
    30 + ((i * 7 + k * 3) % 11) as u8
}

fn qual_line(i: usize, len: usize) -> Vec<u8> {
    (0..len).map(|k| 33 + phred(i, k)).collect()
}

fn fastq_bytes(read1: bool, range: std::ops::Range<usize>) -> Vec<u8> {
    let mut out = Vec::new();
    for i in range {
        let seq = record(i, read1);
        out.extend_from_slice(format!("@READ{i} {}:N:0:AGGT\n", if read1 { 1 } else { 2 }).as_bytes());
        out.extend_from_slice(&seq);
        out.extend_from_slice(b"\n+\n");
        out.extend_from_slice(&qual_line(i, seq.len()));
        out.push(b'\n');
    }
    out
}

fn write_plain(path: &Path, read1: bool) {
    std::fs::write(path, fastq_bytes(read1, 0..N_PAIRS)).unwrap();
}

fn write_gz(path: &Path, read1: bool) {
    let per_member = N_PAIRS / MEMBERS;
    let mut bytes = Vec::new();
    for m in 0..MEMBERS {
        let start = m * per_member;
        let end = if m + 1 == MEMBERS { N_PAIRS } else { start + per_member };
        let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        enc.write_all(&fastq_bytes(read1, start..end)).unwrap();
        bytes.extend_from_slice(&enc.finish().unwrap());
    }
    std::fs::write(path, bytes).unwrap();
}

type RunResult = (u64, u64, Vec<(u64, u64, Vec<u8>, Vec<u8>)>);

fn run(r1: &Path, r2: &Path) -> RunResult {
    let parser = Parser::new(GEOMETRY, TrimSettings::default(), |cb| Some(scrm_core::CellId(cb))).unwrap();
    let collected: Mutex<Vec<ParsedPair>> = Mutex::new(Vec::new());
    let config = StreamConfig {
        threads: 4,
        decoder_threads: 2,
        limit: None,
    };
    let counters = stream_pairs(r1, r2, &parser, &config, |pair| collected.lock().unwrap().push(pair)).unwrap();
    let mut pairs: Vec<(u64, u64, Vec<u8>, Vec<u8>)> = collected
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|p| (p.cb.0, p.umi.0, p.r2, p.qual))
        .collect();
    pairs.sort();
    (counters.pairs_in, counters.kept, pairs)
}

#[test]
fn gzip_and_plain_input_produce_identical_output() {
    let plain1 = tmp("plain.r1.fastq");
    let plain2 = tmp("plain.r2.fastq");
    let gz1 = tmp("gz.r1.fastq.gz");
    let gz2 = tmp("gz.r2.fastq.gz");
    write_plain(&plain1, true);
    write_plain(&plain2, false);
    write_gz(&gz1, true);
    write_gz(&gz2, false);

    let plain = run(&plain1, &plain2);
    let gzipped = run(&gz1, &gz2);

    assert_eq!(plain.0, N_PAIRS as u64, "plain pairs_in");
    assert_eq!(
        gzipped.0,
        N_PAIRS as u64,
        "{} lost records across {MEMBERS} gzip members",
        scrm_fastq::decode::backend()
    );
    assert_eq!(plain.1, gzipped.1, "kept counts differ");
    assert_eq!(
        plain.2, gzipped.2,
        "parsed pairs differ -- bases or QUALITIES, and the qualities are what the assembler weighs"
    );
    assert_eq!(plain.2.len(), N_PAIRS);

    let (mut seen_quals, mut lengths_agree) = (std::collections::HashSet::new(), true);
    for (_, _, r2, q) in &plain.2 {
        lengths_agree &= q.len() == r2.len();
        seen_quals.extend(q.iter().copied());
    }
    assert!(lengths_agree, "every parsed pair must carry one Phred per base");
    assert!(
        seen_quals.len() > 1,
        "the fixture's qualities do not vary, so no quality difference could be detected: {seen_quals:?}"
    );
    let mut want: Vec<Vec<u8>> = (0..N_PAIRS)
        .map(|i| (0..CDNA.len()).map(|k| phred(i, k)).collect())
        .collect();
    let mut got: Vec<Vec<u8>> = plain.2.iter().map(|(.., q)| q.clone()).collect();
    want.sort();
    got.sort();
    assert_eq!(got, want, "the parsed Phreds are not the FASTQ's own");

    for p in [plain1, plain2, gz1, gz2] {
        std::fs::remove_file(p).ok();
    }
}
