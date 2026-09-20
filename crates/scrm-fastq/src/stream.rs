use std::path::Path;
use std::sync::Mutex;

use paraseq::prelude::*;
use paraseq::{ProcessError, fastx};
use scrm_core::ParsedPair;

use crate::{Counters, Parser, RawRead, decode};

#[derive(Debug, thiserror::Error)]
pub enum StreamError {
    #[error(transparent)]
    Pipeline(#[from] scrm_core::Error),

    #[error("cannot open {path}: {source}")]
    Open {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error(transparent)]
    Fastx(#[from] paraseq::Error),

    #[error(transparent)]
    Reader(ProcessError),
}

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub threads: usize,
    pub decoder_threads: usize,
    pub limit: Option<usize>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        Self {
            threads: cores,
            decoder_threads: cores,
            limit: None,
        }
    }
}

pub fn stream_pairs<S>(
    r1_path: &Path,
    r2_path: &Path,
    parser: &Parser,
    config: &StreamConfig,
    sink: S,
) -> Result<Counters, StreamError>
where
    S: Fn(ParsedPair) + Send + Sync,
{
    let r1 = fastx::Reader::new(open(r1_path, config.decoder_threads)?)?;
    let r2 = fastx::Reader::new(open(r2_path, config.decoder_threads)?)?;

    let shared = Mutex::new(Counters::default());
    let mut worker = Worker {
        parser,
        sink: &sink,
        counters: Counters::default(),
        shared: &shared,
    };

    let outcome = match config.limit {
        Some(limit) => r1.process_parallel_paired_range(r2, &mut worker, config.threads, 0..limit),
        None => r1.process_parallel_paired(r2, &mut worker, config.threads),
    };
    outcome.map_err(unwrap_pipeline_error)?;

    Ok(shared.into_inner().expect("counter mutex poisoned"))
}

fn open(path: &Path, decoder_threads: usize) -> Result<decode::Source, StreamError> {
    decode::open(path, decoder_threads).map_err(|source| StreamError::Open {
        path: path.display().to_string(),
        source,
    })
}

fn unwrap_pipeline_error(err: ProcessError) -> StreamError {
    match err {
        ProcessError::Process(boxed) => match boxed.downcast::<scrm_core::Error>() {
            Ok(ours) => StreamError::Pipeline(*ours),
            Err(other) => StreamError::Reader(ProcessError::Process(other)),
        },
        other => StreamError::Reader(other),
    }
}

struct Worker<'p, S> {
    parser: &'p Parser,
    sink: &'p S,
    counters: Counters,
    shared: &'p Mutex<Counters>,
}

impl<S> Clone for Worker<'_, S> {
    fn clone(&self) -> Self {
        Self {
            parser: self.parser,
            sink: self.sink,
            counters: Counters::default(),
            shared: self.shared,
        }
    }
}

impl<Rf: Record, S> PairedParallelProcessor<Rf> for Worker<'_, S>
where
    S: Fn(ParsedPair) + Send + Sync,
{
    fn process_record_pair(&mut self, r1: Rf, r2: Rf) -> Result<(), ProcessError> {
        let r1_qual = r1.qual().unwrap_or(b"");
        let r2_qual = r2.qual().unwrap_or(b"");
        let outcome = self.parser.parse_pair(
            r1.index(),
            RawRead::new(r1.id(), r1.seq_raw(), r1_qual),
            RawRead::new(r2.id(), r2.seq_raw(), r2_qual),
            &mut self.counters,
        );
        match outcome {
            Ok(crate::Outcome::Kept(pair)) => {
                (self.sink)(pair);
                Ok(())
            }
            Ok(crate::Outcome::Dropped(_)) => Ok(()),
            Err(e) => Err(e.into_process_error()),
        }
    }

    fn on_thread_complete(&mut self) -> Result<(), ProcessError> {
        let mut shared = self.shared.lock().expect("counter mutex poisoned");
        shared.merge(&self.counters);
        self.counters = Counters::default();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::sync::Mutex;

    use scrm_trim::TrimSettings;

    use super::*;

    const CB: &[u8] = b"ACGTACGTACGTACGT";
    const UMI: &[u8] = b"TTTTGGGGCCCC";
    const CDNA: &[u8] = b"GCTAGCTAGGCATCGATCGGATCCAAGCTTGGATCCTAGCTAGCTAGGCATCGATCGGAT";

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("scrm-stream-{}-{name}", std::process::id()));
        p
    }

    type Records = Vec<(String, Vec<u8>)>;

    fn write_fastq(path: &Path, records: &[(String, Vec<u8>)]) {
        let mut f = std::fs::File::create(path).unwrap();
        for (id, seq) in records {
            let qual = vec![b'I'; seq.len()];
            f.write_all(format!("@{id}\n").as_bytes()).unwrap();
            f.write_all(seq).unwrap();
            f.write_all(b"\n+\n").unwrap();
            f.write_all(&qual).unwrap();
            f.write_all(b"\n").unwrap();
        }
    }

    fn r1_seq() -> Vec<u8> {
        let mut v = CB.to_vec();
        v.extend_from_slice(UMI);
        v
    }

    fn run(
        r1: &[(String, Vec<u8>)],
        r2: &[(String, Vec<u8>)],
        threads: usize,
        tag: &str,
    ) -> Result<(Counters, Vec<ParsedPair>), StreamError> {
        let p1 = tmp(&format!("{tag}.r1.fastq"));
        let p2 = tmp(&format!("{tag}.r2.fastq"));
        write_fastq(&p1, r1);
        write_fastq(&p2, r2);
        let parser = Parser::new("1{b[16]u[12]}2{r:}", TrimSettings::default(), |cb| {
            Some(scrm_core::CellId(cb))
        })
        .unwrap();
        let collected = Mutex::new(Vec::new());
        let config = StreamConfig {
            threads,
            decoder_threads: 1,
            limit: None,
        };
        let result = stream_pairs(&p1, &p2, &parser, &config, |pair| collected.lock().unwrap().push(pair));
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
        result.map(|c| (c, collected.into_inner().unwrap()))
    }

    fn pairs(n: usize) -> (Records, Records) {
        let r1 = (0..n).map(|i| (format!("READ{i}"), r1_seq())).collect();
        let r2 = (0..n).map(|i| (format!("READ{i}"), CDNA.to_vec())).collect();
        (r1, r2)
    }

    #[test]
    fn a_matched_pair_of_files_streams_through() {
        for threads in [1, 4] {
            let (r1, r2) = pairs(50);
            let (counters, collected) = run(&r1, &r2, threads, "ok").unwrap();
            assert_eq!(counters.pairs_in, 50, "threads {threads}");
            assert_eq!(counters.kept, 50, "threads {threads}");
            assert_eq!(collected.len(), 50, "threads {threads}");
            assert!(counters.balanced());
            assert_eq!(counters.survival(), 1.0);
        }
    }

    #[test]
    fn counts_do_not_scale_with_thread_count() {
        let (r1, r2) = pairs(200);
        let one = run(&r1, &r2, 1, "t1").unwrap().0;
        let many = run(&r1, &r2, 8, "t8").unwrap().0;
        assert_eq!(one.pairs_in, many.pairs_in);
        assert_eq!(one.kept, many.kept);
        assert_eq!(one.artifacts.reads, many.artifacts.reads);
    }

    #[test]
    fn a_swapped_file_fails_the_id_assert() {
        let (r1, mut r2) = pairs(20);
        r2.reverse();
        for threads in [1, 4] {
            let err = run(&r1, &r2, threads, "swap").unwrap_err();
            match err {
                StreamError::Pipeline(scrm_core::Error::PairIdMismatch { r1, r2, .. }) => {
                    assert_ne!(r1, r2);
                }
                other => panic!("expected PairIdMismatch, got {other:?}"),
            }
        }
    }

    #[test]
    fn a_gzipped_pair_streams_through() {
        let p1 = tmp("gz.r1.fastq.gz");
        let p2 = tmp("gz.r2.fastq.gz");
        for (path, seq) in [(&p1, r1_seq()), (&p2, CDNA.to_vec())] {
            let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            for i in 0..30 {
                let qual = vec![b'I'; seq.len()];
                enc.write_all(format!("@READ{i}\n").as_bytes()).unwrap();
                enc.write_all(&seq).unwrap();
                enc.write_all(b"\n+\n").unwrap();
                enc.write_all(&qual).unwrap();
                enc.write_all(b"\n").unwrap();
            }
            std::fs::write(path, enc.finish().unwrap()).unwrap();
        }
        let parser = Parser::new("1{b[16]u[12]}2{r:}", TrimSettings::default(), |cb| {
            Some(scrm_core::CellId(cb))
        })
        .unwrap();
        let counted = std::sync::atomic::AtomicUsize::new(0);
        let counters = stream_pairs(&p1, &p2, &parser, &StreamConfig::default(), |_| {
            counted.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        })
        .unwrap();
        assert_eq!(counters.kept, 30);
        assert_eq!(counted.into_inner(), 30);
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }

    #[test]
    fn the_limit_stops_early() {
        let (r1, r2) = pairs(500);
        let p1 = tmp("lim.r1.fastq");
        let p2 = tmp("lim.r2.fastq");
        write_fastq(&p1, &r1);
        write_fastq(&p2, &r2);
        let parser = Parser::new("1{b[16]u[12]}2{r:}", TrimSettings::default(), |cb| {
            Some(scrm_core::CellId(cb))
        })
        .unwrap();
        let config = StreamConfig {
            threads: 2,
            decoder_threads: 1,
            limit: Some(100),
        };
        let counters = stream_pairs(&p1, &p2, &parser, &config, |_| {}).unwrap();
        assert_eq!(counters.pairs_in, 100);
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }

    #[test]
    fn unequal_file_lengths_are_reported() {
        let (r1, _) = pairs(40);
        let (_, r2) = pairs(20);
        let err = run(&r1, &r2, 2, "len").unwrap_err();
        assert!(matches!(err, StreamError::Reader(_)), "{err:?}");
    }
}
