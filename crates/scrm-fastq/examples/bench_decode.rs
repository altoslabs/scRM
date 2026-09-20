use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use scrm_fastq::decode;

fn cpu_secs() -> f64 {
    let s = std::fs::read_to_string("/proc/self/stat").unwrap_or_default();
    let rest = s.rsplit_once(')').map(|(_, r)| r).unwrap_or("");
    let f: Vec<&str> = rest.split_whitespace().collect();
    let ticks = 100.0;
    let u: f64 = f.get(11).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let k: f64 = f.get(12).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    (u + k) / ticks
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().unwrap_or_else(|| "raw".into());
    let r1 = PathBuf::from(
        args.next()
            .expect("usage: bench_decode MODE R1 R2 [threads] [dec_threads] [limit]"),
    );
    let r2 = PathBuf::from(args.next().expect("R2 required"));
    let threads: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(4);
    let dec: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(4);
    let limit: usize = args.next().map(|s| s.parse().unwrap()).unwrap_or(10_000_000);

    println!("backend        {}", decode::backend());
    println!("mode           {mode}");
    println!(
        "threads        {threads}   decoder_threads {dec}   (cores {})",
        num_cpus()
    );

    let t0 = Instant::now();
    let c0 = cpu_secs();

    let (bytes, pairs) = match mode.as_str() {
        "raw" => {
            let (a, b) = std::thread::scope(|s| {
                let h1 = s.spawn(|| drain(&r1, dec));
                let h2 = s.spawn(|| drain(&r2, dec));
                (h1.join().unwrap(), h2.join().unwrap())
            });
            (a? + b?, 0u64)
        }
        "pipe" => {
            use scrm_fastq::{Parser, StreamConfig, stream_pairs};
            use scrm_trim::TrimSettings;
            let settings = TrimSettings::default();
            let parser = Parser::new("1{b[16]u[12]}2{r:}", settings, |cb| Some(scrm_core::CellId(cb)))?;
            let cfg = StreamConfig {
                threads,
                decoder_threads: dec,
                limit: Some(limit),
            };
            let counters = stream_pairs(&r1, &r2, &parser, &cfg, |_| {})?;
            (0, counters.pairs_in)
        }
        other => return Err(format!("unknown mode {other}").into()),
    };

    let wall = t0.elapsed().as_secs_f64();
    let cpu = cpu_secs() - c0;
    println!("wall           {wall:.2} s");
    println!("cpu            {cpu:.2} s   ({:.2}x wall)", cpu / wall.max(1e-9));
    if bytes > 0 {
        println!("decoded        {:.2} GiB", bytes as f64 / (1u64 << 30) as f64);
        println!(
            "throughput     {:.0} MiB/s decompressed",
            bytes as f64 / (1u64 << 20) as f64 / wall
        );
    }
    if pairs > 0 {
        println!("pairs          {pairs}");
        println!("throughput     {:.0} pairs/s", pairs as f64 / wall);
    }
    Ok(())
}

fn drain(path: &std::path::Path, dec: usize) -> std::io::Result<u64> {
    let mut r = decode::open(path, dec)?;
    let mut buf = vec![0u8; 1 << 20];
    let mut total = 0u64;
    loop {
        match r.read(&mut buf)? {
            0 => break,
            n => total += n as u64,
        }
    }
    Ok(total)
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}
