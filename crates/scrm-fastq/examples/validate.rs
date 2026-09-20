use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use scrm_fastq::{Parser, StreamConfig, stream_pairs};
use scrm_trim::TrimSettings;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let r1 = PathBuf::from(args.next().expect("usage: validate R1 R2 [geometry] [limit] [threads]"));
    let r2 = PathBuf::from(args.next().expect("R2 path required"));
    let geometry = args.next().unwrap_or_else(|| "1{b[16]u[12]}2{r:}".to_string());
    let limit: usize = args
        .next()
        .map(|s| s.parse().expect("limit must be a number"))
        .unwrap_or(100_000);
    let threads: usize = args
        .next()
        .map(|s| s.parse().expect("threads must be a number"))
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));

    let settings = TrimSettings::default();
    let parser = Parser::new(&geometry, settings, |cb| Some(scrm_core::CellId(cb)))?;

    let bases = AtomicU64::new(0);
    let shortest = AtomicU64::new(u64::MAX);
    let longest = AtomicU64::new(0);
    let config = StreamConfig {
        threads,
        decoder_threads: threads,
        limit: Some(limit),
    };

    let start = Instant::now();
    let counters = stream_pairs(&r1, &r2, &parser, &config, |pair| {
        let len = pair.r2.len() as u64;
        bases.fetch_add(len, Ordering::Relaxed);
        shortest.fetch_min(len, Ordering::Relaxed);
        longest.fetch_max(len, Ordering::Relaxed);
    })?;
    let secs = start.elapsed().as_secs_f64();

    let kept = counters.kept.max(1);
    println!("geometry            {geometry}");
    println!(
        "cb_len / umi_len    {} / {}",
        parser.chemistry().cb_len(),
        parser.chemistry().umi_len()
    );
    println!(
        "adapter             {} (trimmed in process by scrm-trim)",
        String::from_utf8_lossy(&parser.trimmer().settings().adapters[0].1)
    );
    println!("decode backend      {}", scrm_fastq::decode::backend());
    println!("threads             {threads}");
    println!();
    println!("pairs_in            {}", counters.pairs_in);
    println!(
        "kept                {}  ({:.3}% survival)",
        counters.kept,
        100.0 * counters.survival()
    );
    println!(
        "dropped_geometry    {}  ({:.3}% of pairs_in {})",
        counters.dropped_geometry,
        pct(counters.dropped_geometry, counters.pairs_in),
        counters.pairs_in
    );
    println!(
        "dropped_malformed   {} record + {} on trim",
        counters.dropped_malformed_record, counters.dropped_malformed_trim
    );
    println!(
        "dropped_n_cbumi     {}  ({:.3}% of geometry-accepted {})",
        counters.dropped_n_cbumi,
        pct(counters.dropped_n_cbumi, counters.geometry_accepted()),
        counters.geometry_accepted()
    );
    println!(
        "dropped_gated       {}  ({:.3}% of gate-offered {})",
        counters.dropped_gated,
        pct(counters.dropped_gated, counters.gate_offered()),
        counters.gate_offered()
    );
    println!(
        "dropped_short       {}  ({:.3}% of trimmed {})",
        counters.dropped_short,
        pct(counters.dropped_short, counters.trimmed()),
        counters.trimmed()
    );
    println!(
        "dropped_n_read      {}  ({:.3}% of trimmed {})",
        counters.dropped_n_read,
        pct(counters.dropped_n_read, counters.trimmed()),
        counters.trimmed()
    );
    println!("balanced            {}", counters.balanced());
    println!(
        "r2 length           mean {:.1}, min {}, max {}",
        bases.load(Ordering::Relaxed) as f64 / kept as f64,
        shortest.load(Ordering::Relaxed),
        longest.load(Ordering::Relaxed)
    );
    println!(
        "throughput          {:.0} pairs/s ({:.2} s wall)",
        counters.pairs_in as f64 / secs,
        secs
    );

    let rates = counters.artifacts.rates();
    println!();
    println!("adapter_fraction    {:.4}%", 100.0 * rates.adapter_fraction);
    println!("tso_fraction        {:.4}%", 100.0 * rates.tso_fraction);
    println!("homopolymer (nonzero only):");
    for (key, value) in &rates.homopolymer {
        if *value > 0.0 {
            println!("  {key:<26} {:.4}%", 100.0 * value);
        }
    }
    Ok(())
}

fn pct(n: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        100.0 * n as f64 / total as f64
    }
}
