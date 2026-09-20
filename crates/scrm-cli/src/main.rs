mod collate;
mod samplesheet;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Parser as ClapParser;
use rayon::prelude::*;
use scrm_asm::{Retention, StatsCollector};
use scrm_chem::Chemistry;
use scrm_core::stamp::{Declared, Measured};
use scrm_corpus::CorpusWriter;
use scrm_correct::Corrector;
use scrm_fastq::{Parser, StreamConfig, decode, stream_pairs};
use scrm_trim::TrimSettings;
use scrm_umi::{UmiStatsAccum, dedup_cell};

use collate::Collator;

#[derive(ClapParser, Debug)]
#[command(name = "scrm-corpus", about = "Build an scRM Arrow corpus from a 10x FASTQ pair")]
struct Args {
    #[arg(long)]
    samplesheet: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    threads: usize,
    #[arg(long, default_value_t = scrm_core::K_DEFAULT)]
    k: usize,
    #[arg(long, default_value_t = scrm_core::MIN_UMI_LEN_DEFAULT)]
    min_umi_len: usize,
    #[arg(long, default_value_t = scrm_core::MAX_UMI_LEN_DEFAULT)]
    max_umi_len: usize,
    #[arg(long, default_value_t = scrm_core::MIN_CELL_MOLECULES_DEFAULT)]
    min_cell_molecules: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let t_all = Instant::now();

    scrm_core::validate_k(args.k).map_err(|e| anyhow::anyhow!("--k: {e}"))?;
    scrm_core::LenFilter::new(args.min_umi_len, args.max_umi_len).map_err(|e| anyhow::anyhow!("{e}"))?;

    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let threads = match args.threads {
        0 => cores,
        n if n > cores => {
            eprintln!("threads       {n} requested but only {cores} available -- capped at {cores}");
            cores
        }
        n => n,
    };
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .context("sizing the rayon pool")?;

    let experiments = samplesheet::read(&args.samplesheet)?;
    eprintln!(
        "samplesheet   {} -> {} experiment(s): {}",
        args.samplesheet.display(),
        experiments.len(),
        experiments
            .iter()
            .map(|e| e.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    for exp in &experiments {
        scrm_corpus::refuse_if_corpus(&args.out.join(&exp.name))?;
    }

    for (i, exp) in experiments.iter().enumerate() {
        eprintln!("\n=== [{}/{}] {} ===", i + 1, experiments.len(), exp.name);
        let out = args.out.join(&exp.name);
        build_experiment(exp, &args, &out, threads).with_context(|| format!("building experiment {:?}", exp.name))?;
    }
    if experiments.len() > 1 {
        eprintln!(
            "\nall {} experiments built in {:.1}s",
            experiments.len(),
            t_all.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn build_experiment(exp: &samplesheet::Experiment, args: &Args, out: &std::path::Path, threads: usize) -> Result<()> {
    let t_start = Instant::now();
    let (r1, r2) = (exp.r1(), exp.r2());
    let n_lanes = r1.len();
    let min_trim_len = scrm_core::min_trim_len_for(args.k);

    let len_filter_proto =
        scrm_core::LenFilter::new(args.min_umi_len, args.max_umi_len).map_err(|e| anyhow::anyhow!("{e}"))?;

    let sample = exp.name.clone();

    for path in r1.iter().chain(&r2).chain(std::iter::once(&exp.barcodes)) {
        std::fs::File::open(path).with_context(|| format!("cannot read input {}", path.display()))?;
    }

    anyhow::ensure!(
        r1.len() == r2.len(),
        "{} --r1 and {} --r2: lanes must pair up",
        r1.len(),
        r2.len()
    );

    let chem = Chemistry::new(&exp.geometry)?;
    let cb_len = chem.cb_len();
    let umi_len = chem.umi_len();
    eprintln!("geometry      {}  cb_len {cb_len}  umi_len {umi_len}", chem.geometry());
    eprintln!("assembly      k {}  trim floor {min_trim_len}", args.k);

    let corrector = Arc::new(
        Corrector::from_filtered_matrix(&exp.barcodes, cb_len)
            .with_context(|| format!("barcode list {}", exp.barcodes.display()))?,
    );
    eprintln!(
        "barcodes      {} cells  sha256 {}",
        corrector.n_barcodes,
        &corrector.list_sha256[..16]
    );

    let settings = TrimSettings {
        min_len: min_trim_len,
        ..TrimSettings::default()
    };

    // ONLY WHAT THIS BINARY ACTS ON -- which is now all four steps, so every one of these
    // parameters is applied. They were dropped from the stamp while trimming was a pre-step,
    // because recording a parameter nothing applied reads downstream as evidence that it was; the
    // same rule puts them back now that `Trimmer::trim` uses them.
    let trim_stamp = serde_json::json!({
        "decode_backend": decode::backend(),
        "phred_offset": settings.phred_offset,
        "quality_cutoff": settings.quality_cutoff,
        "error_rate": settings.error_rate,
        "stringency": settings.stringency,
        "adapters": settings.adapters.iter()
            .map(|(name, seq)| serde_json::json!({
                "name": name, "seq": String::from_utf8_lossy(seq),
            }))
            .collect::<Vec<_>>(),
        "tso": String::from_utf8_lossy(&settings.tso),
        "tso_min_score": settings.tso_min_score,
        "homopolymer_min_run": settings.homopolymer_min_run,
        "min_len": settings.min_len,
        "trimmed_here": true,
    });

    let n_exact = Arc::new(AtomicU64::new(0));
    let n_corrected = Arc::new(AtomicU64::new(0));
    let n_ambiguous = Arc::new(AtomicU64::new(0));
    let n_nomatch = Arc::new(AtomicU64::new(0));
    let parser = {
        let (c, e, k, a, m) = (
            Arc::clone(&corrector),
            Arc::clone(&n_exact),
            Arc::clone(&n_corrected),
            Arc::clone(&n_ambiguous),
            Arc::clone(&n_nomatch),
        );
        Parser::new(&exp.geometry, settings, move |cb| {
            let outcome = c.correct(cb);
            match outcome {
                scrm_correct::Correction::Exact(_) => &e,
                scrm_correct::Correction::Corrected(_) => &k,
                scrm_correct::Correction::Ambiguous => &a,
                scrm_correct::Correction::NoMatch => &m,
            }
            .fetch_add(1, Ordering::Relaxed);
            outcome.cell()
        })
        .context("building parser")?
    };

    let t_stream = Instant::now();
    let collator = Collator::new(corrector.n_barcodes)?;
    let cfg = StreamConfig {
        threads,
        decoder_threads: threads,
        limit: None,
    };
    eprintln!(
        "threads       {threads} workers, {} decoder threads/file, backend {}",
        cfg.decoder_threads,
        decode::backend()
    );

    let spill_err: std::sync::Mutex<Option<anyhow::Error>> = std::sync::Mutex::new(None);
    let mut counters = scrm_fastq::Counters::default();
    for (lane, (r1, r2)) in r1.iter().zip(&r2).enumerate() {
        let lane_counters = stream_pairs(r1, r2, &parser, &cfg, |pair| match corrector.rank(pair.cb) {
            Some(rank) => {
                if let Err(e) = collator.push(rank, &pair) {
                    *spill_err.lock().expect("spill err") = Some(e);
                }
            }
            None => {
                *spill_err.lock().expect("spill err") = Some(anyhow::anyhow!(
                    "corrected barcode {} is not in the list it was corrected against",
                    String::from_utf8_lossy(&corrector.render(pair.cb))
                ));
            }
        })
        .with_context(|| format!("streaming lane {} (R1 {}, R2 {})", lane + 1, r1.display(), r2.display()))?;
        eprintln!(
            "lane {}/{}      {} pairs in, {} kept",
            lane + 1,
            n_lanes,
            lane_counters.pairs_in,
            lane_counters.kept
        );
        counters.merge(&lane_counters);
    }
    anyhow::ensure!(
        counters.balanced(),
        "the pair ledger does not balance: {} pairs in, but kept {} + malformed {} + geometry {} \
         + N in CB/UMI {} + gated {} + short {} + malformed on trim {} + N in R2 {} = {}",
        counters.pairs_in,
        counters.kept,
        counters.dropped_malformed_record,
        counters.dropped_geometry,
        counters.dropped_n_cbumi,
        counters.dropped_gated,
        counters.dropped_short,
        counters.dropped_malformed_trim,
        counters.dropped_n_read,
        counters.kept
            + counters.dropped_malformed_record
            + counters.dropped_geometry
            + counters.dropped_n_cbumi
            + counters.dropped_gated
            + counters.dropped_short
            + counters.dropped_malformed_trim
            + counters.dropped_n_read
    );
    if let Some(e) = spill_err.into_inner().expect("spill err") {
        return Err(e).context("spilling reads");
    }
    let shards = collator.seal()?;
    let secs_stream = t_stream.elapsed().as_secs_f64();

    let cb_stats = scrm_correct::CorrectionStats {
        exact: n_exact.load(Ordering::Relaxed),
        corrected: n_corrected.load(Ordering::Relaxed),
        ambiguous: n_ambiguous.load(Ordering::Relaxed),
        no_match: n_nomatch.load(Ordering::Relaxed),
    };
    eprintln!(
        "pairs         {} in, {} kept by parse ({:.3}%), {} dropped at CB",
        counters.pairs_in,
        counters.kept,
        100.0 * counters.kept as f64 / counters.pairs_in.max(1) as f64,
        cb_stats.dropped()
    );
    eprintln!(
        "spilled       {:.2} GiB across {} shards",
        shards.spilled_bytes() as f64 / (1u64 << 30) as f64,
        shards.len()
    );

    let declared = build_declared(exp, args, &chem, &corrector, trim_stamp)?;
    std::fs::create_dir_all(out).with_context(|| format!("creating corpus dir {}", out.display()))?;

    let t_asm = Instant::now();
    type ShardOut = (
        Option<scrm_corpus::ShardEntry>,
        UmiStatsAccum,
        StatsCollector,
        usize,
        scrm_core::LenFilter,
        (u64, u64, u64),
    );

    let kmer_len = args.k;
    let per_shard: Vec<Result<ShardOut>> = (0..shards.len())
        .into_par_iter()
        .map(|shard| -> Result<ShardOut> {
            let empty = || {
                (
                    None,
                    UmiStatsAccum::new(),
                    StatsCollector::default(),
                    0usize,
                    len_filter_proto.clone(),
                    (0, 0, 0),
                )
            };
            let cells = shards.load(shard).with_context(|| format!("loading shard {shard}"))?;
            if cells.is_empty() {
                return Ok(empty());
            }

            let file = format!("{sample}-shard-{shard:03}.arrow");
            let mut writer =
                CorpusWriter::create(&out.join(&file), &declared).with_context(|| format!("creating {file}"))?;

            let mut umi = UmiStatsAccum::new();
            let mut asm = StatsCollector::default();
            let mut len_filter = len_filter_proto.clone();
            let mut cells_written = 0usize;
            let mut shallow_cells = 0u64;
            let mut shallow_molecules = 0u64;
            let mut emptied_cells = 0u64;
            let mut cell_buf: Vec<scrm_core::MoleculeContigs> = Vec::new();
            for (cell, pairs) in cells {
                let cell_id = String::from_utf8(corrector.render(cell)).context("cell id is not UTF-8")?;
                let deduped = dedup_cell(cell, pairs, umi_len as u8).context("dedup")?;
                umi.add_cell(&deduped.stats);
                cell_buf.clear();
                for molecule in &deduped.molecules {
                    let (mc, diag) = scrm_asm::assemble(molecule, scrm_asm::default_model(), kmer_len);
                    let kept = len_filter.keep(mc.tokens.len());
                    asm.record(
                        molecule,
                        &mc,
                        &diag,
                        if kept {
                            Retention::Kept
                        } else {
                            Retention::LengthFiltered
                        },
                    );
                    if !kept {
                        continue;
                    }
                    cell_buf.push(mc);
                }
                if cell_buf.len() < args.min_cell_molecules {
                    if cell_buf.is_empty() {
                        emptied_cells += 1;
                    } else {
                        shallow_cells += 1;
                        shallow_molecules += cell_buf.len() as u64;
                    }
                    continue;
                }
                for mc in &cell_buf {
                    writer.push(&cell_id, mc).context("writing row")?;
                }
                cells_written += 1;
            }
            let summary = writer.finish().with_context(|| format!("finishing {file}"))?;
            if summary.n_rows == 0 {
                std::fs::remove_file(out.join(&file)).with_context(|| format!("removing empty shard {file}"))?;
                return Ok((
                    None,
                    umi,
                    asm,
                    cells_written,
                    len_filter,
                    (shallow_cells, shallow_molecules, emptied_cells),
                ));
            }
            Ok((
                Some(scrm_corpus::ShardEntry {
                    file,
                    n_rows: summary.n_rows,
                    n_tokens: summary.n_tokens,
                    tokens_sha256: summary.tokens_sha256,
                    first_cell: summary.first_cell,
                    last_cell: summary.last_cell,
                }),
                umi,
                asm,
                cells_written,
                len_filter,
                (shallow_cells, shallow_molecules, emptied_cells),
            ))
        })
        .collect();

    let mut entries = Vec::new();
    let mut umi_acc = UmiStatsAccum::new();
    let mut asm_stats = StatsCollector::default();
    let mut n_cells_written = 0usize;
    let mut molecules_out = 0u64;
    let mut tokens_out = 0u64;
    let mut len_filter = len_filter_proto.clone();
    let mut shallow_cells = 0u64;
    let mut shallow_molecules = 0u64;
    let mut emptied_cells = 0u64;
    for result in per_shard {
        let (entry, umi, asm, cells_written, shard_filter, (shallow_c, shallow_m, emptied_c)) = result?;
        shallow_cells += shallow_c;
        shallow_molecules += shallow_m;
        emptied_cells += emptied_c;
        n_cells_written += cells_written;
        umi_acc.merge(&umi);
        asm_stats.merge(&asm);
        len_filter.merge(&shard_filter);
        if let Some(entry) = entry {
            molecules_out += entry.n_rows as u64;
            tokens_out += entry.n_tokens;
            entries.push(entry);
        }
    }
    if entries.is_empty() {
        let cause = if len_filter.n_dropped() == 0 && shallow_cells > 0 {
            format!(
                "every cell fell below --min-cell-molecules {}: {} cells dropped, holding {} \
                 molecules between them. The molecule bounds dropped nothing.",
                args.min_cell_molecules, shallow_cells, shallow_molecules
            )
        } else if shallow_cells == 0 && len_filter.n_dropped() > 0 {
            format!(
                "the molecule bounds [{}, {}] dropped all {} assembled molecules ({} below the \
                 floor, {} above the cap). Check the bounds against the read length.",
                args.min_umi_len,
                args.max_umi_len,
                len_filter.n_dropped(),
                len_filter.n_short,
                len_filter.n_long
            )
        } else {
            format!(
                "the molecule bounds [{}, {}] dropped {} molecules ({} short, {} long) and \
                 --min-cell-molecules {} then dropped {} cells holding {} molecules.",
                args.min_umi_len,
                args.max_umi_len,
                len_filter.n_dropped(),
                len_filter.n_short,
                len_filter.n_long,
                args.min_cell_molecules,
                shallow_cells,
                shallow_molecules
            )
        };
        anyhow::bail!("there is no corpus to write: {cause}");
    }
    let secs_asm = t_asm.elapsed().as_secs_f64();
    eprintln!("shards        {} written in parallel", entries.len());
    eprintln!("cells         {n_cells_written} written to the corpus");
    eprintln!("molecules     {molecules_out}");

    let measured = build_measured(
        &counters,
        &cb_stats,
        &corrector,
        threads,
        &umi_acc,
        &asm_stats,
        &len_filter,
        (shallow_cells, shallow_molecules, emptied_cells),
        molecules_out,
        tokens_out,
        secs_stream,
        secs_asm,
    )?;
    let u = &measured.umi_stats;
    eprintln!(
        "umi guard     mean H1 degree {:.4} (percolates at {:.1})  largest group {} UMIs (subsampled ref {})  chain depth {} (subsampled ref {})",
        u.mean_hamming1_degree,
        scrm_umi::PERCOLATION_DEGREE,
        u.max_group_umis,
        scrm_umi::OBSERVED_MAX_GROUP_UMIS,
        u.max_chain_depth,
        scrm_umi::OBSERVED_MAX_CHAIN_DEPTH,
    );
    eprintln!(
        "              merge rate {:.2}%  max group {:.4}% of a {}-UMI cell",
        u.merge_rate * 100.0,
        u.max_group_fraction * 100.0,
        u.max_group_fraction_umis,
    );
    if let Some(warning) = scrm_umi::percolation_warning(u) {
        eprintln!("\nUMI PERCOLATION WARNING");
        eprintln!("              {warning}");
        eprintln!("              1MM_CR_flow is transitive, so it has no combinatorial bound on");
        eprintln!("              group size. The corpus is what the rule produces; whether that");
        eprintln!("              molecule definition is acceptable here is a science decision.");
    }

    let manifest = scrm_corpus::Manifest::new(declared, measured, entries);
    manifest.write(out).context("writing manifest")?;

    let stats = scrm_corpus::verify_corpus(out).context("verifying the corpus we just wrote")?;
    anyhow::ensure!(
        stats.n_tokens == tokens_out,
        "manifest says {tokens_out} tokens, the shards hold {}",
        stats.n_tokens
    );

    let wall = t_start.elapsed().as_secs_f64();
    eprintln!("\ncorpus        {}", out.display());
    eprintln!("rows          {}  tokens {}", stats.n_rows, stats.n_tokens);
    eprintln!("sha256 root   {}", stats.root);
    eprintln!("verify        OK");
    eprintln!("timing        stream {secs_stream:.1}s  assemble+write {secs_asm:.1}s  total {wall:.1}s");
    eprintln!(
        "throughput    {:.0} pairs/s  {:.0} molecules/s",
        counters.pairs_in as f64 / wall,
        molecules_out as f64 / wall
    );
    Ok(())
}

fn build_declared(
    exp: &samplesheet::Experiment,
    args: &Args,
    chem: &Chemistry,
    corrector: &Corrector,
    trim_stamp: serde_json::Value,
) -> Result<Declared> {
    use scrm_core::stamp::*;

    let git = |a: &[&str]| -> String {
        std::process::Command::new("git")
            .args(a)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_default()
    };
    let sha = git(&["rev-parse", "HEAD"]);
    let dirty = !git(&["status", "--porcelain"]).is_empty();

    let inputs = exp
        .lanes
        .iter()
        .flat_map(|(a, b)| [a, b])
        .chain(std::iter::once(&exp.barcodes))
        .map(|p| {
            let bytes = std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
            InputFile {
                path: p.display().to_string(),
                bytes,
                sha256: String::new(),
            }
        })
        .collect();

    Ok(Declared {
        build: Build {
            scrm_sha: sha,
            dirty,
            built_at_utc: env!("SCRM_BUILT_AT_UTC").into(),
        },
        config: Config {
            geometry: chem.geometry().to_string(),
            cb_len: chem.cb_len() as u8,
            umi_len: chem.umi_len() as u8,
            whitelist_sha256: corrector.list_sha256.clone(),
            alphabet_size: scrm_core::ALPHABET_SIZE,
            dedup_rule: "1MM_CR_flow".into(),
            k: args.k,
            min_trim_len: scrm_core::min_trim_len_for(args.k),
            trim: trim_stamp,
            read_policy: "R1: CB+UMI only, byte-identical".into(),
            bubble_resolution: scrm_asm::resolution_name(),
            max_windows_per_cell: 0,
            min_umi_len: args.min_umi_len,
            max_umi_len: args.max_umi_len,
            min_cell_molecules: args.min_cell_molecules,
        },
        inputs,
        tools: Tools {
            rustc: env!("SCRM_RUSTC_VERSION").into(),
            paraseq: env!("SCRM_VERSION_paraseq_temp").into(),
            rapidgzip_core: env!("SCRM_VERSION_rapidgzip_core").into(),
            thread_broker: "unused".into(),
            libradicl: env!("SCRM_VERSION_libradicl").into(),
            seq_geom_parser: env!("SCRM_VERSION_seq_geom_parser").into(),
            debruijn: env!("SCRM_VERSION_debruijn").into(),
            arrow: env!("SCRM_VERSION_arrow").into(),
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn build_measured(
    counters: &scrm_fastq::Counters,
    cb_stats: &scrm_correct::CorrectionStats,
    corrector: &Corrector,
    threads: usize,
    umi_acc: &UmiStatsAccum,
    asm_stats: &StatsCollector,
    len_filter: &scrm_core::LenFilter,
    shallow: (u64, u64, u64),
    molecules_out: u64,
    tokens_out: u64,
    secs_stream: f64,
    secs_asm: f64,
) -> Result<Measured> {
    use scrm_core::stamp::*;

    Ok(Measured {
        counts: Counts {
            pairs_in: counters.pairs_in,
            dropped_n: counters.dropped_n(),
            dropped_short: counters.dropped_short,
            dropped_cb: {
                let (gate, parser) = (cb_stats.dropped(), counters.dropped_gated);
                anyhow::ensure!(
                    gate == parser,
                    "gate rejected {gate} barcodes but the parser recorded {parser} gated drops"
                );
                gate
            },
            cells_called: corrector.n_barcodes as u64,
            dropped_umi_short: len_filter.n_short,
            dropped_umi_short_tokens: len_filter.tokens_short,
            dropped_umi_long: len_filter.n_long,
            dropped_umi_long_tokens: len_filter.tokens_long,
            dropped_cells_shallow: shallow.0,
            dropped_cells_molecules: shallow.1,
            dropped_cells_emptied: shallow.2,
            molecules_out,
            tokens_out,
        },
        umi_stats: umi_acc.finish(),
        assembly_stats: asm_stats.finish(),
        artifact_rates: counters.artifacts.rates(),
        throughput: Throughput {
            read_pairs_per_sec: counters.pairs_in as f64 / secs_stream.max(1e-9),
            molecules_per_sec: molecules_out as f64 / secs_asm.max(1e-9),
            secs_stream,
            secs_assemble_write: secs_asm,
            n_threads: threads as u32,
        },
    })
}
