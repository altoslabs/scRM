use std::path::PathBuf;
use std::time::Instant;

use scrm_pack::{BatchSpec, Corpus, build_batch};

const SEQ_LEN: usize = 16384;
const B: usize = 128;
const V: usize = 2;
const TOKENS_PER_STEP: usize = B * V * SEQ_LEN;

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("pass a corpus path, or set SCRM_CORPUS_ROOT"));
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("thread pool");
    }
    let n_threads = rayon::current_num_threads();

    println!("corpus   {}", dir.display());
    println!("threads  {n_threads}");
    println!("step     B={B} V={V} seq_len={SEQ_LEN} -> {TOKENS_PER_STEP} tokens/step");

    let t = Instant::now();
    let corpus = Corpus::open(&dir).expect("open corpus");
    let open_s = t.elapsed().as_secs_f64();
    let n_cells = corpus.cells().len();
    println!("open     {open_s:.2}s to map and index {n_cells} cells");
    assert!(n_cells >= B, "corpus has {n_cells} cells, fewer than B={B}");

    let batches = (n_cells / B).clamp(2, 24);
    println!("batches  {batches} of {B} cells\n");

    let run = |label: &str, seed_base: u64| -> (f64, usize) {
        let t = Instant::now();
        let mut tokens = 0usize;
        let mut short = 0usize;
        for i in 0..batches {
            let cells: Vec<u32> = ((i * B) as u32..((i + 1) * B) as u32).collect();
            let spec = BatchSpec {
                cells,
                v: V,
                seq_len: SEQ_LEN,
                epoch_seed: seed_base + i as u64,
                frac: 1.0,
                frac_seed: 0,
            };
            let batch = build_batch(&corpus, &spec).expect("build_batch");
            tokens += batch.used_tokens();
            short += batch.short_cells;
        }
        let secs = t.elapsed().as_secs_f64();
        let per_batch_ms = secs / batches as f64 * 1000.0;
        let tps = tokens as f64 / secs;
        println!(
            "{label:6}  {secs:6.2}s  {per_batch_ms:7.1} ms/batch  {:.2} M tokens/s  \
             ({:.1} M/s/thread)  short_cells {short}",
            tps / 1e6,
            tps / 1e6 / n_threads as f64,
        );
        (tps, tokens)
    };

    let (cold, _) = run("cold", 1000);
    let (warm, tokens) = run("warm", 2000);

    println!("\naverage window occupancy  {:.2}%", {
        let per_batch = tokens as f64 / batches as f64;
        per_batch / (B * V * SEQ_LEN) as f64 * 100.0
    });

    println!("\n--- against the requirement ---");
    for (label, tps) in [("cold", cold), ("warm", warm)] {
        let ms_per_step = TOKENS_PER_STEP as f64 / tps * 1000.0;
        println!(
            "{label:6}  {:.2} M tokens/s  ->  {ms_per_step:6.1} ms to pack one step's {TOKENS_PER_STEP} tokens",
            tps / 1e6
        );
    }
    println!(
        "\nPacking is free whenever a step's GPU time exceeds the ms above, since packing overlaps\n\
         compute. It is NOT free below that, and the fix would be packing, not more workers."
    );
}
