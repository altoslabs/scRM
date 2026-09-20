use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "scrm-permute",
    about = "Permute molecule payloads across cells, depth held exactly (ablation C3)"
)]
struct Args {
    #[arg(long)]
    src: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    scrm_corpus::refuse_if_corpus(&args.out)?;
    let manifest = scrm_corpus::permute_corpus(&args.src, &args.out, args.seed)
        .with_context(|| format!("permuting {}", args.src.display()))?;
    eprintln!(
        "permuted      {} -> {} (seed {})",
        args.src.display(),
        args.out.display(),
        args.seed
    );
    eprintln!("rows          {}  tokens {}", manifest.n_rows, manifest.n_tokens);
    eprintln!("verify        OK (cell layout unchanged, every payload from a different cell)");
    eprintln!("NEXT          pin it: pinned_digests.py --report, then --write, then commit");
    Ok(())
}
