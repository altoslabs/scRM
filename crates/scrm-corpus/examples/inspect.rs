use std::path::{Path, PathBuf};

use scrm_corpus::{CorpusReader, MANIFEST_FILE, Manifest, stream_digest, verify, verify_corpus};
use sha2::{Digest, Sha256};

struct Identity {
    stream: String,
    n_rows: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let targets: Vec<PathBuf> = std::env::args().skip(1).map(PathBuf::from).collect();
    if targets.is_empty() {
        return Err("usage: inspect CORPUS_DIR_OR_FILE [ANOTHER ...]".into());
    }

    let mut identities = Vec::new();
    for target in &targets {
        let id = if target.join(MANIFEST_FILE).exists() {
            report_corpus(target)?
        } else {
            report_single_file(target)?
        };
        identities.push((target.clone(), id));
    }

    if identities.len() > 1 {
        let (first_path, first) = &identities[0];
        println!();
        let same = identities.iter().all(|(_, i)| i.stream == first.stream);
        if same {
            println!(
                "IDENTICAL rows: all {} targets stream to {}",
                identities.len(),
                first.stream
            );
        } else {
            println!("DIFFER:");
            for (p, i) in &identities {
                let mark = if i.stream == first.stream { "==" } else { "!=" };
                println!("  {mark} {}  {} rows  {}", i.stream, i.n_rows, p.display());
            }
            println!("(compared against {})", first_path.display());
        }
    }
    Ok(())
}

fn report_corpus(dir: &Path) -> Result<Identity, Box<dyn std::error::Error>> {
    let manifest = Manifest::read(dir)?;
    let stats = verify_corpus(dir)?;
    let streamed = stream_digest(dir)?;
    let cfg = &manifest.declared.config;

    println!("{}  (sharded corpus)", dir.display());
    println!("  shards         {}", stats.n_shards);
    println!("  rows           {}", stats.n_rows);
    println!("  tokens         {}", stats.n_tokens);
    println!("  sha256 root    {}", stats.root);
    println!("  sha256 stream  {streamed}");
    println!(
        "  cells          {}..{}",
        manifest.shards.first().map(|s| s.first_cell.as_str()).unwrap_or("-"),
        manifest.shards.last().map(|s| s.last_cell.as_str()).unwrap_or("-")
    );
    println!(
        "  compat_key     alphabet {} k {} umi_len {} dedup {} bubbles {}",
        cfg.alphabet_size, cfg.k, cfg.umi_len, cfg.dedup_rule, cfg.bubble_resolution
    );
    println!(
        "  build          {} {}",
        manifest.declared.build.scrm_sha,
        if manifest.declared.build.dirty { "(dirty)" } else { "" }
    );
    println!(
        "  molecules      {}  pairs_in {}",
        manifest.measured.counts.molecules_out, manifest.measured.counts.pairs_in
    );
    Ok(Identity {
        stream: streamed,
        n_rows: stats.n_rows,
    })
}

fn report_single_file(path: &Path) -> Result<Identity, Box<dyn std::error::Error>> {
    verify(path)?;
    let mut reader = CorpusReader::open(path)?;
    let cfg = &reader.declared().config;

    println!("{}  (single file)", path.display());
    println!("  rows           {}", reader.n_rows());
    println!("  tokens         {}", reader.n_tokens());
    println!("  sha256 stream  {}", reader.tokens_sha256());
    println!("  cells          {}", reader.cells().len());
    println!(
        "  compat_key     alphabet {} k {} umi_len {} dedup {} bubbles {}",
        cfg.alphabet_size, cfg.k, cfg.umi_len, cfg.dedup_rule, cfg.bubble_resolution
    );

    let mut hasher = Sha256::new();
    while let Some(batch) = reader.next_batch() {
        let batch = batch?;
        for i in 0..batch.len() {
            hasher.update(batch.row(i).tokens);
        }
    }
    let stream = format!("{:x}", hasher.finalize());
    Ok(Identity {
        stream,
        n_rows: reader.n_rows(),
    })
}
