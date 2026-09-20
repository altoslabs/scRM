use std::path::{Path, PathBuf};
use std::process::Command;

const TOOLS: &[&str] = &[
    "paraseq-temp",
    "rapidgzip-core",
    "libradicl",
    "seq_geom_parser",
    "debruijn",
    "arrow",
];

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let lock = find_lockfile(&manifest).expect("no Cargo.lock above the manifest directory");
    println!("cargo::rerun-if-changed={}", lock.display());
    let text = std::fs::read_to_string(&lock).expect("reading Cargo.lock");

    for tool in TOOLS {
        let key = tool.replace('-', "_");
        println!("cargo::rustc-env=SCRM_VERSION_{key}={}", version_of(&text, tool));
    }

    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let out = Command::new(&rustc)
        .arg("--version")
        .output()
        .unwrap_or_else(|e| panic!("running `{rustc} --version`: {e}"));
    let rustc_version = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(!rustc_version.is_empty(), "`{rustc} --version` printed nothing");
    println!("cargo::rustc-env=SCRM_RUSTC_VERSION={rustc_version}");

    let out = Command::new("date")
        .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
        .output()
        .unwrap_or_else(|e| panic!("running `date -u`: {e}"));
    let built_at = String::from_utf8_lossy(&out.stdout).trim().to_string();
    assert!(built_at.ends_with('Z'), "`date -u` gave {built_at:?}");
    println!("cargo::rustc-env=SCRM_BUILT_AT_UTC={built_at}");
}

fn find_lockfile(from: &Path) -> Option<PathBuf> {
    from.ancestors().map(|d| d.join("Cargo.lock")).find(|p| p.is_file())
}

fn version_of(lock: &str, name: &str) -> String {
    let mut in_block = false;
    let (mut version, mut source) = (None, None);
    for line in lock.lines() {
        let line = line.trim();
        if line == "[[package]]" {
            if in_block {
                break;
            }
            in_block = false;
            version = None;
            source = None;
            continue;
        }
        if let Some(v) = line.strip_prefix("name = ") {
            in_block = v.trim_matches('"') == name;
            continue;
        }
        if !in_block {
            continue;
        }
        if let Some(v) = line.strip_prefix("version = ") {
            version = Some(v.trim_matches('"').to_string());
        }
        if let Some(s) = line.strip_prefix("source = ") {
            source = Some(s.trim_matches('"').to_string());
        }
    }
    let version = version.unwrap_or_else(|| panic!("Cargo.lock has no version for `{name}`"));
    match source.as_deref().and_then(|s| s.split_once('#')) {
        Some((_, rev)) => format!("{version} (git {})", &rev[..rev.len().min(12)]),
        None => version,
    }
}
