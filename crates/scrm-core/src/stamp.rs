use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stamp {
    pub declared: Declared,
    pub measured: Measured,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Declared {
    pub build: Build,
    pub config: Config,
    pub inputs: Vec<InputFile>,
    pub tools: Tools,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measured {
    pub counts: Counts,
    pub umi_stats: UmiStats,
    pub assembly_stats: AssemblyStats,
    pub artifact_rates: ArtifactRates,
    pub throughput: Throughput,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Build {
    pub scrm_sha: String,
    pub dirty: bool,
    pub built_at_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub geometry: String,
    pub cb_len: u8,
    pub umi_len: u8,
    pub whitelist_sha256: String,
    pub alphabet_size: u8,
    pub dedup_rule: String,
    pub k: usize,
    pub min_trim_len: usize,
    pub trim: serde_json::Value,
    pub read_policy: String,
    pub bubble_resolution: String,
    pub max_windows_per_cell: u32,
    pub min_umi_len: usize,
    pub max_umi_len: usize,
    pub min_cell_molecules: usize,
}

pub type CompatKey<'a> = (u8, usize, &'a str, &'a str, usize, usize, usize);

impl Config {
    pub fn compat_key(&self) -> CompatKey<'_> {
        (
            self.alphabet_size,
            self.k,
            self.dedup_rule.as_str(),
            self.bubble_resolution.as_str(),
            self.min_umi_len,
            self.max_umi_len,
            self.min_cell_molecules,
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputFile {
    pub path: String,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counts {
    pub pairs_in: u64,
    pub dropped_n: u64,
    pub dropped_short: u64,
    pub dropped_cb: u64,
    pub cells_called: u64,
    pub dropped_umi_short: u64,
    pub dropped_umi_short_tokens: u64,
    pub dropped_umi_long: u64,
    pub dropped_umi_long_tokens: u64,
    pub dropped_cells_shallow: u64,
    pub dropped_cells_molecules: u64,
    #[serde(default)]
    pub dropped_cells_emptied: u64,
    pub molecules_out: u64,
    pub tokens_out: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UmiStats {
    pub mean_hamming1_degree: f64,
    pub singleton_fraction: f64,
    pub merge_rate: f64,
    pub max_group_fraction: f64,
    pub max_group_fraction_umis: u64,
    pub max_group_umis: u32,
    pub max_chain_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyStats {
    pub reads_per_molecule_hist: Vec<u64>,
    pub distinct_seqs_per_molecule_hist: Vec<u64>,
    pub pass_through_fraction: f64,
    pub contigs_per_molecule_hist: Vec<u64>,
    pub contig_length_hist: Vec<u64>,
    pub self_loop_molecules: u64,
    pub bubble_n_tokens: u64,
    #[serde(default)]
    pub bubble_n_tie: u64,
    #[serde(default)]
    pub bubble_n_low_confidence: u64,
    #[serde(default)]
    pub bubble_n_indel_interior: u64,
    #[serde(default)]
    pub ambiguous_read_placements: u64,
    #[serde(default)]
    pub branch_decisions: u64,
    #[serde(default)]
    pub origin_splits: u64,
    #[serde(default)]
    pub origin_surprise_hist: Vec<u64>,
    #[serde(default)]
    pub called_q_hist: Vec<u64>,
    #[serde(default)]
    pub origin_k_total: u64,
    #[serde(default)]
    pub origin_lambda_total: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRates {
    pub adapter_fraction: f64,
    pub tso_fraction: f64,
    pub homopolymer: std::collections::BTreeMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Throughput {
    pub read_pairs_per_sec: f64,
    pub molecules_per_sec: f64,
    pub secs_stream: f64,
    pub secs_assemble_write: f64,
    pub n_threads: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tools {
    pub rustc: String,
    pub paraseq: String,
    pub rapidgzip_core: String,
    pub thread_broker: String,
    pub libradicl: String,
    pub seq_geom_parser: String,
    pub debruijn: String,
    pub arrow: String,
}

macro_rules! json_roundtrip {
    ($t:ty) => {
        impl $t {
            pub fn to_json(&self) -> serde_json::Result<String> {
                serde_json::to_string_pretty(self)
            }

            pub fn from_json(s: &str) -> serde_json::Result<Self> {
                serde_json::from_str(s)
            }
        }
    };
}

json_roundtrip!(Stamp);
json_roundtrip!(Declared);
json_roundtrip!(Measured);

impl Stamp {
    pub fn compat_key(&self) -> CompatKey<'_> {
        self.declared.config.compat_key()
    }

    pub fn config(&self) -> &Config {
        &self.declared.config
    }
}
