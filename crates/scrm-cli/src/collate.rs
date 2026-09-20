use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use scrm_core::record::Umi;
use scrm_core::{CellId, ParsedPair};

const SHARDS: usize = 64;

const FLUSH_BYTES: usize = 8 << 20;

const HEADER_BYTES: usize = 8 + 8 + 4;

const BASES_PER_BYTE: usize = 4;

struct Shard {
    buf: Vec<u8>,
    file: Option<BufWriter<File>>,
    path: PathBuf,
    n_records: usize,
}

pub struct Collator {
    dir: PathBuf,
    shards: Vec<Mutex<Shard>>,
    n_cells: usize,
}

impl Collator {
    pub fn new(n_cells: usize) -> Result<Self> {
        static SERIAL: AtomicU64 = AtomicU64::new(0);
        static SWEPT: AtomicU64 = AtomicU64::new(0);
        if SWEPT.fetch_add(1, Ordering::Relaxed) == 0 {
            let n = sweep_stale_spills(&std::env::temp_dir());
            if n > 0 {
                eprintln!("spill         reclaimed {n} stale spill dir(s) from processes that are gone");
            }
        }
        let dir = std::env::temp_dir().join(format!(
            "scrm-collate-{}-{}",
            std::process::id(),
            SERIAL.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).with_context(|| format!("creating spill dir {}", dir.display()))?;
        let shards = (0..SHARDS)
            .map(|k| {
                Mutex::new(Shard {
                    buf: Vec::with_capacity(FLUSH_BYTES + (1 << 12)),
                    file: None,
                    path: dir.join(format!("{k:03}.spill")),
                    n_records: 0,
                })
            })
            .collect();
        Ok(Self {
            dir,
            shards,
            n_cells: n_cells.max(1),
        })
    }

    #[inline]
    fn shard_of(&self, rank: u32) -> usize {
        (rank as usize * SHARDS) / self.n_cells
    }

    pub fn push(&self, rank: u32, pair: &ParsedPair) -> Result<()> {
        let mut record = [0u8; HEADER_BYTES];
        record[..8].copy_from_slice(&pair.cb.0.to_le_bytes());
        record[8..16].copy_from_slice(&pair.umi.0.to_le_bytes());
        record[16..].copy_from_slice(&(pair.r2.len() as u32).to_le_bytes());
        let packed = pack_read(&pair.r2)?;
        let mut quals = pair.qual.clone();
        quals.resize(pair.r2.len(), 0);

        let mut shard = self.shards[self.shard_of(rank)].lock().expect("shard mutex");
        shard.buf.extend_from_slice(&record);
        shard.buf.extend_from_slice(&packed);
        shard.buf.extend_from_slice(&quals);
        shard.n_records += 1;
        if shard.buf.len() >= FLUSH_BYTES {
            shard.spill()?;
        }
        Ok(())
    }

    pub fn seal(mut self) -> Result<SealedShards> {
        let mut sealed = Vec::with_capacity(SHARDS);
        for m in self.shards.drain(..) {
            let mut shard = m.into_inner().expect("shard mutex");
            shard.spill()?;
            if let Some(mut f) = shard.file.take() {
                f.flush().context("flushing spill")?;
            }
            sealed.push((shard.path, shard.n_records));
        }
        let dir = std::mem::take(&mut self.dir);
        Ok(SealedShards { dir, sealed })
    }
}

fn pack_read(tokens: &[u8]) -> Result<Vec<u8>> {
    let mut out = vec![0u8; tokens.len().div_ceil(BASES_PER_BYTE)];
    for (i, &t) in tokens.iter().enumerate() {
        anyhow::ensure!(
            t < 4,
            "token {t} at offset {i} is not A/C/G/T; only N-free reads reach the collator"
        );
        out[i / BASES_PER_BYTE] |= t << (2 * (i % BASES_PER_BYTE));
    }
    Ok(out)
}

fn unpack_read(packed: &[u8], bases: usize) -> Vec<u8> {
    (0..bases)
        .map(|i| (packed[i / BASES_PER_BYTE] >> (2 * (i % BASES_PER_BYTE))) & 3)
        .collect()
}

impl Shard {
    fn spill(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        let file = match &mut self.file {
            Some(f) => f,
            none => {
                let f = File::create(&self.path).with_context(|| format!("creating spill {}", self.path.display()))?;
                none.insert(BufWriter::with_capacity(1 << 20, f))
            }
        };
        file.write_all(&self.buf).with_context(|| {
            format!(
                "writing spill {}\n\
                 The spill device is chosen by TMPDIR (currently {}). On this cluster /tmp is a 10 GB \
                 ephemeral volume unless the job asked for more with --gres=disk:tmp:NG, and a large \
                 dataset needs far more than that -- 8 lanes of SOD1 exhausted 10 GB at lane 6 of 8. \
                 Point TMPDIR at a device with room, or request a bigger /tmp.",
                self.path.display(),
                std::env::var("TMPDIR").unwrap_or_else(|_| std::env::temp_dir().display().to_string()),
            )
        })?;
        self.buf.clear();
        Ok(())
    }
}

pub struct SealedShards {
    dir: PathBuf,
    sealed: Vec<(PathBuf, usize)>,
}

impl SealedShards {
    pub fn load(&self, k: usize) -> Result<Vec<(CellId, Vec<ParsedPair>)>> {
        let (path, n_records) = &self.sealed[k];
        if *n_records == 0 {
            return Ok(Vec::new());
        }
        let mut bytes = Vec::new();
        File::open(path)
            .with_context(|| format!("opening spill {}", path.display()))?
            .read_to_end(&mut bytes)
            .with_context(|| format!("reading spill {}", path.display()))?;

        let mut pairs: Vec<(u64, ParsedPair)> = Vec::with_capacity(*n_records);
        let mut at = 0usize;
        while at < bytes.len() {
            anyhow::ensure!(
                at + HEADER_BYTES <= bytes.len(),
                "truncated spill {}: {} bytes left, need {HEADER_BYTES}",
                path.display(),
                bytes.len() - at
            );
            let cb = u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8 bytes"));
            let umi = u64::from_le_bytes(bytes[at + 8..at + 16].try_into().expect("8 bytes"));
            let bases = u32::from_le_bytes(bytes[at + 16..at + 20].try_into().expect("4 bytes")) as usize;
            at += HEADER_BYTES;
            let packed_len = bases.div_ceil(BASES_PER_BYTE);
            anyhow::ensure!(
                at + packed_len <= bytes.len(),
                "truncated spill {}: a {bases}-base read runs past end",
                path.display()
            );
            anyhow::ensure!(
                at + packed_len + bases <= bytes.len(),
                "truncated spill {}: a {bases}-base read's qualities run past end",
                path.display()
            );
            let r2 = unpack_read(&bytes[at..at + packed_len], bases);
            let qual = bytes[at + packed_len..at + packed_len + bases].to_vec();
            pairs.push((
                cb,
                ParsedPair {
                    cb: CellId(cb),
                    umi: Umi(umi),
                    r2,
                    qual,
                },
            ));
            at += packed_len + bases;
        }
        anyhow::ensure!(
            pairs.len() == *n_records,
            "spill {} holds {} records, {n_records} were written",
            path.display(),
            pairs.len()
        );

        pairs.sort_unstable_by_key(|(cb, _)| *cb);
        let mut cells: Vec<(CellId, Vec<ParsedPair>)> = Vec::new();
        for (cb, pair) in pairs {
            match cells.last_mut() {
                Some((last, group)) if last.0 == cb => group.push(pair),
                _ => cells.push((CellId(cb), vec![pair])),
            }
        }
        Ok(cells)
    }

    pub fn len(&self) -> usize {
        self.sealed.len()
    }

    pub fn spilled_bytes(&self) -> u64 {
        self.sealed
            .iter()
            .filter_map(|(p, _)| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum()
    }
}

impl Drop for Collator {
    fn drop(&mut self) {
        for shard in &self.shards {
            if let Ok(s) = shard.lock() {
                std::fs::remove_file(&s.path).ok();
            }
        }
        std::fs::remove_dir(&self.dir).ok();
    }
}

fn sweep_stale_spills(temp: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(temp) else {
        return 0;
    };
    let mut reclaimed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        let Some(rest) = name.strip_prefix("scrm-collate-") else {
            continue;
        };
        let Some(pid) = rest.split('-').next().and_then(|p| p.parse::<i32>().ok()) else {
            continue;
        };
        if pid_is_alive(pid) {
            continue;
        }
        if std::fs::remove_dir_all(entry.path()).is_ok() {
            reclaimed += 1;
        }
    }
    reclaimed
}

#[cfg(target_os = "linux")]
fn pid_is_alive(pid: i32) -> bool {
    std::path::Path::new(&format!("/proc/{pid}")).exists()
}

#[cfg(not(target_os = "linux"))]
fn pid_is_alive(_pid: i32) -> bool {
    true
}

impl Drop for SealedShards {
    fn drop(&mut self) {
        for (path, _) in &self.sealed {
            std::fs::remove_file(path).ok();
        }
        std::fs::remove_dir(&self.dir).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_stale_spill_dir_from_a_dead_process_is_swept_and_a_live_one_is_not() {
        let temp = std::env::temp_dir().join(format!("scrm-sweep-test-{}", std::process::id()));
        std::fs::create_dir_all(&temp).unwrap();

        let live = temp.join("scrm-collate-1-0");
        let dead = temp.join("scrm-collate-2147483646-0");
        let other = temp.join("something-else-0");
        for d in [&live, &dead, &other] {
            std::fs::create_dir_all(d).unwrap();
            std::fs::write(d.join("000.spill"), b"bytes").unwrap();
        }

        let reclaimed = sweep_stale_spills(&temp);
        assert!(live.exists(), "a live process's spill must never be touched");
        assert!(other.exists(), "only this program's own prefix is in scope");
        if cfg!(target_os = "linux") {
            assert_eq!(reclaimed, 1, "exactly the dead one");
            assert!(!dead.exists(), "a dead process's spill must be reclaimed");
        } else {
            assert_eq!(reclaimed, 0, "no liveness source, so nothing may be reclaimed");
            assert!(dead.exists(), "an unresolvable pid counts as alive");
        }

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn the_sweep_never_reclaims_the_running_process_own_spill_dir() {
        let temp = std::env::temp_dir().join(format!("scrm-sweep-self-{}", std::process::id()));
        std::fs::create_dir_all(&temp).unwrap();

        let mine = temp.join(format!("scrm-collate-{}-0", std::process::id()));
        std::fs::create_dir_all(&mine).unwrap();
        std::fs::write(mine.join("000.spill"), b"bytes").unwrap();

        assert_eq!(sweep_stale_spills(&temp), 0, "nothing here is stale");
        assert!(
            mine.exists(),
            "the sweeping process's own spill dir must survive its own sweep"
        );
        assert!(mine.join("000.spill").exists(), "and so must its contents");

        std::fs::remove_dir_all(&temp).ok();
    }

    #[test]
    fn dropping_a_collator_before_sealing_removes_its_spill_dir() {
        let dir = {
            let c = Collator::new(4).unwrap();
            let dir = c.dir.clone();
            for i in 0..2000u64 {
                let r2 = [0u8, 1, 2, 3].repeat(8);
                c.push((i % 4) as u32, &pair(i % 4, i, &r2)).unwrap();
            }
            assert!(dir.exists(), "the spill dir should exist while the collator does");
            dir
        };
        assert!(!dir.exists(), "the spill dir must be gone once the collator is dropped");
    }

    fn pair(cb: u64, umi: u64, r2: &[u8]) -> ParsedPair {
        let qual: Vec<u8> = (0..r2.len()).map(|i| (i % 41) as u8).collect();
        ParsedPair {
            cb: CellId(cb),
            umi: Umi(umi),
            r2: r2.to_vec(),
            qual,
        }
    }

    #[test]
    fn packing_round_trips_at_every_length() {
        for bases in 0..200usize {
            let tokens: Vec<u8> = (0..bases).map(|i| (i % 4) as u8).collect();
            let packed = pack_read(&tokens).unwrap();
            assert_eq!(
                packed.len(),
                bases.div_ceil(BASES_PER_BYTE),
                "{bases} bases packed to the wrong size"
            );
            assert_eq!(unpack_read(&packed, bases), tokens, "{bases} bases did not survive");
        }
    }

    #[test]
    fn packing_is_four_bases_per_byte() {
        let read = vec![1u8; 92];
        assert_eq!(pack_read(&read).unwrap().len(), 23);
    }

    #[test]
    fn a_non_acgt_token_is_refused_not_aliased() {
        let err = pack_read(&[0, 1, 4, 2]).unwrap_err().to_string();
        assert!(err.contains("not A/C/G/T"), "{err}");
    }

    #[test]
    fn shard_ranges_are_monotonic_and_in_bounds() {
        let c = Collator::new(7377).unwrap();
        let mut last = 0;
        for rank in 0..7377u32 {
            let s = c.shard_of(rank);
            assert!(s < SHARDS, "rank {rank} -> shard {s}");
            assert!(s >= last, "shard went backwards at rank {rank}");
            last = s;
        }
        assert_eq!(c.shard_of(0), 0);
        assert_eq!(c.shard_of(7376), SHARDS - 1);
    }

    #[test]
    fn one_rank_maps_to_exactly_one_shard() {
        for n_cells in [1usize, 2, 63, 64, 65, 1000, 7377] {
            let c = Collator::new(n_cells).unwrap();
            for rank in 0..n_cells as u32 {
                let a = c.shard_of(rank);
                assert_eq!(a, c.shard_of(rank), "rank {rank} of {n_cells} is unstable");
            }
        }
    }

    #[test]
    fn round_trip_preserves_every_pair() {
        let c = Collator::new(4).unwrap();
        let inputs = [
            (0u32, pair(10, 100, b"\x00\x01\x02")),
            (0, pair(10, 101, b"\x03\x02")),
            (1, pair(20, 200, b"\x01")),
            (3, pair(40, 400, &[2u8; 300])),
        ];
        for (rank, p) in &inputs {
            c.push(*rank, p).unwrap();
        }
        let sealed = c.seal().unwrap();

        let mut seen = Vec::new();
        for k in 0..sealed.len() {
            for (cell, pairs) in sealed.load(k).unwrap() {
                for p in pairs {
                    assert_eq!(p.cb, cell, "grouped under the wrong cell");
                    seen.push((p.cb.0, p.umi.0, p.r2));
                }
            }
        }
        assert_eq!(seen.len(), inputs.len());
        let cbs: Vec<u64> = seen.iter().map(|(cb, _, _)| *cb).collect();
        assert_eq!(cbs, vec![10, 10, 20, 40], "cells are not in ascending order");
        assert_eq!(seen[3].2, vec![2u8; 300], "long read did not survive");
    }

    #[test]
    fn spanning_the_flush_threshold_is_lossless() {
        let c = Collator::new(1).unwrap();
        let read = vec![1u8; 4096];
        let on_disk = HEADER_BYTES + read.len().div_ceil(BASES_PER_BYTE);
        let n = (FLUSH_BYTES / on_disk) * 3;
        for i in 0..n {
            c.push(0, &pair(7, i as u64, &read)).unwrap();
        }
        let sealed = c.seal().unwrap();
        assert!(sealed.spilled_bytes() > FLUSH_BYTES as u64, "nothing spilled");
        let cells = sealed.load(0).unwrap();
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].1.len(), n);
        assert!(cells[0].1.iter().all(|p| p.r2 == read));
    }

    #[test]
    fn an_empty_shard_reads_back_empty() {
        let sealed = Collator::new(64).unwrap().seal().unwrap();
        for k in 0..sealed.len() {
            assert!(sealed.load(k).unwrap().is_empty());
        }
        assert_eq!(sealed.spilled_bytes(), 0, "an untouched shard touched the disk");
    }

    #[test]
    fn spill_files_are_removed_on_drop() {
        let c = Collator::new(1).unwrap();
        c.push(0, &pair(1, 1, b"\x00")).unwrap();
        let sealed = c.seal().unwrap();
        let paths: Vec<PathBuf> = sealed.sealed.iter().map(|(p, _)| p.clone()).collect();
        let dir = sealed.dir.clone();
        assert!(paths.iter().any(|p| p.exists()), "nothing was written");
        drop(sealed);
        assert!(!paths.iter().any(|p| p.exists()), "spill files outlived the run");
        assert!(!dir.exists(), "spill dir outlived the run");
    }
}
