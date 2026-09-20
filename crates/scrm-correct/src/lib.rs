//! Cell-barcode correction against an explicit list of called cells.
//!
//! The rule is alevin-fry's `ExplicitList` permit-list mode (COMBINE-lab/alevin-fry),
//! reimplemented here from its description: the list is CellRanger's filtered
//! barcodes, an exact hit is kept as `Exact`, and an unmatched barcode is `Corrected` only when
//! exactly ONE list entry sits at Hamming distance 1. Two or more neighbours is `Ambiguous` and
//! the read is discarded -- the tie is never broken by a count. There is no knee finder and no 10x
//! master whitelist, so the corpus cell set is exactly the list the evaluation labels are keyed by.
//!
//! The Hamming-1 neighbour query itself is `libradicl`'s `BarcodeLookupMap` (COMBINE-lab,
//! BSD-3-Clause), a dependency rather than a reimplementation. Both notices are in
//! THIRD_PARTY_NOTICES.md.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use libradicl::BarcodeLookupMap;
use scrm_core::{CellId, MAX_PACK_LEN, pack_acgt, unpack_acgt};
use sha2::{Digest, Sha256};

#[derive(Debug, thiserror::Error)]
pub enum CorrectError {
    #[error("failed to read barcode list {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("barcode list {path} line {line}: {barcode:?} is not pure ACGT")]
    NotAcgt { path: String, line: usize, barcode: String },

    #[error("barcode list {path} line {line}: length {len} != expected {expected}")]
    WrongLength {
        path: String,
        line: usize,
        len: usize,
        expected: usize,
    },

    #[error("barcode list {path} is empty")]
    Empty { path: String },

    #[error("barcode length {0} exceeds {MAX_PACK_LEN}")]
    TooLong(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Correction {
    Exact(CellId),
    Corrected(CellId),
    Ambiguous,
    NoMatch,
}

impl Correction {
    #[inline]
    pub fn cell(self) -> Option<CellId> {
        match self {
            Correction::Exact(c) | Correction::Corrected(c) => Some(c),
            Correction::Ambiguous | Correction::NoMatch => None,
        }
    }

    #[inline]
    pub fn is_kept(self) -> bool {
        self.cell().is_some()
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CorrectionStats {
    pub exact: u64,
    pub corrected: u64,
    pub ambiguous: u64,
    pub no_match: u64,
}

impl CorrectionStats {
    #[inline]
    pub fn record(&mut self, c: Correction) {
        match c {
            Correction::Exact(_) => self.exact += 1,
            Correction::Corrected(_) => self.corrected += 1,
            Correction::Ambiguous => self.ambiguous += 1,
            Correction::NoMatch => self.no_match += 1,
        }
    }

    pub fn total(&self) -> u64 {
        self.exact + self.corrected + self.ambiguous + self.no_match
    }

    pub fn dropped(&self) -> u64 {
        self.ambiguous + self.no_match
    }
}

pub struct Corrector {
    map: BarcodeLookupMap,
    packed: Vec<u64>,
    bclen: usize,
    pub list_sha256: String,
    pub n_barcodes: usize,
}

impl Corrector {
    pub fn from_filtered_matrix(path: &Path, bclen: usize) -> Result<Self, CorrectError> {
        if bclen > MAX_PACK_LEN {
            return Err(CorrectError::TooLong(bclen));
        }
        let p = path.display().to_string();
        let raw = std::fs::read(path).map_err(|source| CorrectError::Io {
            path: p.clone(),
            source,
        })?;

        let reader: Box<dyn Read> = if path.extension().is_some_and(|e| e == "gz") {
            Box::new(flate2::read::GzDecoder::new(&raw[..]))
        } else {
            Box::new(&raw[..])
        };

        let mut packed = Vec::new();
        for (i, line) in BufReader::new(reader).lines().enumerate() {
            let line = line.map_err(|source| CorrectError::Io {
                path: p.clone(),
                source,
            })?;
            let bc = line.trim();
            if bc.is_empty() {
                continue;
            }
            let bc = bc.split('-').next().unwrap_or(bc);
            if bc.len() != bclen {
                return Err(CorrectError::WrongLength {
                    path: p.clone(),
                    line: i + 1,
                    len: bc.len(),
                    expected: bclen,
                });
            }
            let v = pack_acgt(bc.as_bytes()).ok_or_else(|| CorrectError::NotAcgt {
                path: p.clone(),
                line: i + 1,
                barcode: bc.to_string(),
            })?;
            packed.push(v);
        }

        if packed.is_empty() {
            return Err(CorrectError::Empty { path: p });
        }

        packed.sort_unstable();
        packed.dedup();
        let n_barcodes = packed.len();

        // THE HASH IS OF THE CELL LIST, NOT THE FILE. It was taken over the raw bytes before
        // decompressing, and gzip writes the current time into its header -- so the same Cell
        // Ranger matrix produced a different `whitelist_sha256` on every build, and the field
        // identified when the file was compressed rather than which cells were called. Two
        // identical corpora looked different and the field could not do the one job it has.
        //
        // Taken over the sorted, deduplicated, packed barcodes, so it is a function of the called
        // set alone: stable across compression, across `.gz` and plain, and across re-fetches,
        // and it still differs whenever the set does.
        let mut hasher = Sha256::new();
        hasher.update((n_barcodes as u64).to_le_bytes());
        hasher.update((bclen as u64).to_le_bytes());
        for v in &packed {
            hasher.update(v.to_le_bytes());
        }
        let list_sha256 = format!("{:x}", hasher.finalize());

        Ok(Self {
            map: BarcodeLookupMap::new(packed.clone(), bclen as u32),
            packed,
            bclen,
            list_sha256,
            n_barcodes,
        })
    }

    #[inline]
    pub fn correct(&self, raw: u64) -> Correction {
        match self.map.find_neighbors(raw, true) {
            (Some(idx), 1) => {
                let bc = self.map.barcode_for_idx(idx);
                if bc == raw {
                    Correction::Exact(CellId(bc))
                } else {
                    Correction::Corrected(CellId(bc))
                }
            }
            (_, n) if n >= 2 => Correction::Ambiguous,
            _ => Correction::NoMatch,
        }
    }

    pub fn render(&self, cell: CellId) -> Vec<u8> {
        unpack_acgt(cell.0, self.bclen)
    }

    pub fn bclen(&self) -> usize {
        self.bclen
    }

    #[inline]
    pub fn rank(&self, cell: CellId) -> Option<u32> {
        let i = self.packed.partition_point(|&b| b < cell.0);
        (self.packed.get(i) == Some(&cell.0)).then_some(i as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_list(barcodes: &[&str], gz: bool) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(if gz { ".gz" } else { ".tsv" })
            .tempfile()
            .unwrap();
        let body = barcodes.join("\n");
        if gz {
            let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
            e.write_all(body.as_bytes()).unwrap();
            f.write_all(&e.finish().unwrap()).unwrap();
        } else {
            f.write_all(body.as_bytes()).unwrap();
        }
        f.flush().unwrap();
        f
    }

    const BC: &str = "AAACCCAAGGCATCAG";

    #[test]
    fn rank_is_dense_and_matches_packed_order() {
        let list = [
            "TTTTTTTTTTTTTTTT",
            "AAACCCAAGGCATCAG",
            "AAAAAAAAAAAAAAAA",
            "AAACCCAAGGCATCAG",
        ];
        let f = write_list(&list, false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        assert_eq!(c.n_barcodes, 3, "the duplicate was not collapsed");

        let mut ranked: Vec<(u32, u64)> = ["AAAAAAAAAAAAAAAA", "AAACCCAAGGCATCAG", "TTTTTTTTTTTTTTTT"]
            .iter()
            .map(|bc| {
                let packed = pack_acgt(bc.as_bytes()).unwrap();
                (c.rank(CellId(packed)).expect("called cell has a rank"), packed)
            })
            .collect();
        assert_eq!(ranked.iter().map(|(r, _)| *r).collect::<Vec<_>>(), vec![0, 1, 2]);
        ranked.sort_unstable();
        assert!(ranked.windows(2).all(|w| w[0].1 < w[1].1), "rank order != packed order");
    }

    #[test]
    fn an_uncalled_barcode_has_no_rank() {
        let f = write_list(&[BC], false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        let absent = pack_acgt(b"TTTTTTTTTTTTTTTT").unwrap();
        assert_eq!(c.rank(CellId(absent)), None);
        assert_eq!(c.rank(CellId(pack_acgt(BC.as_bytes()).unwrap())), Some(0));
    }

    #[test]
    fn exact_match_is_exact_not_corrected() {
        let f = write_list(&[&format!("{BC}-1")], false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        let q = pack_acgt(BC.as_bytes()).unwrap();
        assert_eq!(c.correct(q), Correction::Exact(CellId(q)));
    }

    #[test]
    fn gem_well_suffix_is_stripped() {
        let f = write_list(&[&format!("{BC}-1")], false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        assert_eq!(c.n_barcodes, 1);
        assert_eq!(c.render(CellId(pack_acgt(BC.as_bytes()).unwrap())), BC.as_bytes());
    }

    #[test]
    fn gzip_and_plain_load_identically() {
        let plain = write_list(&[&format!("{BC}-1")], false);
        let gz = write_list(&[&format!("{BC}-1")], true);
        let a = Corrector::from_filtered_matrix(plain.path(), 16).unwrap();
        let b = Corrector::from_filtered_matrix(gz.path(), 16).unwrap();
        assert_eq!(a.n_barcodes, b.n_barcodes);
        let q = pack_acgt(BC.as_bytes()).unwrap();
        assert_eq!(a.correct(q), b.correct(q));
    }

    #[test]
    fn hamming_one_is_corrected_to_the_list_entry() {
        let f = write_list(&[&format!("{BC}-1")], false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        let mut m = BC.to_string();
        m.pop();
        m.push('T');
        let q = pack_acgt(m.as_bytes()).unwrap();
        let target = pack_acgt(BC.as_bytes()).unwrap();
        assert_eq!(c.correct(q), Correction::Corrected(CellId(target)));
    }

    #[test]
    fn two_equally_good_neighbours_are_ambiguous_not_guessed() {
        let a = "AAACCCAAGGCATCAA";
        let b = "AAACCCAAGGCATCAC";
        let query = "AAACCCAAGGCATCAG";
        let f = write_list(&[&format!("{a}-1"), &format!("{b}-1")], false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        let q = pack_acgt(query.as_bytes()).unwrap();
        assert_eq!(c.correct(q), Correction::Ambiguous);
        assert!(!c.correct(q).is_kept());
    }

    #[test]
    fn distant_barcode_is_no_match() {
        let f = write_list(&[&format!("{BC}-1")], false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        let q = pack_acgt(b"TTTTTTTTTTTTTTTT").unwrap();
        assert_eq!(c.correct(q), Correction::NoMatch);
    }

    #[test]
    fn non_acgt_in_the_list_is_an_error_not_a_discard() {
        let f = write_list(&["AAACCCAAGGCATCAN-1"], false);
        assert!(matches!(
            Corrector::from_filtered_matrix(f.path(), 16),
            Err(CorrectError::NotAcgt { .. })
        ));
    }

    #[test]
    fn wrong_length_is_rejected() {
        let f = write_list(&["AAACCCAAGG-1"], false);
        assert!(matches!(
            Corrector::from_filtered_matrix(f.path(), 16),
            Err(CorrectError::WrongLength { .. })
        ));
    }

    #[test]
    fn duplicates_do_not_make_an_exact_match_ambiguous() {
        let f = write_list(&[&format!("{BC}-1"), &format!("{BC}-1")], false);
        let c = Corrector::from_filtered_matrix(f.path(), 16).unwrap();
        assert_eq!(c.n_barcodes, 1, "duplicates must be collapsed");
        let q = pack_acgt(BC.as_bytes()).unwrap();
        assert_eq!(c.correct(q), Correction::Exact(CellId(q)));
    }

    #[test]
    fn sha256_is_stable_and_content_dependent() {
        let f1 = write_list(&[&format!("{BC}-1")], false);
        let f2 = write_list(&[&format!("{BC}-1")], false);
        let f3 = write_list(&["AAACCCAAGGCATCAT-1"], false);
        let a = Corrector::from_filtered_matrix(f1.path(), 16).unwrap();
        let b = Corrector::from_filtered_matrix(f2.path(), 16).unwrap();
        let c = Corrector::from_filtered_matrix(f3.path(), 16).unwrap();
        assert_eq!(a.list_sha256, b.list_sha256);
        assert_ne!(a.list_sha256, c.list_sha256);
    }

    /// THE HASH IS OF THE CELL LIST, NOT THE CONTAINER. Both files carry one barcode; one is
    /// gzipped and one is not, and gzip writes a timestamp into its header. This test is why the
    /// hash moved off the raw bytes: taken there, the same Cell Ranger matrix produced a
    /// different `whitelist_sha256` on every build, so two identical corpora looked different.
    /// The version above could not see it -- both its files are uncompressed.
    #[test]
    fn sha256_does_not_depend_on_the_container() {
        let plain = write_list(&[&format!("{BC}-1")], false);
        let zipped = write_list(&[&format!("{BC}-1")], true);
        let a = Corrector::from_filtered_matrix(plain.path(), 16).unwrap();
        let b = Corrector::from_filtered_matrix(zipped.path(), 16).unwrap();
        assert_eq!(
            a.list_sha256, b.list_sha256,
            "the same called set, compressed or not, is the same whitelist"
        );
    }

    /// And the order the publisher happens to list them in is not part of the set.
    #[test]
    fn sha256_does_not_depend_on_the_order_the_barcodes_arrive_in() {
        let x = format!("{BC}-1");
        let y = "AAACCCAAGGCATCAT-1".to_string();
        let f1 = write_list(&[&x, &y], false);
        let f2 = write_list(&[&y, &x], false);
        let a = Corrector::from_filtered_matrix(f1.path(), 16).unwrap();
        let b = Corrector::from_filtered_matrix(f2.path(), 16).unwrap();
        assert_eq!(a.list_sha256, b.list_sha256);
        assert_eq!(a.n_barcodes, 2);
    }

    #[test]
    fn stats_partition_every_outcome() {
        let mut s = CorrectionStats::default();
        s.record(Correction::Exact(CellId(0)));
        s.record(Correction::Corrected(CellId(1)));
        s.record(Correction::Ambiguous);
        s.record(Correction::NoMatch);
        assert_eq!(s.total(), 4);
        assert_eq!(s.dropped(), 2);
    }
}
