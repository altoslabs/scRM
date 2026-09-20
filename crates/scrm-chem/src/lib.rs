use seq_geom_parser::{
    CompiledGeom, FragmentGeom, GeoLen, GeoTagType, ReadGeom, format_errors, parse_geometry, validate_geometry,
};

#[derive(Debug, thiserror::Error)]
pub enum ChemError {
    #[error("geometry {geometry:?} failed to parse:\n{detail}")]
    Parse { geometry: String, detail: String },

    #[error("geometry {geometry:?} is semantically invalid: {detail}")]
    Invalid { geometry: String, detail: String },

    #[error("geometry {geometry:?} could not be compiled: {detail}")]
    Compile { geometry: String, detail: String },

    #[error(
        "geometry {geometry:?} declares the biological read on R1; scRM requires R1 = CB+UMI only \
         and the biological read on R2"
    )]
    ReadNotOnR2 { geometry: String },

    #[error(
        "geometry {geometry:?} declares {count} {tag} segments, lengths {lens:?}; exactly one is supported. seq_geom_parser collapses the split before it is visible, so accepting this would use part of the {tag} and discard the rest with no error. Split barcodes are out of scope: this pipeline reads 10x Chromium single-cell polyA assays."
    )]
    SplitSegment {
        geometry: String,
        tag: &'static str,
        count: usize,
        lens: Vec<usize>,
    },

    #[error("geometry {geometry:?} declares {tag} on R2; CB and UMI must both come from R1")]
    TechnicalOnR2 { geometry: String, tag: String },

    #[error("geometry {geometry:?} declares {levels} barcode levels; scRM supports exactly one")]
    BarcodeLevels { geometry: String, levels: usize },

    #[error("geometry {geometry:?} declares a variable-width {tag}; scRM requires a fixed width")]
    VariableWidth { geometry: String, tag: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segments<'a> {
    pub cb: &'a [u8],
    pub umi: &'a [u8],
    pub read: &'a [u8],
    pub read_offset: usize,
}

#[derive(Debug, Clone)]
pub struct Chemistry {
    geometry: String,
    compiled: CompiledGeom,
    cb_len: usize,
    umi_len: usize,
}

impl Chemistry {
    pub fn new(geometry: &str) -> Result<Self, ChemError> {
        let geom = parse_geometry(geometry).map_err(|errors| ChemError::Parse {
            geometry: geometry.to_string(),
            detail: format_errors(geometry, &errors),
        })?;
        validate_geometry(&geom).map_err(|detail| ChemError::Invalid {
            geometry: geometry.to_string(),
            detail,
        })?;
        Self::check_read_roles(geometry, &geom)?;
        Self::check_single_segments(geometry, &geom)?;

        let compiled = CompiledGeom::from_fragment_geom(&geom).map_err(|detail| ChemError::Compile {
            geometry: geometry.to_string(),
            detail,
        })?;
        let meta = compiled.meta();
        if meta.num_bc_levels != 1 {
            return Err(ChemError::BarcodeLevels {
                geometry: geometry.to_string(),
                levels: meta.num_bc_levels,
            });
        }
        let cb_len = meta.bc_lens[0];
        let umi_len = meta.umi_len;

        Ok(Self {
            geometry: geometry.to_string(),
            compiled,
            cb_len,
            umi_len,
        })
    }

    fn check_single_segments(geometry: &str, geom: &FragmentGeom) -> Result<(), ChemError> {
        let count = |want: fn(&GeoTagType) -> bool| -> Vec<usize> {
            geom.read1
                .parts
                .iter()
                .filter(|p| want(&p.tag))
                .map(|p| match p.len {
                    GeoLen::Fixed(n) => n as usize,
                    _ => 0,
                })
                .collect()
        };
        for (tag, lens) in [
            (
                "barcode",
                count(|t| matches!(t, GeoTagType::Barcode | GeoTagType::NumberedBarcode(_))),
            ),
            ("UMI", count(|t| matches!(t, GeoTagType::Umi))),
        ] {
            if lens.len() != 1 {
                return Err(ChemError::SplitSegment {
                    geometry: geometry.to_string(),
                    tag,
                    count: lens.len(),
                    lens,
                });
            }
        }
        Ok(())
    }

    fn check_read_roles(geometry: &str, geom: &FragmentGeom) -> Result<(), ChemError> {
        if geom.read1.parts.iter().any(|p| p.tag == GeoTagType::Read) {
            return Err(ChemError::ReadNotOnR2 {
                geometry: geometry.to_string(),
            });
        }
        if !geom.read2.parts.iter().any(|p| p.tag == GeoTagType::Read) {
            return Err(ChemError::ReadNotOnR2 {
                geometry: geometry.to_string(),
            });
        }
        for (tag, name) in [(GeoTagType::Barcode, "a barcode"), (GeoTagType::Umi, "a UMI")] {
            if geom.read2.parts.iter().any(|p| p.tag == tag) {
                return Err(ChemError::TechnicalOnR2 {
                    geometry: geometry.to_string(),
                    tag: name.to_string(),
                });
            }
        }
        Self::check_fixed_width(geometry, &geom.read1)?;
        Ok(())
    }

    fn check_fixed_width(geometry: &str, read: &ReadGeom) -> Result<(), ChemError> {
        for part in &read.parts {
            let is_technical = matches!(
                part.tag,
                GeoTagType::Barcode | GeoTagType::NumberedBarcode(_) | GeoTagType::Umi
            );
            if is_technical && !matches!(part.len, GeoLen::Fixed(_)) {
                return Err(ChemError::VariableWidth {
                    geometry: geometry.to_string(),
                    tag: format!("{:?}", part.tag),
                });
            }
        }
        Ok(())
    }

    pub fn geometry(&self) -> &str {
        &self.geometry
    }

    pub fn cb_len(&self) -> usize {
        self.cb_len
    }

    pub fn umi_len(&self) -> usize {
        self.umi_len
    }

    pub fn r1_technical_len(&self) -> usize {
        self.cb_len + self.umi_len
    }

    #[inline]
    pub fn extract<'a>(&self, r1: &'a [u8], r2: &'a [u8]) -> Option<Segments<'a>> {
        let seqs = self.compiled.extract(r1, r2);
        let cb = (*seqs.barcodes.first()?)?;
        let umi = seqs.umi?;
        let read = *seqs.reads.first()?;
        if cb.len() != self.cb_len || umi.len() != self.umi_len {
            return None;
        }
        let read_offset = (read.as_ptr() as usize).wrapping_sub(r2.as_ptr() as usize);
        debug_assert!(read_offset + read.len() <= r2.len(), "read is outside r2");
        Some(Segments {
            cb,
            umi,
            read,
            read_offset,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R1_28: &[u8] = b"ACGTACGTACGTACGTGGGGCCCCTTTT";
    const R2: &[u8] = b"ACGTTTTTACGGGACAAACGTGTGTACAGTCAG";

    #[test]
    fn a_split_barcode_is_rejected_not_truncated() {
        for (geometry, tag, count) in [
            ("1{b[9]u[8]b[10]}2{r:}", "barcode", 2),
            ("1{b[10]b[10]u[8]}2{r:}", "barcode", 2),
            ("1{b[16]u[6]u[6]}2{r:}", "UMI", 2),
        ] {
            match Chemistry::new(geometry) {
                Err(ChemError::SplitSegment { tag: t, count: c, .. }) => {
                    assert_eq!(t, tag, "{geometry}");
                    assert_eq!(c, count, "{geometry}");
                }
                Err(other) => panic!("{geometry}: wrong error {other}"),
                Ok(c) => panic!(
                    "{geometry} was ACCEPTED with cb_len {} umi_len {} -- this is the silent \
                     truncation the check exists to prevent",
                    c.cb_len(),
                    c.umi_len()
                ),
            }
        }
    }

    #[test]
    fn single_segment_geometries_are_unaffected() {
        for (geometry, cb, umi) in [
            ("1{b[16]u[12]}2{r:}", 16, 12),
            ("1{b[16]u[10]}2{r:}", 16, 10),
            ("1{b[16]u[10]x[124]}2{r:}", 16, 10),
        ] {
            let c = Chemistry::new(geometry).unwrap_or_else(|e| panic!("{geometry}: {e}"));
            assert_eq!((c.cb_len(), c.umi_len()), (cb, umi), "{geometry}");
        }
    }

    const VENDOR_SUPPORTED: &[(&str, usize, usize)] = &[
        ("ARC-v1", 16, 12),
        ("SC3Pv2", 16, 10),
        ("SC3Pv3-CS1", 16, 12),
        ("SC3Pv3-polyA", 16, 12),
        ("SC3Pv3HT-CS1", 16, 12),
        ("SC3Pv3HT-polyA", 16, 12),
        ("SC3Pv4-CS1", 16, 12),
        ("SC3Pv4-polyA", 16, 12),
        ("SC5P-R2", 16, 10),
        ("SC5P-R2-v3", 16, 12),
        ("SC5PHT", 16, 10),
        ("SCVDJ-R2", 16, 10),
        ("SCVDJ-R2-v3", 16, 12),
    ];

    #[test]
    fn every_in_scope_cellranger_chemistry_executes() {
        for &(name, cb, umi) in VENDOR_SUPPORTED {
            let geometry = format!("1{{b[{cb}]u[{umi}]}}2{{r:}}");
            let chem = Chemistry::new(&geometry).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(chem.cb_len(), cb, "{name}");
            assert_eq!(chem.umi_len(), umi, "{name}");
            assert_eq!(chem.r1_technical_len(), cb + umi, "{name}");
        }
        let five_prime: Vec<usize> = VENDOR_SUPPORTED
            .iter()
            .filter(|(n, _, _)| n.starts_with("SC5P") || n.starts_with("SCVDJ"))
            .map(|(_, _, u)| *u)
            .collect();
        assert!(five_prime.contains(&10) && five_prime.contains(&12), "{five_prime:?}");
    }

    #[test]
    fn the_vendor_shapes_we_do_not_support_are_refused_by_name() {
        assert!(matches!(
            Chemistry::new("1{u[9]x[2]b[14]b[14]}2{r:}"),
            Err(ChemError::SplitSegment { .. })
        ));
        assert!(matches!(
            Chemistry::new("1{b[16]u[12]}2{r[50]x[10]b[8]}"),
            Err(ChemError::TechnicalOnR2 { .. }) | Err(ChemError::ReadNotOnR2 { .. })
        ));
        assert!(matches!(
            Chemistry::new("1{b[16]u[10]r:}2{r:}"),
            Err(ChemError::ReadNotOnR2 { .. })
        ));
    }

    #[test]
    fn umi_length_comes_from_the_geometry_string_alone() {
        for (geometry, umi_len) in [("1{b[16]u[10]}2{r:}", 10), ("1{b[16]u[12]}2{r:}", 12)] {
            let chem = Chemistry::new(geometry).unwrap();
            assert_eq!(chem.cb_len(), 16, "{geometry}");
            assert_eq!(chem.umi_len(), umi_len, "{geometry}");
            assert_eq!(chem.r1_technical_len(), 16 + umi_len, "{geometry}");

            let seg = chem.extract(R1_28, R2).unwrap();
            assert_eq!(seg.cb, &R1_28[0..16], "{geometry}");
            assert_eq!(seg.umi, &R1_28[16..16 + umi_len], "{geometry}");
            assert_eq!(seg.read, R2, "{geometry}");
        }
    }

    #[test]
    fn ten_and_twelve_differ_by_exactly_two_bases() {
        let ten = Chemistry::new("1{b[16]u[10]}2{r:}").unwrap();
        let twelve = Chemistry::new("1{b[16]u[12]}2{r:}").unwrap();
        let a = ten.extract(R1_28, R2).unwrap();
        let b = twelve.extract(R1_28, R2).unwrap();
        assert_eq!(a.cb, b.cb);
        assert_eq!(b.umi.len() - a.umi.len(), 2);
        assert_eq!(&b.umi[..a.umi.len()], a.umi);
        assert_eq!(&b.umi[a.umi.len()..], &R1_28[26..28]);
    }

    #[test]
    fn explicit_trailing_discard_changes_nothing() {
        let bare = Chemistry::new("1{b[16]u[12]}2{r:}").unwrap();
        let explicit = Chemistry::new("1{b[16]u[12]x:}2{r:}").unwrap();
        assert_eq!(bare.umi_len(), explicit.umi_len());
        assert_eq!(bare.extract(R1_28, R2).unwrap(), explicit.extract(R1_28, R2).unwrap());
    }

    #[test]
    fn read_offset_tracks_a_discarded_r2_prefix() {
        let chem = Chemistry::new("1{b[16]u[12]}2{r:}").unwrap();
        assert_eq!(chem.extract(R1_28, R2).unwrap().read_offset, 0);

        let chem = Chemistry::new("1{b[16]u[12]}2{x[13]r:}").unwrap();
        let seg = chem.extract(R1_28, R2).unwrap();
        assert_eq!(seg.read_offset, 13);
        assert_eq!(seg.read, &R2[13..]);
    }

    #[test]
    fn short_r1_yields_none_rather_than_a_short_barcode() {
        let chem = Chemistry::new("1{b[16]u[12]}2{r:}").unwrap();
        assert!(chem.extract(&R1_28[..20], R2).is_none());
        assert!(chem.extract(b"", R2).is_none());
    }

    #[test]
    fn biological_read_on_r1_is_rejected() {
        let err = Chemistry::new("1{b[16]u[12]r:}2{r:}").unwrap_err();
        assert!(matches!(err, ChemError::ReadNotOnR2 { .. }), "{err:?}");
    }

    #[test]
    fn technical_pieces_on_r2_are_rejected() {
        let err = Chemistry::new("1{u[12]}2{b[16]r:}").unwrap_err();
        assert!(matches!(err, ChemError::TechnicalOnR2 { .. }), "{err:?}");
    }

    #[test]
    fn nonsense_geometry_is_a_parse_error() {
        let err = Chemistry::new("not a geometry").unwrap_err();
        assert!(matches!(err, ChemError::Parse { .. }), "{err:?}");
    }

    #[test]
    fn geometry_without_a_umi_is_invalid() {
        let err = Chemistry::new("1{b[16]}2{r:}").unwrap_err();
        assert!(matches!(err, ChemError::Invalid { .. }), "{err:?}");
    }

    #[test]
    fn variable_width_umi_is_rejected() {
        let err = Chemistry::new("1{b[16]u[10-12]f[ACGTACGTAC]}2{r:}").unwrap_err();
        assert!(matches!(err, ChemError::VariableWidth { .. }), "{err:?}");
    }

    #[test]
    fn five_prime_r1_tail_is_discarded_for_free() {
        let chem = Chemistry::new("1{b[16]u[10]}2{r:}").unwrap();
        let mut r1 = R1_28[..26].to_vec();
        r1.extend_from_slice(b"TTTCTTATATGGG");
        r1.extend_from_slice(&[b'A'; 111]);
        let seg = chem.extract(&r1, R2).unwrap();
        assert_eq!(seg.cb.len() + seg.umi.len(), 26);
        assert_eq!(seg.umi, &r1[16..26]);
    }
}
