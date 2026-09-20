use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Experiment {
    pub name: String,
    pub lanes: Vec<(PathBuf, PathBuf)>,
    pub barcodes: PathBuf,
    pub geometry: String,
}

impl Experiment {
    pub fn r1(&self) -> Vec<PathBuf> {
        self.lanes.iter().map(|(a, _)| a.clone()).collect()
    }
    pub fn r2(&self) -> Vec<PathBuf> {
        self.lanes.iter().map(|(_, b)| b.clone()).collect()
    }
}

const COLUMNS: [&str; 5] = ["experiment", "r1", "r2", "barcodes", "geometry"];

const MAX_NAME: usize = 160;

fn check_name(name: &str, line: usize) -> Result<()> {
    if name.is_empty() {
        bail!("line {line}: empty experiment name");
    }
    if name.len() > MAX_NAME {
        bail!(
            "line {line}: experiment name is {} bytes, over the {MAX_NAME} limit",
            name.len()
        );
    }
    if let Some(c) = name
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || *c == '_' || *c == '-' || *c == '.'))
    {
        bail!(
            "line {line}: experiment name {name:?} contains {c:?}; it becomes a directory and a \
             shard filename, so only ASCII letters, digits, '_', '-' and '.' are allowed"
        );
    }
    if name.starts_with('.') {
        bail!("line {line}: experiment name {name:?} starts with '.', which would hide the corpus");
    }
    Ok(())
}

pub fn read(path: &Path) -> Result<Vec<Experiment>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading samplesheet {}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    parse(&text, base)
}

pub fn parse(text: &str, base: &Path) -> Result<Vec<Experiment>> {
    let resolve = |s: &str| -> PathBuf {
        let p = Path::new(s);
        if p.is_absolute() { p.to_path_buf() } else { base.join(p) }
    };

    let mut rows = text.lines().enumerate().map(|(i, l)| (i + 1, l.trim()));

    let (header_line, header) = rows
        .by_ref()
        .find(|(_, l)| !l.is_empty() && !l.starts_with('#'))
        .ok_or_else(|| anyhow::anyhow!("samplesheet is empty: expected a header {COLUMNS:?}"))?;

    let names: Vec<&str> = header.split(',').map(str::trim).collect();
    let mut index: HashMap<&str, usize> = HashMap::new();
    for (i, n) in names.iter().enumerate() {
        if index.insert(n, i).is_some() {
            bail!("line {header_line}: column {n:?} appears twice in the header");
        }
    }
    let missing: Vec<&str> = COLUMNS.iter().copied().filter(|c| !index.contains_key(c)).collect();
    if !missing.is_empty() {
        bail!("line {header_line}: samplesheet header is missing {missing:?}; found {names:?}");
    }
    let unknown: Vec<&&str> = names.iter().filter(|n| !COLUMNS.contains(n)).collect();
    if !unknown.is_empty() {
        bail!("line {header_line}: samplesheet header has unknown columns {unknown:?}");
    }
    let col = |row: &Vec<&str>, name: &str| -> String { row[index[name]].to_string() };

    let mut order: Vec<String> = Vec::new();
    let mut by_name: HashMap<String, Experiment> = HashMap::new();
    let mut seen_input: HashMap<PathBuf, (usize, &'static str)> = HashMap::new();

    for (line, raw) in rows {
        if raw.is_empty() || raw.starts_with('#') {
            continue;
        }
        let row: Vec<&str> = raw.split(',').map(str::trim).collect();
        if row.len() != names.len() {
            bail!("line {line}: has {} fields, header declares {}", row.len(), names.len());
        }
        let name = col(&row, "experiment");
        check_name(&name, line)?;
        let (r1s, r2s, bcs) = (col(&row, "r1"), col(&row, "r2"), col(&row, "barcodes"));
        let geometry = col(&row, "geometry");
        for (label, v) in [("r1", &r1s), ("r2", &r2s), ("barcodes", &bcs), ("geometry", &geometry)] {
            if v.is_empty() {
                bail!("line {line}: empty {label}");
            }
        }
        let (r1, r2, barcodes) = (resolve(&r1s), resolve(&r2s), resolve(&bcs));
        if r1 == r2 {
            bail!("line {line}: r1 and r2 are the same file, {}", r1.display());
        }
        for (label, path) in [("r1", &r1), ("r2", &r2)] {
            if let Some((first, first_label)) = seen_input.insert(path.clone(), (line, label)) {
                bail!(
                    "line {line}: {label} {} was already listed on line {first} (as {first_label}); \
                     reading a lane twice doubles every molecule it contains and no later check can \
                     detect it",
                    path.display()
                );
            }
        }

        match by_name.get_mut(&name) {
            None => {
                order.push(name.clone());
                by_name.insert(
                    name,
                    Experiment {
                        name: row[index["experiment"]].to_string(),
                        lanes: vec![(r1, r2)],
                        barcodes,
                        geometry,
                    },
                );
            }
            Some(exp) => {
                if exp.barcodes != barcodes {
                    bail!(
                        "line {line}: experiment {name:?} already declared barcodes {}, this row \
                         says {} -- one experiment has one cell list",
                        exp.barcodes.display(),
                        barcodes.display()
                    );
                }
                if exp.geometry != geometry {
                    bail!(
                        "line {line}: experiment {name:?} already declared geometry {:?}, this row \
                         says {geometry:?} -- one experiment's lanes are one library",
                        exp.geometry
                    );
                }
                exp.lanes.push((r1, r2));
            }
        }
    }

    if order.is_empty() {
        bail!("samplesheet has a header but no rows");
    }
    let experiments: Vec<Experiment> = order.into_iter().map(|n| by_name.remove(&n).unwrap()).collect();

    for exp in &experiments {
        for (r1, r2) in &exp.lanes {
            for p in [r1, r2] {
                if !p.is_file() {
                    bail!("experiment {:?}: {} does not exist", exp.name, p.display());
                }
            }
        }
        if !exp.barcodes.is_file() {
            bail!(
                "experiment {:?}: barcodes {} does not exist",
                exp.name,
                exp.barcodes.display()
            );
        }
    }
    Ok(experiments)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> &'static Path {
        Path::new("/tmp/does-not-matter")
    }

    fn parse_paths(text: &str) -> Result<Vec<Experiment>> {
        match parse(text, base()) {
            Ok(v) => Ok(v),
            Err(e) if e.to_string().contains("does not exist") => Err(e),
            Err(e) => Err(e),
        }
    }

    const HEADER: &str = "experiment,r1,r2,barcodes,geometry";
    const GEOM: &str = "1{b[16]u[12]}2{r:}";

    #[test]
    fn a_missing_column_is_named() {
        let err = parse("experiment,r1,r2,barcodes\n", base()).unwrap_err().to_string();
        assert!(err.contains("missing [\"geometry\"]"), "{err}");
    }

    #[test]
    fn an_unknown_column_is_named() {
        let err = parse(&format!("{HEADER},lane\n"), base()).unwrap_err().to_string();
        assert!(err.contains("unknown columns"), "{err}");
    }

    #[test]
    fn column_order_is_free() {
        let a = parse_paths(&format!("{HEADER}\ne,r1.fq,r2.fq,bc.tsv,{GEOM}\n"));
        let b = parse_paths(&format!(
            "geometry,barcodes,r2,r1,experiment\n{GEOM},bc.tsv,r2.fq,r1.fq,e\n"
        ));
        assert_eq!(a.unwrap_err().to_string(), b.unwrap_err().to_string());
    }

    #[test]
    fn comments_and_blank_lines_are_skipped_and_line_numbers_still_match_the_file() {
        let text = format!("# a comment\n\n{HEADER}\n\n# another\nname with space,a,b,c,{GEOM}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(err.starts_with("line 6:"), "expected the real line number, got: {err}");
    }

    #[test]
    fn a_row_with_the_wrong_field_count_is_rejected() {
        let err = parse(&format!("{HEADER}\ne,r1.fq,r2.fq,bc\n"), base())
            .unwrap_err()
            .to_string();
        assert!(err.contains("has 4 fields, header declares 5"), "{err}");
    }

    #[test]
    fn r1_equal_to_r2_is_rejected() {
        let err = parse(&format!("{HEADER}\ne,same.fq,same.fq,bc,{GEOM}\n"), base())
            .unwrap_err()
            .to_string();
        assert!(err.contains("r1 and r2 are the same file"), "{err}");
    }

    #[test]
    fn the_same_lane_listed_twice_is_rejected() {
        let text = format!("{HEADER}\ne,a.fq,b.fq,bc,{GEOM}\ne,a.fq,c.fq,bc,{GEOM}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(err.contains("already listed on line 2"), "{err}");
        assert!(
            err.contains("doubles every molecule"),
            "the reason must be in the message: {err}"
        );
    }

    #[test]
    fn the_same_r2_listed_twice_is_rejected() {
        let text = format!("{HEADER}\ne,a.fq,shared.fq,bc,{GEOM}\ne,c.fq,shared.fq,bc,{GEOM}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(err.contains("already listed on line 2"), "{err}");
        assert!(
            err.contains("doubles every molecule"),
            "the reason must be in the message: {err}"
        );
    }

    #[test]
    fn a_path_used_as_both_r1_and_r2_is_rejected() {
        let text = format!("{HEADER}\ne,a.fq,b.fq,bc,{GEOM}\ne,c.fq,a.fq,bc,{GEOM}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(err.contains("already listed on line 2"), "{err}");
    }

    #[test]
    fn two_barcode_lists_for_one_experiment_are_rejected() {
        let text = format!("{HEADER}\ne,a.fq,b.fq,bc1,{GEOM}\ne,c.fq,d.fq,bc2,{GEOM}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(err.contains("one experiment has one cell list"), "{err}");
    }

    #[test]
    fn an_experiment_name_that_is_not_filename_safe_is_rejected_not_mangled() {
        for bad in ["a/b", "a b", "a:b", ".hidden", ""] {
            let text = format!("{HEADER}\n{bad},a.fq,b.fq,bc,{GEOM}\n");
            let err = parse(&text, base()).unwrap_err().to_string();
            assert!(err.starts_with("line 2:"), "for {bad:?}: {err}");
        }
    }

    #[test]
    fn a_name_ending_in_a_number_survives_verbatim() {
        let text = format!("{HEADER}\nSOD1_G93A_mouse_spinal_cord_P112_specimen_1,a.fq,b.fq,bc,{GEOM}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(
            err.contains("specimen_1\": "),
            "the name must reach the existence check intact: {err}"
        );
    }

    #[test]
    fn a_header_with_no_rows_is_rejected() {
        let err = parse(&format!("{HEADER}\n\n# nothing\n"), base())
            .unwrap_err()
            .to_string();
        assert!(err.contains("no rows"), "{err}");
    }

    #[test]
    fn an_empty_sheet_is_rejected() {
        let err = parse("\n# only comments\n", base()).unwrap_err().to_string();
        assert!(err.contains("expected a header"), "{err}");
    }

    #[test]
    fn two_geometries_for_one_experiment_are_rejected() {
        let text = format!("{HEADER}\ne,a.fq,b.fq,bc,1{{b[16]u[12]}}2{{r:}}\ne,c.fq,d.fq,bc,1{{b[16]u[10]}}2{{r:}}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(err.contains("one experiment's lanes are one library"), "{err}");
    }

    #[test]
    fn two_experiments_may_declare_different_geometries() {
        let text =
            format!("{HEADER}\na,a1.fq,a2.fq,bc,1{{b[16]u[12]}}2{{r:}}\nb,b1.fq,b2.fq,bc,1{{b[16]u[10]}}2{{r:}}\n");
        let err = parse(&text, base()).unwrap_err().to_string();
        assert!(err.contains("does not exist"), "must reach the existence check: {err}");
    }

    #[test]
    fn an_empty_geometry_is_rejected() {
        let err = parse(&format!("{HEADER}\ne,a.fq,b.fq,bc,\n"), base())
            .unwrap_err()
            .to_string();
        assert!(err.contains("empty geometry"), "{err}");
    }

    #[test]
    fn a_duplicate_header_column_is_rejected() {
        let err = parse("experiment,r1,r1,barcodes,geometry\n", base())
            .unwrap_err()
            .to_string();
        assert!(err.contains("appears twice"), "{err}");
    }
}
