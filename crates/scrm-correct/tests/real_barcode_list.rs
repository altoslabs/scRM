use std::path::PathBuf;

use scrm_core::pack_acgt;
use scrm_correct::{Correction, Corrector};

const PROJ_DEFAULT: &str = "";
const LIST_REL: &str = "whitelists/mouse_brain_5k/filtered_feature_bc_matrix/barcodes.tsv.gz";

fn proj_root() -> PathBuf {
    PathBuf::from(
        std::env::var_os("SCRM_PROJ")
            .filter(|v| !v.is_empty())
            .unwrap_or(PROJ_DEFAULT.into()),
    )
}

fn list_path() -> PathBuf {
    proj_root().join(LIST_REL)
}

#[test]
fn loads_and_corrects_against_a_real_cellranger_list() {
    let p = list_path();
    if !p.exists() {
        let proj = proj_root();
        assert!(
            !proj.is_dir(),
            "{} is missing although the project root {} exists -- a relocated or incomplete \
             whitelist tree, not a checkout without the data. Reporting `ok` here would hide it.",
            p.display(),
            proj.display()
        );
        eprintln!("SKIP: {} not present", p.display());
        return;
    }
    let c = Corrector::from_filtered_matrix(&p, 16).expect("real list must load");

    assert_eq!(c.n_barcodes, 7377, "barcode count changed -- did the dataset change?");
    assert_eq!(c.list_sha256.len(), 64);

    let first = c.render(scrm_core::CellId(0));
    assert_eq!(first.len(), 16);

    let known = b"AAACCCAAGGCATCAG";
    let q = pack_acgt(known).unwrap();
    assert!(
        matches!(c.correct(q), Correction::Exact(_)),
        "a barcode from the list must match exactly"
    );

    let mut m = known.to_vec();
    m[15] = b'T';
    let qm = pack_acgt(&m).unwrap();
    match c.correct(qm) {
        Correction::Corrected(cell) => assert_eq!(c.render(cell), known.to_vec()),
        other => panic!("this neighbour is unambiguous in this list; expected Corrected, got {other:?}"),
    }
    let cell = c.correct(qm).cell().expect("a corrected barcode names a cell");
    assert!(
        c.rank(cell).is_some_and(|r| r < c.n_barcodes as u32),
        "the corrected cell must have a rank inside the called list"
    );

    let far = pack_acgt(b"TTTTTTTTTTTTTTTT").unwrap();
    assert!(!c.correct(far).is_kept());
}
