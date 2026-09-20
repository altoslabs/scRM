use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use scrm_pack::{BatchSpec, Corpus as PackCorpus, assert_geometry, build_batch, max_patches};

fn err(e: scrm_pack::PackError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pyclass(module = "scrm_py", frozen)]
pub struct Batch {
    #[pyo3(get)]
    tokens: Py<PyArray2<u8>>,
    #[pyo3(get)]
    valid: Py<PyArray2<bool>>,
    #[pyo3(get)]
    contig_segment_id: Py<PyArray2<u16>>,
    #[pyo3(get)]
    umi_segment_id: Py<PyArray2<u16>>,
    #[pyo3(get)]
    cell_index: Py<PyArray1<u32>>,
    #[pyo3(get)]
    patch_start_ids: Py<PyArray2<u32>>,
    #[pyo3(get)]
    n_patches: Py<PyArray1<u32>>,
    #[pyo3(get)]
    n_windows: usize,
    #[pyo3(get)]
    seq_len: usize,
    #[pyo3(get)]
    short_cells: usize,
}

#[pyclass(module = "scrm_py", frozen)]
pub struct Corpus {
    inner: PackCorpus,
    seq_len: usize,
    min_v: usize,
}

#[pymethods]
impl Corpus {
    #[new]
    #[pyo3(signature = (path, *, seq_len = 16384, max_umi_len = None, min_v = 2))]
    fn new(path: &Bound<'_, PyAny>, seq_len: usize, max_umi_len: Option<usize>, min_v: usize) -> PyResult<Self> {
        if min_v == 0 {
            return Err(PyValueError::new_err("min_v must be at least 1"));
        }
        let dirs: Vec<String> = if let Ok(one) = path.extract::<String>() {
            vec![one]
        } else {
            path.extract::<Vec<String>>()
                .map_err(|_| PyValueError::new_err("path must be a str or a list of str (one per corpus directory)"))?
        };
        let paths: Vec<&std::path::Path> = dirs.iter().map(|d| std::path::Path::new(d.as_str())).collect();
        let inner = PackCorpus::open_many(&paths).map_err(err)?;
        assert_geometry(max_umi_len.unwrap_or(inner.compat_key().max_umi_len), seq_len).map_err(err)?;
        Ok(Self { inner, seq_len, min_v })
    }

    fn compat_key<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let k = self.inner.compat_key();
        let d = pyo3::types::PyDict::new(py);
        d.set_item("alphabet_size", k.alphabet_size)?;
        d.set_item("k", k.k)?;
        d.set_item("dedup_rule", &k.dedup_rule)?;
        d.set_item("bubble_resolution", &k.bubble_resolution)?;
        d.set_item("min_umi_len", k.min_umi_len)?;
        d.set_item("max_umi_len", k.max_umi_len)?;
        d.set_item("min_cell_molecules", k.min_cell_molecules)?;
        Ok(d)
    }

    fn cell_names(&self) -> Vec<String> {
        self.inner.cells().iter().map(|c| c.name.clone()).collect()
    }

    fn n_cells(&self) -> usize {
        self.inner.cells().len()
    }

    fn molecules_per_cell<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner
            .cells()
            .iter()
            .map(|c| c.n_rows)
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }

    #[pyo3(signature = (epoch_seeds, *, v = None, frac = 1.0, frac_seed = 0))]
    fn cells_below_v(
        &self,
        py: Python<'_>,
        epoch_seeds: Vec<u64>,
        v: Option<usize>,
        frac: f64,
        frac_seed: u64,
    ) -> PyResult<Vec<u32>> {
        let v = v.unwrap_or(self.min_v);
        if epoch_seeds.is_empty() {
            return Err(PyValueError::new_err(
                "cells_below_v needs the run's epoch seeds; an empty list would vacuously pass every cell",
            ));
        }
        let inner = &self.inner;
        let seq_len = self.seq_len;
        py.detach(|| scrm_pack::cells_below_v(inner, &epoch_seeds, v, seq_len, frac, frac_seed))
            .map_err(err)
    }

    fn cells_below_min_v(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        let seq_len = self.seq_len;
        let min_v = self.min_v;
        let inner = &self.inner;
        py.detach(|| {
            let mut short = Vec::new();
            let mut lengths = Vec::new();
            for (i, cell) in inner.cells().iter().enumerate() {
                inner.cell_lengths(cell, &mut lengths).map_err(err)?;
                let order: Vec<u32> = (0..lengths.len() as u32).collect();
                let w = scrm_pack::tile_first(&lengths, &order, seq_len, min_v).map_err(err)?;
                if w.len() < min_v {
                    short.push(i as u32);
                }
            }
            Ok(short)
        })
    }

    #[pyo3(signature = (cells, *, epoch_seed, v = 2, allow_short = false, frac = 1.0, frac_seed = 0))]
    #[allow(clippy::too_many_arguments)]
    fn build_batch(
        &self,
        py: Python<'_>,
        cells: Vec<u32>,
        epoch_seed: u64,
        v: usize,
        allow_short: bool,
        frac: f64,
        frac_seed: u64,
    ) -> PyResult<Batch> {
        let n = self.inner.cells().len();
        if let Some(bad) = cells.iter().find(|&&c| c as usize >= n) {
            return Err(PyValueError::new_err(format!(
                "cell index {bad} is outside the corpus's {n} cells"
            )));
        }
        if !frac.is_finite() || frac <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "frac must be positive and finite, got {frac}"
            )));
        }
        let spec = BatchSpec {
            cells,
            v,
            seq_len: self.seq_len,
            epoch_seed,
            frac,
            frac_seed,
        };
        let batch = py.detach(|| build_batch(&self.inner, &spec)).map_err(err)?;
        if batch.short_cells > 0 && !allow_short {
            return Err(PyRuntimeError::new_err(format!(
                "{} of {} cells could not supply v={v} windows. A cell with fewer views than \
                 requested gives the EMA objective nothing to compare, and the loss cannot see \
                 that. Call cells_below_min_v() before training and exclude them, or pass \
                 allow_short=True for eval semantics (design 12.4).",
                batch.short_cells,
                spec_len(&spec)
            )));
        }

        let (nw, sl) = (batch.n_windows, batch.seq_len);
        let mp = max_patches(sl);
        Ok(Batch {
            tokens: two_d(py, batch.tokens, nw, sl)?,
            valid: two_d(py, batch.valid, nw, sl)?,
            contig_segment_id: two_d(py, batch.contig_seg, nw, sl)?,
            umi_segment_id: two_d(py, batch.umi_seg, nw, sl)?,
            cell_index: batch.cell_index.into_pyarray(py).unbind(),
            patch_start_ids: two_d(py, batch.patch_starts, nw, mp)?,
            n_patches: batch.n_patches.into_pyarray(py).unbind(),
            n_windows: nw,
            seq_len: sl,
            short_cells: batch.short_cells,
        })
    }
}

fn spec_len(spec: &BatchSpec) -> usize {
    spec.cells.len()
}

fn two_d<T: numpy::Element>(py: Python<'_>, data: Vec<T>, rows: usize, cols: usize) -> PyResult<Py<PyArray2<T>>> {
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), data)
        .map_err(|e| PyRuntimeError::new_err(format!("internal shape error for {rows}x{cols}: {e}")))?;
    Ok(PyArray2::from_owned_array(py, arr).unbind())
}

#[pyfunction]
fn resolution_name() -> String {
    scrm_asm::resolution_name()
}

#[pyfunction]
#[pyo3(name = "max_patches")]
fn py_max_patches(seq_len: usize) -> usize {
    max_patches(seq_len)
}

#[pymodule]
fn scrm_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Corpus>()?;
    m.add_class::<Batch>()?;
    m.add_function(wrap_pyfunction!(py_max_patches, m)?)?;
    m.add_function(wrap_pyfunction!(resolution_name, m)?)?;
    m.add("MAX_UMI_LEN_DEFAULT", scrm_core::MAX_UMI_LEN_DEFAULT)?;
    m.add("MIN_UMI_LEN_DEFAULT", scrm_core::MIN_UMI_LEN_DEFAULT)?;
    m.add("MIN_CELL_MOLECULES_DEFAULT", scrm_core::MIN_CELL_MOLECULES_DEFAULT)?;
    Ok(())
}
