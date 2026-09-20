use rayon::prelude::*;

use crate::{CellLoc, Corpus, PackError, Window, WindowOut, max_patches, pack, tile_first};

#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[inline]
pub fn derive_seed(epoch_seed: u64, a: u64, b: u64) -> u64 {
    let mut s = epoch_seed ^ 0xA076_1D64_78BD_642F;
    s = s.wrapping_add(a.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let _ = splitmix64(&mut s);
    s = s.wrapping_add(b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    splitmix64(&mut s)
}

fn draw_without_replacement(scratch: &mut [u32], want: usize, seed: u64, out: &mut Vec<u32>) {
    let n = scratch.len();
    let want = want.min(n);
    let mut state = seed;
    out.clear();
    out.reserve(want);
    for i in 0..want {
        let j = i + (splitmix64(&mut state) % (n - i) as u64) as usize;
        scratch.swap(i, j);
        out.push(scratch[i]);
    }
}

#[derive(Debug, Clone)]
pub struct BatchSpec {
    pub cells: Vec<u32>,
    pub v: usize,
    pub seq_len: usize,
    pub epoch_seed: u64,
    pub frac: f64,
    pub frac_seed: u64,
}

pub fn kept_molecules(n: usize, frac: f64, frac_seed: u64, cell: u64) -> Vec<u32> {
    if frac >= 1.0 {
        return (0..n as u32).collect();
    }
    let k = ((frac * n as f64).ceil() as usize).clamp(1, n);
    let mut scratch: Vec<u32> = (0..n as u32).collect();
    let mut out = Vec::new();
    draw_without_replacement(&mut scratch, k, derive_seed(frac_seed, cell, 0), &mut out);
    out.sort_unstable();
    out
}

pub fn windows_for_cell(
    lengths: &[u32],
    kept: &[u32],
    epoch_seed: u64,
    cell: u64,
    seq_len: usize,
    v: usize,
) -> Result<Vec<Window>, PackError> {
    debug_assert!(!kept.is_empty(), "caller guarantees a non-empty cell");
    let mean = (kept.iter().map(|&i| lengths[i as usize] as usize).sum::<usize>() / kept.len()).max(1);
    let per_window = seq_len / (mean + 1) + 1;
    let mut want = (per_window * v).min(kept.len());
    let seed = derive_seed(epoch_seed, cell, 0);
    let mut scratch: Vec<u32> = kept.to_vec();
    let mut order = Vec::new();
    loop {
        draw_without_replacement(&mut scratch, want, seed, &mut order);
        let w = tile_first(lengths, &order, seq_len, v)?;
        if w.len() >= v || want == kept.len() {
            return Ok(w);
        }
        want = kept.len();
        scratch.clear();
        scratch.extend_from_slice(kept);
    }
}

pub fn cells_below_v(
    corpus: &Corpus,
    epoch_seeds: &[u64],
    v: usize,
    seq_len: usize,
    frac: f64,
    frac_seed: u64,
) -> Result<Vec<u32>, PackError> {
    let hits: Result<Vec<Option<u32>>, PackError> = corpus
        .cells()
        .par_iter()
        .enumerate()
        .map(|(i, cell)| {
            let mut lengths = Vec::new();
            corpus.cell_lengths(cell, &mut lengths)?;
            if lengths.is_empty() {
                return Ok(Some(i as u32));
            }
            let kept = kept_molecules(lengths.len(), frac, frac_seed, i as u64);
            for &es in epoch_seeds {
                if windows_for_cell(&lengths, &kept, es, i as u64, seq_len, v)?.len() < v {
                    return Ok(Some(i as u32));
                }
            }
            Ok(None)
        })
        .collect();
    Ok(hits?.into_iter().flatten().collect())
}

#[derive(Debug)]
pub struct Batch {
    pub tokens: Vec<u8>,
    pub valid: Vec<bool>,
    pub contig_seg: Vec<u16>,
    pub umi_seg: Vec<u16>,
    pub cell_index: Vec<u32>,
    pub patch_starts: Vec<u32>,
    pub n_patches: Vec<u32>,
    pub n_windows: usize,
    pub seq_len: usize,
    pub short_cells: usize,
}

impl Batch {
    fn zeroed(n_windows: usize, seq_len: usize) -> Self {
        Self {
            tokens: vec![0; n_windows * seq_len],
            valid: vec![false; n_windows * seq_len],
            contig_seg: vec![0; n_windows * seq_len],
            umi_seg: vec![0; n_windows * seq_len],
            cell_index: vec![0; n_windows],
            patch_starts: vec![0; n_windows * max_patches(seq_len)],
            n_patches: vec![0; n_windows],
            n_windows,
            seq_len,
            short_cells: 0,
        }
    }

    pub fn used_tokens(&self) -> usize {
        self.valid.iter().filter(|&&v| v).count()
    }
}

struct CellOut<'a> {
    tok: &'a mut [u8],
    val: &'a mut [bool],
    cseg: &'a mut [u16],
    useg: &'a mut [u16],
    pstart: &'a mut [u32],
    npatch: &'a mut [u32],
    cidx: &'a mut [u32],
}

pub fn build_batch(corpus: &Corpus, spec: &BatchSpec) -> Result<Batch, PackError> {
    if spec.v == 0 || spec.cells.is_empty() {
        return Err(PackError::Corpus(format!(
            "a batch needs at least one cell and one view: {} cells, v = {}",
            spec.cells.len(),
            spec.v
        )));
    }
    let n_windows = spec.cells.len() * spec.v;
    let mut out = Batch::zeroed(n_windows, spec.seq_len);
    let mp = max_patches(spec.seq_len);
    let per_cell = spec.v * spec.seq_len;
    let all = corpus.cells();

    let mut views: Vec<CellOut<'_>> = Vec::with_capacity(spec.cells.len());
    {
        let mut tok = out.tokens.chunks_mut(per_cell);
        let mut val = out.valid.chunks_mut(per_cell);
        let mut cseg = out.contig_seg.chunks_mut(per_cell);
        let mut useg = out.umi_seg.chunks_mut(per_cell);
        let mut pstart = out.patch_starts.chunks_mut(spec.v * mp);
        let mut npatch = out.n_patches.chunks_mut(spec.v);
        let mut cidx = out.cell_index.chunks_mut(spec.v);
        for _ in 0..spec.cells.len() {
            views.push(CellOut {
                tok: tok.next().expect("chunk"),
                val: val.next().expect("chunk"),
                cseg: cseg.next().expect("chunk"),
                useg: useg.next().expect("chunk"),
                pstart: pstart.next().expect("chunk"),
                npatch: npatch.next().expect("chunk"),
                cidx: cidx.next().expect("chunk"),
            });
        }
    }

    let short: Result<Vec<usize>, PackError> = views
        .into_par_iter()
        .enumerate()
        .map(|(slot, o)| -> Result<usize, PackError> {
            let cell_i = spec.cells[slot];
            let loc: &CellLoc = all.get(cell_i as usize).ok_or(PackError::RowOutOfRange {
                row: cell_i,
                n: all.len(),
            })?;
            let rows = corpus.cell_rows(loc)?;
            let mut lengths = Vec::new();
            corpus.cell_lengths(loc, &mut lengths)?;
            if lengths.is_empty() {
                return Err(PackError::Corpus(format!("cell {:?} has no rows", loc.name)));
            }

            let kept = kept_molecules(lengths.len(), spec.frac, spec.frac_seed, cell_i as u64);
            let windows = windows_for_cell(&lengths, &kept, spec.epoch_seed, cell_i as u64, spec.seq_len, spec.v)?;

            let mut short = 0usize;
            for v in 0..spec.v {
                o.cidx[v] = slot as u32;
                let Some(window) = windows.get(v) else {
                    short = 1;
                    continue;
                };
                let mut w = WindowOut {
                    tokens: &mut o.tok[v * spec.seq_len..(v + 1) * spec.seq_len],
                    valid: &mut o.val[v * spec.seq_len..(v + 1) * spec.seq_len],
                    contig_seg: &mut o.cseg[v * spec.seq_len..(v + 1) * spec.seq_len],
                    umi_seg: &mut o.useg[v * spec.seq_len..(v + 1) * spec.seq_len],
                    patch_starts: &mut o.pstart[v * mp..(v + 1) * mp],
                };
                let mut sc: Vec<u32> = Vec::new();
                let st = pack(
                    window,
                    &rows,
                    |row, n, buf| {
                        let s = derive_seed(spec.epoch_seed, cell_i as u64, row as u64 + 1);
                        sc.clear();
                        sc.extend(0..n as u32);
                        draw_without_replacement(&mut sc, n, s, buf);
                    },
                    &mut w,
                    spec.seq_len,
                )?;
                o.npatch[v] = st.n_patches as u32;
            }
            Ok(short)
        })
        .collect();

    out.short_cells = short?.into_iter().sum();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_seeds_do_not_collide_under_coordinate_swaps() {
        assert_ne!(derive_seed(1, 3, 7), derive_seed(1, 7, 3));
        assert_ne!(derive_seed(1, 0, 0), derive_seed(2, 0, 0));
        assert_ne!(derive_seed(1, 0, 1), derive_seed(1, 1, 0));
        assert_eq!(derive_seed(42, 5, 9), derive_seed(42, 5, 9));
    }

    #[test]
    fn a_draw_is_distinct_and_reproducible() {
        let mut a: Vec<u32> = (0..1000).collect();
        let mut b: Vec<u32> = (0..1000).collect();
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        draw_without_replacement(&mut a, 300, 12345, &mut oa);
        draw_without_replacement(&mut b, 300, 12345, &mut ob);
        assert_eq!(oa, ob, "the same seed must give the same draw");
        assert_eq!(oa.len(), 300);
        let mut sorted = oa.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 300, "a draw must be without replacement");
        assert!(sorted.iter().all(|&x| x < 1000));
    }

    #[test]
    fn a_different_seed_gives_a_different_draw() {
        let mut a: Vec<u32> = (0..1000).collect();
        let mut b: Vec<u32> = (0..1000).collect();
        let (mut oa, mut ob) = (Vec::new(), Vec::new());
        draw_without_replacement(&mut a, 300, 1, &mut oa);
        draw_without_replacement(&mut b, 300, 2, &mut ob);
        assert_ne!(oa, ob);
    }

    #[test]
    fn drawing_everything_is_a_permutation() {
        let mut s: Vec<u32> = (0..64).collect();
        let mut out = Vec::new();
        draw_without_replacement(&mut s, 64, 7, &mut out);
        let mut sorted = out.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..64).collect::<Vec<_>>());
    }

    #[test]
    fn drawing_more_than_exists_is_clamped_rather_than_looping() {
        let mut s: Vec<u32> = (0..10).collect();
        let mut out = Vec::new();
        draw_without_replacement(&mut s, 999, 7, &mut out);
        assert_eq!(out.len(), 10);
    }

    #[test]
    fn kept_molecules_at_full_budget_is_the_identity() {
        assert_eq!(kept_molecules(5, 1.0, 123, 7), vec![0, 1, 2, 3, 4]);
        assert_eq!(kept_molecules(5, 1.5, 123, 7), vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn kept_molecules_is_a_pure_function_of_frac_seed_and_cell_and_never_of_the_epoch() {
        let a = kept_molecules(1000, 0.25, 42, 7);
        assert_eq!(a, kept_molecules(1000, 0.25, 42, 7), "pure");
        assert_eq!(a.len(), 250, "ceil(frac * n)");
        assert_ne!(a, kept_molecules(1000, 0.25, 43, 7), "the seed matters");
        assert_ne!(a, kept_molecules(1000, 0.25, 42, 8), "the cell matters");
        assert!(a.windows(2).all(|w| w[0] < w[1]), "sorted, distinct");
        assert!(a.iter().all(|&i| i < 1000));
    }

    #[test]
    fn kept_molecules_never_returns_an_empty_subset() {
        assert_eq!(kept_molecules(9, 0.001, 1, 1).len(), 1);
    }

    #[test]
    fn windows_for_cell_is_deterministic_and_respects_the_kept_subset() {
        let lengths = vec![100u32; 40];
        let kept = kept_molecules(40, 0.5, 9, 3);
        assert_eq!(kept.len(), 20);
        let a = windows_for_cell(&lengths, &kept, 77, 3, 512, 2).unwrap();
        let b = windows_for_cell(&lengths, &kept, 77, 3, 512, 2).unwrap();
        assert_eq!(a, b, "pure function of (epoch_seed, cell, kept)");
        assert_eq!(a.len(), 2, "tile_first stops at v");
        assert_eq!(a.iter().map(Vec::len).collect::<Vec<_>>(), vec![5, 5]);
        let kept_set: std::collections::HashSet<u32> = kept.iter().copied().collect();
        for w in &a {
            assert!(
                w.iter().all(|m| kept_set.contains(m)),
                "a window may only hold kept molecules"
            );
        }
        let mut all: Vec<u32> = a.iter().flatten().copied().collect();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), 10, "the two views share no molecule");
        let c = windows_for_cell(&lengths, &kept, 78, 3, 512, 2).unwrap();
        assert_ne!(a, c, "a different epoch draws different views");
    }

    #[test]
    fn windows_for_cell_retries_with_the_whole_subset_when_the_mean_underestimates() {
        let mut lengths = vec![10u32; 50];
        lengths[0] = 500;
        lengths[1] = 500;
        let kept: Vec<u32> = (0..50).collect();
        let (epoch_seed, cell, seq_len, v) = (6u64, 2u64, 512usize, 2usize);

        let mean = (kept.iter().map(|&i| lengths[i as usize] as usize).sum::<usize>() / kept.len()).max(1);
        let per_window = seq_len / (mean + 1) + 1;
        let want = (per_window * v).min(kept.len());
        assert_eq!(
            (mean, per_window, want),
            (29, 18, 36),
            "the fixture's sizing arithmetic"
        );
        assert!(
            want < kept.len(),
            "the retry is unreachable unless the first draw is a subset"
        );
        let mut scratch = kept.clone();
        let mut order = Vec::new();
        draw_without_replacement(&mut scratch, want, derive_seed(epoch_seed, cell, 0), &mut order);
        let first = tile_first(&lengths, &order, seq_len, v).unwrap();
        assert_eq!(
            first.len(),
            1,
            "PRECONDITION: this seed's first draw must UNDER-FILL, or the retry never runs"
        );

        let w = windows_for_cell(&lengths, &kept, epoch_seed, cell, seq_len, v).unwrap();
        assert_eq!(w.len(), v, "the retry must reach v windows, not merely more than one");
        let mut scratch = kept.clone();
        let mut order = Vec::new();
        draw_without_replacement(&mut scratch, kept.len(), derive_seed(epoch_seed, cell, 0), &mut order);
        assert_eq!(
            w,
            tile_first(&lengths, &order, seq_len, v).unwrap(),
            "the retry draws the WHOLE subset under the same seed -- not a second guess"
        );
        assert_eq!(w.iter().map(Vec::len).collect::<Vec<_>>(), vec![41, 2]);
    }
}
