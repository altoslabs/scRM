# Third-party notices

This project is MIT (see `LICENSE`). The notices below are owed to third parties whose **code**
this tree reproduces or is closely modelled on. Methods merely *cited* are credited at their use
site and in the paper's bibliography; they are not repeated here. This file exists because the
per-module header is not a place a licence notice can live durably, and because two of the
notices below were owed and missing.

## facebookresearch/lingua — BSD-3-Clause

`model/scrm_model/trunk.py` shares distinctive expression with `lingua/transformer.py`:
the 2x2 real rotation packing `torch.stack((cos, -sin, sin, cos), dim=-1).view(..., 2, 2)`
(the llama reference uses a complex `torch.polar` representation instead, so this form is that
codebase's, not the paper's), the SwiGLU hidden-width rule `int(2 * (4 * dim) / 3)` rounded up to
`multiple_of`, and the `precompute_freqs_cis` / `apply_rotary_emb` / `reset_parameters(init_std,
factor)` naming and signatures.

    BSD 3-Clause License

    Copyright 2024 Meta

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
       list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
       this list of conditions and the following disclaimer in the documentation
       and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its
       contributors may be used to endorse or promote products derived from
       this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## fulcrumgenomics/fgbio — MIT

`crates/scrm-asm/src/consensus.rs` reimplements fgbio's `ConsensusCaller` /
`CallMolecularConsensusReads` quality-weighted consensus model, and carries its parameter
vocabulary by name: `MIN_CONSENSUS_Q`, `OMEGA_LOG10`, `ERROR_RATE_PRE`, `ERROR_RATE_POST`.

    The MIT License (MIT)

    Copyright (c) 2019 Fulcrum Genomics LLC

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

## COMBINE-lab/libradicl — BSD-3-Clause

A build dependency, pinned by git rev in `Cargo.toml`, used by `crates/scrm-correct` for
`BarcodeLookupMap`. Copyright (c) 2022, COMBINE lab. Its full licence text ships with the crate
source that `cargo` fetches.

## Methods credited at their use site, not here

Each of these is either a published method reimplemented here or a library called as a dependency.
None is vendored code, so what is owed at the use site is a citation, and a dependency's own licence
text ships with the package `cargo` or `pip` fetches. The use site is named here too, and so is the
header that carries the citation, so the deferral is checkable rather than a promise:

- cutadapt (MIT) for the suffix-anchored 3' adapter rule — `crates/scrm-trim/src/adapter.rs` —
  and BWA's quality trimming rule as cutadapt documents it — `crates/scrm-trim/src/quality.rs`.
  Both written from the documentation rather than the source; the crate header in
  `crates/scrm-trim/src/lib.rs` says which modules are ours and which are not.
- alevin-fry's `ExplicitList` permit-list rule — `crates/scrm-correct/src/lib.rs`. The Hamming-1
  neighbour query under it is `libradicl`'s `BarcodeLookupMap`, a dependency, noticed above.
- STARsolo's sort/swap kernel for Hamming-1 candidate pairs (alexdobin/STAR) — implemented in
  `crates/scrm-umi/src/pairs.rs` and credited in that crate's header,
  `crates/scrm-umi/src/lib.rs`.
- CellRanger's `1MM_CR_flow` UMI rule — `crates/scrm-umi/src/flow.rs`.
- the `debruijn` crate (10XGenomics/rust-debruijn, MIT), a dependency used at
  `crates/scrm-asm/src/graph.rs` and credited in `crates/scrm-asm/src/lib.rs`; its own licence text
  ships with the crate source `cargo` fetches.
- fgbio's `ConsensusCaller` — `crates/scrm-asm/src/consensus.rs`, whose header carries it; the MIT
  notice itself is above.
- scanpy, decoupler/AUCell, MSigDB and scib on the evaluation side — `eval/scrm_eval/targets.py`,
  `eval/scrm_eval/metrics.py`, `eval/scrm_eval/controls.py`, `eval/scrm_eval/run_probes.py` and
  `measurements/uce_*.py`. `requirements.txt` pins the versions this tree is verified against and
  says they are floors this tree was checked at, not independently established minima.
- BLT (arXiv:2412.09871) for the byte-to-patch design — `model/scrm_model/trunk.py`, whose header
  also itemises the expression it shares with facebookresearch/lingua and the BSD-3-Clause notice
  that requires.
- LeJEPA (arXiv:2511.08544) and Epps & Pulley 1983, Biometrika 70(3) for the SIGReg objective —
  `model/scrm_model/losses.py`, whose header separates what the papers specify from what is ours.
