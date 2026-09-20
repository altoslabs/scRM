"""The baseline panel, declared once, with no dependencies.

`controls.py` re-exports `PANEL` from here, and `eval/countfm/embed_count_fm.py` imports it
directly. That script runs under the foundation models' own interpreter, which has their
dependencies and not this package's, so it cannot import `controls` -- and a second copy of
the list is what `eval/tests/test_no_hardcoded_lists.py` exists to refuse. A module with no
imports is the only thing both can read.
"""

from __future__ import annotations

#: The count-based foundation models the tables compare against, in the paper's order.
PANEL: tuple[str, ...] = ("geneformer", "scgpt", "scvi_pretrained", "genejepa", "uce")

#: Those `eval/countfm/embed_count_fm.py` embeds itself. UCE is fetched and run separately --
#: `measurements/uce_*.py` and `model/jobs/uce_matrix.sbatch` -- so it is not one of these.
EMBEDDABLE: tuple[str, ...] = tuple(m for m in PANEL if m != "uce")
