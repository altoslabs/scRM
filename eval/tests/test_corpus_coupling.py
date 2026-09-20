"""The trained-vs-embedded molecule definition coupling."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "eval"))

from scrm_eval import embed  # noqa: E402

LIKELIHOOD = {
    "alphabet_size": 7, "k": 19, "dedup_rule": "1MM_CR_flow",
    "bubble_resolution": "likelihood-q10-pre45-post40-omega7",
    "min_umi_len": 60, "max_umi_len": 1600, "min_cell_molecules": 500,
}
RETIRED = dict(LIKELIHOOD, bubble_resolution="weighted_split")


class _Ckpt:
    def __init__(self, key):
        self.corpus_compat_key = key


class _Corpus:
    """A corpus that can state its own molecule definition, which is all the check asks of one."""

    def __init__(self, key):
        self._key = key

    def compat_key(self):
        return self._key


class _MuteCorpus:
    """A corpus object that cannot state its identity."""


def test_a_disagreeing_molecule_definition_is_refused():
    """Two disagreeing molecule definitions must never be embedded as one."""
    with pytest.raises(embed.EmbedError) as e:
        embed._assert_trained_and_embedded_corpora_agree(_Ckpt(RETIRED), _Corpus(LIKELIHOOD))
    msg = str(e.value)
    assert "REFUSING" in msg
    assert "bubble_resolution" in msg and "weighted_split" in msg
    assert "likelihood-q10-pre45-post40-omega7" in msg


def test_the_same_definition_passes():
    embed._assert_trained_and_embedded_corpora_agree(_Ckpt(dict(LIKELIHOOD)), _Corpus(dict(LIKELIHOOD)))


def test_any_field_of_the_definition_counts_not_just_the_bubble_rule():
    """`k` and the length bounds must define a molecule as much as the consensus rule does."""
    for field, other in (("k", 23), ("min_umi_len", 40), ("dedup_rule", "exact")):
        with pytest.raises(embed.EmbedError):
            embed._assert_trained_and_embedded_corpora_agree(
                _Ckpt(dict(LIKELIHOOD, **{field: other})), _Corpus(LIKELIHOOD)
            )


def test_a_prefield_checkpoint_skips_and_says_so(capsys):
    """A checkpoint recording no compat key must skip the check visibly."""
    embed._assert_trained_and_embedded_corpora_agree(_Ckpt(None), _Corpus(LIKELIHOOD))
    out = capsys.readouterr().out
    assert "CANNOT be checked" in out and "corpus_compat_key" in out


def test_train_py_records_the_corpus_it_trained_on():
    """`train.py` must record the corpus it trained on, so the comparison has a write side."""
    src = (REPO / "model" / "scrm_model" / "train.py").read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert '"corpus_compat_key": corpus_compat_key,' in code, (
        "checkpoint_payload must record the corpus it trained on, or the coupling check degrades "
        "to a skip for every run"
    )
    assert "corpus_compat_key=dict(corpus.compat_key())" in code, (
        "the training loop must pass the corpus object's key; a path is not a molecule definition"
    )


def test_the_check_runs_BEFORE_the_first_forward():
    """The coupling check must have one call site, above the embedding loop."""
    src = (REPO / "eval" / "scrm_eval" / "embed.py").read_text()
    lines = src.splitlines()
    calls = [n for n, ln in enumerate(lines)
             if "_assert_trained_and_embedded_corpora_agree(" in ln and not ln.startswith("def ")]
    assert len(calls) == 1, f"the coupling check should have ONE call site, found {len(calls)}"
    first_forward = next(n for n, ln in enumerate(lines) if "_drain(" in ln)
    assert calls[0] < first_forward, (
        "the coupling check must be called before any embedding work begins; move it above the loop"
    )


def test_eval_ready_does_not_strip_the_field():
    """`run_eval.eval_ready` must not drop `corpus_compat_key` when it re-saves a checkpoint."""
    src = (REPO / "model" / "scrm_model" / "run_eval.py").read_text()
    body = src[src.index("def eval_ready("):src.index("def main(")]
    assert 'if "arch" in raw:\n        return path' in body, (
        "eval_ready no longer short-circuits for a modern checkpoint"
    )
    assert "raw[" in body and "torch.save(raw," in body, (
        "eval_ready must augment the loaded payload rather than construct a new one, or any key it "
        "does not name is dropped, including corpus_compat_key"
    )


def test_the_corpus_is_only_interrogated_when_there_is_something_to_compare(capsys):
    """A corpus that cannot state its identity must not break the embed path."""
    class _Exploding:
        def compat_key(self):
            raise AssertionError("the corpus must not be asked when the checkpoint records nothing")

    embed._assert_trained_and_embedded_corpora_agree(_Ckpt(None), _Exploding())
    assert "CANNOT be checked" in capsys.readouterr().out


def test_a_corpus_that_cannot_state_its_definition_skips_visibly(capsys):
    """A checkpoint that records a compat key against a mute corpus must skip visibly."""
    embed._assert_trained_and_embedded_corpora_agree(_Ckpt(dict(LIKELIHOOD)), _MuteCorpus())
    out = capsys.readouterr().out
    assert "cannot" in out and "compat_key" in out
