"""`fetch.py --check` must be able to FAIL.

It could not. `--check` was `p.exists()` over five model DIRECTORIES, and
`huggingface_hub.snapshot_download(local_dir=...)` creates the directory before it downloads
anything -- so a fetch interrupted at any point still printed "every path the embedder reads is
present". It also never looked at the cloned code, without which the loader raises at embed
time, after the GPU allocation is spent. A check that cannot fail is worse than no check,
because it reads as coverage; these tests are the permanent controls that it still can.

Nothing here needs the real weights (tens of gigabytes) or the network: the trees are built from
the embedder's own declaration of what it opens, which is the thing under test.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "countfm"))

import fetch  # noqa: E402
import genejepa_loader as loader  # noqa: E402


def _required() -> list[tuple[str, str]]:
    return fetch.required_files()


def _plant(root: Path, *, bytes_for=lambda rel: b"x") -> None:
    """A tree with every file the embedder opens, with contents the caller chooses."""
    for _, rel in _required():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(bytes_for(rel))


def _fake_clone(root: Path, monkeypatch) -> None:
    """A git repo at the clone's path, with the pin moved to whatever HEAD it actually has.

    The pinned commit cannot be fabricated, so the mechanism is tested by pinning the fixture:
    what matters is that `check` compares HEAD against the pin at all.
    """
    src = root / loader.SRC_DIR
    src.mkdir(parents=True, exist_ok=True)
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(root), "PATH": "/usr/bin:/bin"}
    subprocess.run(["git", "init", "--quiet", str(src)], check=True, env=env)
    (src / "seed").write_text("seed")
    subprocess.run(["git", "-C", str(src), "add", "-A"], check=True, env=env)
    subprocess.run(["git", "-C", str(src), "commit", "--quiet", "-m", "t"], check=True, env=env)
    head = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], check=True, env=env,
                          capture_output=True, text=True).stdout.strip()
    url, _, dest = fetch.GENEJEPA_SRC
    monkeypatch.setattr(fetch, "GENEJEPA_SRC", (url, head, dest))


def _pin_the_fixture(root: Path, monkeypatch) -> None:
    """Point `DIGESTS` at the planted tree's real sizes and hashes."""
    pinned = {}
    for _, rel in _required():
        if rel in fetch.DIGESTS:
            blob = (root / rel).read_bytes()
            pinned[rel] = (len(blob), hashlib.sha256(blob).hexdigest())
    monkeypatch.setattr(fetch, "DIGESTS", pinned)


def test_the_declaration_is_read_from_the_embedder_and_is_not_empty():
    """If this were empty every test below would pass by checking nothing."""
    req = _required()
    assert req, "fetch.required_files() names no file at all"
    assert all("/" in rel for _, rel in req), req
    # The clone's own modules are covered too: their absence is what raised at embed time.
    assert [rel for _, rel in req if rel.startswith(loader.SRC_DIR + "/")], (
        f"no file under {loader.SRC_DIR} is checked, which is the gap this closes"
    )


def test_every_downloaded_file_is_pinned_by_a_digest():
    """A file the embedder opens but nothing pins would be fetched unverified."""
    unpinned = [rel for _, rel in _required()
                if rel not in fetch.DIGESTS and not rel.startswith(loader.SRC_DIR + "/")]
    assert not unpinned, (
        f"these files are opened by an embedding but have no entry in fetch.DIGESTS: {unpinned}. "
        "Record their size and sha256 there, or `--check` passes on whatever bytes are present."
    )


def test_the_directories_alone_do_not_pass(tmp_path, capsys):
    """THE ORIGINAL BUG. `snapshot_download` makes the directory first, so this is exactly the
    state an interrupted fetch leaves behind."""
    for _, rel in _required():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    assert fetch.check(tmp_path) > 0, "a tree of empty directories passed the check"
    assert "MISSING" in capsys.readouterr().out


def test_a_complete_tree_passes(tmp_path, monkeypatch, capsys):
    """The positive control: without it, a check that always failed would look the same."""
    _plant(tmp_path, bytes_for=lambda rel: rel.encode())
    _fake_clone(tmp_path, monkeypatch)
    _pin_the_fixture(tmp_path, monkeypatch)
    assert fetch.check(tmp_path) == 0, capsys.readouterr().out


def test_the_right_names_with_the_wrong_bytes_do_not_pass(tmp_path, monkeypatch, capsys):
    """Same names, same sizes, different content: only a digest can tell these apart."""
    _plant(tmp_path, bytes_for=lambda rel: rel.encode())
    _fake_clone(tmp_path, monkeypatch)
    _pin_the_fixture(tmp_path, monkeypatch)
    target = next(rel for _, rel in _required() if rel in fetch.DIGESTS)
    blob = bytearray((tmp_path / target).read_bytes())
    blob[0] ^= 0xFF  # same length, one flipped bit
    (tmp_path / target).write_bytes(bytes(blob))
    assert fetch.check(tmp_path) > 0, "a file with the pinned size but different bytes passed"
    out = capsys.readouterr().out
    assert "WRONG BYTES" in out and target in out


def test_a_truncated_file_does_not_pass(tmp_path, monkeypatch, capsys):
    """The interrupted download, caught on size before anything is hashed."""
    _plant(tmp_path, bytes_for=lambda rel: rel.encode())
    _fake_clone(tmp_path, monkeypatch)
    _pin_the_fixture(tmp_path, monkeypatch)
    target = next(rel for _, rel in _required() if rel in fetch.DIGESTS)
    (tmp_path / target).write_bytes(b"")
    assert fetch.check(tmp_path) > 0, "a truncated file passed"
    assert "WRONG SIZE" in capsys.readouterr().out


def test_an_unpinned_file_does_not_pass(tmp_path, monkeypatch, capsys):
    """A file with no digest is reported, not accepted: that is how a new model's weights would
    otherwise go unverified while the check printed ok."""
    _plant(tmp_path, bytes_for=lambda rel: rel.encode())
    _fake_clone(tmp_path, monkeypatch)
    _pin_the_fixture(tmp_path, monkeypatch)
    dropped = next(iter(fetch.DIGESTS))
    monkeypatch.setattr(fetch, "DIGESTS", {k: v for k, v in fetch.DIGESTS.items() if k != dropped})
    assert fetch.check(tmp_path) > 0, "a file with no pinned digest passed"
    assert "UNPINNED" in capsys.readouterr().out


def test_the_clone_must_be_there_and_at_the_pinned_commit(tmp_path, monkeypatch, capsys):
    """The loader imports the clone; the checkpoint unpickles only against the code that wrote
    it. Present-but-at-another-revision is the failure that looks like success."""
    _plant(tmp_path, bytes_for=lambda rel: rel.encode())
    _pin_the_fixture(tmp_path, monkeypatch)

    # no clone at all
    assert fetch.check(tmp_path) > 0
    assert "NO CLONE" in capsys.readouterr().out

    # a clone, at the wrong revision
    _fake_clone(tmp_path, monkeypatch)
    url, head, dest = fetch.GENEJEPA_SRC
    monkeypatch.setattr(fetch, "GENEJEPA_SRC", (url, "0" * 40, dest))
    assert fetch.check(tmp_path) > 0, "a clone at an unpinned revision passed"
    assert "WRONG REV" in capsys.readouterr().out


def test_the_loader_and_the_fetcher_pin_the_same_clone():
    """One clone, one pin. Two copies of a commit is one copy that goes stale, and the loader is
    the module that imports it."""
    url, commit, dest = fetch.GENEJEPA_SRC
    assert (url, commit, dest) == (loader.UPSTREAM, loader.UPSTREAM_COMMIT, loader.SRC_DIR)


def test_the_loader_refuses_a_directory_with_no_modules_in_it(tmp_path):
    """An interrupted clone leaves the directory behind; `is_dir()` accepted that and the import
    failed several frames away from anything that says how to fix it."""
    (tmp_path / loader.SRC_DIR / loader.SRC_PACKAGE).mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        loader._their_modules(tmp_path)
    assert loader.SRC_DIR in str(exc.value) and loader.UPSTREAM_COMMIT[:12] in str(exc.value)


def _a_model_with_a_nested_required_file() -> tuple[str, str]:
    """Some (model, required path) whose file sits in a subdirectory of the model's own dir --
    asked of the declaration rather than named, as this suite requires."""
    ecf = fetch.embedder()
    for model, files in sorted(ecf.REQUIRED_FILES.items()):
        dest = fetch.WEIGHTS[model][2]
        for rel in files:
            if rel.startswith(dest + "/") and "/" in rel[len(dest) + 1:]:
                return model, rel
    pytest.skip("no model declares a required file below its own directory")


def test_placement_puts_a_file_fetched_at_another_repo_path_where_the_embedder_opens_it(tmp_path):
    """A publisher's layout is not a contract. The gc104M dictionaries live at one path in the
    HuggingFace repo and are read from another, so a fetch that only mirrored the repo left a
    tree the embedder could not read -- and the old `--check` called it complete."""
    model, rel = _a_model_with_a_nested_required_file()
    dest = fetch.WEIGHTS[model][2]
    elsewhere = tmp_path / dest / Path(rel).name  # as a repo-relative mirror would leave it
    elsewhere.parent.mkdir(parents=True, exist_ok=True)
    elsewhere.write_bytes(b"payload")

    assert fetch.place(model, tmp_path, dry=False) >= 0
    assert (tmp_path / rel).read_bytes() == b"payload", f"{rel} was not put where it is opened"
    assert not elsewhere.exists(), "the file was copied rather than moved"


def test_placement_moves_nothing_when_everything_is_already_in_place(tmp_path, capsys):
    """It runs against a tree that is already correct on every fetch, so it must be inert."""
    _plant(tmp_path, bytes_for=lambda rel: rel.encode())
    before = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    for model in sorted(fetch.embedder().REQUIRED_FILES):
        assert fetch.place(model, tmp_path, dry=False) == 0
    after = {p: p.read_bytes() for p in sorted(tmp_path.rglob("*")) if p.is_file()}
    assert after == before, "placement changed a tree that was already correct"
    assert "placed" not in capsys.readouterr().out


def test_placement_refuses_to_choose_between_two_candidates(tmp_path, capsys):
    """Two files with the right name is a choice this script must not make silently."""
    model, rel = _a_model_with_a_nested_required_file()
    dest = tmp_path / fetch.WEIGHTS[model][2]
    for sub in ("a", "b"):
        p = dest / sub / Path(rel).name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(sub.encode())
    assert fetch.place(model, tmp_path, dry=False) > 0
    assert "AMBIGUOUS" in capsys.readouterr().out
    assert not (tmp_path / rel).exists(), "one of two candidates was chosen anyway"


def test_the_embedder_declares_files_for_every_model_it_accepts():
    """Asserted at import in the embedder; asserted here too, because the import that would
    catch it happens under the models' own interpreter, which this suite does not run."""
    ecf = fetch.embedder()
    from scrm_eval import panel

    assert set(ecf.REQUIRED_FILES) == set(panel.EMBEDDABLE)
    for model, files in ecf.REQUIRED_FILES.items():
        assert files, f"{model} declares no file, so nothing about it can be verified"
