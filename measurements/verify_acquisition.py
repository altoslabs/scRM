"""Verify that every corpus the paper uses can be acquired, and print the plan that builds them.

`model/jobs/corpora.tsv` is the one declaration; this reads it and answers, per row, whether the
declared whitelist SCHEME resolves. It checks the scheme, not whether the product already sits on
this machine: a check that passes because the author's disk holds the file answers nothing about a
clean clone.

    --plan        the corpus submissions, one per row, for reproduce.sh
    --net         additionally resolve every read source over the network
    --write-tex   tab:dataurls' body, and dataurls_n's macros for the counts the PROSE states
    --check-tex   assert that both match a fresh derivation
"""

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
JOBS = REPO / "model/jobs"
sys.path.insert(0, str(REPO / "eval"))
from scrm_eval import registry  # noqa: E402

PROJ = registry.PROJ
TABLE = JOBS / "corpora.tsv"
EXECUTOR = "corpus.sbatch"

CDN = "https://cf.10xgenomics.com/samples/cell-exp"
S3H = "https://s3-us-west-2.amazonaws.com/10x.files/samples/cell-exp"
EBI = "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/519/E-MTAB-10519/Files"
FCA_LIST = PROJ / "work/fca/fetch_list_all.json"

#: Datasets the paper cites by accession rather than by 10x release path.
CITE = {
    "fly_antenna": "E-MTAB-10519 (FCA43)",
    "sc_mixology_5cl": "SRA SRR8606521",
    "got_ET01": "SRA SRR7613775; genotypes GSM3309838",
}

COLUMNS = ("key", "role", "kind", "source", "whitelist", "geometry", "species", "tools", "note")


def rows() -> list[dict[str, str]]:
    """Every declared row of corpora.tsv, in file order."""
    out = []
    for line in TABLE.read_text().splitlines():
        if not line or line.startswith("#") or "\t" not in line or line.startswith("key\t"):
            continue
        out.append(dict(zip(COLUMNS, line.split("\t"))))
    return out


def fields(row: dict[str, str]) -> dict[str, str]:
    """The row's `source` column, as its k=v pairs."""
    return dict(p.split("=", 1) for p in row["source"].split(";") if "=" in p)


def tenx_base(f: dict[str, str]) -> str:
    return f"{CDN}/{f['ver']}" if f.get("host") == "cf" else f"{S3H}/{f['ver']}"


def reads_of(row: dict[str, str]) -> str | None:
    """Where the row's reads come from, derived from its own locator fields.

    `None` for a corpus that is derived from another rather than downloaded.
    """
    f = fields(row)
    kind = row["kind"]
    if kind == "tenx-tar":
        return f"{tenx_base(f)}/{f['sample']}/{f['sample']}_fastqs.tar"
    if kind in ("sra-reads", "sra-bam"):
        return f"sra:{f['srr']}"
    if kind == "ebi-files":
        return f"fca:{f['sample']}"
    if kind == "czb-s3":
        return f"s3://{f['bucket']}/{f['prefix']}/{f['channel']}"
    if kind == "permuted-from":
        return None
    raise SystemExit(f"REFUSING: {row['key']} declares unknown kind {kind!r}")


def tracked(rel: str) -> bool:
    r = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", rel],
                       capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def derivations() -> set[str]:
    sys.path.insert(0, str(JOBS))
    try:
        import derive_labels
        return set(derive_labels.DERIVATIONS)
    finally:
        sys.path.pop(0)


def check_whitelist(row: dict[str, str]) -> tuple[bool, str]:
    """Does the row's declared whitelist SCHEME resolve? Never "is the product already on disk"."""
    spec, f = row["whitelist"], fields(row)
    if spec == "UNRESOLVED":
        return False, "declared UNRESOLVED"
    if spec == "derived":
        return True, "derived with the corpus it is permuted from"
    if spec in ("h5", "h5-multi"):
        ok = {"host", "ver", "sample"} <= set(f)
        where = "in the sample's doubled directory" if spec == "h5-multi" else "beside the sample"
        return ok, (f"h5 {where}" if ok else f"{spec} but no host/ver/sample to locate it")
    if spec.startswith("tar:"):
        ok = {"host", "ver", "sample"} <= set(f) and len(spec) > 4
        return ok, (f"member {spec[4:]} of the filtered-matrix tarball" if ok
                    else "tar: but no sample or no member named")
    if spec.startswith("repo:"):
        rel = spec[len("repo:"):]
        return tracked(rel), (f"ships in the repo at {rel}" if tracked(rel)
                              else f"repo:{rel} is NOT tracked by git, so a clone does not have it")
    if spec.startswith("geo:"):
        rest = spec[len("geo:"):]
        ok = "#" in rest and rest.startswith("GSM")
        return ok, (f"GEO supplementary file {rest}" if ok else f"malformed geo: spec {rest!r}")
    if spec.startswith("derive:"):
        ds = spec[len("derive:"):]
        known = derivations()
        return ds in known, (f"derived by derive_labels.py {ds}" if ds in known
                             else f"derive:{ds} is not implemented; known: {sorted(known)}")
    if spec.startswith(("json:", "matrix:")):
        scheme, rest = spec.split(":", 1)
        rel = rest.split("#", 1)[0]
        # These name a file under the DATA ROOT. Nothing in this tree writes one, so a clean clone
        # cannot produce it.
        return False, (f"{scheme}:{rel} names a file under the data root that nothing in this tree "
                       f"produces, so a clean clone cannot obtain it")
    return False, f"unknown whitelist scheme {spec!r}"


def s3_ok(uri: str) -> bool:
    """`aws s3 ls` for an s3:// source; curl cannot HEAD an s3 URI."""
    out = subprocess.run(["aws", "s3", "ls", "--no-sign-request", uri.rstrip("/") + "/"],
                         capture_output=True, text=True)
    return out.returncode == 0 and bool(out.stdout.strip())


def head_ok(url: str) -> bool:
    out = subprocess.run(
        ["curl", "-sI", "--max-time", "25", "-o", "/dev/null", "-w", "%{http_code}", url],
        capture_output=True, text=True,
    ).stdout.strip()
    return out == "200"


def fca_ok(sample: str) -> tuple[bool, str]:
    """Resolve a fly sample against the atlas file list, then HEAD its first read."""
    if not FCA_LIST.is_file():
        return False, f"{FCA_LIST} missing"
    d = json.loads(FCA_LIST.read_text())
    entry = d.get(sample)
    if not entry:
        return False, f"{sample} absent from fetch_list_all.json"
    files = entry["files"] if isinstance(entry, dict) else entry
    r1 = [x for x in files if "_R1_" in x]
    if not r1:
        return False, f"{sample} has no R1 in the list"
    return head_ok(f"{EBI}/{sorted(r1)[0]}"), f"{len(files)} files, first {sorted(r1)[0]}"


TEX_GROUPS = (
    ("train", "Pretraining --- the four the reported model draws its $3{,}000$ cells from"),
    ("eval", "Held-out evaluation"),
    ("eval-pending", "Registered, not yet an evaluation column"),
)

#: The datasets whose CELL COUNT the paper states in PROSE, and the macro stem each one gets.
#:
#: Written out rather than derived from the registry name because a LaTeX control-sequence name is
#: LETTERS ONLY: a mechanical stem would have to spell every digit
#: (`tm_10X_P7_2` -> `TmOneZeroXPSevenTwo`), and dropping the digits collides -- `neuron_1k_v3` and
#: `neuron_10k_v3` both become `NeuronKV`. The keys are asserted against `registry.DATASETS` below,
#: so renaming a dataset fails here instead of silently emitting nothing.
CELL_MACROS = {
    "sc_mixology_5cl": "ScMixology",
    "tm_10X_P7_2": "TabulaMuris",
    "crispr_a549_5k": "Crispr",
}


def _cells_of(ds: str) -> int | None:
    """The corpus's cell count, from its MANIFEST rather than by walking it."""
    m = registry.corpus_root() / ds / "manifest.json"
    if not m.is_file():
        return None
    try:
        c = json.loads(m.read_text())["measured"]["counts"]
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    called = c.get("cells_called")
    if called is None:
        return None
    return called - (c.get("dropped_cells_shallow") or 0)


def _thousands(n: int) -> str:
    """The count as the paper prints it. One spelling, so a macro and its table cell agree."""
    return f"{n:,}".replace(",", "{,}")


def _missing(what: str) -> str:
    """The paper's own missing-cell marker, so this file does not restate its macro namespace."""
    sys.path.insert(0, str(REPO / "measurements"))
    from results_page import TEX_MISSING
    return TEX_MISSING + "{" + what + "}"

def tex_dataurls() -> tuple[str, str, int]:
    """`tab:dataurls`'s body, `dataurls_n`'s macros, and a problem count.

    Columns: dataset, species, MEASURED cells, release. The macros are the same cell counts the
    body prints, for the prose sites that state one; both read ONE map, so a macro cannot disagree
    with the table cell beside it.
    """
    def esc(t: str) -> str:
        return t.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")

    def slug(ds: str, reads: str | None) -> str:
        """The release as the paper shows it: a citation if there is one, else version/sample."""
        if ds in CITE:
            return esc(CITE[ds])
        if not reads:
            return "DERIVED, not downloaded"
        t = reads
        if t.startswith(CDN):
            t = t[len(CDN):].lstrip("/")
        elif t.startswith(S3H):
            t = t[len(S3H):].lstrip("/")
            if t.endswith(".tar"):
                t = t.rsplit("/", 1)[0]
            return (esc(t).replace(r"\_", r"\_\allowbreak ").replace("/", r"/\allowbreak ")
                    + r" \textit{(S3 host, not the CDN)}")
        elif t.startswith("s3://"):
            body = esc(t).replace("/", r"/\allowbreak ").replace("-", r"-\allowbreak ")
            return f"{body} \\textit{{(S3 host, not the CDN)}}"
        t = t.rsplit("/", 1)[0] if t.endswith(".tar") else t
        return esc(t).replace(r"\_", r"\_\allowbreak ").replace("/", r"/\allowbreak ")

    registered = {d[0] for d in registry.DATASETS}
    unknown = sorted(set(CELL_MACROS) - registered)
    if unknown:
        raise SystemExit(f"REFUSING: CELL_MACROS names {unknown}, which the registry does not. "
                         f"A prose macro for an unregistered dataset derives nothing.")

    by_key = {r["key"]: r for r in rows()}
    cells = {ds: _cells_of(ds) for ds in registered}
    body, problems = [], 0
    for role, caption in TEX_GROUPS:
        names = [d[0] for d in registry.DATASETS if d[2] == role]
        if not names:
            continue
        body.append(f"\\multicolumn{{4}}{{@{{}}l}}{{\\textit{{{caption}}}}} \\\\")
        for ds in names:
            species = next(d[1] for d in registry.DATASETS if d[0] == ds)
            row = by_key.get(ds)
            if row is None:
                problems += 1
                cells_s, rel = _missing("cells"), _missing("source")
            else:
                rel = slug(ds, reads_of(row))
                n = cells[ds]
                if n is None:
                    problems += 1
                    cells_s = _missing("cells")
                else:
                    cells_s = _thousands(n)
            body.append(f"{esc(ds)} & {species} & {cells_s} & {rel} \\\\")

    sys.path.insert(0, str(REPO / "measurements"))
    from results_page import TEX_NS
    macros = [
        f"% GENERATED by measurements/{pathlib.Path(__file__).name} --write-tex. Do not edit.",
        "% Each value is cells_called minus dropped_cells_shallow from the corpus manifest -- the",
        "% same quantity tab:dataurls' own cell prints, read out of ONE map so they cannot differ.",
        "% Each count is of the whole WHITELIST, not of a subset, which is why the prose can state",
        "% it: for crispr_a549_5k the called cells ARE the guide-called cells (corpora.tsv says",
        "% derive:crispr_a549_5k) and for tm_10X_P7_2 they ARE the annotated cells, so 'cells carry",
        "% a call' and 'cells over 16 expert classes' are that count by construction.",
    ]
    for ds, stem in CELL_MACROS.items():
        n = cells[ds]
        val = _thousands(n) if n is not None else _missing("cells")
        macros.append("\\newcommand{\\" + TEX_NS + "Cells" + stem + "}{" + val + "}")
    return "\n".join(body), "\n".join(macros), problems


def _plan() -> int:
    """One CORPUS line per declared row: the single executor, keyed by the row it reads.

    A permuted-from row needs its source corpus built first, so those are emitted last.
    """
    rs = rows()
    known = {d for d, _, _ in registry.DATASETS}
    declared = {r["key"] for r in rs}
    if declared != known:
        print(f"# corpora.tsv and registry.DATASETS disagree: "
              f"only-declared {sorted(declared - known)}, only-registered {sorted(known - declared)}")
    ordered = ([r for r in rs if r["kind"] != "permuted-from"]
               + [r for r in rs if r["kind"] == "permuted-from"])
    print(f"# {len(ordered)} declared corpora -> {len(ordered)} submissions of {EXECUTOR}, "
          f"permuted last because each needs its src= corpus built first")
    for r in ordered:
        print(f"CORPUS|{EXECUTOR}|{r['key']}")
    return 0 if declared == known else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", action="store_true", help="also resolve every read source")
    ap.add_argument("--plan", action="store_true", help="print the corpus submissions")
    ap.add_argument("--write-tex", type=pathlib.Path, metavar="DIR")
    ap.add_argument("--check-tex", type=pathlib.Path, metavar="DIR")
    args = ap.parse_args()

    if args.plan:
        return _plan()

    tex_dir = args.write_tex or args.check_tex
    if tex_dir:
        body, macros, problems = tex_dataurls()
        sys.path.insert(0, str(REPO / "measurements"))
        import results_page as _rp
        for name, text in (("dataurls_body", body), ("dataurls_n", macros)):
            text = text.rstrip("\n") + "\n"
            f = pathlib.Path(tex_dir) / f"{name}.tex"
            for b in _rp.validate_tex(name, text):
                print(f"  [tex] MALFORMED {b}")
                problems += 1
            if args.check_tex:
                have = f.read_text() if f.is_file() else None
                if have != text:
                    print(f"  [tex] STALE {f}: "
                          f"{'absent' if have is None else 'differs from a fresh derivation'}")
                    problems += 1
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tex.tmp")
            tmp.write_text(text)
            tmp.replace(f)
            print(f"  [tex] wrote {f} ({len(text.splitlines())} lines)")
        return 1 if problems else 0

    rs = rows()
    known = {d for d, _, _ in registry.DATASETS}
    problems: list[str] = []
    if not (JOBS / EXECUTOR).is_file():
        problems.append(f"{EXECUTOR} does not exist, so nothing can be built")
    for ds in sorted(known - {r["key"] for r in rs}):
        problems.append(f"{ds}: registered but not declared in corpora.tsv, so it cannot be built")
    for ds in sorted({r["key"] for r in rs} - known):
        problems.append(f"{ds}: declared in corpora.tsv but not registered, so nothing consumes it")

    print(f"{'dataset':46} {'kind':15} whitelist")
    for r in rs:
        ok, detail = check_whitelist(r)
        if not ok:
            problems.append(f"{r['key']}: whitelist unresolved -- {detail}")
        print(f"  {r['key']:46} {r['kind']:15} {detail}")

    if args.net:
        print("\nresolve every read source:")
        for r in rs:
            u = reads_of(r)
            if u is None:
                print(f"  {r['key']:46} {'derived, nothing to fetch':26}")
                continue
            if u.startswith("sra:"):
                have = shutil.which("prefetch") is not None
                print(f"  {r['key']:46} {'sra, toolkit ' + ('present' if have else 'MISSING'):26} {u}")
                if not have:
                    problems.append(f"{r['key']}: SRA source but no prefetch binary")
                continue
            if u.startswith("s3://"):
                good = s3_ok(u)
                print(f"  {r['key']:46} {'s3 listable' if good else '*** NOT LISTABLE ***':26} {u}")
                if not good:
                    problems.append(f"{r['key']}: s3 source not listable -- {u}")
                continue
            if u.startswith("fca:"):
                good, detail = fca_ok(u[4:])
                print(f"  {r['key']:46} {'200' if good else '*** NOT 200 ***':26} {detail}")
                if not good:
                    problems.append(f"{r['key']}: fly reads unresolved -- {detail}")
                continue
            good = head_ok(u)
            print(f"  {r['key']:46} {'200' if good else '*** NOT 200 ***':26} {u[-58:]}")
            if not good:
                problems.append(f"{r['key']}: read URL does not return 200 -- {u}")

    print(f"\n{len(rs)} declared corpora, all built by {EXECUTOR}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("every declared corpus has a resolvable whitelist and one executor that builds it")
    return 0


if __name__ == "__main__":
    sys.exit(main())
