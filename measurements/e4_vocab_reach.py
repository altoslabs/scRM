"""How much of a dataset can each model ADDRESS AT ALL, before any accuracy is measured?"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "eval"), str(REPO / "target" / "py")]
from scrm_eval import artifact, controls, registry, targets  # noqa: E402

P = registry.PROJ
#: The weights tree every other producer and consumer of these tables reads -- the PRODUCER of the
#: bridges (`build_ortholog_table.WEIGHTS`), `embed_count_fm`, and the launchers.
W = P / "countfm" / "scfm_weights"
REPORTS = P / "eval_reports"
BRIDGES = {"mouse": W / "ortholog_mouse_human.tsv", "fly": W / "ortholog_fly_human.tsv"}

#: OUR producer key, read from the declaration rather than typed.
_OURS = sorted(artifact.MODEL_PRODUCERS)
assert len(_OURS) == 1, f"this table has one row for our model; MODEL_PRODUCERS holds {_OURS}"
OURS = _OURS[0]


def bridge(species: str):
    """The bridge as `embed_count_fm.to_human` applies it: `(by_source_id, by_source_symbol)`, each
    mapping to the **(human ENSG, human symbol)** pair. `None` for human, which uses no bridge.
    """
    if species == "human":
        return None
    rows = [line.rstrip("\n").split("\t") for line in BRIDGES[species].read_text().splitlines()][1:]
    by_id, by_sym = {}, {}
    for r in rows:
        if len(r) < 4:
            continue
        src_id, src_sym, ensg, hsym = r[0], r[1], r[2], r[3]
        by_id[src_id] = (ensg, hsym.upper())
        if src_sym:
            by_sym[src_sym.upper()] = (ensg, hsym.upper())
    return by_id, by_sym


def vocabularies() -> dict[str, set[str]]:
    """Each count FM's token vocabulary, read from the weights it is actually run with."""
    out: dict[str, set[str]] = {}
    with open(W / "geneformer/dictionaries/gene_name_id_dict_gc104M.pkl", "rb") as fh:
        out["geneformer"] = {str(k).upper() for k in pickle.load(fh)}
    out["scgpt"] = {str(k).upper() for k in json.loads((W / "scgpt_human/vocab.json").read_text())}
    try:
        import pandas as pd
        md = pd.read_parquet(W / "genejepa/gene_metadata.parquet")
        out["genejepa"] = {str(x).upper() for x in md["ensembl_id"].dropna()}
    except Exception as exc:
        print(f"  [warn] genejepa vocabulary unreadable ({exc}); omitted rather than assumed", flush=True)
    return out


VOCAB_KEY = {"geneformer": "symbol", "scgpt": "symbol", "genejepa": "ensembl"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="+",
                    default=["sc_mixology_5cl", "nuclei_2k", "fly_antenna"])
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="where to write the report; the default is the data root's eval_reports/")
    a = ap.parse_args(argv)
    vocab = vocabularies()
    print("  vocabularies: " + ", ".join(f"{k} {len(v):,}" for k, v in sorted(vocab.items())))

    out = {"vocab_sizes": {k: len(v) for k, v in vocab.items()}, "datasets": {}}
    for ds in a.datasets:
        species = targets.SPECIES[ds]
        ad, cells = controls._read_counts(ds)
        syms = [str(s).upper() for s in ad.var_names]
        umis = np.asarray(ad.X.sum(axis=0)).ravel()
        total_u, total_g = float(umis.sum()), len(syms)
        raw_ids = list(ad.var["gene_ids"]) if "gene_ids" in ad.var.columns else list(ad.var_names)
        ids = [str(g).split(".")[0] for g in raw_ids]
        b = bridge(species)
        if b is None:
            pair = [(i, sy) for i, sy in zip(ids, syms)]
            mapped = [True] * len(syms)
        else:
            by_id, by_sym = b
            pair, mapped = [], []
            for i, sy in zip(ids, syms):
                hit = by_id.get(i) or by_sym.get(sy)
                pair.append(hit if hit else ("", ""))
                mapped.append(hit is not None)
        bridged_u = float(sum(u for u, ok in zip(umis, mapped) if ok))
        row = {"species": species, "n_cells": len(cells), "genes_total": total_g,
               "genes_after_bridge": int(sum(mapped)),
               "umi_frac_after_bridge": bridged_u / max(total_u, 1),
               "gene_frac_after_bridge": sum(mapped) / max(total_g, 1), "models": {}}
        for m, v in sorted(vocab.items()):
            k_ens = VOCAB_KEY.get(m) == "ensembl"
            ok = [mp and ((pr[0] in v) if k_ens else (pr[1].upper() in v))
                  for mp, pr in zip(mapped, pair)]
            keep = [sy for sy, k in zip(syms, ok) if k]
            u = float(sum(uu for uu, k in zip(umis, ok) if k))
            row["models"][m] = {"genes_reachable": len(keep),
                                "gene_frac": len(keep) / max(total_g, 1),
                                "umi_frac": u / max(total_u, 1)}
        row["models"][OURS] = {"genes_reachable": total_g, "gene_frac": 1.0, "umi_frac": 1.0,
                               "note": "no gene vocabulary and no bridge; reads bases"}
        out["datasets"][ds] = row
        print(f"\n  {ds} ({species}): {total_g:,} genes, {total_u:,.0f} UMIs")
        print(f"    after ortholog bridge: {int(sum(mapped)):,} genes "
              f"({100*row['gene_frac_after_bridge']:.1f}%), "
              f"{100*row['umi_frac_after_bridge']:.1f}% of UMIs")
        row["vocab_key"] = dict(VOCAB_KEY)
        for m, d in sorted(row["models"].items()):
            print(f"    {m:12s} {d['genes_reachable']:7,d} genes ({100*d['gene_frac']:5.1f}%)  "
                  f"{100*d['umi_frac']:5.1f}% of UMIs")

    dest = a.out or (REPORTS / "e4_vocab_reach.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, sort_keys=True))
    tmp.replace(dest)
    print(f"\n  wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
