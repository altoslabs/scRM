"""The guards' instruments: callable, so the loop can log them every step."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_EVAL = str(Path(__file__).resolve().parents[2] / "eval")
if _EVAL not in sys.path:
    sys.path.insert(0, _EVAL)

from scrm_eval.metrics import effective_rank as _eval_effective_rank  # noqa: E402


def effective_rank(x: torch.Tensor) -> float:
    """`exp(H(p))` on the centred singular values, computed by eval's own function."""
    return float(_eval_effective_rank(x.detach().float().cpu().numpy()))


def gradient_audit(module: torch.nn.Module, prefix: str = "") -> dict[str, float]:
    """Every named parameter's gradient L2 norm; `0.0` for a missing or all-zero gradient."""
    out: dict[str, float] = {}
    for name, p in module.named_parameters():
        g = p.grad
        out[prefix + name] = 0.0 if g is None else float(g.detach().norm())
    return out


def dead_parameters(audit: dict[str, float]) -> list[str]:
    """The audit's verdict: names whose gradient is exactly absent or zero."""
    return [name for name, norm in audit.items() if norm == 0.0]


def parameter_groups(student, projector, nb=None, mol=None) -> dict[str, list]:
    """The 13.12 partition table's rows, as parameter lists. The instrument below measures the
    table empirically: which loss term delivers gradient into which group.
    """
    groups = {
        "local": list(student.embedding.parameters()) + list(student.local.parameters()),
        "pools": list(student.contig_pool.parameters()) + list(student.umi_pool.parameters()),
        "global": list(student.global_trunk.parameters()),
        "window_pool": list(student.window_pool.parameters()),
        "view_pool": list(student.view_pool.parameters()),
        "projector": list(projector.parameters()),
    }
    if nb is not None:
        groups["nb_head"] = list(nb[1].parameters())
    if mol is not None:
        groups["mol_temp"] = list(mol.parameters())
    return groups


def per_term_gradient_norms(terms: dict, groups: dict[str, list]) -> dict[str, float]:
    """For each loss term, the L2 norm of its gradient into each parameter group."""
    import torch

    flat: list = []
    slices: dict[str, slice] = {}
    for name, params in groups.items():
        slices[name] = slice(len(flat), len(flat) + len(params))
        flat.extend(params)
    out: dict[str, float] = {}
    for term_name, term in terms.items():
        grads = torch.autograd.grad(term, flat, retain_graph=True, allow_unused=True)
        for group_name, sl in slices.items():
            sq = 0.0
            for g in grads[sl]:
                if g is not None:
                    sq += float(g.detach().pow(2).sum())
            out[f"gnorm/{term_name}/{group_name}"] = sq**0.5
    return out
