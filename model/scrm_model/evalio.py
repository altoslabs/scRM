"""The checkpoint → eval bridge: a `WindowEncoder` (eval's protocol) backed by `ScrmModel`."""

from __future__ import annotations

import sys

import numpy as np
import torch

from .config import ArchConfig
from .model import ScrmModel


class ScrmWindowEncoder:
    def __init__(self, model: ScrmModel, device: torch.device | None = None,
                 compile_blocks: bool = False, views: int = 2) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        if views < 1:
            raise ValueError(f"views must be >= 1, got {views}")
        self.views = int(views)
        self._compiled = False
        if compile_blocks and self.device.type == "cuda":
            try:
                self.model.compile_blocks()
                self._compiled = True
            except Exception as exc:
                print(f"[evalio] compile_blocks unavailable, staying eager: {exc}", flush=True)

    @property
    def forward_precision(self) -> str:
        """What the forward actually runs, for the artifact's stamp to record."""
        dtype = "bf16" if self.device.type == "cuda" else "fp32"
        return f"{dtype}+{'compiled' if self._compiled else 'eager'}"

    @torch.no_grad()
    def embed_windows(
        self,
        tokens: np.ndarray,
        valid: np.ndarray,
        contig_segment_id: np.ndarray,
        umi_segment_id: np.ndarray,
        patch_start_ids: np.ndarray,
    ) -> np.ndarray:
        dev = self.device
        valid_t = torch.from_numpy(np.asarray(valid)).to(dev)
        useg = torch.from_numpy(np.asarray(umi_segment_id).astype(np.int64)).to(dev)
        n_patches = (torch.where(valid_t, useg, useg.new_full((), -1)).amax(dim=1) + 1).clamp_min(0)
        batch = {
            "tokens": torch.from_numpy(np.asarray(tokens).astype(np.int64)).to(dev),
            "valid": valid_t,
            "contig_segment_id": torch.from_numpy(np.asarray(contig_segment_id).astype(np.int64)).to(dev),
            "umi_segment_id": useg,
            "n_patches": n_patches,
        }
        batch["n_patches"] = batch["n_patches"].clamp_min(1)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            out = self.model(batch)
        if self._level == "umi":
            u = out.umi.float()
            n = batch["n_patches"].to(u.device)
            keep = (torch.arange(u.shape[1], device=u.device)[None, :] < n[:, None]).unsqueeze(-1)
            v = (u * keep).sum(1) / keep.sum(1).clamp_min(1)
            return v.cpu().numpy()
        return out.window.float().cpu().numpy()

    _level = "window"

    @torch.no_grad()
    def pool_pairs(self, window_vecs: np.ndarray) -> np.ndarray:
        """(V*k, dim) adjacent-grouped window embeddings → (k, dim) through the trained view pool."""
        v = self.views
        x = torch.from_numpy(np.asarray(window_vecs, dtype=np.float32)).to(self.device)
        if x.shape[0] % v:
            raise ValueError(f"pool_pairs at V={v} takes a multiple of {v} rows, got {x.shape[0]}")
        return self.model.pool_views(x, v=v).float().cpu().numpy()


def model_factory(arch: dict, state_dict, readout: dict | None = None) -> ScrmWindowEncoder:
    """`embed.Checkpoint.load`'s injection point: (arch, state_dict, readout) → WindowEncoder."""
    fields = {k: v for k, v in arch.items() if k in ArchConfig.__dataclass_fields__}
    if not torch.cuda.is_available() and fields.get("attn_impl") != "sdpa":
        print(f"evalio: no CUDA, running attn_impl={fields.get('attn_impl')!r} as 'sdpa'; the "
              "artifact's arch_sha is the checkpoint's, computed over the recorded implementation",
              file=sys.stderr, flush=True)
        fields["attn_impl"] = "sdpa"
    model = ScrmModel(ArchConfig(**fields))
    model.load_state_dict(state_dict)
    return ScrmWindowEncoder(model, views=int((readout or {}).get("views", 2)))
