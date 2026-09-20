"""The projector: `dim_global → hidden → hidden → proj_dim`, BN and ReLU on hidden layers
ONLY; the MLP ends `Linear → Dropout`.
"""

from torch import nn

from .config import ArchConfig


def build_projector(cfg: ArchConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(cfg.dim_global, cfg.proj_hidden, bias=True),
        nn.BatchNorm1d(cfg.proj_hidden),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(cfg.proj_hidden, cfg.proj_hidden, bias=True),
        nn.BatchNorm1d(cfg.proj_hidden),
        nn.ReLU(),
        nn.Dropout(0.0),
        nn.Linear(cfg.proj_hidden, cfg.proj_dim, bias=True),
        nn.Dropout(0.0),
    )
