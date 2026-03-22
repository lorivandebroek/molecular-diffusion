from __future__ import annotations

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """Concat(Morgan FP, T_scaled) -> log10(D)."""

    def __init__(
        self,
        fp_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        dims = [fp_dim + 1] + list(hidden_dims) + [1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, fp: torch.Tensor, t_scaled: torch.Tensor) -> torch.Tensor:
        if t_scaled.dim() == 1:
            t_scaled = t_scaled.unsqueeze(-1)
        x = torch.cat([fp, t_scaled], dim=-1)
        return self.net(x)
