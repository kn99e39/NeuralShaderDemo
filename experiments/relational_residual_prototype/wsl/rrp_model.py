"""Residual operators for the relational-residual prototype (torch; runs in RNA's WSL venv).

RelationalResidual (candidate):
    per-probe shared encoder phi (probe descriptor -> h)
    -> permutation-stable aggregation [mean_k h, max_k h]
    -> linear projection to the compact dynamic state g (dim G)
    -> decoder psi([g, local]) -> dL (RGB, linear radiance) per query sample.

LocalResidual (matched control):
    local encoder lam (local inputs -> G) -> the same decoder psi([g_local, local]).
    It has no argument through which probe / remote / nonlocal data could enter.

No configuration identifier enters either model; the same weights serve every state.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(sizes: list[int], last_act: bool) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Decoder(nn.Module):
    def __init__(self, g_dim: int, local_dim: int, hidden: list[int]):
        super().__init__()
        self.net = mlp([g_dim + local_dim, *hidden, 3], last_act=False)

    def forward(self, g, local):
        return self.net(torch.cat([g, local], -1))


class RelationalResidual(nn.Module):
    kind = "relational"

    def __init__(self, probe_dim: int, local_dim: int, cfg: dict):
        super().__init__()
        self.phi = mlp([probe_dim, *cfg["probe_encoder"]], last_act=True)
        h = cfg["probe_encoder"][-1]
        self.proj = nn.Linear(2 * h, cfg["state_dim"])
        self.psi = Decoder(cfg["state_dim"], local_dim, cfg["decoder"])

    def state(self, probes):
        """probes (Q, K, P) -> compact dynamic state g (Q, G)."""
        h = self.phi(probes)
        return self.proj(torch.cat([h.mean(1), h.amax(1)], -1))

    def forward(self, local, probes):
        return self.psi(self.state(probes), local)


class LocalResidual(nn.Module):
    kind = "local_only"

    def __init__(self, local_dim: int, cfg: dict):
        super().__init__()
        self.lam = mlp([local_dim, *cfg["local_encoder"], cfg["state_dim"]], last_act=False)
        self.psi = Decoder(cfg["state_dim"], local_dim, cfg["decoder"])

    def state(self, local):
        return self.lam(local)

    def forward(self, local, probes=None):
        if probes is not None:
            raise TypeError("the local-only control takes no probe state")
        return self.psi(self.state(local), local)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
