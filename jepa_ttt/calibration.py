"""Action calibration: adapt what actions mean, with the world model frozen.

Let a_i be the i-th raw command of a block starting at state s_0. Under a shift
of the form a_applied = clip(R(s) a), the state at raw step i is a function of
s_0 and a_0, ..., a_{i-1} (away from walls). So the map from commands to applied
actions has the form

    a_applied_i = clip((I + D_i(s_0, a_0, ..., a_{i-1})) a_i)

for some 2x2 matrix D_i. This module learns D_i from the block's start latent
and the earlier commands. It aims to approximate that map without assuming
rotations in particular, and D_i = 0 is the identity. A finite network behind a
linear bottleneck can only approximate it, and only if the latent reveals s_0.
"""

from __future__ import annotations

import torch
from torch import nn

from .planning import ACTION_BLOCK, recent_errors

ACTION_DIM = 2
HIDDEN_DIM = 256
# Chosen on offline random-policy data, never on the held-out evaluation suite.
BOTTLENECK_DIM = 8
# Fixed before any GPU result, with no tuning.
GATE_MARGIN = 0.5  # the frozen model must have under half the calibrated error
OLD_WORLD_ERROR = 0.1  # frozen one-step error on real no-shift data was about 0.064
FAST_STEPS = 3
FAST_STEP_SIZE = 0.1


class ActionCalibrator(nn.Module):
    """Map normalized command blocks to normalized effective blocks."""

    def __init__(
        self,
        mean,
        std,
        latent_dim: int,
        hidden_dim: int = HIDDEN_DIM,
        bottleneck_dim: int | None = BOTTLENECK_DIM,
    ):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        # The state that sets the shift can be far smaller than the latent.
        self.compress = (
            nn.Identity() if bottleneck_dim is None else nn.Linear(latent_dim, bottleneck_dim)
        )
        state_dim = latent_dim if bottleneck_dim is None else bottleneck_dim
        # Earlier commands of the block, then a one-hot of the position.
        prefix_dim = (ACTION_BLOCK - 1) * ACTION_DIM
        self.net = nn.Sequential(
            nn.Linear(state_dim + prefix_dim + ACTION_BLOCK, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ACTION_DIM * ACTION_DIM),
        )
        # A zero last layer makes every D_i zero, so calibration starts as the identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer(
            "earlier", torch.tril(torch.ones(ACTION_BLOCK, ACTION_BLOCK), diagonal=-1)
        )
        self.register_buffer("positions", torch.eye(ACTION_BLOCK))
        # Before-plus-now: plan with the frozen model alone when it predicts better.
        self.gated = False
        self.enabled = True
        self.default_on = True
        self.last_on = True
        self.last_before_error = float("inf")
        # Fast adaptation: a per-environment matrix (B, 2, 2) set only while planning.
        self.adapt_fast = False
        self.fast: torch.Tensor | None = None

    def deltas(self, z: torch.Tensor, commands: torch.Tensor) -> torch.Tensor:
        """Return D_i (N, K, 2, 2) from latents (N, D) and commands (N, K, 2).

        ``earlier`` masks commands at and after position i, so D_i cannot depend
        on them. The last command never precedes another, so it is left out.
        """
        n = z.size(0)
        visible = commands.unsqueeze(1) * self.earlier[None, :, :, None]
        prefix = visible[:, :, : ACTION_BLOCK - 1].reshape(n, ACTION_BLOCK, -1)
        features = torch.cat(
            [
                self.compress(z).unsqueeze(1).expand(n, ACTION_BLOCK, -1),
                prefix,
                self.positions.unsqueeze(0).expand(n, -1, -1),
            ],
            dim=-1,
        )
        deltas = self.net(features).reshape(n, ACTION_BLOCK, ACTION_DIM, ACTION_DIM)
        if self.fast is None:
            return deltas
        # Rows are grouped by environment, so each repeats its matrix in place.
        if n % self.fast.size(0) != 0:
            raise ValueError(f"{n} rows do not divide among {self.fast.size(0)} environments")
        fast = self.fast.repeat_interleave(n // self.fast.size(0), dim=0)
        return deltas + fast.unsqueeze(1)

    def forward(self, z: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        """Calibrate blocks (..., K * 2) using their start latents (..., D).

        The result equals ``blocks`` exactly when every D_i is zero, including for
        out-of-range planner candidates. In-range commands map to
        clip((I + D_i) a_i), so the clamp mirrors the environment's own.
        """
        if blocks.numel() == 0 or not self.enabled:
            return blocks
        shape = blocks.shape
        raw = blocks.reshape(-1, ACTION_BLOCK, ACTION_DIM) * self.std + self.mean
        commands = raw.clamp(-1.0, 1.0)
        deltas = self.deltas(z.reshape(raw.size(0), -1), commands)
        moved = commands + (deltas @ commands.unsqueeze(-1)).squeeze(-1)
        change = moved.clamp(-1.0, 1.0) - commands
        return blocks + (change / self.std).reshape(shape)


class LatentCorrection(nn.Module):
    """A zero-initialized residual added to the frozen predictor's next latent."""

    def __init__(self, mean, std, latent_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        # It only trains on executed actions, so candidates outside the action range
        # are clamped first, and the planner can't exploit its extrapolation.
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        self.register_buffer("low", ((-1.0 - mean) / std).repeat(ACTION_BLOCK))
        self.register_buffer("high", ((1.0 - mean) / std).repeat(ACTION_BLOCK))
        self.compress = nn.Linear(latent_dim, BOTTLENECK_DIM)
        self.net = nn.Sequential(
            nn.Linear(BOTTLENECK_DIM + ACTION_BLOCK * ACTION_DIM, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
        executable = torch.max(torch.min(blocks, self.high), self.low)
        return self.net(torch.cat([self.compress(z), executable], dim=-1))


def choose_worlds(model, z_history: torch.Tensor, a_history: torch.Tensor, t: int) -> torch.Tensor:
    """Per-environment choice for before-plus-now: True plans with the calibrator.

    Each environment uses the frozen model alone only if that clearly predicted its
    recent blocks better. Before any block has run, it uses the deployment's last choice.
    """
    calibrator = model.calibrator
    if t == 0:
        calibrator.last_before_error = float("inf")
        return torch.full((z_history.size(0),), calibrator.default_on, device=z_history.device)
    with torch.no_grad():
        now = recent_errors(model, z_history, a_history, t)
        calibrator.enabled = False
        before = recent_errors(model, z_history, a_history, t)
        calibrator.enabled = True
    calibrator.last_before_error = float(before[0])
    return before >= GATE_MARGIN * now


def fast_steps(
    model,
    fast: torch.Tensor,
    z_history: torch.Tensor,
    a_history: torch.Tensor,
    t: int,
) -> torch.Tensor:
    """Take normalized gradient steps on each environment's fast matrix.

    Each environment's loss depends only on its own matrix, so one gradient of the
    summed loss holds them all. Only the matrix changes, never a stored weight.
    """
    if t == 0:
        return fast
    calibrator = model.calibrator
    current = fast
    for _ in range(FAST_STEPS):
        current = current.detach().requires_grad_(True)
        calibrator.fast = current
        with torch.enable_grad():
            loss = recent_errors(model, z_history, a_history, t).sum()
            gradient = torch.autograd.grad(loss, current)[0]
        norm = gradient.flatten(1).norm(dim=1).clamp_min(1e-8).reshape(-1, 1, 1)
        current = current - FAST_STEP_SIZE * gradient / norm
    calibrator.fast = None
    return current.detach()
