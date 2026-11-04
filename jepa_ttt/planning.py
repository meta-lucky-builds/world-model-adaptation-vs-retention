"""Latent rollouts and the CEM planner, with the settings from Table A1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

ACTION_BLOCK = 5  # K: raw actions per world-model block
HISTORY = 3  # C: context length of the predictor


@dataclass(frozen=True)
class PlannerConfig:
    num_samples: int = 256
    iterations: int = 6
    elites: int = 32
    horizon: int = 5
    init_std: float = 1.0
    std_floor: float = 0.05
    discount: float = 0.99


def make_reward_head(latent_dim: int = 192) -> nn.Sequential:
    """Sigmoid MLP 192 -> 256 -> 256 -> 1 with ReLU hidden activations."""
    return nn.Sequential(
        nn.Linear(latent_dim, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )


def calibrate_blocks(model, z: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """Apply the model's action calibrator, if it has one, to time-aligned blocks."""
    calibrator = getattr(model, "calibrator", None)
    return blocks if calibrator is None else calibrator(z, blocks)


def predict_with(model, emb: torch.Tensor, blocks: torch.Tensor) -> torch.Tensor:
    """Predict next latents, adding the model's latent correction if it has one."""
    predicted = model.predict(emb, model.action_encoder(blocks))
    correction = getattr(model, "correction", None)
    return predicted if correction is None else predicted + correction(emb, blocks)


def recent_errors(
    model, z_history: torch.Tensor, a_history: torch.Tensor, t: int, blocks: int = 3
) -> torch.Tensor:
    """Per-environment one-step error on the last executed blocks before raw step ``t``."""
    errors = []
    for k in range(max(t - blocks * ACTION_BLOCK, 0), t, ACTION_BLOCK):
        z_ctx, b_ctx = keyframe_context(z_history, a_history, k)
        block = a_history[:, k : k + ACTION_BLOCK].reshape(z_history.size(0), 1, -1)
        executed = calibrate_blocks(model, z_ctx, torch.cat([b_ctx, block], dim=1))
        predicted = predict_with(model, z_ctx, executed)[:, -1]
        target = z_history[:, k + ACTION_BLOCK]
        errors.append((predicted - target).pow(2).mean(dim=-1))
    return torch.stack(errors, dim=1).mean(dim=1)


def rollout_latents(
    model, z_ctx: torch.Tensor, b_ctx: torch.Tensor, blocks: torch.Tensor
) -> torch.Tensor:
    """Roll the predictor forward autoregressively.

    Args:
        z_ctx: Observed latents at keyframes (N, T, D), oldest first, T >= 1.
        b_ctx: Executed blocks following all but the last keyframe (N, T-1, A).
        blocks: Future blocks (N, H, A); ``blocks[:, 0]`` follows ``z_ctx[:, -1]``.

    Returns:
        Predicted latents (N, H, D).
    """
    if b_ctx.size(1) != z_ctx.size(1) - 1:
        raise ValueError(f"expected {z_ctx.size(1) - 1} context blocks, got {b_ctx.size(1)}")
    emb = z_ctx
    # Each block is calibrated once, from the latent at the block's start.
    act = calibrate_blocks(model, z_ctx[:, :-1], b_ctx)
    predictions = []
    for h in range(blocks.size(1)):
        next_block = calibrate_blocks(model, emb[:, -1:], blocks[:, h : h + 1])
        act = torch.cat([act, next_block], dim=1)
        pred = predict_with(model, emb[:, -HISTORY:], act[:, -HISTORY:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        predictions.append(pred)
    return torch.cat(predictions, dim=1)


def score_candidates(
    model,
    reward_head,
    z_ctx: torch.Tensor,
    b_ctx: torch.Tensor,
    candidates: torch.Tensor,
    discount: float,
) -> torch.Tensor:
    """Return J(b) from Eq. 3 for candidates (B, S, H, A) as scores (B, S)."""
    batch, samples, horizon, action_dim = candidates.shape
    z = z_ctx.repeat_interleave(samples, dim=0)
    b = b_ctx.repeat_interleave(samples, dim=0)
    flat = candidates.reshape(batch * samples, horizon, action_dim)
    predicted = rollout_latents(model, z, b, flat)
    rewards = reward_head(predicted).squeeze(-1)  # (B * S, H)
    weights = discount ** torch.arange(1, horizon + 1, device=rewards.device).to(rewards.dtype)
    return (rewards * weights).sum(dim=1).reshape(batch, samples)


@torch.no_grad()
def cem_plan(
    model,
    reward_head,
    z_ctx: torch.Tensor,
    b_ctx: torch.Tensor,
    config: PlannerConfig,
    generator: torch.Generator,
    action_dim: int = 2 * ACTION_BLOCK,
) -> torch.Tensor:
    """Optimize action blocks in normalized space and return the first (B, A).

    The proposal starts at zero mean with std 1 on every replan, as in Table A1.
    Like upstream's CEMSolver, the first candidate of each iteration is the
    current mean.
    """
    batch = z_ctx.size(0)
    device = z_ctx.device
    dtype = z_ctx.dtype
    mean = torch.zeros(batch, config.horizon, action_dim, device=device, dtype=dtype)
    std = torch.full_like(mean, config.init_std)
    batch_index = torch.arange(batch, device=device).unsqueeze(1)
    for _ in range(config.iterations):
        noise = torch.randn(
            batch,
            config.num_samples,
            config.horizon,
            action_dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        candidates = mean.unsqueeze(1) + std.unsqueeze(1) * noise
        candidates[:, 0] = mean
        scores = score_candidates(model, reward_head, z_ctx, b_ctx, candidates, config.discount)
        elite_index = scores.topk(config.elites, dim=1).indices
        elites = candidates[batch_index, elite_index]
        mean = elites.mean(dim=1)
        std = elites.std(dim=1, correction=0).clamp_min(config.std_floor)
    return mean[:, 0]


def keyframe_context(
    z_history: torch.Tensor, a_history: torch.Tensor, t: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build planner context at raw step ``t``.

    Args:
        z_history: Latents per raw step (B, T + 1, D); step ``t`` must be filled.
        a_history: Normalized executed actions per raw step (B, T, 2).
        t: The current raw step, a multiple of the block length.

    Returns:
        Up to ``HISTORY`` keyframe latents (B, n, D) ending at ``t`` and the
        ``n - 1`` executed blocks between them (B, n - 1, A).
    """
    if t % ACTION_BLOCK != 0:
        raise ValueError(f"replanning step {t} is not a block boundary")
    steps = [s for s in range(t - (HISTORY - 1) * ACTION_BLOCK, t + 1, ACTION_BLOCK) if s >= 0]
    z_ctx = z_history[:, steps]
    batch = z_history.size(0)
    if len(steps) == 1:
        return z_ctx, a_history.new_zeros(batch, 0, ACTION_BLOCK * a_history.size(-1))
    b_ctx = torch.stack(
        [a_history[:, s : s + ACTION_BLOCK].reshape(batch, -1) for s in steps[:-1]],
        dim=1,
    )
    return z_ctx, b_ctx
