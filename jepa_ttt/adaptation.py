"""Dense prediction windows, update schedules and test-time training (Sec. 3.3-3.4)."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .calibration import OLD_WORLD_ERROR, ActionCalibrator, LatentCorrection
from .planning import (
    ACTION_BLOCK,
    HISTORY,
    calibrate_blocks,
    predict_with,
    rollout_latents,
)

# Appendix A1.2: a window start is valid once (C + 1) * K raw rows exist.
ROWS_PER_WINDOW = (HISTORY + 1) * ACTION_BLOCK


class DenseWindowStream:
    """Turn one episode's (latent, normalized action) rows into dense windows."""

    def __init__(self):
        self._latents: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []

    def add_row(self, latent: torch.Tensor, action: torch.Tensor):
        """Add a row and return the window (z, b) it completes, if any.

        ``z`` holds the C + 1 keyframe latents (C + 1, D) and ``b`` the C action
        blocks between them (C, K * action_dim), both on the CPU.
        """
        self._actions.append(action.detach().cpu())
        return self.add_observation(latent)

    def add_observation(self, latent: torch.Tensor):
        """Add a row without an action, as the episode's final observation is.

        A window's last block is unused by the loss, so this row can complete one.
        """
        self._latents.append(latent.detach().cpu())
        start = len(self._latents) - ROWS_PER_WINDOW
        if start < 0:
            return None
        z = torch.stack([self._latents[start + j * ACTION_BLOCK] for j in range(HISTORY + 1)])
        b = torch.stack(
            [
                torch.cat(self._actions[start + j * ACTION_BLOCK : start + (j + 1) * ACTION_BLOCK])
                for j in range(HISTORY)
            ]
        )
        return z, b


class WindowBuffer:
    """Growing window store sampled uniformly, without replacement in a batch."""

    def __init__(self):
        self._z: list[torch.Tensor] = []
        self._b: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self._z)

    def add(self, window: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._z.append(window[0])
        self._b.append(window[1])

    def clear(self) -> None:
        self._z.clear()
        self._b.clear()

    def sample(self, batch_size: int, rng: np.random.Generator):
        index = rng.choice(len(self), size=batch_size, replace=False)
        return (
            torch.stack([self._z[i] for i in index]),
            torch.stack([self._b[i] for i in index]),
        )


@dataclass(frozen=True)
class ArmConfig:
    """One update rule.

    Attributes:
        batch_size: Test-time minibatch size.
        episode_local: Keep only the current episode's windows and count them
            per episode, as in the matched persistence control (Appendix A6.1).
        reset_each_episode: Restore pretrained weights and a fresh optimizer at
            every episode boundary.
        updates_per_batch: Updates owed per ``batch_size`` new windows. The
            paper uses 1; a larger value reuses the replay buffer more.
        calibrate: Freeze the whole world model and train an action calibrator.
        lr: Test-time learning rate.
        gated: With a calibrator, plan with the frozen model alone when it clearly
            predicts better, and keep only windows from the calibrated world.
        adapt_fast: With a calibrator, also adapt a per-environment fast matrix
            during every episode, including held-out ones.
        correct: Freeze the whole world model and train a latent correction.
        rollout_weight: Weight of an added free-running rollout loss.
        anchor_weight: Weight of a penalty on drifting from the pretrained weights.
    """

    batch_size: int
    episode_local: bool
    reset_each_episode: bool
    updates_per_batch: int = 1
    calibrate: bool = False
    lr: float = 5e-5
    gated: bool = False
    adapt_fast: bool = False
    correct: bool = False
    rollout_weight: float = 0.0
    anchor_weight: float = 0.0


ARMS = {
    "persistent_replay": ArmConfig(64, episode_local=False, reset_each_episode=False),
    "persistent_local": ArmConfig(8, episode_local=True, reset_each_episode=False),
    "reset_local": ArmConfig(8, episode_local=True, reset_each_episode=True),
    # Extension beyond the paper, fixed before any result exists: same data and
    # learning rate as persistent_replay, four times the updates.
    "persistent_replay_x4": ArmConfig(
        64, episode_local=False, reset_each_episode=False, updates_per_batch=4
    ),
    "persistent_replay_x8": ArmConfig(
        64, episode_local=False, reset_each_episode=False, updates_per_batch=8
    ),
    "persistent_replay_x16": ArmConfig(
        64, episode_local=False, reset_each_episode=False, updates_per_batch=16
    ),
    # Fixed before any result exists. The network starts from zero output and is
    # far smaller than the predictor, so it takes a larger rate and more updates.
    "action_calibration": ArmConfig(
        64,
        episode_local=False,
        reset_each_episode=False,
        updates_per_batch=16,
        calibrate=True,
        lr=1e-3,
    ),
    # Further ideas, also fixed before any of their results. Each changes one thing.
    "calibration_before_now": ArmConfig(
        64,
        episode_local=False,
        reset_each_episode=False,
        updates_per_batch=16,
        calibrate=True,
        lr=1e-3,
        gated=True,
    ),
    "calibration_fast": ArmConfig(
        64,
        episode_local=False,
        reset_each_episode=False,
        updates_per_batch=16,
        calibrate=True,
        lr=1e-3,
        adapt_fast=True,
    ),
    "latent_correction": ArmConfig(
        64,
        episode_local=False,
        reset_each_episode=False,
        updates_per_batch=16,
        correct=True,
        lr=1e-3,
    ),
    "rollout_loss_x4": ArmConfig(
        64,
        episode_local=False,
        reset_each_episode=False,
        updates_per_batch=4,
        rollout_weight=0.5,
    ),
    "anchored_x4": ArmConfig(
        64,
        episode_local=False,
        reset_each_episode=False,
        updates_per_batch=4,
        anchor_weight=0.01,
    ),
}
FROZEN = "frozen"
ALL_ARMS = (FROZEN, *ARMS)
STAGE1_ARMS = (
    "persistent_replay",
    "persistent_local",
    "reset_local",
    "persistent_replay_x4",
)
CALIBRATION_ARMS = (
    "persistent_replay_x4",
    "persistent_replay_x8",
    "persistent_replay_x16",
    "action_calibration",
)
IDEA_ARMS = (
    "calibration_before_now",
    "calibration_fast",
    "latent_correction",
    "rollout_loss_x4",
    "anchored_x4",
)

TEST_WEIGHT_DECAY = 1e-3
GRAD_CLIP = 1.0


def set_trainable(model) -> list[torch.nn.Parameter]:
    """Enable gradients only for the modules Table A1 lists as trainable."""
    model.requires_grad_(False)
    params: list[torch.nn.Parameter] = []
    for module in (model.action_encoder, model.predictor, model.pred_proj):
        module.requires_grad_(True)
        params.extend(module.parameters())
    return params


def attach_calibrator(model, stats, seed: int, arm: ArmConfig) -> list[torch.nn.Parameter]:
    """Freeze the whole world model and give it a new trainable calibrator."""
    if stats is None:
        raise ValueError("action calibration needs the dataset's action statistics")
    model.requires_grad_(False)
    device = next(model.parameters()).device
    latent_dim = model.projector.net[-1].out_features
    torch.manual_seed(seed)
    model.calibrator = ActionCalibrator(stats.mean, stats.std, latent_dim).to(device)
    model.calibrator.gated = arm.gated
    model.calibrator.adapt_fast = arm.adapt_fast
    return list(model.calibrator.parameters())


def attach_correction(model, stats, seed: int) -> list[torch.nn.Parameter]:
    """Freeze the whole world model and give it a new trainable latent correction."""
    if stats is None:
        raise ValueError("latent correction needs the dataset's action statistics")
    model.requires_grad_(False)
    device = next(model.parameters()).device
    latent_dim = model.projector.net[-1].out_features
    torch.manual_seed(seed)
    model.correction = LatentCorrection(stats.mean, stats.std, latent_dim).to(device)
    return list(model.correction.parameters())


def prediction_loss(model, z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Eq. 4: teacher-forced latent prediction over a window's C positions."""
    blocks = calibrate_blocks(model, z[:, :HISTORY], b)
    predicted = predict_with(model, z[:, :HISTORY], blocks)
    return F.mse_loss(predicted, z[:, 1:].detach())


def rollout_loss(model, z: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Free-running error: predict a window's later latents from its first one only."""
    predicted = rollout_latents(model, z[:, :1], b[:, :0], b)
    return F.mse_loss(predicted, z[:, 1:].detach())


class TestTimeTrainer:
    """Own the optimizer, buffer and update schedule of one deployment."""

    __test__ = False

    def __init__(self, model, arm: ArmConfig, seed: int, stats=None):
        self.model = model
        self.arm = arm
        if arm.calibrate:
            self.params = attach_calibrator(model, stats, seed, arm)
        elif arm.correct:
            self.params = attach_correction(model, stats, seed)
        else:
            self.params = set_trainable(model)
        self.anchors = [p.detach().clone() for p in self.params] if arm.anchor_weight else []
        self.episode_blocks = 0
        self.episode_on = 0
        self.gate_blocks = 0
        self.gate_on_blocks = 0
        self.windows_dropped = 0
        self.rng = np.random.default_rng(seed)
        self.buffer = WindowBuffer()
        self.stream = DenseWindowStream()
        self.windows_seen = 0
        self.updates_done = 0
        self.total_updates = 0
        self.last_loss = float("nan")
        self.optimizer = self._make_optimizer()
        modules = (model.action_encoder, model.predictor, model.pred_proj)
        self._pretrained = [copy.deepcopy(m.state_dict()) for m in modules]
        self._modules = modules

    def _make_optimizer(self) -> torch.optim.AdamW:
        return torch.optim.AdamW(self.params, lr=self.arm.lr, weight_decay=TEST_WEIGHT_DECAY)

    def start_episode(self) -> None:
        """Apply the arm's episode-boundary rule before an episode starts."""
        self.stream = DenseWindowStream()
        self.episode_blocks = 0
        self.episode_on = 0
        if self.arm.episode_local:
            self.buffer.clear()
            self.windows_seen = 0
            self.updates_done = 0
        if self.arm.reset_each_episode:
            for module, state in zip(self._modules, self._pretrained):
                module.load_state_dict(state)
            self.optimizer = self._make_optimizer()

    def add_block(self, z_history, a_history, t: int) -> None:
        """Store the rows of the block executed at step ``t``, then maybe update."""
        windows = [
            self.stream.add_row(z_history[0, row], a_history[0, row])
            for row in range(t, t + ACTION_BLOCK)
        ]
        if t + ACTION_BLOCK == z_history.size(1) - 1:
            windows.append(self.stream.add_observation(z_history[0, t + ACTION_BLOCK]))
        keep = True
        if self.arm.gated:
            calibrator = self.model.calibrator
            # Old-world windows would teach the calibrator to undo itself. A frozen
            # model that still predicts badly means this isn't the old world, so
            # those windows stay, and a calibrator that went wrong can recover.
            keep = calibrator.last_on or calibrator.last_before_error >= OLD_WORLD_ERROR
            self.episode_blocks += 1
            self.episode_on += int(calibrator.last_on)
            calibrator.default_on = 2 * self.episode_on >= self.episode_blocks
            self.gate_blocks += 1
            self.gate_on_blocks += int(calibrator.last_on)
        for window in windows:
            if window is None:
                continue
            if keep:
                self.buffer.add(window)
                self.windows_seen += 1
            else:
                self.windows_dropped += 1
        self.maybe_update()

    def gate_stats(self) -> dict:
        """How often before-plus-now used the calibrator, for the logs."""
        if not self.arm.gated:
            return {}
        return {
            "gate_on_rate": self.gate_on_blocks / max(self.gate_blocks, 1),
            "windows_dropped": self.windows_dropped,
        }

    def maybe_update(self) -> bool:
        """Catch up to floor(windows / batch) * updates_per_batch updates.

        A call does at most ``updates_per_batch`` updates, so the paper's arms
        do at most one per replan.
        """
        owed = self.windows_seen // self.arm.batch_size * self.arm.updates_per_batch
        updated = False
        for _ in range(self.arm.updates_per_batch):
            if owed <= self.updates_done:
                break
            self._update_once()
            updated = True
        return updated

    def _update_once(self) -> None:
        device = self.params[0].device
        z, b = self.buffer.sample(self.arm.batch_size, self.rng)
        self.model.eval()
        z, b = z.to(device), b.to(device)
        loss = prediction_loss(self.model, z, b)
        if self.arm.rollout_weight:
            loss = loss + self.arm.rollout_weight * rollout_loss(self.model, z, b)
        if self.anchors:
            drift = sum((p - a).pow(2).sum() for p, a in zip(self.params, self.anchors))
            loss = loss + 0.5 * self.arm.anchor_weight * drift
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, GRAD_CLIP)
        self.optimizer.step()
        self.updates_done += 1
        self.total_updates += 1
        self.last_loss = float(loss.detach())
