"""Closed-loop episodes, held-out evaluation, diagnostics and deployments."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .adaptation import ARMS, FROZEN, TestTimeTrainer
from .calibration import choose_worlds, fast_steps
from .envs import EPISODE_HORIZON, ShiftedTwoRoom
from .planning import (
    ACTION_BLOCK,
    HISTORY,
    PlannerConfig,
    cem_plan,
    keyframe_context,
    rollout_latents,
)

# Seed suites are disjoint: deployment seeds stay below EVAL_SEED_BASE.
EVAL_SEED_BASE = 900_000_000
TRAJECTORY_SEED_BASE = 800_000_000
DEPLOYMENT_SEED_STRIDE = 1_000_000

ACTION_DIM = 2
MSE_FUTURE_BLOCKS = 5
REVERSAL_EVALS = (10, 25, 50)


@dataclass(frozen=True)
class ActionStats:
    """Per-dimension z-score parameters of the raw dataset actions."""

    mean: np.ndarray
    std: np.ndarray

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        return (actions - self.mean) / self.std

    def denormalize(self, actions: np.ndarray) -> np.ndarray:
        return actions * self.std + self.mean


@dataclass
class EpisodeBatch:
    final_scores: np.ndarray
    peak_scores: np.ndarray
    successes: np.ndarray
    z_history: torch.Tensor
    a_history: torch.Tensor
    replans: int
    applied_history: torch.Tensor | None = None


def run_episodes(
    model,
    reward_head,
    envs: list[ShiftedTwoRoom],
    stats: ActionStats,
    planner: PlannerConfig,
    generator: torch.Generator,
    horizon: int = EPISODE_HORIZON,
    stop_on_success: bool = False,
    record_all_steps: bool = False,
    on_block: Callable[[torch.Tensor, torch.Tensor, int], None] | None = None,
) -> EpisodeBatch:
    """Run already-reset environments under receding-horizon CEM in lockstep.

    Only keyframe latents are encoded unless ``record_all_steps`` is set, which
    dense windows and fixed-trajectory diagnostics require. Environments that
    stop on success drop out of later planning calls.
    """
    if stop_on_success and record_all_steps:
        raise ValueError("recorded trajectories must have a fixed length")
    if horizon % ACTION_BLOCK != 0:
        raise ValueError("horizon must be a whole number of action blocks")

    device = next(model.parameters()).device
    n = len(envs)
    first = model.encode_frames(np.stack([env.render() for env in envs]))
    z_history = first.new_zeros(n, horizon + 1, first.size(-1))
    z_history[:, 0] = first
    a_history = first.new_zeros(n, horizon, ACTION_DIM)
    applied_history = a_history.clone() if record_all_steps else None

    finished = np.zeros(n, dtype=bool)
    successes = np.zeros(n, dtype=bool)
    final_scores = np.array([env.score() for env in envs], dtype=np.float64)
    peak_scores = final_scores.copy()
    replans = 0
    fast_state = None

    for t in range(0, horizon, ACTION_BLOCK):
        active = np.flatnonzero(~finished)
        if active.size == 0:
            break
        active_index = torch.as_tensor(active, device=device)
        z_ctx, b_ctx = keyframe_context(z_history[active_index], a_history[active_index], t)
        calibrator = getattr(model, "calibrator", None)
        if calibrator is not None and calibrator.adapt_fast:
            # This is planning-time state, gone at the end of the episode.
            if fast_state is None:
                fast_state = first.new_zeros(n, ACTION_DIM, ACTION_DIM)
            fast_state[active_index] = fast_steps(
                model,
                fast_state[active_index],
                z_history[active_index],
                a_history[active_index],
                t,
            )
            calibrator.fast = fast_state[active_index]
            block = cem_plan(model, reward_head, z_ctx, b_ctx, planner, generator)
            calibrator.fast = None
        elif calibrator is not None and calibrator.gated:
            use_now = choose_worlds(model, z_history[active_index], a_history[active_index], t)
            calibrator.last_on = bool(use_now[0])
            block = first.new_zeros(active.size, ACTION_BLOCK * ACTION_DIM)
            for choice in (True, False):
                chosen = use_now == choice
                if chosen.any():
                    calibrator.enabled = choice
                    block[chosen] = cem_plan(
                        model, reward_head, z_ctx[chosen], b_ctx[chosen], planner, generator
                    )
            calibrator.enabled = True
        else:
            block = cem_plan(model, reward_head, z_ctx, b_ctx, planner, generator)
        replans += 1

        # Execute the clipped command and store exactly what was commanded.
        planned = stats.denormalize(
            block.detach().cpu().numpy().reshape(active.size, ACTION_BLOCK, ACTION_DIM)
        )
        commands = np.clip(planned, -1.0, 1.0).astype(np.float32)
        executed = stats.normalize(commands).astype(np.float32)
        a_history[active_index, t : t + ACTION_BLOCK] = torch.as_tensor(
            executed, device=device, dtype=a_history.dtype
        )

        step_frames: list[list[np.ndarray]] = [[] for _ in range(ACTION_BLOCK)]
        applied = np.zeros_like(commands)
        for row, env_id in enumerate(active):
            env = envs[env_id]
            for offset in range(ACTION_BLOCK):
                env.step(commands[row, offset])
                applied[row, offset] = env.last_applied
                score = env.score()
                final_scores[env_id] = score
                peak_scores[env_id] = max(peak_scores[env_id], score)
                successes[env_id] = successes[env_id] or env.reached_goal()
                if record_all_steps:
                    step_frames[offset].append(env.render())
                if stop_on_success and env.reached_goal():
                    finished[env_id] = True
                    break

        if record_all_steps:
            applied_history[active_index, t : t + ACTION_BLOCK] = torch.as_tensor(
                stats.normalize(applied).astype(np.float32),
                device=device,
                dtype=applied_history.dtype,
            )
            # Encode the whole block in one batch rather than one call per step.
            encoded = model.encode_frames(
                np.stack([frame for frames in step_frames for frame in frames])
            ).reshape(ACTION_BLOCK, active.size, -1)
            for offset in range(ACTION_BLOCK):
                z_history[active_index, t + offset + 1] = encoded[offset]
        else:
            running = np.flatnonzero(~finished)
            if running.size > 0:
                z_history[torch.as_tensor(running, device=device), t + ACTION_BLOCK] = (
                    model.encode_frames(np.stack([envs[i].render() for i in running]))
                )

        if on_block is not None:
            on_block(z_history, a_history, t)

    return EpisodeBatch(
        final_scores=final_scores,
        peak_scores=peak_scores,
        successes=successes,
        z_history=z_history,
        a_history=a_history,
        replans=replans,
        applied_history=applied_history,
    )


def _chunks(total: int, size: int):
    for start in range(0, total, size):
        yield start, min(start + size, total)


def evaluate(
    model,
    reward_head,
    stats: ActionStats,
    shift: str,
    planner: PlannerConfig,
    episodes: int,
    batch_size: int,
    start_mode: str,
    seed_base: int = EVAL_SEED_BASE,
) -> dict:
    """Score the current model on the fixed held-out suite without updating it.

    Held-out episodes stop on success, and their data never reach a buffer.
    """
    device = next(model.parameters()).device
    finals, peaks, successes = [], [], []
    for start, end in _chunks(episodes, batch_size):
        envs = [ShiftedTwoRoom(shift) for _ in range(start, end)]
        for i, env in enumerate(envs):
            env.reset_task(seed_base + start + i, start_mode)
        generator = torch.Generator(device=device).manual_seed(seed_base + start)
        batch = run_episodes(
            model,
            reward_head,
            envs,
            stats,
            planner,
            generator,
            stop_on_success=True,
        )
        finals.append(batch.final_scores)
        peaks.append(batch.peak_scores)
        successes.append(batch.successes)
    return {
        "score_final": float(np.concatenate(finals).mean()),
        "score_peak": float(np.concatenate(peaks).mean()),
        "success_rate": float(np.concatenate(successes).mean()),
    }


def collect_fixed_trajectories(
    model,
    reward_head,
    stats: ActionStats,
    shift: str,
    planner: PlannerConfig,
    episodes: int,
    batch_size: int,
    start_mode: str,
) -> dict[str, torch.Tensor]:
    """Collect 50-step CEM trajectories with the pretrained model (Appendix A5.1)."""
    device = next(model.parameters()).device
    zs, actions, applied = [], [], []
    for start, end in _chunks(episodes, batch_size):
        envs = [ShiftedTwoRoom(shift) for _ in range(start, end)]
        for i, env in enumerate(envs):
            env.reset_task(TRAJECTORY_SEED_BASE + start + i, start_mode)
        generator = torch.Generator(device=device).manual_seed(TRAJECTORY_SEED_BASE + start)
        batch = run_episodes(
            model,
            reward_head,
            envs,
            stats,
            planner,
            generator,
            record_all_steps=True,
        )
        zs.append(batch.z_history.cpu())
        actions.append(batch.a_history.cpu())
        applied.append(batch.applied_history.cpu())
    return {"z": torch.cat(zs), "a": torch.cat(actions), "a_applied": torch.cat(applied)}


@torch.no_grad()
def fixed_trajectory_mse(model, trajectories: dict[str, torch.Tensor]) -> float:
    """Five-block autoregressive latent MSE over every sliding window (Appendix A5.1).

    Each window has three context frames and five future frames five raw steps
    apart. Windows are averaged within an episode before averaging episodes.
    """
    device = next(model.parameters()).device
    z = trajectories["z"].to(device)
    a = trajectories["a"].to(device)
    episodes, steps = z.size(0), z.size(1) - 1
    frames = HISTORY + MSE_FUTURE_BLOCKS
    last_start = steps - (frames - 1) * ACTION_BLOCK
    if last_start < 0:
        raise ValueError(f"{steps} steps cannot hold a {frames}-frame window")

    def block_at(step: int) -> torch.Tensor:
        return a[:, step : step + ACTION_BLOCK].reshape(episodes, -1)

    per_window = []
    for start in range(last_start + 1):
        keyframes = [start + j * ACTION_BLOCK for j in range(frames)]
        context = keyframes[:HISTORY]
        z_ctx = z[:, context]
        b_ctx = torch.stack([block_at(s) for s in context[:-1]], dim=1)
        future = torch.stack([block_at(s) for s in keyframes[HISTORY - 1 : -1]], dim=1)
        predicted = rollout_latents(model, z_ctx, b_ctx, future)
        error = (predicted - z[:, keyframes[HISTORY:]]).pow(2).mean(dim=(1, 2))
        per_window.append(error)
    return float(torch.stack(per_window, dim=1).mean(dim=1).mean())


def deployment_seed(run_seed: int, episode: int) -> int:
    seed = run_seed * DEPLOYMENT_SEED_STRIDE + episode
    if seed >= TRAJECTORY_SEED_BASE:
        raise ValueError(f"run seed {run_seed} collides with the held-out suites")
    return seed


def run_deployment(
    model,
    reward_head,
    stats: ActionStats,
    arm: str,
    shift: str,
    run_seed: int,
    planner: PlannerConfig,
    episodes: int,
    eval_every: int,
    eval_episodes: int,
    eval_batch: int,
    start_mode: str,
    trajectories: dict[str, torch.Tensor] | None,
    log: Callable[[dict], None],
    final_state_path: Path | None = None,
    dense_until: int = 0,
    dense_every: int = 1,
    reversal_episodes: int = 0,
) -> None:
    """Interleave held-out evaluation with one sequence of deployment episodes.

    With ``dense_until`` set, evaluation also runs every ``dense_every`` episodes
    up to that episode. With ``reversal_episodes`` set, the deployment then returns
    to the training dynamics without telling the model, and keeps adapting.
    """
    if arm != FROZEN and arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    device = next(model.parameters()).device

    def record(episode: int, trainer: TestTimeTrainer | None, online: list[float]):
        started = time.perf_counter()
        metrics = evaluate(
            model,
            reward_head,
            stats,
            shift,
            planner,
            eval_episodes,
            eval_batch,
            start_mode,
        )
        row = {
            "kind": "eval",
            "arm": arm,
            "shift": shift,
            "seed": run_seed,
            "episode": episode,
            **metrics,
            "latent_mse": (
                fixed_trajectory_mse(model, trajectories) if trajectories is not None else None
            ),
            "updates": trainer.total_updates if trainer else 0,
            "last_loss": trainer.last_loss if trainer else None,
            "online_score": float(np.mean(online)) if online else None,
            "eval_seconds": time.perf_counter() - started,
            **(trainer.gate_stats() if trainer else {}),
        }
        # The premise of action calibration: the frozen model should predict the
        # shifted world well when it is told the actions that were really applied.
        if trainer is None and trajectories is not None and "a_applied" in trajectories:
            row["latent_mse_applied"] = fixed_trajectory_mse(
                model, {"z": trajectories["z"], "a": trajectories["a_applied"]}
            )
        log(row)

    if arm == FROZEN:
        record(0, None, [])
        return

    trainer = TestTimeTrainer(model, ARMS[arm], seed=run_seed, stats=stats)
    env = ShiftedTwoRoom(shift)
    generator = torch.Generator(device=device).manual_seed(run_seed)
    record(0, trainer, [])
    online: list[float] = []
    for episode in range(1, episodes + 1):
        trainer.start_episode()
        env.reset_task(deployment_seed(run_seed, episode), start_mode)
        batch = run_episodes(
            model,
            reward_head,
            [env],
            stats,
            planner,
            generator,
            record_all_steps=True,
            on_block=trainer.add_block,
        )
        online.append(float(batch.final_scores[0]))
        dense = episode <= dense_until and episode % dense_every == 0
        if episode % eval_every == 0 or dense:
            record(episode, trainer, online)
            online = []
    if final_state_path is not None:
        calibrator = getattr(model, "calibrator", None)
        torch.save(
            {
                "action_encoder": model.action_encoder.state_dict(),
                "predictor": model.predictor.state_dict(),
                "pred_proj": model.pred_proj.state_dict(),
                "calibrator": None if calibrator is None else calibrator.state_dict(),
                "correction": (
                    model.correction.state_dict() if hasattr(model, "correction") else None
                ),
                "total_updates": trainer.total_updates,
            },
            final_state_path,
        )
    if reversal_episodes > 0:

        def record_reversal(episode: int) -> None:
            metrics = evaluate(
                model,
                reward_head,
                stats,
                "none",
                planner,
                eval_episodes,
                eval_batch,
                start_mode,
            )
            log(
                {
                    "kind": "reversal",
                    "arm": arm,
                    "shift": shift,
                    "seed": run_seed,
                    "episode": episode,
                    **metrics,
                    "updates": trainer.total_updates,
                    **trainer.gate_stats(),
                }
            )

        record_reversal(0)
        back = ShiftedTwoRoom("none")
        for episode in range(1, reversal_episodes + 1):
            trainer.start_episode()
            back.reset_task(deployment_seed(run_seed, episodes + episode), start_mode)
            run_episodes(
                model,
                reward_head,
                [back],
                stats,
                planner,
                generator,
                record_all_steps=True,
                on_block=trainer.add_block,
            )
            if episode in REVERSAL_EVALS or episode == reversal_episodes:
                record_reversal(episode)
