"""Why does a trained calibrator keep working when the world switches back?

Usage:
    python retention.py PROJECT RESULTS_FOLDER PREPARE_JSON WEIGHTS CONFIG

It compares the calibrator's change to each action block with the change the
environment really makes, in the shifted world and in the original one. If the
correction is much smaller than the true shift, the calibrator mostly leaves actions
alone. If it matches, it really learned the shift, and re-planning must explain why
it still works after the switch back.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import numpy as np
import torch

from jepa_ttt.calibration import ActionCalibrator
from jepa_ttt.envs import ShiftedTwoRoom
from jepa_ttt.experiment import ActionStats, fixed_trajectory_mse
from jepa_ttt.lewm import load_pretrained

RESULTS, PREPARE, WEIGHTS, CONFIG = map(Path, sys.argv[2:6])
STATES = 400
TRAJECTORIES = 30
BLOCK = 5
LEFT_X, RIGHT_X, Y = (21.0, 99.0), (125.0, 203.0), (21.0, 203.0)

torch.set_num_threads(6)
stats_json = json.loads(PREPARE.read_text())["action_stats"]
STATS = ActionStats(
    mean=np.array(stats_json["mean"], dtype=np.float32),
    std=np.array(stats_json["std"], dtype=np.float32),
)


def sample_states(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Starts across both rooms, and held-direction command blocks like a planner's."""
    left = rng.random(STATES) < 0.5
    x = np.where(left, rng.uniform(*LEFT_X, STATES), rng.uniform(*RIGHT_X, STATES))
    y = rng.uniform(*Y, STATES)
    held = rng.uniform(-1, 1, size=(STATES, 1, 2))
    commands = np.clip(held + rng.normal(0, 0.2, size=(STATES, BLOCK, 2)), -1, 1)
    return np.stack([x, y], axis=1).astype(np.float32), commands.astype(np.float32)


def true_applied(shift: str, starts: np.ndarray, commands: np.ndarray) -> np.ndarray:
    """Simulate each block and return the actions that really moved the agent."""
    env = ShiftedTwoRoom(shift)
    env.reset_task(seed=0, start_mode="fixed")
    applied = np.zeros_like(commands)
    for i, start in enumerate(starts):
        env.agent_position = torch.as_tensor(start.copy())
        for j in range(BLOCK):
            env.step(commands[i, j])
            applied[i, j] = env.last_applied
    return applied


def encode_starts(model, starts: np.ndarray) -> torch.Tensor:
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=0, start_mode="fixed")
    frames = []
    for start in starts:
        env.agent_position = torch.as_tensor(start.copy())
        frames.append(env.render())
    latents = [
        model.encode_frames(np.stack(frames[i : i + 128])) for i in range(0, len(frames), 128)
    ]
    return torch.cat(latents)


def summarize(calibrated: np.ndarray, commands: np.ndarray, truth: np.ndarray, mask=None) -> dict:
    """Compare block displacement, which is what predictions depend on most."""
    if mask is not None:
        calibrated, commands, truth = calibrated[mask], commands[mask], truth[mask]
    correction = calibrated.sum(1) - commands.sum(1)
    needed = truth.sum(1) - commands.sum(1)
    size = float(np.linalg.norm(correction, axis=1).mean())
    needed_size = float(np.linalg.norm(needed, axis=1).mean())
    error = float(np.linalg.norm(truth.sum(1) - calibrated.sum(1), axis=1).mean())
    return {
        "correction": size,
        "needed": needed_size,
        "error": error,
        "do_nothing_error": needed_size,
    }


base = load_pretrained(WEIGHTS, CONFIG, "cpu")
rng = np.random.default_rng(0)
starts, commands = sample_states(rng)
latents = encode_starts(base, starts)
blocks = torch.as_tensor(STATS.normalize(commands).reshape(STATES, -1), dtype=torch.float32)
in_left = starts[:, 0] < 112.0

trajectories = {}
for shift in ("none", "grid", "wave"):
    data = torch.load(RESULTS / f"trajectories__{shift}__uniform_left__n100.pt", weights_only=True)
    trajectories[shift] = {k: v[:TRAJECTORIES] for k, v in data.items()}

truth_by_world = {w: true_applied(w, starts, commands) for w in ("grid", "wave")}
# In the original world, the applied action is simply the command.
truth_by_world["none"] = commands

print("Frozen model rollout error, on commanded actions:")
for world, data in trajectories.items():
    print(f"  {world}: {fixed_trajectory_mse(base, data):.3f}")
print()

print("=== 1. What the calibrator does to action blocks (displacement, action units)")
print(
    f"{'run':14}{'room':>7}{'size of fix':>13}{'fix needed':>12}{'error shifted':>15}"
    f"{'do nothing':>12}{'error old world':>17}"
)
for shift in ("grid", "wave"):
    for seed in (0, 1, 2):
        state = torch.load(
            RESULTS / f"action_calibration__{shift}__seed{seed}__final.pt",
            map_location="cpu",
            weights_only=True,
        )
        calibrator = ActionCalibrator(STATS.mean, STATS.std, latents.size(1))
        calibrator.load_state_dict(state["calibrator"])
        with torch.no_grad():
            out = calibrator(latents, blocks).numpy().reshape(STATES, BLOCK, 2)
        calibrated = STATS.denormalize(out)
        for room, mask in (("left", in_left), ("right", ~in_left), ("all", None)):
            on_shift = summarize(calibrated, commands, truth_by_world[shift], mask)
            on_old = summarize(calibrated, commands, truth_by_world["none"], mask)
            print(
                f"{shift}/seed{seed:<8}{room:>7}{on_shift['correction']:>13.2f}{on_shift['needed']:>12.2f}"
                f"{on_shift['error']:>15.2f}{on_shift['do_nothing_error']:>12.2f}{on_old['error']:>17.2f}"
            )
print()

print("=== 2. Rollout prediction error on the planner's own trajectories")
print(f"{'run':30}{'shifted world':>15}{'old world':>12}")
for shift in ("grid", "wave"):
    for seed in (0, 1, 2):
        state = torch.load(
            RESULTS / f"action_calibration__{shift}__seed{seed}__final.pt",
            map_location="cpu",
            weights_only=True,
        )
        calibrator = ActionCalibrator(STATS.mean, STATS.std, latents.size(1))
        calibrator.load_state_dict(state["calibrator"])
        base.calibrator = calibrator
        shifted = fixed_trajectory_mse(base, trajectories[shift])
        old = fixed_trajectory_mse(base, trajectories["none"])
        base.calibrator = None
        print(f"{'calibration ' + shift + '/seed' + str(seed):30}{shifted:>15.3f}{old:>12.3f}")

        state = torch.load(
            RESULTS / f"persistent_replay_x16__{shift}__seed{seed}__final.pt",
            map_location="cpu",
            weights_only=True,
        )
        tuned = load_pretrained(WEIGHTS, CONFIG, "cpu")
        tuned.action_encoder.load_state_dict(state["action_encoder"])
        tuned.predictor.load_state_dict(state["predictor"])
        tuned.pred_proj.load_state_dict(state["pred_proj"])
        shifted = fixed_trajectory_mse(tuned, trajectories[shift])
        old = fixed_trajectory_mse(tuned, trajectories["none"])
        print(f"{'x16 ' + shift + '/seed' + str(seed):30}{shifted:>15.3f}{old:>12.3f}")
