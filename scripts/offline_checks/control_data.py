"""Shared data collection for the offline calibration checks."""

import os
from pathlib import Path

import numpy as np
import torch

from jepa_ttt.envs import ShiftedTwoRoom
from jepa_ttt.experiment import ActionStats

T = Path(os.environ["JEPA_TTT_OFFLINE_DIR"])  # holds weights.pt and config.json
STATS = ActionStats(
    mean=np.array([0.003093412611633539, -0.05298233404755592], dtype=np.float32),
    std=np.array([0.8674762845039368, 0.8677660226821899], dtype=np.float32),
)


def collect(model, shift, episodes, seed):
    rng = np.random.default_rng(seed)
    zs, cmd, app = [], [], []
    env = ShiftedTwoRoom(shift)
    for ep in range(episodes):
        env.reset_task(seed=seed * 1000 + ep)
        frames, c, a = [env.render()], [], []
        for t in range(50):
            if t % 5 == 0:
                held = rng.uniform(-1, 1, size=2)
            command = np.clip(held + rng.normal(0, 0.3, size=2), -1, 1).astype(np.float32)
            env.step(command)
            c.append(command)
            a.append(env.last_applied.copy())
            frames.append(env.render())
        zs.append(model.encode_frames(np.stack(frames)))
        cmd.append(STATS.normalize(np.stack(c)))
        app.append(STATS.normalize(np.stack(a)))
    return {
        "z": torch.stack(zs),
        "a": torch.as_tensor(np.stack(cmd), dtype=torch.float32),
        "a_applied": torch.as_tensor(np.stack(app), dtype=torch.float32),
    }
