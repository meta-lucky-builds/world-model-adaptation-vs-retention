"""Check calibration's premise and learning on the real checkpoint, offline, on a CPU."""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])

import numpy as np
import torch

from jepa_ttt.adaptation import ARMS, TestTimeTrainer, prediction_loss, set_trainable
from jepa_ttt.envs import ShiftedTwoRoom
from jepa_ttt.experiment import ActionStats, fixed_trajectory_mse
from jepa_ttt.lewm import load_pretrained

torch.set_num_threads(6)
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
                # Held direction with noise, so the agent covers the room.
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


def windows(data):
    """All dense windows of (z, b), as the deployment forms them."""
    zs, bs = [], []
    for e in range(data["z"].size(0)):
        for s in range(0, 32):
            zs.append(torch.stack([data["z"][e, s + 5 * j] for j in range(4)]))
            bs.append(
                torch.stack([data["a"][e, s + 5 * j : s + 5 * j + 5].reshape(-1) for j in range(3)])
            )
    return torch.stack(zs), torch.stack(bs)


def train(model, params, lr, z, b, steps):
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(steps):
        idx = rng.choice(z.size(0), 64, replace=False)
        loss = prediction_loss(model, z[idx], b[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()


for shift in ("grid", "wave"):
    model = load_pretrained(T / "weights.pt", T / "config.json", "cpu")
    start = time.time()
    tr = collect(model, shift, 40, 1)
    te = collect(model, shift, 12, 2)
    none = collect(model, "none", 12, 3)
    commanded = fixed_trajectory_mse(model, {"z": te["z"], "a": te["a"]})
    applied = fixed_trajectory_mse(model, {"z": te["z"], "a": te["a_applied"]})
    baseline = fixed_trajectory_mse(model, {"z": none["z"], "a": none["a"]})
    print(
        f"[{shift}] no-shift {baseline:.4f} | shifted, commanded {commanded:.4f} | "
        f"shifted, applied {applied:.4f} | collect {time.time() - start:.0f}s",
        flush=True,
    )

    z, b = windows(tr)
    print(f"[{shift}] windows {z.size(0)}", flush=True)
    for steps in (50, 150, 300):
        cal = load_pretrained(T / "weights.pt", T / "config.json", "cpu")
        trainer = TestTimeTrainer(cal, ARMS["action_calibration"], seed=0, stats=STATS)
        n = sum(p.numel() for p in trainer.params)
        t0 = time.time()
        train(cal, trainer.params, 1e-3, z, b, steps)
        cal_mse = fixed_trajectory_mse(cal, {"z": te["z"], "a": te["a"]})
        fine = load_pretrained(T / "weights.pt", T / "config.json", "cpu")
        params = set_trainable(fine)
        train(fine, params, 5e-5, z, b, steps)
        fine_mse = fixed_trajectory_mse(fine, {"z": te["z"], "a": te["a"]})
        print(
            f"[{shift}] updates {steps}: calibration ({n} params) {cal_mse:.4f} | "
            f"fine-tune ({sum(p.numel() for p in params)} params) {fine_mse:.4f} | "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
