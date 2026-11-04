"""Separate representation from optimization for the action calibrator."""

import sys
import time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from control_data import STATS, T, collect

from jepa_ttt.adaptation import ARMS, TestTimeTrainer, prediction_loss
from jepa_ttt.experiment import fixed_trajectory_mse
from jepa_ttt.lewm import load_pretrained

torch.set_num_threads(6)


def windows(data):
    zs, bs, applied = [], [], []
    for e in range(data["z"].size(0)):
        for s in range(0, 32):
            zs.append(torch.stack([data["z"][e, s + 5 * j] for j in range(4)]))
            bs.append(
                torch.stack([data["a"][e, s + 5 * j : s + 5 * j + 5].reshape(-1) for j in range(3)])
            )
            applied.append(
                torch.stack(
                    [data["a_applied"][e, s + 5 * j : s + 5 * j + 5].reshape(-1) for j in range(3)]
                )
            )
    return torch.stack(zs), torch.stack(bs), torch.stack(applied)


def block_error(cal, z, b, applied):
    with torch.no_grad():
        return F.mse_loss(cal.calibrator(z[:, :3], b), applied).item()


shift = sys.argv[2]
cache = T / f"data_{shift}.pt"
if cache.exists():
    data = torch.load(cache, weights_only=True)
else:
    model = load_pretrained(T / "weights.pt", T / "config.json", "cpu")
    data = {"train": collect(model, shift, 40, 1), "test": collect(model, shift, 12, 2)}
    torch.save(data, cache)
tr, te = data["train"], data["test"]
z, b, applied = windows(tr)
zt, bt, appliedt = windows(te)
print("identity block error", F.mse_loss(bt, appliedt).item(), flush=True)

# A: supervised on the true applied actions, so the frozen model plays no part.
cal = load_pretrained(T / "weights.pt", T / "config.json", "cpu")
params = TestTimeTrainer(cal, ARMS["action_calibration"], seed=0, stats=STATS).params
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
rng = np.random.default_rng(0)
start = time.time()
for step in range(1, 4001):
    i = rng.choice(z.size(0), 64, replace=False)
    loss = F.mse_loss(cal.calibrator(z[i][:, :3], b[i]), applied[i])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if step in (100, 300, 1000, 2000, 4000):
        mse = fixed_trajectory_mse(cal, {"z": te["z"], "a": te["a"]})
        print(
            f"supervised {step}: block train {block_error(cal, z, b, applied):.4f} "
            f"test {block_error(cal, zt, bt, appliedt):.4f} | rollout mse {mse:.4f} | "
            f"{time.time() - start:.0f}s",
            flush=True,
        )

# B: the real objective, through the frozen predictor, with a longer horizon.
cal = load_pretrained(T / "weights.pt", T / "config.json", "cpu")
params = TestTimeTrainer(cal, ARMS["action_calibration"], seed=0, stats=STATS).params
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-3)
rng = np.random.default_rng(0)
start = time.time()
for step in range(1, 1501):
    i = rng.choice(z.size(0), 64, replace=False)
    loss = prediction_loss(cal, z[i], b[i])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()
    if step in (300, 700, 1500):
        with torch.no_grad():
            train_loss = prediction_loss(cal, z[:512], b[:512]).item()
            test_loss = prediction_loss(cal, zt, bt).item()
        mse = fixed_trajectory_mse(cal, {"z": te["z"], "a": te["a"]})
        print(
            f"self-supervised {step}: one-step train {train_loss:.4f} test {test_loss:.4f} | "
            f"block test {block_error(cal, zt, bt, appliedt):.4f} | rollout mse {mse:.4f} | "
            f"{time.time() - start:.0f}s",
            flush=True,
        )
