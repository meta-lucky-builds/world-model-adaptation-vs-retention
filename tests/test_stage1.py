"""Logic tests with tiny settings; they verify behavior, not benchmark scores.

Set JEPA_TTT_WEIGHTS and JEPA_TTT_CONFIG to also test the published checkpoint,
and JEPA_TTT_UPSTREAM_MODULE to compare the vendored classes with upstream.
"""

from __future__ import annotations

import ast
import io
import math
import os
import subprocess
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch
import zstandard
from torch import nn

from jepa_ttt.adaptation import (
    ARMS,
    ROWS_PER_WINDOW,
    DenseWindowStream,
    TestTimeTrainer,
    WindowBuffer,
    prediction_loss,
    set_trainable,
)
from jepa_ttt.envs import (
    DOORWAY,
    GOAL,
    LEFT_ROOM_X,
    LEFT_ROOM_Y,
    ShiftedTwoRoom,
    rotate,
    rotation_angle,
    task_distance,
    task_score,
    task_scores,
)
from jepa_ttt.experiment import (
    ActionStats,
    collect_fixed_trajectories,
    deployment_seed,
    evaluate,
    fixed_trajectory_mse,
    run_deployment,
    run_episodes,
)
from jepa_ttt.lewm import (
    MLP,
    ARPredictor,
    Embedder,
    LeWorldModel,
    build_model,
    load_pretrained,
    preprocess_frames,
)
from jepa_ttt.planning import (
    PlannerConfig,
    cem_plan,
    keyframe_context,
    make_reward_head,
    rollout_latents,
    score_candidates,
)
from jepa_ttt.prepare import (
    compute_action_stats,
    encode_episodes,
    extract_dataset,
    fit_reward_head,
    require_device,
    validate_dynamics,
    validate_render_match,
)
from jepa_ttt.report import paired_auc_gains, run_auc, summarize
from jepa_ttt.sweep import is_finished, plan_runs, run_jobs

CONFIG = {
    "encoder": {
        "size": "tiny",
        "patch_size": 14,
        "image_size": 224,
        "pretrained": False,
        "use_mask_token": False,
    },
    "predictor": {
        "_target_": "ignored",
        "num_frames": 3,
        "input_dim": 192,
        "hidden_dim": 192,
        "output_dim": 192,
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dim_head": 64,
        "dropout": 0.1,
        "emb_dropout": 0.0,
    },
    "action_encoder": {"_target_": "ignored", "input_dim": 10, "emb_dim": 192},
    "projector": {"input_dim": 192, "output_dim": 192, "hidden_dim": 2048},
    "pred_proj": {"input_dim": 192, "output_dim": 192, "hidden_dim": 2048},
}
STATS = ActionStats(
    mean=np.array([0.1, -0.2], dtype=np.float32),
    std=np.array([0.8, 0.9], dtype=np.float32),
)
TINY_PLANNER = PlannerConfig(num_samples=6, iterations=2, elites=2)


def small_model() -> LeWorldModel:
    """A cheap model with the real module layout and latent size 16."""
    torch.manual_seed(0)

    class TinyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=8, stride=8)

        def forward(self, pixels, interpolate_pos_encoding=False):
            cls = self.conv(pixels).mean(dim=(2, 3))
            return type("Output", (), {"last_hidden_state": cls.unsqueeze(1)})

    model = LeWorldModel(
        encoder=TinyEncoder(),
        predictor=ARPredictor(
            num_frames=3,
            depth=2,
            heads=2,
            mlp_dim=32,
            input_dim=16,
            hidden_dim=16,
            output_dim=16,
            dim_head=8,
            dropout=0.1,
        ),
        action_encoder=Embedder(input_dim=10, smoothed_dim=10, emb_dim=16),
        projector=MLP(16, 32, 16, norm_fn=nn.BatchNorm1d),
        pred_proj=MLP(16, 32, 16, norm_fn=nn.BatchNorm1d),
    )
    # AdaLN-zero starts with actions having no effect, unlike a trained model.
    for layer in model.predictor.transformer.layers:
        nn.init.normal_(layer.adaLN_modulation[-1].weight, std=0.05)
    model.eval()
    model.requires_grad_(False)
    return model


def small_head() -> nn.Module:
    torch.manual_seed(1)
    head = make_reward_head(16)
    head.eval()
    head.requires_grad_(False)
    return head


def reference_rollout(model, z_ctx, b_ctx, blocks, history_size=3):
    """Upstream ``JEPA.rollout``'s loop, restated for comparison."""
    emb = z_ctx.clone()
    act = torch.cat([b_ctx, blocks[:, :1]], dim=1)
    act_future = blocks[:, 1:]
    for t in range(blocks.size(1) - 1):
        act_emb = model.action_encoder(act)
        pred = model.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
        emb = torch.cat([emb, pred], dim=1)
        act = torch.cat([act, act_future[:, t : t + 1]], dim=1)
    act_emb = model.action_encoder(act)
    pred = model.predict(emb[:, -history_size:], act_emb[:, -history_size:])[:, -1:]
    emb = torch.cat([emb, pred], dim=1)
    return emb[:, z_ctx.size(1) :]


# ---------------------------------------------------------------- shifts and score


def test_rotation_matrix_matches_eq_a5():
    assert rotate(np.array([1.0, 0.0]), math.pi / 2) == pytest.approx([0.0, 1.0], abs=1e-6)
    assert rotate(np.array([0.0, 1.0]), math.pi / 2) == pytest.approx([-1.0, 0.0], abs=1e-6)
    assert np.linalg.norm(rotate(np.array([0.3, -0.7]), 1.1)) == pytest.approx(
        math.hypot(0.3, 0.7), rel=1e-6
    )


def test_wave_field_matches_eq_a7():
    assert rotation_angle("wave", 20.0, 77.0) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(rotation_angle("wave", 42.0, 3.0)) == pytest.approx(75.0)
    assert math.degrees(rotation_angle("wave", 86.0, 3.0)) == pytest.approx(-75.0)
    assert rotation_angle("wave", 108.0, 9.0) == pytest.approx(0.0, abs=1e-9)


def test_grid_field_matches_eq_a8():
    assert rotation_angle("grid", 20.0, 20.0) == pytest.approx(0.0, abs=1e-9)
    assert math.degrees(rotation_angle("grid", 44.0, 20.0)) == pytest.approx(75.0)
    assert math.degrees(rotation_angle("grid", 44.0, 68.0)) == pytest.approx(-75.0)
    assert rotation_angle("grid", 44.0, 44.0) == pytest.approx(0.0, abs=1e-9)
    assert rotation_angle("none", 44.0, 20.0) == 0.0
    with pytest.raises(ValueError):
        rotation_angle("other", 0.0, 0.0)


def test_score_matches_appendix_a22():
    assert task_score(GOAL) == pytest.approx(1.0)
    start = np.array([60.0, 112.0])
    expected = np.linalg.norm(start - DOORWAY) + np.linalg.norm(DOORWAY - GOAL)
    assert task_distance(start) == pytest.approx(expected)
    assert task_score(start) == pytest.approx(math.exp(-expected / 64.0))
    right = np.array([130.0, 150.0])
    assert task_distance(right) == pytest.approx(np.linalg.norm(right - GOAL))
    assert GOAL.tolist() == [164.0, 112.0] and DOORWAY.tolist() == [112.0, 49.0]
    # Passing through the door raises the score, unlike straight-line distance.
    assert task_score([100.0, 49.0]) > task_score([100.0, 112.0])


def test_vector_score_matches_scalar():
    rng = np.random.default_rng(0)
    points = rng.uniform(20, 204, size=(200, 2))
    expected = [task_score(p) for p in points]
    assert task_scores(points) == pytest.approx(expected)


def test_env_applies_rotation_from_current_position():
    env = ShiftedTwoRoom("grid")
    env.reset_task(seed=1, start_mode="fixed")
    env.agent_position = torch.tensor([44.0, 20.0 + 96.0], dtype=torch.float32)
    alpha = rotation_angle("grid", 44.0, 116.0)
    assert math.degrees(alpha) == pytest.approx(75.0)
    start = env.agent_position.clone().numpy()
    env.step(np.array([0.5, 0.0], dtype=np.float32))
    moved = env.agent_position.numpy() - start
    assert moved == pytest.approx(
        [5.0 * 0.5 * math.cos(alpha), 5.0 * 0.5 * math.sin(alpha)], abs=1e-4
    )


def test_env_without_shift_matches_base_dynamics():
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=1, start_mode="fixed")
    start = env.agent_position.clone().numpy()
    env.step(np.array([0.4, -0.2], dtype=np.float32))
    assert (env.agent_position.numpy() - start) == pytest.approx([2.0, -1.0], abs=1e-5)


def test_env_clips_command_before_rotation():
    env = ShiftedTwoRoom("grid")
    env.reset_task(seed=1, start_mode="fixed")
    env.agent_position = torch.tensor([44.0, 116.0], dtype=torch.float32)
    start = env.agent_position.clone().numpy()
    env.step(np.array([9.0, 0.0], dtype=np.float32))
    alpha = rotation_angle("grid", 44.0, 116.0)
    assert (env.agent_position.numpy() - start) == pytest.approx(
        [5.0 * math.cos(alpha), 5.0 * math.sin(alpha)], abs=1e-4
    )


def test_reset_task_keeps_task_fixed_and_start_seeded():
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=5)
    first = env.agent_position.clone()
    assert env.target_position.tolist() == GOAL.tolist()
    assert LEFT_ROOM_X[0] <= float(first[0]) <= LEFT_ROOM_X[1]
    assert LEFT_ROOM_Y[0] <= float(first[1]) <= LEFT_ROOM_Y[1]
    assert env.door_positions.tolist() == [49.0]
    assert env.door_sizes.tolist() == [14.0]
    assert float(env.variation_space["agent"]["speed"].value.item()) == 5.0
    env.reset_task(seed=5)
    assert env.agent_position.tolist() == first.tolist()
    env.reset_task(seed=6)
    assert env.agent_position.tolist() != first.tolist()


def test_goal_is_not_rendered_in_frames():
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=2, start_mode="fixed")
    frame = env.render()
    assert frame.shape == (224, 224, 3) and frame.dtype == np.uint8
    # The goal disc is green; with render_target off no pure green pixels exist.
    green = (frame[..., 1] > 200) & (frame[..., 0] < 50) & (frame[..., 2] < 50)
    assert not green.any()


def test_success_matches_env_termination():
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=1, start_mode="fixed")
    env.agent_position = torch.tensor([164.0, 100.0], dtype=torch.float32)
    _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert terminated and env.reached_goal()
    env.agent_position = torch.tensor([164.0, 90.0], dtype=torch.float32)
    _, _, terminated, _, _ = env.step(np.zeros(2, dtype=np.float32))
    assert not terminated and not env.reached_goal()


def test_door_path_is_reachable_without_shift():
    """A straight route through the doorway must reach the goal in 50 steps."""
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=1, start_mode="fixed")
    for _ in range(50):
        position = env.agent_position.numpy()
        waypoint = DOORWAY if position[0] < 112.0 else GOAL
        direction = waypoint - position
        env.step(direction / np.linalg.norm(direction))
        if env.reached_goal():
            break
    assert env.reached_goal()


# ------------------------------------------------------------------- dense windows


def test_rows_per_window_follows_appendix_a12():
    assert ROWS_PER_WINDOW == 20


def test_dense_windows_count_and_content():
    stream = DenseWindowStream()
    windows = []
    for row in range(50):
        latent = torch.full((4,), float(row))
        action = torch.tensor([row + 0.25, row + 0.75])
        window = stream.add_row(latent, action)
        assert (window is None) == (row < 19)
        if window is not None:
            windows.append(window)
    assert len(windows) == 31
    final = stream.add_observation(torch.full((4,), 50.0))
    assert final is not None  # the final observation completes start t = 31
    assert final[0][:, 0].tolist() == [31.0, 36.0, 41.0, 46.0]
    assert final[1].shape == (3, 10)
    assert final[1][-1, -1].item() == 45.75
    z, b = windows[7]  # start t = 7
    assert z[:, 0].tolist() == [7.0, 12.0, 17.0, 22.0]
    assert z.shape == (4, 4) and b.shape == (3, 10)
    expected = [
        [r + offset for r in range(7 + 5 * j, 12 + 5 * j) for offset in (0.25, 0.75)]
        for j in range(3)
    ]
    assert b.tolist() == expected


def test_buffer_samples_without_replacement_and_clears():
    buffer = WindowBuffer()
    for i in range(64):
        buffer.add((torch.full((4, 2), float(i)), torch.zeros(3, 10)))
    z, b = buffer.sample(64, np.random.default_rng(0))
    assert z.shape == (64, 4, 2) and b.shape == (64, 3, 10)
    assert sorted(z[:, 0, 0].tolist()) == [float(i) for i in range(64)]
    with pytest.raises(ValueError):
        buffer.sample(65, np.random.default_rng(0))
    buffer.clear()
    assert len(buffer) == 0


def push_episode(trainer, latent_dim=16, rows=50):
    """Feed one episode of random rows through ``add_block``."""
    trainer.start_episode()
    z = torch.randn(1, rows + 1, latent_dim)
    a = torch.randn(1, rows, 2)
    for t in range(0, rows, 5):
        trainer.add_block(z, a, t)


def test_replay_schedule_is_floor_of_windows_over_batch():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["persistent_replay"], seed=0)
    for episode in range(1, 7):
        push_episode(trainer)
        assert trainer.windows_seen == 32 * episode
        assert trainer.total_updates == (32 * episode) // 64
        assert len(trainer.buffer) == 32 * episode


def test_extension_does_four_updates_per_scheduled_batch():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["persistent_replay_x4"], seed=0)
    for episode in range(1, 7):
        push_episode(trainer)
        assert trainer.total_updates == 4 * ((32 * episode) // 64)
        assert len(trainer.buffer) == 32 * episode


def test_sweep_plan_and_finished_detection(tmp_path):
    frozen, adaptive = plan_runs(["grid"], [0, 1, 2])
    assert frozen == [("frozen", "none", 0), ("frozen", "grid", 0)]
    assert len(adaptive) == 4 * 3
    assert ("persistent_replay", "grid", 2) in adaptive
    path = tmp_path / "run.jsonl"
    assert not is_finished(path)
    path.write_text('{"kind": "eval"}\n')
    assert not is_finished(path)
    path.write_text('{"kind": "eval"}\n{"kind": "done"}\n')
    assert is_finished(path)
    path.write_text('{"kind": "eval"}\n{"kind": "do')
    assert not is_finished(path)


def test_run_jobs_keeps_gpus_evenly_loaded(tmp_path, monkeypatch):
    lock = threading.Lock()
    running = {0: 0, 1: 0}
    peak = {0: 0, 1: 0}
    seen = []

    def fake_run(command, stdout, stderr, env):
        gpu = int(env["CUDA_VISIBLE_DEVICES"])
        with lock:
            running[gpu] += 1
            peak[gpu] = max(peak[gpu], running[gpu])
            seen.append(gpu)
        time.sleep(0.02)
        with lock:
            running[gpu] -= 1
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    args = SimpleNamespace(
        workdir=tmp_path,
        episodes=1,
        eval_every=1,
        eval_episodes=1,
        eval_batch=1,
        trajectories=1,
        num_samples=4,
        iterations=1,
        elites=2,
        device="cuda",
        tag="",
        allow_cpu=False,
        jobs=4,
        gpus=2,
        eval_dense_until=0,
        eval_dense_every=1,
        reversal_episodes=0,
    )
    jobs = [("persistent_replay", "grid", seed) for seed in range(8)]
    outcomes = run_jobs(jobs, args, tmp_path, tmp_path)
    assert [code for _, code in outcomes] == [0] * 8
    assert sorted(set(seen)) == [0, 1]
    assert peak[0] <= 2 and peak[1] <= 2


def test_paired_auc_gains_match_seeds():
    def run(seed, values):
        return {
            "seed": seed,
            "evals": [{"episode": e, "score_final": v} for e, v in zip((0, 50, 100), values)],
        }

    gains = paired_auc_gains(
        [run(0, (0.2, 0.2, 0.2)), run(1, (0.5, 0.5, 0.5))],
        [run(0, (0.3, 0.3, 0.3)), run(2, (0.9, 0.9, 0.9))],
        "score_final",
    )
    assert list(gains) == [0]
    assert gains[0] == pytest.approx(0.1)


def test_local_schedule_counts_per_episode_and_drops_leftovers():
    model = small_model()
    for arm in ("persistent_local", "reset_local"):
        trainer = TestTimeTrainer(model, ARMS[arm], seed=0)
        for episode in range(1, 4):
            push_episode(trainer)
            assert len(trainer.buffer) == 32
            assert trainer.total_updates == 4 * episode


def test_at_most_one_update_per_call():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["persistent_local"], seed=0)
    trainer.start_episode()
    stream_z = torch.randn(1, 51, 16)
    stream_a = torch.randn(1, 50, 2)
    for row in range(50):
        window = trainer.stream.add_row(stream_z[0, row], stream_a[0, row])
        if window is not None:
            trainer.buffer.add(window)
            trainer.windows_seen += 1
    assert trainer.maybe_update() is True
    assert trainer.total_updates == 1
    assert trainer.maybe_update() is True
    assert trainer.total_updates == 2


# ------------------------------------------------------------------------ planning


@pytest.mark.parametrize("context", [1, 2, 3])
def test_rollout_matches_upstream_loop(context):
    model = small_model()
    torch.manual_seed(context)
    z = torch.randn(4, context, 16)
    b = torch.randn(4, context - 1, 10)
    blocks = torch.randn(4, 5, 10)
    with torch.no_grad():
        ours = rollout_latents(model, z, b, blocks)
        theirs = reference_rollout(model, z, b, blocks)
    assert ours.shape == (4, 5, 16)
    assert torch.allclose(ours, theirs)
    with torch.no_grad():
        different = rollout_latents(model, z, b, torch.randn(4, 5, 10))
    assert not torch.allclose(ours, different)


def test_rollout_rejects_misaligned_context():
    model = small_model()
    with pytest.raises(ValueError):
        rollout_latents(model, torch.zeros(1, 3, 16), torch.zeros(1, 3, 10), torch.zeros(1, 5, 10))


def test_score_candidates_applies_eq3_discounting():
    class Model:
        action_encoder = staticmethod(lambda act: act)

        @staticmethod
        def predict(emb, act_emb):
            return emb + 1.0

    class Head(nn.Module):
        def forward(self, z):
            return z.mean(dim=-1, keepdim=True)

    z = torch.zeros(1, 1, 4)
    b = torch.zeros(1, 0, 10)
    candidates = torch.zeros(1, 2, 3, 10)
    scores = score_candidates(Model, Head(), z, b, candidates, discount=0.5)
    # Latents 1, 2, 3 weighted by 0.5, 0.25, 0.125.
    assert scores.shape == (1, 2)
    assert scores[0].tolist() == pytest.approx([1.375, 1.375])


def test_score_candidates_keeps_batches_aligned():
    """Each environment's candidates must be scored from its own context."""
    model, head = small_model(), small_head()
    torch.manual_seed(3)
    z = torch.randn(3, 2, 16)
    b = torch.randn(3, 1, 10)
    candidates = torch.randn(3, 4, 5, 10)
    with torch.no_grad():
        scores = score_candidates(model, head, z, b, candidates, 0.99)
        for env in range(3):
            single = score_candidates(
                model, head, z[env : env + 1], b[env : env + 1], candidates[env : env + 1], 0.99
            )
            assert torch.allclose(scores[env], single[0], atol=1e-6)


def test_cem_moves_toward_higher_reward_and_is_seeded():
    class Model:
        action_encoder = staticmethod(lambda act: act)

        @staticmethod
        def predict(emb, act_emb):
            return emb + act_emb.mean(dim=-1, keepdim=True)

    class Head(nn.Module):
        def forward(self, z):
            return -(z[..., :1] - 2.0).abs()

    z = torch.zeros(2, 1, 4)
    b = torch.zeros(2, 0, 10)
    config = PlannerConfig(num_samples=512, iterations=8, elites=32, horizon=3)
    first = cem_plan(Model, Head(), z, b, config, torch.Generator().manual_seed(0))
    again = cem_plan(Model, Head(), z, b, config, torch.Generator().manual_seed(0))
    assert first.shape == (2, 10)
    assert torch.equal(first, again)
    assert first.mean().item() > 0.3


def test_cem_defaults_match_table_a1():
    config = PlannerConfig()
    assert (config.num_samples, config.iterations, config.elites) == (256, 6, 32)
    assert (config.horizon, config.init_std, config.std_floor) == (5, 1.0, 0.05)
    assert config.discount == 0.99


def test_cem_std_floor_is_applied():
    seen = []

    class Model:
        @staticmethod
        def action_encoder(act):
            seen.append(act.clone())
            return act

        @staticmethod
        def predict(emb, act_emb):
            return emb

    class Head(nn.Module):
        def forward(self, z):
            return z[..., :1]

    # Elites from a tiny first spread fall below the floor, so the floor sets the second.
    config = PlannerConfig(num_samples=64, iterations=2, elites=8, horizon=1, init_std=1e-6)
    cem_plan(
        Model,
        Head(),
        torch.zeros(1, 1, 4),
        torch.zeros(1, 0, 10),
        config,
        torch.Generator().manual_seed(0),
    )
    assert len(seen) == 2
    assert seen[0].std().item() < 1e-5
    assert seen[1].std().item() == pytest.approx(0.05, rel=0.2)


def test_keyframe_context_alignment():
    z = torch.arange(51, dtype=torch.float32).reshape(1, 51, 1)
    a = torch.stack([torch.arange(50.0), -torch.arange(50.0)], dim=-1).unsqueeze(0)
    z0, b0 = keyframe_context(z, a, 0)
    assert z0[0, :, 0].tolist() == [0.0] and b0.shape == (1, 0, 10)
    z5, b5 = keyframe_context(z, a, 5)
    assert z5[0, :, 0].tolist() == [0.0, 5.0]
    assert b5[0, 0].tolist() == [0.0, 0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    z20, b20 = keyframe_context(z, a, 20)
    assert z20[0, :, 0].tolist() == [10.0, 15.0, 20.0]
    assert b20.shape == (1, 2, 10)
    assert b20[0, 0, 0].item() == 10.0 and b20[0, 1, 0].item() == 15.0
    with pytest.raises(ValueError):
        keyframe_context(z, a, 3)


# ----------------------------------------------------------------------- adaptation


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def test_trainable_set_matches_table_a1():
    model = small_model()
    params = set_trainable(model)
    assert params
    for name in ("encoder", "projector"):
        assert not any(p.requires_grad for p in getattr(model, name).parameters())
    for name in ("predictor", "action_encoder", "pred_proj"):
        assert all(p.requires_grad for p in getattr(model, name).parameters())


def test_update_changes_only_trainable_weights_and_not_batchnorm():
    model = small_model()
    head = small_head()
    frozen = {n: snapshot(getattr(model, n)) for n in ("encoder", "projector")}
    bn = snapshot(model.pred_proj.net[1])
    head_state = snapshot(head)
    before = snapshot(model.predictor)
    trainer = TestTimeTrainer(model, ARMS["persistent_local"], seed=0)
    push_episode(trainer)
    assert trainer.total_updates == 4
    assert not same(before, snapshot(model.predictor))
    for name, state in frozen.items():
        assert same(state, snapshot(getattr(model, name)))
    after_bn = snapshot(model.pred_proj.net[1])
    for key in ("running_mean", "running_var", "num_batches_tracked"):
        assert torch.equal(bn[key], after_bn[key])
    assert same(head_state, snapshot(head))
    assert not model.training


def test_updates_run_in_eval_mode_even_if_model_was_training():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["persistent_local"], seed=0)
    modes = []
    dropout = model.predictor.transformer.layers[0].mlp.net[3]
    dropout.register_forward_hook(lambda module, _, __: modes.append(module.training))
    trainer.start_episode()
    z, a = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
    model.train()
    for t in range(0, 50, 5):
        trainer.add_block(z, a, t)
    assert trainer.total_updates == 4
    assert modes and not any(modes)


def test_reset_arm_restores_weights_and_optimizer():
    model = small_model()
    start = snapshot(model.predictor)
    trainer = TestTimeTrainer(model, ARMS["reset_local"], seed=0)
    push_episode(trainer)
    assert not same(start, snapshot(model.predictor))
    trainer.start_episode()
    assert same(start, snapshot(model.predictor))
    assert len(trainer.optimizer.state) == 0
    assert len(trainer.buffer) == 0


def test_persistent_arms_keep_weights_and_optimizer():
    model = small_model()
    for arm in ("persistent_local", "persistent_replay"):
        trainer = TestTimeTrainer(model, ARMS[arm], seed=0)
        push_episode(trainer)
        push_episode(trainer)
        push_episode(trainer)
        state = snapshot(model.predictor)
        optimizer = trainer.optimizer
        assert len(optimizer.state) > 0
        trainer.start_episode()
        assert same(state, snapshot(model.predictor))
        assert trainer.optimizer is optimizer


def test_replay_keeps_buffer_across_episodes():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["persistent_replay"], seed=0)
    push_episode(trainer)
    push_episode(trainer)
    assert len(trainer.buffer) == 64


def test_windows_never_span_episodes():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["persistent_replay"], seed=0)
    trainer.start_episode()
    z = torch.zeros(1, 16, 16)
    a = torch.zeros(1, 15, 2)
    for t in range(0, 15, 5):
        trainer.add_block(z, a, t)
    assert trainer.windows_seen == 0
    trainer.start_episode()
    for t in range(0, 15, 5):
        trainer.add_block(z, a, t)
    assert trainer.windows_seen == 0


def test_prediction_loss_matches_eq4_and_upstream_shape():
    class Model:
        action_encoder = staticmethod(lambda b: b[..., :2])

        @staticmethod
        def predict(z, act_emb):
            return z + act_emb

    z = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [4.0, 3.0]]])
    b = torch.ones(1, 3, 10)
    # Predictions 1,1 / 2,2 / 3,3 against 1,1 / 2,2 / 4,3: one squared error of 1.
    assert prediction_loss(Model, z, b).item() == pytest.approx(1.0 / 6.0)


def test_updates_reduce_prediction_error_on_a_learnable_target():
    model = small_model()
    params = set_trainable(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    torch.manual_seed(0)
    # Each next latent is the previous one plus a fixed map of the action block.
    action_map = torch.randn(10, 16) * 0.3
    b = torch.randn(64, 3, 10)
    z = [torch.randn(64, 16) * 0.2]
    for j in range(3):
        z.append(z[-1] + b[:, j] @ action_map)
    z = torch.stack(z, dim=1)
    first = prediction_loss(model, z, b).item()
    for _ in range(200):
        loss = prediction_loss(model, z, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert prediction_loss(model, z, b).item() < 0.5 * first


# ----------------------------------------------------------------------- experiment


def test_run_episodes_records_all_steps_and_executed_actions():
    model, head = small_model(), small_head()
    env = ShiftedTwoRoom("grid")
    env.reset_task(seed=3)
    blocks = []
    batch = run_episodes(
        model,
        head,
        [env],
        STATS,
        TINY_PLANNER,
        torch.Generator().manual_seed(0),
        horizon=15,
        record_all_steps=True,
        on_block=lambda z, a, t: blocks.append(t),
    )
    assert blocks == [0, 5, 10]
    assert batch.replans == 3
    assert batch.z_history.shape == (1, 16, 16)
    assert batch.a_history.shape == (1, 15, 2)
    assert (batch.z_history[0].abs().sum(dim=1) > 0).all()
    raw = STATS.denormalize(batch.a_history[0].numpy())
    assert (np.abs(raw) <= 1.0 + 1e-5).all()


def test_recorded_latents_match_replayed_frames():
    """z_history[t] must be the encoding of the frame after exactly t actions."""
    model, head = small_model(), small_head()
    env = ShiftedTwoRoom("wave")
    env.reset_task(seed=4)
    batch = run_episodes(
        model,
        head,
        [env],
        STATS,
        TINY_PLANNER,
        torch.Generator().manual_seed(1),
        horizon=10,
        record_all_steps=True,
    )
    replay = ShiftedTwoRoom("wave")
    replay.reset_task(seed=4)
    actions = STATS.denormalize(batch.a_history[0].numpy()).astype(np.float32)
    frames = [replay.render()]
    for action in actions:
        replay.step(action)
        frames.append(replay.render())
    expected = model.encode_frames(np.stack(frames))
    assert torch.allclose(batch.z_history[0], expected, atol=1e-3)


def test_stop_on_success_freezes_a_finished_env():
    model, head = small_model(), small_head()
    near, far = ShiftedTwoRoom("none"), ShiftedTwoRoom("none")
    near.reset_task(seed=1, start_mode="fixed")
    near.agent_position = torch.as_tensor(GOAL.copy())  # one step cannot leave the radius
    far.reset_task(seed=2, start_mode="fixed")
    steps = {id(near): 0, id(far): 0}
    for env in (near, far):
        original = env.step

        def counted(action, env=env, original=original):
            steps[id(env)] += 1
            return original(action)

        env.step = counted
    batch = run_episodes(
        model,
        head,
        [near, far],
        STATS,
        TINY_PLANNER,
        torch.Generator().manual_seed(0),
        horizon=10,
        stop_on_success=True,
    )
    assert batch.successes[0]
    assert steps[id(near)] == 1
    assert steps[id(far)] == 10
    assert batch.final_scores[0] == pytest.approx(near.score())
    assert batch.peak_scores[1] >= batch.final_scores[1]
    assert batch.replans == 2


def test_run_episodes_rejects_invalid_modes():
    model, head = small_model(), small_head()
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=1)
    with pytest.raises(ValueError):
        run_episodes(
            model,
            head,
            [env],
            STATS,
            TINY_PLANNER,
            torch.Generator(),
            stop_on_success=True,
            record_all_steps=True,
        )
    with pytest.raises(ValueError):
        run_episodes(model, head, [env], STATS, TINY_PLANNER, torch.Generator(), horizon=12)


def test_evaluate_is_deterministic_and_does_not_change_model():
    model, head = small_model(), small_head()
    state = snapshot(model)
    kwargs = dict(episodes=3, batch_size=2, start_mode="uniform_left")
    first = evaluate(model, head, STATS, "grid", TINY_PLANNER, **kwargs)
    second = evaluate(model, head, STATS, "grid", TINY_PLANNER, **kwargs)
    assert first == second
    assert same(state, snapshot(model))
    assert set(first) == {"score_final", "score_peak", "success_rate"}
    assert first["score_peak"] >= first["score_final"]


def test_evaluate_uses_the_same_starts_regardless_of_batch_size(monkeypatch):
    model, head = small_model(), small_head()
    original = ShiftedTwoRoom.reset_task
    seeds = []

    def recording(self, seed, start_mode="uniform_left"):
        seeds.append(seed)
        return original(self, seed, start_mode)

    monkeypatch.setattr(ShiftedTwoRoom, "reset_task", recording)
    evaluate(model, head, STATS, "grid", TINY_PLANNER, 3, 2, "uniform_left")
    by_two = list(seeds)
    seeds.clear()
    evaluate(model, head, STATS, "grid", TINY_PLANNER, 3, 3, "uniform_left")
    assert by_two == seeds == [900_000_000, 900_000_001, 900_000_002]


def test_deployment_seeds_are_disjoint_from_held_out_suites():
    seeds = {deployment_seed(run, ep) for run in range(3) for ep in range(1, 501)}
    assert len(seeds) == 1500
    assert max(seeds) < 800_000_000
    with pytest.raises(ValueError):
        deployment_seed(800, 0)


def test_fixed_trajectory_mse_counts_windows_and_horizons():
    calls = []

    class Model:
        @staticmethod
        def parameters():
            return iter([torch.zeros(1)])

        @staticmethod
        def action_encoder(act):
            return act

        @staticmethod
        def predict(emb, act_emb):
            calls.append(emb.size(1))
            # Perfect prediction for a latent that grows by 5 per keyframe.
            next_value = emb[:, -1:] + 5.0
            return next_value.expand(emb.size(0), emb.size(1), emb.size(2))

    z = torch.arange(51, dtype=torch.float32).reshape(1, 51, 1).repeat(2, 1, 3)
    a = torch.zeros(2, 50, 2)
    mse = fixed_trajectory_mse(Model, {"z": z, "a": a})
    assert mse == pytest.approx(0.0)
    assert len(calls) == 16 * 5  # starts 0..15 and five future blocks each


def test_fixed_trajectory_mse_measures_error():
    class Model:
        @staticmethod
        def parameters():
            return iter([torch.zeros(1)])

        @staticmethod
        def action_encoder(act):
            return act

        @staticmethod
        def predict(emb, act_emb):
            return emb[:, -1:].expand(emb.size(0), emb.size(1), emb.size(2)) + 6.0

    z = torch.arange(51, dtype=torch.float32).reshape(1, 51, 1)
    mse = fixed_trajectory_mse(Model, {"z": z, "a": torch.zeros(1, 50, 2)})
    # Horizon h has error h squared: (1 + 4 + 9 + 16 + 25) / 5.
    assert mse == pytest.approx(11.0)


@pytest.mark.parametrize("arm", ["frozen", "persistent_replay", "persistent_local", "reset_local"])
def test_deployment_logs_expected_schedule(arm):
    model, head = small_model(), small_head()
    trajectories = collect_fixed_trajectories(
        model, head, STATS, "grid", TINY_PLANNER, 2, 2, "uniform_left"
    )
    assert trajectories["z"].shape == (2, 51, 16)
    assert trajectories["a"].shape == (2, 50, 2)
    rows = []
    start = snapshot(model.predictor)
    run_deployment(
        model,
        head,
        STATS,
        arm,
        "grid",
        run_seed=0,
        planner=TINY_PLANNER,
        episodes=3,
        eval_every=3,
        eval_episodes=2,
        eval_batch=2,
        start_mode="uniform_left",
        trajectories=trajectories,
        log=rows.append,
    )
    if arm == "frozen":
        assert [r["episode"] for r in rows] == [0]
        assert same(start, snapshot(model.predictor))
        return
    assert [r["episode"] for r in rows] == [0, 3]
    assert rows[0]["updates"] == 0
    expected = {"persistent_replay": 96 // 64, "persistent_local": 12, "reset_local": 12}[arm]
    assert rows[-1]["updates"] == expected
    assert not same(start, snapshot(model.predictor))


def test_action_stats_round_trip():
    actions = np.array([[0.5, -0.5], [1.0, 1.0]], dtype=np.float32)
    assert STATS.denormalize(STATS.normalize(actions)) == pytest.approx(actions, abs=1e-6)


# --------------------------------------------------------------------------- report


def test_auc_is_normalized_trapezoid():
    episodes = np.array([0, 50, 100])
    assert run_auc(episodes, np.array([0.0, 1.0, 1.0])) == pytest.approx(0.75)
    assert run_auc(np.array([0]), np.array([0.4])) == pytest.approx(0.4)


def test_summary_best_excludes_episode_zero_and_uses_mean_curve():
    def run(values):
        return {
            "evals": [
                {
                    "episode": e,
                    "score_final": v,
                    "success_rate": 1.0,
                    "latent_mse": 0.5,
                    "updates": 2,
                }
                for e, v in zip((0, 50, 100), values)
            ]
        }

    summary = summarize([run((0.9, 0.2, 0.6)), run((0.9, 0.4, 0.2))], "score_final")
    assert summary["best"] == pytest.approx(0.4)
    assert summary["runs"] == 2


# -------------------------------------------------------------------------- prepare


def test_preprocess_matches_upstream_normalization():
    frame = np.full((1, 224, 224, 3), 255, dtype=np.uint8)
    out = preprocess_frames(frame, "cpu")
    assert out.shape == (1, 3, 224, 224)
    assert out[0, :, 0, 0].tolist() == pytest.approx(
        [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225], rel=1e-6
    )
    with pytest.raises(ValueError):
        preprocess_frames(np.zeros((1, 64, 64, 3), dtype=np.uint8), "cpu")
    with pytest.raises(ValueError):
        preprocess_frames(np.zeros((1, 224, 224, 3), dtype=np.float32), "cpu")


def make_h5(path: Path, env_frames=True) -> None:
    rng = np.random.default_rng(0)
    env = ShiftedTwoRoom("none")
    env.reset_task(seed=0, start_mode="fixed")
    lengths = [6, 7, 5]
    total = sum(lengths)
    positions = np.stack([rng.uniform(25, 95, total), rng.uniform(25, 195, total)], axis=1).astype(
        np.float32
    )
    pixels = np.zeros((total, 224, 224, 3), dtype=np.uint8)
    for row, position in enumerate(positions):
        env.agent_position = torch.as_tensor(position)
        pixels[row] = env.render() if env_frames else 0
    actions = rng.normal(size=(total, 2)).astype(np.float32)
    actions[np.cumsum(lengths) - 1] = np.nan
    with h5py.File(path, "w") as h5:
        h5["pixels"] = pixels
        h5["proprio"] = positions
        h5["action"] = actions
        h5["ep_len"] = np.array(lengths)
        h5["ep_offset"] = np.array([0, 6, 13])


def test_action_stats_skip_nan_and_use_sample_std(tmp_path):
    path = tmp_path / "d.h5"
    make_h5(path)
    with h5py.File(path, "r") as h5:
        stats = compute_action_stats(h5)
        raw = h5["action"][:]
    clean = raw[~np.isnan(raw).any(axis=1)]
    assert len(clean) == 15
    assert stats["mean"] == pytest.approx(clean.mean(axis=0).tolist())
    assert stats["std"] == pytest.approx(clean.std(axis=0, ddof=1).tolist())


def test_render_validation_separates_match_from_mismatch(tmp_path):
    good, bad = tmp_path / "good.h5", tmp_path / "bad.h5"
    make_h5(good)
    make_h5(bad, env_frames=False)
    with h5py.File(good, "r") as h5:
        assert validate_render_match(h5, samples=5) == pytest.approx(0.0)
    with h5py.File(bad, "r") as h5:
        assert validate_render_match(h5, samples=5) > 100.0


def make_dynamics_h5(path: Path, shift_actions: bool) -> None:
    """Rows where a[t] moves p[t] to p[t + 1], or a[t + 1] does if shifted."""
    rng = np.random.default_rng(0)
    lengths = [40, 40]
    positions, actions = [], []
    for length in lengths:
        steps = rng.uniform(-1, 1, size=(length, 2)).astype(np.float32)
        steps[-1] = np.nan
        moved = np.nan_to_num(steps) * 5.0
        track = np.concatenate([[[60.0, 100.0]], np.cumsum(moved[:-1], axis=0) + [60.0, 100.0]])
        positions.append(track.astype(np.float32))
        if shift_actions:
            steps = np.concatenate([np.full((1, 2), np.nan, dtype=np.float32), steps[:-1]])
        actions.append(steps)
    with h5py.File(path, "w") as h5:
        h5["proprio"] = np.concatenate(positions)
        h5["action"] = np.concatenate(actions)
        h5["ep_len"] = np.array(lengths)
        h5["ep_offset"] = np.array([0, 40])


def test_validate_dynamics_measures_speed_and_alignment(tmp_path):
    aligned, shifted = tmp_path / "a.h5", tmp_path / "s.h5"
    make_dynamics_h5(aligned, shift_actions=False)
    make_dynamics_h5(shifted, shift_actions=True)
    with h5py.File(aligned, "r") as h5:
        report = validate_dynamics(h5)
    assert report["speed"] == pytest.approx(5.0, abs=1e-3)
    assert report["aligned_error"] < 1e-3 < min(report["previous_error"], report["next_error"])
    with h5py.File(shifted, "r") as h5:
        report = validate_dynamics(h5)
    assert report["aligned_error"] > report["next_error"]


def test_encode_episodes_keeps_episode_ownership(tmp_path):
    path = tmp_path / "d.h5"
    make_h5(path)
    model = small_model()
    with h5py.File(path, "r") as h5:
        latents, positions, owners = encode_episodes(model, h5, np.array([2, 0]), batch_size=4)
        stored = h5["proprio"][:]
    assert latents.shape == (11, 16)
    assert owners.tolist() == [0] * 6 + [2] * 5
    assert positions.tolist() == np.concatenate([stored[:6], stored[13:18]]).tolist()


def test_reward_head_fits_and_restores_best_state():
    torch.manual_seed(0)
    x = torch.randn(600, 8)
    y = torch.sigmoid(x[:, 0] * 2.0)
    mask = torch.arange(600) < 500
    head, metrics = fit_reward_head(x, y, mask, torch.device("cpu"), max_epochs=60, patience=15)
    with torch.no_grad():
        val = torch.nn.functional.mse_loss(head(x[~mask]).squeeze(-1), y[~mask]).item()
    assert val == pytest.approx(metrics["val_mse"], rel=1e-6)
    assert val < 0.02
    assert not any(p.requires_grad for p in head.parameters())


def test_extract_dataset_round_trip_and_reuse(tmp_path):
    payload = b"content"
    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        info = tarfile.TarInfo("inner/tworoom.h5")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    archive = tmp_path / "a.tar.zst"
    archive.write_bytes(zstandard.ZstdCompressor().compress(tar_bytes.getvalue()))
    target = tmp_path / "out"
    extracted = extract_dataset(archive, target)
    assert extracted.read_bytes() == payload
    archive.unlink()
    assert extract_dataset(archive, target) == extracted


def test_cpu_requires_explicit_permission():
    with pytest.raises(SystemExit):
        require_device("cpu", allow_cpu=False)
    assert require_device("cpu", allow_cpu=True).type == "cpu"


# -------------------------------------------------------- published artifacts


def test_build_model_ignores_hydra_targets():
    model = build_model(CONFIG)
    assert model.predictor.pos_embedding.shape == (1, 3, 192)
    assert sum(p.numel() for p in model.parameters()) > 10_000_000


@pytest.mark.skipif("JEPA_TTT_WEIGHTS" not in os.environ, reason="checkpoint not supplied")
def test_published_checkpoint_loads_strictly_and_plans():
    model = load_pretrained(os.environ["JEPA_TTT_WEIGHTS"], os.environ["JEPA_TTT_CONFIG"], "cpu")
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    env = ShiftedTwoRoom("grid")
    env.reset_task(seed=0, start_mode="fixed")
    z = model.encode_frames(env.render()[None])
    assert z.shape == (1, 192)
    assert torch.isfinite(z).all()
    head = make_reward_head()
    head.eval()
    block = cem_plan(
        model,
        head,
        z.unsqueeze(1),
        z.new_zeros(1, 0, 10),
        PlannerConfig(num_samples=4, iterations=1, elites=2),
        torch.Generator().manual_seed(0),
    )
    assert block.shape == (1, 10) and torch.isfinite(block).all()

    # One real gradient update must leave the frozen modules untouched.
    encoder_before = snapshot(model.encoder)
    projector_before = snapshot(model.projector)
    predictor_before = snapshot(model.predictor)
    trainer = TestTimeTrainer(model, ARMS["persistent_local"], seed=0)
    trainer.start_episode()
    z_history = z.repeat(1, 51, 1) + 0.01 * torch.randn(1, 51, 192)
    a_history = torch.randn(1, 50, 2)
    for t in range(0, 30, 5):
        trainer.add_block(z_history, a_history, t)
    assert trainer.total_updates >= 1
    assert math.isfinite(trainer.last_loss)
    assert same(encoder_before, snapshot(model.encoder))
    assert same(projector_before, snapshot(model.projector))
    assert not same(predictor_before, snapshot(model.predictor))


@pytest.mark.skipif(
    "JEPA_TTT_UPSTREAM_MODULE" not in os.environ, reason="upstream file not supplied"
)
def test_vendored_classes_equal_upstream():
    def classes(source):
        tree = ast.parse(source)
        return {
            node.name: ast.dump(node)
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }

    ours = classes(Path(__file__).parents[1].joinpath("jepa_ttt/lewm.py").read_text())
    theirs = classes(Path(os.environ["JEPA_TTT_UPSTREAM_MODULE"]).read_text())
    for name in (
        "modulate",
        "FeedForward",
        "Attention",
        "ConditionalBlock",
        "Block",
        "Transformer",
        "Embedder",
        "MLP",
        "ARPredictor",
    ):
        assert ours[name] == theirs[name], name
