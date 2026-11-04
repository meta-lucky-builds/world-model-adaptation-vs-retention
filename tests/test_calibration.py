"""The mathematical properties of action calibration, on small models."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import torch
from torch import nn

from jepa_ttt.adaptation import ARMS, TestTimeTrainer, prediction_loss
from jepa_ttt.calibration import ActionCalibrator
from jepa_ttt.envs import ShiftedTwoRoom, rotate, rotation_angle
from jepa_ttt.experiment import (
    ActionStats,
    collect_fixed_trajectories,
    fixed_trajectory_mse,
    run_deployment,
    run_episodes,
)
from jepa_ttt.planning import PlannerConfig, cem_plan, rollout_latents, score_candidates
from jepa_ttt.report_calibration import (
    episodes_to_reach,
    judge_shift,
    premise_holds,
    recovers_faster,
    reversal_score,
)
from jepa_ttt.sweep import check_baselines
from tests.test_stage1 import same, small_head, small_model, snapshot

STATS = ActionStats(
    mean=np.array([0.1, -0.2], dtype=np.float32),
    std=np.array([0.8, 0.9], dtype=np.float32),
)
TINY_PLANNER = PlannerConfig(num_samples=6, iterations=2, elites=2)


def make(latent_dim: int = 16, randomize: bool = False) -> ActionCalibrator:
    torch.manual_seed(0)
    calibrator = ActionCalibrator(STATS.mean, STATS.std, latent_dim, hidden_dim=32)
    if randomize:
        nn.init.normal_(calibrator.net[-1].weight, std=0.1)
    return calibrator


def to_normalized(raw: torch.Tensor) -> torch.Tensor:
    return (raw - torch.as_tensor(STATS.mean)) / torch.as_tensor(STATS.std)


def to_raw(blocks: torch.Tensor) -> torch.Tensor:
    return blocks * torch.as_tensor(STATS.std) + torch.as_tensor(STATS.mean)


def test_starts_as_exact_identity_including_out_of_range():
    calibrator = make()
    z = torch.randn(7, 3, 16)
    blocks = torch.randn(7, 3, 10) * 5.0  # many are far outside the action range
    assert torch.equal(calibrator(z, blocks), blocks)


def test_shapes_and_empty_context():
    calibrator = make(randomize=True)
    assert calibrator(torch.randn(4, 3, 16), torch.randn(4, 3, 10)).shape == (4, 3, 10)
    assert calibrator(torch.randn(4, 1, 16), torch.randn(4, 1, 10)).shape == (4, 1, 10)
    empty = torch.zeros(4, 0, 10)
    assert calibrator(torch.zeros(4, 0, 16), empty).shape == (4, 0, 10)


def test_each_position_sees_only_earlier_commands():
    calibrator = make(randomize=True)
    z = torch.randn(1, 16)
    commands = torch.rand(1, 5, 2) * 0.5
    base = calibrator.deltas(z, commands)
    for position in range(5):
        changed = commands.clone()
        changed[0, position] = -0.3
        result = calibrator.deltas(z, changed)
        # Positions up to and including this one cannot see the change.
        assert torch.equal(result[0, : position + 1], base[0, : position + 1])
        if position < 4:
            assert not torch.equal(result[0, position + 1 :], base[0, position + 1 :])


def test_function_class_contains_the_true_clipped_map():
    """With D_i = R(alpha_i) - I, the output is exactly clip(R(alpha_i) a_i)."""
    calibrator = make()
    angles = torch.tensor([0.3, -1.1, 0.7, 1.3, -0.4])

    def true_deltas(z, commands):
        c, s = angles.cos(), angles.sin()
        rotation = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
        return (rotation - torch.eye(2)).unsqueeze(0).expand(z.size(0), -1, -1, -1)

    calibrator.deltas = true_deltas
    rng = np.random.default_rng(0)
    raw = rng.uniform(-1, 1, size=(20, 5, 2)).astype(np.float32)
    blocks = to_normalized(torch.as_tensor(raw)).reshape(20, 10)
    out = to_raw(calibrator(torch.zeros(20, 16), blocks).reshape(20, 5, 2)).numpy()
    for episode in range(20):
        for i in range(5):
            rotated = rotate(raw[episode, i], float(angles[i]))
            expected = np.clip(rotated, -1.0, 1.0)
            assert out[episode, i] == pytest.approx(expected, abs=1e-5)


def test_out_of_range_part_of_a_candidate_is_kept():
    calibrator = make(randomize=True)
    z = torch.randn(1, 16)
    inside = torch.full((1, 5, 2), 0.5)
    outside = inside.clone()
    outside[0, 2, 0] = 3.0  # commands clip to 1.0 here, so this differs by 2.0
    clipped = inside.clone()
    clipped[0, 2, 0] = 1.0
    out_outside = to_raw(calibrator(z, to_normalized(outside).reshape(1, 10)).reshape(1, 5, 2))
    out_clipped = to_raw(calibrator(z, to_normalized(clipped).reshape(1, 10)).reshape(1, 5, 2))
    expected = out_clipped.clone()
    expected[0, 2, 0] += 2.0
    assert torch.allclose(out_outside, expected, atol=1e-5)


def test_rollout_is_unchanged_until_the_calibrator_learns():
    model = small_model()
    torch.manual_seed(1)
    z, b, blocks = torch.randn(3, 3, 16), torch.randn(3, 2, 10), torch.randn(3, 5, 10)
    with torch.no_grad():
        plain = rollout_latents(model, z, b, blocks)
        model.calibrator = make()
        calibrated = rollout_latents(model, z, b, blocks)
        assert torch.equal(plain, calibrated)
        nn.init.normal_(model.calibrator.net[-1].weight, std=0.1)
        assert not torch.equal(rollout_latents(model, z, b, blocks), plain)


def test_rollout_calibrates_each_block_from_its_start_latent():
    model = small_model()
    seen = []

    class Recording(nn.Module):
        def forward(self, z, blocks):
            seen.append((z.detach().clone(), blocks.detach().clone()))
            return blocks

    model.calibrator = Recording()
    torch.manual_seed(1)
    z, b, blocks = torch.randn(2, 3, 16), torch.randn(2, 2, 10), torch.randn(2, 4, 10)
    with torch.no_grad():
        predicted = rollout_latents(model, z, b, blocks)
    assert len(seen) == 1 + 4
    assert torch.equal(seen[0][0], z[:, :2]) and torch.equal(seen[0][1], b)
    assert torch.equal(seen[1][0], z[:, -1:]) and torch.equal(seen[1][1], blocks[:, :1])
    for h in range(1, 4):
        assert torch.equal(seen[h + 1][0], predicted[:, h - 1 : h])
        assert torch.equal(seen[h + 1][1], blocks[:, h : h + 1])


def test_planner_uses_the_calibrator():
    model, head = small_model(), small_head()
    torch.manual_seed(2)
    z, b = torch.randn(1, 1, 16), torch.zeros(1, 0, 10)
    candidates = torch.randn(1, 6, 5, 10)
    with torch.no_grad():
        plain = score_candidates(model, head, z, b, candidates, 0.99)
        model.calibrator = make(randomize=True)
        calibrated = score_candidates(model, head, z, b, candidates, 0.99)
    assert not torch.equal(plain, calibrated)
    block = cem_plan(model, head, z, b, TINY_PLANNER, torch.Generator().manual_seed(0))
    assert block.shape == (1, 10) and torch.isfinite(block).all()


def test_applied_actions_reproduce_the_shifted_world_exactly():
    """The core premise: a shifted step equals a base step with the applied action."""
    for shift in ("grid", "wave"):
        shifted, base = ShiftedTwoRoom(shift), ShiftedTwoRoom("none")
        shifted.reset_task(seed=3)
        base.reset_task(seed=3)
        rng = np.random.default_rng(1)
        for _ in range(60):
            shifted.step(rng.uniform(-1, 1, size=2).astype(np.float32))
            base.step(shifted.last_applied)
            assert torch.equal(shifted.agent_position, base.agent_position)


def test_last_applied_matches_the_rotation_field():
    env = ShiftedTwoRoom("wave")
    env.reset_task(seed=1, start_mode="fixed")
    x, y = float(env.agent_position[0]), float(env.agent_position[1])
    command = np.array([0.6, 0.2], dtype=np.float32)
    env.step(command)
    expected = np.clip(rotate(command, rotation_angle("wave", x, y)), -1, 1)
    assert env.last_applied == pytest.approx(expected)


def test_recorded_applied_history_matches_a_replay():
    model, head = small_model(), small_head()
    env = ShiftedTwoRoom("grid")
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
    replay = ShiftedTwoRoom("none")
    replay.reset_task(seed=4)
    actions = STATS.denormalize(batch.applied_history[0].numpy())
    for action in actions:
        replay.step(np.clip(action.astype(np.float32), -1, 1))
    assert replay.agent_position.numpy() == pytest.approx(env.agent_position.numpy(), abs=1e-3)


class LinearWorld(nn.Module):
    """A frozen world where a block moves the latent by its summed actions."""

    def __init__(self):
        super().__init__()
        self.projector = nn.Module()
        self.projector.net = nn.Sequential(nn.Linear(4, 4))
        self.requires_grad_(False)

    @staticmethod
    def action_encoder(blocks):
        raw = blocks.reshape(*blocks.shape[:-1], 5, 2).sum(dim=-2)
        return torch.cat([raw, torch.zeros_like(raw)], dim=-1)

    @staticmethod
    def predict(z, act_emb):
        return z + act_emb

    def eval(self):
        return self


def test_gradient_training_recovers_a_constant_rotation():
    """The loss must drive calibration to the shift, through a frozen model."""
    stats = ActionStats(mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    model = LinearWorld()
    torch.manual_seed(0)
    calibrator = ActionCalibrator(stats.mean, stats.std, 4, hidden_dim=64)
    model.calibrator = calibrator
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=1e-3, weight_decay=1e-3)
    angle = math.radians(60.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )

    def sample():
        raw = torch.rand(64, 3, 5, 2) * 0.8 - 0.4
        applied = raw @ rotation.T
        z = [torch.randn(64, 4) * 0.1]
        for j in range(3):
            step = torch.cat([applied[:, j].sum(dim=1), torch.zeros(64, 2)], dim=-1)
            z.append(z[-1] + step)
        return torch.stack(z, dim=1), raw.reshape(64, 3, 10)

    z, b = sample()
    initial = prediction_loss(model, z, b).item()
    for _ in range(500):
        z, b = sample()
        loss = prediction_loss(model, z, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    z, b = sample()
    assert prediction_loss(model, z, b).item() < 0.05 * initial

    # The learned map should now approximate the true rotation on fresh commands.
    raw = torch.rand(32, 1, 5, 2) * 0.8 - 0.4
    with torch.no_grad():
        out = calibrator(z[:32, :1], raw.reshape(32, 1, 10)).reshape(32, 1, 5, 2)
    summed_out = out.sum(dim=2)
    summed_true = (raw @ rotation.T).sum(dim=2)
    error = (summed_out - summed_true).norm(dim=-1).mean()
    assert error < 0.15 * summed_true.norm(dim=-1).mean()


def test_only_the_calibrator_trains_and_the_model_is_untouched():
    model = small_model()
    before = snapshot(model)
    trainer = TestTimeTrainer(model, ARMS["action_calibration"], seed=0, stats=STATS)
    trainer.start_episode()
    z, a = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
    calibrator_before = copy.deepcopy(model.calibrator.state_dict())
    for t in range(0, 50, 5):
        trainer.add_block(z, a, t)
    trainer.start_episode()
    for t in range(0, 50, 5):
        trainer.add_block(z, a, t)
    assert trainer.windows_seen == 64
    assert trainer.total_updates == 16
    after = {k: v for k, v in snapshot(model).items() if not k.startswith("calibrator.")}
    assert same(before, after)
    # The zero-initialized last layer can only move through a real gradient.
    assert model.calibrator.net[-1].weight.abs().sum() > 0
    assert not same(calibrator_before, model.calibrator.state_dict())
    assert all(
        not p.requires_grad for n, p in model.named_parameters() if not n.startswith("calibrator.")
    )
    assert all(p.requires_grad for p in model.calibrator.parameters())


def test_calibrator_needs_stats_and_is_seeded():
    with pytest.raises(ValueError):
        TestTimeTrainer(small_model(), ARMS["action_calibration"], seed=0)
    first, second = small_model(), small_model()
    TestTimeTrainer(first, ARMS["action_calibration"], seed=3, stats=STATS)
    TestTimeTrainer(second, ARMS["action_calibration"], seed=3, stats=STATS)
    assert same(first.calibrator.state_dict(), second.calibrator.state_dict())


def test_fine_tuning_arms_have_no_calibrator():
    model = small_model()
    TestTimeTrainer(model, ARMS["persistent_replay_x16"], seed=0)
    assert getattr(model, "calibrator", None) is None


def test_update_schedules_scale_with_the_arm():
    for arm, per_batch in (("persistent_replay_x8", 8), ("persistent_replay_x16", 16)):
        model = small_model()
        trainer = TestTimeTrainer(model, ARMS[arm], seed=0)
        for _ in range(4):
            trainer.start_episode()
            z, a = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
            for t in range(0, 50, 5):
                trainer.add_block(z, a, t)
        assert trainer.total_updates == per_batch * (32 * 4 // 64)


def test_deployment_follows_dense_schedule_and_reverses():
    model, head = small_model(), small_head()
    trajectories = collect_fixed_trajectories(
        model, head, STATS, "grid", TINY_PLANNER, 2, 2, "uniform_left"
    )
    assert trajectories["a_applied"].shape == (2, 50, 2)
    frozen_state = {k: v.clone() for k, v in model.state_dict().items()}
    rows = []
    run_deployment(
        model,
        head,
        STATS,
        "action_calibration",
        "grid",
        run_seed=0,
        planner=TINY_PLANNER,
        episodes=4,
        eval_every=4,
        eval_episodes=2,
        eval_batch=2,
        start_mode="uniform_left",
        trajectories=trajectories,
        log=rows.append,
        dense_until=2,
        dense_every=1,
        reversal_episodes=2,
    )
    evals = [r for r in rows if r["kind"] == "eval"]
    reversal = [r for r in rows if r["kind"] == "reversal"]
    assert [r["episode"] for r in evals] == [0, 1, 2, 4]
    assert [r["episode"] for r in reversal] == [0, 2]
    assert [r["updates"] for r in reversal] == [32, 48]  # 128, then 192 windows
    assert all("latent_mse_applied" not in r for r in evals)
    after = {k: v for k, v in model.state_dict().items() if not k.startswith("calibrator.")}
    assert same(frozen_state, after)


def test_frozen_row_logs_the_premise_check():
    model, head = small_model(), small_head()
    trajectories = collect_fixed_trajectories(
        model, head, STATS, "grid", TINY_PLANNER, 2, 2, "uniform_left"
    )
    rows = []
    run_deployment(
        model,
        head,
        STATS,
        "frozen",
        "grid",
        run_seed=0,
        planner=TINY_PLANNER,
        episodes=1,
        eval_every=1,
        eval_episodes=2,
        eval_batch=2,
        start_mode="uniform_left",
        trajectories=trajectories,
        log=rows.append,
    )
    expected = fixed_trajectory_mse(model, {"z": trajectories["z"], "a": trajectories["a_applied"]})
    assert rows[0]["latent_mse_applied"] == pytest.approx(expected)


def run_row(arm, seed, scores, reversal=(), mse=0.1):
    episodes = (0, 5, 50)
    return {
        "arm": arm,
        "seed": seed,
        "evals": [
            {
                "episode": e,
                "score_final": s,
                "success_rate": 1.0,
                "latent_mse": mse,
                "latent_mse_applied": mse / 4,
                "updates": 8,
            }
            for e, s in zip(episodes, scores)
        ],
        "reversal": [{"episode": e, "score_final": s} for e, s in reversal],
    }


def test_report_helpers():
    episodes = np.array([0, 5, 10])
    assert episodes_to_reach(episodes, np.array([0.1, 0.8, 0.9]), 0.7) == 5
    assert episodes_to_reach(episodes, np.array([0.1, 0.2, 0.3]), 0.7) is None
    assert episodes_to_reach(episodes, np.array([0.8, 0.2, 0.9]), 0.7) == 10  # must hold
    assert recovers_faster(5, 10) and not recovers_faster(6, 10)
    assert recovers_faster(100, None, 500) and not recovers_faster(300, None, 500)
    assert not recovers_faster(None, 10)
    assert reversal_score(
        [run_row("a", 0, (0, 0, 0), ((10, 0.4),)), run_row("a", 1, (0, 0, 0), ((10, 0.6),))], 10
    ) == pytest.approx(0.5)
    assert premise_holds(run_row("frozen", 0, (0.2, 0.2, 0.2)))[0]


def winning_groups(seeds=range(3)):
    def arm(name, scores):
        return [run_row(name, s, scores) for s in seeds]

    return {
        ("grid", "persistent_replay_x4"): arm("persistent_replay_x4", (0.2, 0.3, 0.8)),
        ("grid", "persistent_replay_x16"): arm("persistent_replay_x16", (0.2, 0.3, 0.8)),
        ("grid", "action_calibration"): arm("action_calibration", (0.2, 0.8, 0.8)),
        ("grid", "frozen"): [run_row("frozen", 0, (0.2, 0.2, 0.2))],
    }


def test_judge_passes_only_when_every_condition_holds(capsys):
    assert judge_shift(winning_groups(), "grid", 0.8, "score_final") is True

    # Same AUC as the baseline fails, even with faster recovery elsewhere.
    grouped = winning_groups()
    grouped[("grid", "persistent_replay_x16")] = [
        run_row("persistent_replay_x16", s, (0.2, 0.8, 0.8)) for s in range(3)
    ]
    assert judge_shift(grouped, "grid", 0.8, "score_final") is False

    # Higher AUC without recovering in half the episodes fails.
    grouped = winning_groups()
    grouped[("grid", "persistent_replay_x4")] = [
        run_row("persistent_replay_x4", s, (0.2, 0.2, 0.2)) for s in range(3)
    ]
    grouped[("grid", "action_calibration")] = [
        run_row("action_calibration", s, (0.2, 0.3, 0.8)) for s in range(3)
    ]
    grouped[("grid", "persistent_replay_x16")] = [
        run_row("persistent_replay_x16", s, (0.2, 0.2, 0.8)) for s in range(3)
    ]
    assert judge_shift(grouped, "grid", 0.8, "score_final") is False

    # A failed premise fails even if the scores look good.
    grouped = winning_groups()
    grouped[("grid", "frozen")][0]["evals"][0]["latent_mse_applied"] = 1.0
    assert judge_shift(grouped, "grid", 0.8, "score_final") is False
    capsys.readouterr()


def test_judge_is_incomplete_without_runs_or_matching_seeds(capsys):
    for key in winning_groups():
        grouped = winning_groups()
        del grouped[key]
        assert judge_shift(grouped, "grid", 0.8, "score_final") is None
    grouped = winning_groups()
    grouped[("grid", "action_calibration")] = grouped[("grid", "action_calibration")][:1]
    assert judge_shift(grouped, "grid", 0.8, "score_final") is None
    capsys.readouterr()


def test_baseline_gate_keeps_only_shifts_that_hurt(tmp_path):
    for shift, score in (("none", 0.8), ("grid", 0.2), ("wave", 0.7)):
        row = {"kind": "eval", "score_final": score}
        (tmp_path / f"frozen__{shift}__seed0.jsonl").write_text(json.dumps(row) + "\n")
    assert check_baselines(tmp_path, ["grid", "wave"], 0) == ["grid"]
    with pytest.raises(SystemExit):
        check_baselines(tmp_path, ["wave"], 0)
    row = {"kind": "eval", "score_final": 0.5}
    (tmp_path / "frozen__none__seed0.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(SystemExit):
        check_baselines(tmp_path, ["grid"], 0)


def test_training_calibrates_each_block_from_its_own_start_latent():
    model = small_model()
    seen = []

    class Recording(nn.Module):
        def forward(self, z, blocks):
            seen.append((z.clone(), blocks.clone()))
            return blocks

    model.calibrator = Recording()
    z, b = torch.randn(2, 4, 16), torch.randn(2, 3, 10)
    prediction_loss(model, z, b)
    assert torch.equal(seen[0][0], z[:, :3]) and torch.equal(seen[0][1], b)


def test_gradient_training_learns_a_state_dependent_shift():
    """The rotation's sign depends on the latent, so misaligned inputs could not fit it."""
    stats = ActionStats(mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    model = LinearWorld()
    torch.manual_seed(0)
    calibrator = ActionCalibrator(stats.mean, stats.std, 4, hidden_dim=64)
    model.calibrator = calibrator
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=1e-3, weight_decay=1e-3)
    angle = math.radians(60.0)
    c, s = math.cos(angle), math.sin(angle)

    def sample():
        raw = torch.rand(128, 3, 5, 2) * 0.8 - 0.4
        z = [torch.randn(128, 4)]
        for j in range(3):
            sign = torch.sign(z[j][:, 2:3])
            sums = raw[:, j].sum(dim=1)
            moved = torch.cat(
                [
                    c * sums[:, :1] - sign * s * sums[:, 1:],
                    sign * s * sums[:, :1] + c * sums[:, 1:],
                ],
                dim=-1,
            )
            z.append(z[j] + torch.cat([moved, torch.zeros(128, 2)], dim=-1))
        return torch.stack(z, dim=1), raw.reshape(128, 3, 10)

    z, b = sample()
    initial = prediction_loss(model, z, b).item()
    for _ in range(800):
        z, b = sample()
        loss = prediction_loss(model, z, b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    z, b = sample()
    assert prediction_loss(model, z, b).item() < 0.25 * initial


def test_saturated_commands_keep_finite_gradients():
    model = LinearWorld()
    calibrator = make(latent_dim=4, randomize=True)
    model.calibrator = calibrator
    z = torch.randn(4, 4, 4)
    b = to_normalized(torch.ones(4, 3, 5, 2)).reshape(4, 3, 10)  # every command at the limit
    prediction_loss(model, z, b).backward()
    for param in calibrator.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
