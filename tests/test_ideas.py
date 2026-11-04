"""The further ideas: before-plus-now, fast adaptation, correction, rollout, anchoring."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from jepa_ttt.adaptation import (
    ARMS,
    CALIBRATION_ARMS,
    IDEA_ARMS,
    STAGE1_ARMS,
    TestTimeTrainer,
    prediction_loss,
    rollout_loss,
)
from jepa_ttt.calibration import (
    FAST_STEP_SIZE,
    FAST_STEPS,
    ActionCalibrator,
    LatentCorrection,
    choose_worlds,
    fast_steps,
)
from jepa_ttt.envs import ShiftedTwoRoom
from jepa_ttt.experiment import ActionStats, evaluate, run_deployment, run_episodes
from jepa_ttt.planning import (
    PlannerConfig,
    predict_with,
    recent_errors,
    rollout_latents,
    score_candidates,
)
from jepa_ttt.report_calibration import IDEA_BASELINES, TABLE_ARMS, judge_ideas
from tests.test_stage1 import same, small_head, small_model, snapshot

STATS = ActionStats(
    mean=np.array([0.1, -0.2], dtype=np.float32),
    std=np.array([0.8, 0.9], dtype=np.float32),
)
TINY_PLANNER = PlannerConfig(num_samples=6, iterations=2, elites=2)


def history(envs: int = 3, steps: int = 20):
    torch.manual_seed(5)
    return torch.randn(envs, steps + 1, 16), torch.randn(envs, steps, 2)


def calibrated_model(randomize: bool = True):
    model = small_model()
    torch.manual_seed(0)
    model.calibrator = ActionCalibrator(STATS.mean, STATS.std, 16, hidden_dim=32)
    if randomize:
        nn.init.normal_(model.calibrator.net[-1].weight, std=0.1)
    return model


def test_ideas_are_new_runs_and_earlier_arms_are_not_rerun():
    assert not set(IDEA_ARMS) & set(CALIBRATION_ARMS)
    assert not set(IDEA_ARMS) & set(STAGE1_ARMS)
    for arm in IDEA_ARMS:
        assert arm in ARMS and arm in TABLE_ARMS
    assert set(IDEA_BASELINES) | {"calibration_before_now"} == set(IDEA_ARMS)
    assert set(IDEA_BASELINES.values()) <= set(CALIBRATION_ARMS)


def test_each_idea_changes_only_what_it_claims():
    plain = ARMS["action_calibration"]
    assert ARMS["calibration_before_now"].gated and not plain.gated
    assert ARMS["calibration_fast"].adapt_fast and not plain.adapt_fast
    assert (ARMS["calibration_before_now"].lr, ARMS["calibration_fast"].lr) == (plain.lr, plain.lr)
    x4 = ARMS["persistent_replay_x4"]
    assert ARMS["rollout_loss_x4"].updates_per_batch == x4.updates_per_batch
    assert ARMS["anchored_x4"].updates_per_batch == x4.updates_per_batch
    assert ARMS["rollout_loss_x4"].lr == ARMS["anchored_x4"].lr == x4.lr


def test_earlier_arms_keep_their_settings():
    """Saved results stay comparable only if these never changed."""
    for name, per_batch in (
        ("persistent_replay_x4", 4),
        ("persistent_replay_x8", 8),
        ("persistent_replay_x16", 16),
    ):
        arm = ARMS[name]
        assert (arm.updates_per_batch, arm.lr, arm.rollout_weight, arm.anchor_weight) == (
            per_batch,
            5e-5,
            0.0,
            0.0,
        )
    plain = ARMS["action_calibration"]
    assert (plain.updates_per_batch, plain.lr, plain.calibrate) == (16, 1e-3, True)
    assert not (plain.gated or plain.adapt_fast or plain.correct)


def test_prediction_is_unchanged_without_a_correction():
    model = small_model()
    z, b = torch.randn(4, 3, 16), torch.randn(4, 3, 10)
    with torch.no_grad():
        assert torch.equal(predict_with(model, z, b), model.predict(z, model.action_encoder(b)))


def test_recent_errors_match_a_direct_one_step_prediction():
    model = small_model()
    z, a = history(steps=20)
    with torch.no_grad():
        errors = recent_errors(model, z, a, 5)
        block = a[:, 0:5].reshape(3, 1, -1)
        predicted = model.predict(z[:, :1], model.action_encoder(block))[:, -1]
        expected = (predicted - z[:, 5]).pow(2).mean(dim=-1)
    assert torch.allclose(errors, expected)
    assert errors.shape == (3,)


def test_recent_errors_average_the_last_three_blocks():
    model = small_model()
    z, a = history(steps=30)
    with torch.no_grad():
        combined = recent_errors(model, z, a, 25, blocks=3)
        singles = [
            recent_errors(model, z[:, : k + 6], a[:, : k + 5], k + 5, blocks=1)
            for k in (10, 15, 20)
        ]
    assert torch.allclose(combined, torch.stack(singles).mean(dim=0), atol=1e-6)


def test_gate_uses_default_first_and_needs_clear_evidence(monkeypatch):
    model = calibrated_model()
    z, a = history()
    model.calibrator.default_on = False
    assert not choose_worlds(model, z, a, 0).any()
    model.calibrator.default_on = True
    assert choose_worlds(model, z, a, 0).all()

    errors = {True: torch.tensor([1.0, 1.0, 1.0]), False: torch.tensor([0.4, 0.5, 0.9])}

    def fake(model, z_history, a_history, t):
        return errors[model.calibrator.enabled]

    monkeypatch.setattr("jepa_ttt.calibration.recent_errors", fake)
    assert choose_worlds(model, z, a, 5).tolist() == [False, True, True]
    assert model.calibrator.enabled is True


def test_disabled_calibrator_is_the_frozen_model():
    model = calibrated_model()
    z, b, blocks = torch.randn(2, 3, 16), torch.randn(2, 2, 10), torch.randn(2, 5, 10)
    with torch.no_grad():
        model.calibrator.enabled = False
        off = rollout_latents(model, z, b, blocks)
        model.calibrator = None
        plain = rollout_latents(model, z, b, blocks)
    assert torch.equal(off, plain)


def test_gated_planning_matches_each_world_per_environment(monkeypatch):
    model, head = calibrated_model(), small_head()
    model.calibrator.gated = True
    seen = []

    def fake_cem(model, reward_head, z_ctx, b_ctx, planner, generator):
        seen.append((model.calibrator.enabled, z_ctx.size(0)))
        value = 1.0 if model.calibrator.enabled else -1.0
        return torch.full((z_ctx.size(0), 10), value)

    monkeypatch.setattr("jepa_ttt.experiment.cem_plan", fake_cem)
    monkeypatch.setattr(
        "jepa_ttt.experiment.choose_worlds",
        lambda m, z, a, t: torch.tensor([True, False, True]),
    )
    envs = [ShiftedTwoRoom("none") for _ in range(3)]
    for i, env in enumerate(envs):
        env.reset_task(seed=i)
    batch = run_episodes(
        model,
        head,
        envs,
        STATS,
        TINY_PLANNER,
        torch.Generator(),
        horizon=5,
        record_all_steps=True,
    )
    assert seen == [(True, 2), (False, 1)]
    assert model.calibrator.enabled is True
    # Environment 1 received the frozen plan and the others the calibrated plan.
    raw = STATS.denormalize(batch.a_history[:, 0].numpy())
    expected_on = np.clip(STATS.denormalize(np.array([1.0, 1.0], np.float32)), -1, 1)
    expected_off = np.clip(STATS.denormalize(np.array([-1.0, -1.0], np.float32)), -1, 1)
    assert raw[0] == pytest.approx(expected_on, abs=1e-5)
    assert raw[2] == pytest.approx(expected_on, abs=1e-5)
    assert raw[1] == pytest.approx(expected_off, abs=1e-5)


def test_gated_trainer_skips_old_world_windows_but_keeps_streams_contiguous():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["calibration_before_now"], seed=0, stats=STATS)
    trainer.start_episode()
    z, a = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
    model.calibrator.last_on = True
    for t in range(0, 25, 5):
        trainer.add_block(z, a, t)  # rows 0..24: starts 0..5 finish, 6 windows
    kept = len(trainer.buffer)
    assert kept == 6
    # Off, and the frozen model predicts well: this looks like the old world.
    model.calibrator.last_on = False
    model.calibrator.last_before_error = 0.01
    trainer.add_block(z, a, 25)
    assert len(trainer.buffer) == kept and trainer.windows_dropped == 5
    model.calibrator.last_on = True
    trainer.add_block(z, a, 30)
    # Rows 30..34 finish starts 11..15; the skipped rows still entered the stream.
    assert len(trainer.buffer) == kept + 5
    assert trainer.buffer._z[-1][0].tolist() == z[0, 15].tolist()
    assert trainer.gate_stats() == {"gate_on_rate": 6 / 7, "windows_dropped": 5}


def test_gate_keeps_learning_while_the_frozen_model_predicts_badly():
    """Off alone must not stop learning, or a calibrator gone wrong could never recover."""
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["calibration_before_now"], seed=0, stats=STATS)
    trainer.start_episode()
    z, a = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
    model.calibrator.last_on = False
    model.calibrator.last_before_error = 1.0
    for t in range(0, 50, 5):
        trainer.add_block(z, a, t)
    assert len(trainer.buffer) == 32 and trainer.windows_dropped == 0


def test_default_world_follows_the_episode_majority():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["calibration_before_now"], seed=0, stats=STATS)
    trainer.start_episode()
    z, a = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
    for t, on in zip(range(0, 25, 5), (True, True, True, False, False)):
        model.calibrator.last_on = on
        trainer.add_block(z, a, t)
    assert model.calibrator.default_on is True  # one last off block doesn't decide it
    trainer.start_episode()
    assert model.calibrator.default_on is True  # kept until the new episode has a block
    for t, on in zip(range(0, 15, 5), (False, False, True)):
        model.calibrator.last_on = on
        trainer.add_block(z, a, t)
    assert model.calibrator.default_on is False


def test_held_out_evaluation_leaves_deployment_state_alone():
    model, head = small_model(), small_head()
    trainer = TestTimeTrainer(model, ARMS["calibration_before_now"], seed=0, stats=STATS)
    model.calibrator.default_on = False
    weights = snapshot(model.calibrator)
    evaluate(model, head, STATS, "grid", TINY_PLANNER, 2, 2, "uniform_left")
    assert model.calibrator.default_on is False
    assert len(trainer.buffer) == 0 and trainer.total_updates == 0
    assert same(weights, snapshot(model.calibrator))


def test_fast_steps_are_per_environment_and_normalized():
    model = calibrated_model()
    z, a = history(envs=3, steps=20)
    fast = torch.zeros(3, 2, 2)
    moved = fast_steps(model, fast, z, a, 15)
    assert model.calibrator.fast is None
    distances = moved.flatten(1).norm(dim=1)
    assert (distances <= FAST_STEPS * FAST_STEP_SIZE + 1e-5).all()
    assert (distances > 0).all()
    # An environment's matrix depends only on its own history.
    alone = fast_steps(model, torch.zeros(1, 2, 2), z[1:2], a[1:2], 15)
    assert torch.allclose(alone[0], moved[1], atol=1e-5)
    assert torch.equal(fast_steps(model, fast, z, a, 0), fast)


def test_fast_matrices_apply_to_the_right_environment_in_a_rollout():
    model = calibrated_model(randomize=False)
    model.calibrator.fast = torch.stack([torch.zeros(2, 2), 0.3 * torch.eye(2)])
    z = torch.randn(2, 16).repeat_interleave(4, dim=0).unsqueeze(1)
    blocks = torch.zeros(8, 1, 10) + 0.1
    out = model.calibrator(z, blocks)
    # The first environment has no fast matrix, so its blocks are unchanged.
    assert torch.equal(out[:4], blocks[:4])
    assert not torch.allclose(out[4:], blocks[4:])
    with pytest.raises(ValueError):
        model.calibrator(z[:3], blocks[:3])
    model.calibrator.fast = None


def test_fast_matrices_stay_aligned_through_candidate_scoring():
    """Each environment's scores must match scoring that environment alone."""
    model, head = calibrated_model(), small_head()
    torch.manual_seed(4)
    z = torch.randn(3, 3, 16)
    b = torch.randn(3, 2, 10) * 0.5
    candidates = torch.randn(3, 4, 5, 10) * 0.5
    fast = torch.stack([torch.zeros(2, 2), 0.3 * torch.eye(2), -0.3 * torch.eye(2)])
    with torch.no_grad():
        model.calibrator.fast = fast
        together = score_candidates(model, head, z, b, candidates, 0.99)
        for env in range(3):
            model.calibrator.fast = fast[env : env + 1]
            alone = score_candidates(
                model, head, z[env : env + 1], b[env : env + 1], candidates[env : env + 1], 0.99
            )
            assert torch.allclose(together[env], alone[0], atol=1e-5)
    model.calibrator.fast = None


def test_fast_state_is_gone_by_the_time_weights_update():
    model, head = small_model(), small_head()
    trainer = TestTimeTrainer(model, ARMS["calibration_fast"], seed=0, stats=STATS)
    env = ShiftedTwoRoom("grid")
    env.reset_task(seed=1)
    fast_at_update = []
    original = trainer._update_once

    def spy():
        fast_at_update.append(model.calibrator.fast)
        original()

    trainer._update_once = spy
    trainer.start_episode()
    for _ in range(3):
        env.reset_task(seed=1)
        trainer.start_episode()
        run_episodes(
            model,
            head,
            [env],
            STATS,
            TINY_PLANNER,
            torch.Generator().manual_seed(0),
            record_all_steps=True,
            on_block=trainer.add_block,
        )
    assert trainer.total_updates > 0
    assert fast_at_update and all(fast is None for fast in fast_at_update)
    assert model.calibrator.fast is None


def test_correction_starts_at_zero_and_only_it_trains():
    model = small_model()
    z, b = torch.randn(4, 3, 16), torch.randn(4, 3, 10)
    with torch.no_grad():
        plain = predict_with(model, z, b)
    before = snapshot(model)
    with pytest.raises(ValueError):
        TestTimeTrainer(small_model(), ARMS["latent_correction"], seed=0)
    trainer = TestTimeTrainer(model, ARMS["latent_correction"], seed=0, stats=STATS)
    with torch.no_grad():
        assert torch.equal(predict_with(model, z, b), plain)
    zh, ah = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
    for _ in range(2):
        trainer.start_episode()
        for t in range(0, 50, 5):
            trainer.add_block(zh, ah, t)
    assert trainer.total_updates == 16
    after = {k: v for k, v in snapshot(model).items() if not k.startswith("correction.")}
    assert same(before, after)
    assert model.correction.net[-1].weight.abs().sum() > 0
    with torch.no_grad():
        assert not torch.equal(predict_with(model, z, b), plain)


def test_rollout_loss_is_free_running_and_trains():
    model = small_model()
    z, b = torch.randn(3, 4, 16), torch.randn(3, 3, 10)
    with torch.no_grad():
        predicted = rollout_latents(model, z[:, :1], b[:, :0], b)
        expected = torch.nn.functional.mse_loss(predicted, z[:, 1:])
        assert rollout_loss(model, z, b).item() == pytest.approx(expected.item())
    # A later real latent should change only its own target term, not the predictions.
    changed = z.clone()
    changed[:, 1] += 1.0
    with torch.no_grad():
        direct = torch.nn.functional.mse_loss(predicted, changed[:, 1:]).item()
        assert rollout_loss(model, changed, b).item() == pytest.approx(direct)
    trainer = TestTimeTrainer(model, ARMS["rollout_loss_x4"], seed=0)
    start = snapshot(model.predictor)
    for _ in range(2):
        trainer.start_episode()
        zh, ah = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
        for t in range(0, 50, 5):
            trainer.add_block(zh, ah, t)
    assert trainer.total_updates == 4
    assert not same(start, snapshot(model.predictor))


def test_anchor_pulls_toward_pretrained_weights():
    def run(arm):
        model = small_model()
        start = [p.detach().clone() for p in model.predictor.parameters()]
        trainer = TestTimeTrainer(model, ARMS[arm], seed=0)
        torch.manual_seed(9)
        for _ in range(8):
            trainer.start_episode()
            zh, ah = torch.randn(1, 51, 16), torch.randn(1, 50, 2)
            for t in range(0, 50, 5):
                trainer.add_block(zh, ah, t)
        return sum((p - s).pow(2).sum() for p, s in zip(model.predictor.parameters(), start))

    assert run("anchored_x4") < run("persistent_replay_x4")


def test_anchor_loss_has_the_stated_form_over_all_trainable_weights():
    model = small_model()
    trainer = TestTimeTrainer(model, ARMS["anchored_x4"], seed=0)
    modules = (model.action_encoder, model.predictor, model.pred_proj)
    assert sum(len(list(m.parameters())) for m in modules) == len(trainer.anchors)
    with torch.no_grad():
        for param in trainer.params:
            param.add_(0.01)
    expected_drift = sum(p.numel() for p in trainer.params) * 0.01**2
    z, b = torch.randn(64, 4, 16), torch.randn(64, 3, 10)
    trainer.buffer = type("Fixed", (), {"sample": lambda self, n, rng: (z, b)})()
    base = prediction_loss(model, z, b).item()
    trainer._update_once()
    assert trainer.last_loss == pytest.approx(base + 0.5 * 0.01 * expected_drift, rel=1e-3)


def test_correction_clamps_out_of_range_candidates():
    torch.manual_seed(0)
    correction = LatentCorrection(STATS.mean, STATS.std, 16, hidden_dim=8)
    nn.init.normal_(correction.net[-1].weight, std=0.1)
    z = torch.randn(2, 1, 16)
    limit = (1.0 - torch.as_tensor(STATS.mean)) / torch.as_tensor(STATS.std)
    at_limit = limit.repeat(5).expand(2, 1, 10)
    assert torch.equal(correction(z, at_limit), correction(z, at_limit + 3.0))
    inside = at_limit * 0.5
    assert not torch.equal(correction(z, inside), correction(z, at_limit))


@pytest.mark.parametrize("arm", IDEA_ARMS)
def test_every_idea_runs_a_full_deployment_with_reversal(arm):
    model, head = small_model(), small_head()
    rows = []
    run_deployment(
        model,
        head,
        STATS,
        arm,
        "grid",
        run_seed=0,
        planner=TINY_PLANNER,
        episodes=2,
        eval_every=2,
        eval_episodes=2,
        eval_batch=2,
        start_mode="uniform_left",
        trajectories=None,
        log=rows.append,
        reversal_episodes=1,
    )
    assert [r["kind"] for r in rows] == ["eval", "eval", "reversal", "reversal"]
    assert all(np.isfinite(r["score_final"]) for r in rows)
    first = evaluate(model, head, STATS, "grid", TINY_PLANNER, 2, 2, "uniform_left")
    second = evaluate(model, head, STATS, "grid", TINY_PLANNER, 2, 2, "uniform_left")
    assert first == second
    assert ("gate_on_rate" in rows[-1]) == (arm == "calibration_before_now")


def test_latent_correction_shape():
    correction = LatentCorrection(STATS.mean, STATS.std, 16, hidden_dim=8)
    assert correction(torch.zeros(2, 3, 16), torch.zeros(2, 3, 10)).shape == (2, 3, 16)


def row(seed, scores, back=None):
    episodes = (0, 5, 50)
    return {
        "seed": seed,
        "evals": [{"episode": e, "score_final": s} for e, s in zip(episodes, scores)],
        "reversal": [] if back is None else [{"episode": 10, "score_final": back}],
    }


def test_idea_rules_pass_and_fail_correctly(capsys):
    def arm(scores, back=None):
        return [row(s, scores, back) for s in range(3)]

    low, high = (0.2, 0.3, 0.5), (0.2, 0.6, 0.8)
    grouped = {
        ("grid", "action_calibration"): arm(low, 0.3),
        ("grid", "calibration_fast"): arm(high),
        ("grid", "persistent_replay_x16"): arm(high),
        ("grid", "latent_correction"): arm(low),
        ("grid", "persistent_replay_x4"): arm(low),
        ("grid", "rollout_loss_x4"): arm(high),
        ("grid", "calibration_before_now"): arm(low, 0.8),
    }
    verdicts = judge_ideas(grouped, "grid", 0.8, "score_final")
    assert verdicts["calibration_fast"] is True
    assert verdicts["latent_correction"] is False
    assert verdicts["rollout_loss_x4"] is True
    assert verdicts["anchored_x4"] is None
    assert verdicts["calibration_before_now"] is True

    grouped[("grid", "calibration_before_now")] = arm(low, 0.5)  # recovers too little
    assert judge_ideas(grouped, "grid", 0.8, "score_final")["calibration_before_now"] is False
    grouped[("grid", "calibration_before_now")] = arm((0.1, 0.1, 0.1), 0.8)  # costs too much
    assert judge_ideas(grouped, "grid", 0.8, "score_final")["calibration_before_now"] is False
    grouped[("grid", "calibration_before_now")] = arm(low)  # no switch-back scores
    assert judge_ideas(grouped, "grid", 0.8, "score_final")["calibration_before_now"] is None
    grouped[("grid", "rollout_loss_x4")] = arm(high)[:1]  # unmatched seeds
    assert judge_ideas(grouped, "grid", 0.8, "score_final")["rollout_loss_x4"] is None
    # One finished seed in both arms isn't enough either.
    grouped[("grid", "persistent_replay_x4")] = arm(low)[:1]
    assert judge_ideas(grouped, "grid", 0.8, "score_final")["rollout_loss_x4"] is None
    capsys.readouterr()


def test_prediction_loss_uses_the_correction():
    model = small_model()
    z, b = torch.randn(3, 4, 16), torch.randn(3, 3, 10)
    plain = prediction_loss(model, z, b).item()
    torch.manual_seed(0)
    model.correction = LatentCorrection(STATS.mean, STATS.std, 16, hidden_dim=8)
    assert prediction_loss(model, z, b).item() == pytest.approx(plain)
    nn.init.normal_(model.correction.net[-1].weight, std=0.1)
    assert prediction_loss(model, z, b).item() != pytest.approx(plain)
