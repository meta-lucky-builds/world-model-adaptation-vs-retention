"""Two-Room task score and the dynamics shifts from Appendix A2.2 of the paper."""

from __future__ import annotations

import math

import numpy as np
from stable_worldmodel.envs.two_room.env import TwoRoomEnv

GOAL = np.array([164.0, 112.0], dtype=np.float32)
DOORWAY = np.array([112.0, 49.0], dtype=np.float32)
WALL_X = 112.0
SUCCESS_RADIUS = 16.0
SCORE_SCALE = 64.0
EPISODE_HORIZON = 50

# Reachable agent centres in the left room for the default geometry: the border
# stops the centre at 14 + 7 = 21 and the wall at 112 - 5 - 7 = 100.
LEFT_ROOM_X = (21.0, 99.0)
LEFT_ROOM_Y = (21.0, 203.0)
DEFAULT_START = np.array([60.0, 112.0], dtype=np.float32)

SHIFTS = ("none", "wave", "grid")
MAX_ROTATION = math.radians(75.0)


def rotation_angle(shift: str, x: float, y: float) -> float:
    """Return alpha(s) in radians: Eq. A7 for ``wave`` and Eq. A8 for ``grid``."""
    if shift == "none":
        return 0.0
    if shift == "wave":
        return MAX_ROTATION * math.sin(2.0 * math.pi * (x - 20.0) / 88.0)
    if shift == "grid":
        return (
            MAX_ROTATION
            * math.sin(2.0 * math.pi * (x - 20.0) / 96.0)
            * math.cos(2.0 * math.pi * (y - 20.0) / 96.0)
        )
    raise ValueError(f"unknown shift {shift!r}; expected one of {SHIFTS}")


def rotate(action: np.ndarray, angle: float) -> np.ndarray:
    """Apply the planar rotation R(angle) from Eq. A5."""
    c, s = math.cos(angle), math.sin(angle)
    x, y = float(action[0]), float(action[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float32)


def task_distance(position) -> float:
    """D(s, g): through the doorway while left of the wall, direct afterwards."""
    s = np.asarray(position, dtype=np.float64)
    if s[0] < WALL_X:
        return float(np.linalg.norm(s - DOORWAY) + np.linalg.norm(DOORWAY - GOAL))
    return float(np.linalg.norm(s - GOAL))


def task_score(position) -> float:
    """The planning score exp(-D(s, g) / 64)."""
    return math.exp(-task_distance(position) / SCORE_SCALE)


def task_scores(positions: np.ndarray) -> np.ndarray:
    """Vectorized ``task_score`` for positions (N, 2)."""
    s = np.asarray(positions, dtype=np.float64)
    via_door = np.linalg.norm(s - DOORWAY, axis=-1) + np.linalg.norm(DOORWAY - GOAL)
    direct = np.linalg.norm(s - GOAL, axis=-1)
    return np.exp(-np.where(s[..., 0] < WALL_X, via_door, direct) / SCORE_SCALE)


class ShiftedTwoRoom(TwoRoomEnv):
    """TwoRoomEnv whose commanded action passes through R(alpha(s_t))."""

    def __init__(self, shift: str = "none", **kwargs):
        if shift not in SHIFTS:
            raise ValueError(f"unknown shift {shift!r}; expected one of {SHIFTS}")
        super().__init__(**kwargs)
        self.shift = shift
        self.last_applied = np.zeros(2, dtype=np.float32)

    def step(self, action):
        command = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        x, y = float(self.agent_position[0]), float(self.agent_position[1])
        rotated = rotate(command, rotation_angle(self.shift, x, y))
        # The action that really moves the agent, after the base step's own clamp.
        self.last_applied = np.clip(rotated, -1.0, 1.0)
        return super().step(rotated)

    def reset_task(self, seed: int, start_mode: str = "uniform_left"):
        """Reset to the fixed task from a seeded start in the left room.

        Only the start varies with ``seed``; door, wall, speed and appearance
        stay at their default values, and the goal is the paper's fixed goal.
        """
        if start_mode == "uniform_left":
            rng = np.random.default_rng(seed)
            start = np.array(
                [rng.uniform(*LEFT_ROOM_X), rng.uniform(*LEFT_ROOM_Y)],
                dtype=np.float32,
            )
        elif start_mode == "fixed":
            start = DEFAULT_START.copy()
        else:
            raise ValueError(f"unknown start mode {start_mode!r}")
        return self.reset(
            seed=seed,
            options={"variation": (), "state": start, "target_state": GOAL.copy()},
        )

    def score(self) -> float:
        return task_score(self.agent_position.detach().cpu().numpy())

    def reached_goal(self) -> bool:
        position = self.agent_position.detach().cpu().numpy()
        return bool(np.linalg.norm(position - GOAL) < SUCCESS_RADIUS)
