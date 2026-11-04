"""Draw the README figures from results/summary.csv and the saved run logs.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SHIFT_TITLES = {"grid": "Grid shift", "wave": "Wave shift"}

FAMILIES = {
    "Fine-tunes the predictor": (BLUE, "o"),
    "Freezes the predictor, translates actions": (ORANGE, "s"),
    "Freezes the predictor, corrects its output": (AQUA, "^"),
}
ARMS = {
    "persistent_replay_x4": ("x4", "Fine-tunes the predictor"),
    "persistent_replay_x8": ("x8", "Fine-tunes the predictor"),
    "persistent_replay_x16": ("x16", "Fine-tunes the predictor"),
    "rollout_loss_x4": ("rollout loss", "Fine-tunes the predictor"),
    "anchored_x4": ("anchored", "Fine-tunes the predictor"),
    "action_calibration": ("calibration", "Freezes the predictor, translates actions"),
    "calibration_before_now": ("before-plus-now", "Freezes the predictor, translates actions"),
    "calibration_fast": ("fast", "Freezes the predictor, translates actions"),
    "latent_correction": ("latent correction", "Freezes the predictor, corrects its output"),
}
LABEL_OFFSETS = {
    ("grid", "persistent_replay_x8"): (-14, -13),
    ("grid", "persistent_replay_x16"): (6, -13),
    ("grid", "persistent_replay_x4"): (6, -4),
    ("grid", "anchored_x4"): (-8, -14),
    ("grid", "rollout_loss_x4"): (-16, -15),
    ("grid", "calibration_fast"): (6, 5),
    ("grid", "action_calibration"): (-20, -15),
    ("grid", "calibration_before_now"): (-30, 7),
    ("wave", "calibration_fast"): (-12, -15),
    ("wave", "action_calibration"): (-70, -4),
    ("wave", "calibration_before_now"): (-40, 7),
    ("wave", "persistent_replay_x8"): (-5, -14),
    ("wave", "persistent_replay_x16"): (7, -3),
    ("wave", "persistent_replay_x4"): (7, -3),
}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)


def draw_tradeoff() -> None:
    with (RESULTS / "summary.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    report = (RESULTS / "calibration" / "report.txt").read_text()
    no_shift = float(report.splitlines()[0].rsplit(" ", 1)[1])
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, facecolor=SURFACE)
    for ax, shift in zip(axes, SHIFT_TITLES):
        style(ax)
        ax.axhline(no_shift, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (4, 3)))
        ax.margins(x=0.09)
        # Each label goes on the side of its panel that has no points near the line.
        on_right = shift == "grid"
        ax.annotate(
            "frozen model, no shift",
            (0.98 if on_right else 0.02, no_shift),
            xycoords=("axes fraction", "data"),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=8.5,
            color=TEXT_SECONDARY,
            ha="right" if on_right else "left",
        )
        for row in rows:
            if row["shift"] != shift or row["arm"] not in ARMS:
                continue
            label, family = ARMS[row["arm"]]
            color, marker = FAMILIES[family]
            x, y = float(row["auc"]), float(row["back_0"])
            ax.scatter(
                x, y, s=80, c=color, marker=marker, edgecolors=SURFACE, linewidths=1.5, zorder=3
            )
            ax.annotate(
                label,
                (x, y),
                xytext=LABEL_OFFSETS.get((shift, row["arm"]), (6, 5)),
                textcoords="offset points",
                fontsize=8.5,
                color=TEXT,
            )
        ax.set_title(SHIFT_TITLES[shift], loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel(
            "Adapting to the shift (AUC, higher is better)", fontsize=9, color=TEXT_SECONDARY
        )
    axes[0].set_ylabel("Score right after switching back", fontsize=9, color=TEXT_SECONDARY)
    axes[0].set_ylim(0.1, 0.87)
    handles = [
        plt.Line2D([], [], color=c, marker=m, linestyle="", markersize=8, label=name)
        for name, (c, m) in FAMILIES.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(FIGURES / "adapt_vs_retain.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


def mean_curve(arm: str, shift: str) -> tuple[np.ndarray, np.ndarray]:
    curves, episodes = [], None
    for seed in range(3):
        path = RESULTS / "calibration" / "runs" / f"{arm}__{shift}__seed{seed}.jsonl"
        evals = [json.loads(line) for line in path.read_text().splitlines()]
        evals = [e for e in evals if e.get("kind") == "eval"]
        episodes = np.array([e["episode"] for e in evals])
        curves.append([e["score_final"] for e in evals])
    return episodes, np.mean(curves, axis=0)


def draw_curves() -> None:
    series = {
        "persistent_replay_x4": ("x4", BLUE),
        "persistent_replay_x8": ("x8", ORANGE),
        "persistent_replay_x16": ("x16", AQUA),
        "action_calibration": ("calibration", YELLOW),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, facecolor=SURFACE)
    for ax, shift in zip(axes, SHIFT_TITLES):
        style(ax)
        for arm, (label, color) in series.items():
            episodes, curve = mean_curve(arm, shift)
            ax.plot(episodes, curve, color=color, linewidth=2, label=label)
        ax.set_title(SHIFT_TITLES[shift], loc="left", fontsize=11, color=TEXT)
        ax.set_xlabel("Deployment episodes", fontsize=9, color=TEXT_SECONDARY)
        ax.set_xlim(0, 500)
    axes[0].set_ylabel("Held-out score, mean of 3 seeds", fontsize=9, color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(FIGURES / "adaptation_curves.png", dpi=170, facecolor=SURFACE)
    plt.close(figure)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    draw_tradeoff()
    draw_curves()
    print(f"Figures written to {FIGURES}")
