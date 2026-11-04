"""The machine checks that protect a paid run."""

from __future__ import annotations

from pathlib import Path

import pytest

from jepa_ttt.preflight import (
    check_disk,
    read_cgroup_limit,
    resolve_gpus,
    resolve_jobs,
)


@pytest.mark.parametrize(
    "gpus, cores, memory, expected",
    [
        (4, 96, 32, 24),  # the rented 4x RTX 5090: 6 runs per card
        (4, 32, 24, 24),  # the earlier 4x RTX 3090: 6 per card, within 30 usable cores
        (4, 16, 24, 14),  # too few cores, so two are left spare
        (1, 96, 8, 2),  # small card
        (2, 2, 32, 1),  # never below one run
    ],
)
def test_automatic_jobs_fit_cores_and_memory(gpus, cores, memory, expected):
    assert resolve_jobs("auto", gpus, cores, memory) == expected


def test_requested_jobs_are_respected():
    assert resolve_jobs("9", 4, 2, 8) == 9


def test_gpu_count_is_automatic_and_checked():
    assert resolve_gpus("auto", 4) == 4
    assert resolve_gpus("2", 4) == 2
    with pytest.raises(SystemExit):
        resolve_gpus("5", 4)
    with pytest.raises(SystemExit):
        resolve_gpus("auto", 0)


def test_reads_cgroup_v2_quota(tmp_path: Path):
    (tmp_path / "cpu.max").write_text("9600000 100000\n")
    assert read_cgroup_limit(tmp_path) == 96
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert read_cgroup_limit(tmp_path) is None


def test_reads_cgroup_v1_quota(tmp_path: Path):
    folder = tmp_path / "cpu"
    folder.mkdir()
    (folder / "cpu.cfs_quota_us").write_text("3200000\n")
    (folder / "cpu.cfs_period_us").write_text("100000\n")
    assert read_cgroup_limit(tmp_path) == 32
    (folder / "cpu.cfs_quota_us").write_text("-1\n")
    assert read_cgroup_limit(tmp_path) is None


def test_missing_or_broken_cgroup_means_no_limit(tmp_path: Path):
    assert read_cgroup_limit(tmp_path) is None
    (tmp_path / "cpu.max").write_text("nonsense\n")
    assert read_cgroup_limit(tmp_path) is None


def test_disk_check_depends_on_whether_setup_is_done(tmp_path: Path, monkeypatch):
    usage = type("Usage", (), {"free": 20 * 1024**3})
    monkeypatch.setattr("jepa_ttt.preflight.shutil.disk_usage", lambda path: usage)
    with pytest.raises(SystemExit):
        check_disk(tmp_path)
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "prepare.json").write_text("{}")
    check_disk(tmp_path)
