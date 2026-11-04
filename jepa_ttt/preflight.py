"""Check a GPU machine before paying for a run, and choose how many runs to start.

Usage:
    python -m jepa_ttt.preflight --workdir W --jobs auto --gpus auto

Details go to stderr. The only stdout line is "<jobs> <gpus>", for the run scripts.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

GIB = 1024**3
MEMORY_PER_RUN_GIB = 4  # above the roughly 3.5 GiB per run measured on RTX 5090
MAX_RUNS_PER_CARD = 6
SPARE_CORES = 2
MIN_DISK_GIB_TO_PREPARE = 40
MIN_DISK_GIB_PREPARED = 10
MIN_TORCH = (2, 1)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_cgroup_limit(root: Path = Path("/sys/fs/cgroup")) -> float | None:
    """Return the container's CPU quota in cores, or None if it has no quota."""
    try:
        quota, period = (root / "cpu.max").read_text().split()
        return None if quota == "max" else int(quota) / int(period)
    except (OSError, ValueError):
        pass
    try:
        quota = int((root / "cpu" / "cpu.cfs_quota_us").read_text())
        period = int((root / "cpu" / "cpu.cfs_period_us").read_text())
        return None if quota <= 0 else quota / period
    except (OSError, ValueError):
        return None


def usable_cores(root: Path = Path("/sys/fs/cgroup")) -> int:
    """Cores this process can really use, which on a rental can be far below the host's."""
    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    limit = read_cgroup_limit(root)
    return max(1, cores if limit is None else min(cores, int(limit)))


def resolve_jobs(requested: str, gpus: int, cores: int, memory_gib: float) -> int:
    """Pick runs at once from cores and card memory, unless a number was requested."""
    per_card = min(MAX_RUNS_PER_CARD, max(1, int(memory_gib // MEMORY_PER_RUN_GIB)))
    automatic = max(1, min(per_card * gpus, cores - SPARE_CORES))
    return automatic if requested == "auto" else int(requested)


def resolve_gpus(requested: str, found: int) -> int:
    """Use every visible GPU unless told otherwise, and refuse to ask for missing ones."""
    gpus = found if requested == "auto" else int(requested)
    if gpus < 1 or gpus > found:
        raise SystemExit(f"Asked for {requested} GPUs, but {found} are visible.")
    return gpus


def check_python() -> None:
    version = sys.version_info
    # Safe tar extraction first appeared in these versions.
    if not (version >= (3, 11, 4) or (3, 10, 12) <= version < (3, 11)):
        raise SystemExit(f"Python 3.10.12+ or 3.11.4+ is required, found {sys.version.split()[0]}.")


def check_disk(workdir: Path) -> None:
    prepared = (workdir / "artifacts" / "prepare.json").exists()
    needed = MIN_DISK_GIB_PREPARED if prepared else MIN_DISK_GIB_TO_PREPARE
    workdir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(workdir).free / GIB
    log(f"Disk: {free:.0f} GiB free at {workdir}")
    if free < needed:
        raise SystemExit(f"Only {free:.0f} GiB is free at {workdir}; this step needs {needed} GiB.")


def check_tools() -> None:
    if shutil.which("git") is None:
        raise SystemExit("git is required to install stable-worldmodel.")


def check_gpus(torch, gpus: int) -> float:
    """Run real work on every card, which catches a PyTorch too old for the hardware."""
    smallest = float("inf")
    for index in range(gpus):
        name = torch.cuda.get_device_name(index)
        try:
            x = torch.randn(64, 64, device=f"cuda:{index}")
            value = float((x @ x.T).sum())
        except RuntimeError as error:
            raise SystemExit(
                f"GPU {index} ({name}) failed a test: {error}. Try a newer PyTorch image."
            ) from error
        if value != value:
            raise SystemExit(f"GPU {index} ({name}) returned an invalid result.")
        memory = torch.cuda.get_device_properties(index).total_memory / GIB
        smallest = min(smallest, memory)
        log(f"GPU {index}: {name}, {memory:.0f} GiB, passed")
    return smallest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--jobs", default="auto")
    parser.add_argument("--gpus", default="auto")
    args = parser.parse_args()

    check_python()
    check_tools()
    check_disk(args.workdir)

    import torch
    import torchvision  # noqa: F401  (stable-worldmodel imports it)

    major_minor = tuple(int(part) for part in torch.__version__.split(".")[:2])
    if major_minor < MIN_TORCH:
        raise SystemExit(f"PyTorch 2.1+ is required, found {torch.__version__}.")
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU is visible.")
    log(f"PyTorch {torch.__version__}")

    gpus = resolve_gpus(args.gpus, torch.cuda.device_count())
    memory = check_gpus(torch, gpus)
    cores = usable_cores()
    jobs = resolve_jobs(args.jobs, gpus, cores, memory)
    log(f"CPU cores usable: {cores} (the host reports {os.cpu_count()})")
    log(f"Running {jobs} at once across {gpus} GPUs")
    if jobs > cores:
        log(f"WARNING: {jobs} runs on {cores} cores will compete for CPU time.")
    print(f"{jobs} {gpus}")


if __name__ == "__main__":
    main()
