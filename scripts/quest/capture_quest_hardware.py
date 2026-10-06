#!/usr/bin/env python3
"""Capture best-effort host and GPU metadata for a Quest benchmark phase.

The benchmark should not fail merely because one diagnostic command is absent,
so command failures are recorded in the JSON and the script still exits zero.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _command(*args: str) -> dict[str, object]:
    executable = shutil.which(args[0])
    if executable is None:
        return {"command": list(args), "available": False, "error": "not found"}
    try:
        result = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": list(args), "available": True, "error": str(exc)}
    return {
        "command": list(args),
        "available": True,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _slurm_environment() -> dict[str, str]:
    names = (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NODELIST",
        "SLURM_JOB_CPUS_PER_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_MEM_PER_NODE",
        "SLURM_GPUS",
        "SLURM_GPUS_ON_NODE",
        "SLURM_JOB_PARTITION",
        "SLURM_JOB_ACCOUNT",
    )
    return {name: os.environ[name] for name in names if os.environ.get(name)}


def _gpu_inventory() -> list[dict[str, str]]:
    command = shutil.which("nvidia-smi")
    if command is None:
        return []
    query = "index,name,uuid,driver_version,memory.total"
    try:
        result = subprocess.run(
            [command, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    fields = [field for field in query.split(",")]
    rows = []
    for raw in result.stdout.splitlines():
        values = [value.strip() for value in raw.split(",")]
        if len(values) != len(fields):
            continue
        rows.append(dict(zip(fields, values)))
    return rows


def capture(phase: str) -> dict[str, object]:
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = []
    return {
        "phase": phase,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "slurm": _slurm_environment(),
        "gpus": _gpu_inventory(),
        "commands": {
            "scontrol_job": _command(
                "scontrol",
                "show",
                "job",
                os.environ.get("SLURM_JOB_ID", ""),
            )
            if os.environ.get("SLURM_JOB_ID")
            else {"available": False, "error": "SLURM_JOB_ID is unset"},
            "lscpu": _command("lscpu"),
            "uptime": _command("uptime"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("start", "end"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(capture(args.phase), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
