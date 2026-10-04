#!/usr/bin/env python3
"""Check the Python, GPU, Java, and Forge pieces needed by Quest jobs.

This intentionally does not install anything. Run it on a GPU allocation with
``--require-gpu`` after bootstrapping the virtual environment.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path


def _arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forge-dir",
        type=Path,
        default=None,
        help="Forge checkout; defaults to FORGE_DIR when set",
    )
    parser.add_argument(
        "--require-jar",
        action="store_true",
        help="fail unless a Forge uber-jar has already been built",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail unless CUDA is available and a small GPU matmul succeeds",
    )
    return parser


def _print(label: str, value: object) -> None:
    print(f"[quest-check] {label}: {value}")


def main() -> int:
    args = _arg_parser().parse_args()
    problems: list[str] = []

    _print("python", sys.executable)
    _print("python-version", platform.python_version())
    _print("host", platform.platform())
    _print("slurm-job", os.environ.get("SLURM_JOB_ID", "not in a Slurm job"))
    _print("cuda-visible-devices", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))

    java = shutil.which("java")
    _print("java", java or "missing")
    if args.require_jar and java is None:
        problems.append("java is not on PATH")

    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - exercised by broken envs
        torch = None
        problems.append(f"cannot import torch: {exc}")

    if torch is not None:
        _print("torch", torch.__version__)
        _print("torch-cuda-runtime", torch.version.cuda or "none")
        _print("torch-hip-runtime", torch.version.hip or "none")
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        _print("cuda-available", available)
        _print("gpu-count", device_count)
        if available:
            _print("gpu-0", torch.cuda.get_device_name(0))

        if args.require_gpu:
            if not available:
                problems.append("torch.cuda.is_available() is false")
            elif device_count < 1:
                problems.append("PyTorch reports no CUDA devices")
            else:
                try:
                    device = torch.device("cuda:0")
                    left = torch.randn((256, 256), device=device)
                    right = torch.randn((256, 256), device=device)
                    result = left @ right
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(result).all().item()):
                        problems.append("float32 GPU matmul produced non-finite values")
                    else:
                        _print("float32-matmul", "ok")

                    # The serving path normally uses autocast. Keep this a
                    # separate check so a GPU with float32 support but no
                    # usable BF16 path is diagnosed before a long run.
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        mixed = left @ right
                    torch.cuda.synchronize()
                    if not bool(torch.isfinite(mixed.float()).all().item()):
                        problems.append("BF16 autocast matmul produced non-finite values")
                    else:
                        _print("bf16-autocast-matmul", "ok")
                except Exception as exc:  # pragma: no cover - hardware-specific
                    problems.append(f"GPU smoke test failed: {exc}")

    for module_name in ("grpc", "numpy", "safetensors"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - exercised by broken envs
            problems.append(f"cannot import {module_name}: {exc}")
        else:
            _print(f"python-{module_name}", "ok")

    forge_text = str(args.forge_dir) if args.forge_dir else os.environ.get("FORGE_DIR", "")
    if forge_text:
        forge_dir = Path(forge_text).expanduser().resolve()
        _print("forge-dir", forge_dir)
        if not forge_dir.is_dir():
            problems.append(f"Forge checkout does not exist: {forge_dir}")
        else:
            jars = sorted(
                (forge_dir / "forge-gui-desktop" / "target").glob(
                    "*-jar-with-dependencies.jar"
                ),
                key=lambda path: path.stat().st_mtime,
            )
            _print("forge-jars", len(jars))
            if jars:
                _print("forge-jar", jars[-1])
            elif args.require_jar:
                problems.append(f"no Forge uber-jar under {forge_dir}/forge-gui-desktop/target")
    elif args.require_jar:
        problems.append("FORGE_DIR or --forge-dir is required")

    if problems:
        for problem in problems:
            print(f"[quest-check] ERROR: {problem}", file=sys.stderr)
        return 1

    print("[quest-check] environment looks usable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
