"""Environment manifest collection.

Every session records ``environment.json`` so results can be interpreted and
reproduced later. Hardware is detected, never hardcoded.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from .version import BENCHMARK_VERSION, verify_version_file

#: Environment variables recorded when set (framework configuration, not secrets).
RECORDED_ENV_VARS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_CACHE",
    "TOKENIZERS_PARALLELISM",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NVIDIA_VISIBLE_DEVICES",
)


def _run_git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def git_commit() -> str | None:
    return _run_git("rev-parse", "HEAD")


def git_branch() -> str | None:
    return _run_git("rev-parse", "--abbrev-ref", "HEAD")


def git_dirty() -> bool | None:
    out = _run_git("status", "--porcelain")
    if out is None:
        return None
    return bool(out.strip())


def _cpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "model": platform.processor() or None,
        "physical_cores": os.cpu_count(),
        "logical_cores": os.cpu_count(),
    }
    try:
        import psutil  # type: ignore

        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
    except Exception:
        pass
    if info["model"] in (None, "") and platform.system() == "Darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if out.returncode == 0:
                info["model"] = out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def _system_ram_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().total / (1024 * 1024), 1)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 * 1024), 1)
    except (ValueError, OSError, AttributeError):
        return None


def _gpu_info() -> dict[str, Any]:
    try:
        from ..telemetry.gpu import probe_gpus  # local import: optional stack

        return {"gpus": probe_gpus(), "telemetry_backend": None}
    except Exception as exc:  # pragma: no cover - defensive
        return {"gpus": [], "error": str(exc)}


def _optional_package_version(module_name: str) -> str | None:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def collect_environment(
    cli_args: list[str] | None = None,
    telemetry_interval_ms: int | None = None,
    runtime_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the environment manifest for one benchmark session."""
    verify_version_file()
    gpu_block = _gpu_info()
    gpus = gpu_block.get("gpus") or []
    primary = gpus[0] if gpus else {}
    env_vars = {
        name: os.environ[name]
        for name in RECORDED_ENV_VARS
        if name in os.environ
    }
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "pradium_git_commit": git_commit(),
        "pradium_git_branch": git_branch(),
        "pradium_git_dirty": git_dirty(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "packages": {
            "torch": _optional_package_version("torch"),
            "psutil": _optional_package_version("psutil"),
            "pynvml": _optional_package_version("pynvml"),
        },
        "gpu": {
            "name": primary.get("name"),
            "uuid": primary.get("uuid"),
            "vram_total_mb": primary.get("vram_total_mb"),
            "driver": primary.get("driver"),
            "cuda": primary.get("cuda"),
            "all_gpus": gpus,
        },
        "cpu": _cpu_info(),
        "system_ram_mb": _system_ram_mb(),
        "runtime": runtime_info or {},
        "cli_arguments": list(cli_args or []),
        "environment_variables": env_vars,
        "telemetry_interval_ms": telemetry_interval_ms,
    }
