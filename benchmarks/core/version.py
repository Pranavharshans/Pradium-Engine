"""Benchmark version constant and version-bump policy."""

from __future__ import annotations

from pathlib import Path

#: Frozen benchmark identifier. Stored in every result file.
BENCHMARK_VERSION = "PRADIUM-RUNTIME-BENCH-v1"

#: Frozen benchmark components. Any change to one of these requires a version
#: bump of :data:`BENCHMARK_VERSION` (and of the ``VERSION`` file).
FROZEN_COMPONENTS: tuple[str, ...] = (
    "prompt corpus",
    "token-length definitions",
    "workload definitions",
    "output lengths",
    "timing definitions",
    "warmup count",
    "measured-run count",
    "statistical methodology",
    "concurrency methodology",
    "cache test methodology",
)

#: File mirroring :data:`BENCHMARK_VERSION`; kept in sync and validated at runtime.
VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def read_version_file() -> str:
    """Return the benchmark version recorded in the ``VERSION`` file."""
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def verify_version_file() -> str:
    """Return the benchmark version, failing if the ``VERSION`` file disagrees.

    Raises:
        RuntimeError: if the file is missing or inconsistent with the constant.
    """
    if not VERSION_FILE.exists():
        raise RuntimeError(f"benchmark VERSION file missing: {VERSION_FILE}")
    file_version = read_version_file()
    if file_version != BENCHMARK_VERSION:
        raise RuntimeError(
            f"benchmark version mismatch: constant={BENCHMARK_VERSION!r} "
            f"file={file_version!r}"
        )
    return BENCHMARK_VERSION
