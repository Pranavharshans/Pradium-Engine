"""Benchmark session management.

A session owns one results directory::

    results/<session-id>/
        environment.json
        config.json
        corpus_manifest.json
        raw.jsonl
        summary.json
        report.md
        failures.jsonl
        traces/
        plots/

Session directories are never overwritten: a colliding session id is treated
as an error (the id already contains a random suffix, so collisions indicate a
bug or clock manipulation).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .results import FailureRecord, RawResult, TokenTrace
from .version import BENCHMARK_VERSION

RESULTS_DIRNAME = "results"


class SessionError(RuntimeError):
    """Raised for invalid session lifecycle operations."""


def make_session_id(runtime: str, mode: str, now: datetime | None = None) -> str:
    """Build a readable unique session id.

    Format: ``YYYYMMDD-HHMMSS_<runtime>_<mode>_<6-hex-suffix>``.
    """
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    safe_runtime = "".join(c if c.isalnum() or c in "-_" else "-" for c in runtime)
    safe_mode = "".join(c if c.isalnum() or c in "-_" else "-" for c in mode)
    return f"{stamp}_{safe_runtime}_{safe_mode}_{uuid.uuid4().hex[:6]}"


class Session:
    """One benchmark session: append-only raw data plus final artifacts."""

    def __init__(
        self,
        session_id: str,
        results_root: Path | str,
        runtime: str,
        benchmark_mode: str,
        run_policy: str,
    ) -> None:
        self.session_id = session_id
        self.results_root = Path(results_root)
        self.directory = self.results_root / session_id
        if self.directory.exists():
            raise SessionError(
                f"refusing to overwrite existing session directory: {self.directory}"
            )
        self.runtime = runtime
        self.benchmark_mode = benchmark_mode
        self.run_policy = run_policy
        self.directory.mkdir(parents=True, exist_ok=False)
        (self.directory / "traces").mkdir()
        (self.directory / "plots").mkdir()
        self._raw_path = self.directory / "raw.jsonl"
        self._failures_path = self.directory / "failures.jsonl"
        self._closed = False
        self._raw_count = 0

    # --- artifacts written up front ---------------------------------------

    def write_environment(self, manifest: dict[str, Any]) -> None:
        self._write_json("environment.json", manifest)

    def write_config(self, config: dict[str, Any]) -> None:
        self._write_json("config.json", config)

    def write_corpus_manifest(self, manifest: dict[str, Any]) -> None:
        self._write_json("corpus_manifest.json", manifest)

    # --- append-only raw data ---------------------------------------------

    def append_raw(self, record: RawResult) -> None:
        if self._closed:
            raise SessionError("session already closed")
        with self._raw_path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")
        self._raw_count += 1

    def append_failure(self, record: FailureRecord) -> None:
        with self._failures_path.open("a", encoding="utf-8") as fh:
            fh.write(record.to_json() + "\n")

    def write_trace(self, trace: TokenTrace) -> Path:
        path = self.directory / "traces" / f"{trace.request_id}.json"
        path.write_text(
            json.dumps(trace.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return path

    # --- final artifacts ---------------------------------------------------

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json("summary.json", summary)

    def write_report(self, markdown: str) -> None:
        (self.directory / "report.md").write_text(markdown, encoding="utf-8")

    def close(self) -> None:
        self._closed = True

    # --- helpers ----------------------------------------------------------

    def _write_json(self, name: str, payload: dict[str, Any]) -> None:
        path = self.directory / name
        if path.exists():
            raise SessionError(f"refusing to overwrite existing artifact: {path}")
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )

    @property
    def raw_count(self) -> int:
        return self._raw_count

    @property
    def raw_path(self) -> Path:
        return self._raw_path

    @property
    def failures_path(self) -> Path:
        return self._failures_path

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "benchmark_version": BENCHMARK_VERSION,
            "runtime": self.runtime,
            "benchmark_mode": self.benchmark_mode,
            "run_policy": self.run_policy,
            "directory": str(self.directory),
            "raw_records": self._raw_count,
        }


def default_results_root() -> Path:
    """Default results root: ``benchmarks/results`` (gitignored)."""
    return Path(__file__).resolve().parent.parent / RESULTS_DIRNAME


def read_raw_records(directory: Path | str) -> list[dict[str, Any]]:
    """Read ``raw.jsonl`` back (analysis never needs a rerun)."""
    path = Path(directory) / "raw.jsonl"
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
