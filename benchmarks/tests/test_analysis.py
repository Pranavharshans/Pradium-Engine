"""Analysis tests: statistics, aggregation, summary schema, report, plots."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.analysis.aggregate import build_summary, group_key, summarize_group
from benchmarks.analysis.plots import generate_plots
from benchmarks.analysis.report import render_report
from benchmarks.analysis.statistics import describe
from benchmarks.core.schema import load_schema, validate_schema
from benchmarks.core.version import BENCHMARK_VERSION

SCHEMAS_DIR = Path(__file__).resolve().parent.parent / "schemas"


def fake_record(**overrides):
    record = {
        "benchmark_version": BENCHMARK_VERSION,
        "session_id": "s",
        "timestamp": "t",
        "suite": "matrix",
        "runtime": "mock",
        "profile": "SS",
        "input_class": "S",
        "output_class": "S",
        "run_index": 0,
        "warmup": False,
        "benchmark_mode": "validation",
        "run_policy": "quick",
        "performance_valid": False,
        "prompt_tokens": 128,
        "requested_output_tokens": 64,
        "actual_output_tokens": 64,
        "concurrency": 1,
        "batch_size": 1,
        "execution_status": "SUCCESS",
        "correctness_status": "PASS",
        "metric_provenance": {},
        "metadata": {},
        "ttft_ms": 10.0,
        "decode_tok_s": 50.0,
        "tpot_ms": 20.0,
        "e2e_ms": 100.0,
    }
    record.update(overrides)
    return record


class TestStatistics(unittest.TestCase):
    def test_describe(self) -> None:
        stats = describe([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["count"], 4)
        self.assertEqual(stats["mean"], 2.5)
        self.assertEqual(stats["median"], 2.5)
        self.assertEqual(stats["min"], 1.0)
        self.assertEqual(stats["max"], 4.0)
        self.assertEqual(stats["p50"], 2.5)
        self.assertIsNotNone(stats["cv"])

    def test_describe_empty_and_none(self) -> None:
        stats = describe([])
        self.assertEqual(stats["count"], 0)
        self.assertIsNone(stats["median"])
        stats = describe([None, None, 5.0])
        self.assertEqual(stats["count"], 1)

    def test_headline_is_median_not_min(self) -> None:
        stats = describe([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(stats["median"], 2.5)
        self.assertNotEqual(stats["median"], stats["min"])


class TestAggregation(unittest.TestCase):
    def _session(self, tmp: str) -> Path:
        session = Path(tmp) / "session"
        (session / "traces").mkdir(parents=True)
        records = [
            fake_record(run_index=0, warmup=True, ttft_ms=999.0),
            fake_record(run_index=0, ttft_ms=10.0, decode_tok_s=50.0),
            fake_record(run_index=1, ttft_ms=12.0, decode_tok_s=55.0),
            fake_record(
                profile="LL",
                prompt_tokens=4096,
                requested_output_tokens=1024,
                actual_output_tokens=1024,
                ttft_ms=40.0,
                execution_status="OOM",
                correctness_status="SKIPPED",
                error="oom",
            ),
        ]
        with (session / "raw.jsonl").open("w") as fh:
            for record in records:
                fh.write(json.dumps(record) + "\n")
        with (session / "failures.jsonl").open("w") as fh:
            fh.write(
                json.dumps(
                    {
                        "benchmark_version": BENCHMARK_VERSION,
                        "session_id": "s",
                        "timestamp": "t",
                        "suite": "matrix",
                        "runtime": "mock",
                        "model": None,
                        "profile": "LL",
                        "concurrency": 1,
                        "input_tokens": 4096,
                        "requested_output_tokens": 1024,
                        "status": "OOM",
                        "error": "oom",
                    }
                )
                + "\n"
            )
        return session

    def test_warmups_excluded_from_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_summary(self._session(tmp))
            ss = next(
                g
                for g in summary["groups"]
                if g["profile"] == "SS" and g["suite"] == "matrix"
            )
            self.assertEqual(ss["warmup_requests"], 1)
            self.assertEqual(ss["measured_requests"], 2)
            self.assertEqual(ss["stats"]["ttft_ms"]["median"], 11.0)
            self.assertEqual(ss["stats"]["ttft_ms"]["max"], 12.0)

    def test_failures_and_unsupported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_summary(self._session(tmp))
            self.assertEqual(len(summary["failures"]), 1)
            self.assertTrue(summary["unsupported_measurements"])
            ll = next(g for g in summary["groups"] if g["profile"] == "LL")
            self.assertEqual(ll["execution_status_counts"], {"OOM": 1})

    def test_summary_matches_schema(self) -> None:
        schema = load_schema(SCHEMAS_DIR / "summary.schema.json")
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_summary(self._session(tmp))
            validate_schema(summary, schema)

    def test_group_key_metadata(self) -> None:
        record = fake_record(metadata={"context_input_tokens": 4096, "phase": "cold"})
        key = group_key(record)
        self.assertEqual(key["context_input_tokens"], 4096)
        self.assertEqual(key["phase"], "cold")

    def test_correctness_vs_performance_separated(self) -> None:
        record = fake_record(correctness_status="WARNING", ttft_ms=5.0)
        summary = summarize_group([record])
        self.assertEqual(summary["correctness_status_counts"], {"WARNING": 1})
        self.assertEqual(summary["execution_status_counts"], {"SUCCESS": 1})
        self.assertIsNotNone(summary["stats"]["ttft_ms"]["median"])


class TestReportAndPlots(unittest.TestCase):
    def test_report_sections(self) -> None:
        summary = {
            "benchmark_version": BENCHMARK_VERSION,
            "session_id": "s",
            "runtime": "mock",
            "runtime_version": "1.0.0",
            "model": "mock-model",
            "model_revision": "r1",
            "tokenizer": "pradium-simple-v1",
            "tokenizer_revision": "t1",
            "benchmark_mode": "validation",
            "run_policy": "quick",
            "performance_valid": False,
            "raw_record_count": 3,
            "environment": {
                "os": {"system": "Linux", "release": "1"},
                "python": {"version": "3.14"},
                "gpu": {"name": "TestGPU", "uuid": "u", "driver": "d", "cuda": "c"},
                "cpu": {"model": "cpu", "logical_cores": 8},
                "system_ram_mb": 1024,
                "pradium_git_commit": "abc",
            },
            "config": {
                "matrix": {"inputs": {"S": 128}, "outputs": {"S": 64}},
                "telemetry": {"interval_ms": 100},
            },
            "capabilities": {"supports_streaming": "SUPPORTED"},
            "groups": [
                {
                    "suite": "matrix",
                    "profile": "SS",
                    "concurrency": 1,
                    "batch_size": 1,
                    "measured_requests": 2,
                    "warmup_requests": 1,
                    "stats": {
                        "ttft_ms": _stats(10.0),
                        "decode_tok_s": _stats(50.0),
                        "tpot_ms": _stats(20.0),
                        "e2e_ms": _stats(100.0),
                    },
                    "derived": {},
                    "execution_status_counts": {"SUCCESS": 2},
                }
            ],
            "suite_blocks": {},
            "suite_extras": {
                "startup": {"runtime_init_ms": 12.0, "model_load_ms": 30.0}
            },
            "failures": [],
            "unsupported_measurements": [],
        }
        report = render_report(summary)
        for section in (
            "## Session",
            "## Environment (facts)",
            "## Runtime capabilities (declared)",
            "## 3x3 workload matrix",
            "TTFT (ms)",
            "Sustained decode",
            "## Startup (facts)",
            "## Failures and unsupported measurements",
            "no winner is declared",
            "TTFT = first_token_ts − submit_ts",
        ):
            self.assertIn(section, report)

    def test_plots_generated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session = Path(tmp)
            (session / "traces").mkdir()
            summary = {
                "groups": [
                    {
                        "suite": "matrix",
                        "profile": "SS",
                        "concurrency": 1,
                        "batch_size": 1,
                        "stats": {"ttft_ms": _stats(10.0), "decode_tok_s": _stats(50.0)},
                    },
                    {
                        "suite": "matrix",
                        "profile": "MM",
                        "concurrency": 1,
                        "batch_size": 1,
                        "stats": {"ttft_ms": _stats(20.0), "decode_tok_s": _stats(40.0)},
                    },
                ],
                "suite_blocks": {
                    "concurrency_scaling": {
                        "SS": {
                            "1": {
                                "aggregate_decode_tok_s": 50.0,
                                "decode_tok_s_per_request": 50.0,
                                "ttft_ms": 10.0,
                                "tpot_ms": 20.0,
                            },
                            "4": {
                                "aggregate_decode_tok_s": 150.0,
                                "decode_tok_s_per_request": 40.0,
                                "ttft_ms": 15.0,
                                "tpot_ms": 25.0,
                            },
                        }
                    },
                    "context_scaling": [
                        {
                            "input_tokens": 128,
                            "ttft_ms": 5.0,
                            "prefill_tok_s": 100.0,
                            "decode_tok_s": 50.0,
                            "vram_peak_mb": 100.0,
                        }
                    ],
                    "prefix_reuse": [
                        {
                            "ratio": 0.0,
                            "scenario": "same_session",
                            "ttft_ms": 10.0,
                            "prefill_ms": 8.0,
                        },
                        {
                            "ratio": 1.0,
                            "scenario": "same_session",
                            "ttft_ms": 4.0,
                            "prefill_ms": 2.0,
                        },
                    ],
                },
                "suite_extras": {"startup": {"runtime_init_ms": 12.0}},
            }
            written = generate_plots(session, summary)
            names = {p.name for p in written}
            self.assertIn("matrix_ttft_by_workload.svg", names)
            self.assertIn("concurrency_aggregate_decode_tok_s.svg", names)
            self.assertIn("context_ttft_ms.svg", names)
            self.assertIn("prefix_ttft_ms.svg", names)
            self.assertIn("startup_breakdown.svg", names)
            for path in written:
                content = path.read_text()
                self.assertIn("<svg", content)


def _stats(value: float) -> dict:
    return {
        "count": 1,
        "mean": value,
        "median": value,
        "min": value,
        "max": value,
        "stdev": 0.0,
        "p50": value,
        "p90": value,
        "p95": value,
        "p99": value,
        "cv": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
