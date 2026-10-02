"""End-to-end CLI tests: full pipeline over the mock runtime."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.cli import main


class TestCliEndToEnd(unittest.TestCase):
    def test_matrix_quick_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "run",
                    "matrix",
                    "--runtime",
                    "mock",
                    "--quick",
                    "--time-scale",
                    "0.0",
                    "--results-dir",
                    tmp,
                    "--telemetry-interval",
                    "20",
                ]
            )
            self.assertEqual(code, 0)
            sessions = list(Path(tmp).iterdir())
            self.assertEqual(len(sessions), 1)
            session = sessions[0]
            for artifact in (
                "environment.json",
                "config.json",
                "corpus_manifest.json",
                "raw.jsonl",
                "summary.json",
                "report.md",
            ):
                self.assertTrue((session / artifact).exists(), artifact)
            self.assertTrue((session / "traces").is_dir())

            raw_lines = (session / "raw.jsonl").read_text().strip().splitlines()
            self.assertTrue(raw_lines)
            for line in raw_lines:
                record = json.loads(line)
                self.assertEqual(record["benchmark_version"], "PRADIUM-RUNTIME-BENCH-v1")
                self.assertEqual(record["runtime"], "mock")
                self.assertEqual(record["benchmark_mode"], "validation")
                self.assertFalse(record["performance_valid"])

            summary = json.loads((session / "summary.json").read_text())
            self.assertEqual(summary["runtime"], "mock")
            self.assertFalse(summary["performance_valid"])
            report = (session / "report.md").read_text()
            self.assertIn("no winner is declared", report)

    def test_no_session_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "run",
                "matrix",
                "--runtime",
                "mock",
                "--quick",
                "--time-scale",
                "0.0",
                "--results-dir",
                tmp,
            ]
            self.assertEqual(main(argv), 0)
            self.assertEqual(main(argv), 0)
            self.assertEqual(len(list(Path(tmp).iterdir())), 2)

    def test_report_regenerates_from_raw(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            main(
                [
                    "run",
                    "matrix",
                    "--runtime",
                    "mock",
                    "--quick",
                    "--time-scale",
                    "0.0",
                    "--results-dir",
                    tmp,
                ]
            )
            session = next(Path(tmp).iterdir())
            (session / "report.md").unlink()
            (session / "summary.json").unlink()
            code = main(["report", session.name, "--results-dir", tmp])
            self.assertEqual(code, 0)
            self.assertTrue((session / "report.md").exists())
            self.assertTrue((session / "summary.json").exists())

    def test_corpus_commands(self) -> None:
        self.assertEqual(main(["corpus", "validate"]), 0)

    def test_version_and_suites(self) -> None:
        self.assertEqual(main(["version"]), 0)
        self.assertEqual(main(["suites"]), 0)
        self.assertEqual(main(["adapters"]), 0)

    def test_soak_excluded_from_quick(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = main(
                [
                    "run",
                    "soak",
                    "--runtime",
                    "mock",
                    "--quick",
                    "--results-dir",
                    tmp,
                ]
            )
            self.assertEqual(code, 2)

    def test_unknown_profile_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main(
                    [
                        "run",
                        "matrix",
                        "--runtime",
                        "mock",
                        "--quick",
                        "--profiles",
                        "ZZ",
                        "--results-dir",
                        tmp,
                    ]
                )


if __name__ == "__main__":
    unittest.main()
