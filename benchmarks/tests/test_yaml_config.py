"""Configuration and minimal-YAML loader tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.core.config import (
    CONFIGS_DIR,
    ConfigError,
    load_config,
)
from benchmarks.core.yaml_min import YamlError, load_file, loads


class TestYamlMin(unittest.TestCase):
    def test_scalars(self) -> None:
        doc = loads(
            "a: 1\nb: 2.5\nc: true\nd: false\ne: null\nf: hello\n"
            "g: 'quoted: text'\nh: \"double # not comment\"\n"
        )
        self.assertEqual(
            doc,
            {
                "a": 1,
                "b": 2.5,
                "c": True,
                "d": False,
                "e": None,
                "f": "hello",
                "g": "quoted: text",
                "h": "double # not comment",
            },
        )

    def test_nested_mapping_and_lists(self) -> None:
        doc = loads(
            "top:\n  inner:\n    x: 1\n  items: [1, 2, 3]\nplain: -5\n"
            "block_list:\n  - alpha\n  - beta\n"
        )
        self.assertEqual(
            doc,
            {
                "top": {"inner": {"x": 1}, "items": [1, 2, 3]},
                "plain": -5,
                "block_list": ["alpha", "beta"],
            },
        )

    def test_comments_and_blanks(self) -> None:
        doc = loads("# leading comment\n\na: 1  # trailing\n\nb: 2\n")
        self.assertEqual(doc, {"a": 1, "b": 2})

    def test_empty_flow_list(self) -> None:
        self.assertEqual(loads("a: []\n"), {"a": []})

    def test_bad_syntax(self) -> None:
        with self.assertRaises(YamlError):
            loads("no colon here\n")
        with self.assertRaises(YamlError):
            loads("a: 1\na: 2\n")
        with self.assertRaises(YamlError):
            loads("\tx: 1\n")


class TestConfigLoading(unittest.TestCase):
    def test_load_and_validate(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.matrix.inputs["S"], 128)
        self.assertEqual(cfg.matrix.inputs["M"], 1024)
        self.assertEqual(cfg.matrix.inputs["L"], 4096)
        self.assertEqual(cfg.matrix.outputs["S"], 64)
        self.assertEqual(cfg.matrix.outputs["M"], 256)
        self.assertEqual(cfg.matrix.outputs["L"], 1024)
        self.assertEqual(
            cfg.matrix.profiles, ("SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL")
        )
        self.assertEqual(cfg.concurrency.levels, (1, 2, 4, 8))
        self.assertEqual(cfg.batching.batch_sizes, (1, 2, 4, 8))
        self.assertEqual(cfg.context.input_tokens[-1], 32768)
        self.assertEqual(cfg.context.output_tokens, 256)

    def test_run_policies(self) -> None:
        cfg = load_config()
        quick = cfg.mode_policy("quick")
        self.assertEqual((quick.warmups, quick.measured_runs), (1, 2))
        self.assertFalse(quick.performance_valid)
        official = cfg.mode_policy("official")
        self.assertEqual((official.warmups, official.measured_runs), (3, 10))
        self.assertTrue(official.performance_valid)
        self.assertEqual(
            official.profiles, ("SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL")
        )
        validation = cfg.mode_policy("validation")
        self.assertFalse(validation.performance_valid)

    def test_frozen_values_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in CONFIGS_DIR.glob("*.yaml"):
                (root / name.name).write_text(
                    name.read_text(encoding="utf-8"), encoding="utf-8"
                )
            matrix = root / "matrix.yaml"
            matrix.write_text(
                matrix.read_text(encoding="utf-8").replace("long: 4096", "long: 8192"),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigError):
                load_config(root)

    def test_missing_config_dir(self) -> None:
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/configs")

    def test_load_file_helper(self) -> None:
        doc = load_file(CONFIGS_DIR / "concurrency.yaml")
        self.assertEqual(doc["levels"], [1, 2, 4, 8])


if __name__ == "__main__":
    unittest.main()
