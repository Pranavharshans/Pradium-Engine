"""Workload / profile definition tests (3x3 matrix generation)."""

from __future__ import annotations

import unittest

from benchmarks.core.workloads import (
    CONTEXT_INPUT_TOKENS,
    CONTEXT_OUTPUT_TOKENS,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    PROFILES,
    PROFILE_ORDER,
    context_cases,
    matrix_layout,
    prefix_ratio_spec,
)


class TestProfiles(unittest.TestCase):
    def test_all_nine_profiles_exist(self) -> None:
        self.assertEqual(sorted(PROFILES), sorted(PROFILE_ORDER))
        self.assertEqual(len(PROFILES), 9)

    def test_profile_token_sizes(self) -> None:
        expected = {
            "SS": (128, 64),
            "SM": (128, 256),
            "SL": (128, 1024),
            "MS": (1024, 64),
            "MM": (1024, 256),
            "ML": (1024, 1024),
            "LS": (4096, 64),
            "LM": (4096, 256),
            "LL": (4096, 1024),
        }
        for name, (prompt, output) in expected.items():
            self.assertEqual(PROFILES[name].prompt_tokens, prompt, name)
            self.assertEqual(PROFILES[name].output_tokens, output, name)

    def test_input_output_class_labels(self) -> None:
        self.assertEqual(PROFILES["SS"].input_class, "S")
        self.assertEqual(PROFILES["SS"].output_class, "S")
        self.assertEqual(PROFILES["LM"].input_class, "L")
        self.assertEqual(PROFILES["LM"].output_class, "M")

    def test_purposes_are_documented(self) -> None:
        for name in PROFILE_ORDER:
            self.assertTrue(PROFILES[name].purpose.strip(), name)

    def test_matrix_layout(self) -> None:
        layout = matrix_layout()
        self.assertEqual(layout["inputs"], INPUT_TOKENS)
        self.assertEqual(layout["outputs"], OUTPUT_TOKENS)
        rows = layout["rows"]
        self.assertEqual(len(rows), 3)
        for row in rows:
            self.assertEqual(len(row["columns"]), 3)

    def test_context_cases(self) -> None:
        cases = context_cases()
        self.assertEqual([c.input_tokens for c in cases], list(CONTEXT_INPUT_TOKENS))
        for case in cases:
            self.assertEqual(case.output_tokens, CONTEXT_OUTPUT_TOKENS)

    def test_prefix_ratio_spec(self) -> None:
        self.assertEqual(prefix_ratio_spec(0.0, 1024), (0, 1024))
        self.assertEqual(prefix_ratio_spec(0.25, 1024), (256, 768))
        self.assertEqual(prefix_ratio_spec(0.5, 1024), (512, 512))
        self.assertEqual(prefix_ratio_spec(0.75, 1024), (768, 256))
        self.assertEqual(prefix_ratio_spec(0.9, 1024), (922, 102))
        shared, suffix = prefix_ratio_spec(1.0, 1024)
        self.assertEqual(shared + suffix, 1024)
        self.assertEqual(suffix, 16)
        with self.assertRaises(ValueError):
            prefix_ratio_spec(1.5, 1024)


if __name__ == "__main__":
    unittest.main()
