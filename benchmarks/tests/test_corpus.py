"""Frozen corpus: hashing, mutation detection, exact token counts."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmarks.corpus import materialize as mat
from benchmarks.corpus.materialize import (
    CORPUS_DIR,
    MANIFEST_PATH,
    SOURCES_DIR,
    CorpusValidationError,
    build_token_stream,
    load_manifest,
    load_simple_tokenizer,
    materialize_prefix_group,
    materialize_prompt,
    sha256_file,
    sha256_text,
    tokenize_pieces,
    validate_corpus,
)
from benchmarks.core.workloads import CONTEXT_INPUT_TOKENS, PROFILES, PROFILE_ORDER


class TestPieceTokenizer(unittest.TestCase):
    def test_lossless_round_trip(self) -> None:
        text = "hello world\n\n  indented\ttext! (with) punctuation, indeed."
        pieces = tokenize_pieces(text)
        self.assertEqual("".join(pieces), text)

    def test_pieces_attach_leading_whitespace(self) -> None:
        self.assertEqual(tokenize_pieces("a b"), ["a", " b"])


class TestFrozenCorpus(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = load_simple_tokenizer()

    def test_manifest_exists_and_validates(self) -> None:
        self.assertTrue(MANIFEST_PATH.exists())
        result = validate_corpus()
        self.assertEqual(result["status"], "OK")

    def test_sources_committed(self) -> None:
        manifest = load_manifest()
        expected = {"conversation.md", "technical.md", "code.md", "structured.md", "mixed.md"}
        self.assertEqual(set(manifest["sources"]), expected)
        for name in expected:
            self.assertTrue((SOURCES_DIR / name).exists())

    def test_exact_matrix_token_counts(self) -> None:
        for offset, name in enumerate(PROFILE_ORDER):
            profile = PROFILES[name]
            prompt = materialize_prompt(
                self.tokenizer, name, profile.prompt_tokens, source_offset=offset
            )
            self.assertEqual(len(prompt.input_ids), profile.prompt_tokens, name)
            self.assertEqual(prompt.prompt_tokens, profile.prompt_tokens, name)

    def test_exact_context_token_counts(self) -> None:
        for offset, length in enumerate(CONTEXT_INPUT_TOKENS):
            prompt = materialize_prompt(
                self.tokenizer, f"CTX-{length}", length, source_offset=offset
            )
            self.assertEqual(len(prompt.input_ids), length)

    def test_deterministic_materialization(self) -> None:
        a = materialize_prompt(self.tokenizer, "MM", 1024, source_offset=4)
        b = materialize_prompt(self.tokenizer, "MM", 1024, source_offset=4)
        self.assertEqual(a.input_ids_sha256, b.input_ids_sha256)
        self.assertEqual(a.prompt_sha256, b.prompt_sha256)

    def test_round_trip_decode(self) -> None:
        prompt = materialize_prompt(self.tokenizer, "SS", 128, source_offset=0)
        self.assertEqual(self.tokenizer.decode(prompt.input_ids), prompt.prompt_text)

    def test_prompts_are_not_degenerate(self) -> None:
        prompt = materialize_prompt(self.tokenizer, "LL", 4096, source_offset=8)
        self.assertGreater(len(set(prompt.input_ids)), 500)

    def test_prefix_groups(self) -> None:
        for ratio in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
            group = materialize_prefix_group(self.tokenizer, ratio, 1024, 4, 16)
            seen: set[str] = set()
            for request in group.requests:
                self.assertEqual(len(request.input_ids), 1024)
                if group.shared_prefix_tokens:
                    self.assertEqual(
                        request.input_ids[: group.shared_prefix_tokens],
                        group.shared_prefix_ids,
                    )
                self.assertNotIn(request.input_ids_sha256, seen)
                seen.add(request.input_ids_sha256)
            self.assertAlmostEqual(
                group.expected_reuse_ratio(),
                group.shared_prefix_tokens / group.total_input_tokens,
            )
        # ~100% reuse: large shared prefix, small unique suffix
        full = materialize_prefix_group(self.tokenizer, 1.0, 1024, 4, 16)
        self.assertEqual(full.shared_prefix_tokens, 1008)
        self.assertEqual(full.unique_suffix_tokens, 16)

    def test_source_mutation_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "sources"
            shutil.copytree(SOURCES_DIR, fake)
            target = fake / "conversation.md"
            target.write_text(
                target.read_text(encoding="utf-8") + "\nmutated\n", encoding="utf-8"
            )
            with mock.patch.object(mat, "SOURCES_DIR", fake):
                with self.assertRaises(CorpusValidationError) as ctx:
                    validate_corpus()
                message = str(ctx.exception)
                self.assertTrue(
                    "hash mismatch" in message or "corpus mutated" in message,
                    message,
                )

    def test_manifest_hashes_are_stable(self) -> None:
        manifest = load_manifest()
        for name, block in manifest["sources"].items():
            actual = sha256_text((SOURCES_DIR / name).read_text(encoding="utf-8"))
            self.assertEqual(actual, block["sha256"], name)
        self.assertEqual(sha256_file(CORPUS_DIR / "tokenizers" / "pradium-simple-v1" / "vocab.json"),
                         manifest["tokenizer"]["vocab_sha256"])

    def test_unknown_piece_rejected(self) -> None:
        with self.assertRaises(CorpusValidationError):
            self.tokenizer.encode("zzz_unknown_token_zzz")

    def test_stream_exactness_by_truncation(self) -> None:
        ids = build_token_stream(self.tokenizer, 333)
        self.assertEqual(len(ids), 333)
        ids2 = build_token_stream(self.tokenizer, 334)
        self.assertEqual(ids2[:333], ids)


if __name__ == "__main__":
    unittest.main()
