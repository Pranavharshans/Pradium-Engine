"""Frozen prompt corpus and deterministic materialization.

Source text and tokenized prompts are separated: the corpus sources are
human-readable documents committed to Git, while prompt token sequences are
materialized deterministically from them with a specific tokenizer.

Materialization guarantees EXACT token counts (128, 1024, 4096, ...): prompts
are built on the token-id stream and truncated to precisely the requested
length. The materialized input IDs are what runtimes should consume.

The frozen corpus for PRADIUM-RUNTIME-BENCH-v1 is validated by SHA-256 hashes
stored in ``corpus/manifest.json``. Any mutation of a source file or of the
tokenizer vocabulary fails validation; official runs must not proceed on
validation failure.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from ..core.request import hash_token_ids
from ..core.version import BENCHMARK_VERSION

CORPUS_DIR = Path(__file__).resolve().parent
SOURCES_DIR = CORPUS_DIR / "sources"
MANIFEST_PATH = CORPUS_DIR / "manifest.json"
TOKENIZERS_DIR = CORPUS_DIR / "tokenizers"
MATERIALIZED_DIR = CORPUS_DIR / "materialized"

#: Canonical source order used to build long token streams.
SOURCE_ORDER: tuple[str, ...] = (
    "mixed.md",
    "technical.md",
    "conversation.md",
    "code.md",
    "structured.md",
)

#: Marker vocabularies are bounded so the tokenizer vocabulary stays closed.
MAX_SEGMENT_INDEX = 1023
MAX_REQUEST_INDEX = 63
#: Markers reserved per prefix request so suffix streams never collide.
REQUEST_MARKERS_PER_STREAM = 16

SEGMENT_MARKER = "\n\n[segment {index}]\n\n"
REQUEST_MARKER = "\n\n[request {index}]\n\n"

#: Piece splitting: leading whitespace attaches to the following non-space run;
#: trailing whitespace runs become their own piece. ``"".join(pieces) == text``.
PIECE_RE = re.compile(r"\s*\S+|\s+")

UNK_TOKEN = "<unk>"


class CorpusValidationError(RuntimeError):
    """Raised when the frozen corpus does not match its manifest hashes."""


def tokenize_pieces(text: str) -> list[str]:
    """Split text into lossless tokenizer pieces."""
    return PIECE_RE.findall(text)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tokenizer(Protocol):
    """Minimal tokenizer protocol used by the materializer."""

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class SimpleTokenizer:
    """Deterministic lossless piece tokenizer with a frozen vocabulary.

    ``encode`` raises on pieces outside the vocabulary so corpus mutation is
    caught instead of silently becoming ``<unk>``. ``decode(encode(text))``
    reproduces ``text`` exactly for known text.
    """

    def __init__(self, vocab: dict[str, int]) -> None:
        if UNK_TOKEN not in vocab:
            raise ValueError(f"vocabulary must contain {UNK_TOKEN!r}")
        self._vocab = dict(vocab)
        self._inv = {idx: piece for piece, idx in self._vocab.items()}

    @property
    def name(self) -> str:
        return "pradium-simple-v1"

    @property
    def revision(self) -> str:
        return vocab_revision(self._vocab)

    def __len__(self) -> int:
        return len(self._vocab)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in tokenize_pieces(text):
            idx = self._vocab.get(piece)
            if idx is None:
                raise CorpusValidationError(
                    f"piece not in frozen vocabulary (corpus mutated?): {piece!r}"
                )
            ids.append(idx)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        parts: list[str] = []
        for idx in ids:
            piece = self._inv.get(int(idx))
            parts.append(piece if piece is not None else f"<unk:{int(idx)}>")
        return "".join(parts)


class HFTokenizerWrapper:
    """Wrapper for a HuggingFace tokenizer (used only when one is provided).

    Requires the ``transformers`` package; the benchmark core never depends on
    it. The tokenizer revision is recorded so materializations are reproducible.
    """

    def __init__(self, name_or_path: str, revision: str | None = None) -> None:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers is required for hf: tokenizer specs but is not "
                "installed; use --tokenizer simple or install the tokenizer"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(name_or_path, revision=revision)
        self._name = name_or_path
        self._revision = revision or str(
            getattr(self._tokenizer, "init_kwargs", {}).get("revision", "default")
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def revision(self) -> str:
        return self._revision

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def decode(self, ids: Iterable[int]) -> str:
        return str(self._tokenizer.decode(list(ids), skip_special_tokens=True))


def vocab_revision(vocab: dict[str, int]) -> str:
    """Deterministic revision id for a vocabulary (content hash)."""
    payload = json.dumps(vocab, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- Vocabulary construction ----------------------------------------------


def _all_marker_texts() -> list[str]:
    markers = [SEGMENT_MARKER.format(index=i) for i in range(MAX_SEGMENT_INDEX + 1)]
    markers += [REQUEST_MARKER.format(index=i) for i in range(MAX_REQUEST_INDEX + 1)]
    return markers


def build_vocab() -> dict[str, int]:
    """Build the frozen vocabulary from sources + marker families, deterministically."""
    pieces: set[str] = set()
    for name in SOURCE_ORDER:
        text = (SOURCES_DIR / name).read_text(encoding="utf-8")
        pieces.update(tokenize_pieces(text))
    for marker in _all_marker_texts():
        pieces.update(tokenize_pieces(marker))
    ordered = sorted(pieces)
    vocab = {UNK_TOKEN: 0}
    for i, piece in enumerate(ordered, start=1):
        vocab[piece] = i
    return vocab


def vocab_path(tokenizer_id: str = "pradium-simple-v1") -> Path:
    return TOKENIZERS_DIR / tokenizer_id / "vocab.json"


def write_vocab(force: bool = False) -> Path:
    """Write the frozen vocabulary file (idempotent without ``force``)."""
    path = vocab_path()
    if path.exists() and not force:
        return path
    vocab = build_vocab()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(vocab, sort_keys=True, ensure_ascii=True, indent=0) + "\n",
        encoding="utf-8",
    )
    return path


def load_simple_tokenizer() -> SimpleTokenizer:
    path = vocab_path()
    if not path.exists():
        raise CorpusValidationError(
            f"frozen vocabulary missing: {path}. Run: python -m benchmarks corpus build-vocab"
        )
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return SimpleTokenizer(vocab)


def get_tokenizer(spec: str = "simple") -> Tokenizer:
    """Resolve a tokenizer spec: ``simple`` or ``hf:<name>[@<revision>]``."""
    if spec in ("simple", "pradium-simple-v1"):
        return load_simple_tokenizer()
    if spec.startswith("hf:"):
        rest = spec[3:]
        if "@" in rest:
            name, revision = rest.rsplit("@", 1)
        else:
            name, revision = rest, None
        return HFTokenizerWrapper(name, revision)
    raise ValueError(f"unknown tokenizer spec: {spec!r}")


# --- Stream materialization ------------------------------------------------


def load_sources() -> list[tuple[str, str]]:
    """Return ``(name, text)`` for the frozen sources in canonical order."""
    out: list[tuple[str, str]] = []
    for name in SOURCE_ORDER:
        path = SOURCES_DIR / name
        if not path.exists():
            raise CorpusValidationError(f"corpus source missing: {path}")
        out.append((name, path.read_text(encoding="utf-8")))
    return out


def build_token_stream(
    tokenizer: Tokenizer,
    target_tokens: int,
    source_offset: int = 0,
    marker: str = SEGMENT_MARKER,
    marker_offset: int = 0,
) -> list[int]:
    """Compose a deterministic token stream of EXACTLY ``target_tokens`` ids.

    Sources cycle in canonical order; each is preceded by an index marker so
    long contexts resemble multi-document inputs rather than repeated tokens.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    sources = [(name, tokenizer.encode(text)) for name, text in load_sources()]
    ids: list[int] = []
    i = 0
    while len(ids) < target_tokens:
        marker_index = marker_offset + i
        max_index = MAX_SEGMENT_INDEX if marker is SEGMENT_MARKER else MAX_REQUEST_INDEX
        if marker_index > max_index:
            raise CorpusValidationError(f"marker index exceeds bound: {marker_index}")
        name, src_ids = sources[(i + source_offset) % len(sources)]
        ids.extend(tokenizer.encode(marker.format(index=marker_index)))
        ids.extend(src_ids)
        i += 1
    return ids[:target_tokens]


@dataclass(frozen=True)
class MaterializedPrompt:
    """One exact-length materialized prompt with its frozen hashes."""

    name: str
    prompt_tokens: int
    input_ids: tuple[int, ...]
    prompt_text: str
    prompt_sha256: str
    input_ids_sha256: str
    tokenizer_name: str
    tokenizer_revision: str
    benchmark_version: str = BENCHMARK_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "prompt_tokens": self.prompt_tokens,
            "prompt_text": self.prompt_text,
            "prompt_sha256": self.prompt_sha256,
            "input_ids_sha256": self.input_ids_sha256,
            "tokenizer": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "benchmark_version": self.benchmark_version,
        }


def materialize_prompt(
    tokenizer: Tokenizer, name: str, target_tokens: int, source_offset: int = 0
) -> MaterializedPrompt:
    """Materialize one exact-length prompt from the frozen sources."""
    ids = build_token_stream(tokenizer, target_tokens, source_offset=source_offset)
    if len(ids) != target_tokens:
        raise CorpusValidationError(
            f"materialization error: {name} has {len(ids)} tokens, expected {target_tokens}"
        )
    text = tokenizer.decode(ids)
    return MaterializedPrompt(
        name=name,
        prompt_tokens=target_tokens,
        input_ids=tuple(ids),
        prompt_text=text,
        prompt_sha256=sha256_text(text),
        input_ids_sha256=hash_token_ids(list(ids)),
        tokenizer_name=tokenizer.name,
        tokenizer_revision=tokenizer.revision,
    )


@dataclass(frozen=True)
class MaterializedPrefixGroup:
    """One prefix-reuse group: shared prefix + unique per-request suffixes."""

    ratio: float
    total_input_tokens: int
    shared_prefix_tokens: int
    unique_suffix_tokens: int
    prefix_id: str | None
    shared_prefix_ids: tuple[int, ...]
    requests: tuple[MaterializedPrompt, ...]

    def expected_reuse_ratio(self) -> float:
        return self.shared_prefix_tokens / self.total_input_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio,
            "total_input_tokens": self.total_input_tokens,
            "shared_prefix_tokens": self.shared_prefix_tokens,
            "unique_suffix_tokens": self.unique_suffix_tokens,
            "expected_reuse_ratio": self.expected_reuse_ratio(),
            "prefix_id": self.prefix_id,
            "requests": [
                {
                    "name": r.name,
                    "input_ids_sha256": r.input_ids_sha256,
                    "prompt_sha256": r.prompt_sha256,
                }
                for r in self.requests
            ],
        }


def materialize_prefix_group(
    tokenizer: Tokenizer,
    ratio: float,
    total_input_tokens: int,
    request_count: int,
    unique_suffix_tokens_at_full_reuse: int = 16,
) -> MaterializedPrefixGroup:
    """Build a frozen prefix-reuse group for one reuse ratio.

    The reuse ratio is defined on token IDs: ``shared_prefix_tokens / total``.
    At ratio ~1.0 a large identical prefix is paired with a small unique suffix
    per request (never exact duplicate full prompts).
    """
    from ..core.workloads import prefix_ratio_spec

    shared_n, suffix_n = prefix_ratio_spec(
        ratio, total_input_tokens, unique_suffix_tokens_at_full_reuse
    )
    prefix_id = f"prefix-{ratio:.2f}-{shared_n}" if shared_n > 0 else None
    shared_ids = (
        tuple(build_token_stream(tokenizer, shared_n, marker=SEGMENT_MARKER))
        if shared_n > 0
        else ()
    )
    requests: list[MaterializedPrompt] = []
    suffix_first_ids: set[int] = set()
    for i in range(request_count):
        if suffix_n > 0:
            marker_offset = i * REQUEST_MARKERS_PER_STREAM
            if marker_offset + REQUEST_MARKERS_PER_STREAM - 1 > MAX_REQUEST_INDEX:
                raise CorpusValidationError("request marker range exhausted")
            suffix_ids = build_token_stream(
                tokenizer,
                suffix_n,
                source_offset=i + 1,
                marker=REQUEST_MARKER,
                marker_offset=marker_offset,
            )
        else:
            suffix_ids = []
        # Diverge immediately after the declared shared prefix, including 0%.
        # Marker tokenization can otherwise introduce an accidental common prefix.
        if suffix_ids:
            distinct = next((j for j, token in enumerate(suffix_ids)
                             if token not in suffix_first_ids), None)
            if distinct is None:
                raise CorpusValidationError("cannot construct distinct suffix starts")
            suffix_ids[0], suffix_ids[distinct] = suffix_ids[distinct], suffix_ids[0]
            suffix_first_ids.add(suffix_ids[0])
        ids = list(shared_ids) + suffix_ids
        if len(ids) != total_input_tokens:
            raise CorpusValidationError(
                f"prefix group {ratio}: request {i} has {len(ids)} tokens, "
                f"expected {total_input_tokens}"
            )
        text = tokenizer.decode(ids)
        requests.append(
            MaterializedPrompt(
                name=f"prefix-{ratio:.2f}-req-{i}",
                prompt_tokens=total_input_tokens,
                input_ids=tuple(ids),
                prompt_text=text,
                prompt_sha256=sha256_text(text),
                input_ids_sha256=hash_token_ids(ids),
                tokenizer_name=tokenizer.name,
                tokenizer_revision=tokenizer.revision,
            )
        )
    return MaterializedPrefixGroup(
        ratio=ratio,
        total_input_tokens=total_input_tokens,
        shared_prefix_tokens=shared_n,
        unique_suffix_tokens=suffix_n,
        prefix_id=prefix_id,
        shared_prefix_ids=shared_ids,
        requests=tuple(requests),
    )


# --- Manifest --------------------------------------------------------------


def source_hashes() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for name in SOURCE_ORDER:
        path = SOURCES_DIR / name
        text = path.read_text(encoding="utf-8")
        out[name] = {
            "sha256": sha256_text(text),
            "bytes": path.stat().st_size,
            "pieces": len(tokenize_pieces(text)),
        }
    return out


def compute_materialization_hashes(tokenizer: Tokenizer) -> dict[str, Any]:
    """Hash every frozen materialization (matrix, context, prefix groups)."""
    from ..core.workloads import (
        CONTEXT_INPUT_TOKENS,
        PREFIX_REQUESTS_PER_RATIO,
        PREFIX_REUSE_RATIOS,
        PREFIX_TOTAL_INPUT_TOKENS,
        PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE,
        PROFILES,
        PROFILE_ORDER,
    )

    matrix: dict[str, Any] = {}
    for offset, name in enumerate(PROFILE_ORDER):
        profile = PROFILES[name]
        prompt = materialize_prompt(tokenizer, name, profile.prompt_tokens, source_offset=offset)
        matrix[name] = {
            "prompt_tokens": profile.prompt_tokens,
            "output_tokens": profile.output_tokens,
            "prompt_sha256": prompt.prompt_sha256,
            "input_ids_sha256": prompt.input_ids_sha256,
        }

    context: dict[str, Any] = {}
    for offset, length in enumerate(CONTEXT_INPUT_TOKENS):
        prompt = materialize_prompt(tokenizer, f"CTX-{length}", length, source_offset=offset)
        context[str(length)] = {
            "prompt_tokens": length,
            "prompt_sha256": prompt.prompt_sha256,
            "input_ids_sha256": prompt.input_ids_sha256,
        }

    prefix: dict[str, Any] = {}
    for ratio in PREFIX_REUSE_RATIOS:
        group = materialize_prefix_group(
            tokenizer,
            ratio,
            PREFIX_TOTAL_INPUT_TOKENS,
            PREFIX_REQUESTS_PER_RATIO,
            PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE,
        )
        prefix[f"{ratio:.2f}"] = group.to_dict()

    return {"matrix": matrix, "context": context, "prefix": prefix}


def build_manifest(tokenizer: Tokenizer | None = None) -> dict[str, Any]:
    """Build the full corpus manifest (sources + tokenizer + materializations)."""
    tokenizer = tokenizer or load_simple_tokenizer()
    from datetime import datetime, timezone

    return {
        "benchmark_version": BENCHMARK_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_hashes(),
        "tokenizer": {
            "id": tokenizer.name,
            "type": "simple" if isinstance(tokenizer, SimpleTokenizer) else "external",
            "vocab_file": str(vocab_path().relative_to(CORPUS_DIR)),
            "vocab_sha256": sha256_file(vocab_path()),
            "revision": tokenizer.revision,
            "vocabulary_size": len(
                json.loads(vocab_path().read_text(encoding="utf-8"))
            ),
        },
        "materializations": compute_materialization_hashes(tokenizer),
    }


def write_manifest() -> Path:
    manifest = build_manifest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return MANIFEST_PATH


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise CorpusValidationError(f"corpus manifest missing: {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def validate_corpus() -> dict[str, Any]:
    """Validate the frozen corpus against its manifest. Raises on any mutation.

    Checks: benchmark version, vocabulary hash, every source hash, and every
    materialization hash rebuilt from source (which also proves determinism and
    exact token counts).
    """
    manifest = load_manifest()
    problems: list[str] = []

    if manifest.get("benchmark_version") != BENCHMARK_VERSION:
        problems.append(
            f"manifest benchmark_version {manifest.get('benchmark_version')!r} "
            f"!= {BENCHMARK_VERSION!r}"
        )

    # Vocabulary / tokenizer
    tokenizer_block = manifest.get("tokenizer", {})
    if not vocab_path().exists():
        problems.append(f"vocabulary file missing: {vocab_path()}")
    else:
        actual_vocab_sha = sha256_file(vocab_path())
        if actual_vocab_sha != tokenizer_block.get("vocab_sha256"):
            problems.append("tokenizer vocabulary hash mismatch (tokenizer mutated)")
        tokenizer: Tokenizer = load_simple_tokenizer()
        if tokenizer.revision != tokenizer_block.get("revision"):
            problems.append("tokenizer revision mismatch")

        # Sources
        for name, expected in manifest.get("sources", {}).items():
            path = SOURCES_DIR / name
            if not path.exists():
                problems.append(f"source missing: {name}")
                continue
            actual = sha256_text(path.read_text(encoding="utf-8"))
            if actual != expected.get("sha256"):
                problems.append(f"source hash mismatch: {name}")

        # Materializations (rebuilt deterministically)
        rebuilt = compute_materialization_hashes(tokenizer)
        frozen = manifest.get("materializations", {})
        for section in ("matrix", "context"):
            for key, expected in frozen.get(section, {}).items():
                actual = rebuilt.get(section, {}).get(key)
                if actual is None:
                    problems.append(f"materialization missing: {section}/{key}")
                elif actual.get("input_ids_sha256") != expected.get("input_ids_sha256"):
                    problems.append(f"input_ids hash mismatch: {section}/{key}")
                elif actual.get("prompt_tokens") != expected.get("prompt_tokens"):
                    problems.append(f"token count mismatch: {section}/{key}")
        for key, expected in frozen.get("prefix", {}).items():
            actual = rebuilt.get("prefix", {}).get(key)
            if actual is None:
                problems.append(f"prefix group missing: {key}")
            else:
                exp_ids = [r["input_ids_sha256"] for r in expected.get("requests", [])]
                act_ids = [r["input_ids_sha256"] for r in actual.get("requests", [])]
                if exp_ids != act_ids:
                    problems.append(f"prefix group hash mismatch: {key}")

    if problems:
        raise CorpusValidationError(
            "frozen corpus validation FAILED:\n  - " + "\n  - ".join(problems)
        )
    return {
        "status": "OK",
        "benchmark_version": BENCHMARK_VERSION,
        "sources": len(manifest.get("sources", {})),
        "tokenizer": tokenizer_block.get("id"),
        "tokenizer_revision": tokenizer_block.get("revision"),
    }


# --- Optional on-disk materialization --------------------------------------


def materialize_to_disk(tokenizer_spec: str = "simple") -> Path:
    """Write materialized prompts to ``corpus/materialized/<tokenizer>/``.

    Input IDs for every frozen workload are stored as JSON so runtimes can
    consume identical pre-tokenized inputs. Regenerating is deterministic.
    """
    tokenizer = get_tokenizer(tokenizer_spec)
    from ..core.workloads import (
        CONTEXT_INPUT_TOKENS,
        PREFIX_REQUESTS_PER_RATIO,
        PREFIX_REUSE_RATIOS,
        PREFIX_TOTAL_INPUT_TOKENS,
        PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE,
        PROFILES,
        PROFILE_ORDER,
    )

    root = MATERIALIZED_DIR / tokenizer.name
    if root.exists():
        import shutil

        shutil.rmtree(root)

    for offset, name in enumerate(PROFILE_ORDER):
        profile = PROFILES[name]
        prompt = materialize_prompt(tokenizer, name, profile.prompt_tokens, source_offset=offset)
        _write_prompt(root / "matrix" / f"{name}.json", prompt, profile.output_tokens)

    for offset, length in enumerate(CONTEXT_INPUT_TOKENS):
        prompt = materialize_prompt(tokenizer, f"CTX-{length}", length, source_offset=offset)
        _write_prompt(root / "context" / f"{length}.json", prompt, 256)

    for ratio in PREFIX_REUSE_RATIOS:
        group = materialize_prefix_group(
            tokenizer,
            ratio,
            PREFIX_TOTAL_INPUT_TOKENS,
            PREFIX_REQUESTS_PER_RATIO,
            PREFIX_UNIQUE_SUFFIX_TOKENS_AT_FULL_REUSE,
        )
        for prompt in group.requests:
            path = root / "prefix" / f"{ratio:.2f}" / f"{prompt.name}.json"
            _write_prompt(path, prompt, 256, extra=group.to_dict())

    side_manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "tokenizer": {
            "id": tokenizer.name,
            "revision": tokenizer.revision,
        },
        "materializations": compute_materialization_hashes(tokenizer),
    }
    (root / "manifest.json").write_text(
        json.dumps(side_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return root


def _write_prompt(
    path: Path,
    prompt: MaterializedPrompt,
    output_tokens: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = prompt.to_dict()
    payload["output_tokens"] = output_tokens
    payload["input_ids"] = list(prompt.input_ids)
    if extra:
        payload["group"] = {k: v for k, v in extra.items() if k != "requests"}
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


# --- CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.corpus.materialize",
        description="Frozen corpus vocabulary, materialization and validation",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-vocab", help="write the frozen tokenizer vocabulary")
    build = sub.add_parser("build", help="write vocabulary + corpus manifest")
    build.add_argument("--force", action="store_true")
    sub.add_parser("validate", help="validate the frozen corpus against manifest.json")
    mat = sub.add_parser("materialize", help="write exact-length prompts to disk")
    mat.add_argument("--tokenizer", default="simple")
    args = parser.parse_args(argv)

    if args.command == "build-vocab":
        path = write_vocab(force=getattr(args, "force", False))
        print(f"vocabulary written: {path}")
    elif args.command == "build":
        write_vocab(force=getattr(args, "force", False))
        path = write_manifest()
        print(f"corpus manifest written: {path}")
    elif args.command == "validate":
        result = validate_corpus()
        print(f"corpus validation: {result['status']} ({result['sources']} sources)")
    elif args.command == "materialize":
        root = materialize_to_disk(args.tokenizer)
        print(f"materialized prompts written under: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
