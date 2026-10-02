"""Frozen prompt corpus package.

Exports are lazy so ``python -m benchmarks.corpus.materialize`` runs without
runpy import warnings.
"""

from typing import Any

__all__ = [
    "CorpusValidationError",
    "MaterializedPrompt",
    "SimpleTokenizer",
    "get_tokenizer",
    "load_manifest",
    "materialize_prompt",
    "validate_corpus",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import materialize

        return getattr(materialize, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
