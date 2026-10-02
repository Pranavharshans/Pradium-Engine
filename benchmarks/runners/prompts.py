"""Prompt provisioning for suite runners.

Materialized prompts (exact token counts, frozen hashes) are cached per
tokenizer and converted into :class:`BenchmarkRequest` objects. Every request
carries its ``prompt_hash`` and ``input_ids_hash`` so raw records can prove
identical workloads across runtimes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..corpus.materialize import (
    MaterializedPrefixGroup,
    MaterializedPrompt,
    Tokenizer,
    get_tokenizer,
    materialize_prefix_group,
    materialize_prompt,
)
from ..core.request import BenchmarkRequest, new_request_id
from ..core.workloads import PROFILES, PROFILE_ORDER


@dataclass
class PromptProvider:
    """Caches materialized prompts and builds benchmark requests."""

    tokenizer: Tokenizer
    _cache: dict[str, MaterializedPrompt] = field(default_factory=dict)
    _groups: dict[str, MaterializedPrefixGroup] = field(default_factory=dict)

    @classmethod
    def from_spec(cls, spec: str = "simple") -> "PromptProvider":
        return cls(tokenizer=get_tokenizer(spec))

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name

    @property
    def tokenizer_revision(self) -> str:
        return self.tokenizer.revision

    def profile_prompt(self, name: str) -> MaterializedPrompt:
        if name not in PROFILES:
            raise KeyError(f"unknown profile: {name}")
        cache_key = f"matrix:{name}"
        if cache_key not in self._cache:
            offset = PROFILE_ORDER.index(name)
            self._cache[cache_key] = materialize_prompt(
                self.tokenizer, name, PROFILES[name].prompt_tokens, source_offset=offset
            )
        return self._cache[cache_key]

    def context_prompt(self, input_tokens: int) -> MaterializedPrompt:
        cache_key = f"context:{input_tokens}"
        if cache_key not in self._cache:
            self._cache[cache_key] = materialize_prompt(
                self.tokenizer, f"CTX-{input_tokens}", input_tokens
            )
        return self._cache[cache_key]

    def prefix_group(
        self,
        ratio: float,
        total_input_tokens: int,
        request_count: int,
        unique_suffix_tokens_at_full_reuse: int = 16,
    ) -> MaterializedPrefixGroup:
        cache_key = f"prefix:{ratio}:{total_input_tokens}:{request_count}"
        if cache_key not in self._groups:
            self._groups[cache_key] = materialize_prefix_group(
                self.tokenizer,
                ratio,
                total_input_tokens,
                request_count,
                unique_suffix_tokens_at_full_reuse,
            )
        return self._groups[cache_key]

    def request(
        self,
        prompt: MaterializedPrompt,
        output_tokens: int,
        *,
        run_index: int = 0,
        warmup: bool = False,
        profile: str | None = None,
        input_class: str | None = None,
        output_class: str | None = None,
        request_id: str | None = None,
        prefix_id: str | None = None,
        prefix_token_count: int = 0,
        concurrency_group: str | None = None,
        batch_size: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkRequest:
        meta = dict(metadata or {})
        meta.setdefault("prompt_hash", prompt.prompt_sha256)
        meta.setdefault("prompt_name", prompt.name)
        meta.setdefault("tokenizer_revision", prompt.tokenizer_revision)
        return BenchmarkRequest(
            request_id=request_id or new_request_id(),
            input_ids=list(prompt.input_ids),
            prompt_token_count=prompt.prompt_tokens,
            max_new_tokens=output_tokens,
            profile=profile,
            input_class=input_class,
            output_class=output_class,
            prefix_id=prefix_id,
            prefix_token_count=prefix_token_count,
            concurrency_group=concurrency_group,
            batch_size=batch_size,
            warmup=warmup,
            run_index=run_index,
            metadata=meta,
        )

    def profile_request(
        self, name: str, run_index: int = 0, warmup: bool = False, **kwargs: Any
    ) -> BenchmarkRequest:
        profile = PROFILES[name]
        return self.request(
            self.profile_prompt(name),
            profile.output_tokens,
            run_index=run_index,
            warmup=warmup,
            profile=name,
            input_class=profile.input_class,
            output_class=profile.output_class,
            **kwargs,
        )
