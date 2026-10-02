"""Generic runtime adapter conformance suite.

Every runtime adapter (mock today; ExLlamaV3 / SGLang / vLLM / Pradium later)
must pass the same behavioral requirements. New adapters reuse this suite by
subclassing :class:`AdapterConformanceMixin` in their own test module.
"""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.core.capabilities import CAPABILITY_NAMES, CapabilityStatus
from benchmarks.core.request import BenchmarkRequest, new_request_id


@dataclass
class ConformanceCheck:
    name: str
    passed: bool
    detail: str = ""


def make_conformance_request(
    prompt_tokens: int = 32, output_tokens: int = 8, **kwargs
) -> BenchmarkRequest:
    request_id = kwargs.pop("request_id", None) or new_request_id()
    return BenchmarkRequest(
        request_id=request_id,
        input_ids=list(range(100, 100 + prompt_tokens)),
        prompt_token_count=prompt_tokens,
        max_new_tokens=output_tokens,
        profile=kwargs.pop("profile", None),
        **kwargs,
    )


def run_conformance_suite(adapter) -> list[ConformanceCheck]:
    """Run behavioral conformance checks against an initialized+loaded adapter."""
    checks: list[ConformanceCheck] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append(ConformanceCheck(name, ok, detail))

    # --- capability declaration ---
    try:
        caps = adapter.get_capabilities()
        names_ok = all(hasattr(caps, name) for name in CAPABILITY_NAMES)
        statuses_ok = all(
            isinstance(caps.status(name), CapabilityStatus) for name in CAPABILITY_NAMES
        )
        record("capability_declaration_valid", names_ok and statuses_ok)
    except Exception as exc:  # noqa: BLE001
        record("capability_declaration_valid", False, str(exc))

    # --- runtime info ---
    try:
        info = adapter.get_runtime_info()
        record("runtime_info_available", isinstance(info, dict) and bool(info.get("name")))
    except Exception as exc:  # noqa: BLE001
        record("runtime_info_available", False, str(exc))

    # --- non-streaming generate ---
    request = make_conformance_request(32, 8)
    try:
        result = adapter.generate(request)
        record("generate_returns_tokens", len(result.token_ids) > 0)
        record(
            "generate_token_count_correct",
            len(result.token_ids) == request.max_new_tokens
            or not adapter.get_capabilities().is_usable("supports_fixed_decode_length"),
            f"got {len(result.token_ids)}, requested {request.max_new_tokens}",
        )
    except Exception as exc:  # noqa: BLE001
        record("generate_returns_tokens", False, str(exc))

    # --- streaming ---
    request = make_conformance_request(32, 6, request_id=new_request_id())
    try:
        tokens = list(adapter.generate_stream(request))
        record("stream_produces_tokens", len(tokens) >= 1)
        emitted = [t for t in tokens if not t.done]
        record(
            "stream_maintains_order",
            [t.index for t in emitted] == list(range(len(emitted))),
        )
        record("stream_final_done_event", bool(tokens) and tokens[-1].done)
        record(
            "stream_final_has_finish_reason",
            bool(tokens) and bool(tokens[-1].finish_reason),
        )
    except Exception as exc:  # noqa: BLE001
        record("stream_produces_tokens", False, str(exc))

    # --- request id preservation / determinism ---
    request = make_conformance_request(16, 4, request_id="conformance-fixed-id")
    try:
        first = [t.token_id for t in adapter.generate_stream(request) if not t.done]
        second = [t.token_id for t in adapter.generate_stream(request) if not t.done]
        record("outputs_deterministic", first == second, f"{first} vs {second}")
    except Exception as exc:  # noqa: BLE001
        record("outputs_deterministic", False, str(exc))

    # --- actual token count ---
    request = make_conformance_request(16, 5)
    try:
        result = adapter.generate(request)
        record(
            "actual_token_count_correct",
            len(result.token_ids) == 5,
            f"got {len(result.token_ids)}",
        )
    except Exception as exc:  # noqa: BLE001
        record("actual_token_count_correct", False, str(exc))

    return checks
