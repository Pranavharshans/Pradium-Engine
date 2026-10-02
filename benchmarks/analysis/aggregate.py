"""Aggregation: raw records -> session summary.

Groups every measured request, computes descriptive statistics per metric,
derives efficiency metrics, and assembles suite-specific blocks (concurrency
scaling, context scaling, prefix reuse, scheduler interference, startup,
batching, soak). Warmups are recorded but excluded from all statistics.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..core.results import EXEC_SUCCESS
from ..core.version import BENCHMARK_VERSION
from .statistics import describe, safe_ratio

#: Numeric metrics aggregated per group (raw record field -> summary name).
METRIC_FIELDS: tuple[str, ...] = (
    "ttft_ms",
    "prefill_ms",
    "prefill_tok_s",
    "decode_ms",
    "decode_tok_s",
    "tpot_ms",
    "itl_mean_ms",
    "itl_median_ms",
    "itl_p95_ms",
    "itl_p99_ms",
    "itl_max_ms",
    "itl_stdev_ms",
    "e2e_ms",
    "queue_ms",
    "scheduler_ms",
    "output_tok_s_per_request",
    "gpu_util_avg",
    "gpu_util_peak",
    "gpu_mem_util_avg",
    "gpu_power_avg",
    "gpu_power_median",
    "gpu_power_peak",
    "gpu_temperature_avg",
    "gpu_temperature_peak",
    "vram_before_mb",
    "vram_peak_mb",
    "vram_after_mb",
    "ram_before_mb",
    "ram_peak_mb",
    "ram_after_mb",
    "cpu_avg",
    "cpu_peak",
    "system_cpu_avg",
    "threads_peak",
    "aggregate_output_tok_s",
    "aggregate_prompt_tok_s",
    "aggregate_total_tok_s",
    "aggregate_decode_tok_s",
    "requests_per_second",
    "completed_requests_per_minute",
    "cache_hit_rate",
    "prefix_reused_tokens",
    "prefix_recomputed_tokens",
)

#: Metadata keys that participate in group identity.
GROUP_METADATA_KEYS: tuple[str, ...] = (
    "context_input_tokens",
    "reuse_ratio",
    "scenario",
    "phase",
    "mixed_workload",
    "role",
    "soak_iteration",
)


def group_key(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("metadata") or {}
    return {
        "suite": record.get("suite"),
        "profile": record.get("profile"),
        "concurrency": record.get("concurrency", 1),
        "batch_size": record.get("batch_size", 1),
        **{key: meta.get(key) for key in GROUP_METADATA_KEYS},
    }


def _is_measured_success(record: dict[str, Any]) -> bool:
    return not record.get("warmup", False) and record.get("execution_status") == EXEC_SUCCESS


def _derived_metrics(records: list[dict[str, Any]], stats: dict[str, Any]) -> dict[str, Any]:
    """Efficiency metrics (median-based; no single overall score)."""
    decode = stats.get("decode_tok_s", {}).get("median")
    aggregate = stats.get("aggregate_output_tok_s", {}).get("median")
    vram_peak = stats.get("vram_peak_mb", {}).get("median")
    ram_peak = stats.get("ram_peak_mb", {}).get("median")
    power_avg = stats.get("gpu_power_avg", {}).get("median")
    ttft = stats.get("ttft_ms", {}).get("median")
    concurrency = records[0].get("concurrency", 1) if records else 1
    prompt_tokens = records[0].get("prompt_tokens") if records else None

    vram_gb = (vram_peak / 1024.0) if vram_peak else None
    out: dict[str, Any] = {
        "decode_tok_s_per_gb_vram": safe_ratio(decode, vram_gb),
        "aggregate_tok_s_per_gb_vram": safe_ratio(aggregate, vram_gb),
        "decode_tok_s_per_watt": safe_ratio(decode, power_avg),
        "aggregate_tok_s_per_watt": safe_ratio(aggregate, power_avg),
        "output_tokens_per_joule": None,
        "ttft_per_1k_prompt_tokens": (
            ttft / (prompt_tokens / 1000.0) if ttft and prompt_tokens else None
        ),
        "vram_per_concurrent_request_mb": (
            vram_peak / concurrency if vram_peak else None
        ),
        "ram_per_concurrent_request_mb": (
            ram_peak / concurrency if ram_peak else None
        ),
        "max_successful_concurrency": None,
    }
    if power_avg and records:
        e2e = stats.get("e2e_ms", {}).get("median")
        output_tokens = sum(r.get("actual_output_tokens", 0) for r in records)
        if e2e:
            joules = power_avg * (e2e / 1000.0) * len(records)
            if joules > 0:
                out["output_tokens_per_joule"] = output_tokens / joules
    return out


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [r for r in records if not r.get("warmup", False)]
    successful = [r for r in measured if r.get("execution_status") == EXEC_SUCCESS]
    stats = {
        field: describe([r.get(field) for r in successful])
        for field in METRIC_FIELDS
    }
    status_counts: dict[str, int] = {}
    for record in measured:
        status = record.get("execution_status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    correctness_counts: dict[str, int] = {}
    for record in measured:
        status = record.get("correctness_status", "UNKNOWN")
        correctness_counts[status] = correctness_counts.get(status, 0) + 1
    return {
        "measured_requests": len(measured),
        "warmup_requests": len(records) - len(measured),
        "successful_requests": len(successful),
        "stats": stats,
        "derived": _derived_metrics(successful, stats),
        "execution_status_counts": status_counts,
        "correctness_status_counts": correctness_counts,
    }


def _suite_blocks(groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Suite-specific derived blocks computed from group summaries."""
    blocks: dict[str, Any] = {}

    # --- concurrency scaling ---
    conc_groups = [g for g in groups if g["key"]["suite"] == "concurrency"]
    scaling: dict[str, Any] = {}
    for group in conc_groups:
        profile = group["key"]["profile"]
        level = group["key"]["concurrency"]
        entry = scaling.setdefault(profile, {})
        decode = group["summary"]["stats"]["decode_tok_s"]["median"]
        agg_decode = group["summary"]["stats"]["aggregate_decode_tok_s"]["median"]
        entry[str(level)] = {
            "decode_tok_s_per_request": decode,
            "aggregate_decode_tok_s": agg_decode,
            "ttft_ms": group["summary"]["stats"]["ttft_ms"]["median"],
            "tpot_ms": group["summary"]["stats"]["tpot_ms"]["median"],
        }
    for profile, levels in scaling.items():
        baseline = levels.get("1", {})
        base_decode = baseline.get("decode_tok_s_per_request")
        base_ttft = baseline.get("ttft_ms")
        base_tpot = baseline.get("tpot_ms")
        for level, entry in levels.items():
            n = int(level)
            entry["scaling_efficiency"] = (
                entry["aggregate_decode_tok_s"] / (n * base_decode)
                if entry["aggregate_decode_tok_s"] and base_decode and n > 0
                else None
            )
            entry["per_request_throughput_degradation"] = (
                entry["decode_tok_s_per_request"] / base_decode - 1.0
                if entry["decode_tok_s_per_request"] and base_decode
                else None
            )
            entry["ttft_degradation_vs_c1"] = (
                entry["ttft_ms"] / base_ttft - 1.0
                if entry["ttft_ms"] and base_ttft
                else None
            )
            entry["tpot_degradation_vs_c1"] = (
                entry["tpot_ms"] / base_tpot - 1.0
                if entry["tpot_ms"] and base_tpot
                else None
            )
    if scaling:
        blocks["concurrency_scaling"] = scaling

    # --- context scaling ---
    ctx_groups = [g for g in groups if g["key"]["suite"] == "context"]
    if ctx_groups:
        blocks["context_scaling"] = [
            {
                "input_tokens": g["key"]["context_input_tokens"],
                "profile": g["key"]["profile"],
                "ttft_ms": g["summary"]["stats"]["ttft_ms"]["median"],
                "prefill_tok_s": g["summary"]["stats"]["prefill_tok_s"]["median"],
                "decode_tok_s": g["summary"]["stats"]["decode_tok_s"]["median"],
                "tpot_ms": g["summary"]["stats"]["tpot_ms"]["median"],
                "vram_peak_mb": g["summary"]["stats"]["vram_peak_mb"]["median"],
                "kv_memory_mb": g["summary"]["stats"].get("kv_memory_mb", {}).get("median"),
                "execution_status_counts": g["summary"]["execution_status_counts"],
            }
            for g in sorted(
                ctx_groups, key=lambda g: g["key"]["context_input_tokens"] or 0
            )
        ]

    # --- prefix reuse ---
    prefix_groups = [g for g in groups if g["key"]["suite"] == "prefix-cache"]
    if prefix_groups:
        baselines: dict[tuple, dict[str, Any]] = {}
        for g in prefix_groups:
            if (g["key"]["reuse_ratio"] or 0) == 0.0:
                baselines[g["key"]["scenario"]] = g["summary"]["stats"]
        blocks["prefix_reuse"] = [
            {
                "ratio": g["key"]["reuse_ratio"],
                "scenario": g["key"]["scenario"],
                "ttft_ms": g["summary"]["stats"]["ttft_ms"]["median"],
                "prefill_ms": g["summary"]["stats"]["prefill_ms"]["median"],
                "decode_tok_s": g["summary"]["stats"]["decode_tok_s"]["median"],
                "reused_tokens": g["summary"]["stats"]["prefix_reused_tokens"]["median"],
                "cache_hit_rate": g["summary"]["stats"]["cache_hit_rate"]["median"],
                "ttft_speedup_vs_0": _speedup(
                    baselines.get(g["key"]["scenario"], {}).get("ttft_ms", {}).get("median"),
                    g["summary"]["stats"]["ttft_ms"]["median"],
                ),
                "prefill_speedup_vs_0": _speedup(
                    baselines.get(g["key"]["scenario"], {}).get("prefill_ms", {}).get("median"),
                    g["summary"]["stats"]["prefill_ms"]["median"],
                ),
                "execution_status_counts": g["summary"]["execution_status_counts"],
            }
            for g in sorted(
                prefix_groups,
                key=lambda g: (g["key"]["reuse_ratio"] or 0, str(g["key"]["scenario"])),
            )
        ]

    # --- max successful concurrency (per suite/profile) ---
    max_conc: dict[str, int] = {}
    for group in groups:
        if group["summary"]["successful_requests"] > 0:
            key = f'{group["key"]["suite"]}:{group["key"]["profile"]}'
            max_conc[key] = max(max_conc.get(key, 0), group["key"]["concurrency"] or 1)
    blocks["max_successful_concurrency"] = max_conc

    return blocks


def _speedup(baseline: float | None, current: float | None) -> float | None:
    if not baseline or not current:
        return None
    return baseline / current


def build_summary(
    session_dir: Path | str,
    suite_extras: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    capabilities: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the session summary from raw data (regenerable at any time)."""
    session_dir = Path(session_dir)
    records: list[dict[str, Any]] = []
    raw_path = session_dir / "raw.jsonl"
    if raw_path.exists():
        with raw_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    grouped: dict[str, list[dict[str, Any]]] = {}
    keys: dict[str, dict[str, Any]] = {}
    for record in records:
        key = group_key(record)
        key_str = json.dumps(key, sort_keys=True)
        grouped.setdefault(key_str, []).append(record)
        keys[key_str] = key

    groups = []
    for key_str, group_records in grouped.items():
        groups.append(
            {
                "key": keys[key_str],
                "summary": summarize_group(group_records),
            }
        )
    groups.sort(
        key=lambda g: (
            str(g["key"]["suite"]),
            str(g["key"]["profile"]),
            g["key"]["concurrency"] or 0,
            g["key"]["context_input_tokens"] or 0,
            g["key"]["reuse_ratio"] if g["key"]["reuse_ratio"] is not None else -1,
            str(g["key"]["scenario"]),
            str(g["key"]["phase"]),
        )
    )
    # Flatten group summaries into schema-friendly shape.
    flat_groups = []
    for group in groups:
        entry = dict(group["key"])
        entry.update(group["summary"])
        flat_groups.append(entry)

    failures: list[dict[str, Any]] = []
    failures_path = session_dir / "failures.jsonl"
    if failures_path.exists():
        with failures_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    failures.append(json.loads(line))

    sample = records[0] if records else {}
    blocks = _suite_blocks(groups)

    unsupported: list[str] = []
    for group in groups:
        for status in group["summary"]["execution_status_counts"]:
            if status in ("OOM", "UNSUPPORTED", "FAILED", "TIMEOUT"):
                unsupported.append(
                    f'{group["key"]["suite"]}/{group["key"]["profile"]}'
                    f'@C{group["key"]["concurrency"]}: {status}'
                )

    return {
        "benchmark_version": sample.get("benchmark_version", BENCHMARK_VERSION),
        "session_id": sample.get("session_id", session_dir.name),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": sample.get("runtime"),
        "runtime_version": sample.get("runtime_version"),
        "runtime_config": sample.get("runtime_config", {}),
        "model": sample.get("model"),
        "model_revision": sample.get("model_revision"),
        "tokenizer": sample.get("tokenizer"),
        "tokenizer_revision": sample.get("tokenizer_revision"),
        "benchmark_mode": sample.get("benchmark_mode"),
        "run_policy": sample.get("run_policy"),
        "performance_valid": bool(records) and all(r.get("performance_valid", False) for r in records if not r.get("warmup", False)),
        "environment": environment or {},
        "config": config or {},
        "capabilities": capabilities or {},
        "groups": flat_groups,
        "suite_blocks": blocks,
        "failures": failures,
        "unsupported_measurements": sorted(set(unsupported)),
        "raw_record_count": len(records),
    }
