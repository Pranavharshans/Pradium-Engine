"""Markdown report generation from a session summary.

Facts and interpretation are kept distinguishable: the report presents measured
facts and derived formulas only. It never names a winner or ranks runtimes.
"""

from __future__ import annotations

from typing import Any

from ..core.version import BENCHMARK_VERSION

NUMBER_FORMATS = {
    "ttft_ms": "{:.2f}",
    "prefill_ms": "{:.2f}",
    "prefill_tok_s": "{:.1f}",
    "decode_tok_s": "{:.2f}",
    "tpot_ms": "{:.3f}",
    "itl_mean_ms": "{:.3f}",
    "itl_p95_ms": "{:.3f}",
    "itl_p99_ms": "{:.3f}",
    "itl_max_ms": "{:.3f}",
    "e2e_ms": "{:.2f}",
    "vram_peak_mb": "{:.1f}",
    "ram_peak_mb": "{:.1f}",
    "cpu_avg": "{:.1f}",
    "gpu_util_avg": "{:.1f}",
    "gpu_power_avg": "{:.1f}",
    "aggregate_output_tok_s": "{:.2f}",
    "decode_tok_s_per_gb_vram": "{:.2f}",
    "decode_tok_s_per_watt": "{:.3f}",
    "ttft_per_1k_prompt_tokens": "{:.3f}",
}


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        template = NUMBER_FORMATS.get(key, "{:.2f}")
        try:
            return template.format(value)
        except (ValueError, TypeError):
            return str(value)
    return str(value)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _group_lookup(summary: dict[str, Any], suite: str) -> list[dict[str, Any]]:
    return [g for g in summary.get("groups", []) if g.get("suite") == suite]


def _matrix_groups(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for group in _group_lookup(summary, "matrix"):
        if group.get("concurrency", 1) == 1 and not group.get("phase"):
            out[group.get("profile")] = group
    return out


def render_report(summary: dict[str, Any]) -> str:
    """Render ``report.md`` content from a summary dict."""
    lines: list[str] = []
    add = lines.append

    add(f"# Pradium Runtime Benchmark Report — {BENCHMARK_VERSION}")
    add("")
    add("This report contains **measured facts** and **derived metrics** only.")
    add("Interpretation is left to the reader; no winner is declared.")
    add("")

    # --- identity / modes ---
    add("## Session")
    add("")
    add(f"- Session: `{summary.get('session_id')}`")
    add(f"- Runtime: `{summary.get('runtime')}` ({summary.get('runtime_version')})")
    add(f"- Model: `{summary.get('model')}` rev `{summary.get('model_revision')}`")
    add(
        f"- Tokenizer: `{summary.get('tokenizer')}` rev `{summary.get('tokenizer_revision')}`"
    )
    add(f"- Benchmark mode: `{summary.get('benchmark_mode')}` (run policy: `{summary.get('run_policy')}`)")
    add(f"- Performance valid: `{summary.get('performance_valid')}`")
    add(f"- Raw records: {summary.get('raw_record_count', 0)}")
    add("")

    # --- environment ---
    add("## Environment (facts)")
    add("")
    env = summary.get("environment", {})
    gpu = env.get("gpu", {})
    cpu = env.get("cpu", {})
    rows = [
        ["OS", f"{env.get('os', {}).get('system')} {env.get('os', {}).get('release')}"],
        ["Python", env.get("python", {}).get("version")],
        ["GPU", gpu.get("name")],
        ["GPU UUID", gpu.get("uuid")],
        ["Driver", gpu.get("driver")],
        ["CUDA", gpu.get("cuda")],
        ["CPU", f"{cpu.get('model')} ({cpu.get('logical_cores')} cores)"],
        ["System RAM (MiB)", _fmt(env.get("system_ram_mb"))],
        ["Git commit", env.get("pradium_git_commit")],
    ]
    lines.extend(_table(["Item", "Value"], [[k, _fmt(v)] for k, v in rows]))
    add("")

    # --- capabilities ---
    add("## Runtime capabilities (declared)")
    add("")
    caps = summary.get("capabilities", {})
    cap_rows = [
        [k, str(v)] for k, v in sorted(caps.items()) if k != "notes"
    ]
    if cap_rows:
        lines.extend(_table(["Capability", "Status"], cap_rows))
        notes = caps.get("notes") or {}
        if notes:
            add("")
            for key, note in sorted(notes.items()):
                add(f"- note `{key}`: {note}")
    else:
        add("_No capability declaration recorded._")
    add("")

    # --- benchmark configuration ---
    add("## Benchmark configuration (frozen)")
    add("")
    config = summary.get("config", {})
    matrix = config.get("matrix", {})
    if matrix:
        add(
            f"- Input lengths: {matrix.get('inputs')}  (S/M/L classes)"
        )
        add(f"- Output lengths: {matrix.get('outputs')}")
    add(f"- Run policy: {summary.get('run_policy')} (see config.json for warmups/measured runs)")
    add(f"- Telemetry interval: {config.get('telemetry', {}).get('interval_ms')} ms")
    add("")

    # --- 3x3 matrix layout ---
    add("## 3x3 workload matrix")
    add("")
    add("```text")
    add("              Output tokens")
    add("Input        64      256      1024")
    add("128          SS      SM       SL")
    add("1024         MS      MM       ML")
    add("4096         LS      LM       LL")
    add("```")
    add("")

    groups = _matrix_groups(summary)
    if groups:
        order = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]
        present = [p for p in order if p in groups]

        def metric_table(title: str, key: str) -> None:
            add(f"### {title}")
            add("")
            rows = []
            for profile in present:
                group = groups[profile]
                stats = group.get("stats", {}).get(key, {})
                rows.append(
                    [
                        profile,
                        _fmt(stats.get("median"), key),
                        _fmt(stats.get("mean"), key),
                        _fmt(stats.get("min"), key),
                        _fmt(stats.get("max"), key),
                        _fmt(stats.get("p95"), key),
                        _fmt(stats.get("cv"), key),
                    ]
                )
            lines.extend(
                _table(
                    ["Profile", "median", "mean", "min", "max", "p95", "cv"],
                    rows,
                )
            )
            add("")

        metric_table("TTFT (ms) — first token minus submission", "ttft_ms")
        metric_table("Prefill (tok/s, runtime-reported)", "prefill_tok_s")
        metric_table("Sustained decode (tok/s per request)", "decode_tok_s")
        metric_table("TPOT (ms/token)", "tpot_ms")
        metric_table("End-to-end latency (ms)", "e2e_ms")
        metric_table("ITL p95 (ms)", "itl_p95_ms")
        metric_table("Peak VRAM (MiB)", "vram_peak_mb")
        metric_table("Peak host RAM (MiB)", "ram_peak_mb")
        metric_table("CPU utilization avg (%)", "cpu_avg")
        metric_table("GPU power avg (W, GPU-side only)", "gpu_power_avg")

    # --- context scaling ---
    context = summary.get("suite_blocks", {}).get("context_scaling")
    if context:
        add("## Context scaling (facts)")
        add("")
        rows = [
            [
                str(c["input_tokens"]),
                _fmt(c["ttft_ms"], "ttft_ms"),
                _fmt(c["prefill_tok_s"], "prefill_tok_s"),
                _fmt(c["decode_tok_s"], "decode_tok_s"),
                _fmt(c["tpot_ms"], "tpot_ms"),
                _fmt(c["vram_peak_mb"], "vram_peak_mb"),
                str(c["execution_status_counts"]),
            ]
            for c in context
        ]
        lines.extend(
            _table(
                ["Input tokens", "TTFT ms", "prefill tok/s", "decode tok/s", "TPOT ms", "peak VRAM MiB", "statuses"],
                rows,
            )
        )
        add("")

    # --- concurrency ---
    scaling = summary.get("suite_blocks", {}).get("concurrency_scaling")
    if scaling:
        add("## Concurrency scaling (facts + derived)")
        add("")
        for profile in sorted(scaling):
            add(f"### Profile {profile}")
            add("")
            rows = []
            for level in sorted(scaling[profile], key=int):
                entry = scaling[profile][level]
                rows.append(
                    [
                        f"C{level}",
                        _fmt(entry.get("ttft_ms"), "ttft_ms"),
                        _fmt(entry.get("tpot_ms"), "tpot_ms"),
                        _fmt(entry.get("decode_tok_s_per_request"), "decode_tok_s"),
                        _fmt(entry.get("aggregate_decode_tok_s"), "decode_tok_s"),
                        _fmt(entry.get("scaling_efficiency")),
                        _fmt(entry.get("per_request_throughput_degradation")),
                    ]
                )
            lines.extend(
                _table(
                    [
                        "Concurrency",
                        "TTFT ms",
                        "TPOT ms",
                        "decode tok/s/req",
                        "agg decode tok/s",
                        "scaling eff.",
                        "req. throughput degr.",
                    ],
                    rows,
                )
            )
            add("")
        add(
            "scaling_efficiency(Cn) = aggregate_decode(Cn) / (n × decode(C1)); "
            "degradation = current/baseline − 1."
        )
        add("")

    # --- prefix reuse ---
    prefix = summary.get("suite_blocks", {}).get("prefix_reuse")
    if prefix:
        add("## Prefix / KV reuse (facts + derived)")
        add("")
        rows = [
            [
                f"{int((p['ratio'] or 0) * 100)}%",
                str(p["scenario"]),
                _fmt(p["ttft_ms"], "ttft_ms"),
                _fmt(p["prefill_ms"], "prefill_ms"),
                _fmt(p["decode_tok_s"], "decode_tok_s"),
                _fmt(p["reused_tokens"]),
                _fmt(p["ttft_speedup_vs_0"]),
                _fmt(p["prefill_speedup_vs_0"]),
            ]
            for p in prefix
        ]
        lines.extend(
            _table(
                [
                    "Reuse",
                    "scenario",
                    "TTFT ms",
                    "prefill ms",
                    "decode tok/s",
                    "reused tok",
                    "TTFT speedup",
                    "prefill speedup",
                ],
                rows,
            )
        )
        add("")

    # --- scheduler interference ---
    for group in _group_lookup(summary, "scheduler"):
        pass  # per-request facts live in the matrix-style tables below
    sched = (summary.get("suite_blocks", {}) or {}).get("scheduler_interference") or (
        summary.get("suite_extras", {}) or {}
    ).get("scheduler_interference")
    if sched:
        add("## Scheduler interference (facts)")
        add("")
        add(f"- Status: {sched.get('status')}")
        add(f"- Intruder TTFT (ms): {_fmt(sched.get('intruder_ttft_ms'), 'ttft_ms')}")
        add(f"- Intruder prefill (ms): {_fmt(sched.get('intruder_prefill_ms'), 'prefill_ms')}")
        add(f"- ITL before large prefill (ms): {_fmt(sched.get('itl_before_ms'))}")
        add(f"- ITL during large prefill (ms): {_fmt(sched.get('itl_during_ms'))}")
        add(f"- ITL after large prefill (ms): {_fmt(sched.get('itl_after_ms'))}")
        add(f"- Maximum output-token stall (ms): {_fmt(sched.get('max_output_token_stall_ms'))}")
        add(f"- p95 ITL (ms): {_fmt(sched.get('p95_itl_ms'))}")
        add("")

    # --- batching ---
    batching = (summary.get("suite_extras", {}) or {}).get("batching")
    if batching:
        add("## Static batching (facts)")
        add("")
        rows = [
            [f"B{size}", str(block.get("supported")), str(block.get("successful_requests"))]
            for size, block in sorted(batching.items(), key=lambda kv: int(kv[0]))
        ]
        lines.extend(_table(["Batch size", "supported", "successful requests"], rows))
        add("")
        add("Bn (static batch) is not equivalent to Cn (concurrency).")
        add("")

    # --- startup ---
    startup = (summary.get("suite_extras", {}) or {}).get("startup")
    if startup:
        add("## Startup (facts)")
        add("")
        rows = [[key, _fmt(value)] for key, value in sorted(startup.items())]
        lines.extend(_table(["Milestone", "Value (ms unless noted)"], rows))
        add("")

    # --- stability / soak ---
    soak = (summary.get("suite_extras", {}) or {}).get("soak")
    if soak:
        add("## Stability / soak (facts)")
        add("")
        analysis = soak.get("analysis", {})
        rows = [
            ["requested duration (s)", _fmt(soak.get("duration_requested_s"))],
            ["actual duration (s)", _fmt(soak.get("duration_actual_s"))],
            ["iterations", str(soak.get("iterations"))],
            ["failed requests", str(analysis.get("failed_requests_total"))],
            ["RAM growth (MiB)", _fmt(analysis.get("ram_growth_mb"))],
            ["VRAM growth (MiB)", _fmt(analysis.get("vram_growth_mb"))],
            ["TTFT growth ratio", _fmt(analysis.get("ttft_growth_ratio"))],
            ["decode throughput change", _fmt(analysis.get("decode_throughput_change_ratio"))],
        ]
        lines.extend(_table(["Metric", "Value"], rows))
        add("")

    # --- mixed workloads ---
    mixed = (summary.get("suite_extras", {}) or {}).get("mixed")
    if mixed:
        add("## Mixed workloads (facts)")
        add("")
        rows = [
            [name, str(block.get("concurrency")), str(block.get("requests")), str(block.get("records"))]
            for name, block in sorted(mixed.items())
        ]
        lines.extend(_table(["Workload", "concurrency", "requests", "records"], rows))
        add("")

    # --- failures / unsupported ---
    add("## Failures and unsupported measurements")
    add("")
    failures = summary.get("failures", [])
    if failures:
        rows = [
            [
                str(f.get("suite")),
                str(f.get("profile")),
                str(f.get("status")),
                str(f.get("input_tokens")),
                str(f.get("requested_output_tokens")),
                (f.get("error") or "")[:80],
            ]
            for f in failures[:50]
        ]
        lines.extend(
            _table(
                ["suite", "profile", "status", "in tok", "req out tok", "error"],
                rows,
            )
        )
        if len(failures) > 50:
            add(f"\n_... {len(failures) - 50} more failures in failures.jsonl_")
    else:
        add("_No failures recorded._")
    add("")
    unsupported = summary.get("unsupported_measurements", [])
    if unsupported:
        add("### Unsupported / unavailable")
        add("")
        for item in unsupported:
            add(f"- {item}")
        add("")

    add("---")
    add("")
    add(
        "Metric formulas: TTFT = first_token_ts − submit_ts; "
        "decode window = last_token_ts − first_token_ts; "
        "decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); "
        "aggregate_output_tok_s = total_output_tokens/wall_clock_s; "
        "ITL_k = ts_k − ts_{k−1}."
    )
    add("")
    return "\n".join(lines) + "\n"
