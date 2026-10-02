"""Plot generation (SVG, dependency-free).

Plot generation is strictly independent from benchmark execution: plots are
rendered from persisted raw data/summaries and can be regenerated at any time.

A Matplotlib backend can be substituted later; the SVG renderer keeps the
suite dependency-free while producing real, inspectable plots.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

#: Plot dimensions
WIDTH = 720
HEIGHT = 420
MARGIN_L = 70
MARGIN_R = 20
MARGIN_T = 40
MARGIN_B = 60

_PALETTE = [
    "#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280",
    "#098699", "#9c4221", "#5a67d8",
]


class SvgCanvas:
    """Minimal SVG chart canvas (bars, lines, scatter, log-free axes)."""

    def __init__(self, title: str, x_label: str, y_label: str) -> None:
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.elements: list[str] = []
        self._x_min = 0.0
        self._x_max = 1.0
        self._y_min = 0.0
        self._y_max = 1.0

    def _sx(self, x: float) -> float:
        span = self._x_max - self._x_min or 1.0
        return MARGIN_L + (x - self._x_min) / span * (WIDTH - MARGIN_L - MARGIN_R)

    def _sy(self, y: float) -> float:
        span = self._y_max - self._y_min or 1.0
        return HEIGHT - MARGIN_B - (y - self._y_min) / span * (HEIGHT - MARGIN_T - MARGIN_B)

    def set_x_range(self, lo: float, hi: float) -> None:
        self._x_min, self._x_max = lo, hi

    def set_y_range(self, lo: float, hi: float) -> None:
        self._y_min, self._y_max = lo, hi

    def line(self, points: Sequence[tuple[float, float]], color: str, label: str = "") -> None:
        coords = " ".join(f"{self._sx(x):.1f},{self._sy(y):.1f}" for x, y in points)
        self.elements.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y in points:
            self.elements.append(
                f'<circle cx="{self._sx(x):.1f}" cy="{self._sy(y):.1f}" r="3" fill="{color}"/>'
            )
        if label:
            self.elements.append(f'<text x="0" y="0" fill="{color}" font-size="11">{_xml(label)}</text>')

    def bars(self, values: Sequence[tuple[str, float]], color: str = "#2b6cb0") -> None:
        n = max(1, len(values))
        slot = (WIDTH - MARGIN_L - MARGIN_R) / n
        bar_w = slot * 0.6
        for i, (label, value) in enumerate(values):
            x = MARGIN_L + i * slot + (slot - bar_w) / 2
            y = self._sy(value)
            h = (HEIGHT - MARGIN_B) - y
            self.elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{max(0.0, h):.1f}" fill="{color}"/>'
            )
            self.elements.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{HEIGHT - MARGIN_B + 14:.1f}" '
                f'font-size="10" text-anchor="middle">{_xml(label)}</text>'
            )

    def band(self, x0: float, x1: float, color: str = "#eeeeee") -> None:
        self.elements.append(
            f'<rect x="{self._sx(x0):.1f}" y="{MARGIN_T}" '
            f'width="{max(1.0, self._sx(x1) - self._sx(x0)):.1f}" '
            f'height="{HEIGHT - MARGIN_T - MARGIN_B}" fill="{color}" opacity="0.7"/>'
        )

    def _axes(self) -> list[str]:
        out = []
        y_ticks = 5
        for i in range(y_ticks + 1):
            value = self._y_min + (self._y_max - self._y_min) * i / y_ticks
            y = self._sy(value)
            out.append(
                f'<line x1="{MARGIN_L}" y1="{y:.1f}" x2="{WIDTH - MARGIN_R}" y2="{y:.1f}" '
                f'stroke="#dddddd" stroke-width="1"/>'
            )
            out.append(
                f'<text x="{MARGIN_L - 8:.1f}" y="{y + 4:.1f}" font-size="10" '
                f'text-anchor="end">{value:.2g}</text>'
            )
        out.append(
            f'<line x1="{MARGIN_L}" y1="{MARGIN_T}" x2="{MARGIN_L}" '
            f'y2="{HEIGHT - MARGIN_B}" stroke="#333333" stroke-width="1.5"/>'
        )
        out.append(
            f'<line x1="{MARGIN_L}" y1="{HEIGHT - MARGIN_B}" x2="{WIDTH - MARGIN_R}" '
            f'y2="{HEIGHT - MARGIN_B}" stroke="#333333" stroke-width="1.5"/>'
        )
        return out

    def render(self) -> str:
        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">',
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="white"/>',
            f'<text x="{WIDTH / 2:.0f}" y="20" font-size="14" text-anchor="middle">'
            f"{_xml(self.title)}</text>",
            f'<text x="{WIDTH / 2:.0f}" y="{HEIGHT - 12}" font-size="11" text-anchor="middle">'
            f"{_xml(self.x_label)}</text>",
            f'<text x="16" y="{HEIGHT / 2:.0f}" font-size="11" text-anchor="middle" '
            f'transform="rotate(-90 16 {HEIGHT / 2:.0f})">{_xml(self.y_label)}</text>',
        ]
        parts.extend(self._axes())
        parts.extend(self.elements)
        parts.append("</svg>")
        return "\n".join(parts)


def _xml(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write(canvas: SvgCanvas, path: Path) -> Path:
    path.write_text(canvas.render(), encoding="utf-8")
    return path


# --- plot builders ---------------------------------------------------------

MATRIX_METRICS = {
    "ttft": ("ttft_ms", "TTFT by workload (median ms)"),
    "prefill": ("prefill_tok_s", "Prefill throughput by workload (median tok/s)"),
    "decode": ("decode_tok_s", "Decode throughput by workload (median tok/s)"),
    "tpot": ("tpot_ms", "TPOT by workload (median ms/token)"),
    "e2e": ("e2e_ms", "E2E latency by workload (median ms)"),
    "vram": ("vram_peak_mb", "Peak VRAM by workload (median MiB)"),
    "ram": ("ram_peak_mb", "Peak RAM by workload (median MiB)"),
    "cpu": ("cpu_avg", "CPU by workload (median %)"),
    "power": ("gpu_power_avg", "GPU power by workload (median W)"),
}


def _matrix_groups(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        g["profile"]: g
        for g in summary.get("groups", [])
        if g.get("suite") == "matrix" and g.get("concurrency", 1) == 1 and not g.get("phase")
    }


def generate_plots(
    session_dir: Path | str,
    summary: dict[str, Any],
    suite_extras: dict[str, Any] | None = None,
) -> list[Path]:
    """Render all applicable plots for a session into ``plots/``."""
    session_dir = Path(session_dir)
    plots_dir = session_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    suite_extras = suite_extras or summary.get("suite_extras", {}) or {}

    # --- metric by workload (matrix) ---
    matrix = _matrix_groups(summary)
    order = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]
    profiles = [p for p in order if p in matrix]
    for key, (field, title) in MATRIX_METRICS.items():
        values = []
        for profile in profiles:
            value = matrix[profile].get("stats", {}).get(field, {}).get("median")
            if value is not None:
                values.append((profile, float(value)))
        if values:
            canvas = SvgCanvas(title, "workload profile", field)
            canvas.set_x_range(-0.5, len(values) - 0.5)
            canvas.set_y_range(0.0, max(v for _, v in values) * 1.15 or 1.0)
            canvas.bars(values)
            written.append(_write(canvas, plots_dir / f"matrix_{key}_by_workload.svg"))

    # --- vs concurrency ---
    conc = summary.get("suite_blocks", {}).get("concurrency_scaling") or {}
    for field, title, y_label in (
        ("aggregate_decode_tok_s", "Aggregate decode tok/s vs concurrency", "tok/s"),
        ("decode_tok_s_per_request", "Per-request decode tok/s vs concurrency", "tok/s"),
        ("ttft_ms", "TTFT vs concurrency (median ms)", "ms"),
        ("tpot_ms", "TPOT vs concurrency (median ms/token)", "ms/token"),
    ):
        canvas = SvgCanvas(title, "concurrency", y_label)
        series = 0
        max_x = 1
        max_y = 0.0
        plotted = []
        for i, profile in enumerate(sorted(conc)):
            points = []
            for level, entry in sorted(conc[profile].items(), key=lambda kv: int(kv[0])):
                value = entry.get(field)
                if value is not None:
                    points.append((int(level), float(value)))
                    max_x = max(max_x, int(level))
                    max_y = max(max_y, float(value))
            if points:
                plotted.append((points, i))
        if plotted:
            canvas.set_x_range(0, max_x * 1.1)
            canvas.set_y_range(0.0, max_y * 1.15 or 1.0)
            for points, i in plotted:
                canvas.line(points, _PALETTE[i % len(_PALETTE)])
            written.append(_write(canvas, plots_dir / f"concurrency_{field}.svg"))

    # --- vs context length ---
    context = summary.get("suite_blocks", {}).get("context_scaling") or []
    for field, title, y_label in (
        ("ttft_ms", "TTFT vs context length", "ms"),
        ("prefill_tok_s", "Prefill tok/s vs context length", "tok/s"),
        ("decode_tok_s", "Decode tok/s vs context length", "tok/s"),
        ("vram_peak_mb", "Peak VRAM vs context length", "MiB"),
    ):
        points = [
            (float(c["input_tokens"]), float(c[field]))
            for c in context
            if c.get(field) is not None
        ]
        if points:
            canvas = SvgCanvas(title, "input tokens", y_label)
            canvas.set_x_range(min(x for x, _ in points), max(x for x, _ in points))
            canvas.set_y_range(0.0, max(y for _, y in points) * 1.15 or 1.0)
            canvas.line(points, _PALETTE[0])
            written.append(_write(canvas, plots_dir / f"context_{field}.svg"))

    # --- prefix reuse ---
    prefix = summary.get("suite_blocks", {}).get("prefix_reuse") or []
    for field, title, y_label in (
        ("ttft_ms", "Prefix reuse % vs TTFT", "ms"),
        ("prefill_ms", "Prefix reuse % vs prefill latency", "ms"),
    ):
        by_scenario: dict[str, list[tuple[float, float]]] = {}
        for entry in prefix:
            if entry.get(field) is None:
                continue
            scenario = str(entry.get("scenario"))
            by_scenario.setdefault(scenario, []).append(
                ((entry.get("ratio") or 0.0) * 100.0, float(entry[field]))
            )
        if by_scenario:
            canvas = SvgCanvas(title, "reusable prefix %", y_label)
            all_points = [p for pts in by_scenario.values() for p in pts]
            canvas.set_x_range(0.0, max(x for x, _ in all_points) * 1.1 or 1.0)
            canvas.set_y_range(0.0, max(y for _, y in all_points) * 1.15 or 1.0)
            for i, (scenario, points) in enumerate(sorted(by_scenario.items())):
                canvas.line(sorted(points), _PALETTE[i % len(_PALETTE)], label=scenario)
            written.append(_write(canvas, plots_dir / f"prefix_{field}.svg"))

    # --- scheduler ITL timeline ---
    sched = suite_extras.get("scheduler_interference") or {}
    timeline = _scheduler_timeline(session_dir, summary, sched)
    if timeline is not None:
        written.append(_write(timeline, plots_dir / "scheduler_itl_timeline.svg"))

    # --- startup breakdown ---
    startup = suite_extras.get("startup") or {}
    startup_points = [
        (key.replace("_ms", ""), float(value))
        for key, value in startup.items()
        if isinstance(value, (int, float)) and key.endswith("_ms")
    ]
    if startup_points:
        canvas = SvgCanvas("Startup timing breakdown (ms)", "milestone", "ms")
        canvas.set_x_range(-0.5, len(startup_points) - 0.5)
        canvas.set_y_range(0.0, max(v for _, v in startup_points) * 1.15 or 1.0)
        canvas.bars(startup_points, color="#2f855a")
        written.append(_write(canvas, plots_dir / "startup_breakdown.svg"))

    return written


def _scheduler_timeline(
    session_dir: Path, summary: dict[str, Any], sched: dict[str, Any]
) -> SvgCanvas | None:
    """ITL timeline of background streams with the intruder prefill window."""
    records = []
    raw_path = session_dir / "raw.jsonl"
    if not raw_path.exists():
        return None
    with raw_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    background = [
        r for r in records
        if r.get("suite") == "scheduler" and (r.get("metadata") or {}).get("role") == "background"
    ]
    if not background:
        return None
    canvas = SvgCanvas(
        "Scheduler: inter-token latency timeline during large prefill",
        "time from first background token (ms)",
        "ITL (ms)",
    )
    all_itls: list[tuple[float, float]] = []
    for record in background:
        trace_path = session_dir / "traces" / f"{record.get('request_id')}.json"
        if not trace_path.exists():
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        tokens = trace.get("tokens", [])
        base = record.get("request_submitted_ns")
        for i in range(1, len(tokens)):
            t0 = tokens[i - 1]["relative_timestamp_ns"] / 1e6
            t1 = tokens[i]["relative_timestamp_ns"] / 1e6
            all_itls.append((t0, t1 - t0))
    if not all_itls:
        return None
    intruder = next(
        (
            r for r in records
            if r.get("suite") == "scheduler" and (r.get("metadata") or {}).get("role") == "intruder"
        ),
        None,
    )
    if intruder and intruder.get("request_submitted_ns") and background[0].get("request_submitted_ns"):
        start_ms = (intruder["request_submitted_ns"] - background[0]["request_submitted_ns"]) / 1e6
        end_ms = (
            (intruder.get("first_token_ns") or intruder["request_submitted_ns"])
            - background[0]["request_submitted_ns"]
        ) / 1e6
        canvas.band(start_ms, end_ms, color="#ffe0e0")
    canvas.set_x_range(0.0, max(x for x, _ in all_itls) * 1.05 or 1.0)
    canvas.set_y_range(0.0, max(y for _, y in all_itls) * 1.15 or 1.0)
    canvas.line(sorted(all_itls), _PALETTE[0])
    return canvas
