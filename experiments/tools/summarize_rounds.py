#!/usr/bin/env python3
"""Summarize harness matrix sessions into decode-speedup tables.

    summarize_rounds.py label=session_dir [label=session_dir ...] [--json-out PATH]

Each session dir must contain raw.jsonl (the harness session directory). Warmup
records are excluded, medians of the measured runs are reported per workload,
and the last two sessions are compared against the frozen baseline and against
each other.

The frozen reference is read from FROZEN_SESSION (default: the campaign's
official matrix session on the VM) when it exists.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

FROZEN_DEFAULT = (
    "/workspace/pradium/benchmarks/campaigns/rtx3060-minicpm5-exl3-b0/"
    "sessions/20261002-090349_exllamav3_official_537aee"
)

# Frozen campaign medians (derived-tables.md), used when the session is not present.
FROZEN_MEDIANS = {
    "SS": 109.80, "SM": 85.58, "SL": 96.45, "MS": 109.36, "MM": 101.31,
    "ML": 87.95, "LS": 99.81, "LM": 89.29, "LL": 77.19,
}

ORDER = ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]


def load_session(path: Path) -> dict[str, dict]:
    raw = path / "raw.jsonl"
    if not raw.exists():
        raise SystemExit(f"no raw.jsonl in {path}")
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for line in raw.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("warmup"):
            continue
        profile = rec.get("profile") or rec.get("workload")
        if profile:
            by_profile[profile].append(rec)
    out = {}
    for profile, recs in by_profile.items():
        dec = [r["decode_tok_s"] for r in recs if r.get("decode_tok_s")]
        ttft = [r["ttft_ms"] for r in recs if r.get("ttft_ms") is not None]
        tpot = [r["tpot_ms"] for r in recs if r.get("tpot_ms") is not None]
        p99 = [r["itl_p99_ms"] for r in recs if r.get("itl_p99_ms") is not None]
        vram = [r["vram_peak_mb"] for r in recs if r.get("vram_peak_mb") is not None]
        statuses = [r.get("execution_status") for r in recs]
        out[profile] = {
            "n": len(recs),
            "decode_median": st.median(dec) if dec else None,
            "decode_min": min(dec) if dec else None,
            "decode_max": max(dec) if dec else None,
            "decode_stdev": st.stdev(dec) if len(dec) > 1 else 0.0,
            "ttft_median": st.median(ttft) if ttft else None,
            "tpot_median": st.median(tpot) if tpot else None,
            "itl_p99_median": st.median(p99) if p99 else None,
            "vram_peak": max(vram) if vram else None,
            "statuses": sorted(set(s for s in statuses if s)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sessions", nargs="+", help="label=session_dir")
    ap.add_argument("--frozen", default=os.environ.get("FROZEN_SESSION", FROZEN_DEFAULT))
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    sessions: list[tuple[str, dict]] = []
    for spec in args.sessions:
        label, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"expected label=session_dir, got {spec!r}")
        sessions.append((label, load_session(Path(path))))

    frozen = dict(FROZEN_MEDIANS)
    if Path(args.frozen).exists():
        f = load_session(Path(args.frozen))
        for profile, row in f.items():
            if row["decode_median"]:
                frozen[profile] = row["decode_median"]

    print("| profile | frozen | " + " | ".join(label for label, _ in sessions) + " | " +
          " | ".join(f"x vs frozen ({label})" for label, _ in sessions) + " |")
    print("|---" * (2 + 2 * len(sessions)) + "|")
    summary = {"frozen": frozen, "sessions": {}}
    for label, data in sessions:
        summary["sessions"][label] = data
    for profile in ORDER:
        row = [f"| {profile} | {frozen.get(profile, float('nan')):.2f} "]
        for label, data in sessions:
            v = data.get(profile, {}).get("decode_median")
            row.append(f"| {v:.2f} " if v else "| n/a ")
        for label, data in sessions:
            v = data.get(profile, {}).get("decode_median")
            row.append(f"| {v / frozen[profile]:.3f}x " if v else "| n/a ")
        print("".join(row) + "|")

    print()
    for label, data in sessions:
        ratios = [data[p]["decode_median"] / frozen[p] for p in ORDER
                  if data.get(p, {}).get("decode_median")]
        if ratios:
            print(f"{label}: geomean {st.geometric_mean(ratios):.3f}x  "
                  f"min {min(ratios):.3f}x  max {max(ratios):.3f}x  (n={len(ratios)})")
        bad = [p for p in ORDER if data.get(p, {}).get("statuses") and
               data[p]["statuses"] not in (["SUCCESS"], ["PASS"])]
        if bad:
            print(f"  non-SUCCESS statuses: {[(p, data[p]['statuses']) for p in bad]}")

    if len(sessions) >= 2:
        print()
        (lab_a, a), (lab_b, b) = sessions[-2], sessions[-1]
        ratios = []
        for p in ORDER:
            va, vb = a.get(p, {}).get("decode_median"), b.get(p, {}).get("decode_median")
            if va and vb:
                ratios.append(vb / va)
                print(f"{p}: {lab_a} {va:.2f} -> {lab_b} {vb:.2f}  ({(vb / va - 1) * 100:+.1f}%)")
        if ratios:
            print(f"{lab_b} vs {lab_a}: geomean {st.geometric_mean(ratios):.3f}x")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
