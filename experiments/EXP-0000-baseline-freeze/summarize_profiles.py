#!/usr/bin/env python3
"""Aggregate profile_decode.py JSON outputs into one comparison table."""

from __future__ import annotations

import glob
import json
import sys


def row(path: str) -> dict | None:
    with open(path) as fh:
        data = json.load(fh)
    measured = [r for r in data["requests"] if r["role"] == "measured"]
    if not measured:
        return None
    m = measured[-1]
    prof = m.get("profiled", {}) or {}
    itl = m.get("itl_ms", {}) or {}
    steps = prof.get("steps") or 0
    top = prof.get("top_kernels", []) or []
    top3 = ", ".join(
        f"{k['key'].split('(')[0][:34]}:{(k['self_device_us'] / steps / 1000.0) if steps else 0:.2f}ms"
        for k in top[:3]
    )
    return {
        "label": data.get("label"),
        "rev": (data.get("repo") or {}).get("revision", "?")[:8],
        "dirty": (data.get("repo") or {}).get("dirty"),
        "exl": (data.get("exllamav3_file") or "?").replace("/workspace/", ""),
        "prompt": data["config"]["prompt_tokens"],
        "new": data["config"]["max_new_tokens"],
        "decode_tok_s": m.get("decode_tok_s"),
        "itl_med": itl.get("median_ms"),
        "itl_p99": itl.get("p99_ms"),
        "kernel_ms": prof.get("kernel_ms_per_step"),
        "gap_ms": (
            (prof.get("span_ms_per_step") or 0.0) - (prof.get("kernel_ms_per_step") or 0.0)
            if prof.get("span_ms_per_step") is not None
            and prof.get("kernel_ms_per_step") is not None
            else None
        ),
        "steps": steps,
        "d2h": prof.get("d2h_per_step"),
        "top3": top3,
    }


def main() -> int:
    paths: list[str] = []
    for arg in sys.argv[1:]:
        paths.extend(sorted(glob.glob(arg)))
    if not paths:
        print("usage: summarize_profiles.py <profile-*.json | glob> ...", file=sys.stderr)
        return 2
    print(
        "| label | prompt/new | rev | exllamav3 | decode tok/s | ITL med ms | ITL p99 ms | kernel ms/step | gap ms/step | d2h/step | top kernels |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for path in paths:
        try:
            r = row(path)
        except Exception as exc:  # noqa: BLE001
            print(f"| ERROR {path}: {exc} |", file=sys.stderr)
            continue
        if not r:
            continue

        def f(v, nd=2):
            return "n/a" if v is None else f"{v:.{nd}f}"

        print(
            f"| {r['label']} | {r['prompt']}/{r['new']} | {r['rev']}{'*' if r['dirty'] else ''} | "
            f"{r['exl']} | {f(r['decode_tok_s'])} | {f(r['itl_med'],3)} | {f(r['itl_p99'],2)} | "
            f"{f(r['kernel_ms'],3)} | {f(r['gap_ms'],3)} | {r['d2h']} | {r['top3']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
