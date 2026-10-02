#!/usr/bin/env python3
"""Compare greedy token IDs between two harness sessions (correctness gate).

    compare_tokens.py <baseline-session-dir> <candidate-session-dir> [--json-out PATH]

Matches traces by (workload profile, run index, warmup flag) and reports, for
every matched pair, whether the generated token ID sequences are identical.
Different prompts or different decoded lengths are reported as mismatches, not
silently skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_session(session: Path) -> dict:
    raw = session / "raw.jsonl"
    if not raw.exists():
        raise SystemExit(f"missing raw.jsonl in {session}")
    records = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    traces_dir = session / "traces"
    traces = {}
    for path in sorted(traces_dir.glob("req-*.json")):
        data = json.loads(path.read_text())
        traces[data["request_id"]] = [t["token_id"] for t in data["tokens"]]
    keyed = {}
    for r in records:
        rid = r.get("request_id")
        if rid not in traces:
            continue
        key = (
            r.get("profile") or r.get("workload"),
            r.get("run_index"),
            bool(r.get("warmup")),
        )
        keyed[key] = {"request_id": rid, "tokens": traces[rid], "decode_tok_s": r.get("decode_tok_s")}
    return keyed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    base = load_session(args.baseline)
    cand = load_session(args.candidate)

    by_profile: dict[str, dict[str, int]] = defaultdict(lambda: {"match": 0, "mismatch": 0, "missing": 0})
    mismatches = []
    for key, b in sorted(base.items(), key=lambda kv: str(kv[0])):
        profile = key[0]
        c = cand.get(key)
        if c is None:
            by_profile[profile]["missing"] += 1
            continue
        if b["tokens"] == c["tokens"]:
            by_profile[profile]["match"] += 1
        else:
            by_profile[profile]["mismatch"] += 1
            first = next(
                (i for i, (x, y) in enumerate(zip(b["tokens"], c["tokens"])) if x != y),
                min(len(b["tokens"]), len(c["tokens"])),
            )
            mismatches.append(
                {
                    "profile": profile,
                    "run_index": key[1],
                    "warmup": key[2],
                    "baseline_len": len(b["tokens"]),
                    "candidate_len": len(c["tokens"]),
                    "first_divergence_index": first,
                    "baseline_tokens": b["tokens"][: max(0, first + 4)],
                    "candidate_tokens": c["tokens"][: max(0, first + 4)],
                }
            )

    print("| profile | identical | diverged | missing |")
    print("|---|---|---|---|")
    total = {"match": 0, "mismatch": 0, "missing": 0}
    for profile in sorted(by_profile):
        row = by_profile[profile]
        for k in total:
            total[k] += row[k]
        print(f"| {profile} | {row['match']} | {row['mismatch']} | {row['missing']} |")
    print(f"| **total** | {total['match']} | {total['mismatch']} | {total['missing']} |")

    for m in mismatches[:10]:
        print(
            f"DIVERGENCE {m['profile']} run={m['run_index']} warmup={m['warmup']} "
            f"len {m['baseline_len']}->{m['candidate_len']} first_diff={m['first_divergence_index']}"
        )
        print(f"  baseline : {m['baseline_tokens']}")
        print(f"  candidate: {m['candidate_tokens']}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"by_profile": by_profile, "mismatches": mismatches}, indent=2) + "\n"
        )
    return 0 if total["mismatch"] == 0 and total["missing"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
