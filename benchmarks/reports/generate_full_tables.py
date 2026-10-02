#!/usr/bin/env python3
"""Emit the full campaign benchmark as markdown tables, straight from raw.jsonl."""

from __future__ import annotations

import glob
import json
import os
import statistics as st

# Campaign directory, resolved relative to this script so the generator is portable.
BASE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "campaigns",
    "rtx3060-minicpm5-exl3-b0"))


def load(suite: str, base: str = "exllamav3"):
    p = os.path.join(BASE, base, suite, "raw.jsonl")
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def med(rs, f, nd=2):
    v = sorted(r[f] for r in rs if isinstance(r.get(f), (int, float)))
    if not v:
        return "-"
    x = st.median(v)
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def g(rs):
    return [r for r in rs if not r.get("warmup")]


print("### 1. 3x3 core matrix — all nine workloads (median of 10 measured runs)\n")
print("| WL | in→out | TTFT ms | pref ms | pref tok/s | dec ms | dec tok/s | TPOT ms |"
      " ITL med | ITL p95 | ITL p99 | ITL max | E2E ms | VRAM MB | CPU % | GPU % | W | °C |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
R = load("matrix")
for p in ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]:
    rs = [r for r in g(R) if r.get("profile") == p]
    print(f"| {p} | {rs[0]['prompt_tokens']}→{rs[0]['requested_output_tokens']} | "
          f"{med(rs,'ttft_ms')} | {med(rs,'prefill_ms')} | {med(rs,'prefill_tok_s',0)} | "
          f"{med(rs,'decode_ms',0)} | {med(rs,'decode_tok_s')} | {med(rs,'tpot_ms')} | "
          f"{med(rs,'itl_median_ms')} | {med(rs,'itl_p95_ms')} | {med(rs,'itl_p99_ms')} | "
          f"{med(rs,'itl_max_ms')} | {med(rs,'e2e_ms')} | {med(rs,'vram_peak_mb',0)} | "
          f"{med(rs,'cpu_avg',0)} | {med(rs,'gpu_util_avg')} | {med(rs,'gpu_power_avg')} | "
          f"{med(rs,'gpu_temperature_peak',0)} |")

print("\n### 2. Context scaling — 128 → 32768 input, 256 output\n")
print("| input | TTFT ms | pref ms | pref tok/s | dec tok/s | TPOT ms | ITL p95 | E2E ms |"
      " VRAM MB | GPU % | W | °C | runs |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
C = load("context")
for L in (128, 512, 1024, 2048, 4096, 8192, 16384, 32768):
    rs = [r for r in g(C) if r.get("prompt_tokens") == L] if L != 32768 else []
    if not rs:
        continue
    print(f"| {L} | {med(rs,'ttft_ms')} | {med(rs,'prefill_ms')} | {med(rs,'prefill_tok_s',0)} | "
          f"{med(rs,'decode_tok_s')} | {med(rs,'tpot_ms')} | {med(rs,'itl_p95_ms')} | "
          f"{med(rs,'e2e_ms')} | {med(rs,'vram_peak_mb',0)} | {med(rs,'gpu_util_avg')} | "
          f"{med(rs,'gpu_power_avg')} | {med(rs,'gpu_temperature_peak',0)} | {len(rs)} |")
print("| 32768 † | FAILED | — | — | — | — | — | — | — | — | — | — | 0/13 |")
S = load("context", "supplementary")
rs = g(S)
print(f"| 32768 ‡ | {med(rs,'ttft_ms')} | {med(rs,'prefill_ms')} | {med(rs,'prefill_tok_s',0)} | "
      f"{med(rs,'decode_tok_s')} | {med(rs,'tpot_ms')} | {med(rs,'itl_p95_ms')} | "
      f"{med(rs,'e2e_ms')} | {med(rs,'vram_peak_mb',0)} | {med(rs,'gpu_util_avg')} | "
      f"{med(rs,'gpu_power_avg')} | {med(rs,'gpu_temperature_peak',0)} | {len(rs)} |")

print("\n### 3. Concurrency C1/C2/C4/C8 — all nine profiles\n")
print("| WL | metric | C1 | C2 | C4 | C8 |")
print("|---|---|---|---|---|---|")
N = load("concurrency")
for p in ["SS", "SM", "SL", "MS", "MM", "ML", "LS", "LM", "LL"]:
    row = {}
    for c in (1, 2, 4, 8):
        rs = [r for r in g(N) if r.get("profile") == p and r.get("concurrency") == c]
        row[c] = rs
    if not row[1]:
        continue
    a1, a8 = (st.median([r["aggregate_output_tok_s"] for r in row[c]
                         if isinstance(r.get("aggregate_output_tok_s"), (int, float))] or [0])
              for c in (1, 8))
    fac = a8 / a1 if a1 else 0
    eff = fac / 8 * 100
    d1, d8 = (st.median([r["decode_tok_s"] for r in row[c]
                         if isinstance(r.get("decode_tok_s"), (int, float))] or [0])
              for c in (1, 8))
    ret = d8 / d1 * 100 if d1 else 0
    print(f"| {p} | aggregate tok/s | " + " | ".join(med(row[c], "aggregate_output_tok_s")
                                                     for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | per-req decode tok/s | " + " | ".join(med(row[c], "decode_tok_s")
                                                          for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | TTFT ms | " + " | ".join(med(row[c], "ttft_ms") for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | TPOT ms | " + " | ".join(med(row[c], "tpot_ms") for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | ITL p99 ms | " + " | ".join(med(row[c], "itl_p99_ms") for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | E2E ms | " + " | ".join(med(row[c], "e2e_ms") for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | peak VRAM MB | " + " | ".join(med(row[c], "vram_peak_mb", 0) for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | GPU % | " + " | ".join(med(row[c], "gpu_util_avg") for c in (1, 2, 4, 8)) + " |")
    print(f"| {p} | power W | " + " | ".join(med(row[c], "gpu_power_avg") for c in (1, 2, 4, 8)) + " |")
    print(f"| **{p}** | **C1→C8** | | | | **{fac:.2f}× ({eff:.1f}% of ideal), per-req {ret:.1f}%** |")

print("\n### 4. Prefix / KV reuse — 1024-token inputs, 256 output\n")
print("| reuse | scenario | TTFT ms | pref ms | pref tok/s | speedup | dec tok/s | E2E ms |")
print("|---|---|---|---|---|---|---|---|")
P = load("prefix-cache")
grp = {}
for r in g(P):
    grp.setdefault((r.get("expected_reuse_ratio"),
                    (r.get("metadata") or {}).get("scenario")), []).append(r)
ctrl = {k[0]: st.median([x["prefill_tok_s"] for x in v
                         if isinstance(x.get("prefill_tok_s"), (int, float))])
        for k, v in grp.items() if k[1] == "cross_session"}
for k in sorted(grp, key=lambda x: (x[0] if x[0] is not None else -1, str(x[1]))):
    rs = grp[k]
    cur = st.median([x["prefill_tok_s"] for x in rs
                     if isinstance(x.get("prefill_tok_s"), (int, float))])
    sp = cur / ctrl[k[0]] if ctrl.get(k[0]) else None
    print(f"| {k[0]} | {k[1]} | {med(rs,'ttft_ms')} | {med(rs,'prefill_ms')} | {cur:.0f} | "
          f"{('%.2f×' % sp) if sp else '-'} | {med(rs,'decode_tok_s')} | {med(rs,'e2e_ms')} |")

print("\n### 5. Scheduler interference — 3 resident decoders + one 4096-token prefill\n")
print("| metric | value |")
print("|---|---|")
res = json.load(open(os.path.join(BASE, "exllamav3", "scheduler", "summary.json")))
si = (res.get("suite_extras") or {}).get("scheduler_interference", {})
for k in ("background_requests", "background_aggregate_decode_tok_s", "intruder_profile",
          "intruder_input_tokens", "intruder_prefill_ms", "intruder_ttft_ms",
          "itl_before_ms", "itl_during_ms", "itl_after_ms", "max_output_token_stall_ms",
          "p95_itl_ms"):
    v = si.get(k)
    print(f"| {k} | {('%.2f' % v) if isinstance(v, float) else v} |")

print("\n### 6. Mixed request workloads\n")
print("| scenario | TTFT ms | dec tok/s | aggregate tok/s | TPOT ms | ITL p99 | E2E ms | VRAM MB | GPU % |")
print("|---|---|---|---|---|---|---|---|---|")
M = load("mixed")
mg = {}
for r in g(M):
    mg.setdefault((r.get("metadata") or {}).get("mixed_workload"), []).append(r)
for k in sorted(mg, key=str):
    rs = mg[k]
    print(f"| `{k}` | {med(rs,'ttft_ms')} | {med(rs,'decode_tok_s')} | "
          f"{med(rs,'aggregate_output_tok_s')} | {med(rs,'tpot_ms')} | {med(rs,'itl_p99_ms')} | "
          f"{med(rs,'e2e_ms')} | {med(rs,'vram_peak_mb',0)} | {med(rs,'gpu_util_avg')} |")

print("\n### 7. Batching B1/B2/B4/B8 at MM\n")
print("| batch | requests | aggregate tok/s | E2E ms | scaling vs B1 |")
print("|---|---|---|---|---|")
B = load("batching")
b1 = None
for b in (1, 2, 4, 8):
    rs = [r for r in g(B) if r.get("batch_size") == b]
    cur = st.median([x["aggregate_output_tok_s"] for x in rs
                     if isinstance(x.get("aggregate_output_tok_s"), (int, float))])
    if b == 1:
        b1 = cur
    print(f"| B{b} | {len(rs)} | {cur:.2f} | {med(rs,'e2e_ms')} | {cur/b1:.2f}× |")

print("\n### 8. Startup\n")
print("| milestone | value |")
print("|---|---|")
su = json.load(open(os.path.join(BASE, "exllamav3", "startup", "summary.json")))
sd = (su.get("suite_extras") or {}).get("startup", {})
for k in ("runtime_init_ms", "cuda_init_ms", "model_load_ms", "process_start_to_ready_ms",
          "cold_first_request_ttft_ms", "process_start_to_first_token_ms",
          "warm_request_ttft_ms_median", "idle_vram_mb", "idle_ram_mb",
          "cuda_graph_capture_ms", "weight_mapping_ms"):
    v = sd.get(k)
    print(f"| {k} | {('%.4f' % v) if isinstance(v, float) else v} |")

print("\n### 9. Soak / stability — 30 minutes\n")
print("| metric | value |")
print("|---|---|")
so = json.load(open(os.path.join(BASE, "exllamav3", "soak", "summary.json")))
sk = (so.get("suite_extras") or {}).get("soak", {})
a = sk.get("analysis", {})
print(f"| duration actual s | {sk.get('duration_actual_s'):.1f} |")
print(f"| iterations | {sk.get('iterations')} |")
print(f"| new failures | {a.get('failed_requests_total')} |")
print(f"| VRAM growth MB | {a.get('vram_growth_mb'):+.2f} |")
print(f"| RAM growth MB | {a.get('ram_growth_mb'):+.2f} |")
print(f"| TTFT change | {a.get('ttft_growth_ratio')*100:+.1f}% |")
print(f"| decode throughput change | {a.get('decode_throughput_change_ratio')*100:+.1f}% |")
