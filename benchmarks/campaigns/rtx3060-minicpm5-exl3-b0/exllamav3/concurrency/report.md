# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-095935_exllamav3_official_e107da`
- Runtime: `exllamav3` (1.5.3+cu128.torch2.10.0)
- Model: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Tokenizer: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Benchmark mode: `official` (run policy: `official`)
- Performance valid: `True`
- Raw records: 1755

## Environment (facts)

| Item | Value |
|---|---|
| OS | Linux 6.8.0-60-generic |
| Python | 3.12.3 |
| GPU | NVIDIA GeForce RTX 3060 |
| GPU UUID | GPU-9bdacdb0-33f1-e196-6c1f-dc65129fb653 |
| Driver | 550.144.03 |
| CUDA | — |
| CPU | x86_64 (4 cores) |
| System RAM (MiB) | 32030.90 |
| Git commit | cf4e8754e6edcf588ebf0a207472e19c24f233d3 |

## Runtime capabilities (declared)

| Capability | Status |
|---|---|
| supports_cache_reset | SUPPORTED |
| supports_chunked_prefill | SUPPORTED |
| supports_continuous_batching | SUPPORTED |
| supports_cross_request_prefix_cache | SUPPORTED |
| supports_cuda_graphs | UNSUPPORTED |
| supports_fixed_decode_length | SUPPORTED |
| supports_input_ids | SUPPORTED |
| supports_internal_kv_metrics | SUPPORTED |
| supports_internal_queue_metrics | PARTIAL |
| supports_internal_scheduler_metrics | UNSUPPORTED |
| supports_kv_quantization | SUPPORTED |
| supports_prefix_cache | SUPPORTED |
| supports_static_batching | SUPPORTED |
| supports_streaming | SUPPORTED |

- note `supports_cache_reset`: No public free_cache in v1.5.3; drains the queue and calls pagetable.reset_page_table(), verified by get_cache_stats().
- note `supports_chunked_prefill`: max_chunk_size bounds prefill chunks.
- note `supports_cuda_graphs`: v1.5.3 has no CUDA-graph capture path.
- note `supports_internal_kv_metrics`: Generator.get_cache_stats().
- note `supports_internal_queue_metrics`: time_enqueued (queue wait) only.
- note `supports_kv_quantization`: Available (CacheLayer_quant) but the official configuration runs the default fp16 cache.
- note `supports_prefix_cache`: Automatic, page-granular (256-token pages) via a chained content-hash page table; best-effort under eviction.
- note `supports_static_batching`: Real batched decode: all sequences are enqueued before any decode step and advance in lockstep, one token per iterate().

## Benchmark configuration (frozen)

- Input lengths: {'L': 4096, 'M': 1024, 'S': 128}  (S/M/L classes)
- Output lengths: {'L': 1024, 'M': 256, 'S': 64}
- Run policy: official (see config.json for warmups/measured runs)
- Telemetry interval: 100 ms

## 3x3 workload matrix

```text
              Output tokens
Input        64      256      1024
128          SS      SM       SL
1024         MS      MM       ML
4096         LS      LM       LL
```

## Concurrency scaling (facts + derived)

### Profile LL

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 876.97 | 15.036 | 66.64 | 63.03 | 0.95 | 0.00 |
| C2 | 881.35 | 16.303 | 61.39 | 116.58 | 0.87 | -0.08 |
| C4 | 889.56 | 16.022 | 62.41 | 236.54 | 0.89 | -0.06 |
| C8 | 956.87 | 24.810 | 40.31 | 309.98 | 0.58 | -0.40 |

### Profile LM

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 878.92 | 10.882 | 91.91 | 69.73 | 0.76 | 0.00 |
| C2 | 880.94 | 14.215 | 70.36 | 113.14 | 0.62 | -0.23 |
| C4 | 888.63 | 16.329 | 61.25 | 201.39 | 0.55 | -0.33 |
| C8 | 952.12 | 24.841 | 40.26 | 279.31 | 0.38 | -0.56 |

### Profile LS

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 879.69 | 10.097 | 99.04 | 41.54 | 0.42 | 0.00 |
| C2 | 886.25 | 12.393 | 80.69 | 75.44 | 0.38 | -0.19 |
| C4 | 891.81 | 15.694 | 63.72 | 132.60 | 0.33 | -0.36 |
| C8 | 953.75 | 25.037 | 39.94 | 195.00 | 0.25 | -0.60 |

### Profile ML

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 237.31 | 12.784 | 78.39 | 76.99 | 0.98 | 0.00 |
| C2 | 240.28 | 14.966 | 66.88 | 131.63 | 0.84 | -0.15 |
| C4 | 258.74 | 17.033 | 58.93 | 232.02 | 0.74 | -0.25 |
| C8 | 339.25 | 17.315 | 57.75 | 452.43 | 0.72 | -0.26 |

### Profile MM

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 237.47 | 10.182 | 98.56 | 90.21 | 0.92 | 0.00 |
| C2 | 241.07 | 12.092 | 82.85 | 150.86 | 0.77 | -0.16 |
| C4 | 258.55 | 20.557 | 48.67 | 180.54 | 0.46 | -0.51 |
| C8 | 316.19 | 16.782 | 59.59 | 441.05 | 0.56 | -0.40 |

### Profile MS

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 237.07 | 9.243 | 108.19 | 76.85 | 0.71 | 0.00 |
| C2 | 240.09 | 11.014 | 90.80 | 134.69 | 0.62 | -0.16 |
| C4 | 259.99 | 13.460 | 74.32 | 219.91 | 0.51 | -0.31 |
| C8 | 353.56 | 16.393 | 61.00 | 359.96 | 0.42 | -0.44 |

### Profile SL

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 70.03 | 11.034 | 90.65 | 90.09 | 0.99 | 0.00 |
| C2 | 71.29 | 12.929 | 77.34 | 153.84 | 0.85 | -0.15 |
| C4 | 87.61 | 14.481 | 69.06 | 274.46 | 0.76 | -0.24 |
| C8 | 285.67 | 17.964 | 55.67 | 439.06 | 0.61 | -0.39 |

### Profile SM

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 69.90 | 10.071 | 99.37 | 96.75 | 0.97 | 0.00 |
| C2 | 72.00 | 11.231 | 89.15 | 173.90 | 0.88 | -0.10 |
| C4 | 90.28 | 12.293 | 81.36 | 313.78 | 0.79 | -0.18 |
| C8 | 279.34 | 13.582 | 73.64 | 543.00 | 0.68 | -0.26 |

### Profile SS

| Concurrency | TTFT ms | TPOT ms | decode tok/s/req | agg decode tok/s | scaling eff. | req. throughput degr. |
|---|---|---|---|---|---|---|
| C1 | 69.44 | 9.132 | 109.51 | 97.73 | 0.89 | 0.00 |
| C2 | 72.15 | 9.670 | 103.41 | 182.14 | 0.83 | -0.06 |
| C4 | 92.77 | 10.870 | 91.99 | 325.50 | 0.74 | -0.16 |
| C8 | 282.30 | 13.717 | 73.15 | 426.90 | 0.49 | -0.33 |

scaling_efficiency(Cn) = aggregate_decode(Cn) / (n × decode(C1)); degradation = current/baseline − 1.

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

