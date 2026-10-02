# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-110241_exllamav3_official_7cf380`
- Runtime: `exllamav3` (1.5.3+cu128.torch2.10.0)
- Model: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Tokenizer: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Benchmark mode: `official` (run policy: `official`)
- Performance valid: `True`
- Raw records: 1440

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

## Prefix / KV reuse (facts + derived)

| Reuse | scenario | TTFT ms | prefill ms | decode tok/s | reused tok | TTFT speedup | prefill speedup |
|---|---|---|---|---|---|---|---|
| 0% | cross_request | — | — | — | — | — | — |
| 0% | cross_request | 237.64 | 227.94 | 98.48 | — | 1.00 | 1.00 |
| 0% | cross_session | — | — | — | — | — | — |
| 0% | cross_session | 237.59 | 227.97 | 102.26 | — | 1.00 | 1.00 |
| 0% | same_session | — | — | — | — | — | — |
| 0% | same_session | 237.76 | 228.29 | 97.82 | — | 1.00 | 1.00 |
| 25% | cross_request | — | — | — | — | — | — |
| 25% | cross_request | 189.21 | 179.70 | 99.23 | — | 1.26 | 1.27 |
| 25% | cross_session | — | — | — | — | — | — |
| 25% | cross_session | 238.35 | 228.83 | 102.32 | — | 1.00 | 1.00 |
| 25% | same_session | — | — | — | — | — | — |
| 25% | same_session | 189.31 | 179.68 | 104.34 | — | 1.26 | 1.27 |
| 50% | cross_request | — | — | — | — | — | — |
| 50% | cross_request | 138.34 | 128.81 | 101.13 | — | 1.72 | 1.77 |
| 50% | cross_session | — | — | — | — | — | — |
| 50% | cross_session | 238.01 | 228.27 | 102.51 | — | 1.00 | 1.00 |
| 50% | same_session | — | — | — | — | — | — |
| 50% | same_session | 138.06 | 128.68 | 102.82 | — | 1.72 | 1.77 |
| 75% | cross_request | — | — | — | — | — | — |
| 75% | cross_request | 93.14 | 83.83 | 97.03 | — | 2.55 | 2.72 |
| 75% | cross_session | — | — | — | — | — | — |
| 75% | cross_session | 238.02 | 228.39 | 97.14 | — | 1.00 | 1.00 |
| 75% | same_session | — | — | — | — | — | — |
| 75% | same_session | 93.10 | 83.74 | 102.22 | — | 2.55 | 2.73 |
| 90% | cross_request | — | — | — | — | — | — |
| 90% | cross_request | 66.82 | 57.20 | 104.98 | — | 3.56 | 3.98 |
| 90% | cross_session | — | — | — | — | — | — |
| 90% | cross_session | 238.32 | 228.78 | 92.59 | — | 1.00 | 1.00 |
| 90% | same_session | — | — | — | — | — | — |
| 90% | same_session | 66.66 | 57.21 | 101.17 | — | 3.57 | 3.99 |
| 100% | cross_request | — | — | — | — | — | — |
| 100% | cross_request | 21.55 | 12.24 | 101.53 | — | 11.03 | 18.63 |
| 100% | cross_session | — | — | — | — | — | — |
| 100% | cross_session | 237.78 | 228.23 | 100.02 | — | 1.00 | 1.00 |
| 100% | same_session | — | — | — | — | — | — |
| 100% | same_session | 21.93 | 12.35 | 102.44 | — | 10.84 | 18.49 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

