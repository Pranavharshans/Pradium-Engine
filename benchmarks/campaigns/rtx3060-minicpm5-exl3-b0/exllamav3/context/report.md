# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-095216_exllamav3_official_c72190`
- Runtime: `exllamav3` (1.5.3+cu128.torch2.10.0)
- Model: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Tokenizer: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Benchmark mode: `official` (run policy: `official`)
- Performance valid: `False`
- Raw records: 104

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

## Context scaling (facts)

| Input tokens | TTFT ms | prefill tok/s | decode tok/s | TPOT ms | peak VRAM MiB | statuses |
|---|---|---|---|---|---|---|
| 128 | 68.81 | 2124.2 | 101.00 | 9.906 | 2799.0 | {'SUCCESS': 10} |
| 512 | 132.23 | 4160.9 | 103.53 | 9.659 | 2911.0 | {'SUCCESS': 10} |
| 1024 | 237.53 | 4493.5 | 103.04 | 9.705 | 2951.0 | {'SUCCESS': 10} |
| 2048 | 414.28 | 5069.9 | 100.99 | 9.902 | 3031.0 | {'SUCCESS': 10} |
| 4096 | 877.54 | 4725.5 | 88.12 | 11.363 | 3031.0 | {'SUCCESS': 10} |
| 8192 | 2021.67 | 4076.4 | 87.70 | 11.403 | 3031.0 | {'SUCCESS': 10} |
| 16384 | 4930.08 | 3333.2 | 71.69 | 13.949 | 3031.0 | {'SUCCESS': 10} |
| 32768 | — | — | — | — | — | {'FAILED': 10} |

## Failures and unsupported measurements

| suite | profile | status | in tok | req out tok | error |
|---|---|---|---|---|---|
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |
| context | CTX-32768 | FAILED | 32768 | 256 | AssertionError: Job requires 130 pages (only 128 available) and cannot be enqueu |

### Unsupported / unavailable

- context/CTX-32768@C1: FAILED

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

