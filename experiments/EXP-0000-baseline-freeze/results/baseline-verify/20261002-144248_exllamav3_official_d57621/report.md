# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-144248_exllamav3_official_d57621`
- Runtime: `exllamav3` (1.5.3+cu128.torch2.10.0)
- Model: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Tokenizer: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Benchmark mode: `official` (run policy: `official`)
- Performance valid: `True`
- Raw records: 39

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
| Git commit | 04fa9206ee3e1252255d138af781cfe80a8075f8 |

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

- note `supports_cache_reset`: v1.5.3 has no public free_cache. Rebuilding the Generator (fresh PageTable over the same Cache) is the only reset whose effect is observable: cached_tokens == 0 and the next identical prompt reports alloc_cached_pages == 0. Verified by the adapter conformance tests.
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

### TTFT (ms) — first token minus submission

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 68.78 | 69.42 | 68.52 | 74.45 | 72.32 | 0.03 |
| MM | 235.79 | 235.77 | 234.38 | 237.25 | 236.98 | 0.00 |
| LL | 882.96 | 883.33 | 876.70 | 889.42 | 888.24 | 0.00 |

### Prefill (tok/s, runtime-reported)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2127.6 | 2112.0 | 1974.0 | 2132.4 | 2132.0 | 0.0 |
| MM | 4529.0 | 4529.2 | 4500.0 | 4550.2 | 4546.2 | 0.0 |
| LL | 4693.4 | 4695.2 | 4675.4 | 4726.4 | 4718.0 | 0.0 |

### Sustained decode (tok/s per request)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 112.42 | 106.90 | 70.41 | 114.65 | 114.07 | 0.12 |
| MM | 102.02 | 101.09 | 90.34 | 108.77 | 108.65 | 0.07 |
| LL | 75.28 | 74.70 | 55.65 | 92.82 | 91.47 | 0.17 |

### TPOT (ms/token)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 8.895 | 9.539 | 8.722 | 14.203 | 12.162 | 0.174 |
| MM | 9.802 | 9.932 | 9.193 | 11.069 | 11.042 | 0.068 |
| LL | 13.286 | 13.744 | 10.773 | 17.971 | 17.216 | 0.174 |

### End-to-end latency (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 629.01 | 670.40 | 618.01 | 963.72 | 837.63 | 0.16 |
| MM | 2734.88 | 2768.32 | 2580.10 | 3058.72 | 3051.07 | 0.06 |
| LL | 14477.44 | 14942.99 | 11902.01 | 19265.13 | 18493.20 | 0.16 |

### ITL p95 (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 18.857 | 20.520 | 15.484 | 40.342 | 31.312 | 0.346 |
| MM | 19.964 | 20.142 | 19.199 | 22.730 | 21.773 | 0.050 |
| LL | 34.064 | 39.663 | 19.181 | 63.629 | 62.859 | 0.464 |

### Peak VRAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 0.0 |
| MM | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 0.0 |
| LL | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 0.0 |

### Peak host RAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 1725.4 | 1726.1 | 1725.3 | 1727.3 | 1727.3 | 0.0 |
| MM | 1969.8 | 1969.5 | 1968.7 | 1970.0 | 1970.0 | 0.0 |
| LL | 1988.6 | 1988.7 | 1987.5 | 1990.2 | 1990.0 | 0.0 |

### CPU utilization avg (%)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 173.6 | 171.5 | 155.0 | 174.6 | 174.5 | 0.0 |
| MM | 169.7 | 167.4 | 159.1 | 171.7 | 171.5 | 0.0 |
| LL | 160.0 | 159.7 | 151.9 | 167.2 | 165.9 | 0.0 |

### GPU power avg (W, GPU-side only)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 121.9 | 118.5 | 101.0 | 123.6 | 123.5 | 0.1 |
| MM | 130.3 | 128.6 | 119.3 | 133.2 | 133.0 | 0.0 |
| LL | 120.9 | 119.4 | 98.7 | 135.0 | 134.4 | 0.1 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

