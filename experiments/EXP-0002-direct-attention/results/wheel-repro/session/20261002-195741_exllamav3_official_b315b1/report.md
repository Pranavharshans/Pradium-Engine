# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-195741_exllamav3_official_b315b1`
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
| Git commit | f1c1c4862af82584932d22a43f0d9efcea2b3e63 |

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
| SM | 68.81 | 69.00 | 68.63 | 70.09 | 69.84 | 0.01 |
| LM | 874.16 | 873.65 | 869.78 | 877.10 | 876.91 | 0.00 |
| LL | 877.45 | 878.06 | 875.34 | 884.45 | 882.34 | 0.00 |

### Prefill (tok/s, runtime-reported)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 2124.0 | 2124.6 | 2114.4 | 2131.8 | 2131.5 | 0.0 |
| LM | 4741.9 | 4745.3 | 4729.5 | 4765.2 | 4764.0 | 0.0 |
| LL | 4724.9 | 4721.6 | 4689.1 | 4735.7 | 4731.6 | 0.0 |

### Sustained decode (tok/s per request)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 100.31 | 92.66 | 53.95 | 108.42 | 108.27 | 0.21 |
| LM | 97.92 | 87.85 | 47.90 | 100.16 | 99.68 | 0.21 |
| LL | 79.81 | 76.88 | 57.84 | 93.46 | 91.98 | 0.18 |

### TPOT (ms/token)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 9.970 | 11.390 | 9.223 | 18.534 | 17.370 | 0.282 |
| LM | 10.213 | 12.096 | 9.984 | 20.876 | 19.109 | 0.309 |
| LL | 12.534 | 13.414 | 10.700 | 17.288 | 17.208 | 0.190 |

### End-to-end latency (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 2611.00 | 2973.38 | 2422.00 | 4795.11 | 4498.56 | 0.28 |
| LM | 3478.41 | 3958.25 | 3421.68 | 6200.43 | 5746.81 | 0.24 |
| LL | 13698.82 | 14600.95 | 11822.90 | 18570.22 | 18486.24 | 0.18 |

### ITL p95 (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 21.347 | 31.214 | 19.202 | 75.561 | 70.940 | 0.669 |
| LM | 20.536 | 28.678 | 18.032 | 63.401 | 62.659 | 0.624 |
| LL | 23.780 | 38.866 | 19.706 | 78.494 | 71.471 | 0.592 |

### Peak VRAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 0.0 |
| LM | 2979.0 | 2979.0 | 2979.0 | 2979.0 | 2979.0 | 0.0 |
| LL | 2979.0 | 2979.0 | 2979.0 | 2979.0 | 2979.0 | 0.0 |

### Peak host RAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 1726.3 | 1726.3 | 1725.8 | 1726.7 | 1726.6 | 0.0 |
| LM | 1761.8 | 1761.9 | 1761.5 | 1762.7 | 1762.5 | 0.0 |
| LL | 1781.5 | 1781.3 | 1764.4 | 1797.6 | 1797.6 | 0.0 |

### CPU utilization avg (%)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 169.6 | 167.4 | 152.2 | 175.6 | 175.5 | 0.1 |
| LM | 158.6 | 155.1 | 143.9 | 159.4 | 159.4 | 0.0 |
| LL | 162.1 | 160.6 | 152.2 | 167.7 | 167.1 | 0.0 |

### GPU power avg (W, GPU-side only)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SM | 118.6 | 114.6 | 92.4 | 126.5 | 126.2 | 0.1 |
| LM | 141.8 | 134.0 | 97.7 | 145.0 | 144.9 | 0.1 |
| LL | 124.1 | 120.8 | 100.6 | 137.4 | 135.7 | 0.1 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

