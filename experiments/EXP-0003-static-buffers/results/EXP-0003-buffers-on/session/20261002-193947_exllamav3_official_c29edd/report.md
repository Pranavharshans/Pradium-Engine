# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-193947_exllamav3_official_c29edd`
- Runtime: `exllamav3` (1.5.3+cu128.torch2.10.0)
- Model: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Tokenizer: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` rev `e36fbb568f466739d7c244a5100d01acf93d3b7f`
- Benchmark mode: `official` (run policy: `official`)
- Performance valid: `True`
- Raw records: 117

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
| SS | 69.22 | 70.00 | 68.85 | 77.13 | 73.85 | 0.04 |
| SM | 69.16 | 72.08 | 68.75 | 98.02 | 85.38 | 0.13 |
| SL | 71.52 | 76.64 | 69.16 | 104.91 | 96.73 | 0.15 |
| MS | 240.15 | 245.55 | 239.95 | 285.43 | 268.88 | 0.06 |
| MM | 240.33 | 242.04 | 239.27 | 257.30 | 250.42 | 0.02 |
| ML | 239.43 | 239.67 | 238.50 | 242.98 | 241.89 | 0.01 |
| LS | 900.30 | 900.84 | 897.36 | 908.94 | 906.20 | 0.00 |
| LM | 898.62 | 899.39 | 895.60 | 905.76 | 904.08 | 0.00 |
| LL | 898.33 | 900.18 | 895.89 | 912.32 | 910.27 | 0.01 |

### Prefill (tok/s, runtime-reported)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2105.8 | 2083.5 | 1900.8 | 2111.0 | 2110.7 | 0.0 |
| SM | 2103.7 | 2045.1 | 1511.7 | 2111.0 | 2110.5 | 0.1 |
| SL | 2076.4 | 2010.8 | 1609.0 | 2105.2 | 2103.2 | 0.1 |
| MS | 4505.3 | 4438.6 | 3829.9 | 4512.5 | 4510.4 | 0.0 |
| MM | 4504.5 | 4475.2 | 4186.3 | 4521.7 | 4519.7 | 0.0 |
| ML | 4519.4 | 4517.4 | 4480.0 | 4532.6 | 4532.2 | 0.0 |
| LS | 4696.7 | 4693.9 | 4657.3 | 4712.4 | 4710.5 | 0.0 |
| LM | 4704.6 | 4701.5 | 4669.2 | 4719.2 | 4717.0 | 0.0 |
| LL | 4707.5 | 4699.7 | 4645.7 | 4721.0 | 4717.9 | 0.0 |

### Sustained decode (tok/s per request)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 109.36 | 96.49 | 26.90 | 115.69 | 115.05 | 0.31 |
| SM | 86.37 | 78.11 | 40.61 | 106.01 | 105.59 | 0.32 |
| SL | 69.11 | 70.07 | 57.33 | 87.82 | 84.87 | 0.15 |
| MS | 80.50 | 78.79 | 67.85 | 80.79 | 80.71 | 0.05 |
| MM | 72.74 | 67.03 | 34.21 | 76.40 | 76.12 | 0.21 |
| ML | 60.91 | 59.99 | 41.70 | 68.44 | 68.35 | 0.14 |
| LS | 96.22 | 80.84 | 23.65 | 99.94 | 99.72 | 0.37 |
| LM | 74.18 | 69.66 | 26.25 | 99.63 | 99.03 | 0.41 |
| LL | 70.49 | 69.18 | 46.10 | 87.36 | 86.91 | 0.23 |

### TPOT (ms/token)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 9.144 | 12.733 | 8.644 | 37.174 | 28.230 | 0.705 |
| SM | 11.590 | 14.425 | 9.433 | 24.625 | 24.356 | 0.402 |
| SL | 14.471 | 14.552 | 11.386 | 17.443 | 17.338 | 0.147 |
| MS | 12.422 | 12.726 | 12.378 | 14.738 | 14.014 | 0.058 |
| MM | 13.747 | 15.921 | 13.089 | 29.234 | 25.631 | 0.331 |
| ML | 16.419 | 17.022 | 14.611 | 23.983 | 22.084 | 0.168 |
| LS | 10.393 | 16.172 | 10.006 | 42.278 | 38.490 | 0.727 |
| LM | 14.283 | 17.570 | 10.038 | 38.102 | 32.172 | 0.525 |
| LL | 14.372 | 15.203 | 11.447 | 21.691 | 20.581 | 0.242 |

### End-to-end latency (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 645.02 | 872.17 | 613.88 | 2411.52 | 1847.90 | 0.65 |
| SM | 3024.99 | 3750.56 | 2495.86 | 6348.10 | 6279.86 | 0.39 |
| SL | 14885.62 | 14963.53 | 11717.50 | 17914.48 | 17806.42 | 0.15 |
| MS | 1022.76 | 1047.32 | 1019.79 | 1168.55 | 1126.80 | 0.05 |
| MM | 3746.06 | 4301.98 | 3594.97 | 7695.06 | 6775.72 | 0.31 |
| ML | 17037.80 | 17653.46 | 15185.76 | 24773.54 | 22831.62 | 0.17 |
| LS | 1552.88 | 1919.65 | 1532.97 | 3564.65 | 3324.60 | 0.39 |
| LM | 4541.15 | 5379.83 | 3458.08 | 10612.74 | 9103.08 | 0.44 |
| LL | 15599.49 | 16452.65 | 12606.85 | 23088.46 | 21958.83 | 0.23 |

### ITL p95 (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 21.375 | 32.771 | 17.911 | 94.747 | 83.663 | 0.819 |
| SM | 24.194 | 44.722 | 19.604 | 88.559 | 84.540 | 0.662 |
| SL | 36.913 | 40.430 | 21.619 | 70.737 | 67.994 | 0.473 |
| MS | 19.390 | 19.605 | 18.015 | 24.124 | 22.057 | 0.085 |
| MM | 22.462 | 32.126 | 19.483 | 83.721 | 74.985 | 0.704 |
| ML | 25.554 | 33.136 | 19.694 | 73.202 | 67.937 | 0.561 |
| LS | 20.084 | 35.754 | 17.425 | 99.631 | 96.918 | 0.902 |
| LM | 45.698 | 55.033 | 19.367 | 110.050 | 101.671 | 0.691 |
| LL | 45.853 | 47.621 | 21.332 | 80.267 | 77.534 | 0.546 |

### Peak VRAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 0.0 |
| SM | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 0.0 |
| SL | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 0.0 |
| MS | 2949.0 | 2949.0 | 2949.0 | 2949.0 | 2949.0 | 0.0 |
| MM | 2949.0 | 2949.0 | 2949.0 | 2949.0 | 2949.0 | 0.0 |
| ML | 2949.0 | 2949.0 | 2949.0 | 2949.0 | 2949.0 | 0.0 |
| LS | 3035.0 | 3035.0 | 3035.0 | 3035.0 | 3035.0 | 0.0 |
| LM | 3035.0 | 3035.0 | 3035.0 | 3035.0 | 3035.0 | 0.0 |
| LL | 3035.0 | 3035.0 | 3035.0 | 3035.0 | 3035.0 | 0.0 |

### Peak host RAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 1724.7 | 1725.0 | 1724.7 | 1725.4 | 1725.4 | 0.0 |
| SM | 1728.9 | 1728.5 | 1727.6 | 1728.9 | 1728.9 | 0.0 |
| SL | 1731.9 | 1732.0 | 1730.7 | 1733.4 | 1733.3 | 0.0 |
| MS | 1972.9 | 1973.0 | 1971.9 | 1974.2 | 1974.0 | 0.0 |
| MM | 1975.6 | 1975.6 | 1975.6 | 1975.7 | 1975.7 | 0.0 |
| ML | 1978.1 | 1978.2 | 1977.2 | 1979.8 | 1979.6 | 0.0 |
| LS | 2013.0 | 2012.9 | 2012.3 | 2013.2 | 2013.1 | 0.0 |
| LM | 2015.9 | 2015.7 | 2015.0 | 2015.9 | 2015.9 | 0.0 |
| LL | 2017.7 | 2017.8 | 2016.4 | 2019.3 | 2019.0 | 0.0 |

### CPU utilization avg (%)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 173.8 | 166.1 | 134.7 | 177.4 | 176.4 | 0.1 |
| SM | 166.3 | 161.3 | 143.5 | 173.7 | 173.6 | 0.1 |
| SL | 162.5 | 163.0 | 156.3 | 173.5 | 170.7 | 0.0 |
| MS | 159.8 | 158.2 | 152.3 | 162.7 | 161.9 | 0.0 |
| MM | 167.8 | 165.1 | 148.1 | 172.4 | 171.8 | 0.0 |
| ML | 167.5 | 166.7 | 152.7 | 173.1 | 172.9 | 0.0 |
| LS | 135.6 | 133.6 | 122.5 | 138.2 | 137.6 | 0.0 |
| LM | 151.2 | 150.2 | 137.0 | 159.4 | 159.3 | 0.1 |
| LL | 159.5 | 158.4 | 147.7 | 165.9 | 165.9 | 0.0 |

### GPU power avg (W, GPU-side only)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 117.0 | 108.2 | 69.0 | 126.1 | 125.8 | 0.2 |
| SM | 115.0 | 104.9 | 73.1 | 124.8 | 124.3 | 0.2 |
| SL | 101.6 | 102.5 | 91.6 | 117.6 | 115.2 | 0.1 |
| MS | 122.1 | 119.4 | 98.4 | 122.7 | 122.5 | 0.1 |
| MM | 111.1 | 104.9 | 78.6 | 113.4 | 113.4 | 0.1 |
| ML | 99.7 | 97.8 | 80.0 | 105.3 | 105.1 | 0.1 |
| LS | 147.6 | 139.2 | 103.0 | 151.6 | 151.4 | 0.1 |
| LM | 117.2 | 116.8 | 79.8 | 142.6 | 142.1 | 0.2 |
| LL | 116.6 | 113.6 | 90.3 | 130.7 | 130.2 | 0.1 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

