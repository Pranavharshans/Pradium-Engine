# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-090349_exllamav3_official_537aee`
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

### TTFT (ms) — first token minus submission

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 69.53 | 70.39 | 69.07 | 74.15 | 73.62 | 0.02 |
| SM | 69.01 | 69.85 | 68.59 | 77.38 | 73.94 | 0.04 |
| SL | 69.88 | 70.48 | 69.37 | 73.45 | 72.69 | 0.02 |
| MS | 236.34 | 236.34 | 235.18 | 237.56 | 237.29 | 0.00 |
| MM | 237.13 | 239.19 | 236.04 | 251.40 | 247.80 | 0.02 |
| ML | 237.49 | 237.52 | 236.35 | 239.12 | 238.78 | 0.00 |
| LS | 880.25 | 879.66 | 875.55 | 883.40 | 882.37 | 0.00 |
| LM | 878.99 | 878.56 | 875.00 | 880.35 | 880.22 | 0.00 |
| LL | 875.30 | 875.73 | 872.41 | 879.08 | 878.87 | 0.00 |

### Prefill (tok/s, runtime-reported)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2103.9 | 2096.2 | 2044.3 | 2108.2 | 2108.2 | 0.0 |
| SM | 2117.7 | 2100.7 | 1916.8 | 2130.6 | 2130.5 | 0.0 |
| SL | 2101.1 | 2099.1 | 2075.7 | 2115.0 | 2109.4 | 0.0 |
| MS | 4516.4 | 4517.1 | 4505.5 | 4534.5 | 4529.7 | 0.0 |
| MM | 4509.0 | 4473.5 | 4282.8 | 4519.7 | 4519.1 | 0.0 |
| ML | 4508.8 | 4502.8 | 4483.8 | 4518.3 | 4515.7 | 0.0 |
| LS | 4714.0 | 4719.2 | 4706.8 | 4739.9 | 4739.3 | 0.0 |
| LM | 4715.1 | 4718.4 | 4708.2 | 4736.1 | 4730.7 | 0.0 |
| LL | 4737.7 | 4734.1 | 4713.2 | 4752.0 | 4750.3 | 0.0 |

### Sustained decode (tok/s per request)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 109.80 | 105.63 | 73.16 | 112.97 | 112.55 | 0.11 |
| SM | 85.58 | 81.33 | 42.98 | 108.71 | 108.58 | 0.28 |
| SL | 96.45 | 88.76 | 62.91 | 102.62 | 102.47 | 0.18 |
| MS | 109.36 | 99.92 | 31.11 | 111.63 | 111.54 | 0.25 |
| MM | 101.31 | 89.30 | 54.05 | 109.52 | 108.52 | 0.24 |
| ML | 87.95 | 83.45 | 53.85 | 102.78 | 102.21 | 0.22 |
| LS | 99.81 | 97.74 | 79.21 | 102.01 | 101.62 | 0.07 |
| LM | 89.29 | 85.92 | 65.56 | 100.22 | 99.89 | 0.17 |
| LL | 77.19 | 76.48 | 51.04 | 94.11 | 94.03 | 0.19 |

### TPOT (ms/token)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 9.108 | 9.614 | 8.852 | 13.668 | 11.963 | 0.151 |
| SM | 11.699 | 13.439 | 9.199 | 23.267 | 21.308 | 0.345 |
| SL | 10.368 | 11.658 | 9.744 | 15.896 | 15.511 | 0.208 |
| MS | 9.145 | 11.601 | 8.958 | 32.144 | 22.311 | 0.623 |
| MM | 9.872 | 11.941 | 9.131 | 18.501 | 17.425 | 0.290 |
| ML | 11.501 | 12.589 | 9.730 | 18.571 | 17.490 | 0.246 |
| LS | 10.019 | 10.283 | 9.803 | 12.625 | 11.565 | 0.081 |
| LM | 11.327 | 11.964 | 9.978 | 15.254 | 14.934 | 0.178 |
| LL | 12.963 | 13.566 | 10.626 | 19.592 | 18.820 | 0.218 |

### End-to-end latency (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 644.73 | 676.05 | 627.19 | 930.18 | 825.02 | 0.14 |
| SM | 3056.44 | 3496.70 | 2414.60 | 6001.67 | 5502.36 | 0.34 |
| SL | 10676.98 | 11996.83 | 10040.18 | 16335.39 | 15939.19 | 0.21 |
| MS | 811.57 | 967.22 | 800.78 | 2261.31 | 1642.43 | 0.47 |
| MM | 2754.25 | 3284.23 | 2579.74 | 4953.80 | 4682.65 | 0.27 |
| ML | 12002.67 | 13115.61 | 10192.75 | 19234.30 | 18129.86 | 0.24 |
| LS | 1511.31 | 1527.45 | 1498.72 | 1675.87 | 1608.68 | 0.03 |
| LM | 3767.44 | 3929.32 | 3423.35 | 4767.37 | 4686.85 | 0.14 |
| LL | 14136.55 | 14753.57 | 11746.06 | 20921.22 | 20128.51 | 0.21 |

### ITL p95 (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 18.107 | 20.938 | 15.806 | 49.517 | 36.045 | 0.483 |
| SM | 26.332 | 39.850 | 19.438 | 73.941 | 71.500 | 0.580 |
| SL | 21.464 | 35.401 | 19.954 | 82.694 | 73.986 | 0.678 |
| MS | 18.802 | 24.809 | 17.490 | 79.294 | 52.678 | 0.772 |
| MM | 20.518 | 34.971 | 19.045 | 75.844 | 71.030 | 0.658 |
| ML | 27.992 | 38.404 | 19.816 | 70.913 | 67.800 | 0.555 |
| LS | 19.106 | 19.989 | 17.093 | 28.437 | 25.065 | 0.162 |
| LM | 22.644 | 30.196 | 18.588 | 63.721 | 57.271 | 0.523 |
| LL | 29.009 | 35.774 | 19.788 | 71.829 | 69.973 | 0.538 |

### Peak VRAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 0.0 |
| SM | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 0.0 |
| SL | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 2799.0 | 0.0 |
| MS | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 0.0 |
| MM | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 0.0 |
| ML | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 2951.0 | 0.0 |
| LS | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 0.0 |
| LM | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 0.0 |
| LL | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 3031.0 | 0.0 |

### Peak host RAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 1720.5 | 1720.8 | 1719.8 | 1722.0 | 1722.0 | 0.0 |
| SM | 1725.1 | 1724.8 | 1724.1 | 1725.3 | 1725.2 | 0.0 |
| SL | 1728.1 | 1728.1 | 1727.0 | 1729.0 | 1728.9 | 0.0 |
| MS | 1969.1 | 1969.4 | 1968.9 | 1970.5 | 1970.4 | 0.0 |
| MM | 1972.0 | 1972.0 | 1972.0 | 1972.0 | 1972.0 | 0.0 |
| ML | 1984.1 | 1984.0 | 1982.7 | 1985.1 | 1985.0 | 0.0 |
| LS | 2002.4 | 2002.4 | 2002.4 | 2002.4 | 2002.4 | 0.0 |
| LM | 2002.4 | 2002.8 | 2002.4 | 2003.7 | 2003.7 | 0.0 |
| LL | 2004.9 | 2004.9 | 2003.7 | 2006.7 | 2006.4 | 0.0 |

### CPU utilization avg (%)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 171.6 | 169.1 | 142.1 | 174.3 | 174.1 | 0.1 |
| SM | 161.7 | 163.0 | 149.0 | 176.9 | 176.3 | 0.1 |
| SL | 169.8 | 165.9 | 149.2 | 173.7 | 173.3 | 0.1 |
| MS | 157.0 | 155.3 | 135.6 | 160.2 | 160.1 | 0.0 |
| MM | 167.1 | 164.1 | 151.9 | 171.0 | 171.0 | 0.0 |
| ML | 166.2 | 163.9 | 150.4 | 171.5 | 171.4 | 0.0 |
| LS | 136.4 | 135.3 | 129.4 | 137.5 | 137.3 | 0.0 |
| LM | 155.0 | 154.4 | 148.3 | 159.1 | 159.0 | 0.0 |
| LL | 160.9 | 160.4 | 148.9 | 168.6 | 168.4 | 0.0 |

### GPU power avg (W, GPU-side only)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 116.3 | 114.0 | 100.4 | 118.8 | 118.7 | 0.1 |
| SM | 111.1 | 104.8 | 78.5 | 124.0 | 123.4 | 0.2 |
| SL | 123.5 | 117.9 | 97.3 | 130.6 | 129.8 | 0.1 |
| MS | 136.4 | 130.9 | 97.6 | 137.9 | 137.4 | 0.1 |
| MM | 120.5 | 119.8 | 97.8 | 134.7 | 134.0 | 0.1 |
| ML | 119.8 | 116.2 | 90.4 | 132.4 | 132.4 | 0.1 |
| LS | 152.1 | 151.5 | 148.8 | 152.5 | 152.4 | 0.0 |
| LM | 135.5 | 134.3 | 120.8 | 145.5 | 145.1 | 0.1 |
| LL | 120.7 | 119.4 | 93.8 | 137.3 | 136.5 | 0.1 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

