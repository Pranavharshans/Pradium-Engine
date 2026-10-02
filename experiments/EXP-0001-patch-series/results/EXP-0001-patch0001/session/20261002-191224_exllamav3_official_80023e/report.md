# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-191224_exllamav3_official_80023e`
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
| SS | 69.28 | 71.97 | 68.81 | 96.09 | 84.41 | 0.12 |
| SM | 69.08 | 69.18 | 68.80 | 69.98 | 69.74 | 0.00 |
| SL | 69.70 | 70.64 | 69.42 | 78.52 | 74.91 | 0.04 |
| MS | 237.80 | 238.54 | 236.91 | 244.90 | 242.24 | 0.01 |
| MM | 238.84 | 239.24 | 236.73 | 248.27 | 244.20 | 0.01 |
| ML | 237.55 | 238.24 | 236.67 | 244.29 | 241.72 | 0.01 |
| LS | 876.31 | 876.92 | 874.45 | 882.00 | 880.59 | 0.00 |
| LM | 881.11 | 880.97 | 876.55 | 887.79 | 885.50 | 0.00 |
| LL | 877.45 | 877.48 | 873.93 | 883.12 | 881.94 | 0.00 |

### Prefill (tok/s, runtime-reported)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2115.0 | 2060.9 | 1560.9 | 2123.8 | 2123.7 | 0.1 |
| SM | 2116.8 | 2117.9 | 2111.2 | 2123.4 | 2122.8 | 0.0 |
| SL | 2098.9 | 2095.2 | 2075.7 | 2102.3 | 2102.0 | 0.0 |
| MS | 4489.2 | 4481.1 | 4402.6 | 4501.2 | 4497.8 | 0.0 |
| MM | 4477.8 | 4470.0 | 4365.2 | 4504.1 | 4500.2 | 0.0 |
| ML | 4493.8 | 4484.2 | 4364.4 | 4523.2 | 4516.6 | 0.0 |
| LS | 4730.4 | 4728.0 | 4704.6 | 4740.8 | 4737.5 | 0.0 |
| LM | 4707.7 | 4708.2 | 4686.7 | 4729.2 | 4724.1 | 0.0 |
| LL | 4727.1 | 4725.3 | 4693.1 | 4742.5 | 4741.6 | 0.0 |

### Sustained decode (tok/s per request)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 109.18 | 99.34 | 24.12 | 111.57 | 111.23 | 0.27 |
| SM | 105.15 | 91.25 | 40.46 | 109.56 | 108.95 | 0.27 |
| SL | 83.35 | 82.77 | 51.22 | 102.33 | 100.31 | 0.19 |
| MS | 108.75 | 105.44 | 74.00 | 110.90 | 110.83 | 0.11 |
| MM | 101.23 | 94.32 | 56.25 | 108.40 | 107.79 | 0.18 |
| ML | 72.70 | 75.78 | 53.22 | 101.02 | 98.25 | 0.19 |
| LS | 99.74 | 95.68 | 77.23 | 101.43 | 101.19 | 0.09 |
| LM | 88.39 | 84.66 | 55.82 | 98.35 | 97.86 | 0.18 |
| LL | 72.76 | 72.10 | 55.89 | 91.20 | 87.97 | 0.17 |

### TPOT (ms/token)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 9.159 | 12.526 | 8.963 | 41.459 | 27.628 | 0.813 |
| SM | 9.511 | 12.209 | 9.128 | 24.716 | 21.717 | 0.427 |
| SL | 12.004 | 12.567 | 9.773 | 19.525 | 17.509 | 0.232 |
| MS | 9.196 | 9.615 | 9.017 | 13.514 | 11.679 | 0.143 |
| MM | 9.896 | 11.041 | 9.225 | 17.778 | 15.688 | 0.245 |
| ML | 13.765 | 13.654 | 9.899 | 18.790 | 17.519 | 0.195 |
| LS | 10.026 | 10.550 | 9.859 | 12.948 | 12.741 | 0.109 |
| LM | 11.327 | 12.231 | 10.168 | 17.916 | 17.075 | 0.218 |
| LL | 13.768 | 14.259 | 10.964 | 17.893 | 17.654 | 0.176 |

### End-to-end latency (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 646.09 | 861.09 | 634.82 | 2681.49 | 1810.10 | 0.74 |
| SM | 2494.22 | 3182.42 | 2396.64 | 6371.77 | 5607.35 | 0.42 |
| SL | 12353.92 | 12926.65 | 10067.12 | 20043.13 | 17981.56 | 0.23 |
| MS | 817.48 | 844.28 | 805.01 | 1089.00 | 973.64 | 0.10 |
| MM | 2762.35 | 3054.73 | 2591.67 | 4772.21 | 4238.74 | 0.23 |
| ML | 14318.66 | 14206.67 | 10364.69 | 19466.87 | 18163.46 | 0.19 |
| LS | 1510.50 | 1541.55 | 1498.07 | 1691.38 | 1678.49 | 0.05 |
| LM | 3773.69 | 3999.76 | 3474.13 | 5447.11 | 5234.27 | 0.17 |
| LL | 14960.44 | 15464.50 | 12090.80 | 19183.51 | 18940.74 | 0.17 |

### ITL p95 (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 18.770 | 25.631 | 16.254 | 85.575 | 58.103 | 0.826 |
| SM | 20.020 | 33.657 | 19.149 | 83.815 | 82.768 | 0.775 |
| SL | 27.944 | 36.834 | 20.129 | 84.604 | 75.121 | 0.587 |
| MS | 18.599 | 21.526 | 16.544 | 51.797 | 37.231 | 0.496 |
| MM | 20.285 | 27.493 | 18.739 | 75.891 | 56.029 | 0.637 |
| ML | 43.220 | 45.365 | 19.919 | 76.167 | 73.890 | 0.464 |
| LS | 19.724 | 21.439 | 17.401 | 33.645 | 30.777 | 0.237 |
| LM | 21.675 | 30.685 | 19.040 | 71.131 | 68.782 | 0.653 |
| LL | 39.242 | 43.612 | 21.013 | 72.672 | 70.441 | 0.496 |

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
| SS | 1732.3 | 1732.4 | 1731.5 | 1733.4 | 1733.4 | 0.0 |
| SM | 1736.4 | 1736.2 | 1735.0 | 1736.9 | 1736.9 | 0.0 |
| SL | 1739.0 | 1738.9 | 1737.8 | 1740.2 | 1740.0 | 0.0 |
| MS | 1979.3 | 1979.3 | 1978.7 | 1979.8 | 1979.7 | 0.0 |
| MM | 1981.8 | 1981.7 | 1981.0 | 1981.8 | 1981.8 | 0.0 |
| ML | 1983.4 | 1983.6 | 1982.8 | 1984.9 | 1984.7 | 0.0 |
| LS | 2002.0 | 2002.0 | 2002.0 | 2002.0 | 2002.0 | 0.0 |
| LM | 2002.9 | 2002.9 | 2002.9 | 2003.1 | 2003.0 | 0.0 |
| LL | 2004.7 | 2005.0 | 2004.4 | 2006.0 | 2006.0 | 0.0 |

### CPU utilization avg (%)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 170.4 | 166.0 | 141.5 | 175.6 | 175.0 | 0.1 |
| SM | 172.1 | 167.1 | 144.0 | 177.0 | 176.9 | 0.1 |
| SL | 164.6 | 164.1 | 149.5 | 172.6 | 172.1 | 0.0 |
| MS | 157.6 | 156.8 | 151.0 | 161.6 | 160.2 | 0.0 |
| MM | 166.2 | 164.9 | 150.8 | 172.0 | 171.5 | 0.0 |
| ML | 159.9 | 161.0 | 149.4 | 171.9 | 170.8 | 0.0 |
| LS | 135.2 | 134.3 | 127.3 | 138.2 | 137.9 | 0.0 |
| LM | 154.4 | 154.4 | 145.7 | 159.7 | 159.5 | 0.0 |
| LL | 159.2 | 158.9 | 151.7 | 167.8 | 166.2 | 0.0 |

### GPU power avg (W, GPU-side only)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 118.7 | 107.8 | 60.3 | 119.9 | 119.8 | 0.2 |
| SM | 119.2 | 111.9 | 74.7 | 126.2 | 125.5 | 0.1 |
| SL | 113.0 | 112.2 | 85.5 | 128.7 | 127.0 | 0.1 |
| MS | 138.2 | 134.5 | 102.3 | 139.3 | 139.3 | 0.1 |
| MM | 128.3 | 126.2 | 94.1 | 138.0 | 137.4 | 0.1 |
| ML | 109.9 | 110.1 | 86.8 | 132.4 | 130.1 | 0.1 |
| LS | 151.8 | 150.1 | 143.9 | 152.8 | 152.8 | 0.0 |
| LM | 135.2 | 134.0 | 115.6 | 145.6 | 144.6 | 0.1 |
| LL | 116.7 | 116.3 | 100.1 | 133.9 | 131.8 | 0.1 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

