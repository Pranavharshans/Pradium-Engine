# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-200233_exllamav3_official_014627`
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
| Git commit | 9350de11edd6638ff1b2d97f0f9471fbb210cad0 |

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
| SS | 70.00 | 72.26 | 69.04 | 79.60 | 79.52 | 0.06 |
| SM | 69.53 | 72.86 | 69.10 | 93.57 | 86.63 | 0.11 |
| SL | 69.47 | 70.62 | 69.07 | 78.66 | 75.26 | 0.04 |
| MS | 238.43 | 239.17 | 237.56 | 242.49 | 242.33 | 0.01 |
| MM | 237.25 | 237.30 | 236.38 | 238.77 | 238.44 | 0.00 |
| ML | 237.82 | 238.87 | 235.93 | 252.32 | 246.12 | 0.02 |
| LS | 880.60 | 880.42 | 878.30 | 882.08 | 881.81 | 0.00 |
| LM | 881.52 | 881.69 | 879.77 | 885.68 | 884.00 | 0.00 |
| LL | 876.98 | 876.49 | 873.18 | 879.63 | 879.58 | 0.00 |

### Prefill (tok/s, runtime-reported)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2105.9 | 2086.9 | 1931.6 | 2108.3 | 2107.9 | 0.0 |
| SM | 2101.4 | 2029.4 | 1628.9 | 2104.5 | 2104.2 | 0.1 |
| SL | 2101.1 | 2081.2 | 1902.4 | 2105.2 | 2104.9 | 0.0 |
| MS | 4488.5 | 4485.0 | 4440.3 | 4499.6 | 4495.9 | 0.0 |
| MM | 4505.7 | 4501.1 | 4483.2 | 4512.2 | 4511.7 | 0.0 |
| ML | 4497.4 | 4470.7 | 4215.4 | 4519.3 | 4517.3 | 0.0 |
| LS | 4710.0 | 4710.3 | 4705.0 | 4718.8 | 4716.2 | 0.0 |
| LM | 4703.1 | 4703.9 | 4697.6 | 4713.1 | 4712.0 | 0.0 |
| LL | 4727.9 | 4730.2 | 4711.4 | 4750.0 | 4748.0 | 0.0 |

### Sustained decode (tok/s per request)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 108.88 | 98.83 | 18.90 | 116.30 | 114.48 | 0.29 |
| SM | 104.27 | 101.81 | 82.03 | 106.03 | 106.02 | 0.07 |
| SL | 74.97 | 74.88 | 63.57 | 87.32 | 84.21 | 0.09 |
| MS | 104.84 | 95.59 | 45.25 | 109.14 | 108.89 | 0.22 |
| MM | 91.04 | 86.34 | 39.69 | 106.06 | 105.60 | 0.25 |
| ML | 85.47 | 77.90 | 41.35 | 94.90 | 94.29 | 0.22 |
| LS | 95.31 | 87.51 | 22.40 | 99.78 | 99.72 | 0.27 |
| LM | 90.64 | 85.11 | 69.99 | 96.55 | 96.44 | 0.13 |
| LL | 71.99 | 72.57 | 55.70 | 90.28 | 88.79 | 0.18 |

### TPOT (ms/token)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 9.185 | 13.680 | 8.598 | 52.918 | 33.842 | 1.009 |
| SM | 9.591 | 9.875 | 9.432 | 12.190 | 11.208 | 0.084 |
| SL | 13.339 | 13.457 | 11.452 | 15.731 | 15.250 | 0.093 |
| MS | 9.539 | 11.242 | 9.162 | 22.100 | 18.426 | 0.362 |
| MM | 10.991 | 12.638 | 9.429 | 25.196 | 21.209 | 0.386 |
| ML | 11.700 | 13.668 | 10.538 | 24.183 | 20.977 | 0.311 |
| LS | 10.492 | 14.002 | 10.022 | 44.637 | 30.131 | 0.770 |
| LM | 11.034 | 11.940 | 10.357 | 14.289 | 14.160 | 0.137 |
| LL | 13.894 | 14.208 | 11.077 | 17.952 | 17.764 | 0.186 |

### End-to-end latency (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 648.07 | 934.09 | 611.22 | 3409.31 | 2205.64 | 0.93 |
| SM | 2515.26 | 2590.99 | 2476.85 | 3177.68 | 2938.28 | 0.08 |
| SL | 13715.30 | 13837.13 | 11784.46 | 16171.17 | 15674.72 | 0.09 |
| MS | 838.64 | 947.39 | 817.60 | 1629.97 | 1399.22 | 0.27 |
| MM | 3039.84 | 3460.07 | 2640.73 | 6661.78 | 5645.48 | 0.36 |
| ML | 12207.92 | 14221.31 | 11016.21 | 24977.20 | 21697.14 | 0.31 |
| LS | 1542.29 | 1762.57 | 1512.18 | 3692.01 | 2777.83 | 0.39 |
| LM | 3694.72 | 3926.40 | 3521.97 | 4524.69 | 4492.34 | 0.11 |
| LL | 15091.22 | 15410.98 | 12210.94 | 19244.71 | 19051.84 | 0.18 |

### ITL p95 (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 18.616 | 26.065 | 17.264 | 92.842 | 60.786 | 0.902 |
| SM | 20.946 | 21.377 | 19.574 | 26.627 | 24.815 | 0.098 |
| SL | 38.926 | 41.376 | 22.622 | 61.858 | 60.602 | 0.338 |
| MS | 18.762 | 25.409 | 17.642 | 68.520 | 54.525 | 0.641 |
| MM | 21.691 | 32.724 | 19.048 | 79.591 | 77.636 | 0.723 |
| ML | 23.474 | 39.947 | 21.222 | 88.055 | 81.019 | 0.637 |
| LS | 20.164 | 27.612 | 17.987 | 94.217 | 62.248 | 0.849 |
| LM | 21.266 | 24.925 | 19.766 | 43.027 | 38.285 | 0.305 |
| LL | 35.111 | 44.961 | 20.944 | 82.833 | 79.743 | 0.555 |

### Peak VRAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 0.0 |
| SM | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 2797.0 | 0.0 |
| SL | 2805.0 | 2805.0 | 2805.0 | 2805.0 | 2805.0 | 0.0 |
| MS | 2957.0 | 2957.0 | 2957.0 | 2957.0 | 2957.0 | 0.0 |
| MM | 2957.0 | 2957.0 | 2957.0 | 2957.0 | 2957.0 | 0.0 |
| ML | 2957.0 | 2957.0 | 2957.0 | 2957.0 | 2957.0 | 0.0 |
| LS | 3037.0 | 3037.0 | 3037.0 | 3037.0 | 3037.0 | 0.0 |
| LM | 3037.0 | 3037.0 | 3037.0 | 3037.0 | 3037.0 | 0.0 |
| LL | 3037.0 | 3037.0 | 3037.0 | 3037.0 | 3037.0 | 0.0 |

### Peak host RAM (MiB)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 1724.9 | 1725.1 | 1724.8 | 1725.5 | 1725.5 | 0.0 |
| SM | 1729.0 | 1728.6 | 1727.5 | 1729.0 | 1729.0 | 0.0 |
| SL | 1736.7 | 1736.7 | 1735.5 | 1737.9 | 1737.9 | 0.0 |
| MS | 1977.4 | 1977.4 | 1976.3 | 1978.7 | 1978.6 | 0.0 |
| MM | 1980.2 | 1980.2 | 1980.2 | 1980.2 | 1980.2 | 0.0 |
| ML | 1981.4 | 1981.8 | 1981.2 | 1983.2 | 1983.0 | 0.0 |
| LS | 2000.4 | 2000.4 | 2000.4 | 2000.4 | 2000.4 | 0.0 |
| LM | 2000.7 | 2000.7 | 2000.4 | 2001.0 | 2001.0 | 0.0 |
| LL | 2002.8 | 2003.0 | 2001.6 | 2004.7 | 2004.6 | 0.0 |

### CPU utilization avg (%)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 170.9 | 166.9 | 140.4 | 173.7 | 173.6 | 0.1 |
| SM | 175.5 | 173.8 | 160.6 | 176.4 | 176.2 | 0.0 |
| SL | 162.7 | 162.7 | 158.2 | 167.8 | 166.6 | 0.0 |
| MS | 157.9 | 156.0 | 145.4 | 161.0 | 160.3 | 0.0 |
| MM | 166.5 | 163.8 | 146.0 | 171.1 | 171.0 | 0.1 |
| ML | 165.6 | 162.8 | 146.4 | 170.7 | 170.4 | 0.0 |
| LS | 135.5 | 134.8 | 127.1 | 138.4 | 138.2 | 0.0 |
| LM | 156.0 | 155.5 | 150.2 | 158.9 | 158.9 | 0.0 |
| LL | 159.8 | 160.5 | 153.4 | 167.7 | 167.7 | 0.0 |

### GPU power avg (W, GPU-side only)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 123.1 | 112.8 | 62.1 | 127.2 | 127.0 | 0.2 |
| SM | 127.6 | 125.7 | 111.2 | 129.7 | 129.3 | 0.0 |
| SL | 106.9 | 106.4 | 92.2 | 115.9 | 114.2 | 0.1 |
| MS | 134.1 | 127.9 | 87.5 | 138.5 | 137.7 | 0.1 |
| MM | 123.9 | 119.4 | 81.9 | 133.9 | 133.6 | 0.1 |
| ML | 116.4 | 111.2 | 78.7 | 126.4 | 125.5 | 0.1 |
| LS | 149.7 | 141.4 | 102.3 | 152.3 | 152.2 | 0.1 |
| LM | 136.4 | 135.0 | 123.4 | 143.9 | 143.2 | 0.1 |
| LL | 118.9 | 117.3 | 100.8 | 134.4 | 132.9 | 0.1 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

