# Pradium Runtime Benchmark Report — PRADIUM-RUNTIME-BENCH-v2

This report contains **measured facts** and **derived metrics** only.
Interpretation is left to the reader; no winner is declared.

## Session

- Session: `20261002-192457_exllamav3_official_826a38`
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
| SS | 69.37 | 72.17 | 68.70 | 92.95 | 84.53 | 0.10 |
| SM | 69.42 | 69.54 | 68.99 | 71.05 | 70.66 | 0.01 |
| SL | 69.30 | 70.52 | 68.89 | 81.00 | 76.26 | 0.05 |
| MS | 240.28 | 246.35 | 239.72 | 295.63 | 272.29 | 0.07 |
| MM | 240.05 | 240.40 | 239.25 | 242.45 | 242.15 | 0.00 |
| ML | 239.44 | 239.77 | 238.91 | 241.68 | 241.53 | 0.00 |
| LS | 899.58 | 899.57 | 896.29 | 904.23 | 902.84 | 0.00 |
| LM | 893.75 | 894.65 | 891.05 | 901.63 | 900.28 | 0.00 |
| LL | 898.53 | 898.63 | 896.37 | 902.94 | 901.68 | 0.00 |

### Prefill (tok/s, runtime-reported)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 2099.0 | 2086.1 | 1951.7 | 2117.3 | 2117.0 | 0.0 |
| SM | 2101.2 | 2098.1 | 2089.4 | 2102.3 | 2102.2 | 0.0 |
| SL | 2100.0 | 2097.9 | 2070.7 | 2115.2 | 2114.8 | 0.0 |
| MS | 4505.3 | 4418.9 | 3700.4 | 4513.1 | 4512.0 | 0.1 |
| MM | 4508.3 | 4507.1 | 4472.4 | 4522.1 | 4521.6 | 0.0 |
| ML | 4522.4 | 4518.7 | 4499.9 | 4526.5 | 4526.3 | 0.0 |
| LS | 4700.5 | 4699.5 | 4678.3 | 4713.6 | 4713.6 | 0.0 |
| LM | 4727.8 | 4725.3 | 4694.2 | 4744.6 | 4744.0 | 0.0 |
| LL | 4705.5 | 4704.2 | 4686.5 | 4713.6 | 4713.4 | 0.0 |

### Sustained decode (tok/s per request)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 113.75 | 104.72 | 31.00 | 120.31 | 120.23 | 0.26 |
| SM | 99.12 | 94.30 | 48.32 | 108.38 | 108.29 | 0.20 |
| SL | 76.14 | 71.70 | 53.28 | 86.74 | 85.12 | 0.17 |
| MS | 79.90 | 71.74 | 34.81 | 80.85 | 80.76 | 0.24 |
| MM | 72.54 | 66.79 | 48.26 | 76.29 | 76.05 | 0.16 |
| ML | 59.20 | 57.89 | 46.33 | 65.10 | 65.02 | 0.12 |
| LS | 99.42 | 96.72 | 83.17 | 101.78 | 101.62 | 0.07 |
| LM | 61.24 | 63.00 | 22.01 | 98.65 | 98.29 | 0.44 |
| LL | 77.33 | 74.04 | 45.63 | 87.60 | 86.12 | 0.17 |

### TPOT (ms/token)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 8.791 | 11.231 | 8.312 | 32.263 | 22.358 | 0.660 |
| SM | 10.089 | 11.218 | 9.227 | 20.697 | 17.217 | 0.313 |
| SL | 13.149 | 14.361 | 11.529 | 18.768 | 18.373 | 0.186 |
| MS | 12.515 | 15.133 | 12.368 | 28.727 | 25.771 | 0.374 |
| MM | 13.796 | 15.349 | 13.107 | 20.721 | 19.302 | 0.175 |
| ML | 16.942 | 17.508 | 15.360 | 21.583 | 21.109 | 0.126 |
| LS | 10.059 | 10.383 | 9.825 | 12.023 | 11.700 | 0.072 |
| LM | 16.332 | 19.873 | 10.137 | 45.429 | 38.454 | 0.564 |
| LL | 12.945 | 13.969 | 11.415 | 21.915 | 19.337 | 0.223 |

### End-to-end latency (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 622.83 | 779.73 | 593.11 | 2103.15 | 1478.34 | 0.60 |
| SM | 2642.69 | 2930.01 | 2421.88 | 5346.81 | 4460.02 | 0.31 |
| SL | 13521.44 | 14762.04 | 11863.40 | 19269.02 | 18865.03 | 0.19 |
| MS | 1030.05 | 1199.71 | 1019.66 | 2049.54 | 1863.67 | 0.29 |
| MM | 3758.11 | 4154.28 | 3584.14 | 5523.30 | 5161.44 | 0.16 |
| ML | 17570.46 | 18150.08 | 15954.88 | 22318.64 | 21833.92 | 0.12 |
| LS | 1534.46 | 1553.72 | 1519.77 | 1653.73 | 1633.60 | 0.03 |
| LM | 5059.48 | 5962.32 | 3477.92 | 12485.96 | 10706.14 | 0.48 |
| LL | 14141.59 | 15188.59 | 12573.90 | 23318.30 | 20681.98 | 0.21 |

### ITL p95 (ms)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 17.697 | 24.370 | 13.002 | 89.569 | 58.552 | 0.946 |
| SM | 20.404 | 28.774 | 19.093 | 84.120 | 64.608 | 0.712 |
| SL | 24.784 | 39.350 | 21.370 | 70.683 | 69.638 | 0.555 |
| MS | 19.581 | 34.919 | 18.446 | 102.049 | 98.228 | 0.951 |
| MM | 23.485 | 36.642 | 19.390 | 72.746 | 69.660 | 0.570 |
| ML | 36.187 | 41.500 | 22.279 | 79.537 | 72.656 | 0.513 |
| LS | 18.765 | 20.688 | 16.997 | 34.365 | 28.999 | 0.244 |
| LM | 65.114 | 58.176 | 18.950 | 96.995 | 94.035 | 0.518 |
| LL | 32.189 | 38.543 | 20.447 | 86.859 | 75.956 | 0.560 |

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
| SS | 1760.4 | 1760.4 | 1760.4 | 1760.4 | 1760.4 | 0.0 |
| SM | 1764.1 | 1764.0 | 1763.4 | 1764.1 | 1764.1 | 0.0 |
| SL | 1766.3 | 1766.8 | 1765.3 | 1769.3 | 1769.1 | 0.0 |
| MS | 2019.5 | 2016.3 | 2007.7 | 2020.7 | 2020.5 | 0.0 |
| MM | 2022.0 | 2022.0 | 2022.0 | 2022.0 | 2022.0 | 0.0 |
| ML | 2023.9 | 2024.2 | 2023.4 | 2025.6 | 2025.4 | 0.0 |
| LS | 2046.2 | 2046.0 | 2045.0 | 2046.5 | 2046.5 | 0.0 |
| LM | 2047.9 | 2047.7 | 2046.8 | 2047.9 | 2047.9 | 0.0 |
| LL | 2050.2 | 2050.3 | 2049.2 | 2051.7 | 2051.6 | 0.0 |

### CPU utilization avg (%)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 172.4 | 167.1 | 134.1 | 175.1 | 174.9 | 0.1 |
| SM | 170.7 | 168.5 | 146.0 | 175.8 | 175.7 | 0.1 |
| SL | 166.6 | 164.0 | 153.7 | 172.2 | 171.1 | 0.0 |
| MS | 161.1 | 158.0 | 142.0 | 165.5 | 165.5 | 0.0 |
| MM | 165.5 | 164.1 | 152.0 | 173.0 | 172.5 | 0.0 |
| ML | 165.6 | 164.5 | 156.0 | 169.8 | 169.7 | 0.0 |
| LS | 135.4 | 134.4 | 129.6 | 138.5 | 137.5 | 0.0 |
| LM | 145.5 | 146.8 | 132.4 | 159.3 | 158.9 | 0.1 |
| LL | 161.5 | 159.8 | 145.8 | 164.9 | 164.5 | 0.0 |

### GPU power avg (W, GPU-side only)

| Profile | median | mean | min | max | p95 | cv |
|---|---|---|---|---|---|---|
| SS | 125.1 | 116.6 | 76.4 | 128.9 | 128.6 | 0.2 |
| SM | 122.9 | 119.1 | 90.7 | 129.3 | 129.0 | 0.1 |
| SL | 107.9 | 104.1 | 87.9 | 116.5 | 115.7 | 0.1 |
| MS | 120.9 | 115.8 | 89.7 | 122.1 | 122.0 | 0.1 |
| MM | 110.0 | 105.6 | 90.1 | 114.3 | 113.7 | 0.1 |
| ML | 98.1 | 96.7 | 86.1 | 103.9 | 103.5 | 0.1 |
| LS | 150.2 | 149.9 | 145.5 | 152.8 | 152.7 | 0.0 |
| LM | 110.6 | 110.6 | 68.0 | 141.7 | 139.0 | 0.2 |
| LL | 122.6 | 118.2 | 88.9 | 132.0 | 130.2 | 0.1 |

## Failures and unsupported measurements

_No failures recorded._

---

Metric formulas: TTFT = first_token_ts − submit_ts; decode window = last_token_ts − first_token_ts; decode_tok_s = (N−1)/decode_window_s; TPOT = decode_window_ms/(N−1); aggregate_output_tok_s = total_output_tokens/wall_clock_s; ITL_k = ts_k − ts_{k−1}.

