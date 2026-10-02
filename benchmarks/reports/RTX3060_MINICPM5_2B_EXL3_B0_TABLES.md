# PRADIUM EXL3 B0 — full benchmark tables

Campaign `PRADIUM-EXL3-RTX3060-B0` · benchmark `PRADIUM-RUNTIME-BENCH-v2` ·
model `ewin-reg/MiniCPM5-2B-EXL3-Quantized` @ `e36fbb568f466739d7c244a5100d01acf93d3b7f`
(EXL3 4.0 bpw) · NVIDIA RTX 3060 12 GB, sm_86, driver 550.144.03 · Pradium `cf4e875`.

Every table below is generated directly from the committed `raw.jsonl` records
(4034 records, 11 sessions); warmups are excluded and each figure is the median
of the measured runs in its group. Official configuration: 3 warmups + 10
measured runs, `max_num_tokens=32768`, `max_batch_size=16`, fp16 KV cache,
`ArgmaxSampler` (greedy).

Only ExLlamaV3 1.5.3 could genuinely execute this checkpoint on this VM; SGLang
0.5.20 + `sglang_exl3` and vLLM 0.30.0 + `vllm-exl3` are `INSTALL_FAILURE`
(their pinned `torch==2.13.0` is CUDA 13 only, the driver caps at CUDA 12.4), and
TensorFold 0.6.1 is `UNSUPPORTED_MODEL` (no MiniCPM family, and it refuses sm_86).
See `compatibility.md` and the campaign report for the full reasoning.

### 1. 3x3 core matrix — all nine workloads (median of 10 measured runs)

| WL | in→out | TTFT ms | pref ms | pref tok/s | dec ms | dec tok/s | TPOT ms | ITL med | ITL p95 | ITL p99 | ITL max | E2E ms | VRAM MB | CPU % | GPU % | W | °C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SS | 128→64 | 69.53 | 60.84 | 2104 | 574 | 109.80 | 9.11 | 7.89 | 18.11 | 20.69 | 21.46 | 644.73 | 2799 | 172 | 73.17 | 116.27 | 50 |
| SM | 128→256 | 69.01 | 60.44 | 2118 | 2983 | 85.58 | 11.70 | 7.94 | 26.33 | 69.99 | 81.46 | 3056.44 | 2799 | 162 | 59.87 | 111.14 | 57 |
| SL | 128→1024 | 69.88 | 60.92 | 2101 | 10606 | 96.45 | 10.37 | 7.98 | 21.46 | 50.16 | 77.98 | 10676.98 | 2799 | 170 | 71.07 | 123.54 | 66 |
| MS | 1024→64 | 236.34 | 226.73 | 4516 | 576 | 109.36 | 9.14 | 8.00 | 18.80 | 21.23 | 21.48 | 811.57 | 2951 | 157 | 78.38 | 136.37 | 65 |
| MM | 1024→256 | 237.13 | 227.10 | 4509 | 2517 | 101.31 | 9.87 | 8.01 | 20.52 | 30.96 | 47.77 | 2754.25 | 2951 | 167 | 76.17 | 120.48 | 66 |
| ML | 1024→1024 | 237.49 | 227.11 | 4509 | 11765 | 87.95 | 11.50 | 8.11 | 27.99 | 58.61 | 87.23 | 12002.67 | 2951 | 166 | 66.08 | 119.81 | 68 |
| LS | 4096→64 | 880.25 | 868.91 | 4714 | 631 | 99.81 | 10.02 | 8.88 | 19.11 | 22.88 | 23.87 | 1511.31 | 3031 | 136 | 88.17 | 152.10 | 69 |
| LM | 4096→256 | 878.99 | 868.70 | 4715 | 2888 | 89.29 | 11.33 | 8.91 | 22.64 | 37.02 | 41.14 | 3767.44 | 3031 | 155 | 78.76 | 135.54 | 70 |
| LL | 4096→1024 | 875.30 | 864.55 | 4738 | 13262 | 77.19 | 12.96 | 8.98 | 29.01 | 69.09 | 98.99 | 14136.55 | 3031 | 161 | 67.15 | 120.66 | 68 |

### 2. Context scaling — 128 → 32768 input, 256 output

| input | TTFT ms | pref ms | pref tok/s | dec tok/s | TPOT ms | ITL p95 | E2E ms | VRAM MB | GPU % | W | °C | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 68.81 | 60.26 | 2124 | 101.00 | 9.91 | 20.30 | 2594.83 | 2799 | 73.03 | 118.78 | 57 | 10 |
| 512 | 132.23 | 123.05 | 4161 | 103.53 | 9.66 | 19.97 | 2596.21 | 2911 | 75.69 | 125.58 | 64 | 10 |
| 1024 | 237.53 | 227.88 | 4494 | 103.04 | 9.71 | 20.01 | 2712.09 | 2951 | 76.12 | 130.09 | 67 | 10 |
| 2048 | 414.28 | 403.96 | 5070 | 100.99 | 9.90 | 19.79 | 2939.40 | 3031 | 77.94 | 136.81 | 69 | 10 |
| 4096 | 877.54 | 866.80 | 4725 | 88.12 | 11.36 | 21.13 | 3779.47 | 3031 | 75.42 | 127.63 | 69 | 10 |
| 8192 | 2021.67 | 2009.60 | 4076 | 87.70 | 11.40 | 20.13 | 4929.65 | 3031 | 88.11 | 149.68 | 70 | 10 |
| 16384 | 4930.08 | 4915.39 | 3333 | 71.69 | 13.95 | 22.47 | 8485.95 | 3031 | 92.15 | 157.64 | 73 | 10 |
| 32768 † | FAILED | — | — | — | — | — | — | — | — | — | — | 0/13 |
| 32768 ‡ | 13445.14 | 13417.58 | 2442 | 58.75 | 17.02 | 17.65 | 17775.69 | 3659 | 97.81 | 165.58 | 74 | 10 |

### 3. Concurrency C1/C2/C4/C8 — all nine profiles

| WL | metric | C1 | C2 | C4 | C8 |
|---|---|---|---|---|---|
| SS | aggregate tok/s | 99.28 | 185.03 | 330.66 | 433.68 |
| SS | per-req decode tok/s | 109.51 | 103.41 | 91.99 | 73.15 |
| SS | TTFT ms | 69.44 | 72.15 | 92.77 | 282.30 |
| SS | TPOT ms | 9.13 | 9.67 | 10.87 | 13.72 |
| SS | ITL p99 ms | 22.04 | 23.34 | 22.83 | 51.00 |
| SS | E2E ms | 644.69 | 683.44 | 763.63 | 1142.34 |
| SS | peak VRAM MB | 2799 | 2815 | 2857 | 2946 |
| SS | GPU % | 71.94 | 69.19 | 60.79 | 44.90 |
| SS | power W | 123.74 | 122.84 | 115.70 | 65.76 |
| **SS** | **C1→C8** | | | | **4.37× (54.6% of ideal), per-req 66.8%** |
| SM | aggregate tok/s | 97.13 | 174.59 | 315.01 | 545.13 |
| SM | per-req decode tok/s | 99.37 | 89.15 | 81.36 | 73.64 |
| SM | TTFT ms | 69.90 | 72.00 | 90.28 | 279.34 |
| SM | TPOT ms | 10.07 | 11.23 | 12.29 | 13.58 |
| SM | ITL p99 ms | 38.06 | 45.14 | 45.25 | 52.95 |
| SM | E2E ms | 2637.59 | 2935.32 | 3235.82 | 3722.91 |
| SM | peak VRAM MB | 2961 | 2961 | 2961 | 2961 |
| SM | GPU % | 72.06 | 69.92 | 66.97 | 62.06 |
| SM | power W | 108.26 | 120.04 | 116.73 | 105.51 |
| **SM** | **C1→C8** | | | | **5.61× (70.2% of ideal), per-req 74.1%** |
| SL | aggregate tok/s | 90.18 | 153.99 | 274.73 | 439.49 |
| SL | per-req decode tok/s | 90.65 | 77.34 | 69.06 | 55.67 |
| SL | TTFT ms | 70.03 | 71.29 | 87.61 | 285.67 |
| SL | TPOT ms | 11.03 | 12.93 | 14.48 | 17.96 |
| SL | ITL p99 ms | 60.15 | 74.56 | 76.96 | 96.55 |
| SL | E2E ms | 11357.58 | 13297.79 | 14894.69 | 18613.45 |
| SL | peak VRAM MB | 2971 | 2971 | 2971 | 2971 |
| SL | GPU % | 67.22 | 63.90 | 68.23 | 61.89 |
| SL | power W | 119.12 | 113.80 | 111.33 | 105.02 |
| **SL** | **C1→C8** | | | | **4.87× (60.9% of ideal), per-req 61.4%** |
| MS | aggregate tok/s | 78.07 | 136.83 | 223.41 | 365.67 |
| MS | per-req decode tok/s | 108.19 | 90.80 | 74.32 | 61.00 |
| MS | TTFT ms | 237.07 | 240.09 | 259.99 | 353.56 |
| MS | TPOT ms | 9.24 | 11.01 | 13.46 | 16.39 |
| MS | ITL p99 ms | 21.05 | 24.24 | 31.32 | 32.24 |
| MS | E2E ms | 819.75 | 933.80 | 1131.80 | 1379.34 |
| MS | peak VRAM MB | 3123 | 3123 | 3123 | 3123 |
| MS | GPU % | 77.75 | 74.72 | 68.39 | 72.77 |
| MS | power W | 136.42 | 130.82 | 118.96 | 97.06 |
| **MS** | **C1→C8** | | | | **4.68× (58.5% of ideal), per-req 56.4%** |
| MM | aggregate tok/s | 90.56 | 151.45 | 181.24 | 442.78 |
| MM | per-req decode tok/s | 98.56 | 82.85 | 48.67 | 59.59 |
| MM | TTFT ms | 237.47 | 241.07 | 258.55 | 316.19 |
| MM | TPOT ms | 10.18 | 12.09 | 20.56 | 16.78 |
| MM | ITL p99 ms | 35.65 | 43.19 | 93.07 | 38.86 |
| MM | E2E ms | 2835.15 | 3380.73 | 5634.64 | 4601.13 |
| MM | peak VRAM MB | 3123 | 3123 | 3123 | 3123 |
| MM | GPU % | 72.53 | 66.67 | 51.32 | 80.28 |
| MM | power W | 125.57 | 112.60 | 97.05 | 110.78 |
| **MM** | **C1→C8** | | | | **4.89× (61.1% of ideal), per-req 60.5%** |
| ML | aggregate tok/s | 77.06 | 131.76 | 232.25 | 452.87 |
| ML | per-req decode tok/s | 78.39 | 66.88 | 58.93 | 57.75 |
| ML | TTFT ms | 237.31 | 240.28 | 258.74 | 339.25 |
| ML | TPOT ms | 12.78 | 14.97 | 17.03 | 17.31 |
| ML | ITL p99 ms | 79.44 | 74.81 | 92.21 | 36.23 |
| ML | E2E ms | 13315.00 | 15549.33 | 17675.04 | 18052.67 |
| ML | peak VRAM MB | 3123 | 3123 | 3123 | 3123 |
| ML | GPU % | 60.14 | 56.55 | 64.14 | 85.85 |
| ML | power W | 112.24 | 106.28 | 110.26 | 117.37 |
| **ML** | **C1→C8** | | | | **5.88× (73.5% of ideal), per-req 73.7%** |
| LS | aggregate tok/s | 42.20 | 76.64 | 134.70 | 198.09 |
| LS | per-req decode tok/s | 99.04 | 80.69 | 63.72 | 39.94 |
| LS | TTFT ms | 879.69 | 886.25 | 891.81 | 953.75 |
| LS | TPOT ms | 10.10 | 12.39 | 15.69 | 25.04 |
| LS | ITL p99 ms | 21.11 | 24.83 | 26.72 | 35.54 |
| LS | E2E ms | 1516.73 | 1666.58 | 1890.86 | 2536.81 |
| LS | peak VRAM MB | 3203 | 3203 | 3203 | 3203 |
| LS | GPU % | 87.20 | 86.78 | 84.75 | 85.94 |
| LS | power W | 151.42 | 147.04 | 141.31 | 120.50 |
| **LS** | **C1→C8** | | | | **4.69× (58.7% of ideal), per-req 40.3%** |
| LM | aggregate tok/s | 70.00 | 113.58 | 202.18 | 280.40 |
| LM | per-req decode tok/s | 91.91 | 70.36 | 61.25 | 40.26 |
| LM | TTFT ms | 878.92 | 880.94 | 888.63 | 952.12 |
| LM | TPOT ms | 10.88 | 14.22 | 16.33 | 24.84 |
| LM | ITL p99 ms | 34.34 | 53.36 | 46.68 | 34.19 |
| LM | E2E ms | 3657.48 | 4505.22 | 5055.33 | 7282.29 |
| LM | peak VRAM MB | 3203 | 3203 | 3203 | 3203 |
| LM | GPU % | 82.69 | 74.83 | 84.63 | 89.74 |
| LM | power W | 134.97 | 129.32 | 130.54 | 119.33 |
| **LM** | **C1→C8** | | | | **4.01× (50.1% of ideal), per-req 43.8%** |
| LL | aggregate tok/s | 63.09 | 116.70 | 236.77 | 310.28 |
| LL | per-req decode tok/s | 66.64 | 61.39 | 62.41 | 40.31 |
| LL | TTFT ms | 876.97 | 881.35 | 889.56 | 956.87 |
| LL | TPOT ms | 15.04 | 16.30 | 16.02 | 24.81 |
| LL | ITL p99 ms | 83.47 | 86.00 | 40.27 | 34.99 |
| LL | E2E ms | 16259.41 | 17557.53 | 17285.62 | 26357.43 |
| LL | peak VRAM MB | 3203 | 3203 | 3203 | 3203 |
| LL | GPU % | 58.76 | 64.89 | 86.78 | 91.34 |
| LL | power W | 109.17 | 115.70 | 133.15 | 119.87 |
| **LL** | **C1→C8** | | | | **4.92× (61.5% of ideal), per-req 60.5%** |

### 4. Prefix / KV reuse — 1024-token inputs, 256 output

| reuse | scenario | TTFT ms | pref ms | pref tok/s | speedup | dec tok/s | E2E ms |
|---|---|---|---|---|---|---|---|
| 0.0 | cross_request | 237.64 | 227.94 | 4492 | 1.00× | 98.48 | 2827.16 |
| 0.0 | cross_session | 237.59 | 227.97 | 4492 | 1.00× | 102.26 | 2735.00 |
| 0.0 | same_session | 237.76 | 228.29 | 4486 | 1.00× | 97.82 | 2845.28 |
| 0.25 | cross_request | 189.21 | 179.70 | 5698 | 1.27× | 99.23 | 2758.58 |
| 0.25 | cross_session | 238.35 | 228.83 | 4475 | 1.00× | 102.32 | 2730.20 |
| 0.25 | same_session | 189.31 | 179.68 | 5699 | 1.27× | 104.34 | 2633.29 |
| 0.5 | cross_request | 138.34 | 128.81 | 7949 | 1.77× | 101.13 | 2660.22 |
| 0.5 | cross_session | 238.01 | 228.27 | 4486 | 1.00× | 102.51 | 2725.92 |
| 0.5 | same_session | 138.06 | 128.68 | 7958 | 1.77× | 102.82 | 2617.28 |
| 0.75 | cross_request | 93.14 | 83.83 | 12216 | 2.72× | 97.03 | 2721.48 |
| 0.75 | cross_session | 238.02 | 228.39 | 4484 | 1.00× | 97.14 | 2863.49 |
| 0.75 | same_session | 93.10 | 83.74 | 12228 | 2.73× | 102.22 | 2587.88 |
| 0.900390625 | cross_request | 66.82 | 57.20 | 17901 | 4.00× | 104.98 | 2504.04 |
| 0.900390625 | cross_session | 238.32 | 228.78 | 4476 | 1.00× | 92.59 | 2992.32 |
| 0.900390625 | same_session | 66.66 | 57.21 | 17900 | 4.00× | 101.17 | 2587.25 |
| 0.984375 | cross_request | 21.55 | 12.24 | 83683 | 18.65× | 101.53 | 2533.83 |
| 0.984375 | cross_session | 237.78 | 228.23 | 4487 | 1.00× | 100.02 | 2787.78 |
| 0.984375 | same_session | 21.93 | 12.35 | 82930 | 18.48× | 102.44 | 2517.07 |

### 5. Scheduler interference — 3 resident decoders + one 4096-token prefill

| metric | value |
|---|---|
| background_requests | 3 |
| background_aggregate_decode_tok_s | 196.92 |
| intruder_profile | LS |
| intruder_input_tokens | 4096 |
| intruder_prefill_ms | 898.10 |
| intruder_ttft_ms | 2236.04 |
| itl_before_ms | None |
| itl_during_ms | 329.95 |
| itl_after_ms | 14.62 |
| max_output_token_stall_ms | 499.69 |
| p95_itl_ms | 28.41 |

### 6. Mixed request workloads

| scenario | TTFT ms | dec tok/s | aggregate tok/s | TPOT ms | ITL p99 | E2E ms | VRAM MB | GPU % |
|---|---|---|---|---|---|---|---|---|
| `agent_like` | 259.24 | 76.82 | 235.60 | 13.02 | 25.05 | 1078.47 | 3059 | 72.55 |
| `interactive_burst` | 152.59 | 84.84 | 197.03 | 11.79 | 26.85 | 953.61 | 2898 | 60.17 |
| `long_job_interference` | 551.70 | 51.76 | 84.52 | 19.35 | 94.81 | 3288.62 | 3139 | 81.39 |
| `mixed_generation` | 695.96 | 71.89 | 89.97 | 13.91 | 71.71 | 4187.06 | 3059 | 71.33 |
| `mixed_prompt_sizes` | 718.24 | 61.14 | 154.55 | 16.36 | 31.96 | 4946.06 | 3139 | 86.30 |

### 7. Batching B1/B2/B4/B8 at MM

| batch | requests | aggregate tok/s | E2E ms | scaling vs B1 |
|---|---|---|---|---|
| B1 | 10 | 75.13 | 3407.61 | 1.00× |
| B2 | 20 | 129.26 | 4043.08 | 1.72× |
| B4 | 40 | 247.10 | 4159.26 | 3.29× |
| B8 | 80 | 424.24 | 4828.88 | 5.65× |

### 8. Startup

| milestone | value |
|---|---|
| runtime_init_ms | 0.0924 |
| cuda_init_ms | 0.0859 |
| model_load_ms | 1561.7776 |
| process_start_to_ready_ms | 1561.8768 |
| cold_first_request_ttft_ms | 1574.0058 |
| process_start_to_first_token_ms | 3211.6334 |
| warm_request_ttft_ms_median | 11.2726 |
| idle_vram_mb | 2711.0000 |
| idle_ram_mb | 1676.4300 |
| cuda_graph_capture_ms | None |
| weight_mapping_ms | None |

### 9. Soak / stability — 30 minutes

| metric | value |
|---|---|
| duration actual s | 1805.5 |
| iterations | 170 |
| new failures | 0 |
| VRAM growth MB | +3.86 |
| RAM growth MB | +36.54 |
| TTFT change | -7.9% |
| decode throughput change | +3.7% |
