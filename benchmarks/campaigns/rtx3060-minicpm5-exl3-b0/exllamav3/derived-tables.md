# Derived comparison tables — ExLlamaV3 1.5.3 on RTX 3060

Campaign `PRADIUM-EXL3-RTX3060-B0`, benchmark `PRADIUM-RUNTIME-BENCH-v2`,
model `ewin-reg/MiniCPM5-2B-EXL3-Quantized` @
`e36fbb568f466739d7c244a5100d01acf93d3b7f`, generated from the committed
`raw.jsonl` records (warmups excluded).

Provenance: every value marked **harness** is the benchmark's own reported or
derived metric. Columns marked **DERIVED** are computed here, only because the
harness leaves the corresponding field null; the formula is stated inline.

## 3x3 core matrix (harness, medians of 10 measured runs)

| Workload | TTFT ms | prefill tok/s | decode tok/s | TPOT ms | ITL p99 ms | E2E ms | peak VRAM MB | GPU util % | power W |
|---|---|---|---|---|---|---|---|---|---|
| SS | 69.53 | 2103.92 | 109.80 | 9.11 | 20.69 | 644.73 | 2799.00 | 73.17 | 116.27 |
| SM | 69.01 | 2117.74 | 85.58 | 11.70 | 69.99 | 3056.44 | 2799.00 | 59.87 | 111.14 |
| SL | 69.88 | 2101.14 | 96.45 | 10.37 | 50.16 | 10676.98 | 2799.00 | 71.07 | 123.54 |
| MS | 236.34 | 4516.35 | 109.36 | 9.14 | 21.23 | 811.57 | 2951.00 | 78.38 | 136.37 |
| MM | 237.13 | 4508.99 | 101.31 | 9.87 | 30.96 | 2754.25 | 2951.00 | 76.17 | 120.48 |
| ML | 237.49 | 4508.85 | 87.95 | 11.50 | 58.61 | 12002.67 | 2951.00 | 66.08 | 119.81 |
| LS | 880.25 | 4713.97 | 99.81 | 10.02 | 22.88 | 1511.31 | 3031.00 | 88.17 | 152.10 |
| LM | 878.99 | 4715.08 | 89.29 | 11.33 | 37.02 | 3767.44 | 3031.00 | 78.76 | 135.54 |
| LL | 875.30 | 4737.73 | 77.19 | 12.96 | 69.09 | 14136.55 | 3031.00 | 67.15 | 120.66 |

## Concurrency C1/C2/C4/C8

### Aggregate output tok/s (harness: from concurrent group wall time)

| Profile | C1 | C2 | C4 | C8 | C1→C8 factor **DERIVED** | efficiency vs ideal 8x **DERIVED** |
|---|---|---|---|---|---|---|
| SS | 99.28 | 185.03 | 330.66 | 433.68 | 4.37x | 54.60% |
| SM | 97.13 | 174.59 | 315.01 | 545.13 | 5.61x | 70.16% |
| SL | 90.18 | 153.99 | 274.73 | 439.49 | 4.87x | 60.92% |
| MS | 78.07 | 136.83 | 223.41 | 365.67 | 4.68x | 58.55% |
| MM | 90.56 | 151.45 | 181.24 | 442.78 | 4.89x | 61.11% |
| ML | 77.06 | 131.76 | 232.25 | 452.87 | 5.88x | 73.46% |
| LS | 42.20 | 76.64 | 134.70 | 198.09 | 4.69x | 58.68% |
| LM | 70.00 | 113.58 | 202.18 | 280.40 | 4.01x | 50.07% |
| LL | 63.09 | 116.70 | 236.77 | 310.28 | 4.92x | 61.48% |

### Per-request decode tok/s and C1→C8 retention (**DERIVED** = decode(C8)/decode(C1))

| Profile | C1 | C2 | C4 | C8 | retention |
|---|---|---|---|---|---|
| SS | 109.51 | 103.41 | 91.99 | 73.15 | 66.80% |
| SM | 99.37 | 89.15 | 81.36 | 73.64 | 74.10% |
| SL | 90.65 | 77.34 | 69.06 | 55.67 | 61.41% |
| MS | 108.19 | 90.80 | 74.32 | 61.00 | 56.39% |
| MM | 98.56 | 82.85 | 48.67 | 59.59 | 60.46% |
| ML | 78.39 | 66.88 | 58.93 | 57.75 | 73.68% |
| LS | 99.04 | 80.69 | 63.72 | 39.94 | 40.33% |
| LM | 91.91 | 70.36 | 61.25 | 40.26 | 43.80% |
| LL | 66.64 | 61.39 | 62.41 | 40.31 | 60.49% |

### TTFT / ITL p99 / peak VRAM / GPU utilisation / power by level (harness)

| Profile | C | TTFT ms | TPOT ms | ITL p99 ms | peak VRAM MB | GPU util % | power W |
|---|---|---|---|---|---|---|---|
| SS | 1 | 69.44 | 9.13 | 22.04 | 2799.00 | 71.94 | 123.74 |
| SS | 2 | 72.15 | 9.67 | 23.34 | 2815.00 | 69.19 | 122.84 |
| SS | 4 | 92.77 | 10.87 | 22.83 | 2857.00 | 60.79 | 115.70 |
| SS | 8 | 282.30 | 13.72 | 51.00 | 2946.00 | 44.90 | 65.76 |
| MM | 1 | 237.47 | 10.18 | 35.65 | 3123.00 | 72.53 | 125.57 |
| MM | 2 | 241.07 | 12.09 | 43.19 | 3123.00 | 66.67 | 112.60 |
| MM | 4 | 258.55 | 20.56 | 93.07 | 3123.00 | 51.32 | 97.05 |
| MM | 8 | 316.19 | 16.78 | 38.86 | 3123.00 | 80.28 | 110.78 |
| LL | 1 | 876.97 | 15.04 | 83.47 | 3203.00 | 58.76 | 109.17 |
| LL | 2 | 881.35 | 16.30 | 86.00 | 3203.00 | 64.89 | 115.70 |
| LL | 4 | 889.56 | 16.02 | 40.27 | 3203.00 | 86.78 | 133.15 |
| LL | 8 | 956.87 | 24.81 | 34.99 | 3203.00 | 91.34 | 119.87 |

## Prefix / KV reuse

Speedup **DERIVED** = prefill tok/s for the scenario ÷ prefill tok/s of the
no-reuse `cross_session` control at the same nominal ratio.

| nominal reuse | scenario | TTFT ms | prefill tok/s | speedup | decode tok/s |
|---|---|---|---|---|---|
| 0.0 | cross_request | 237.6 | 4492 | 1.00x | 98.5 |
| 0.0 | cross_session | 237.6 | 4492 | 1.00x | 102.3 |
| 0.0 | same_session | 237.8 | 4486 | 1.00x | 97.8 |
| 0.25 | cross_request | 189.2 | 5698 | 1.27x | 99.2 |
| 0.25 | cross_session | 238.3 | 4475 | 1.00x | 102.3 |
| 0.25 | same_session | 189.3 | 5699 | 1.27x | 104.3 |
| 0.5 | cross_request | 138.3 | 7949 | 1.77x | 101.1 |
| 0.5 | cross_session | 238.0 | 4486 | 1.00x | 102.5 |
| 0.5 | same_session | 138.1 | 7958 | 1.77x | 102.8 |
| 0.75 | cross_request | 93.1 | 12216 | 2.72x | 97.0 |
| 0.75 | cross_session | 238.0 | 4484 | 1.00x | 97.1 |
| 0.75 | same_session | 93.1 | 12228 | 2.73x | 102.2 |
| 0.900390625 | cross_request | 66.8 | 17901 | 4.00x | 105.0 |
| 0.900390625 | cross_session | 238.3 | 4476 | 1.00x | 92.6 |
| 0.900390625 | same_session | 66.7 | 17900 | 4.00x | 101.2 |
| 0.984375 | cross_request | 21.5 | 83683 | 18.65x | 101.5 |
| 0.984375 | cross_session | 237.8 | 4487 | 1.00x | 100.0 |
| 0.984375 | same_session | 21.9 | 82930 | 18.48x | 102.4 |

## Mixed request workloads (harness)

| scenario | TTFT ms | decode tok/s | aggregate tok/s | TPOT ms | ITL p99 ms | E2E ms | peak VRAM MB | GPU util % |
|---|---|---|---|---|---|---|---|---|
| `agent_like` | 259.24 | 76.82 | 235.60 | 13.02 | 25.05 | 1078.47 | 3059.00 | 72.55 |
| `interactive_burst` | 152.59 | 84.84 | 197.03 | 11.79 | 26.85 | 953.61 | 2898.00 | 60.17 |
| `long_job_interference` | 551.70 | 51.76 | 84.52 | 19.35 | 94.81 | 3288.62 | 3139.00 | 81.39 |
| `mixed_generation` | 695.96 | 71.89 | 89.97 | 13.91 | 71.71 | 4187.06 | 3059.00 | 71.33 |
| `mixed_prompt_sizes` | 718.24 | 61.14 | 154.55 | 16.36 | 31.96 | 4946.06 | 3139.00 | 86.30 |

## Batching B1/B2/B4/B8 at MM (harness; group-aggregate accounting, so
per-request TTFT/TPOT are UNAVAILABLE rather than estimated)

| batch | measured requests | aggregate tok/s | E2E ms | scaling vs B1 **DERIVED** |
|---|---|---|---|---|
| B1 | 10 | 75.13 | 3407.61 | 1.00x |
| B2 | 20 | 129.26 | 4043.08 | 1.72x |
| B4 | 40 | 247.10 | 4159.26 | 3.29x |
| B8 | 80 | 424.24 | 4828.88 | 5.65x |

## Startup (harness)

| Milestone | Value |
|---|---|
| idle_vram_mb | 2711.00 |
| idle_ram_mb | 1676.43 |
| runtime_init_ms | 0.09 |
| cuda_init_ms | 0.09 |
| model_load_ms | 1561.78 |
| process_start_to_ready_ms | 1561.88 |
| cold_first_request_ttft_ms | 1574.01 |
| process_start_to_first_token_ms | 3211.63 |
| warm_request_ttft_ms_median | 11.27 |
| warm_request_count | 5 |
| cuda_graph_capture_ms | - |
| weight_mapping_ms | - |

The `idle_*` values were sampled while the harness's CLI already had the model
loaded, so they are not true idle; true idle baselines are in the campaign report
(217-300 MB VRAM, ~15 W).

## Soak / stability (harness)

| Metric | Value |
|---|---|
| duration_requested_s | 1800.0 |
| duration_actual_s | 1805.5 |
| iterations | 170 |
| profiles | ['MM', 'SL', 'LL'] |
| failed_requests_total | 0 |
| vram_growth_mb | 3.86 |
| ram_growth_mb | 36.54 |
| ttft_growth_ratio | -0.0793 |
| decode_throughput_change_ratio | 0.0374 |

## Supplementary `CTX-32768` at a 49152-token KV pool (harness)

| Metric | Value |
|---|---|
| ttft_ms | 13445.14 |
| prefill_ms | 13417.58 |
| prefill_tok_s | 2442.17 |
| decode_tok_s | 58.75 |
| tpot_ms | 17.02 |
| itl_p95_ms | 17.65 |
| e2e_ms | 17775.69 |
| vram_peak_mb | 3659.00 |
| gpu_util_avg | 97.81 |
| gpu_power_avg | 165.58 |
| measured_runs | 10 |
| max_num_tokens | 49152 |

