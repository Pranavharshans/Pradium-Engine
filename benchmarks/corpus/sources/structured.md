# Structured Information Corpus

Tables, lists, key/value data, log lines, and configuration fragments. Agent
and retrieval workloads process large amounts of exactly this kind of
structured content.

---

## Hardware inventory table

| Host    | GPU            | VRAM  | CPU                | Cores | RAM   | Region    | Role        |
|---------|----------------|-------|--------------------|-------|-------|-----------|-------------|
| node-01 | RTX 4090       | 24 GB | Ryzen 9 7950X      | 16    | 128GB | us-east   | inference   |
| node-02 | RTX 4090       | 24 GB | Ryzen 9 7950X      | 16    | 128GB | us-east   | inference   |
| node-03 | RTX 3090       | 24 GB | Core i9-13900K     | 24    | 96 GB | us-east   | inference   |
| node-04 | RTX 6000 Ada   | 48GB  | Threadripper 7970X | 32    | 256GB | us-west   | evaluation  |
| node-05 | A100 80GB      | 80 GB | EPYC 9354P         | 32    | 512GB | us-west   | training    |
| node-06 | H100 SXM       | 80 GB | EPYC 9654          | 96    | 1 TB  | us-west   | training    |
| node-07 | RTX 4070       | 12 GB | Core i7-13700      | 16    | 64 GB | eu-west   | staging     |
| node-08 | RTX 4060 Ti    | 16 GB | Ryzen 7 7700X      | 8     | 64 GB | eu-west   | staging     |

## Release schedule table

| Quarter | Milestone                | Owner    | Status      | Notes                                  |
|---------|--------------------------|----------|-------------|----------------------------------------|
| 2026Q1  | Token cache v2           | Inference| shipped     | prefix reuse across requests           |
| 2026Q2  | Scheduler rewrite        | Runtime  | shipped     | continuous batching, chunked prefill   |
| 2026Q3  | Quantization pipeline    | Model    | in progress | 4-bit and 6-bit paths                  |
| 2026Q4  | Multi-GPU serving        | Runtime  | planned     | tensor parallel, 2-4 devices           |
| 2027Q1  | Speculative decoding     | Research | planned     | draft model verification               |

## Key/value configuration dump

```ini
[service]
name = bench-worker
instance = 7
region = us-east-1b
replicas = 4
max_concurrency = 8

[model]
path = /models/primary
revision = 9f3ac21
quantization = q4_k_m
context_length = 32768
rope_scaling = linear
rope_factor = 2.0

[runtime]
backend = eager
device = cuda
device_index = 0
kv_cache_bytes = 6442450944
kv_cache_dtype = fp16
graph_capture = enabled
warmup_iterations = 3

[limits]
prompt_tokens = 32768
new_tokens = 4096
requests_per_minute = 600
queue_depth = 256
timeout_ms = 120000
```

## Log excerpt

```text
2026-03-14T09:12:01.221Z INFO  scheduler: admitted request req-8f21c0 prompt=1024 new=256 queue_depth=3
2026-03-14T09:12:01.224Z DEBUG kv_cache: prefix hit tokens=768 lookup_ms=0.42 insert_ms=0.11
2026-03-14T09:12:01.318Z INFO  prefill: completed tokens=256 elapsed_ms=93.7 tok_s=2732.1
2026-03-14T09:12:02.884Z INFO  decode: request=req-8f21c0 tokens=64 elapsed_ms=1552.3 tok_s=41.2
2026-03-14T09:12:02.901Z WARN  telemetry: nvml sampling missed 2 intervals
2026-03-14T09:12:03.004Z INFO  scheduler: admitted request req-4b7e12 prompt=4096 new=64 queue_depth=1
2026-03-14T09:12:03.771Z ERROR kv_cache: eviction triggered used=6442450944 budget=6442450944 evicted_blocks=128
2026-03-14T09:12:03.912Z INFO  decode: request=req-4b7e12 tokens=64 elapsed_ms=812.9 tok_s=78.7
2026-03-14T09:12:04.100Z INFO  metrics: agg_output_tok_s=118.4 gpu_util=0.93 power_w=214.7 vram_mb=21340
2026-03-14T09:12:04.318Z DEBUG scheduler: batch formed size=4 formation_ms=0.83 policy=continuous
```

## Capability matrix

| Runtime      | Streaming | Input IDs | Fixed decode | Prefix cache | Continuous batching | Chunked prefill | CUDA graphs |
|--------------|-----------|-----------|--------------|--------------|---------------------|-----------------|-------------|
| reference-a  | yes       | yes       | yes          | partial      | yes                 | yes             | yes         |
| reference-b  | yes       | no        | yes          | no           | yes                 | no              | no          |
| reference-c  | partial   | yes       | no           | yes          | no                  | no              | yes         |
| reference-d  | yes       | yes       | yes          | yes          | yes                 | yes             | partial     |

## Event counts by hour

| Hour | Requests | Prompt tokens | Output tokens | OOM | Timeouts | p50 TTFT ms | p99 TTFT ms |
|------|----------|---------------|---------------|-----|----------|-------------|-------------|
| 08   | 12,411   | 8,102,338     | 2,214,120     | 0   | 1        | 82          | 241         |
| 09   | 21,004   | 14,882,771    | 3,910,233     | 2   | 4        | 96          | 388         |
| 10   | 24,550   | 17,221,412    | 4,412,880     | 5   | 2        | 118         | 441         |
| 11   | 22,118   | 15,993,210    | 4,021,774     | 1   | 0        | 109         | 362         |
| 12   | 13,877   | 9,004,221     | 2,550,118     | 0   | 1        | 88          | 251         |
| 13   | 19,236   | 13,512,884    | 3,612,900     | 3   | 2        | 101         | 402         |
| 14   | 20,774   | 14,662,115    | 3,801,442     | 2   | 3        | 104         | 377         |

## Endpoint list

- `POST /v1/generate` — text generation; body: model, prompt, max_tokens, temperature
- `POST /v1/generate_stream` — SSE stream of token events; same body
- `GET  /v1/models` — list loaded models with revisions and context lengths
- `GET  /v1/metrics` — Prometheus exposition format
- `GET  /healthz` — liveness probe; returns 200 when the scheduler loop is alive
- `GET  /readyz` — readiness probe; returns 200 when the model is loaded and warm
- `POST /admin/cache/reset` — clears the KV prefix cache; returns bytes reclaimed
- `GET  /admin/queue` — current queue depth, oldest waiting age, admission state

## Error codes

| Code | Name               | Meaning                                          | Retry |
|------|--------------------|--------------------------------------------------|-------|
| 4000 | BAD_REQUEST        | malformed body or missing required field         | no    |
| 4001 | PROMPT_TOO_LONG    | prompt exceeds model context length              | no    |
| 4002 | OUTPUT_TOO_LONG    | max_new_tokens exceeds per-request limit         | no    |
| 4290 | RATE_LIMITED       | requests-per-minute exceeded                     | yes   |
| 4291 | QUEUE_FULL         | admission queue at capacity                      | yes   |
| 5000 | INTERNAL           | unhandled runtime error                          | yes   |
| 5001 | OOM                | model or KV cache allocation failed              | no    |
| 5002 | TIMEOUT            | generation exceeded request timeout              | yes   |
| 5003 | MODEL_UNAVAILABLE  | requested model revision not loaded              | no    |

## Sparse key/value store dump

```text
session:8841:role            = assistant
session:8841:model           = primary
session:8841:created         = 1773470400
session:8841:tokens_used     = 18224
session:8841:last_activity   = 1773474122
session:8842:role            = assistant
session:8842:model           = primary
session:8842:created         = 1773470511
session:8842:tokens_used     = 2211
session:8842:last_activity   = 1773470900
cache:prefix:7c21f9:tokens   = 1024
cache:prefix:7c21f9:bytes    = 1048576
cache:prefix:7c21f9:hits     = 42
cache:prefix:7c21f9:created  = 1773469000
```
