# Mixed Document Corpus

A realistic engineering document mixing prose, code, and structured data —
the shape of material that agent and coding systems actually process.

---

## Prefill and decode are different problems

Serving a large language model means running two workloads with opposite
characteristics inside one process. Prefill is compute-bound: the entire prompt
is processed in parallel, arithmetic intensity is high, and the metric that
matters is prompt tokens per second. Decode is bandwidth-bound: each step moves
the full key/value cache to produce one token, arithmetic intensity is low, and
the metric that matters is time per output token. Optimizing one often
regresses the other, which is why a single "tokens per second" number is
uninterpretable without knowing which phase it describes.

The practical consequence: measure prefill and decode separately, and never
infer one from the other. Time to first token includes queueing, scheduling,
tokenization, prefill, sampling, and delivery. It is an excellent user-facing
metric and a poor prefill metric.

### Phase-separated instrumentation

```python
from dataclasses import dataclass


@dataclass
class PhaseTimings:
    queue_ms: float | None = None
    prefill_ms: float | None = None
    first_token_ms: float | None = None
    decode_ms: float | None = None
    sample_ms: float | None = None


def derive_throughput(prompt_tokens: int, output_tokens: int, t: PhaseTimings):
    out = {}
    if t.prefill_ms:
        out["prefill_tok_s"] = prompt_tokens / (t.prefill_ms / 1000.0)
    if t.decode_ms and output_tokens > 1:
        # Sustained decode: N-1 intervals between N emitted tokens.
        out["decode_tok_s"] = (output_tokens - 1) / (t.decode_ms / 1000.0)
        out["tpot_ms"] = t.decode_ms / (output_tokens - 1)
    return out
```

## Memory accounting

The KV cache dominates memory for long contexts. Per token, a transformer with
`L` layers, `H` KV heads and head dimension `D` stores `2 * L * H * D` elements
in the cache; multiply by batch tokens and by the element size to get bytes.

| Model size | Layers | KV heads | Head dim | KV bytes/token (fp16) | 4K ctx   | 32K ctx   |
|------------|--------|----------|----------|-----------------------|----------|-----------|
| 1.1B       | 22     | 4        | 64       | 22,528                | 90 MB    | 720 MB    |
| 3B         | 32     | 8        | 128      | 131,072               | 524 MB   | 4.2 GB    |
| 8B         | 32     | 8        | 128      | 131,072               | 524 MB   | 4.2 GB    |
| 34B        | 60     | 8        | 128      | 245,760               | 983 MB   | 7.9 GB    |

Grouped-query attention shrinks this proportionally to the ratio of KV heads to
query heads; multi-query attention shrinks it to a single head per layer. These
architectural choices are invisible in output quality benchmarks and decisive
for serving capacity.

### Allocation policy

```json
{
  "kv_cache": {
    "strategy": "preallocate",
    "budget_bytes": 6442450944,
    "block_size_tokens": 16,
    "eviction": "lru",
    "watermark": 0.92
  },
  "admission": {
    "max_batch_tokens": 8192,
    "max_running_requests": 8,
    "reject_on_full": true
  }
}
```

## Chunked prefill

Long prompts used to block every other request until prefill finished. Chunked
prefill splits the prompt into bounded chunks and interleaves them with decode
steps, trading a little prefill throughput for dramatically better decode
latency under mixed load. The scheduling loop below shows the shape of the
decision:

```c
// Simplified scheduler step: interleave decode and prefill chunks.
typedef struct {
    Request* running[MAX_RUNNING];
    size_t   running_len;
    size_t   chunk_budget_tokens;
} Scheduler;

void scheduler_step(Scheduler* s, Pool* pool) {
    size_t budget = s->chunk_budget_tokens;

    // 1. Every running request gets one decode step.
    for (size_t i = 0; i < s->running_len; ++i) {
        request_decode_step(s->running[i], pool);
    }

    // 2. Spend the remaining budget on prefill chunks.
    while (budget > 0 && queue_has_waiting()) {
        Request* r = queue_peek_prefill();
        size_t chunk = request_next_chunk(r, budget);
        if (chunk == 0) break;
        request_prefill_chunk(r, chunk, pool);
        budget -= chunk;
        if (request_prefill_done(r)) {
            queue_promote_to_running(s, r);
        }
    }
}
```

The measurement that validates chunked prefill is not aggregate throughput but
inter-token latency of the running streams while a long prompt arrives. See the
scheduler-interference workload of the benchmark suite.

## Warmup is part of the measurement contract

Cold-start behavior and steady-state behavior must be recorded separately.

```text
phase                     metric                        cold / warm
------------------------- ----------------------------- -----------
model load                load_ms                       cold
cuda graph capture        capture_ms                    cold
first request             ttft_ms                       cold
steady request            ttft_ms                       warm
steady decode             tpot_ms                       warm
```

A run that mixes cold and warm samples into one distribution reports a number
that describes no real deployment. The benchmark therefore tags every request
with its cache state and warmup status, and aggregates them in separate groups.

## Validation checklist before publishing numbers

1. Confirm the model revision, quantization and tokenizer are identical across
   compared runs.
2. Confirm input IDs are identical (hash them; do not assume).
3. Confirm `requested_output_tokens == actual_output_tokens` or that early
   termination is explained by finish reason.
4. Confirm warmup runs are excluded from headline statistics.
5. Confirm telemetry gaps are recorded as `UNAVAILABLE`, never zero-filled.
6. Confirm the session manifest captures GPU, driver, and CUDA versions.
7. Confirm failures (OOM, timeout) are listed with their workload, not dropped.

## Operator notes

```bash
# Inspect the last session's failures without rerunning anything.
jq -c 'select(.execution_status != "SUCCESS")' results/<session-id>/raw.jsonl

# Recompute a summary from raw data months later.
python -m benchmarks report <session-id>

# Regenerate plots only (reporting is independent of measurement).
python -m benchmarks report <session-id> --plots
```

The design principle behind all of the above: raw records are the source of
truth, and every derived artifact must be reconstructible from them alone.
