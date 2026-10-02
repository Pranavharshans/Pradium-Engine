# Pradium Runtime Benchmark — `PRADIUM-RUNTIME-BENCH-v2`

A **framework-neutral inference-runtime benchmark**. The benchmark owns its
workloads, prompts, token accounting, telemetry, statistics, result format and
reporting; inference runtimes plug in through a generic adapter API.

```text
                    Pradium Runtime Benchmark
                              |
             +----------------+----------------+
             |                |                |
          Workloads         Metrics          Reports
             |                |                |
       S/M/L matrix      latency/VRAM       JSONL
       Context scale     CPU/GPU/power      JSON
       Concurrency       throughput         Markdown
       Prefix reuse      ITL                plots
       Mixed loads       RAM
             |
             v
                    Generic Runtime API
                              |
          +-------------------+-------------------+
          |                   |                   |
       ExLlamaV3           SGLang              vLLM
          |
       future adapters
```

The benchmark core **never** contains framework-specific inference logic.
Runtime-specific behavior lives only in adapters. No real runtime is installed
or assumed here; the bundled `mock` adapter validates the whole pipeline.

---

## Purpose and philosophy

* One frozen definition of "how to measure an inference runtime", reusable
  across ExLlamaV3, SGLang, TensorFold, vLLM, Pradium and future runtimes.
* Exact, identical workloads for every runtime: same token IDs, same lengths,
  same generation parameters, same warmup/measured-run policy.
* Raw data is sacred: every request produces a raw JSON record immediately;
  summaries, reports and plots are derived artifacts regenerable at any time.
* Honest numbers: metric provenance is recorded, unavailable data stays
  unavailable, correctness status is never conflated with performance validity.
* No database, no web UI, no external datasets, no network access at run time.

## Quick start

```bash
# validate the frozen prompt corpus
python -m benchmarks corpus validate

# quick development run over the mock runtime (never official)
python -m benchmarks run matrix --runtime mock --quick

# official-style full matrix (3 warmups, 10 measured runs; mock = validation data)
python -m benchmarks run matrix --runtime mock

# a subset of the matrix
python -m benchmarks run matrix --runtime mock --profiles SS,MM,LL

# every suite
python -m benchmarks run concurrency --runtime mock
python -m benchmarks run context --runtime mock
python -m benchmarks run prefix-cache --runtime mock
python -m benchmarks run scheduler --runtime mock
python -m benchmarks run mixed --runtime mock
python -m benchmarks run startup --runtime mock
python -m benchmarks run batching --runtime mock
python -m benchmarks run soak --runtime mock --duration 60

# regenerate summary/report/plots from raw data (months later, no rerun)
python -m benchmarks report <session-id>

# benchmark identity and frozen components
python -m benchmarks version --frozen
```

`python -m benchmarks --help` lists everything.

---

## The 3x3 core matrix

All combinations of input length and output length are official workloads:

```text
                OUTPUT
              S=64   M=256   L=1024
INPUT S=128    SS      SM      SL
INPUT M=1024   MS      MM      ML
INPUT L=4096   LS      LM      LL
```

| Profile | prompt tokens | output tokens | Purpose |
|---------|---------------|---------------|---------|
| SS | 128 | 64 | tiny interactive request; framework fixed overhead |
| SM | 128 | 256 | normal short-prompt generation; interactive decode |
| SL | 128 | 1024 | decode-heavy; sustained generation speed |
| MS | 1024 | 64 | prefill-heavy short answer; coding/agent lookup |
| MM | 1024 | 256 | canonical normal workload |
| ML | 1024 | 1024 | substantial prefill + substantial decode |
| LS | 4096 | 64 | strongly prefill-dominated; RAG / long agent context |
| LM | 4096 | 256 | long-context interactive workload |
| LL | 4096 | 1024 | heavy end-to-end workload |

**Why is L = 4096?** Deliberately: the core matrix must stay practical on
constrained GPUs and larger models. Extreme contexts live in the separate
context-scaling suite. Do not change L to 32K just because a model supports it.

Definitions are frozen in `benchmarks/configs/matrix.yaml` and validated
against `benchmarks/core/workloads.py` at load time.

## Extended context scaling

```bash
python -m benchmarks run context --runtime mock
```

Input lengths `128, 512, 1024, 2048, 4096, 8192, 16384, 32768`, fixed 256-token
output. Cases that cannot run are recorded as `OOM` / `UNSUPPORTED` / `FAILED`
and the suite continues — never silently dropped. Produces TTFT, prefill tok/s,
decode tok/s, TPOT, peak VRAM and KV memory vs context length.

---

## Frozen prompt corpus

Prompts live in the repository (`benchmarks/corpus/`) and are never randomly
generated at run time. Five realistic source documents (conversational, technical,
code, structured, mixed) are frozen with SHA-256 hashes in
`corpus/manifest.json`. See [`corpus/README.md`](corpus/README.md).

Exact token counts are materialized deterministically: a token stream is
composed from the sources and truncated to **exactly** 128 / 1024 / 4096 / ...
token IDs. Materialized prompts record `prompt_hash`, `input_ids_hash`,
tokenizer name/revision and token count. Validation fails on any mutation.

```bash
python -m benchmarks corpus validate
python -m benchmarks corpus materialize --tokenizer simple
python -m benchmarks corpus materialize --tokenizer hf:<name>@<revision>
```

Runtimes should consume `BenchmarkRequest.input_ids` so one runtime's
tokenization cannot change the workload; text-only use is recorded as a
fairness deviation.

---

## Metric definitions (exact formulas)

All timing uses a monotonic clock (`time.perf_counter_ns`). Wall-clock is
metadata only. Units: ms, tok/s, ms/token.

```text
TTFT          = first_token_ts - request_submit_ts                      [ms]

decode_window = last_token_ts - first_token_ts                          [ms]
decode_tok_s  = (N - 1) / decode_window_s                (N > 1 tokens)
TPOT          = decode_window_ms / (N - 1)                (N > 1)  [ms/token]

E2E           = last_token_ts - request_submit_ts                       [ms]

ITL_k         = token_ts_k - token_ts_(k-1)              (k = 2..N)     [ms]

output_tok_s_per_request = N / e2e_s
prefill_tok_s            = prompt_tokens / prefill_s     (runtime-reported only)
```

**Prefill is never inferred from TTFT.** TTFT includes queueing, scheduling,
preprocessing, prefill, sampling and first-token delivery. `prefill_ms` is
stored only when the runtime reports it; otherwise provenance is `UNAVAILABLE`.

Aggregate metrics use the **global wall-clock interval** of a concurrent group,
never sums of per-request rates:

```text
aggregate_output_tok_s  = total output tokens / wall_clock_s
aggregate_prompt_tok_s  = total prompt tokens / wall_clock_s
aggregate_total_tok_s   = (output + prompt tokens) / wall_clock_s
aggregate_decode_tok_s  = sum(N_i - 1) / wall_clock_s
requests_per_second     = completed requests / wall_clock_s
completed_requests_per_minute = completed requests / (wall_clock_s / 60)
```

Concurrency scaling (relative to C1):

```text
scaling_efficiency(Cn)            = aggregate_decode(Cn) / (n * decode(C1))
per_request_throughput_degradation = decode_per_request(Cn) / decode_per_request(C1) - 1
TTFT_degradation_vs_C1            = TTFT(Cn) / TTFT(C1) - 1
TPOT_degradation_vs_C1            = TPOT(Cn) / TPOT(C1) - 1
```

Prefix reuse:

```text
TTFT_speedup      = median TTFT(reuse=0%) / median TTFT(reuse=r%)
prefill_speedup   = median prefill(reuse=0%) / median prefill(reuse=r%)
memory_cost_per_cached_token = additional VRAM / cached tokens (where reported)
```

Efficiency metrics (median-based; **no single overall score**):

```text
decode_tok_s_per_gb_vram       = decode_tok_s / (vram_peak_mb / 1024)
aggregate_tok_s_per_gb_vram    = aggregate_output_tok_s / (vram_peak_mb / 1024)
decode_tok_s_per_watt          = decode_tok_s / gpu_power_avg_w
aggregate_tok_s_per_watt       = aggregate_output_tok_s / gpu_power_avg_w
output_tokens_per_joule        = output_tokens / (gpu_power_avg * e2e_s)
ttft_per_1k_prompt_tokens      = ttft_ms / (prompt_tokens / 1000)
vram_per_concurrent_request_mb = vram_peak_mb / concurrency
ram_per_concurrent_request_mb  = ram_peak_mb / concurrency
```

Power is **NVML GPU-side power**, not whole-system power. Derived metrics carry
provenance `DERIVED`; statistics cover `count, mean, median, min, max, stdev,
p50, p90, p95, p99, cv`. The **headline statistic is the median** — never the
fastest run. Warmup runs are recorded but excluded from all statistics.

### Metric provenance

Every origin-sensitive metric records provenance:

`MEASURED_EXTERNAL` (harness-measured), `REPORTED_RUNTIME` (runtime-reported),
`DERIVED`, `ESTIMATED`, `UNAVAILABLE` — especially for prefill latency, queue
latency, scheduler latency, KV memory and cache hit rate.

### Statuses are separated

```text
execution_status   = SUCCESS | OOM | FAILED | TIMEOUT | UNSUPPORTED
performance_valid  = true | false
correctness_status = PASS | WARNING | FAIL | SKIPPED
```

A correctness warning never erases valid telemetry. Failures are persisted to
`failures.jsonl` with runtime/model/profile/concurrency/tokens/error/timestamp,
and the suite continues where safe.

---

## Suites

| Suite | Command | What it measures |
|-------|---------|------------------|
| matrix | `run matrix` | the 3x3 SS..LL workload matrix |
| context | `run context` | 128..32768 input scaling, 256-token output |
| concurrency | `run concurrency` | full matrix at C1/C2/C4/C8 |
| prefix-cache | `run prefix-cache` | KV/prefix reuse at 0/25/50/75/90/~100% |
| scheduler | `run scheduler` | large prefill arriving into active decoders |
| mixed | `run mixed` | interactive burst, agent-like, mixed generation/sizes, long-job interference |
| startup | `run startup` | process start → ready → first token; cold vs warm |
| batching | `run batching` | static batches B1/B2/B4/B8 (≠ concurrency) |
| soak | `run soak` | long-run stability, memory growth, degradation |

### Concurrency (C1/C2/C4/C8)

Cn means **n simultaneous active requests** released from a shared start gate —
never sequential submission. Actual submission timestamps are recorded per
request; every request keeps its own timing record. Aggregate throughput comes
from the group's global wall clock. The harness supports the entire 3x3 matrix
at every level:

```bash
python -m benchmarks run concurrency --runtime mock --profiles SS,SM,SL,MS,MM,ML,LS,LM,LL --levels 1,2,4,8
```

### Prefix / KV reuse

Ratios are defined on **token IDs** (`shared_prefix_tokens / total`). At ~100%
each request is a large identical prefix (1008 tokens) plus a small unique
suffix (16 tokens) — never exact duplicate prompts. Scenarios:
`same_session`, `cross_request`, `cross_session` (a session boundary between
prime and measure). Measures TTFT/prefill/decode, reused vs recomputed tokens,
cache hit rate and speedups vs the 0% baseline; internal cache metrics stay
`UNAVAILABLE` when the runtime does not expose them.

### Scheduler interference

Requests A/B/C decode while D arrives with a large prefill (LS: 4096 tokens).
From the per-token traces the harness derives ITL **before / during / after**
D's prefill window, the maximum output-token stall, p95 ITL, D's TTFT, and
aggregate throughput — exposing whether a long prefill stalls active streams.

### Cold vs warm

`cold runtime / warm runtime`, `cold cache / warm cache`, `cold first request /
steady-state request` are recorded as metadata and **never combined into one
statistic**.

---

## Telemetry

Background sampling at 100 ms (configurable; `--deep-telemetry` uses 25 ms):

* **GPU**: utilization, memory-controller utilization, clocks, temperature,
  power draw, VRAM usage, PCIe RX/TX (NVML or `nvidia-smi`; optional)
* **VRAM**: before init / after init / after load / after warmup / before
  request / peak / after / steady state / after shutdown (MiB), plus optional
  runtime-reported decomposition (weights, KV cache, CUDA graphs, workspace)
* **RAM**: process RSS + peak, system totals (MiB)
* **CPU**: process/system CPU %, process CPU time, thread counts

Missing telemetry is `UNAVAILABLE` — never fabricated or zero-filled. Sample
overhead is measured (`overhead_per_sample_ms`), not assumed. Nsight-class deep
profiling is out of scope for normal runs.

---

## Benchmark modes

| Mode | Warmups | Measured | Profiles | Official? |
|------|---------|----------|----------|-----------|
| `quick` | 1 | 2 | SS, MM (subset per suite) | never |
| `validation` | 1 | 2 | SS, MM, LL | never |
| `official` | 3 | 10 | SS..LL | yes |

`--quick` is shorthand for quick policy. **Mock-runtime sessions are always
recorded as `benchmark_mode = validation`, `performance_valid = false`,
whatever the policy** — mock numbers are infrastructure data, never real
benchmark results.

---

## Result structure

```text
results/<session-id>/
├── environment.json     # OS, Python, GPU, driver, CUDA, CPU, RAM, git, CLI
├── config.json          # full benchmark configuration + CLI arguments
├── corpus_manifest.json # corpus validation report + tokenizer revision
├── raw.jsonl            # every request: the source of truth
├── summary.json         # aggregated statistics + derived metrics
├── report.md            # human-readable report (facts, no winner)
├── failures.jsonl       # every failure with context
├── traces/<request>.json# per-token timing traces (token_index, id, rel ts)
└── plots/*.svg          # plots (regenerable)
```

Session ids look like `20261002-091530_mock_validation_a1b2c3`
(timestamp, runtime, mode, unique suffix). Existing sessions are **never**
overwritten. Raw records contain the full field set: benchmark version,
runtime/model/tokenizer revisions, hardware, profile, run index, warmup flag,
prompt/input-ids hashes, token counts, batch/concurrency placement, prefix
data, monotonic timestamps, TTFT/prefill/decode/TPOT/ITL/E2E, queue/scheduler,
GPU/CPU/RAM/VRAM telemetry, aggregates, correctness data, provenance and
statuses.

---

## Using the mock runtime

The mock adapter simulates TTFT, inter-token latency, concurrency contention,
prefill interference, prefix-cache reuse, deterministic outputs and injectable
failures (OOM, exception, early EOS, partial generation, timeout) — without any
model. It validates the pipeline end to end.

```bash
python -m benchmarks run matrix --runtime mock --quick --time-scale 0.05
```

`--time-scale` scales simulated delays (0 = instantaneous; default 0.1). Mock
results always carry `runtime = mock`, `benchmark_mode = validation`,
`performance_valid = false`. **Never present mock performance as real data.**

---

## How to add a runtime adapter

A future agent should be able to implement an ExLlamaV3 (or other) adapter
without touching benchmark core. Steps:

1. **Implement the adapter** — subclass
   `benchmarks.adapters.base.RuntimeAdapter` in
   `benchmarks/adapters/<runtime>.py`, implementing
   `initialize()`, `load_model()`, `unload_model()`, `shutdown()`,
   `generate()`, `generate_stream()`, `get_capabilities()`,
   `get_runtime_info()`, `get_memory_info()`; optionally
   `generate_batch()`, `reset_cache()`, `clear_prefix_cache()`,
   `get_startup_milestones()`, `get_internal_metrics()`.
2. **Declare capabilities** — return a `RuntimeCapabilities` with
   `SUPPORTED` / `PARTIAL` / `UNSUPPORTED` / `UNKNOWN` for every capability
   (`supports_streaming`, `supports_input_ids`, `supports_fixed_decode_length`,
   `supports_prefix_cache`, `supports_cross_request_prefix_cache`,
   `supports_continuous_batching`, `supports_static_batching`,
   `supports_chunked_prefill`, `supports_cuda_graphs`,
   `supports_kv_quantization`, `supports_cache_reset`,
   `supports_internal_queue_metrics`, `supports_internal_scheduler_metrics`,
   `supports_internal_kv_metrics`). The harness capability-gates everything.
3. **Provide runtime metadata** — `get_runtime_info()` must return name,
   version, commit, model and model revision (recorded in every raw record).
4. **Implement stream generation** — `generate_stream()` yields `StreamToken`
   objects in emission order and a final `done=True` event carrying
   `finish_reason` plus runtime-reported `prefill_ms` / `queue_ms` /
   `scheduler_ms` / internal metrics where available. If you only have
   non-streaming generation, yield tokens as they are produced internally or
   declare `supports_streaming = UNSUPPORTED` (ITL metrics will record
   `UNAVAILABLE`).
5. **Map input IDs** — consume `BenchmarkRequest.input_ids` directly. If the
   runtime requires text, use `request.prompt_text` and declare
   `supports_input_ids = UNSUPPORTED`; the harness records the fairness
   deviation.
6. **Map fixed output length** — respect `max_new_tokens` with
   `fixed_decode_length` / `ignore_eos` semantics where the runtime allows;
   declare `supports_fixed_decode_length` honestly. Always return the true
   `actual_output_tokens` and `finish_reason`.
7. **Expose internal metrics** — queue/scheduler/KV/cache values via the final
   stream event or `get_internal_metrics()`; mark
   `supports_internal_*_metrics` accordingly. Do not fabricate values.
8. **Register** — `register_adapter("exllamav3", factory)` in
   `benchmarks/adapters/registry.py` (or your module imported by it).
9. **Run adapter conformance tests** — subclass
   `benchmarks.tests.conformance.AdapterConformanceMixin` in your test module
   and implement `make_adapter()`; the shared suite checks lifecycle, stream
   order, token counts, determinism and capability declaration.
10. **Run QUICK, then OFFICIAL**:

```bash
python -m benchmarks run matrix --runtime exllamav3 --quick
python -m benchmarks corpus materialize --tokenizer hf:<model-tokenizer>@<rev>
python -m benchmarks run matrix --runtime exllamav3
python -m benchmarks run concurrency --runtime exllamav3
```

Fairness: identical model revision, quantization, tokenizer, input IDs, lengths,
generation parameters (greedy, temperature 0), concurrency, ordering, warmups,
measured runs and hardware across runtimes. Framework-specific configuration
differences must be recorded, never silently tuned.

---

## Versioning

`PRADIUM-RUNTIME-BENCH-v2` is stored in every result file. The following are
frozen; changing any requires a version bump (`VERSION`,
`benchmarks/core/version.py`, schemas):

prompt corpus · token-length definitions · workload definitions · output
lengths · timing definitions · warmup count · measured-run count · statistical
methodology · concurrency methodology · cache test methodology.

Config loading fails validation if the frozen lengths/levels change without a
version bump. `python -m benchmarks version --frozen` lists the frozen items.

## Repository layout

```text
benchmarks/
├── VERSION, README.md, __main__.py, cli.py
├── core/        # request, capabilities, timing, results, session, config, ...
├── corpus/      # frozen sources, tokenizer, materializer, manifest.json
├── configs/     # matrix/context/concurrency/prefix_cache/scheduler/... yaml
├── adapters/    # base.py (interface), mock.py, registry.py
├── runners/     # matrix, context, concurrency, prefix_cache, scheduler,
│                # mixed, startup, soak, batching + shared collection
├── telemetry/   # gpu.py, cpu.py, memory.py, sampler.py
├── analysis/    # statistics, aggregate, report, plots
├── schemas/     # raw_result.schema.json, summary.schema.json
└── tests/       # unit + conformance + end-to-end pipeline tests
```

Run tests:

```bash
python -m unittest discover -s benchmarks/tests -t .
```

## v2 measurement corrections

v2 results must not be directly compared with v1 results. Concurrent group
throughput now uses earliest submission to latest output token, excluding
telemetry processing and trace writes, matching the C1 boundary. Pre-request
telemetry snapshots are outside submission timing.

Ordinary matrix, context and concurrency groups clear prefix caches before
execution when cache reset is supported. A cache-enabled adapter without reset
support makes the context performance-invalid. Prefix cases clear caches and
prime a different suffix, so the measured prompt shares only the intended
prefix; the zero-reuse baseline also uses a distinct prime. Text-only adapters
receive the materialized prompt text; retokenization remains a fairness deviation.

A streaming deadline covers blocked token delivery, final events and stream
exhaustion. Python cannot forcibly kill an adapter thread: on timeout, no more
generation is started through that context. Adapters must ensure shutdown can
clean up outstanding work; process isolation is needed for hard cancellation.
Static batch calls currently rely on adapter-side deadlines. Missing batch
outputs fail and preserve every requested row and failure record.

Correctness PASS checks the output-length/stream contract, not semantic model
accuracy. Truncated, empty or failed streaming results are performance-invalid;
raw telemetry remains available. Semantic validation and real GPU adapters
remain future work.
