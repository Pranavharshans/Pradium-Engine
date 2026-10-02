# Pradium Engine

Repository for the Pradium inference engine and its tooling.

## Pradium Runtime Benchmark

`benchmarks/` contains **PRADIUM-RUNTIME-BENCH-v2**, a framework-neutral
inference-runtime benchmark suite. It measures any runtime (ExLlamaV3, SGLang,
TensorFold, vLLM, Pradium, future runtimes) through a generic adapter API and
owns its workloads, prompts, token accounting, telemetry, statistics and
reporting.

```bash
python -m benchmarks --help
python -m benchmarks corpus validate
python -m benchmarks run matrix --runtime mock --quick
```

See [`benchmarks/README.md`](benchmarks/README.md) for the full documentation:
workload matrix, metric formulas, suites, result format, and how to add a
runtime adapter.

## Upstream inference engines

`engine/exllamav3/` contains the official ExLlamaV3 engine as a pinned Git
submodule. See [`engine/README.md`](engine/README.md) for the upstream source
and clone instructions.
