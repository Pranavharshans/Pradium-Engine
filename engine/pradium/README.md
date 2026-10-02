# Pradium standalone inference work

Goal: 2x single-request decode throughput over the pinned ExLlamaV3 baseline,
with matching weights, precision, cache conditions, and output quality.

## Measured status (RTX 3060 12 GB, MiniCPM5-2B EXL3, batch 1, 2026-10-02)

Nine core workloads (SS..LL), medians of 10 measured runs after 3 warmups under
the frozen harness configuration, on the authorized VM. Raw records, paired
step-time profiles, correctness diffs and per-experiment verdicts:
`experiments/` (ledger: `experiments/LEDGER.md`).

| patch | verdict | measured effect |
|---|---|---|
| 0001 graph replay argument bindings | KEEP | no regression, no measurable change (host bookkeeping only); the build reproduces the frozen session's greedy tokens exactly, 117/117 |
| 0002 direct single-request attention | REJECT as configured | wins at 192-token contexts (step 7.84 -> 7.49 ms, combine kernel eliminated, kernel total 7.239 -> 6.846 ms) but loses 2-76% of the step at 384-2048 tokens; the padded-width gate cannot separate those cases |
| 0003 static decode buffers | REJECT | within +-1% of its parent on every workload's steady-state step time |
| 0004 combine kernel optional for regime 2 | KEEP (prerequisite) | without it every direct-regime call raises in `configure` |
| 0005 precise live-token gate for the direct regime | REJECT (correctness) | removes every coarse-gate loss and keeps the SS gain, but still flips a greedy token in SM (13/13 runs, first divergence at token 73) |

**No 2x, and no accepted net speedup.** The measured ceiling explains it: a
decoded token reads 1.19 GB of EXL3 weights (991 MB in the 294 layer linears
at 4.01 bpw, 191 MB in the 6-bit LM head) and the VM's measured peak read
bandwidth is ~330 GB/s, so the pure weight-read floor is 3.6 ms/token against
the baseline's 8.9 ms at SS. A 2.00x step (4.45 ms) would have to fit
attention, activations, sampling, host work and sync into 0.85 ms. The
remaining gap to the floor is EXL3 kernel efficiency at the M=1 layer shapes
(199-250 GB/s vs the 321 GB/s the same format reaches in the LM head), which
the forced-shape sweep shows is not recoverable by dispatch.

Reproduction: `experiments/EXP-0000-baseline-freeze/README.md` (baseline and
profiling), `experiments/EXP-0001-patch-series/README.md` (build + rounds),
`experiments/tools/run_full_campaign.sh` (unattended measurement).

The previous text of this section — "No speedup or GPU correctness has been
demonstrated for these changes yet" — is superseded by the table above.

The official source stays in `../exllamav3`, pinned to
`d3739fd393337b1ff4d6c2a342b12f0c87a9592f`. Our changes are a versioned patch
series rather than an upstream history dump or an unpublished submodule commit.
The generated runtime retains the upstream license.

## Implemented changes

- `0001`: cache ordered CUDA graph replay argument bindings; avoid copying the
  argument vector into `Graph::launch`. Repeated parameter names remain separate
  sites. Pointer and packed scalar updates retain the original byte comparisons,
  runtime/driver API distinction, and changed-node update behavior. This reduces
  host bookkeeping, not the number of graph launches or GPU operations.
- `0002`: an experimental separate attention graph slot for single-request,
  single-token, symmetric-head dense attention. It uses the existing kernel's
  `FINAL` output path and omits the combine launch and its replay parameter.
  Sinks are included in the direct kernel. Quantized KV uses the existing
  rotation/dequantization path but has not been validated here. QSA, asymmetric
  V heads, multiple requests, and multi-token queries use upstream paths.
  Crossing the configured context bound selects the original graph slot again.

Direct attention is **disabled by default**. A larger single-pass tile traversal
can lose GPU parallelism, so enabling it is an experiment, not an assumed win.
Selection uses block-table capacity plus the appended token without a GPU-to-CPU
length readback. For example a 512-token table needs a limit of at least 513.

## Build and CPU checks

From the repository root (Python 3.12+ and Git):

```sh
python3 -m unittest discover -s engine/pradium/tests -v
python3 engine/pradium/build.py --output engine/pradium/runtime
```

The output directory must not exist. The builder exports the immutable upstream
pin, applies the patches, and writes `PRADIUM_BUILD.json` with patch SHA256s.
It never edits the upstream submodule. `runtime/` is ignored by Git.
The native replay-plan test uses a C++17 compiler with address/undefined-behavior
sanitizers and compares 10,000 randomized layouts against the original scan.
These CPU checks do not compile the CUDA extension.

## Required CUDA gates

On the authorized CUDA benchmark host, install the generated runtime using the
upstream installation instructions. Confirm `exllamav3.__file__` points to it;
use separate upstream and patched environments so imports cannot mix revisions.
Then run this synthetic kernel differential test from the repository root:

```sh
python3 engine/pradium/tests/gpu_direct_attention.py
```

Missing CUDA/dependencies fail this gate. It compares direct and split attention
with an FP32 reference across page boundaries, padded head dimensions, windows,
and sinks, plus differential 4/8-bit KV cache cases. It does not certify the C++
block integration, other quantized KV configurations, or model
output quality; those need the full campaign below.

Run the existing campaign and conformance suite first with patch 0001 only,
then both patches with direct attention disabled, then with:

```sh
EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=513
```

Use identical prompts and precision, cold-cache and warm-cache cases separately,
full greedy token-ID comparisons, baseline/patched alternation, and warmups
excluded from measurements. Cover transitions between direct and split slots,
context growth, cache reuse/reset, graph pointer changes, FP16 and quantized KV,
and unsupported shapes. Run compute-sanitizer before accepting the CUDA changes.
Report C1 TTFT, decode throughput, ITL p99, VRAM and total runtime. Do not mark
2x achieved from kernel-only timings or from a different concurrency level.

## Remaining requested work

| Program | Status |
|---|---|
| Quantization-aware compiler and persistent GPU execution | Not implemented |
| GPU-managed request/cache state and decode control | Not implemented; patch 0001 only reduces existing host replay work |
| Cross-operator fusion beyond existing upstream fused blocks | Not implemented |
| New GEMV/GEMM kernels or measured cross-family autotuning | Not implemented |
| Attention/KV improvements | Direct attention experiment implemented; broader KV redesign not implemented |

A persistent execution backend needs dependency and memory-lifetime handling,
GPU state ownership, capture/dispatch compatibility, and differential GPU tests.
Those are substantial remaining implementation tasks. These patches are a first
increment, not completion of the five-item program or a 2x engine.
