# PRADIUM EXL3 B0 — RTX 3060 / MiniCPM5-2B EXL3 cross-runtime baseline

**Campaign:** `PRADIUM-EXL3-RTX3060-B0`
**Benchmark:** `PRADIUM-RUNTIME-BENCH-v2` (campaign spec label: `PRADIUM-RUNTIME-BENCH-v1`)
**Pradium commit:** `cf4e8754e6edcf588ebf0a207472e19c24f233d3`
**Branch:** `bench/exl3-runtime-b0-rtx3060`
**Date:** 2026-10-02

---

## 1. Executive summary

The campaign set out to produce a trustworthy cross-runtime EXL3 baseline for
`ewin-reg/MiniCPM5-2B-EXL3-Quantized` on the supplied RTX 3060. It produced one
full, verified baseline and three honest negative results.

**ExLlamaV3 1.5.3 is the only stack that could genuinely execute the checkpoint
on this VM, and it passed the entire official suite.** Its execution was proved
to be real EXL3 and not a silent dequantization: all **295/295** linear modules
report `quant_type == "exl3"`, at a measured **4.0127 bpw layer / 6.0079 bpw
head** against the checkpoint's declared 4.0 / 6. The adapter refuses to load if
that audit fails, so a fallback run cannot masquerade as an EXL3 result.

**The three other pinned runtimes could not run here, for two unrelated reasons:**

* **SGLang 0.5.20 + `sglang_exl3`** and **vLLM 0.30.0 + `vllm-exl3`** are blocked by
  a **CUDA-major/driver mismatch in the supplied environment**. Both pins hard-pin
  `torch==2.13.0`, which exists only as a CUDA 13 build, while the VM's driver
  (550.144.03) tops out at CUDA 12.4. No CUDA 12 build of torch 2.13.0 exists on
  any PyTorch index, and the VM is an unprivileged container so the driver cannot
  be changed. A direct probe returns `cudaSetDevice(0) → rc=35, "CUDA driver
  version is insufficient for CUDA runtime version"`. This is a property of the
  environment: it says nothing about EXL3 support in either framework, and nothing
  about MiniCPM5. Both plugins were cloned and inspected at their exact pinned
  revisions, so the re-run recipe is precise.
* **TensorFold 0.6.1** is `UNSUPPORTED_MODEL`: the pinned revision has no
  MiniCPM / dense-Llama family at all, and independently it refuses to start on
  this GPU (`MIN_CAPABILITY = (8, 9)`; RTX 3060 is 8.6). No porting and no
  substitute model were attempted, so TensorFold contributes no performance row.

Headline result on the one supported runtime (medians of 10 measured runs):

| Workload | TTFT | prefill tok/s | decode tok/s | TPOT | peak VRAM |
|---|---|---|---|---|---|
| SS 128→64 | 69.56 ms | 2106 | 109.89 | 9.12 ms | 2799 MB |
| MM 1024→256 | 237.44 ms | 4513 | 102.27 | 9.97 ms | 2951 MB |
| LL 4096→1024 | 875.31 ms | 4739 | 79.19 | 13.30 ms | 3031 MB |

The decisive structural finding for Pradium: on this 2-core host, decode is
**97–112 tok/s** and is essentially flat from 128 to 2048 input tokens, degrading
to **~72 tok/s at 16 K** context; prefill throughput *rises* with prompt length
(2100 → 4700 tok/s) because short prompts are launch/overhead dominated. The
model is tiny (1.14 GB of weights in 1.6 GB of VRAM), so on a 12 GB card the GPU
is never the binding constraint — **per-request latency and single-request decode
bandwidth are**, which is where Pradium's optimization attention belongs.

## 2. RTX 3060 environment (verified, not assumed)

| Item | Actual measured value |
|---|---|
| GPU | NVIDIA GeForce RTX 3060, 1 × 12288 MiB |
| GPU UUID | `GPU-9bdacdb0-33f1-e196-6c1f-dc65129fb653` |
| Architecture | Ampere, compute capability **8.6** (verified via PyTorch `get_device_capability`) |
| NVIDIA driver | **550.144.03** — supports CUDA runtime **12.4** maximum (`cuDriverGetVersion` = 12040) |
| System CUDA toolkit | 12.8.93 (`nvcc V12.8.93`) |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.8.0-60-generic |
| CPU | Intel Pentium Gold G5420 @ 3.80 GHz — **2 physical / 4 logical cores** |
| System RAM | 31 GiB |
| Disk | 80 GB overlay, ~70 GB free (container filesystem, not a host volume) |
| Python | 3.12.3 |
| Idle GPU state | ~217–300 MiB VRAM, 15 W, ~43 °C, SM 1935 MHz |
| Container | Unprivileged Docker container. **No Docker-in-Docker**, so container images (including the published `sglang-exl3` reference image) are unusable; the driver is host-managed. |

Two environment facts shaped the whole campaign and should be recorded for the
next one:

1. **The driver is the ceiling, not the GPU.** CUDA 12.4 max rules out the entire
   CUDA 13 generation of runtime stacks, which is where both pinned plugin stacks
   now live.
2. **The CPU is very weak** (2 cores, no AVX-512). Any runtime whose host-side
   tokenizer/scheduler work is non-trivial will be CPU-bound here long before it
   is GPU-bound. CPU utilisation under generation was already 130–175 %.

## 3. Exact model revision and hashes

| Item | Value |
|---|---|
| Repo | `ewin-reg/MiniCPM5-2B-EXL3-Quantized` |
| **Frozen revision** | **`e36fbb568f466739d7c244a5100d01acf93d3b7f`** |
| Base model | `openbmb/MiniCPM5-2B` (observed revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`) |
| Quantization | EXL3, 4.0 bpw, `head_bits` 6, codebook `mul1`, `out_scales` always, quantizer version 1.4.6 |
| Checkpoint size | 1 740 500 926 bytes (1.621 GiB), 1 shard, **no** `model.safetensors.index.json` |
| Architecture | `LlamaForCausalLM` / `model_type: llama`; 42 layers, hidden 2048, intermediate 6144, 16 heads, 2 KV heads, head_dim 128, vocab 130560, `max_position_embeddings` 131072, `tie_word_embeddings` false |
| Canonical dir | `/workspace/models/minicpm5-2b-exl3` (the only copy; every runtime loads this path) |

Key file hashes (full manifest in
[`../campaigns/rtx3060-minicpm5-exl3-b0/model_manifest.json`](../campaigns/rtx3060-minicpm5-exl3-b0/model_manifest.json)):

```text
config.json                dfc339b30dd6055b815a7e546b415345074e51eb7485b0571ac933203e1ea536
quantization_config.json   b176f78672cf686c9bda9fac0f973fab648364aba96349f5bcb0f9880bca0dc9
tokenizer.json             3e065a558a034185fe299917b398685c1facd0169a9eea1e629eb30c171fed81
tokenizer_config.json      e9b1064649e771d7a8e15637c68b2d2749724877ba3648bd74a4ece31c26303b
model.safetensors          4a310a644144c54f8146c3dbf58fcedc5ab97e8198d21e241c09fcfad5b6b2be
  header_sha256            55e9ea37a9d43c61bf673ccf144d7cc907d3369936cc34902b22897c61eac9e6
  data_section_sha256      ce4150b52c24e17ad10b3afb3d37f45b2cfcb8b9cd87fac25eada879d1c97256
  tensor_manifest_sha256   57b2969f1f0a867bb85bd96a927ac801b8361577da52d5ad1cef9af444da2a6b
  n_tensors                1266
```

The quantized linear tensor group in the shard is the standard exllamav3 EXL3 set
`trellis` / `suh` / `svh` / `mul1` (e.g. `model.layers.1.self_attn.q_proj.trellis`
= `I16[128,128,64]`, i.e. 16×16 tiles at 4 bpw), and `quantization_config.json` is
an exllamav3-native artifact — it is written by
`exllamav3/conversion/quant_config.py::create_quantization_config_json`, not by a
third-party converter. This was established from the metadata, not from the
repository name.

## 4. PRADIUM-RUNTIME-BENCH revision

The campaign spec names `PRADIUM-RUNTIME-BENCH-v1`. The repository's frozen
identity is **`PRADIUM-RUNTIME-BENCH-v2`**, and the campaign freezes on it:

| | |
|---|---|
| v1 | `32ecc1f` (first commit; the whole harness was developed under the v1 label) |
| v2 = HEAD | `cf4e875` — *"Fix benchmark cache fairness, timing, deadlines and batch accounting"* |
| Starting commit | `cf4e8754e6edcf588ebf0a207472e19c24f233d3` on `bench/generic-runtime-benchmark-v1` |
| Campaign branch | `bench/exl3-runtime-b0-rtx3060` |

The version constant in `benchmarks/core/version.py` and the `VERSION` file were
bumped v1 → v2 in the same commit that fixed cache fairness, timing, deadlines and
batch accounting. Those are genuine measurement-correctness fixes, so freezing on
the earlier v1 would mean knowingly benchmarking with broken fairness and timing.
No benchmark definition was modified: prompts, token lengths, output lengths,
warmup count, measured-run count, timing formulas, concurrency methodology,
prefix-cache methodology and report metric definitions are all unchanged from the
committed tree.

Verification performed before any real-runtime work: the suite's own tests pass
(**119 passed**) and the mock end-to-end pipeline runs
(`python -m benchmarks run matrix --runtime mock --quick` → 6 raw records).

**Frozen inputs.** Prompt IDs were materialized once with the exact model
tokenizer, and every runtime consumes those IDs directly (the adapter never
re-tokenizes):

```bash
python -m benchmarks corpus materialize \
  --tokenizer hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@e36fbb568f466739d7c244a5100d01acf93d3b7f
```

This writes a per-tokenizer side manifest holding `input_ids_sha256` and
`prompt_sha256` per workload to
`benchmarks/corpus/materialized/ewin-reg/MiniCPM5-2B-EXL3-Quantized/manifest.json`.
The frozen simple-tokenizer `benchmarks/corpus/manifest.json` is untouched and
still validates (`corpus validation: OK (5 sources)`).

**Repeat policy.** 3 warmups + 10 measured runs per official workload; warmups are
never counted and never cherry-picked; every individual run is stored in
`raw.jsonl`. Headline statistic is the **median**, with mean, std, p95 and p99
retained.

## 5. Runtime compatibility matrix

| Runtime stack | EXL3 route | MiniCPM5 EXL3 status | Benchmark status |
|---|---|---|---|
| **ExLlamaV3 1.5.3** @ `d3739fd` | Native | `SUPPORTED_NATIVE` | **Full official suite run** |
| **SGLang 0.5.20** @ `94602c9` + **`sglang_exl3`** @ `1ace59c4` | Plugin (`sglang.srt.plugins`) | `INSTALL_FAILURE` — CUDA 13 / driver 550 | Not runnable on this VM |
| **vLLM 0.30.0** @ `ced6857` + **`vllm-exl3`** @ `223e246f` | Plugin (`--quantization exl3`) | `INSTALL_FAILURE` — CUDA 13 / driver 550; model support *not disproven* | Not runnable on this VM |
| **TensorFold 0.6.1** @ `17c73e18` | EXL3 exists, family-gated | `UNSUPPORTED_MODEL` (no MiniCPM family) + refuses sm_86 | Not runnable on this GPU |

Full reasoning: [`../campaigns/rtx3060-minicpm5-exl3-b0/compatibility.md`](../campaigns/rtx3060-minicpm5-exl3-b0/compatibility.md).

### 5.1 Why the two plugin stacks failed (and why it is not their fault)

Both pinned stacks resolve to a CUDA 13 dependency set:

```text
sglang==0.5.20  ->  torch==2.13.0, nvidia-cudnn-cu13==9.20.0.48, nvidia-nvshmem-cu13==3.4.5,
                    nvidia-cuda-crt==13.4.92, flashinfer-python==0.6.18, sglang-kernel==0.4.7   (200 pkgs)
vllm==0.30.0    ->  torch==2.13.0, nvidia-cudnn-cu13==9.20.0.48, nvidia-nvshmem-cu13==3.4.5,
                    nvidia-cutlass-dsl[cu13]==4.7.1, flashinfer-python==0.6.18.post1            (192 pkgs)
```

`torch==2.13.0` on PyPI is a CUDA 13.0 build (`cuda-toolkit[...]==13.0.3`,
`nvidia-*-cu13`, `cuda-bindings>=13.0.3`), and it has **no CUDA 12 build on any
PyTorch index** — cu126, cu128 and cu129 all stop at 2.9.1. CUDA minor-version
compatibility does not cross major versions, so the whole generation is
unreachable from driver 550. The direct probe:

```text
cudaRuntimeGetVersion -> 13.0
cudaDriverGetVersion  -> 12.4
cudaSetDevice(0)      -> rc=35: CUDA driver version is insufficient for CUDA runtime version
```

**Re-run recipe:** retry both pinned stacks unchanged on a host with driver
r580+ (or any CUDA-13-capable driver). The EXL3 compatibility gates still have to
be passed empirically there — nothing in this campaign proves or disproves
MiniCPM5 support in either framework.

### 5.2 What the plugins actually are (inspected at their pinned revisions)

* `0xSero/trellis-serve` @ `1ace59c4` — registers `--quantization exl3` through the
  `sglang.srt.plugins` entry point `exl3 = sglang_exl3.plugin:activate`.
  `get_min_capability()` is 80, so Ampere is *in scope by declaration*. Kernel
  backends are `auto|exllamav3|marlin`; with `trellis_exl3_kernels` absent, `auto`
  falls back to **ExLlamaV3's own bit-faithful kernels**, which is the documented
  path and would have avoided a multi-hour Marlin build on this 2-core CPU.
* `vcruz305/vllm-exl3` @ `223e246f` — version 0.5.0, entry point
  `vllm.general_plugins → vllm_exl3.bf16_madv_compat:register`. It registers a
  generic `@register_quantization_config("exl3")` method whose
  `get_config_filenames()` is exactly `["quantization_config.json"]` (the file this
  checkpoint ships) and whose `get_min_capability()` is 80
  (*"LinearEXL3 uses CUDA >= Ampere; GB10 is SM121"*), serving dense "declared EXL3
  tensors" through `exllamav3_ext`. Its *declared* qualifications, however, are
  model-family specific — GLM-5.3-Flash, DeepSeek-V4/V4.1, Qwen4Exp — with the
  primary target being one **DGX Spark GB10 (sm_121 Blackwell)**. That is why
  MiniCPM5 support is recorded as **undetermined rather than negative**: no dense
  family integration is declared, but the quant method is architecturally generic
  and Ampere-capable, so it is plausible and simply could not be tested here.

## 6. Exact runtime/plugin stacks

| Runtime | Version | Commit | Artifact |
|---|---|---|---|
| ExLlamaV3 | 1.5.3 | `d3739fd393337b1ff4d6c2a342b12f0c87a9592f` | upstream wheel `exllamav3-1.5.3+cu128.torch2.10.0-cp312-cp312-linux_x86_64.whl` |
| SGLang | 0.5.20 | `94602c9c2b7cbdb8efd5c52802dac6a1c180089e` | not installable here (CUDA 13) |
| `sglang_exl3` (trellis-serve) | — | `1ace59c4b43ca16a50fb6b7acf8b3fd7e2351f96` | cloned + inspected |
| vLLM | 0.30.0 | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` | not installable here (CUDA 13) |
| `vllm-exl3` | 0.5.0 | `223e246ff8da552d2ace976c4bcbed804a8ec04c` | cloned + inspected |
| TensorFold | 0.6.1 | `17c73e189f5e6a5304cda7ea37f086f9c49b4788` | installed, refused at startup |

Every pin was verified against the remote before use. All six resolve exactly as
specified, including the two 7-character short pins (`94602c9` → the v0.5.20 tag
object; `ced6857` → the v0.30.0 tag object) and the annotated TensorFold tag
`v0.6.1` → commit `17c73e18`.

Installed ExLlamaV3 stack (also in
[`environments/exllamav3.pip-freeze.txt`](../campaigns/rtx3060-minicpm5-exl3-b0/environments/exllamav3.pip-freeze.txt)):
Python 3.12.3, torch 2.10.0+cu128, exllamav3 1.5.3+cu128.torch2.10.0,
transformers 5.18.0, tokenizers 0.23.2, numpy 2.5.3, safetensors 0.8.0, triton
3.6.0, llguidance 1.9.1, marisa-trie 1.4.1, pydantic 2.13.5, ninja 1.13.2.
Environment freeze is complete: `uv pip freeze` records all 64 packages.

## 7. 3×3 core workload results (ExLlamaV3)

Official mode, 3 warmups + 10 measured runs, medians. All 117 records:
`(SUCCESS, PASS)`, `performance_valid = true`.

| Workload | in→out | TTFT ms | prefill ms | prefill tok/s | decode ms | decode tok/s | TPOT ms (=ITL mean) | ITL med | ITL p95 | ITL p99 | ITL max | E2E ms | peak VRAM MB | RAM MB | CPU % | GPU util % | power W | temp °C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SS | 128→64 | 69.53 | 60.84 | 2106 | 573.23 | 109.89 | 9.11 | 7.89 | 18.11 | 20.69 | 21.46 | 645.87 | 2799 | 1721 | 171.60 | 73.33 | 116.86 | 50 |
| SM | 128→256 | 69.01 | 60.44 | 2118 | 2983.21 | 88.55 | 11.70 | 7.94 | 26.33 | 69.99 | 81.46 | 3164.16 | 2799 | 1725 | 162.24 | 64.71 | 113.30 | 58 |
| SL | 128→1024 | 69.88 | 60.92 | 2101 | 10606.34 | 96.85 | 10.37 | 7.98 | 21.46 | 50.16 | 77.98 | 10721.96 | 2799 | 1728 | 170.14 | 71.31 | 123.84 | 67 |
| MS | 1024→64 | 236.34 | 226.73 | 4517 | 576.12 | 110.51 | 9.14 | 8.00 | 18.80 | 21.23 | 21.48 | 817.85 | 2951 | 1969 | 157.38 | 78.50 | 136.39 | 65 |
| MM | 1024→256 | 237.13 | 227.10 | 4513 | 2476.86 | 102.27 | 9.87 | 8.01 | 20.52 | 30.96 | 47.77 | 2777.56 | 2951 | 1972 | 167.41 | 77.22 | 124.15 | 67 |
| ML | 1024→1024 | 237.49 | 227.11 | 4509 | 11765.42 | 97.35 | 11.50 | 8.11 | 27.99 | 58.61 | 87.23 | 13259.04 | 2951 | 1984 | 169.53 | 72.61 | 127.16 | 68 |
| LS | 4096→64 | 880.25 | 868.91 | 4717 | 631.20 | 100.11 | 10.02 | 8.88 | 19.11 | 22.88 | 23.87 | 1513.18 | 3031 | 2002 | 136.58 | 88.27 | 152.16 | 69 |
| LM | 4096→256 | 878.99 | 868.70 | 4715 | 2888.27 | 98.75 | 11.33 | 8.91 | 22.64 | 37.02 | 41.14 | 4072.27 | 3031 | 2002 | 157.89 | 83.53 | 136.83 | 70 |
| LL | 4096→1024 | 875.30 | 864.25 | 4739 | 12891.11 | 79.19 | 12.96 | 8.98 | 29.01 | 69.09 | 98.99 | 14481.06 | 3031 | 2005 | 161.80 | 69.19 | 123.72 | 68 |

All nine workloads produced exactly the requested output token count
(`finish_reason = max_new_tokens` in every measured run), and every run was
`correctness_status: PASS`. Every value above is a median over 10 measured runs
in session `20261002-090349_exllamav3_official_537aee`; warmups are excluded.

## 8. TTFT

TTFT is `MEASURED_EXTERNAL` (client timestamp of first output token minus
submission) and is dominated by prefill because queue latency is negligible
(`queue_ms` is runtime-reported and sub-millisecond on a single-request run).
TTFT is close to linear in prompt tokens with a large fixed component:

| prompt tokens | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|
| TTFT ms | 68.81 | 132.23 | 237.53 | 414.28 | 877.54 | 2021.67 | 4930.08 | FAILED |
| prefill ms | 60.26 | 123.05 | 227.89 | 403.96 | 866.80 | 2009.60 | 4915.39 | — |

(medians of 10 measured runs per point, session `20261002-095216_exllamav3_official_c72190`;
the 32768 point is covered by the supplementary run, §16–22.)

## 9. Prefill

`prefill_ms`/`prefill_tok_s` are `REPORTED_RUNTIME` (ExLlamaV3's own
`time_prefill`), never a framework "TPS" substituted into the benchmark formula.
Every value below is measured, not interpolated:

| prompt tokens | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| prefill tok/s (context suite) | 2124 | 4161 | 4494 | 5070 | 4725 | 4076 | 3333 |
| prefill tok/s (3×3 matrix) | 2101–2118 | — | 4509–4517 | — | 4715–4739 | — | — |

Prefill efficiency **rises then falls**: 128-token prompts reach only ~2.1 k tok/s
because a single small forward pass is dominated by fixed launch cost, peaking at
~5.1 k tok/s around 2048 tokens and then declining to 3.3 k tok/s at 16 K as
attention cost grows. The practical consequence is that this runtime is *worst*
at exactly the short-prompt case an interactive product hits most often.

## 10. Decode

Decode is `DERIVED` from externally measured `decode_ms` using the benchmark's
sustained-decode definition. It is flat up to ~2 K of context and then decays:

| context at decode (output 256) | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|
| decode tok/s | 101.0 | 103.5 | 103.0 | 101.0 | 88.1 | 87.7 | 71.7 |

In the 3×3 matrix (output 64/256/1024) decode spans **79.2–110.5 tok/s**, with the
best case at short context and short output. Beyond ~4 K of context decode falls
to **~72 tok/s at 16 K — a ~30 % loss**, which is attention-side cost, not a
compute ceiling (the GPU is only 69–92 % utilised while doing it).

## 11. TPOT and ITL

TPOT (`DERIVED`) is the mean inter-token latency. The **median** step time and
the **tail** tell very different stories:

| Workload | ITL median | TPOT (mean) | ITL p95 | ITL p99 | ITL max | p99 / median |
|---|---|---|---|---|---|---|
| SS | 7.89 | 9.11 | 18.11 | 20.69 | 21.46 | 2.6× |
| SM | 7.94 | 11.70 | 26.33 | 69.99 | 81.46 | 8.8× |
| SL | 7.98 | 10.37 | 21.46 | 50.16 | 77.98 | 6.3× |
| MS | 8.00 | 9.14 | 18.80 | 21.23 | 21.48 | 2.7× |
| MM | 8.01 | 9.87 | 20.52 | 30.96 | 47.77 | 3.9× |
| ML | 8.11 | 11.50 | 27.99 | 58.61 | 87.23 | 7.2× |
| LS | 8.88 | 10.02 | 19.11 | 22.88 | 23.87 | 2.6× |
| LM | 8.91 | 11.33 | 22.64 | 37.02 | 41.14 | 4.2× |
| LL | 8.98 | 12.96 | 29.01 | 69.09 | 98.99 | 7.7× |

This is the sharpest single result in the campaign. **The median decode step is
essentially context-insensitive — 7.89 ms at SS versus 8.98 ms at LL, a 14 %
change — while the tail explodes: p99 rises from 20.69 ms to 69.09 ms (3.3×) and
the worst observed step from 21.46 ms to 98.99 ms (4.6×).** Decode *throughput*
degrades with context (109.9 → 79.2 tok/s) almost entirely because of these
outlier steps, not because the typical step got slower.

`ITL mean/median/p95/p99/max` are retained per run in the raw records.

## 12. End-to-end latency

| Workload | E2E ms | of which prefill | of which decode |
|---|---|---|---|
| SS | 645.87 | 60.78 (9 %) | 593 (91 %) |
| MM | 2777.56 | 227.78 (8 %) | 2571 (92 %) |
| LL | 14481.06 | 864.88 (6 %) | 13016 (94 %) |

For every workload with meaningful output length, **the output phase dominates
end-to-end time (91–94 %)**. Prefill reduction cannot move these numbers much;
decode rate can.

## 13. VRAM

| State | VRAM |
|---|---|
| Idle before load | 217–300 MiB |
| Weights loaded (measured 4.0127 bpw) | ~1140 MiB |
| KV cache allocated (32768 tokens, fp16) | 1344 MiB |
| Peak during SS/SM/SL | 2799 MiB |
| Peak during MS/MM/ML | 2951 MiB |
| Peak during LS/LM/LL | 3031 MiB |
| After unload | back to ~217 MiB |

Peak VRAM is **under 3 GB of 12 GB** — the board is 75 % idle in capacity terms
even at LL, which is why per-request latency rather than memory is the binding
constraint. Runtime-reported extras: `torch_peak_reserved` 2736 MB, KV 1344 MB,
weights 1139.5 MB at 4.0127 bpw.

## 14. RAM and CPU

Host RAM peak 1718–2005 MB for the benchmark process. CPU utilisation 130–175 %
of a single core — i.e. the 2-physical-core host is substantially occupied by
feeding a GPU that is itself only 65–92 % busy. Given that the CPU has 2 cores
and no AVX-512, **host-side overhead is a first-class cost on this machine**, not
a rounding error.

## 15. GPU utilisation and power

GPU utilisation 65–92 % (highest at LS), board power 113–152 W against a 170 W
limit — **no power throttling**, and clocks held 1897–1950 MHz throughout.
Temperature 50–73 °C, far from thermal limits. Sessions were therefore not
thermally or power limited, and the GPU was never the bottleneck.

## 16–22. Concurrency, context, prefix, scheduler, batching, startup, soak

Full derived tables: [`../campaigns/rtx3060-minicpm5-2b-exl3-b0/exllamav3/derived-tables.md`](../campaigns/rtx3060-minicpm5-2b-exl3-b0/exllamav3/derived-tables.md).

### 16. Workload / concurrency coverage

All nine profiles ran at C1, C2, C4 and C8 with genuinely overlapping requests
released from a shared start gate (never sequential submission): **1755 raw
records, 0 failures**, session `20261002-095935_exllamav3_official_e107da`.
Median of 10 measured runs per cell.

### 17. Concurrency scaling (C1 / C2 / C4 / C8)

Aggregate output throughput (tok/s) — from actual concurrent group wall time:

| Profile | C1 | C2 | C4 | C8 | C1→C8 factor | efficiency vs ideal 8× |
|---|---|---|---|---|---|---|
| SS | 99.28 | 185.03 | 330.66 | 433.68 | 4.37× | 54.6% |
| SM | 97.13 | 174.59 | 315.01 | 545.13 | 5.61× | 70.2% |
| SL | 90.18 | 153.99 | 274.73 | 439.49 | 4.87× | 60.9% |
| MS | 78.07 | 136.83 | 223.41 | 365.67 | 4.68× | 58.5% |
| MM | 90.56 | 151.45 | 181.24 | 442.78 | 4.89× | 61.1% |
| ML | 77.06 | 131.76 | 232.25 | 452.87 | 5.88× | 73.5% |
| LS | 42.20 | 76.64 | 134.70 | 198.09 | 4.69× | 58.7% |
| LM | 70.00 | 113.58 | 202.18 | 280.40 | 4.01× | 50.1% |
| LL | 63.09 | 116.70 | 236.77 | 310.28 | 4.92× | 61.5% |

Per-request decode throughput (tok/s) and its retention from C1 to C8:

| Profile | C1 | C2 | C4 | C8 | C1→C8 retention |
|---|---|---|---|---|---|
| SS | 109.51 | 103.41 | 91.99 | 73.15 | 66.8% |
| SM | 99.37 | 89.15 | 81.36 | 73.64 | 74.1% |
| SL | 90.65 | 77.34 | 69.06 | 55.67 | 61.4% |
| MS | 108.19 | 90.80 | 74.32 | 61.00 | 56.4% |
| MM | 98.56 | 82.85 | 48.67 | 59.59 | 60.5% |
| ML | 78.39 | 66.88 | 58.93 | 57.75 | 73.7% |
| LS | 99.04 | 80.69 | 63.72 | 39.94 | 40.3% |
| LM | 91.91 | 70.36 | 61.25 | 40.26 | 43.8% |
| LL | 66.64 | 61.39 | 62.41 | 40.31 | 60.5% |

TTFT, TPOT, ITL p99, E2E, VRAM, utilisation and power by level:

| Profile | C | TTFT ms | TPOT ms | ITL p99 ms | E2E ms | peak VRAM MB | GPU util % | power W |
|---|---|---|---|---|---|---|---|---|
| SS | 1 | 69.44 | 9.13 | 22.04 | 644.69 | 2799 | 71.94 | 123.74 |
| SS | 2 | 72.15 | 9.67 | 23.34 | 683.44 | 2815 | 69.19 | 122.84 |
| SS | 4 | 92.77 | 10.87 | 22.83 | 763.63 | 2857 | 60.79 | 115.70 |
| SS | 8 | 282.30 | 13.72 | 51.00 | 1142.34 | 2946 | 44.90 | 65.76 |
| MM | 1 | 237.47 | 10.18 | 35.65 | 2835.15 | 3123 | 72.53 | 125.57 |
| MM | 2 | 241.07 | 12.09 | 43.19 | 3380.73 | 3123 | 66.67 | 112.60 |
| MM | 4 | 258.55 | 20.56 | 93.07 | 5634.64 | 3123 | 51.32 | 97.05 |
| MM | 8 | 316.19 | 16.78 | 38.86 | 4601.13 | 3123 | 80.28 | 110.78 |
| LL | 1 | 876.97 | 15.04 | 83.47 | 16259.41 | 3203 | 58.76 | 109.17 |
| LL | 2 | 881.35 | 16.30 | 86.00 | 17557.53 | 3203 | 64.89 | 115.70 |
| LL | 4 | 889.56 | 16.02 | 40.27 | 17285.62 | 3203 | 86.78 | 133.15 |
| LL | 8 | 956.87 | 24.81 | 34.99 | 26357.43 | 3203 | 91.34 | 119.87 |

**Concurrency is not free and it is not linear.** At C8 the engine reaches only
**50–74 % of ideal 8× scaling**, while per-request decode falls to **40–74 %** of
its C1 rate. TTFT inflates sharply — for SS it quadruples (69 ms → 282 ms) —
because requests now queue behind each other rather than because prefill got
slower. Peak VRAM grows only modestly (2799 → 2946 MB for SS) and is flat at
3203 MB for the long-input profiles: **memory is not what caps concurrency here.**

The most diagnostic cell in the whole campaign is **SS at C8**: GPU utilisation
drops to **44.9 %** and board power falls from ~124 W to **65.8 W**, i.e. the GPU
is idle more than half the time, while aggregate throughput reaches only 4.37× of
C1. A GPU that is starved while eight small requests are in flight points at the
**host side** (2 physical cores doing tokenisation, sampling and per-step
orchestration), not at the kernels.

At C8 the single-request *typical* step survives well (ITL p99 falls at high
concurrency for the long profiles, e.g. ML 79.4 → 36.2 ms), which is consistent
with the engine interleaving more work per step and amortising its own stalls —
but the aggregate is still host-limited.

**Measured anomaly, reported rather than smoothed:** MM at C4 (aggregate 181.24
tok/s, GPU 51.32 %, ITL p99 93.07 ms) sits *below* its own C2 and C8 points.
The values are as measured; the cause was not investigated in B0.

### 18. Context scaling

Official session `20261002-095216_exllamav3_official_c72190`: 8 lengths ×
(3 warmups + 10 measured). **7 of 8 points are valid** (all `performance_valid`,
all correctness PASS). `CTX-32768` is FAILED — see §23.

| input tokens | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
|---|---|---|---|---|---|---|---|---|
| TTFT ms | 68.81 | 132.23 | 237.53 | 414.28 | 877.54 | 2021.67 | 4930.08 | FAILED |
| prefill ms | 60.26 | 123.05 | 227.89 | 403.96 | 866.80 | 2009.60 | 4915.39 | — |
| prefill tok/s | 2124 | 4161 | 4494 | 5070 | 4725 | 4076 | 3333 | — |
| decode tok/s | 101.0 | 103.5 | 103.0 | 101.0 | 88.1 | 87.7 | 71.7 | — |
| peak VRAM MB | 2799 | 2911 | 2951 | 3031 | 3031 | 3031 | 3031 | — |
| GPU util % | 73.0 | 75.7 | 76.1 | 77.9 | 75.4 | 88.1 | 92.1 | — |
| power W | 118.8 | 125.6 | 130.1 | 136.8 | 127.6 | 149.7 | 157.6 | — |
| temp °C | 57 | 63.5 | 67 | 69 | 69 | 70.5 | 73 | — |

TTFT scales worse than linearly beyond ~4 K (4096 → 16384 is 4× the tokens for
5.6× the TTFT), decode decays ~30 % by 16 K, and VRAM saturates at 3031 MB from
2048 tokens onward because the full 32 K KV pool is allocated up front rather
than on demand. Failed points are never interpolated in the plot data.

## 23. Failed / unsupported cases

| Runtime / suite | Case | Result | Cause |
|---|---|---|---|
| ExLlamaV3 context | `CTX-32768` (32768 in / 256 out) | **FAILED** (13/13 runs) | `Job requires 130 pages (only 128 available)` — 32768 + 256 = 33024 tokens exceeds the 32768-token KV pool. Model supports 131072, so this is KV-pool sizing at the chosen configuration, **not** a model limit and not OOM. Preserved in `failures.jsonl`. |
| SGLang 0.5.20 + sglang_exl3 | install | **INSTALL_FAILURE** | CUDA 13 stack vs driver 550 (CUDA 12.4) |
| vLLM 0.30.0 + vllm-exl3 | install | **INSTALL_FAILURE** | CUDA 13 stack vs driver 550 (CUDA 12.4) |
| TensorFold 0.6.1 | model + startup | **UNSUPPORTED_MODEL** | no MiniCPM family at the pinned revision; independently refuses sm_86 (`MIN_CAPABILITY = (8,9)`) |

Failures are never silently discarded: each is written to the session's
`failures.jsonl` with runtime, profile, concurrency, input/output size, timestamp,
error text and status. No retry was made for the capacity failure, because it is
not an infrastructure fault — the official 32K-pool result stands as measured,
and the point is additionally covered by the labelled supplementary run
(§24).

## 24. Architectural observations

Each item is separated into **MEASURED FACT** and **INTERPRETATION**. No
optimization was implemented in this campaign.

1. **Fixed overhead dominates short requests.**
   **MEASURED:** TTFT is 68.8 ms at 128 input tokens vs 237.6 ms at 1024 — the
   first 128 tokens cost ~69 ms (~540 µs/token), while the marginal cost from 128
   to 1024 tokens is ~169 ms for 896 tokens (~189 µs/token). Prefill throughput
   rises 2106 → 4514 tok/s.
   **INTERPRETATION:** ~60 ms per request is paid before useful work scales.
   For an interactive product at short prompts this fixed cost is the single
   largest lever, and it is unlikely to be GPU compute — a 2.5 B model at 4 bpw
   should not need 60 ms for 128 tokens.

2. **Decode is flat and shallow up to 4 K, then decays.**
   **MEASURED:** 109.9 tok/s at SS → 100.1 at LS (4096) → 71.8 at 16 K.
   **INTERPRETATION:** no severe KV-locality cliff in the 1–4 K band; the decay
   past 8 K is attention-cost growth, which is where a Pradium landing-zone design
   would have to earn its keep.

3. **Decode tail latency, not average step time, is the defect.**
   **MEASURED:** across all nine official workloads the ITL *median* stays within
   7.89–8.98 ms (14 % spread) while ITL p99 moves 20.69 → 69.09 ms and the worst
   step 21.46 → 98.99 ms. Decode throughput falls 109.9 → 79.2 tok/s, but the
   typical step barely changes.
   **INTERPRETATION:** the throughput loss at long output/context is produced by a
   small number of very slow steps, not by uniformly slower decoding. Optimising
   average step cost would therefore buy very little; finding and removing the
   stall (a scheduler or synchronisation event, not arithmetic) is where the
   user-visible 3–5× tail sits. Any future comparison should report p99, not TPS.

4. **The host CPU, not the GPU, is the first saturation point.**
   **MEASURED:** CPU 130–175 % with 2 physical cores while GPU sits at 65–92 %
   and 113–152 W of a 170 W limit, at ≤3 GB of 12 GB VRAM.
   **INTERPRETATION:** on this class of host, reducing per-step host work (fewer
   synchronisations, fewer kernel launches per token, cheaper sampling loop) is
   worth more than kernel-level arithmetic optimization. Any future comparison
   must hold CPU work constant or it will measure the CPU, not the engine.

5. **The KV pool, not the GPU, limits the largest workloads.**
   **MEASURED:** 32 K pool = 128 pages of 256 tokens; a 32768-token prompt with
   256 output tokens needs 130 pages and is refused outright rather than
   swapping or degrading.
   **INTERPRETATION:** capacity is enforced by hard refusal. Any concurrency or
   context capability claim must be stated together with the pool size, because
   the same GPU can serve the same model at a different pool size and produce a
   different pass/fail boundary.

6. **Prefix reuse is automatic and page-granular.**
   **MEASURED:** 256-token pages with a chained content-hash index; a 9000-token
   prompt leaves 8960 cached tokens (35 pages), and a repeated identical prompt is
   served from cache (`alloc_cached_pages > 0`). A rebuild of the engine's page
   table resets this verifiably (`cached_tokens == 0`, `alloc_cached_pages == 0`).
   **INTERPRETATION:** reuse granularity (256 tokens) sets the floor on how much
   of a shared system prefix can be reused; system prompts shorter than the page
   size gain nothing.

7. **The runtime engine cannot be driven from multiple threads.**
   **MEASURED:** `Generator.iterate()` is a single-threaded driver; concurrency
   only exists inside one scheduler loop.
   **INTERPRETATION:** concurrency architecture is a property of the engine's API
   shape, and an adapter that drove it from N threads would have serialized
   silently while *appearing* concurrent. Architecture decisions for Pradium
   should treat "one driver loop, N in-flight sequences" as the real model.

8. **Runtime configuration is not optional to record.**
   **MEASURED:** a single KV-pool parameter (32768 vs 49152) moves the boundary
   between PASS and FAILED for two frozen workloads.
   **INTERPRETATION:** every future cross-runtime number must carry its pool size,
   batch width and cache dtype, or it is not comparable.

9. **Concurrency is host-limited, and the GPU says so.**
   **MEASURED:** at C8 on the SS profile, GPU utilisation falls to 44.9 % and board
   power to 65.8 W (from ~124 W at C1) while aggregate throughput reaches only
   4.37× of C1 and per-request decode drops to 66.8 %. Across all profiles C8
   delivers 50–74 % of ideal 8× scaling. Peak VRAM barely moves (2799 → 2946 MB
   on SS).
   **INTERPRETATION:** the limiter at high concurrency is not VRAM and not the
   kernels — it is per-step host work on 2 physical cores. This is the single most
   actionable result for Pradium: on this hardware class, a scheduler that reduces
   *host* work per decode step (fewer synchronisations, fewer launches, cheaper
   sampling, less Python in the step path) converts directly into aggregate
   throughput, whereas kernel-level arithmetic work has little headroom to win.

10. **Aggregate and per-request throughput must always be reported together.**
    **MEASURED:** C8 raises aggregate throughput 4.0–5.9× while lowering each
    request's rate to 40–74 % of C1, and TTFT inflates (SS: 69 → 282 ms).
    **INTERPRETATION:** a throughput-only headline hides a real user-visible
    regression. Any future Pradium concurrency claim should state both axes plus
    TTFT inflation, since the two move in opposite directions.

## 25. Exact reproduction instructions

```bash
# --- VM, unprivileged container, RTX 3060 12 GB, driver 550.144.03 ---
mkdir -p /workspace/{pradium,models,runtimes,envs,logs}

# 1. freeze the model revision FIRST, then download that exact revision
git ls-remote https://huggingface.co/ewin-reg/MiniCPM5-2B-EXL3-Quantized HEAD
# -> e36fbb568f466739d7c244a5100d01acf93d3b7f
MODEL_REVISION=e36fbb568f466739d7c244a5100d01acf93d3b7f
uv venv --python 3.12 /workspace/envs/tools && source /workspace/envs/tools/bin/activate
uv pip install huggingface_hub hf_transfer
export HF_HOME=/workspace/.hf_home
hf download ewin-reg/MiniCPM5-2B-EXL3-Quantized \
  --revision "$MODEL_REVISION" --local-dir /workspace/models/minicpm5-2b-exl3

# verify the checkpoint matches the campaign hashes
sha256sum /workspace/models/minicpm5-2b-exl3/model.safetensors
# 4a310a644144c54f8146c3dbf58fcedc5ab97e8198d21e241c09fcfad5b6b2be

# 2. Pradium + campaign branch
git clone https://github.com/Pranavharshans/Pradium-Engine.git /workspace/pradium
cd /workspace/pradium && git checkout bench/exl3-runtime-b0-rtx3060

# 3. ExLlamaV3 1.5.3 in its own environment (pin matches the wheel tag exactly)
git clone https://github.com/turboderp-org/exllamav3.git /workspace/runtimes/exllamav3
cd /workspace/runtimes/exllamav3 && git checkout d3739fd && git rev-parse HEAD
# d3739fd393337b1ff4d6c2a342b12f0c87a9592f
uv venv --python 3.12 /workspace/envs/exllamav3 && source /workspace/envs/exllamav3/bin/activate
export UV_CACHE_DIR=/workspace/.uvcache TORCH_CUDA_ARCH_LIST=8.6
uv pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.3/exllamav3-1.5.3%2Bcu128.torch2.10.0-cp312-cp312-linux_x86_64.whl"
uv pip install transformers pytest

# 4. frozen inputs, materialized once with the exact model tokenizer
python -m benchmarks corpus validate
python -m benchmarks corpus materialize \
  --tokenizer "hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@${MODEL_REVISION}"

# 5. adapter conformance must pass before any number is trusted
export PYTHONPATH=/workspace/pradium
export PRADIUM_EXLLAMAV3_MODEL=/workspace/models/minicpm5-2b-exl3
python -m pytest benchmarks/tests/test_adapter_exllamav3.py -q      # 15 passed

# 6. suites (3 warmups + 10 measured; never --quick for official figures)
CAMPAIGN=/workspace/pradium/benchmarks/campaigns/rtx3060-minicpm5-exl3-b0
RC='{"model_dir":"/workspace/models/minicpm5-2b-exl3","max_num_tokens":32768,"max_batch_size":16,"max_chunk_size":2048}'
for suite in matrix context concurrency prefix-cache scheduler mixed batching startup; do
  python -m benchmarks run "$suite" --runtime exllamav3 \
    --tokenizer "hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@${MODEL_REVISION}" \
    --model ewin-reg/MiniCPM5-2B-EXL3-Quantized --model-revision "$MODEL_REVISION" \
    --quantization EXL3 --bpw 4.0 \
    --results-dir "$CAMPAIGN/sessions" --runtime-config "$RC"
done
python -m benchmarks run soak --runtime exllamav3 ... --duration 30
```

Runtime environments were kept strictly separate (`/workspace/envs/exllamav3`,
`/workspace/envs/sglang-exl3`…), no framework was installed into another's
environment, and the model directory was never duplicated, symlinked or
re-quantized.

### Supplementary configuration

Two frozen workloads do not fit a 32768-token KV pool and are refused by the
runtime: `CTX-32768` (needs 130 of 128 pages) and concurrency `C8` on
`LM`/`LL` (144 / 168 pages). Those results are kept as FAILED in the official
sessions, and are additionally covered by a **labelled supplementary run** at
`max_num_tokens=49152` (192 pages) whose sessions record that pool size in their
`runtime_config`, so the official 32 K configuration is never silently mixed with
the supplementary one.
