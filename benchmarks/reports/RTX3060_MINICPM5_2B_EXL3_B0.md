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
| SS 128→64 | 69.53 ms | 2104 | 109.80 | 9.11 ms | 2799 MB |
| MM 1024→256 | 237.13 ms | 4509 | 101.31 | 9.87 ms | 2951 MB |
| LL 4096→1024 | 875.30 ms | 4738 | 77.19 | 12.96 ms | 3031 MB |

The decisive structural findings for Pradium:

* **Decode is 97–112 tok/s and essentially flat to 2 K of context, falling to
  58.8 tok/s at 32 K** — and the fall is *tail*, not typical: the ITL median stays
  within 7.89–8.98 ms across all nine official workloads while ITL p99 moves
  20.69 → 69.09 ms.
* **The GPU is not the constraint until 32 K.** Up to 16 K the board sits at
  60–92 % utilisation and 97–158 W of a 170 W limit with ≤3 GB of 12 GB VRAM in
  use; only at 32 K does it reach 97.8 % and 165.6 W.
* **The 2-core host is the limiter for concurrency**: at C8 on short prompts the
  GPU drops to 44.9 % utilisation and 65.8 W while aggregate throughput reaches
  only 4.37× of C1.
* **One long prefill freezes every concurrent stream**: resident inter-token
  latency goes 14.62 → 329.95 ms with a 499.69 ms worst-case stall.

Trustworthiness and reproducibility were treated as more important than producing
a full grid: where a stack could not genuinely run the exact checkpoint, that is
recorded as a result rather than engineered around.

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
| SS | 128→64 | 69.53 | 60.84 | 2104 | 573.23 | 109.80 | 9.11 | 7.89 | 18.11 | 20.69 | 21.46 | 644.73 | 2799 | 1721 | 171.60 | 73.17 | 116.27 | 50 |
| SM | 128→256 | 69.01 | 60.44 | 2118 | 2983.21 | 85.58 | 11.70 | 7.94 | 26.33 | 69.99 | 81.46 | 3056.44 | 2799 | 1725 | 162.24 | 59.87 | 111.14 | 58 |
| SL | 128→1024 | 69.88 | 60.92 | 2101 | 10606.34 | 96.45 | 10.37 | 7.98 | 21.46 | 50.16 | 77.98 | 10676.98 | 2799 | 1728 | 170.14 | 71.07 | 123.54 | 67 |
| MS | 1024→64 | 236.34 | 226.73 | 4516 | 576.12 | 109.36 | 9.14 | 8.00 | 18.80 | 21.23 | 21.48 | 811.57 | 2951 | 1969 | 157.38 | 78.38 | 136.37 | 65 |
| MM | 1024→256 | 237.13 | 227.10 | 4509 | 2476.86 | 101.31 | 9.87 | 8.01 | 20.52 | 30.96 | 47.77 | 2754.25 | 2951 | 1972 | 167.41 | 76.17 | 120.48 | 67 |
| ML | 1024→1024 | 237.49 | 227.11 | 4509 | 11765.42 | 87.95 | 11.50 | 8.11 | 27.99 | 58.61 | 87.23 | 12002.67 | 2951 | 1984 | 169.53 | 66.08 | 119.81 | 68 |
| LS | 4096→64 | 880.25 | 868.91 | 4714 | 631.20 | 99.81 | 10.02 | 8.88 | 19.11 | 22.88 | 23.87 | 1511.31 | 3031 | 2002 | 136.58 | 88.17 | 152.10 | 69 |
| LM | 4096→256 | 878.99 | 868.70 | 4715 | 2888.27 | 89.29 | 11.33 | 8.91 | 22.64 | 37.02 | 41.14 | 3767.44 | 3031 | 2002 | 157.89 | 78.76 | 135.54 | 70 |
| LL | 4096→1024 | 875.30 | 864.25 | 4738 | 12891.11 | 77.19 | 12.96 | 8.98 | 29.01 | 69.09 | 98.99 | 14136.55 | 3031 | 2005 | 161.80 | 67.15 | 120.66 | 68 |

All nine workloads produced exactly the requested output token count
(`finish_reason = max_new_tokens` in every measured run), and every run was
`correctness_status: PASS`. Every value above is a median over 10 measured runs
in session `20261002-090349_exllamav3_official_537aee`; warmups are excluded.
Prefill ms and RAM are medians of the same runs; `prefill_tok_s` and
`prefill_ms` are runtime-reported, everything else is externally measured or
derived by the harness (see the provenance note beneath §7's tables in
[`derived-tables.md`](../campaigns/rtx3060-minicpm5-exl3-b0/exllamav3/derived-tables.md),
which is generated directly from the committed records).

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

| input tokens | 128 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 † |
|---|---|---|---|---|---|---|---|---|
| TTFT ms | 68.81 | 132.23 | 237.53 | 414.28 | 877.54 | 2021.67 | 4930.08 | 13445.14 † |
| prefill ms | 60.26 | 123.05 | 227.89 | 403.96 | 866.80 | 2009.60 | 4915.39 | 13417.58 † |
| prefill tok/s | 2124 | 4161 | 4494 | 5070 | 4725 | 4076 | 3333 | 2442 † |
| decode tok/s | 101.0 | 103.5 | 103.0 | 101.0 | 88.1 | 87.7 | 71.7 | 58.8 † |
| peak VRAM MB | 2799 | 2911 | 2951 | 3031 | 3031 | 3031 | 3031 | 3659 † |
| GPU util % | 73.0 | 75.7 | 76.1 | 77.9 | 75.4 | 88.1 | 92.1 | **97.8 †** |
| power W | 118.8 | 125.6 | 130.1 | 136.8 | 127.6 | 149.7 | 157.6 | **165.6 †** |
| temp °C | 57 | 63.5 | 67 | 69 | 69 | 70.5 | 73 | 74.5 † |

† The 32768 point does not fit the official 32768-token KV pool (it needs 130
pages of 128), so it comes from the **labelled supplementary run** at a
49152-token pool — see §25. It is a different runtime configuration and is marked
as such everywhere it appears; the official row is the FAILED one preserved in
§23. It is not mixed into the official medians.

TTFT scales worse than linearly beyond ~4 K (4096 → 16384 is 4× the tokens for
5.6× the TTFT; 16384 → 32768 is 2× the tokens for 2.7× the TTFT), and decode
decays from ~101 tok/s at ≤2 K to **58.8 tok/s at 32 K**. VRAM saturates at
3031 MB from 2048 tokens onward under the official pool because the full 32 K KV
pool is allocated up front rather than on demand.

**The 32 K point is also the only one where the GPU becomes the constraint**: GPU
utilisation reaches **97.8 %** and board power **165.6 W** against a 170 W limit,
versus 60–92 % and 97–158 W everywhere else. Up to 16 K this workload is
host-limited; at 32 K it finally saturates the SM. Failed points are never
interpolated in the plot data.

### 19. Prefix / KV reuse

Session `20261002-110241_exllamav3_official_7cf380`. Frozen methodology:
1024-token inputs, 256-token outputs, 4 requests per ratio, scenarios
`same_session`, `cross_request` and `cross_session`, 10 measured runs per cell,
shared prefix defined on token IDs. Every prime is a full cold prefill (the
adapter's `clear_prefix_cache()` runs before each request), so the measured
request can only reuse what the prime actually deposited.

| reuse | scenario | TTFT ms | prefill ms | prefill tok/s | decode tok/s | E2E ms |
|---|---|---|---|---|---|---|
| 0 % | same_session | 237.8 | 228.3 | 4486 | 97.8 | 2845.3 |
| 0 % | cross_request | 237.6 | 227.9 | 4492 | 98.5 | 2827.2 |
| 0 % | cross_session | 237.6 | 228.0 | 4492 | 102.3 | 2735.0 |
| 25 % | same_session | 189.3 | 179.7 | 5699 | 104.3 | 2633.3 |
| 25 % | cross_request | 189.2 | 179.7 | 5698 | 99.2 | 2758.6 |
| 25 % | **cross_session** | **238.3** | **228.8** | **4475** | 102.3 | 2730.2 |
| 50 % | same_session | 138.1 | 128.7 | 7958 | 102.8 | 2617.3 |
| 50 % | cross_request | 138.3 | 128.8 | 7949 | 101.1 | 2660.2 |
| 50 % | **cross_session** | **238.0** | **228.3** | **4486** | 102.5 | 2725.9 |
| 75 % | same_session | 93.1 | 83.7 | 12228 | 102.2 | 2587.9 |
| 75 % | cross_request | 93.1 | 83.8 | 12216 | 97.0 | 2721.5 |
| 75 % | **cross_session** | **238.0** | **228.4** | **4484** | 97.1 | 2863.5 |
| 90 % | same_session | 66.7 | 57.2 | 17900 | 101.2 | 2587.2 |
| 90 % | cross_request | 66.8 | 57.2 | 17901 | 105.0 | 2504.0 |
| 90 % | **cross_session** | **238.3** | **228.8** | **4476** | 92.6 | 2992.3 |
| ~98.4 % | same_session | **21.9** | **12.3** | **82930** | 102.4 | 2517.1 |
| ~98.4 % | cross_request | **21.5** | **12.2** | **83683** | 101.5 | 2533.8 |
| ~98.4 % | **cross_session** | **237.8** | **228.2** | **4487** | 100.0 | 2787.8 |

Effective prefill speedup vs the no-reuse `cross_session` control:

| nominal reuse | 0 % | 25 % | 50 % | 75 % | 90 % | ~98.4 % |
|---|---|---|---|---|---|---|
| same_session prefill tok/s | 4486 | 5699 | 7958 | 12228 | 17900 | 82930 |
| speedup | 1.00× | 1.27× | 1.77× | 2.72× | 3.99× | **18.5×** |
| TTFT ms | 237.8 | 189.3 | 138.1 | 93.1 | 66.7 | **21.9** |

The `cross_session` column is the point of the whole table: it is the same
prompts at the same nominal reuse with the engine's page table rebuilt between
prime and measure, and it refuses to move — **4475–4487 tok/s and 237.6–238.3 ms
at every ratio**. Reuse is therefore genuine on this runtime and the benchmark's
session boundary is genuinely cold, which is what makes the other rows
trustworthy. Decode throughput is unaffected throughout (97–105 tok/s): prefix
reuse buys TTFT and prefill, not decode.

`prefix_reused_tokens` / `prefix_recomputed_tokens` are recorded as `UNAVAILABLE`
for ExLlamaV3 (v1.5.3 reports only aggregate page counters, not per-request
reused-token counts), so the reuse evidence here is the measured TTFT and prefill
throughput plus the control — never a framework-reported figure.

### 20. Scheduler interference

Session `20261002-122339_exllamav3_official_e418c8`. Frozen workload: 3 requests
actively decoding, then one large prefill (4096 tokens, LS profile) arrives.
Phase statistics are derived from the persisted per-token timing traces.

| Metric | Value |
|---|---|
| Background requests | 3 |
| Intruder | LS, 4096 input tokens |
| Intruder prefill | 898.10 ms |
| Intruder TTFT | 2236.04 ms (vs 880 ms isolated for LS → **2.5× worse**) |
| ITL during the prefill | **329.95 ms** (6 samples) |
| ITL after the prefill | **14.62 ms** (3063 samples) |
| **Max single-token stall** | **499.69 ms** |
| p95 ITL (whole run) | 28.41 ms |
| Background aggregate decode | 196.92 tok/s |

Per resident request, the picture is identical and unambiguous:

| request | during-prefill ITL | after-prefill ITL | max ITL | p95 ITL | decode tok/s | TTFT ms |
|---|---|---|---|---|---|---|
| 1 | 319.10 ms | 14.61 ms | 499.39 | 28.68 | 64.51 | 1588.70 |
| 2 | 499.69 ms | 14.63 ms | 499.69 | 28.14 | 66.20 | 1828.97 |
| 3 | 261.36 ms | 14.62 ms | 499.55 | 28.28 | 66.21 | 1831.54 |

**A single 4 K prefill in flight costs each resident decoder up to half a second
of stall — 22.6× its normal inter-token latency — and it slows the intruder itself
by 2.5×.** Recovery is complete and immediate (14.6 ms afterwards), so this is a
transient preemption stall, not degradation. `itl_before_ms` is `null` because the
intruder arrived before any resident request had emitted a token
(`phase_token_counts.before == 0`); that window is empty by construction and is
reported as unavailable rather than as zero.

This is the most decision-relevant result in the campaign for Pradium. On this
machine a large prefill does not merely run slowly itself — it **visibly freezes
every concurrent stream** for up to half a second at a time and multiplies their
inter-token latency by ~23×, while the intruder is itself delayed 2.5×. Because
recovery is complete, interleaving or chunking prefill is a pure win: it trades
some intruder TTFT for eliminating the resident-stream stall almost entirely.

### 20b. Mixed workloads

Session `20261002-122404_exllamav3_official_a96d32`, 260 records / 200 measured.
Frozen scenarios and their concurrent request mixes:

| scenario | concurrent requests | n | TTFT ms | decode tok/s | aggregate tok/s | TPOT ms | ITL p99 | E2E ms | peak VRAM MB | GPU util % |
|---|---|---|---|---|---|---|---|---|---|---|
| `interactive_burst` | 4×SS + 2×SM | 60 | 152.59 | 84.84 | 197.03 | 11.79 | 26.85 | 953.61 | 2898 | 60.17 |
| `agent_like` | 4×MS (repeated medium prefixes, small suffixes) | 40 | 259.24 | 76.82 | 235.60 | 13.02 | 25.05 | 1078.47 | 3059 | 72.55 |
| `mixed_generation` | MS + MM + ML together | 30 | 695.96 | 71.89 | 89.97 | 13.91 | 71.71 | 4187.06 | 3059 | 71.33 |
| `mixed_prompt_sizes` | SM + MM + LM (128/1024/4096 inputs) | 30 | 718.24 | 61.14 | 154.55 | 16.36 | 31.96 | 4946.06 | 3139 | 86.30 |
| `long_job_interference` | LL + 2×SS + SM | 40 | 551.70 | 51.76 | 84.52 | **19.35** | **94.81** | 3288.62 | 3139 | 81.39 |

`long_job_interference` is the worst case on every axis that matters to a user:
per-request decode falls to **51.76 tok/s**, TPOT rises to 19.35 ms, and ITL p99
reaches **94.81 ms** — roughly 3.8× the tail of `agent_like` (25.05 ms), for the
same hardware and model. One long job is enough to degrade every interactive
request sharing the GPU, which is the mixed-workload form of §20's finding.

### 21. Batching

Session `20261002-123327_exllamav3_official_50d6af`, frozen B1/B2/B4/B8 at the MM
profile. ExLlamaV3 has no separate static-batch API; the adapter's `generate_batch`
enqueues every sequence before any decode step, so the sequences advance in
lockstep, which is what this suite is defined to measure.

| batch | measured requests | aggregate output tok/s | E2E ms | scaling vs B1 |
|---|---|---|---|---|
| B1 | 10 | 75.13 | 3407.61 | 1.00× |
| B2 | 20 | 129.26 | 4043.08 | 1.72× |
| B4 | 40 | 247.10 | 4159.26 | 3.29× |
| B8 | 80 | 424.24 | 4828.88 | **5.65×** |

Same shape as concurrency, from a different code path: **B8 delivers 5.65× of B1
(70.6 % of ideal 8×)** while per-batch E2E latency grows 1.42×. Batching therefore
buys aggregate throughput at the cost of individual request latency — the same
trade §17 measures from the concurrency side, which is a useful consistency check
because the two suites exercise the engine differently.

The batching suite's records carry group-aggregate accounting only, so per-request
TTFT/TPOT are reported `UNAVAILABLE` there; they are not silently estimated.

### 22. Startup

Session `20261002-123701_exllamav3_official_46dd9a`.

| Milestone | Value |
|---|---|
| Adapter runtime init | 0.09 ms |
| CUDA init | 0.09 ms |
| Model load (weights → VRAM) | **1561.78 ms** |
| Runner start → ready | 1561.88 ms |
| **Cold first request TTFT** | **1574.01 ms** |
| Start → first token | 3211.63 ms |
| **Warm request TTFT (median of 5)** | **11.27 ms** |
| CUDA graph capture | not applicable (v1.5.3 has no CUDA-graph path) |
| Weight mapping / server ready | not reported by this runtime |

The one number to take from this is the **cold-to-warm ratio: 1574.01 ms → 11.27 ms
(140×)**. The first request ever seen by a freshly loaded model pays ~1.57 s before
its first token, and everything after it is ~11 ms.

**Measurement caveat, stated rather than hidden.** The harness's CLI loads the
model before dispatching any suite, so the startup suite's `initialize()` /
`load_model()` calls exercise the *re-load* path inside an already-warm process.
Two consequences: `cuda_init_ms` is ~0.09 ms because the CUDA context already
existed, and the suite's `idle_vram_mb` (2711 MB) is **not** true idle — it was
sampled while the CLI's model was still resident. The genuine idle baselines are
the ones in §2 and §13 (217–300 MB VRAM, ~15 W). A true cold-process startup
measurement would need the runner to own process launch, which is a harness change
and therefore out of scope for a frozen benchmark.

### 22b. Soak / stability

Session `20261002-123734_exllamav3_official_d3f059`, frozen configuration:
30-minute duration, concurrency 1, profiles MM / SL / LL, 1000 ms telemetry poll.

| Metric | Result |
|---|---|
| Duration requested / actual | 1800 s / **1805.54 s** |
| Iterations | 170 |
| **Failed requests** | **0** |
| **VRAM growth (first vs last quartile)** | **+3.86 MB** (against 2949 MB resident) |
| RAM growth | +36.54 MB |
| TTFT change ratio | **−0.079** (TTFT improved ~7.9 %) |
| Decode throughput change ratio | **+0.037** (throughput improved ~3.7 %) |

Thirty minutes of continuous load produced **no failures, no VRAM leak, no RAM
leak and no degradation** — the first-vs-last-quartile comparison actually moves
slightly in the *favourable* direction for both latency and throughput, which is
what a stable, non-throttling system looks like (GPU temperature stayed 50–73 °C
and clocks 1897–1950 MHz throughout, so there was no thermal drift to confound it).
The 3.86 MB VRAM drift is <0.2 % of the model+KV footprint and is allocator noise,
not growth. **Stability is not a concern for this stack on this hardware.**

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

5. **The KV pool, not the GPU, limits the largest workloads — and pool size
   interacts with workload shape, not just prompt length.**
   **MEASURED:** a 32 K pool is 128 pages of 256 tokens; a 32768-token prompt with
   256 output tokens needs 130 pages and is refused outright rather than swapped
   or degraded. The same pool handled concurrency C8 on the 4096-token profiles
   with zero failures (1755/1755 records) — because the C slots send the
   *identical* prompt and therefore share its pages through the content-hash
   index. A 49152-token pool let `CTX-32768` through cleanly (7 of 8 context
   points under 32 K, the eighth only under the larger pool).
   **INTERPRETATION:** capacity is enforced by hard refusal, and whether a given
   concurrency level fits depends on how much of the concurrent prompts is
   *shared*, not only on their length. A concurrency suite built from distinct long
   prompts per slot would have exhausted the same pool that the identical-prompt
   suite sailed through. Any concurrency or context capability claim must therefore
   state its pool size *and* its prompt-sharing pattern, or it is not reproducible.

6. **Prefix reuse is automatic, and its reset is verifiable.**
   **MEASURED:** a 9000-token prompt leaves 8960 cached tokens (35 pages of 256),
   and a repeated identical prompt is served from cache (`alloc_cached_pages > 0`).
   Rebuilding the engine's page table resets this verifiably (`cached_tokens == 0`
   and `alloc_cached_pages == 0` on the next job), and the prefix suite's
   `cross_session` control independently confirms a real session boundary (see
   observation 11).
   **INTERPRETATION:** reuse requires no client-side action — a repeated prefix is
   picked up automatically — but the *observable* state is page counters, not
   per-request reused-token counts, so per-request reuse must be measured through
   TTFT and prefill time rather than read from the runtime.

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

11. **Prefix reuse is highly effective, and a session boundary really is one.**
    **MEASURED:** measured prefill throughput at 0 / 25 / 50 / 75 / 90 / ~98 %
    nominal reuse is **4486 / 5699 / 7958 / 12228 / 17900 / 82930 tok/s**
    (1.00× / 1.27× / 1.77× / 2.72× / 3.99× / **18.5×** versus no reuse), with TTFT
    falling 237.8 → 21.9 ms at the top of the range. The `cross_session` control —
    identical prompts with the engine's page table rebuilt between prime and
    measure — holds **1.00× (≈4480 tok/s) at every single ratio**.
    **INTERPRETATION:** the control is the important half of this measurement: it
    proves the reuse is genuine rather than an artefact of warm weights, and it
    proves the reset used for the session boundary is a real wipe. Intermediate
    speedups are below the ideal (0.75 reuse gives 2.72× rather than 4.0×, 0.5
    gives 1.77× rather than 2.0×) while the ~98 % case reaches 18.5× — so reuse is
    essentially continuous rather than quantised to whole 256-token pages, and the
    shortfall at intermediate ratios is consistent with a fixed per-request
    prefill overhead of roughly 10–25 ms sitting on top of the incremental work.
    That overhead, not cache granularity, is what caps the gain at moderate reuse.
    For Pradium this says the reusable-prefix path is worth having and the fixed
    prefill cost is worth attacking.

12. **A large prefill freezes every concurrent stream.**
    **MEASURED:** with three requests decoding, a single 4096-token prefill
    arriving pushes resident inter-token latency from **14.62 ms to 329.95 ms**
    (22.6×), with a **499.69 ms worst-case single-token stall**, and slows the
    intruder itself from 880 ms to 2236 ms TTFT (2.5×). Recovery afterwards is
    complete (14.62 ms) and the background aggregate still reaches 196.92 tok/s.
    **INTERPRETATION:** prefill on this engine is effectively uninterruptible at
    request granularity, so a single long prompt is a scheduler-wide stall event
    rather than a cost borne only by its own request. This is the strongest
    argument in the data for interleaved/chunked prefill, and it is a
    *scheduling* problem — not a kernel-arithmetic one, and not a VRAM one.

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

One frozen workload does not fit a 32768-token KV pool and is refused by the
runtime: `CTX-32768` needs 130 pages of the 128 available. That result is kept as
**FAILED** in the official context session (§23) and is additionally covered by a
**labelled supplementary run** at `max_num_tokens=49152` (192 pages), session
`20261002-130942_exllamav3_official_6eeced`, whose records carry that pool size in
their `runtime_config` so the official 32 K configuration can never be silently
mixed with the supplementary one.

| `CTX-32768` at 49152-token pool | value |
|---|---|
| records / measured runs | 13 / 10, **all SUCCESS + PASS**, `performance_valid` |
| TTFT | 13445.14 ms |
| prefill | 13417.58 ms → **2442 tok/s** |
| decode | **58.76 tok/s** |
| TPOT / ITL p95 | 17.02 ms / 17.65 ms |
| E2E | 17775.69 ms |
| peak VRAM | 3659 MB |
| GPU utilisation / power / temp | **97.8 %** / **165.58 W** / 74.5 °C |

So the point is attainable — the model supports 131072 positions — and the
official 32 K-pool failure was purely a sizing decision in the adapter
configuration, not a capability limit. No retry was applied to the official
session: rewriting a measured FAILED result because a *different* configuration
succeeds would be exactly the kind of selective re-run the campaign forbids.

The concurrency suite needed no supplement: all nine profiles passed at C8 under
the official 32768-token pool (1755/1755 records, 0 failures), because the C slots
share the identical prompt prefix through the content-hash page table. This is
worth recording as a finding in its own right — a concurrency workload that used
*distinct* long prompts per slot would have exhausted the same 32 K pool.
