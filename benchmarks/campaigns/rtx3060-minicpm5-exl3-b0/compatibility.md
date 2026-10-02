# PRADIUM EXL3 B0 — Runtime compatibility results

Campaign: `PRADIUM-EXL3-RTX3060-B0`
Benchmark: `PRADIUM-RUNTIME-BENCH-v2` (campaign spec label: `PRADIUM-RUNTIME-BENCH-v1`)
Model: `ewin-reg/MiniCPM5-2B-EXL3-Quantized` @ `e36fbb568f466739d7c244a5100d01acf93d3b7f`
GPU: NVIDIA GeForce RTX 3060 12 GB, Ampere sm_86, driver 550.144.03 (CUDA 12.4)

## Compatibility matrix

| Runtime stack | EXL3 route | MiniCPM5 EXL3 status | Benchmark status |
|---|---|---|---|
| ExLlamaV3 1.5.3 @ `d3739fd` | Native | `SUPPORTED_NATIVE` — 295/295 linear modules execute as EXL3, measured 4.0127 bpw layer / 6.0079 bpw head against the declared 4.0/6 | Full official suite run |
| SGLang 0.5.20 @ `94602c9` + `sglang_exl3` @ `1ace59c4` | Plugin (`sglang.srt.plugins` entry point) | Not reached — `INSTALL_FAILURE` | Not runnable on this VM |
| vLLM 0.30.0 @ `ced6857` + `vllm-exl3` @ `223e246f` | Plugin (`--quantization exl3`) | Not reached — `INSTALL_FAILURE` | Not runnable on this VM |
| TensorFold 0.6.1 @ `17c73e18` | EXL3 exists, but family-gated | `UNSUPPORTED_MODEL` (no MiniCPM family) and additionally refused on sm_86 | Not runnable on this GPU |

## Why each verdict

### ExLlamaV3 1.5.3 — SUPPORTED_NATIVE

`quantization_config.json` in the checkpoint is an **exllamav3-native artifact**: it is written by
`exllamav3/conversion/quant_config.py::create_quantization_config_json`, and the quantized linear
tensor group in `model.safetensors` is the standard EXL3 set `trellis` / `suh` / `svh` / `mul1`
(e.g. `model.layers.1.self_attn.q_proj.trellis` is `I16[128,128,64]` = 16×16 tiles, 4 bpw).

Authenticity was verified rather than assumed. Every `Linear` module's `quant_type` was inspected
after load:

```text
linear_modules: 295
quant_type_counts: {"exl3": 295}
non_exl3_linears: []
storage_info: bpw_layer = 4.012678, bpw_head = 6.007935
```

The adapter refuses to load unless this holds (`require_exl3=True`), so a silent fp16 fallback — which
ExLlamaV3 otherwise performs without any warning — cannot be reported as an EXL3 result.

### SGLang 0.5.20 + sglang_exl3 — INSTALL_FAILURE (environment driver/CUDA, not the plugin)

The plugin itself was obtained and inspected at the pinned commit
(`cuda/src/sglang_exl3`, registering `--quantization exl3` through the `sglang.srt.plugins` entry
point, with `get_min_capability() == 80`, i.e. Ampere would be covered). It could not be installed
because the pinned core runtime is not installable on this VM:

* `sglang==0.5.20` hard-pins `torch==2.13.0`; its resolved dependency set is CUDA 13 only
  (`nvidia-cudnn-cu13==9.20.0.48`, `nvidia-nvshmem-cu13`, `nvidia-cuda-crt==13.4.92`,
  `flashinfer-python==0.6.18`, `sglang-kernel==0.4.7`).
* `torch==2.13.0` on PyPI is a CUDA 13.0 build (`cuda-toolkit[...]==13.0.3`, `nvidia-*-cu13`,
  `cuda-bindings>=13.0.3`), and there is **no CUDA 12 build of torch 2.13.0 on any PyTorch index**
  (cu126/cu128/cu129 stop at 2.9.1), so no CUDA 12 route exists for the pin.
* This VM's driver supports **CUDA 12.4 maximum**. Direct probe of the CUDA 13 runtime:

  ```text
  cudaRuntimeGetVersion -> 13.0
  cudaDriverGetVersion  -> 12.4
  cudaSetDevice(0) -> rc=35: CUDA driver version is insufficient for CUDA runtime version
  ```

* The VM is an unprivileged container (no Docker-in-Docker), so the validated reference image
  `ghcr.io/0xsero/sglang-exl3:v0.7.0-ampere@sha256:c8922bd7256caf1b273af7ac49221e652be9e6072c1cdec4b959f506fd415d79`
  could not be used either, and being CUDA 13 based it would hit the same driver limit.

CUDA minor-version compatibility does not cross major versions, so a CUDA 13 runtime cannot run on a
CUDA 12.4 driver. **This says nothing about EXL3 support in SGLang or about MiniCPM5.** On a host with
driver r580+ (or any CUDA-13-capable driver) the same pinned stack should be re-attempted, and the
compatibility gate (architecture → EXL3 tensor layout → dense linears → embeddings → lm_head) still
has to be passed empirically for MiniCPM5 specifically.

### vLLM 0.30.0 + vllm-exl3 — INSTALL_FAILURE (same cause)

Identical blocker: `vllm==0.30.0` also hard-pins `torch==2.13.0`, and its resolution is likewise
CUDA 13 only (`nvidia-cudnn-cu13==9.20.0.48`, `nvidia-nvshmem-cu13`, `nvidia-cutlass-dsl[cu13]`).
No CUDA 12 path exists for the pin. Plugin commit `223e246f` was resolved and verified in the remote
(the repository HEAD equals the pinned revision).

Note for the future re-run: the architecture question for MiniCPM5 is *not* the blocker here — the
checkpoint declares `architectures: ["LlamaForCausalLM"]`, which vLLM supports. The gate to test
remains whether `vllm-exl3` loads this checkpoint's packed EXL3 tensor group and executes an
Ampere-correct kernel path (sm_86, never a Blackwell-only route).

### TensorFold 0.6.1 — UNSUPPORTED_MODEL

Two independent blockers, both verifiable in the pinned source and confirmed at runtime:

1. **No MiniCPM family.** The pinned revision's family registry is
   `bonsai`, `deepseek_v4`, `gemma4`, `glm5_next`, `nemotron_h`, `qwen3_5`, `qwen3_5_moe`,
   `qwen4_exp` (`src/tensorfold/families/`), and `grep -ri minicpm` over the repository returns
   nothing. `tensorfold models` lists EXL3 only for GLM-5.3-Flash, Qwen3.8-27B and Qwen3.8 Flash Next.
2. **The GPU is refused outright.** `src/tensorfold/cuda/build.py:13` sets `MIN_CAPABILITY = (8, 9)`,
   and `refuse_old_gpu()` runs at startup before any weights load. Executed against this GPU with the
   installed package:

   ```text
   ValueError: TensorFold's CUDA kernels need compute capability 8.9 or newer (FP8 MMA);
   this GPU (NVIDIA GeForce RTX 3060) is 8.6
   ```

No porting, no family addition, and no substitute model was attempted: another TensorFold-supported
model's numbers would not be comparable to this checkpoint, so TensorFold contributes **no**
performance row.

## Global rule observed

No model substitution, no re-quantization, no format change, no third-party kernel modification, and
no Pradium optimization was performed. Where a runtime could not genuinely execute the exact EXL3
checkpoint, that is the recorded result.
