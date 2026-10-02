# PRADIUM EXL3 B0 — RTX 3060 / MiniCPM5-2B EXL3

First real, reproducible, cross-runtime EXL3 baseline produced on the supplied
RTX 3060 12 GB VM, against one frozen model revision and one frozen benchmark.

| | |
|---|---|
| Campaign | `PRADIUM-EXL3-RTX3060-B0` |
| Benchmark | `PRADIUM-RUNTIME-BENCH-v2` (campaign spec label `PRADIUM-RUNTIME-BENCH-v1`) |
| Model | `ewin-reg/MiniCPM5-2B-EXL3-Quantized` @ `e36fbb568f466739d7c244a5100d01acf93d3b7f` |
| Quantization | EXL3, 4.0 bpw (head 6 bpw) |
| GPU | NVIDIA GeForce RTX 3060 12 GB, Ampere sm_86, driver 550.144.03 |
| Pradium commit | `cf4e8754e6edcf588ebf0a207472e19c24f233d3` |
| Campaign branch | `bench/exl3-runtime-b0-rtx3060` |

## Result at a glance

| Runtime stack | EXL3 route | MiniCPM5 EXL3 status | Benchmark status |
|---|---|---|---|
| ExLlamaV3 1.5.3 | Native | `SUPPORTED_NATIVE` | Full official suite |
| SGLang 0.5.20 + sglang_exl3 | Plugin | `INSTALL_FAILURE` (driver/CUDA 13) | Not runnable on this VM |
| vLLM 0.30.0 + vllm-exl3 | Plugin | `INSTALL_FAILURE` (driver/CUDA 13) | Not runnable on this VM |
| TensorFold 0.6.1 | EXL3 exists, family-gated | `UNSUPPORTED_MODEL` (+ sm_86 refused) | Not runnable here |

Only ExLlamaV3 genuinely executed the exact EXL3 checkpoint, so it is the only
stack with performance numbers. The other three verdicts are environment and
family gates, documented rather than worked around — see
[`compatibility.md`](compatibility.md).

## Layout

```text
campaigns/rtx3060-minicpm5-exl3-b0/
├── campaign.json          campaign manifest: hardware, model, pins, hashes
├── model_manifest.json    per-file SHA-256 + safetensors header/data hashes
├── compatibility.md       the compatibility matrix and why each verdict holds
├── environments/          one environment manifest per runtime stack
│   ├── exllamav3.json
│   ├── sglang-exl3.json
│   ├── vllm-exl3.json
│   └── tensorfold.json
├── verification/          pre-benchmark evidence for ExLlamaV3
│   ├── exllamav3_smoke.json
│   └── exllamav3_verify.json
├── exllamav3/             results, summary and report per suite
└── sessions/              raw harness sessions (git-ignored; see below)
```

`benchmarks/results/` and `results/` are ignored by the repository's own policy
("generated; never commit"), so the raw harness sessions are written to
`sessions/` via `--results-dir` and the derived artifacts for each runtime are
collected under the runtime directory for review.

## Reproducing

The benchmark must run *inside* each runtime's own environment, because the
adapter imports the runtime under test.

```bash
# 1. frozen model, pinned revision
MODEL_REVISION=e36fbb568f466739d7c244a5100d01acf93d3b7f
hf download ewin-reg/MiniCPM5-2B-EXL3-Quantized \
  --revision "$MODEL_REVISION" --local-dir /workspace/models/minicpm5-2b-exl3

# 2. runtimes (each in its own environment, see environments/*.json for exact versions)
uv venv --python 3.12 /workspace/envs/exllamav3
uv pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cu128
uv pip install "https://github.com/turboderp-org/exllamav3/releases/download/v1.5.3/exllamav3-1.5.3%2Bcu128.torch2.10.0-cp312-cp312-linux_x86_64.whl"
uv pip install transformers pytest

# 3. frozen inputs, materialized once with the exact model tokenizer
python -m benchmarks corpus materialize \
  --tokenizer "hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@${MODEL_REVISION}"

# 4. adapter conformance must pass before any benchmark is trusted
PRADIUM_EXLLAMAV3_MODEL=/workspace/models/minicpm5-2b-exl3 \
  python -m pytest benchmarks/tests/test_adapter_exllamav3.py -q

# 5. suites
python -m benchmarks run matrix --runtime exllamav3 --tokenizer "hf:ewin-reg/MiniCPM5-2B-EXL3-Quantized@${MODEL_REVISION}" \
  --model ewin-reg/MiniCPM5-2B-EXL3-Quantized --model-revision "$MODEL_REVISION" \
  --quantization EXL3 --bpw 4.0 --results-dir <campaign>/sessions \
  --runtime-config '{"model_dir":"/workspace/models/minicpm5-2b-exl3","max_num_tokens":32768,"max_batch_size":16,"max_chunk_size":2048}'
```

## Integrity rules applied

* No model substitution, no re-quantization, no format change (never BF16/FP16/INT8/AWQ/GPTQ/GGUF/HQQ/NVFP4/BitsAndBytes).
* No third-party kernels modified.
* Benchmark prompts, token lengths, output lengths, warmup/measured counts and metric formulas unchanged.
* Pradium itself was not optimized in this campaign.
* EXL3 execution is *proved*, not assumed: the adapter refuses to load unless every
  linear module reports `quant_type == "exl3"`, which is how a silent fp16 fallback
  (which ExLlamaV3 performs without warning) is prevented from being reported as EXL3.
