#!/usr/bin/env python3
"""Direct vs split attention: numeric and token-level differential (EXP-0002 gate).

Runs the same prompt twice inside one process against the candidate build —
once with ``EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS=0`` (the existing split/combine
schedule) and once with the gate enabled (the direct single-pass regime) — and
compares, step by step:

  * greedy token IDs (identical or first divergence index)
  * the full logits vector of each step (max / mean |delta|, argmax agreement)

The engine reads the gate env var on every call, so both paths run against the
identical weights, cache layout and sampler without a rebuild or a second load.
Sampling semantics are unchanged (ArgmaxSampler); only the attention reduction
schedule differs, so the expected difference is float-reduction noise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

GATE = "EXL3_PRADIUM_DIRECT_ATTN_MAX_TOKENS"


def eos_ids(cfg) -> list[int]:
    raw = getattr(cfg, "eos_token_id_list", None) or getattr(cfg, "eos_token_id", None)
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return []


def run_once(generator, cfg, prompt, max_new_tokens, return_logits=True):
    from exllamav3 import ArgmaxSampler, Job

    ids = torch.tensor([prompt], dtype=torch.long)
    job = Job(
        input_ids=ids,
        max_new_tokens=int(max_new_tokens),
        min_new_tokens=int(max_new_tokens),
        stop_conditions=eos_ids(cfg),
        identifier="diff",
        sampler=ArgmaxSampler(),
        return_logits=return_logits,
    )
    generator.enqueue(job)
    tokens: list[int] = []
    logits: list[list[float]] = []
    steps = 0
    safety = max_new_tokens + 32
    while steps < safety:
        results = generator.iterate()
        done = False
        for res in results:
            new_ids = res.get("token_ids")
            if new_ids is not None:
                tokens.extend(int(t) for t in new_ids.flatten().tolist())
            lg = res.get("logits")
            if lg is not None:
                logits.append(lg[0, -1].float().cpu().tolist())
            if res.get("eos"):
                done = True
        steps += 1
        if done:
            break
    return tokens, logits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--gate", type=int, default=4097)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import exllamav3
    from exllamav3 import Cache, Config, Generator, Model, Tokenizer

    cfg = Config.from_directory(args.model_dir)
    tokenizer = Tokenizer.from_config(cfg)
    model = Model.from_config(cfg)
    cache = Cache(model, max_num_tokens=32768, max_batch_size=16, max_history=0)
    model.load(progressbar=False, device="cuda:0")
    generator = Generator(
        model=model, cache=cache, tokenizer=tokenizer,
        max_batch_size=16, max_chunk_size=2048, cpu_cache_size=0,
    )
    torch.cuda.synchronize()

    prompt = [1000 + (i * 7919) % 30000 for i in range(args.prompt_tokens)]

    os.environ[GATE] = "0"
    split_tokens, split_logits = run_once(generator, cfg, prompt, args.max_new_tokens)

    os.environ[GATE] = str(args.gate)
    direct_tokens, direct_logits = run_once(generator, cfg, prompt, args.max_new_tokens)

    n = min(len(split_tokens), len(direct_tokens))
    first_div = next((i for i in range(n) if split_tokens[i] != direct_tokens[i]), None)
    steps = min(len(split_logits), len(direct_logits))
    max_abs = 0.0
    mean_abs = 0.0
    argmax_agree = 0
    for i in range(steps):
        a = torch.tensor(split_logits[i])
        b = torch.tensor(direct_logits[i])
        d = (a - b).abs()
        max_abs = max(max_abs, float(d.max()))
        mean_abs += float(d.mean())
        argmax_agree += int(int(a.argmax()) == int(b.argmax()))
    if steps:
        mean_abs /= steps

    result = {
        "exllamav3_file": exllamav3.__file__,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "gate": args.gate,
        "tokens_compared": n,
        "tokens_identical": first_div is None and len(split_tokens) == len(direct_tokens),
        "first_divergence_index": first_div,
        "split_tokens": split_tokens[:24],
        "direct_tokens": direct_tokens[:24],
        "logit_steps_compared": steps,
        "logit_max_abs_delta": max_abs,
        "logit_mean_abs_delta": mean_abs,
        "argmax_agreement_steps": argmax_agree,
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            fh.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
