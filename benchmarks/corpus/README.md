# Frozen Prompt Corpus (PRADIUM-RUNTIME-BENCH-v1)

This directory is the **frozen prompt corpus** of the Pradium Runtime
Benchmark. Official runs never generate random prompts and never require an
external dataset or network access.

## Layout

```text
corpus/
├── README.md           # this file
├── manifest.json       # frozen SHA-256 hashes of sources + materializations
├── sources/            # human-readable source documents (committed)
│   ├── conversation.md # conversational prose and dialogue
│   ├── technical.md    # distributed systems, GPU, caching, scheduling, ...
│   ├── code.md         # Python, C/C++, JSON, shell, YAML fragments
│   ├── structured.md   # tables, lists, key/value data, logs, config
│   └── mixed.md        # realistic prose + code + structured document
├── tokenizers/
│   └── pradium-simple-v1/vocab.json   # frozen vocabulary (committed)
├── materialize.py      # deterministic materializer + validation
└── materialized/       # generated input_ids (gitignored, regenerable)
```

## Source material

The five source documents contain realistic material of the kinds real chat,
coding, and agent systems process: natural paragraphs and dialogue, technical
prose about systems topics, real code snippets in several languages,
tables/lists/logs/configuration, and a mixed engineering document. Prompts are
never built by repeating one token or phrase.

## Exact token counts

Different models use different tokenizers, so source text and tokenized prompts
are separated:

```text
source corpus  +  model tokenizer  ->  exact 128-token prompt
                                     -> exact 1024-token prompt
                                     -> exact 4096-token prompt
                                     -> ...
```

Materialization composes a deterministic token stream from the sources (each
segment preceded by a bounded index marker, sources cycled in a fixed order)
and truncates the **token-id stream** to exactly the requested length. Official
workloads therefore use exact counts — 128, 1024, 4096, 512, 2048, 8192, 16384,
32768 — never approximately.

The token-id stream is authoritative. Prompt text is stored for runtimes that
can only consume text (recorded as a fairness deviation in results).

## The frozen tokenizer

`pradium-simple-v1` is a deterministic, lossless piece tokenizer whose
vocabulary is frozen in `tokenizers/pradium-simple-v1/vocab.json`. Its revision
is the content hash of that vocabulary. It is used for infrastructure
validation and mock runs. For real runtimes, materialize with the model's own
tokenizer:

```bash
python -m benchmarks corpus materialize --tokenizer hf:<model-or-tokenizer>[@<revision>]
```

Materialized files land in `corpus/materialized/<tokenizer-id>/` and include
`input_ids`, tokenizer name/revision, prompt hash, input-ids hash and token
count for every frozen workload.

## Immutability and validation

`manifest.json` records SHA-256 hashes of every source file, the tokenizer
vocabulary, and every materialized prompt (prompt hash + input-ids hash), plus
the benchmark version. Validation rebuilds all materializations from source and
compares hashes — which also proves determinism and exact token counts:

```bash
python -m benchmarks corpus validate
```

If a frozen prompt, source, or the vocabulary changes unexpectedly, validation
**FAILS** and official runs must not proceed. Any intentional change requires a
benchmark version bump (`PRADIUM-RUNTIME-BENCH-v2`).

## Prefix / KV-reuse groups

`materialize.py` also builds frozen prefix groups for ratios 0 / 25 / 50 / 75 /
90 / ~100% of reusable prefix. Ratios are defined on **token IDs**
(`shared_prefix_tokens / total_input_tokens`), never on characters. At ~100%
each request is a large identical prefix (1008 tokens) plus a small unique
suffix (16 tokens) — never exact duplicate full prompts. Every request records
`total_input_tokens`, `shared_prefix_tokens`, `unique_suffix_tokens` and
`expected_reuse_ratio`.

## Commands

```bash
# rebuild the frozen vocabulary and manifest (maintainers only)
python -m benchmarks.corpus.materialize build

# validate the frozen corpus (also runs automatically before official runs)
python -m benchmarks corpus validate

# materialize exact-length input_ids for a tokenizer
python -m benchmarks corpus materialize --tokenizer simple
python -m benchmarks corpus materialize --tokenizer hf:<name>@<revision>
```
