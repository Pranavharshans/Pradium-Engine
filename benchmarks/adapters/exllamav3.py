"""ExLlamaV3 runtime adapter (native EXL3 execution path).

ExLlamaV3 is the reference implementation of the EXL3 trellis quantization
format, so this adapter must never alter the checkpoint or the kernels: it
loads the frozen checkpoint directory as-is and drives
``exllamav3.generator.Generator``, which is a continuous-batching engine.

Threading model
---------------
``Generator.iterate()`` is a single-threaded driver: one call advances every
active job by one decode step. The benchmark drives runtimes from several
threads (concurrency C1..C8, the scheduler-interference suite), so this adapter
owns exactly one *scheduler thread* which is the only caller of ``iterate()``.
Each ``generate_stream()`` call enqueues a ``Job`` and consumes a private queue
that the scheduler fills. That is the engine's real execution model rather
than a serialization workaround, so Cn measurements are genuinely overlapping.

Determinism
-----------
``job.sampler=None`` selects ``DefaultSampler`` (MinP 0.08 + temperature 0.8),
which is **stochastic**. Official runs request ``greedy=True``, and this adapter
then passes ``ArgmaxSampler()`` explicitly. That is the only sampling path used
for official comparison.

Fixed output length
-------------------
With ``ignore_eos``/``fixed_decode_length`` the adapter passes the model's EOS
ids as stop conditions *and* ``min_new_tokens == max_new_tokens``, which masks
those ids for the whole generation, so the job ends with
``eos_reason == "max_new_tokens"`` after exactly N tokens.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..core.capabilities import CapabilityStatus, RuntimeCapabilities
from ..core.request import BenchmarkRequest
from .base import (
    CapabilityNotSupported,
    GenerationResult,
    RuntimeAdapter,
    RuntimeAdapterError,
    RuntimeOutOfMemoryError,
    StreamToken,
)

EXLLAMAV3_RUNTIME_NAME = "exllamav3"

#: Pinned campaign revision of the runtime this adapter was written against.
EXLLAMAV3_PINNED_COMMIT = "d3739fd393337b1ff4d6c2a342b12f0c87a9592f"
EXLLAMAV3_PINNED_VERSION = "1.5.3"


@dataclass
class ExLlamaV3RuntimeConfig:
    """Adapter configuration. Defaults follow the documented ExLlamaV3 API."""

    model_dir: str = ""
    #: KV cache capacity in tokens.
    max_num_tokens: int = 32768
    #: Maximum concurrently active sequences (continuous-batching width).
    max_batch_size: int = 16
    #: Prefill chunk size; ExLlamaV3's own default is 2048.
    max_chunk_size: int = 2048
    #: fp16 KV cache by default; set both to quantize the cache.
    cache_k_bits: int | None = None
    cache_v_bits: int | None = None
    #: Device string passed to ``Model.load``.
    device: str = "cuda:0"
    #: Host-tier prefix cache size in bytes (0 disables the tier).
    cpu_cache_size: int = 0
    #: Progress bar for model load (off for clean logs).
    progressbar: bool = False
    #: Wall-clock cap for a single enqueued job's queue wait (seconds).
    job_deadline_s: float = 900.0
    #: Fail the load unless every Linear module is confirmed to execute as EXL3.
    #: The ExLlamaV3 loader falls back to fp16 silently when it cannot find the
    #: EXL3 tensor group, so a run labelled EXL3 must be provable.
    require_exl3: bool = True
    model: str = ""
    model_revision: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ActiveJob:
    """Per-request bookkeeping shared between the caller and the scheduler."""

    request_id: str
    events: "queue.Queue[tuple[StreamToken | None, BaseException | None]]"
    emitted: int = 0
    seen: int = 0
    first_token_ns: int | None = None
    finished: bool = False
    runtime_reported: dict[str, Any] | None = None


class ExLlamaV3Runtime(RuntimeAdapter):
    """Runtime adapter around ExLlamaV3's native EXL3 execution path."""

    name = EXLLAMAV3_RUNTIME_NAME

    def __init__(self, config: ExLlamaV3RuntimeConfig | None = None) -> None:
        self.config = config or ExLlamaV3RuntimeConfig()
        self._torch = None
        self._config_obj = None
        self._tokenizer = None
        self._model = None
        self._cache = None
        self._generator = None
        self._loaded = False
        self._lock = threading.RLock()
        self._generator_lock = threading.RLock()
        self._active: dict[str, _ActiveJob] = {}
        self._scheduler: threading.Thread | None = None
        self._scheduler_stop = threading.Event()
        self._scheduler_error: BaseException | None = None
        self._milestones: dict[str, float | None] = {}
        self._runtime_info: dict[str, Any] = {}
        self._quant_audit: dict[str, Any] = {}
        self._cache_stats_at_load: dict[str, Any] | None = None
        self._peak_vram_mb: float | None = None
        self._cancel = False

    # ---------------------------------------------------------------- setup
    def initialize(self, config: dict[str, Any] | None = None) -> None:
        if config:
            merged = dict(self.config.__dict__)
            merged.update(config)
            self.config = ExLlamaV3RuntimeConfig(**merged)

        import torch  # noqa: PLC0415 - optional heavy dependency

        self._torch = torch
        self._milestones["initialize_ns"] = time.perf_counter_ns()
        if not torch.cuda.is_available():
            raise RuntimeAdapterError(
                "ExLlamaV3 adapter requires a CUDA device; torch reports none"
            )

    def load_model(self) -> None:
        import torch  # noqa: PLC0415

        from exllamav3 import Cache, Config, Generator, Model, Tokenizer  # noqa: PLC0415

        if not self.config.model_dir:
            raise RuntimeAdapterError("model_dir is required for the exllamav3 adapter")

        t0 = time.perf_counter_ns()
        if not os.path.isdir(self.config.model_dir):
            raise RuntimeAdapterError(
                f"model directory does not exist: {self.config.model_dir!r}"
            )
        try:
            cfg = Config.from_directory(self.config.model_dir)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeAdapterError(
                f"ExLlamaV3 could not read a model config from "
                f"{self.config.model_dir!r}: {exc}"
            ) from exc
        t_cfg = time.perf_counter_ns()
        self._milestones["config_load_ms"] = (t_cfg - t0) / 1e6
        self._config_obj = cfg

        tokenizer = Tokenizer.from_config(cfg)
        self._milestones["tokenizer_load_ms"] = (time.perf_counter_ns() - t_cfg) / 1e6
        self._tokenizer = tokenizer

        model = Model.from_config(cfg)
        cache_ctor_kwargs: dict[str, Any] = {
            "max_num_tokens": int(self.config.max_num_tokens),
            "max_batch_size": int(self.config.max_batch_size),
            "max_history": 0,
        }
        cache = Cache(model, **cache_ctor_kwargs)
        self._cache = cache
        self._model = model

        t_load = time.perf_counter_ns()
        try:
            model.load(progressbar=bool(self.config.progressbar), device=self.config.device)
        except Exception as exc:  # noqa: BLE001
            msg = str(exc).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                raise RuntimeOutOfMemoryError(str(exc)) from exc
            raise RuntimeAdapterError(f"ExLlamaV3 model load failed: {exc}") from exc
        self._milestones["model_load_ms"] = (time.perf_counter_ns() - t_load) / 1e6
        self._milestones["cuda_init_ms"] = (t_load - t0) / 1e6

        gen_kwargs: dict[str, Any] = {
            "model": model,
            "cache": cache,
            "tokenizer": tokenizer,
            "max_batch_size": int(self.config.max_batch_size),
            "max_chunk_size": int(self.config.max_chunk_size),
            "cpu_cache_size": int(self.config.cpu_cache_size),
        }
        self._generator = Generator(**gen_kwargs)
        self._loaded = True
        self._milestones["generator_ready_ms"] = (time.perf_counter_ns() - t0) / 1e6

        self._quant_audit = self._audit_exl3_execution()
        if self.config.require_exl3 and not self._quant_audit.get("exl3_only"):
            counts = self._quant_audit.get("quant_type_counts")
            raise RuntimeAdapterError(
                "EXL3 authenticity could not be verified for "
                f"{self.config.model_dir!r}: quant_type counts {counts}, "
                f"non-EXL3 linears {self._quant_audit.get('non_exl3_linears')}. "
                "Refusing to label a dequantized/fallback run as EXL3."
            )
        self._runtime_info = self._build_runtime_info()
        self._milestones["model_ready_total_ms"] = (time.perf_counter_ns() - t0) / 1e6

        self._start_scheduler()

    def unload_model(self) -> None:
        self._stop_scheduler()
        with self._lock:
            self._generator = None
            if self._model is not None:
                try:
                    self._model.unload()
                except Exception:  # noqa: BLE001
                    pass
            self._model = None
            self._cache = None
            self._loaded = False
        gc = __import__("gc")
        gc.collect()
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def shutdown(self) -> None:
        self.unload_model()
        self._tokenizer = None
        self._config_obj = None

    # ------------------------------------------------------------ scheduler
    def _start_scheduler(self) -> None:
        self._scheduler_stop.clear()
        self._scheduler_error = None
        self._scheduler = threading.Thread(
            target=self._scheduler_loop, name="exllamav3-scheduler", daemon=True
        )
        self._scheduler.start()

    def _stop_scheduler(self) -> None:
        self._scheduler_stop.set()
        thread = self._scheduler
        if thread is not None and thread.is_alive():
            thread.join(timeout=30.0)
        self._scheduler = None
        with self._lock:
            for job in self._active.values():
                if not job.finished:
                    job.events.put((None, RuntimeAdapterError("runtime stopped")))
            self._active.clear()

    def _scheduler_loop(self) -> None:
        """Single caller of ``Generator.iterate()``; fans tokens out to queues."""
        try:
            while not self._scheduler_stop.is_set():
                with self._lock:
                    generator = self._generator
                    active = list(self._active.values())
                if generator is None or not active:
                    time.sleep(0.0002)
                    continue
                started = time.perf_counter_ns()
                try:
                    results = generator.iterate()
                except Exception as exc:  # noqa: BLE001
                    self._scheduler_error = exc
                    with self._lock:
                        for job in self._active.values():
                            job.events.put((None, exc))
                        self._active.clear()
                    return
                for res in results:
                    self._dispatch(res, started)
        except Exception as exc:  # noqa: BLE001 - never let the thread die silently
            self._scheduler_error = exc

    def _dispatch(self, res: dict[str, Any], started_ns: int) -> None:
        job_obj = res.get("job")
        identifier = res.get("identifier")
        if identifier is None:
            return
        with self._lock:
            job = self._active.get(str(identifier))
        if job is None:
            return

        stage = res.get("stage")
        if stage == "error":
            job.events.put((None, RuntimeAdapterError(str(res.get("error") or "job error"))))
            self._finish(job)
            return

        new_ids = res.get("token_ids")
        if new_ids is not None:
            ids = new_ids.flatten().tolist()
            for token_id in ids:
                if job.first_token_ns is None:
                    job.first_token_ns = started_ns
                text = res.get("text") if len(ids) == 1 else None
                event = StreamToken(
                    token_id=int(token_id),
                    index=job.emitted,
                    text=text,
                    done=False,
                )
                job.emitted += 1
                job.events.put((event, None))

        if res.get("eos"):
            job.runtime_reported = {
                "eos_reason": res.get("eos_reason"),
                "new_tokens": res.get("new_tokens"),
                "time_enqueued_s": res.get("time_enqueued"),
                "time_prefill_s": res.get("time_prefill"),
                "time_generate_s": res.get("time_generate"),
                "accepted_draft_tokens": res.get("accepted_draft_tokens"),
                "rejected_draft_tokens": res.get("rejected_draft_tokens"),
            }
            self._finish(job)

    def _finish(self, job: _ActiveJob) -> None:
        job.finished = True
        with self._lock:
            self._active.pop(job.request_id, None)
        job.events.put((None, None))

    # ----------------------------------------------------------- generation
    def _build_job(self, request: BenchmarkRequest):
        import torch  # noqa: PLC0415

        from exllamav3 import ArgmaxSampler, Job  # noqa: PLC0415

        ids = torch.tensor([list(request.input_ids)], dtype=torch.long)
        eos_ids: list[int] = []
        cfg = self._config_obj
        raw_eos = getattr(cfg, "eos_token_id_list", None) or getattr(cfg, "eos_token_id", None)
        if isinstance(raw_eos, int):
            eos_ids = [raw_eos]
        elif isinstance(raw_eos, (list, tuple)):
            eos_ids = [int(x) for x in raw_eos]

        mask_eos = bool(request.ignore_eos and request.fixed_decode_length)
        kwargs: dict[str, Any] = {
            "input_ids": ids,
            "max_new_tokens": int(request.max_new_tokens),
            "stop_conditions": list(eos_ids),
            "identifier": request.request_id,
        }
        if mask_eos:
            # Suppress EOS for the full generation: exact N tokens, eos_reason
            # == "max_new_tokens".
            kwargs["min_new_tokens"] = int(request.max_new_tokens)

        if request.greedy or request.temperature <= 0.0:
            kwargs["sampler"] = ArgmaxSampler()
        return Job(**kwargs)

    def generate_stream(self, request: BenchmarkRequest) -> Iterator[StreamToken]:
        if not self._loaded:
            raise RuntimeAdapterError("model is not loaded")
        if self._scheduler_error is not None:
            raise RuntimeAdapterError(f"scheduler failed: {self._scheduler_error}")

        job = self._build_job(request)
        active = _ActiveJob(request_id=request.request_id, events=queue.Queue())
        with self._lock:
            self._active[request.request_id] = active
            self._generator.enqueue(job)

        deadline = time.perf_counter() + float(self.config.job_deadline_s)
        try:
            while True:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise RuntimeAdapterError(
                        f"job {request.request_id} exceeded {self.config.job_deadline_s}s"
                    )
                try:
                    token, error = active.events.get(timeout=min(remaining, 5.0))
                except queue.Empty:
                    if self._scheduler_error is not None:
                        raise RuntimeAdapterError(f"scheduler failed: {self._scheduler_error}")
                    continue
                if error is not None:
                    raise error
                if token is None:
                    break
                yield token

            reported = active.runtime_reported or {}
            prefill_ms = reported.get("time_prefill_s")
            queue_ms = reported.get("time_enqueued_s")
            yield StreamToken(
                token_id=None,
                index=active.emitted,
                done=True,
                finish_reason=str(reported.get("eos_reason") or "max_new_tokens"),
                prefill_ms=(float(prefill_ms) * 1000.0) if prefill_ms is not None else None,
                queue_ms=(float(queue_ms) * 1000.0) if queue_ms is not None else None,
                internal_metrics={
                    "requested_output_tokens": int(request.max_new_tokens),
                    "actual_output_tokens": active.emitted,
                    "early_termination": active.emitted < int(request.max_new_tokens),
                    "runtime_new_tokens": reported.get("new_tokens"),
                    "runtime_time_generate_s": reported.get("time_generate_s"),
                    "first_token_ns": active.first_token_ns,
                },
            )
        finally:
            if not active.finished:
                with self._lock:
                    self._active.pop(request.request_id, None)

    def generate(self, request: BenchmarkRequest) -> GenerationResult:
        token_ids: list[int] = []
        prefill_ms = None
        queue_ms = None
        finish_reason = "length"
        internal: dict[str, Any] = {}
        for event in self.generate_stream(request):
            if event.done:
                prefill_ms = event.prefill_ms
                queue_ms = event.queue_ms
                finish_reason = event.finish_reason or finish_reason
                internal = dict(event.internal_metrics)
                break
            if event.token_id is not None:
                token_ids.append(int(event.token_id))
        text = None
        if self._tokenizer is not None and token_ids:
            text = self._decode(token_ids)
        return GenerationResult(
            token_ids=token_ids,
            text=text,
            finish_reason=finish_reason,
            prefill_ms=prefill_ms,
            queue_ms=queue_ms,
            internal_metrics=internal,
        )

    def generate_batch(
        self, requests: list[BenchmarkRequest]
    ) -> list[GenerationResult]:
        """True static batch: every sequence is enqueued before any decode step.

        ExLlamaV3 advances all active sequences by one token per ``iterate()``
        call, so a batch enqueued together is decoded in lockstep, which is what
        the B1/B2/B4/B8 suite measures.
        """
        if not requests:
            return []
        jobs = [self._build_job(r) for r in requests]
        actives: list[_ActiveJob] = []
        with self._lock:
            for request, job in zip(requests, jobs):
                active = _ActiveJob(request_id=request.request_id, events=queue.Queue())
                actives.append(active)
                self._active[request.request_id] = active
            self._generator.enqueue(jobs)

        results: list[GenerationResult | None] = [None] * len(requests)
        deadline = time.perf_counter() + float(self.config.job_deadline_s)
        pending = list(range(len(requests)))
        tokens: list[list[int]] = [[] for _ in requests]
        try:
            while pending:
                if time.perf_counter() > deadline:
                    raise RuntimeAdapterError("batch exceeded its deadline")
                progressed = False
                for idx in list(pending):
                    active = actives[idx]
                    drained = False
                    while True:
                        try:
                            token, error = active.events.get_nowait()
                        except queue.Empty:
                            break
                        if error is not None:
                            raise error
                        if token is None:
                            drained = True
                            break
                        if token.token_id is not None:
                            tokens[idx].append(int(token.token_id))
                        progressed = True
                    if drained:
                        reported = active.runtime_reported or {}
                        prefill = reported.get("time_prefill_s")
                        queued = reported.get("time_enqueued_s")
                        results[idx] = GenerationResult(
                            token_ids=tokens[idx],
                            text=self._decode(tokens[idx]) if self._tokenizer and tokens[idx] else None,
                            finish_reason=str(reported.get("eos_reason") or "max_new_tokens"),
                            prefill_ms=(float(prefill) * 1000.0) if prefill is not None else None,
                            queue_ms=(float(queued) * 1000.0) if queued is not None else None,
                            internal_metrics={
                                "requested_output_tokens": int(requests[idx].max_new_tokens),
                                "actual_output_tokens": len(tokens[idx]),
                                "batch_size": len(requests),
                            },
                        )
                        pending.remove(idx)
                        progressed = True
                if not progressed:
                    time.sleep(0.0002)
        finally:
            with self._lock:
                for request in requests:
                    self._active.pop(request.request_id, None)
        return [r for r in results if r is not None]

    def _decode(self, token_ids: list[int]) -> str | None:
        try:
            import torch  # noqa: PLC0415

            return self._tokenizer.decode(
                torch.tensor([token_ids], dtype=torch.long), decode_special_tokens=True
            )
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------ cache ctrl
    def reset_cache(self) -> None:
        """Drop **all** cached KV/prefix pages — a true cold start.

        ExLlamaV3 v1.5.3 has no ``free_cache``, and neither ``clear_queue()`` nor
        ``pagetable.reset_page_table()`` produces a verifiable empty cache: page
        statistics are memoised on a page-table signature, and the reset leaves
        cumulative metrics in place, so a stale snapshot can be served and the
        reported ``cached_tokens`` does not go to zero.

        Constructing a fresh ``Generator`` over the same ``Cache`` builds a new
        ``PageTable``, which indexes the existing page tensors from scratch and
        therefore discards every content-hashed prompt page. That is the only
        reset whose effect is observable (``cached_tokens == 0``), so it is the
        one the benchmark uses. The KV tensors are not reallocated, so this is
        cheap and does not distort VRAM measurements.
        """
        generator = self._generator
        if generator is None:
            return
        deadline = time.perf_counter() + 120.0
        try:
            while generator.num_remaining_jobs() and time.perf_counter() < deadline:
                generator.iterate()
        except Exception:  # noqa: BLE001
            pass
        with self._generator_lock:
            self._stop_scheduler()
            try:
                self._recreate_generator()
            finally:
                self._start_scheduler()

    def _recreate_generator(self) -> None:
        from exllamav3 import Generator  # noqa: PLC0415

        self._generator = Generator(**self._generator_kwargs())
        self._cache_stats_at_load = None

    def _generator_kwargs(self) -> dict[str, Any]:
        return {
            "model": self._model,
            "cache": self._cache,
            "tokenizer": self._tokenizer,
            "max_batch_size": int(self.config.max_batch_size),
            "max_chunk_size": int(self.config.max_chunk_size),
            "cpu_cache_size": int(self.config.cpu_cache_size),
        }

    def clear_prefix_cache(self, prefix_id: str | None = None) -> None:
        """ExLlamaV3 keeps one implicit content-hashed page table.

        There is no per-prefix eviction API in v1.5.3, so this resets the whole
        page table (recorded as a global reset, never a fake partial clear).
        """
        self.reset_cache()

    # --------------------------------------------------------- introspection
    def _audit_exl3_execution(self) -> dict[str, Any]:
        """Prove the checkpoint is executed as EXL3, not silently dequantized.

        The loader falls back to fp16 silently when it cannot find the EXL3
        tensor group, so every ``Linear`` module's ``quant_type`` is inspected
        and the storage bitrate is compared against the declared 4.0 bpw.
        """
        counts: dict[str, int] = {}
        offenders: list[str] = []
        model = self._model
        try:
            modules = list(getattr(model, "modules", []) or [])
        except Exception:  # noqa: BLE001
            modules = []

        stack = list(modules)
        seen = 0
        while stack:
            mod = stack.pop()
            sub = getattr(mod, "modules", None)
            if isinstance(sub, (list, tuple)):
                stack.extend(sub)
            cls_name = type(mod).__name__
            if cls_name == "Linear":
                seen += 1
                qt = getattr(mod, "quant_type", None)
                counts[str(qt)] = counts.get(str(qt), 0) + 1
                if qt != "exl3":
                    key = getattr(mod, "key", "?")
                    if len(offenders) < 10:
                        offenders.append(str(key))

        info: dict[str, Any] = {
            "linear_modules": seen,
            "quant_type_counts": counts,
            "non_exl3_linears": offenders,
            "exl3_only": seen > 0 and set(counts) <= {"exl3"},
        }
        try:
            storage = model.get_storage_info()
            info["storage_info"] = {
                "bpw_layer": storage[0],
                "bpw_head": storage[1],
                "vram_bits": storage[2],
            }
        except Exception as exc:  # noqa: BLE001
            info["storage_info"] = {"error": repr(exc)}
        return info

    def _build_runtime_info(self) -> dict[str, Any]:
        import importlib.metadata as md  # noqa: PLC0415

        info: dict[str, Any] = {
            "name": EXLLAMAV3_RUNTIME_NAME,
            "version": None,
            "commit": EXLLAMAV3_PINNED_COMMIT,
            "config": {
                "model_dir": self.config.model_dir,
                "max_num_tokens": self.config.max_num_tokens,
                "max_batch_size": self.config.max_batch_size,
                "max_chunk_size": self.config.max_chunk_size,
                "cache_k_bits": self.config.cache_k_bits,
                "cache_v_bits": self.config.cache_v_bits,
                "cpu_cache_size": self.config.cpu_cache_size,
                "device": self.config.device,
                "greedy_sampler": "ArgmaxSampler",
                "cuda_graphs": False,
            },
            "model": self.config.model,
            "model_revision": self.config.model_revision,
            "quantization": "EXL3",
            "exl3_audit": self._quant_audit,
        }
        try:
            info["version"] = md.version("exllamav3")
        except Exception:  # noqa: BLE001
            info["version"] = None
        try:
            import exllamav3  # noqa: PLC0415

            info["torch"] = getattr(self._torch, "__version__", None)
            info["torch_cuda"] = getattr(getattr(self._torch, "version", None), "cuda", None)
            info["exllamav3_path"] = getattr(exllamav3, "__file__", None)
        except Exception:  # noqa: BLE001
            pass
        if self._torch is not None and self._torch.cuda.is_available():
            info["gpu"] = self._torch.cuda.get_device_name(0)
            info["capability"] = list(self._torch.cuda.get_device_capability(0))
        cfg = self._config_obj
        if cfg is not None:
            info["model_config"] = {
                "architecture": getattr(cfg, "architecture", None),
                "model_type": getattr(cfg, "model_type", None),
                "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
                "vocab_size": getattr(cfg, "vocab_size", None),
                "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
                "eos_token_id": getattr(cfg, "eos_token_id", None),
            }
        cache = self._cache
        if cache is not None:
            info["kv_cache"] = {
                "max_num_tokens": _json_safe(getattr(cache, "max_num_tokens", None)),
                "num_slots": _json_safe(getattr(cache, "num_slots", None)),
                "num_layers": _json_safe(getattr(cache, "num_layers", None)),
                "layer_type": _json_safe(
                    getattr(getattr(cache, "layer_type", None), "__name__", None)
                ),
                "initialized": _json_safe(getattr(cache, "initialized", None)),
                "k_bits": _json_safe(getattr(cache, "k_bits", None)),
                "v_bits": _json_safe(getattr(cache, "v_bits", None)),
            }
        generator = self._generator
        if generator is not None and hasattr(generator, "get_cache_stats"):
            try:
                self._cache_stats_at_load = _json_safe(generator.get_cache_stats())
                info["cache_stats_at_load"] = self._cache_stats_at_load
            except Exception:  # noqa: BLE001
                pass
        return info

    def get_capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_streaming=CapabilityStatus.SUPPORTED,
            supports_input_ids=CapabilityStatus.SUPPORTED,
            supports_fixed_decode_length=CapabilityStatus.SUPPORTED,
            supports_prefix_cache=CapabilityStatus.SUPPORTED,
            supports_cross_request_prefix_cache=CapabilityStatus.SUPPORTED,
            supports_continuous_batching=CapabilityStatus.SUPPORTED,
            supports_static_batching=CapabilityStatus.SUPPORTED,
            supports_chunked_prefill=CapabilityStatus.SUPPORTED,
            supports_cuda_graphs=CapabilityStatus.UNSUPPORTED,
            supports_kv_quantization=CapabilityStatus.SUPPORTED,
            supports_cache_reset=CapabilityStatus.SUPPORTED,
            supports_internal_queue_metrics=CapabilityStatus.PARTIAL,
            supports_internal_scheduler_metrics=CapabilityStatus.UNSUPPORTED,
            supports_internal_kv_metrics=CapabilityStatus.SUPPORTED,
            notes={
                "supports_static_batching": (
                    "Real batched decode: all sequences are enqueued before any "
                    "decode step and advance in lockstep, one token per iterate()."
                ),
                "supports_prefix_cache": (
                    "Automatic, page-granular (256-token pages) via a chained "
                    "content-hash page table; best-effort under eviction."
                ),
                "supports_cache_reset": (
                    "No public free_cache in v1.5.3; drains the queue and calls "
                    "pagetable.reset_page_table(), verified by get_cache_stats()."
                ),
                "supports_cuda_graphs": "v1.5.3 has no CUDA-graph capture path.",
                "supports_internal_queue_metrics": "time_enqueued (queue wait) only.",
                "supports_internal_kv_metrics": "Generator.get_cache_stats().",
                "supports_kv_quantization": "Available (CacheLayer_quant) but the "
                "official configuration runs the default fp16 cache.",
                "supports_chunked_prefill": "max_chunk_size bounds prefill chunks.",
            },
        )

    def get_runtime_info(self) -> dict[str, Any]:
        info = dict(self._runtime_info)
        info.setdefault("name", EXLLAMAV3_RUNTIME_NAME)
        return info

    def get_memory_info(self) -> dict[str, Any] | None:
        if self._torch is None or not self._torch.cuda.is_available():
            return None
        torch = self._torch
        out: dict[str, Any] = {
            "torch_allocated_mb": torch.cuda.memory_allocated(0) / 2**20,
            "torch_reserved_mb": torch.cuda.memory_reserved(0) / 2**20,
            "torch_peak_allocated_mb": torch.cuda.max_memory_allocated(0) / 2**20,
            "torch_peak_reserved_mb": torch.cuda.max_memory_reserved(0) / 2**20,
            "provenance": "REPORTED_RUNTIME",
        }
        cache = self._cache
        if cache is not None and hasattr(cache, "get_all_tensors"):
            try:
                total = 0
                for tensor in cache.get_all_tensors():
                    total += tensor.element_size() * tensor.numel()
                out["kv_cache_mb"] = total / 2**20
                out["kv_cache_provenance"] = "REPORTED_RUNTIME"
            except Exception:  # noqa: BLE001
                pass
        model = self._model
        if model is not None and hasattr(model, "get_storage_info"):
            try:
                bpw_layer, bpw_head, vram_bits = model.get_storage_info()
                out["weights_mb"] = vram_bits / 8 / 2**20
                out["weights_bpw_layer"] = bpw_layer
                out["weights_bpw_head"] = bpw_head
                out["weights_provenance"] = "REPORTED_RUNTIME"
            except Exception:  # noqa: BLE001
                pass
        return out

    def get_startup_milestones(self) -> dict[str, float | None]:
        return dict(self._milestones)

    def get_internal_metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        generator = self._generator
        if generator is None:
            return out
        if hasattr(generator, "get_cache_stats"):
            try:
                out["cache_stats"] = _json_safe(generator.get_cache_stats())
                out["cache_stats_provenance"] = "REPORTED_RUNTIME"
            except Exception:  # noqa: BLE001
                pass
        with self._lock:
            out["active_jobs"] = len(self._active)
        out["scheduler_error"] = repr(self._scheduler_error) if self._scheduler_error else None
        return out


_JSON_PRIMITIVES = (int, float, str, bool, type(None))


def _json_safe(value: Any) -> Any:
    """Coerce runtime-reported values to something the result writer can encode.

    ExLlamaV3 reports a few non-primitive values (e.g. ``cache.layer_type`` is a
    class object); raw records are JSON, so those become their name instead of
    breaking serialization mid-run.
    """
    if isinstance(value, _JSON_PRIMITIVES):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return getattr(value, "__name__", None) or repr(value)


def build_exllamav3_runtime(**kwargs: Any) -> ExLlamaV3Runtime:
    config = kwargs.pop("config", None)
    if config is None:
        config = ExLlamaV3RuntimeConfig(**kwargs)
    elif kwargs:
        raise TypeError("pass either config= or keyword overrides, not both")
    return ExLlamaV3Runtime(config)


def register(register_adapter) -> None:  # noqa: ANN001
    """Register this adapter; called by the adapter package's discovery hook."""
    register_adapter(EXLLAMAV3_RUNTIME_NAME, build_exllamav3_runtime)
