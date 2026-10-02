"""Pradium Runtime Benchmark CLI.

Usage::

    python -m benchmarks --help
    python -m benchmarks run matrix --runtime mock --quick
    python -m benchmarks run concurrency --runtime mock
    python -m benchmarks run context --runtime mock
    python -m benchmarks run prefix-cache --runtime mock
    python -m benchmarks run scheduler --runtime mock
    python -m benchmarks run startup --runtime mock
    python -m benchmarks run soak --runtime mock --duration 60
    python -m benchmarks report <session-id>
    python -m benchmarks corpus validate
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

from .adapters.registry import available_adapters, create_adapter
from .analysis.aggregate import build_summary
from .analysis.plots import generate_plots
from .analysis.report import render_report
from .core.config import load_config
from .core.environment import collect_environment
from .core.session import Session, default_results_root, make_session_id
from .core.version import BENCHMARK_VERSION, FROZEN_COMPONENTS
from .core.workloads import (
    CONCURRENCY_LEVELS,
    CONTEXT_INPUT_TOKENS,
    MIXED_WORKLOADS,
    PREFIX_REUSE_RATIOS,
    PROFILE_ORDER,
)
from .corpus.materialize import (
    validate_corpus,
    materialize_to_disk,
    write_manifest,
    write_vocab,
)
from .runners.base import RunnerContext
from .runners.batching import run_batching
from .runners.concurrency import run_concurrency
from .runners.context import run_context
from .runners.matrix import run_matrix
from .runners.mixed import run_mixed
from .runners.prefix_cache import run_prefix_cache
from .runners.prompts import PromptProvider
from .runners.scheduler import run_scheduler
from .runners.soak import run_soak
from .runners.startup import run_startup
from .telemetry.sampler import TelemetrySampler

SUITE_NAMES = (
    "matrix",
    "context",
    "concurrency",
    "prefix-cache",
    "scheduler",
    "mixed",
    "startup",
    "batching",
    "soak",
)

QUICK_SUBSETS: dict[str, Any] = {
    "matrix": {"profiles": ("SS", "MM")},
    "context": {"lengths": (128, 1024, 8192)},
    "concurrency": {"profiles": ("SS", "MM"), "levels": (1, 4)},
    "prefix-cache": {"ratios": (0.0, 1.0), "scenarios": ("same_session",)},
    "mixed": {"workloads": ("interactive_burst", "mixed_generation")},
    "startup": {"warm_request_count": 2},
    "batching": {"batch_sizes": (2,)},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=f"Pradium Runtime Benchmark Suite ({BENCHMARK_VERSION})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- run ---
    run = sub.add_parser("run", help="run a benchmark suite")
    run.add_argument("suite", choices=SUITE_NAMES)
    run.add_argument("--runtime", default="mock", help="registered runtime adapter")
    run.add_argument("--quick", action="store_true", help="quick development run (never official)")
    run.add_argument(
        "--mode",
        default="official",
        choices=("quick", "validation", "official"),
        help="benchmark mode / run policy (default: official)",
    )
    run.add_argument("--profiles", help="comma-separated profiles, e.g. SS,MM,LL")
    run.add_argument("--levels", help="comma-separated concurrency levels, e.g. 1,2,4,8")
    run.add_argument("--lengths", help="comma-separated context input lengths")
    run.add_argument("--ratios", help="comma-separated prefix reuse ratios, e.g. 0.0,0.5,1.0")
    run.add_argument("--scenarios", help="comma-separated prefix cache scenarios")
    run.add_argument("--workloads", help="comma-separated mixed workload names")
    run.add_argument("--batch-sizes", help="comma-separated static batch sizes")
    run.add_argument("--duration", type=float, help="soak duration in minutes")
    run.add_argument("--concurrency", type=int, default=1, help="matrix suite concurrency")
    run.add_argument("--tokenizer", default="simple", help="tokenizer spec (simple or hf:<name>)")
    run.add_argument("--model", help="model name for the environment manifest")
    run.add_argument("--model-revision", help="model revision")
    run.add_argument("--quantization", help="quantization label")
    run.add_argument("--bpw", type=float, help="bits per weight")
    run.add_argument("--time-scale", type=float, default=0.1, help="mock runtime time scale")
    run.add_argument("--telemetry-interval", type=int, help="telemetry interval ms (default 100)")
    run.add_argument("--deep-telemetry", action="store_true", help="use deep telemetry interval")
    run.add_argument("--timeout", type=float, help="per-request timeout seconds")
    run.add_argument("--results-dir", help="results root directory")
    run.add_argument("--runtime-config", help="JSON config passed to the adapter")
    run.add_argument("--no-plots", action="store_true", help="skip plot generation")

    # --- report ---
    report = sub.add_parser("report", help="regenerate summary/report/plots from raw data")
    report.add_argument("session", help="session id or session directory")
    report.add_argument("--results-dir", help="results root directory")
    report.add_argument("--no-plots", action="store_true")

    # --- corpus ---
    corpus = sub.add_parser("corpus", help="frozen corpus operations")
    corpus_sub = corpus.add_subparsers(dest="corpus_command", required=True)
    corpus_sub.add_parser("validate", help="validate the frozen corpus hashes")
    mat = corpus_sub.add_parser("materialize", help="materialize exact-length prompts")
    mat.add_argument("--tokenizer", default="simple")
    corpus_sub.add_parser("build-vocab", help="rebuild the frozen vocabulary")
    corpus_sub.add_parser("build", help="rebuild vocabulary + manifest")

    # --- info ---
    sub.add_parser("suites", help="list benchmark suites")
    sub.add_parser("adapters", help="list registered runtime adapters")
    version = sub.add_parser("version", help="show benchmark version and frozen components")
    version.add_argument("--frozen", action="store_true")

    return parser


def _parse_csv(value: str | None, cast=str) -> list[Any] | None:
    if not value:
        return None
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _resolve_mode(args: argparse.Namespace) -> tuple[str, str]:
    """Return ``(benchmark_mode, run_policy)``."""
    run_policy = "quick" if args.quick else args.mode
    benchmark_mode = run_policy
    if args.runtime == "mock":
        # Mock results are always validation data and never performance-valid.
        benchmark_mode = "validation"
    return benchmark_mode, run_policy


def _create_adapter(args: argparse.Namespace):
    extra = json.loads(args.runtime_config) if args.runtime_config else {}
    if args.runtime == "mock":
        extra.setdefault("time_scale", args.time_scale)
        return create_adapter("mock", **extra)
    return create_adapter(args.runtime, **extra)


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config()
    run_policy_name = "quick" if args.quick else args.mode
    policy = config.mode_policy(run_policy_name)
    warmups, measured_runs = policy.warmups, policy.measured_runs
    benchmark_mode, run_policy = _resolve_mode(args)
    performance_valid = (
        policy.performance_valid and args.runtime != "mock" and not args.quick
    )
    quick = run_policy_name == "quick"

    if args.suite == "soak" and quick:
        print("error: soak is excluded from quick mode", file=sys.stderr)
        return 2

    # --- frozen corpus must validate before any run ---
    try:
        corpus_report = validate_corpus()
    except Exception as exc:  # noqa: BLE001
        print(f"corpus validation FAILED — refusing to run:\n{exc}", file=sys.stderr)
        return 1

    adapter = _create_adapter(args)
    provider = PromptProvider.from_spec(args.tokenizer)
    interval_ms = args.telemetry_interval or (
        config.telemetry.deep_interval_ms if args.deep_telemetry else config.telemetry.interval_ms
    )

    session_id = make_session_id(args.runtime, benchmark_mode)
    results_root = Path(args.results_dir) if args.results_dir else default_results_root()
    session = Session(session_id, results_root, args.runtime, benchmark_mode, run_policy)

    runtime_info: dict[str, Any] = {}
    telemetry = TelemetrySampler(interval_ms=interval_ms, deep=args.deep_telemetry)
    telemetry.start()
    suite_extras: dict[str, Any] = {}
    try:
        adapter.initialize(json.loads(args.runtime_config) if args.runtime_config else None)
        try:
            adapter.load_model()
        except Exception as exc:  # noqa: BLE001 - record, keep session artifacts
            print(f"model load failed: {exc}", file=sys.stderr)
            raise
        runtime_info = adapter.get_runtime_info()

        environment = collect_environment(
            cli_args=sys.argv[1:],
            telemetry_interval_ms=interval_ms,
            runtime_info=runtime_info,
        )
        session.write_environment(environment)
        session.write_config(
            {
                **config.to_dict(),
                "cli": {
                    key: value
                    for key, value in vars(args).items()
                    if value is not None
                },
                "warmups": warmups,
                "measured_runs": measured_runs,
            }
        )
        session.write_corpus_manifest(
            {
                "validated": corpus_report,
                "tokenizer": provider.tokenizer_name,
                "tokenizer_revision": provider.tokenizer_revision,
            }
        )

        ctx = RunnerContext(
            adapter=adapter,
            session=session,
            telemetry=telemetry,
            config=config,
            suite=args.suite,
            benchmark_mode=benchmark_mode,
            run_policy=run_policy,
            performance_valid=performance_valid,
            environment=environment,
            model=args.model,
            model_revision=args.model_revision,
            tokenizer=provider.tokenizer_name,
            tokenizer_revision=provider.tokenizer_revision,
            quantization=args.quantization,
            bpw=args.bpw,
            timeout_s=args.timeout,
        )

        suite_extras = _dispatch_suite(ctx, provider, args, policy)
    finally:
        try:
            adapter.shutdown()
        except Exception:  # noqa: BLE001
            pass
        telemetry.stop()

    # --- analysis (never destroys raw data on failure) ---
    _finalize(session, True, suite_extras, environment, adapter, args)
    print(f"session: {session.directory}")
    print(f"raw records: {session.raw_count}")
    return 0


def _dispatch_suite(
    ctx: RunnerContext,
    provider: PromptProvider,
    args: argparse.Namespace,
    policy,
) -> dict[str, Any]:
    quick = args.quick or args.mode == "quick"
    warmups, measured_runs = policy.warmups, policy.measured_runs
    subset = QUICK_SUBSETS if quick else {}
    suite = args.suite
    extras: dict[str, Any] = {}

    profiles = _parse_csv(args.profiles) or list(policy.profiles)
    for name in profiles:
        if name not in PROFILE_ORDER:
            raise SystemExit(f"error: unknown profile {name!r}")

    if suite == "matrix":
        run_matrix(
            ctx,
            provider,
            profiles=profiles,
            warmups=warmups,
            measured_runs=measured_runs,
            concurrency=args.concurrency,
        )
    elif suite == "context":
        lengths = _parse_csv(args.lengths, int) or list(
            subset.get("context", {}).get("lengths", CONTEXT_INPUT_TOKENS)
        )
        run_context(
            ctx,
            provider,
            input_tokens=lengths,
            warmups=warmups,
            measured_runs=measured_runs,
        )
    elif suite == "concurrency":
        levels = _parse_csv(args.levels, int) or list(
            subset.get("concurrency", {}).get("levels", CONCURRENCY_LEVELS)
        )
        run_concurrency(
            ctx,
            provider,
            profiles=profiles,
            levels=levels,
            warmups=warmups,
            measured_runs=measured_runs,
        )
    elif suite == "prefix-cache":
        ratios = _parse_csv(args.ratios, float) or list(
            subset.get("prefix-cache", {}).get("ratios", PREFIX_REUSE_RATIOS)
        )
        scenarios = _parse_csv(args.scenarios) or list(
            subset.get("prefix-cache", {}).get(
                "scenarios", ("same_session", "cross_request", "cross_session")
            )
        )
        _, extras = run_prefix_cache(
            ctx,
            provider,
            ratios=ratios,
            scenarios=scenarios,
            measured_runs=measured_runs,
        )
    elif suite == "scheduler":
        _, extras = run_scheduler(
            ctx,
            provider,
            intruder_delay_ms=ctx.config.scheduler.intruder_delay_ms,
        )
    elif suite == "mixed":
        workloads = _parse_csv(args.workloads) or list(
            subset.get("mixed", {}).get("workloads", tuple(MIXED_WORKLOADS))
        )
        _, extras = run_mixed(
            ctx,
            provider,
            workloads=workloads,
            warmups=warmups,
            measured_runs=measured_runs,
        )
    elif suite == "startup":
        warm_count = subset.get("startup", {}).get(
            "warm_request_count", ctx.config.startup.warm_request_count
        )
        _, extras = run_startup(
            ctx,
            provider,
            request_profile=ctx.config.startup.request_profile,
            warm_request_count=warm_count,
        )
    elif suite == "batching":
        batch_sizes = _parse_csv(args.batch_sizes, int) or list(
            subset.get("batching", {}).get("batch_sizes", ctx.config.batching.batch_sizes)
        )
        _, extras = run_batching(
            ctx,
            provider,
            profile=ctx.config.batching.profile,
            batch_sizes=batch_sizes,
            warmups=min(warmups, 1),
            measured_runs=max(1, measured_runs),
        )
    elif suite == "soak":
        duration_s = (args.duration * 60.0) if args.duration else (
            ctx.config.soak.duration_minutes * 60.0
        )
        _, extras = run_soak(
            ctx,
            provider,
            duration_s=duration_s,
            profiles=ctx.config.soak.profiles,
        )
    else:  # pragma: no cover - argparse guards this
        raise SystemExit(f"error: unknown suite {suite}")
    return extras


def _finalize(
    session: Session,
    allow_plots: bool,
    suite_extras: dict[str, Any],
    environment: dict[str, Any],
    adapter,
    args: argparse.Namespace,
) -> None:
    try:
        capabilities = adapter.get_capabilities().to_dict()
    except Exception:  # noqa: BLE001
        capabilities = {}
    config_payload: dict[str, Any] = {}
    for name in ("config.json", "environment.json"):
        path = session.directory / name
        if path.exists():
            config_payload[name.replace(".json", "")] = json.loads(
                path.read_text(encoding="utf-8")
            )
    summary = build_summary(
        session.directory,
        suite_extras=suite_extras,
        environment=environment or config_payload.get("environment", {}),
        config=config_payload.get("config", {}),
        capabilities=capabilities,
    )
    summary["suite_extras"] = suite_extras
    session.write_summary(summary)
    session.write_report(render_report(summary))
    if allow_plots and not args.no_plots:
        try:
            generate_plots(session.directory, summary, suite_extras)
        except Exception:  # noqa: BLE001 - plots must never destroy results
            print(f"plot generation failed (raw data intact):\n{traceback.format_exc()}")
    session.close()


def cmd_report(args: argparse.Namespace) -> int:
    results_root = Path(args.results_dir) if args.results_dir else default_results_root()
    session_dir = Path(args.session)
    if not session_dir.is_dir():
        session_dir = results_root / args.session
    if not session_dir.is_dir():
        print(f"error: session not found: {args.session}", file=sys.stderr)
        return 1

    existing_summary: dict[str, Any] = {}
    summary_path = session_dir / "summary.json"
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    environment = existing_summary.get("environment", {})
    env_path = session_dir / "environment.json"
    if not environment and env_path.exists():
        environment = json.loads(env_path.read_text(encoding="utf-8"))
    config_payload = existing_summary.get("config", {})
    config_path = session_dir / "config.json"
    if not config_payload and config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))

    summary = build_summary(
        session_dir,
        suite_extras=existing_summary.get("suite_extras", {}),
        environment=environment,
        config=config_payload,
        capabilities=existing_summary.get("capabilities", {}),
    )
    summary["suite_extras"] = existing_summary.get("suite_extras", {})
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    (session_dir / "report.md").write_text(render_report(summary), encoding="utf-8")
    if not args.no_plots:
        try:
            generate_plots(session_dir, summary, summary.get("suite_extras", {}))
        except Exception:  # noqa: BLE001
            print(f"plot generation failed:\n{traceback.format_exc()}")
    print(f"report regenerated: {session_dir / 'report.md'}")
    return 0


def cmd_corpus(args: argparse.Namespace) -> int:
    if args.corpus_command == "validate":
        result = validate_corpus()
        print(f"corpus validation: {result['status']} ({result['sources']} sources)")
        return 0
    if args.corpus_command == "materialize":
        root = materialize_to_disk(args.tokenizer)
        print(f"materialized prompts: {root}")
        return 0
    if args.corpus_command == "build-vocab":
        print(f"vocabulary: {write_vocab()}")
        return 0
    if args.corpus_command == "build":
        write_vocab()
        print(f"manifest: {write_manifest()}")
        return 0
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    if args.command == "corpus":
        return cmd_corpus(args)
    if args.command == "suites":
        for name in SUITE_NAMES:
            print(name)
        return 0
    if args.command == "adapters":
        for name in available_adapters():
            print(name)
        return 0
    if args.command == "version":
        print(BENCHMARK_VERSION)
        if args.frozen:
            print("frozen components (changes require a version bump):")
            for component in FROZEN_COMPONENTS:
                print(f"  - {component}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
