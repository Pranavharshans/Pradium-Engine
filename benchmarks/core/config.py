"""Benchmark configuration loading and validation.

Benchmark definitions live in version-controlled YAML files under
``benchmarks/configs/`` — never as magic numbers scattered through code. The
loader validates that frozen workload definitions still match
PRADIUM-RUNTIME-BENCH-v1; editing them without a version bump fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import yaml_min
from .workloads import (
    BATCH_SIZES,
    CONCURRENCY_LEVELS,
    CONTEXT_INPUT_TOKENS,
    CONTEXT_OUTPUT_TOKENS,
    INPUT_TOKENS,
    OUTPUT_TOKENS,
    PREFIX_REUSE_RATIOS,
    PROFILE_ORDER,
)

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class ConfigError(ValueError):
    """Raised when benchmark configuration is invalid."""


@dataclass(frozen=True)
class RunPolicy:
    warmups: int
    measured_runs: int

    def to_dict(self) -> dict[str, int]:
        return {"warmups": self.warmups, "measured_runs": self.measured_runs}


@dataclass(frozen=True)
class ModePolicy:
    name: str
    warmups: int
    measured_runs: int
    profiles: tuple[str, ...]
    performance_valid: bool
    description: str = ""

    def run_policy(self) -> RunPolicy:
        return RunPolicy(self.warmups, self.measured_runs)


@dataclass(frozen=True)
class MatrixConfig:
    inputs: dict[str, int]
    outputs: dict[str, int]
    profiles: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "profiles": list(self.profiles),
        }


@dataclass(frozen=True)
class ContextConfig:
    input_tokens: tuple[int, ...]
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {"input_tokens": list(self.input_tokens), "output_tokens": self.output_tokens}


@dataclass(frozen=True)
class ConcurrencyConfig:
    levels: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"levels": list(self.levels)}


@dataclass(frozen=True)
class PrefixCacheConfig:
    ratios: tuple[float, ...]
    total_input_tokens: int
    requests_per_ratio: int
    unique_suffix_tokens_at_full_reuse: int
    scenarios: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratios": list(self.ratios),
            "total_input_tokens": self.total_input_tokens,
            "requests_per_ratio": self.requests_per_ratio,
            "unique_suffix_tokens_at_full_reuse": self.unique_suffix_tokens_at_full_reuse,
            "scenarios": list(self.scenarios),
        }


@dataclass(frozen=True)
class SchedulerConfig:
    background_requests: int
    background_profile: str
    intruder_profile: str
    intruder_delay_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "background_requests": self.background_requests,
            "background_profile": self.background_profile,
            "intruder_profile": self.intruder_profile,
            "intruder_delay_ms": self.intruder_delay_ms,
        }


@dataclass(frozen=True)
class SoakConfig:
    duration_minutes: int
    profiles: tuple[str, ...]
    poll_interval_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_minutes": self.duration_minutes,
            "profiles": list(self.profiles),
            "poll_interval_ms": self.poll_interval_ms,
        }


@dataclass(frozen=True)
class StartupConfig:
    warm_request_count: int
    request_profile: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "warm_request_count": self.warm_request_count,
            "request_profile": self.request_profile,
        }


@dataclass(frozen=True)
class BatchingConfig:
    batch_sizes: tuple[int, ...]
    profile: str

    def to_dict(self) -> dict[str, Any]:
        return {"batch_sizes": list(self.batch_sizes), "profile": self.profile}


@dataclass(frozen=True)
class TelemetryConfig:
    interval_ms: int
    deep_interval_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "interval_ms": self.interval_ms,
            "deep_interval_ms": self.deep_interval_ms,
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    matrix: MatrixConfig
    context: ContextConfig
    concurrency: ConcurrencyConfig
    prefix_cache: PrefixCacheConfig
    scheduler: SchedulerConfig
    soak: SoakConfig
    startup: StartupConfig
    batching: BatchingConfig
    telemetry: TelemetryConfig
    modes: dict[str, ModePolicy] = field(default_factory=dict)
    configs_dir: Path = CONFIGS_DIR

    def mode_policy(self, name: str) -> ModePolicy:
        if name not in self.modes:
            raise ConfigError(f"unknown benchmark mode: {name!r}")
        return self.modes[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": self.matrix.to_dict(),
            "context": self.context.to_dict(),
            "concurrency": self.concurrency.to_dict(),
            "prefix_cache": self.prefix_cache.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "soak": self.soak.to_dict(),
            "startup": self.startup.to_dict(),
            "batching": self.batching.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "modes": {k: v.run_policy().to_dict() for k, v in self.modes.items()},
        }


#: Size-class aliases accepted in configuration files.
_CLASS_ALIASES: dict[str, str] = {
    "short": "S",
    "medium": "M",
    "long": "L",
    "s": "S",
    "m": "M",
    "l": "L",
}


def _class_key(name: Any) -> str:
    key = str(name)
    return _CLASS_ALIASES.get(key.lower(), key)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing key {key!r} in {where}")
    return mapping[key]


def _require_int(value: Any, where: str, minimum: int = 1) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{where} must be an integer >= {minimum}, got {value!r}")
    return value


def load_config(configs_dir: Path | str | None = None) -> BenchmarkConfig:
    """Load and validate all benchmark configuration files."""
    root = Path(configs_dir) if configs_dir else CONFIGS_DIR
    if not root.is_dir():
        raise ConfigError(f"config directory not found: {root}")

    def load(name: str) -> dict[str, Any]:
        data = yaml_min.load_file(root / name)
        if not isinstance(data, dict):
            raise ConfigError(f"{name} must contain a mapping")
        return data

    matrix_raw = load("matrix.yaml")
    context_raw = load("context.yaml")
    concurrency_raw = load("concurrency.yaml")
    prefix_raw = load("prefix_cache.yaml")
    scheduler_raw = load("scheduler.yaml")
    soak_raw = load("soak.yaml")
    startup_raw = load("startup.yaml")
    batching_raw = load("batching.yaml")
    telemetry_raw = load("telemetry.yaml")
    modes_raw = load("modes.yaml")

    # --- matrix ---
    matrix_block = _require(matrix_raw, "matrix", "matrix.yaml")
    inputs = _require(matrix_block, "inputs", "matrix.yaml#matrix")
    outputs = _require(matrix_block, "outputs", "matrix.yaml#matrix")
    profiles_raw = _require(matrix_block, "profiles", "matrix.yaml#matrix")
    inputs = {_class_key(k): v for k, v in inputs.items()}
    outputs = {_class_key(k): v for k, v in outputs.items()}
    for cls, tokens in inputs.items():
        if tokens != INPUT_TOKENS.get(cls):
            raise ConfigError(
                f"frozen input length changed: {cls}={tokens} (expected "
                f"{INPUT_TOKENS.get(cls)}). A benchmark version bump is required."
            )
    for cls, tokens in outputs.items():
        if tokens != OUTPUT_TOKENS.get(cls):
            raise ConfigError(
                f"frozen output length changed: {cls}={tokens} (expected "
                f"{OUTPUT_TOKENS.get(cls)}). A benchmark version bump is required."
            )
    profiles = tuple(profiles_raw)
    if tuple(p for p in PROFILE_ORDER if p in profiles) != profiles:
        raise ConfigError("matrix profiles must follow the canonical order")
    if set(profiles) != set(PROFILE_ORDER):
        raise ConfigError("matrix profiles must define all nine SS..LL workloads")

    matrix = MatrixConfig(
        inputs={str(k): int(v) for k, v in inputs.items()},
        outputs={str(k): int(v) for k, v in outputs.items()},
        profiles=profiles,
    )

    # --- context ---
    ctx_inputs = tuple(int(x) for x in _require(context_raw, "input_tokens", "context.yaml"))
    if ctx_inputs != tuple(CONTEXT_INPUT_TOKENS):
        raise ConfigError("frozen context input lengths changed; version bump required")
    ctx_output = _require_int(_require(context_raw, "output_tokens", "context.yaml"), "context.yaml#output_tokens")
    if ctx_output != CONTEXT_OUTPUT_TOKENS:
        raise ConfigError("frozen context output length changed; version bump required")
    context = ContextConfig(input_tokens=ctx_inputs, output_tokens=ctx_output)

    # --- concurrency ---
    levels = tuple(
        _require_int(x, "concurrency.yaml#levels") for x in _require(concurrency_raw, "levels", "concurrency.yaml")
    )
    if levels != tuple(CONCURRENCY_LEVELS):
        raise ConfigError("frozen concurrency levels changed; version bump required")
    concurrency = ConcurrencyConfig(levels=levels)

    # --- prefix cache ---
    ratios = tuple(float(x) for x in _require(prefix_raw, "ratios", "prefix_cache.yaml"))
    if ratios != tuple(PREFIX_REUSE_RATIOS):
        raise ConfigError("frozen prefix reuse ratios changed; version bump required")
    prefix_cache = PrefixCacheConfig(
        ratios=ratios,
        total_input_tokens=_require_int(_require(prefix_raw, "total_input_tokens", "prefix_cache.yaml"), "prefix_cache.yaml#total_input_tokens"),
        requests_per_ratio=_require_int(_require(prefix_raw, "requests_per_ratio", "prefix_cache.yaml"), "prefix_cache.yaml#requests_per_ratio"),
        unique_suffix_tokens_at_full_reuse=_require_int(_require(prefix_raw, "unique_suffix_tokens_at_full_reuse", "prefix_cache.yaml"), "prefix_cache.yaml#unique_suffix_tokens_at_full_reuse"),
        scenarios=tuple(_require(prefix_raw, "scenarios", "prefix_cache.yaml")),
    )

    # --- scheduler ---
    scheduler = SchedulerConfig(
        background_requests=_require_int(_require(scheduler_raw, "background_requests", "scheduler.yaml"), "scheduler.yaml#background_requests"),
        background_profile=str(_require(scheduler_raw, "background_profile", "scheduler.yaml")),
        intruder_profile=str(_require(scheduler_raw, "intruder_profile", "scheduler.yaml")),
        intruder_delay_ms=_require_int(_require(scheduler_raw, "intruder_delay_ms", "scheduler.yaml"), "scheduler.yaml#intruder_delay_ms", minimum=0),
    )

    # --- soak ---
    soak = SoakConfig(
        duration_minutes=_require_int(_require(soak_raw, "duration_minutes", "soak.yaml"), "soak.yaml#duration_minutes"),
        profiles=tuple(_require(soak_raw, "profiles", "soak.yaml")),
        poll_interval_ms=_require_int(_require(soak_raw, "poll_interval_ms", "soak.yaml"), "soak.yaml#poll_interval_ms"),
    )

    # --- startup ---
    startup = StartupConfig(
        warm_request_count=_require_int(_require(startup_raw, "warm_request_count", "startup.yaml"), "startup.yaml#warm_request_count"),
        request_profile=str(_require(startup_raw, "request_profile", "startup.yaml")),
    )

    # --- batching ---
    batch_sizes = tuple(
        _require_int(x, "batching.yaml#batch_sizes") for x in _require(batching_raw, "batch_sizes", "batching.yaml")
    )
    if batch_sizes != tuple(BATCH_SIZES):
        raise ConfigError("frozen batch sizes changed; version bump required")
    batching = BatchingConfig(
        batch_sizes=batch_sizes,
        profile=str(_require(batching_raw, "profile", "batching.yaml")),
    )

    # --- telemetry ---
    telemetry = TelemetryConfig(
        interval_ms=_require_int(_require(telemetry_raw, "interval_ms", "telemetry.yaml"), "telemetry.yaml#interval_ms", minimum=10),
        deep_interval_ms=_require_int(_require(telemetry_raw, "deep_interval_ms", "telemetry.yaml"), "telemetry.yaml#deep_interval_ms", minimum=10),
    )

    # --- modes ---
    modes: dict[str, ModePolicy] = {}
    modes_block = _require(modes_raw, "modes", "modes.yaml")
    for name, spec in modes_block.items():
        warmups = _require_int(_require(spec, "warmups", f"modes.yaml#{name}"), f"modes.yaml#{name}#warmups", minimum=0)
        runs = _require_int(_require(spec, "measured_runs", f"modes.yaml#{name}"), f"modes.yaml#{name}#measured_runs")
        mode_profiles = tuple(_require(spec, "profiles", f"modes.yaml#{name}"))
        for p in mode_profiles:
            if p not in PROFILE_ORDER:
                raise ConfigError(f"modes.yaml#{name}: unknown profile {p!r}")
        perf = bool(_require(spec, "performance_valid", f"modes.yaml#{name}"))
        modes[str(name)] = ModePolicy(
            name=str(name),
            warmups=warmups,
            measured_runs=runs,
            profiles=mode_profiles,
            performance_valid=perf,
            description=str(spec.get("description", "")),
        )
    for required in ("quick", "validation", "official"):
        if required not in modes:
            raise ConfigError(f"modes.yaml must define {required!r}")

    config = BenchmarkConfig(
        matrix=matrix,
        context=context,
        concurrency=concurrency,
        prefix_cache=prefix_cache,
        scheduler=scheduler,
        soak=soak,
        startup=startup,
        batching=batching,
        telemetry=telemetry,
        modes=modes,
        configs_dir=root,
    )
    validate_config(config)
    return config


def validate_config(config: BenchmarkConfig) -> None:
    """Cross-field validation of a loaded configuration."""
    if config.matrix.inputs != dict(INPUT_TOKENS):
        raise ConfigError("matrix inputs do not match frozen definitions")
    if config.matrix.outputs != dict(OUTPUT_TOKENS):
        raise ConfigError("matrix outputs do not match frozen definitions")
    if config.telemetry.deep_interval_ms > config.telemetry.interval_ms:
        raise ConfigError("deep telemetry interval must be <= normal interval")
    official = config.modes.get("official")
    if official and (official.warmups, official.measured_runs) != (3, 10):
        raise ConfigError("official mode must use 3 warmups and 10 measured runs")
    quick = config.modes.get("quick")
    if quick and quick.measured_runs < 1:
        raise ConfigError("quick mode must run at least one measured run")
