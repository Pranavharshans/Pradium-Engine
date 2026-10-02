"""Adapter registry.

Runtime adapters register here; the benchmark core stays framework-neutral.
Real-runtime adapters (ExLlamaV3, SGLang, vLLM, ...) live in their own modules
in this package and are picked up by a framework-neutral discovery hook: any
module in ``benchmarks.adapters`` exposing ``register(register_adapter)`` is
given the chance to register itself. A module whose optional heavy dependency
(e.g. ``torch``/``exllamav3``) is absent is skipped, never fatal — which is why
adapter modules must import their runtime lazily inside functions.

There are deliberately no stub files pretending to be working adapters.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
from typing import Any, Callable

from .base import RuntimeAdapter

AdapterFactory = Callable[..., RuntimeAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}

#: Modules that are infrastructure rather than runtime adapters.
_NON_ADAPTER_MODULES = frozenset({"__init__", "base", "registry"})


class UnknownAdapterError(ValueError):
    """Raised when a runtime adapter name is not registered."""


def register_adapter(name: str, factory: AdapterFactory) -> None:
    """Register a runtime adapter factory under ``name``."""
    _REGISTRY[name] = factory


def create_adapter(name: str, **kwargs: Any) -> RuntimeAdapter:
    """Instantiate a registered runtime adapter."""
    if name not in _REGISTRY:
        raise UnknownAdapterError(
            f"unknown runtime adapter {name!r}; available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name](**kwargs)


def available_adapters() -> list[str]:
    return sorted(_REGISTRY)


def discover_adapters() -> list[str]:
    """Register every sibling adapter module that offers a ``register`` hook.

    Framework-neutral by construction: this function knows nothing about any
    particular runtime. Import failures and failing hooks are contained so a
    missing optional dependency can never break the benchmark, and an adapter
    that raises during discovery never prevents the others from registering.
    """
    registered: list[str] = []
    here = os.path.dirname(os.path.abspath(__file__))
    for info in pkgutil.iter_modules([here]):
        if info.name in _NON_ADAPTER_MODULES or info.name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{__package__}.{info.name}")
        except Exception:  # noqa: BLE001 - optional dependency absent, etc.
            continue
        hook = getattr(module, "register", None)
        if not callable(hook):
            continue
        before = set(_REGISTRY)
        try:
            hook(register_adapter)
        except Exception:  # noqa: BLE001 - a broken adapter must not break the rest
            _REGISTRY_keys = set(_REGISTRY)
            restored = {k: v for k, v in _REGISTRY.items() if k in before}
            _REGISTRY.clear()
            _REGISTRY.update(restored)
            continue
        registered.extend(sorted(set(_REGISTRY) - before))
    return registered


def _mock_factory(**kwargs: Any) -> RuntimeAdapter:
    from .mock import MockRuntime, MockRuntimeConfig

    config = kwargs.pop("config", None)
    if config is None:
        config = MockRuntimeConfig(**kwargs)
    elif kwargs:
        raise TypeError("pass either config= or keyword overrides, not both")
    return MockRuntime(config)


register_adapter("mock", _mock_factory)

#: Adapters registered by discovery at import time (mock is built in).
DISCOVERED_ADAPTERS: list[str] = discover_adapters()
