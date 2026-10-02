"""Adapter registry.

Future runtime adapters register here; the benchmark core stays
framework-neutral. Real-runtime adapters (ExLlamaV3, SGLang, vLLM, ...) are
introduced in their own integration tasks — there are deliberately no stub
files pretending to be working adapters.
"""

from __future__ import annotations

from typing import Any, Callable

from .base import RuntimeAdapter

AdapterFactory = Callable[..., RuntimeAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}


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


def _mock_factory(**kwargs: Any) -> RuntimeAdapter:
    from .mock import MockRuntime, MockRuntimeConfig

    config = kwargs.pop("config", None)
    if config is None:
        config = MockRuntimeConfig(**kwargs)
    elif kwargs:
        raise TypeError("pass either config= or keyword overrides, not both")
    return MockRuntime(config)


register_adapter("mock", _mock_factory)
