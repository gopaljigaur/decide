from __future__ import annotations

import importlib
import importlib.util

from decide.backends.base import Backend, BaseBackend, Capabilities
from decide.errors import ConfigError

REGISTRY: dict[str, str] = {
    "typesafe": "decide.backends.typesafe:TypeSafeBackend",
    "openrouter": "decide.backends.openrouter:OpenRouterBackend",
    "laya": "decide.backends.laya:LayaBackend",
    "laya_mlx": "decide.backends.laya_mlx:LayaMLXBackend",
    "crossencoder": "decide.backends.crossencoder:CrossEncoderBackend",
    "llm": "decide.backends.llm:LLMBackend",
}

# The extra (if any) that installs the third-party dependency a backend needs.
_EXTRAS: dict[str, str | None] = {
    "typesafe": None,
    "openrouter": None,
    "laya": "pydecide[laya]",
    "laya_mlx": "pydecide[mlx]",
    "crossencoder": "pydecide[st]",
    "llm": None,
}

# The third-party package whose presence determines whether a backend is importable.
_REQUIRES: dict[str, str] = {
    "typesafe": "httpx",
    "openrouter": "httpx",
    "laya": "laya",
    "laya_mlx": "laya_mlx",
    "crossencoder": "sentence_transformers",
    "llm": "httpx",
}


def load(name: str, **kwargs: object) -> Backend:
    """Load and instantiate a backend by its registry name, forwarding kwargs untouched."""
    target = REGISTRY.get(name)
    if target is None:
        raise ConfigError(f"unknown backend {name!r}; available: {sorted(REGISTRY)}")

    module_path, _, class_name = target.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        extra = _EXTRAS.get(name)
        if extra:
            message = (
                f"backend {name!r} requires the {extra!r} extra; install it to use this backend"
            )
        else:
            message = f"backend {name!r} is not available: {module_path} could not be imported"
        raise ConfigError(message) from exc

    cls = getattr(module, class_name)
    return cls(**kwargs)


def available() -> dict[str, bool]:
    """Report, for each registered backend, whether its dependency is importable."""
    return {name: importlib.util.find_spec(pkg) is not None for name, pkg in _REQUIRES.items()}


__all__ = ["REGISTRY", "Backend", "BaseBackend", "Capabilities", "available", "load"]
