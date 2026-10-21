"""One configuration format for Python clients and interactive CLI setup."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from decide.backends import REGISTRY
from decide.errors import ConfigError

DEFAULT_CONFIG = Path("decide.json")


@dataclass(frozen=True)
class Config:
    """An ordered backend chain and its defaults; empty backends discover local models.

    `options` maps backend names to constructor arguments, such as `model` or
    `base_url`. Credentials normally come from the environment, not the file.
    """

    backends: Sequence[str] = ()
    options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    min_confidence: float = 0.0
    on_low_confidence: Literal["accept", "raise"] = "accept"

    def __post_init__(self) -> None:
        if isinstance(self.backends, str) or not isinstance(self.backends, (list, tuple)):
            raise ConfigError("backends must be a list of backend names")
        if any(not isinstance(name, str) or name not in REGISTRY for name in self.backends):
            raise ConfigError(f"backends must contain names from {sorted(REGISTRY)}")
        object.__setattr__(self, "backends", tuple(dict.fromkeys(self.backends)))
        if not isinstance(self.options, Mapping) or any(
            name not in self.backends
            or not isinstance(values, Mapping)
            or any(not isinstance(key, str) for key in values)
            for name, values in self.options.items()
        ):
            raise ConfigError("options must map selected backend names to objects")
        if (
            isinstance(self.min_confidence, bool)
            or not isinstance(self.min_confidence, (int, float))
            or not math.isfinite(self.min_confidence)
            or not 0 <= self.min_confidence <= 1
        ):
            raise ConfigError("min_confidence must be a number between 0 and 1")
        if self.on_low_confidence not in ("accept", "raise"):
            raise ConfigError("on_low_confidence must be 'accept' or 'raise'")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> Config:
        """Read a JSON config, reporting malformed or unreadable files as ConfigError."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("config must be a JSON object")
        unknown = data.keys() - {"backends", "options", "min_confidence", "on_low_confidence"}
        if unknown:
            raise ConfigError(f"unknown config fields: {sorted(unknown)}")
        return cls(**data)

    def save(self, path: str | Path = DEFAULT_CONFIG, *, overwrite: bool = False) -> None:
        """Write JSON; existing files are preserved unless overwrite is explicit."""
        data = {
            "backends": list(self.backends),
            "options": {name: dict(values) for name, values in self.options.items()},
            "min_confidence": self.min_confidence,
            "on_low_confidence": self.on_low_confidence,
        }
        try:
            payload = json.dumps(data, indent=2, allow_nan=False) + "\n"
            with Path(path).open("w" if overwrite else "x", encoding="utf-8") as stream:
                stream.write(payload)
        except (OSError, TypeError, ValueError) as exc:
            raise ConfigError(f"cannot write config {path}: {exc}") from exc
