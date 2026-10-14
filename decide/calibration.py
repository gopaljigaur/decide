"""Fit gate parameters on answers whose correct label you know.

Both fitters take the same rows: the probabilities a backend reported, and the
index of the label that was right. Pure Python, so core install gains nothing.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

__all__ = [
    "Calibration",
    "fit_conformal_quantile",
    "fit_temperature",
    "temper",
]

_EPSILON = 1e-12
_MIN_TEMPERATURE = 0.05
_MAX_TEMPERATURE = 10.0


def temper(probabilities: Sequence[float], temperature: float) -> list[float]:
    """Rescale `probabilities` by `temperature` and renormalize.

    Backends hand us probabilities, not logits, so `log(p)` stands in for the
    logit. That is exact whenever the probabilities came from a softmax.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = [math.log(max(p, _EPSILON)) / temperature for p in probabilities]
    ceiling = max(logits)
    weights = [math.exp(x - ceiling) for x in logits]
    total = sum(weights)
    if total <= 0:  # pragma: no cover - only reachable with a degenerate input
        return [1.0 / len(probabilities)] * len(probabilities)
    return [w / total for w in weights]


def _nll(rows: Sequence[tuple[Sequence[float], int]], temperature: float) -> float:
    total = 0.0
    for probabilities, correct in rows:
        scaled = temper(probabilities, temperature)
        total -= math.log(max(scaled[correct], _EPSILON))
    return total / len(rows)


def fit_temperature(
    rows: Sequence[tuple[Sequence[float], int]],
    *,
    tolerance: float = 1e-4,
) -> float:
    """The temperature that best explains `rows`, by negative log likelihood.

    Above 1 means the backend was overconfident on this data. Ternary search,
    which suffices because the objective is unimodal in temperature.
    """
    if not rows:
        raise ValueError("fit_temperature needs at least one row")
    for probabilities, correct in rows:
        if not 0 <= correct < len(probabilities):
            raise ValueError(f"correct index {correct} outside {len(probabilities)} probabilities")

    low, high = _MIN_TEMPERATURE, _MAX_TEMPERATURE
    while high - low > tolerance:
        third = (high - low) / 3
        left, right = low + third, high - third
        if _nll(rows, left) < _nll(rows, right):
            high = right
        else:
            low = left
    return round((low + high) / 2, 4)


def fit_conformal_quantile(
    rows: Sequence[tuple[Sequence[float], int]],
    *,
    alpha: float = 0.1,
) -> float:
    """The split-conformal cut for `1 - alpha` coverage of the true label.

    Scores `1 - p(true)` at the `ceil((n + 1)(1 - alpha)) / n` quantile; the
    prediction set is every label at or above `1 - quantile`. Holds for inputs
    drawn like the calibration rows, and says nothing about another workload.
    """
    if not rows:
        raise ValueError("fit_conformal_quantile needs at least one row")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    scores = sorted(1.0 - probabilities[correct] for probabilities, correct in rows)
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:  # too few rows to promise this coverage, so admit everything
        return 1.0
    return round(scores[rank - 1], 6)


@dataclass(frozen=True)
class Calibration:
    """Fitted gate parameters per question. Missing names fall back to neutral defaults."""

    temperatures: dict[str, float] = field(default_factory=dict)
    quantiles: dict[str, float] = field(default_factory=dict)
    note: str = ""

    def temperature_for(self, question: str) -> float:
        return self.temperatures.get(question, 1.0)

    def quantile_for(self, question: str) -> float:
        return self.quantiles.get(question, 1.0)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Calibration:
        data = json.loads(text)
        return cls(
            temperatures={str(k): float(v) for k, v in data.get("temperatures", {}).items()},
            quantiles={str(k): float(v) for k, v in data.get("quantiles", {}).items()},
            note=str(data.get("note", "")),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> Calibration:
        return cls.from_json(Path(path).read_text())
