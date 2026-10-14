"""Gating policies that read more than the winning probability.

Each is a drop-in for `Gate` in `Client(policy=...)`. Score answers are never
gated, for the reason `Gate` documents. See the README for when to pick which.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from decide.calibration import temper
from decide.types import ChoiceAnswer, NoulAnswer, Response, ScoreAnswer

__all__ = [
    "CalibratedGate",
    "ConformalGate",
    "EntropyGate",
    "MarginGate",
    "Policy",
]

_TYPE_NAMES: Mapping[type, str] = {
    ChoiceAnswer: "choice",
    ScoreAnswer: "score",
    NoulAnswer: "noul",
}


@runtime_checkable
class Policy(Protocol):
    """What `Client` needs from a gating policy."""

    on_error: Literal["next", "raise"]

    def confidence(self, response: Response) -> dict[str, float]:
        """Per question, higher means more willing to accept. Score answers are omitted."""

    def passes(self, response: Response) -> tuple[bool, str]:
        """`(True, "ok")`, or `(False, reason)` naming the first question that failed."""


def _distribution(answer: object) -> list[float] | None:
    """The answer as a probability vector, or None when it is not gated."""
    if isinstance(answer, ChoiceAnswer):
        return list(answer.probabilities.values())
    if isinstance(answer, NoulAnswer):
        return [1.0 - answer.noul, answer.noul]
    return None


def _resolve(
    name: str,
    answer: object,
    per_question: Mapping[str, float] | None,
    per_type: Mapping[str, float] | None,
    default: float,
) -> float:
    """The parameter for one question: by name, else by answer type, else the default."""
    if per_question and name in per_question:
        return per_question[name]
    if per_type:
        type_name = _TYPE_NAMES.get(type(answer))
        if type_name is not None and type_name in per_type:
            return per_type[type_name]
    return default


def _check_per_type(per_type: Mapping[str, float] | None, policy: str) -> None:
    if per_type is not None:
        unknown = set(per_type) - {"choice", "score", "noul"}
        if unknown:
            raise ValueError(f"{policy}.per_type has unknown keys: {sorted(unknown)}")


def _normalized_entropy(probabilities: list[float]) -> float:
    """Shannon entropy, divided by that of a flat distribution. 0.0 peaked, 1.0 tied."""
    if len(probabilities) < 2:
        return 0.0
    total = 0.0
    for p in probabilities:
        if p > 0:
            total -= p * math.log(p)
    return min(1.0, total / math.log(len(probabilities)))


def _margin(probabilities: list[float]) -> float:
    """Gap between the winner and the runner-up."""
    if len(probabilities) < 2:
        return 1.0
    first, second = sorted(probabilities, reverse=True)[:2]
    return first - second


@dataclass(frozen=True)
class CalibratedGate:
    """`Gate` on probabilities rescaled by a fitted temperature.

    Fit it with `decide.calibration.fit_temperature`. Above 1 pulls an
    overconfident backend down, so more questions escalate.
    """

    min_confidence: float = 0.0
    temperature: float = 1.0
    temperatures: Mapping[str, float] | None = None
    per_question: Mapping[str, float] | None = None
    per_type: Mapping[str, float] | None = None
    on_error: Literal["next", "raise"] = "next"

    def __post_init__(self) -> None:
        _check_per_type(self.per_type, "CalibratedGate")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")

    def _temperature_for(self, name: str) -> float:
        if self.temperatures and name in self.temperatures:
            return self.temperatures[name]
        return self.temperature

    def confidence(self, response: Response) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, answer in response.answers.items():
            probabilities = _distribution(answer)
            if probabilities is None:
                continue
            out[name] = max(temper(probabilities, self._temperature_for(name)))
        return out

    def passes(self, response: Response) -> tuple[bool, str]:
        scores = self.confidence(response)
        for name, answer in response.answers.items():
            if name not in scores:
                continue
            threshold = _resolve(
                name, answer, self.per_question, self.per_type, self.min_confidence
            )
            if scores[name] < threshold:
                return (False, f"low_confidence:{name}={scores[name]:.2f}<{threshold:.2f}")
        return (True, "ok")


@dataclass(frozen=True)
class EntropyGate:
    """Escalate when the distribution is spread out, not when its peak is low.

    `max_entropy` is normalized: 0.0 accepts only all the mass on one
    candidate, 1.0 accepts anything.
    """

    max_entropy: float = 1.0
    per_question: Mapping[str, float] | None = None
    per_type: Mapping[str, float] | None = None
    on_error: Literal["next", "raise"] = "next"

    def __post_init__(self) -> None:
        _check_per_type(self.per_type, "EntropyGate")
        if not 0.0 <= self.max_entropy <= 1.0:
            raise ValueError(f"max_entropy must be in [0, 1], got {self.max_entropy}")

    def confidence(self, response: Response) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, answer in response.answers.items():
            probabilities = _distribution(answer)
            if probabilities is None:
                continue
            out[name] = 1.0 - _normalized_entropy(probabilities)
        return out

    def passes(self, response: Response) -> tuple[bool, str]:
        for name, answer in response.answers.items():
            probabilities = _distribution(answer)
            if probabilities is None:
                continue
            ceiling = _resolve(name, answer, self.per_question, self.per_type, self.max_entropy)
            entropy = _normalized_entropy(probabilities)
            if entropy > ceiling:
                return (False, f"high_entropy:{name}={entropy:.2f}>{ceiling:.2f}")
        return (True, "ok")


@dataclass(frozen=True)
class MarginGate:
    """Escalate when first and second place are close.

    The margin, not the peak, is what decides whether the winner would change.
    """

    min_margin: float = 0.0
    per_question: Mapping[str, float] | None = None
    per_type: Mapping[str, float] | None = None
    on_error: Literal["next", "raise"] = "next"

    def __post_init__(self) -> None:
        _check_per_type(self.per_type, "MarginGate")
        if not 0.0 <= self.min_margin <= 1.0:
            raise ValueError(f"min_margin must be in [0, 1], got {self.min_margin}")

    def confidence(self, response: Response) -> dict[str, float]:
        out: dict[str, float] = {}
        for name, answer in response.answers.items():
            probabilities = _distribution(answer)
            if probabilities is None:
                continue
            out[name] = _margin(probabilities)
        return out

    def passes(self, response: Response) -> tuple[bool, str]:
        for name, answer in response.answers.items():
            probabilities = _distribution(answer)
            if probabilities is None:
                continue
            floor = _resolve(name, answer, self.per_question, self.per_type, self.min_margin)
            margin = _margin(probabilities)
            if margin < floor:
                return (False, f"low_margin:{name}={margin:.2f}<{floor:.2f}")
        return (True, "ok")


@dataclass(frozen=True)
class ConformalGate:
    """Accept only when one candidate survives the conformal cut.

    Fit `quantile` with `decide.calibration.fit_conformal_quantile`. The
    guarantee is exchangeability: it holds for inputs like the ones fitted on.
    """

    quantile: float = 1.0
    quantiles: Mapping[str, float] | None = None
    on_error: Literal["next", "raise"] = "next"

    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError(f"quantile must be in [0, 1], got {self.quantile}")

    def _quantile_for(self, name: str) -> float:
        if self.quantiles and name in self.quantiles:
            return self.quantiles[name]
        return self.quantile

    def prediction_set_size(self, name: str, answer: object) -> int | None:
        """How many candidates survive the cut, or None when not gated."""
        probabilities = _distribution(answer)
        if probabilities is None:
            return None
        floor = 1.0 - self._quantile_for(name)
        return max(1, sum(1 for p in probabilities if p >= floor))

    def confidence(self, response: Response) -> dict[str, float]:
        """One over the set size, so a single survivor scores 1.0."""
        out: dict[str, float] = {}
        for name, answer in response.answers.items():
            size = self.prediction_set_size(name, answer)
            if size is None:
                continue
            out[name] = 1.0 / size
        return out

    def passes(self, response: Response) -> tuple[bool, str]:
        for name, answer in response.answers.items():
            size = self.prediction_set_size(name, answer)
            if size is None:
                continue
            if size > 1:
                return (False, f"ambiguous:{name}=set_size={size}")
        return (True, "ok")
