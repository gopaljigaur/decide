"""Alternatives to thresholding the top probability.

`Gate` escalates when the winning probability falls below a number you pick.
That is the cascade baseline, and the published work on cascades keeps finding
the same two weaknesses in it: the probability it reads is usually not
calibrated, so the number you pick does not mean what it looks like, and the
top probability alone throws away the shape of the rest of the distribution.

Each policy here is a drop-in for `Gate` in `Client(policy=...)`:

- `CalibratedGate` rescales the probabilities first, so a threshold of 0.9 is
  closer to being right 90% of the time. Fit the temperature with
  `decide.calibration.fit_temperature`.
- `EntropyGate` reads the whole distribution instead of its peak, so a close
  three-way split escalates even when the winner clears a threshold.
- `MarginGate` reads the gap between first and second place, which is the
  quantity that decides whether the winner would change.
- `ConformalGate` escalates whenever more than one candidate survives the cut
  fitted by `decide.calibration.fit_conformal_quantile`, which ties escalation
  to a coverage rate you choose instead of a threshold you guess.

Score answers are never gated by any of these, for the same reason `Gate`
skips them: an expected value over ordered levels has no single confidence
behind it.
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
    """What `Client` needs from a gating policy.

    `Gate` and every policy in this module satisfy it.
    """

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
    """Shannon entropy over `probabilities`, divided by the entropy of a flat one.

    0.0 means all the mass sits on one candidate, 1.0 means they are tied.
    """
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
    """`Gate`, but on probabilities rescaled by a fitted temperature.

    Fit `temperatures` with `decide.calibration.fit_temperature` on answers
    whose correct label you know. A temperature above 1 means the backend was
    overconfident, so its probabilities are pulled down before the threshold
    reads them, and more questions escalate than `Gate` would have escalated at
    the same number.
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

    `max_entropy` is on a normalized scale: 0.0 accepts only a distribution with
    all its mass on one candidate, 1.0 accepts anything. Two answers with the
    same winning probability can differ here, because one may have a clear
    runner-up and the other a three-way tie.
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

    The margin is what decides whether the winner would change, so a 0.51 that
    beat a 0.02 runner-up is a different answer from a 0.51 that beat a 0.49,
    and `Gate` cannot tell them apart.
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

    Fit `quantile` with `decide.calibration.fit_conformal_quantile` at the
    coverage you want. The prediction set is every candidate whose probability
    is at least `1 - quantile`; the answer is accepted when that set holds
    exactly one candidate, and escalated when it holds more.

    This ties escalation to a coverage rate measured on your own data instead of
    a threshold picked by feel. The guarantee is exchangeability: it holds for
    inputs that look like the ones you fitted on.
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
        """How many candidates survive the cut, or None when the answer is not gated."""
        probabilities = _distribution(answer)
        if probabilities is None:
            return None
        floor = 1.0 - self._quantile_for(name)
        return max(1, sum(1 for p in probabilities if p >= floor))

    def confidence(self, response: Response) -> dict[str, float]:
        """One over the set size, so a single surviving candidate scores 1.0."""
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
