from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

Content = str | Mapping[str, Any] | Sequence[Any]


def render_content(c: Content) -> str:
    """Render to text: strings unchanged, else compact sorted JSON."""
    if isinstance(c, str):
        return c
    return json.dumps(c, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _frozen_mapping(m: Mapping) -> Mapping:
    return MappingProxyType(dict(m))


@dataclass(frozen=True)
class Choice:
    """Pick one of several candidates.

    `criteria` maps candidate name to optional description. A non-str `Sequence`
    of candidate names (e.g. `["billing", "engineering"]`) is also accepted and
    normalized to a mapping of name to `None`.
    """

    instructions: Content
    criteria: Mapping[str, Content | None] | Sequence[str]

    def __post_init__(self) -> None:
        if not self.criteria:
            raise ValueError("Choice.criteria must contain at least one candidate")
        if not isinstance(self.criteria, Mapping):
            if any(not isinstance(c, str) for c in self.criteria):
                raise ValueError("Choice.criteria entries must be strings")
            object.__setattr__(self, "criteria", dict.fromkeys(self.criteria))
        if any(not isinstance(k, str) or not k for k in self.criteria):
            raise ValueError("Choice.criteria keys must be non-empty strings")
        object.__setattr__(self, "criteria", _frozen_mapping(self.criteria))


@dataclass(frozen=True)
class Score:
    """Place the state on an ordered scale. `criteria` lists the levels from lowest to highest."""

    instructions: Content
    criteria: Sequence[Content]

    def __post_init__(self) -> None:
        if len(self.criteria) < 2:
            raise ValueError("Score.criteria must list at least two ordered levels")
        object.__setattr__(self, "criteria", tuple(self.criteria))


@dataclass(frozen=True)
class Noul:
    """Yes/no judgement as probability in [0, 1].

    Optional descriptions for 'true' and 'false'. A 1- or 2-element non-str
    `Sequence` (e.g. `["yes it is", "no it is not"]`) is also accepted and
    normalized positionally: the first element becomes 'true', the second
    (if present) becomes 'false'.
    """

    instructions: Content
    criteria: Mapping[str, Content] | Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.criteria is not None:
            if not isinstance(self.criteria, Mapping):
                if not 1 <= len(self.criteria) <= 2 or any(
                    not isinstance(c, str) for c in self.criteria
                ):
                    raise ValueError(
                        "Noul.criteria as a sequence must be 1 or 2 strings, true then false"
                    )
                keys = ("true", "false")
                object.__setattr__(self, "criteria", dict(zip(keys, self.criteria, strict=False)))
            if set(self.criteria) - {"true", "false"} or not self.criteria:
                raise ValueError("Noul.criteria keys must be 'true' and/or 'false'")
            object.__setattr__(self, "criteria", _frozen_mapping(self.criteria))


Question = Choice | Score | Noul


@dataclass(frozen=True)
class Request:
    state: Content
    questions: Mapping[str, Question]
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.questions or any(not isinstance(k, str) or not k for k in self.questions):
            raise ValueError("Request.questions must be a non-empty mapping with non-empty names")
        object.__setattr__(self, "questions", _frozen_mapping(self.questions))


@dataclass(frozen=True)
class ChoiceAnswer:
    """`probabilities` are the backend's outputs. They are not guarantees of correctness."""

    choice: str
    probabilities: dict[str, float]


@dataclass(frozen=True)
class ScoreAnswer:
    """`score` is the expected level index in [0, len(levels) - 1].

    `probabilities` are the backend's outputs. They are not guarantees of correctness.
    """

    score: float
    probabilities: list[float]
    levels: list[str]


@dataclass(frozen=True)
class NoulAnswer:
    """`noul` is the model output in [0, 1]. It is not a guarantee of correctness."""

    noul: float


Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer


@dataclass(frozen=True)
class Meta:
    backend: str
    model: str | None
    latency_ms: float
    route: list[str] = field(default_factory=list)
    raw: Any = None


@dataclass(frozen=True)
class Response:
    answers: dict[str, Answer]
    meta: Meta

    def __getitem__(self, name: str) -> Answer:
        return self.answers[name]

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, ScoreAnswer)}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {k: v for k, v in self.answers.items() if isinstance(v, NoulAnswer)}

    def with_meta(self, **changes: Any) -> Response:
        return replace(self, meta=replace(self.meta, **changes))
