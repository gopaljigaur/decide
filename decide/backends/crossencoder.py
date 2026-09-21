"""CrossEncoder backend: local structured judgments on top of a sentence-transformers reranker.

Turns a `Request` (state plus `Choice`/`Score`/`Noul` questions) into `(query, document)` pairs,
scores every pair in a single `CrossEncoder.predict` call and aggregates the raw logits back into
answers:

* `Choice`: one pair per candidate; softmax over the raw scores gives per-candidate probabilities
  and the argmax.
* `Score`: one pair per ordered level; softmax over the levels, then the expected level index gives
  a continuous score in `[0, len(levels) - 1]`.
* `Noul`: sigmoid of the raw score, or of the difference between a "true" and a "false" criterion
  when both are supplied.

Ported from a prototype (`sentence_transformers.cross_encoder.judge.StructuredJudge`); the
request/response vocabulary follows the Jev "System One" API and the templating/aggregation scheme
follows the open KaLM-Jev reference implementation (https://github.com/KaLM-Embedding/KaLM-Jev).

All outputs are *normalized scores*, not calibrated probabilities: a softmax always distributes
mass over the supplied candidates, so a `Choice` still picks a winner when every candidate is
unsuitable, and a `Noul` of 0.9 does not mean the condition holds 90% of the time. Templates are
model-dependent, and a reranker trained on query/passage relevance will follow instructions only
loosely. Calibrate thresholds (or `temperature`) on your own data before relying on the numbers.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from decide.backends.base import BaseBackend, Capabilities
from decide.errors import ConfigError
from decide.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Meta,
    Noul,
    NoulAnswer,
    Request,
    Response,
    Score,
    ScoreAnswer,
    render_content,
)

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.model import CrossEncoder

_DEFAULT_NOUL_TEXT = {
    "true": "The answer to the question is yes.",
    "false": "The answer to the question is no.",
}


def _noul_document(side: Literal["true", "false"], description: str | None) -> str:
    """Document text for one `Noul` side with criteria: the description, or a generic fallback."""
    return description if description is not None else _DEFAULT_NOUL_TEXT[side]


# ----------------------------------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------------------------------


class Template:
    """Turns (state, instructions, candidate) into the `(query, document)` pair the model scores.

    All `state`/`instructions`/`criteria` values are already rendered to strings when these methods
    are called. Subclass to match the prompt format a specific model was trained with, and pass an
    instance as `CrossEncoderBackend(model, template=...)`.
    """

    def choice(
        self, state: str, instructions: str, candidate: str, description: str | None
    ) -> tuple[str, str]:
        """Pair for one `Choice` candidate. `description` is `None` when the candidate has none."""
        raise NotImplementedError

    def score(
        self, state: str, instructions: str, level: str, index: int, n: int
    ) -> tuple[str, str]:
        """Pair for one `Score` level. `index` counts from 0 (lowest) to `n - 1` (highest)."""
        raise NotImplementedError

    def noul(
        self,
        state: str,
        instructions: str,
        side: Literal["true", "false"] | None,
        description: str | None,
    ) -> tuple[str, str]:
        """Pair for a `Noul`. `side` is `None` when the question has no criteria, in which case the
        instructions themselves are judged against the state. Otherwise `description` is the
        rendered "true"/"false" text supplied on the question, or `None` when only the other side
        was given (use a generic fallback)."""
        raise NotImplementedError


class GenericTemplate(Template):
    """Default wording: works reasonably with relevance rerankers (e.g. `cross-encoder/ms-marco-*`).

    The query holds the state followed by the question; the document holds the candidate.
    """

    def _query(self, state: str, instructions: str) -> str:
        return f"{state}\n\nQuestion: {instructions}"

    def choice(
        self, state: str, instructions: str, candidate: str, description: str | None
    ) -> tuple[str, str]:
        document = candidate if description is None else f"{candidate}: {description}"
        return self._query(state, instructions), document

    def score(
        self, state: str, instructions: str, level: str, index: int, n: int
    ) -> tuple[str, str]:
        return self._query(state, instructions), level

    def noul(
        self,
        state: str,
        instructions: str,
        side: Literal["true", "false"] | None,
        description: str | None,
    ) -> tuple[str, str]:
        document = instructions if side is None else _noul_document(side, description)
        return self._query(state, instructions), document


class KaLMJevTemplate(Template):
    """Template modelled after KaLM-Jev (https://github.com/KaLM-Embedding/KaLM-Jev), the reference
    implementation of Jev-style judgments on top of the KaLM-Reranker-V1-R2 models.

    KaLM-Jev feeds the instructions plus a per-question adapter sentence as the task instruction,
    the serialized state as the query and the candidate as the document, using the
    `<Instruct>`/`<Query>`/`<Document>` layout common to instruction-following rerankers. A
    `CrossEncoder` only receives a pair, so the instruction is folded into the query text here. The
    adapter sentences below paraphrase the ones in KaLM-Jev's own `templates.py`.
    """

    choice_adapter = (
        "Decide whether the option named in the Document is a fitting answer to the question "
        "above, given the Query. Judge it by its stated description, not merely by topical overlap."
    )
    score_adapter = (
        "Decide whether the level named in the Document correctly characterizes the Query with "
        "respect to the assessment above. Judge it by its stated criteria, not merely by topical "
        "overlap."
    )
    noul_adapter = (
        "Given the Query, decide whether the criterion named in the Document correctly describes "
        "the answer to the question above. Answer yes when the criterion holds, otherwise no."
    )

    def _instruct_query(self, state: str, instructions: str, adapter: str) -> str:
        return f"<Instruct>: {instructions}\n\n{adapter}\n<Query>: {state}"

    def choice(
        self, state: str, instructions: str, candidate: str, description: str | None
    ) -> tuple[str, str]:
        document = candidate if description is None else f"{candidate}: {description}"
        query = self._instruct_query(state, instructions, self.choice_adapter)
        return query, f"<Document>: {document}"

    def score(
        self, state: str, instructions: str, level: str, index: int, n: int
    ) -> tuple[str, str]:
        query = self._instruct_query(state, instructions, self.score_adapter)
        return query, f"<Document>: {level}"

    def noul(
        self,
        state: str,
        instructions: str,
        side: Literal["true", "false"] | None,
        description: str | None,
    ) -> tuple[str, str]:
        # Without criteria, KaLM-Jev judges the serialized state itself as the sole document.
        document = state if side is None else _noul_document(side, description)
        query = self._instruct_query(state, instructions, self.noul_adapter)
        return query, f"<Document>: {document}"


# ----------------------------------------------------------------------------------------------
# Math helpers (no numeric dependency: plain `math`, element-wise `float()` of model outputs)
# ----------------------------------------------------------------------------------------------


def _softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _to_floats(scores) -> list[float]:
    """Convert model outputs (numpy array, torch tensor, or list) to one float per pair."""
    try:
        return [float(x) for x in scores]
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            "crossencoder model returned more than one score per pair; pass a model with "
            "num_labels=1 (a scalar CrossEncoder), since CrossEncoderBackend needs exactly one "
            "logit per (query, document) pair"
        ) from exc


def _identity_activation():
    """A no-op activation that bypasses the model's own (typically sigmoid) output activation, so
    `predict` returns raw logits. `torch` is only imported here, lazily, so the module stays
    importable without it."""
    try:
        import torch
    except ImportError:
        return lambda x: x
    return torch.nn.Identity()


# ----------------------------------------------------------------------------------------------
# Backend
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Task:
    question: str
    kind: Literal["choice", "score", "noul"]
    key: str
    pair: tuple[str, str]


class CrossEncoderBackend(BaseBackend):
    """Answer `Choice`/`Score`/`Noul` questions locally with a sentence-transformers `CrossEncoder`.

    See the module docstring for the aggregation scheme and its caveats. Outputs are normalized
    scores, not calibrated probabilities.

    Args:
        model: A loaded `CrossEncoder`, or a model name/path to load lazily (requires the
            `pydecide[st]` extra).
        template: How to render pairs for the model. Defaults to `GenericTemplate`; use
            `KaLMJevTemplate` for KaLM-style instruction rerankers, or subclass `Template`.
        temperature: Divides the raw logits before the softmax/sigmoid. Below 1 sharpens the
            distributions, above 1 flattens them. Rerankers differ widely in logit scale, so tune
            this on held-out data. Must be a positive, finite number.
        device: Forwarded to `CrossEncoder(...)` when `model` is a string.
        batch_size: Forwarded to `CrossEncoder.predict(...)`.

    Raises:
        ValueError: `temperature` is not a positive, finite number.
        ConfigError: `model` is a string and `sentence_transformers` cannot be imported.
    """

    name = "crossencoder"

    def __init__(
        self,
        model: CrossEncoder | str,
        *,
        template: Template | None = None,
        temperature: float = 1.0,
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(f"temperature must be a positive finite number, got {temperature!r}")
        if isinstance(model, str):
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ConfigError(
                    "the crossencoder backend needs the 'sentence-transformers' package; "
                    "install it with the 'pydecide[st]' extra"
                ) from exc
            self.model = model
            self._model = CrossEncoder(model, device=device)
        else:
            self.model = getattr(model, "model_name_or_path", None) or type(model).__name__
            self._model = model
        self.template = template if template is not None else GenericTemplate()
        self.temperature = temperature
        self.batch_size = batch_size

    def capabilities(self) -> Capabilities:
        return Capabilities(batch=True, local=True)

    # ---- pair building and aggregation --------------------------------------------------------

    def _compile(self, request: Request) -> list[_Task]:
        state = render_content(request.state)
        tasks: list[_Task] = []
        for name, question in request.questions.items():
            instructions = render_content(question.instructions)
            if isinstance(question, Choice):
                for candidate, description in question.criteria.items():
                    rendered = None if description is None else render_content(description)
                    pair = self.template.choice(state, instructions, candidate, rendered)
                    tasks.append(_Task(name, "choice", candidate, pair))
            elif isinstance(question, Score):
                n = len(question.criteria)
                for index, level in enumerate(question.criteria):
                    rendered_level = render_content(level)
                    pair = self.template.score(state, instructions, rendered_level, index, n)
                    tasks.append(_Task(name, "score", str(index), pair))
            elif isinstance(question, Noul):
                if question.criteria is None:
                    pair = self.template.noul(state, instructions, None, None)
                    tasks.append(_Task(name, "noul", "", pair))
                else:
                    for side in ("true", "false"):
                        description = question.criteria.get(side)
                        rendered = None if description is None else render_content(description)
                        pair = self.template.noul(state, instructions, side, rendered)
                        tasks.append(_Task(name, "noul", side, pair))
            else:
                raise TypeError(
                    f"question {name!r} must be a Choice, Score or Noul, "
                    f"got {type(question).__name__}"
                )
        return tasks

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores = self._model.predict(
            [list(pair) for pair in pairs],
            activation_fn=_identity_activation(),
            batch_size=self.batch_size,
        )
        return _to_floats(scores)

    def _aggregate(
        self, request: Request, tasks: list[_Task], scores: list[float]
    ) -> dict[str, Answer]:
        grouped: dict[str, list[tuple[str, float]]] = {}
        for task, score in zip(tasks, scores, strict=True):
            grouped.setdefault(task.question, []).append((task.key, score))

        answers: dict[str, Answer] = {}
        for name, question in request.questions.items():
            items = grouped[name]
            if isinstance(question, Noul):
                if question.criteria is None:
                    raw = items[0][1]
                else:
                    by_side = dict(items)
                    raw = by_side["true"] - by_side["false"]
                answers[name] = NoulAnswer(noul=_sigmoid(raw / self.temperature))
            elif isinstance(question, Choice):
                keys = [key for key, _ in items]
                probabilities = _softmax([value / self.temperature for _, value in items])
                best = max(range(len(keys)), key=lambda i: probabilities[i])
                answers[name] = ChoiceAnswer(
                    choice=keys[best], probabilities=dict(zip(keys, probabilities, strict=True))
                )
            elif isinstance(question, Score):
                probabilities = _softmax([value / self.temperature for _, value in items])
                score = sum(i * p for i, p in enumerate(probabilities))
                levels = [render_content(level) for level in question.criteria]
                answers[name] = ScoreAnswer(score=score, probabilities=probabilities, levels=levels)
        return answers

    # ---- BaseBackend hooks -----------------------------------------------------------------------

    def _decide(self, request: Request) -> tuple[dict[str, Answer], object, str | None]:
        tasks = self._compile(request)
        pairs = [task.pair for task in tasks]
        scores = self._predict(pairs)
        answers = self._aggregate(request, tasks, scores)
        raw = {"pairs": pairs, "logits": scores}
        return answers, raw, self.model

    def decide_batch(self, requests: Sequence[Request]) -> list[Response]:
        start = time.perf_counter()
        per_request_tasks = [self._compile(request) for request in requests]
        all_pairs = [task.pair for tasks in per_request_tasks for task in tasks]
        all_scores = self._predict(all_pairs)

        responses: list[Response] = []
        offset = 0
        for request, tasks in zip(requests, per_request_tasks, strict=True):
            n = len(tasks)
            request_scores = all_scores[offset : offset + n]
            offset += n
            pairs = [task.pair for task in tasks]
            answers = self._aggregate(request, tasks, request_scores)
            latency_ms = (time.perf_counter() - start) * 1000
            responses.append(
                Response(
                    answers=answers,
                    meta=Meta(
                        backend=self.name,
                        model=self.model,
                        latency_ms=latency_ms,
                        route=[],
                        raw={"pairs": pairs, "logits": request_scores},
                    ),
                )
            )
        return responses
