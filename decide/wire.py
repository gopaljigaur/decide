"""Codec between decide's Request/Response types and TypeSafe's wire JSON shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from decide.errors import BadResponseError
from decide.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Request,
    Response,
    Score,
    ScoreAnswer,
    render_content,
)


def _wire_question(question: Choice | Score | Noul) -> dict[str, Any]:
    if isinstance(question, Choice):
        return {
            "type": "choice",
            "instructions": question.instructions,
            "criteria": dict(question.criteria),
        }
    if isinstance(question, Score):
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": list(question.criteria),
        }
    if isinstance(question, Noul):
        w: dict[str, Any] = {"type": "noul", "instructions": question.instructions}
        if question.criteria is not None:
            w["criteria"] = dict(question.criteria)
        return w
    raise TypeError(f"unknown question type: {type(question)!r}")


def to_wire_request(req: Request) -> dict[str, Any]:
    """Render a Request into TypeSafe's wire JSON shape."""
    wire: dict[str, Any] = {"state": req.state}
    if req.model is not None:
        wire["model"] = req.model
    wire["questions"] = {name: _wire_question(q) for name, q in req.questions.items()}
    return wire


def _parse_wire_question(name: str, wire: Any) -> Question:
    """Parse one wire question dict into a Choice/Score/Noul, the inverse of `_wire_question`."""
    if not isinstance(wire, Mapping):
        raise ValueError(f"question '{name}' must be a mapping")

    qtype = wire.get("type")

    if qtype == "choice":
        if "instructions" not in wire:
            raise ValueError(f"question '{name}' is missing 'instructions'")
        if "criteria" not in wire:
            raise ValueError(f"question '{name}' is missing 'criteria'")
        criteria = wire["criteria"]
        if not isinstance(criteria, Mapping):
            raise ValueError(
                f"question '{name}' criteria must be a mapping of candidate name to "
                "description for type 'choice'"
            )
        return Choice(wire["instructions"], dict(criteria))

    if qtype == "score":
        if "instructions" not in wire:
            raise ValueError(f"question '{name}' is missing 'instructions'")
        if "criteria" not in wire:
            raise ValueError(f"question '{name}' is missing 'criteria'")
        criteria = wire["criteria"]
        if isinstance(criteria, str | bytes) or not isinstance(criteria, Sequence):
            raise ValueError(
                f"question '{name}' criteria must be an ordered list of levels for type 'score'"
            )
        return Score(wire["instructions"], list(criteria))

    if qtype == "noul":
        if "instructions" not in wire:
            raise ValueError(f"question '{name}' is missing 'instructions'")
        criteria = wire.get("criteria")
        if criteria is None:
            return Noul(wire["instructions"])
        if not isinstance(criteria, Mapping):
            raise ValueError(
                f"question '{name}' criteria must be a mapping with 'true'/'false' keys "
                "for type 'noul'"
            )
        return Noul(wire["instructions"], dict(criteria))

    raise ValueError(f"question '{name}' has unknown type {qtype!r}")


def parse_wire_request(body: dict) -> Request:
    """Parse a TypeSafe wire request JSON body into a Request, the inverse of `to_wire_request`.

    Raises `ValueError` (never `KeyError`/`TypeError`) with a message naming the
    offending field when the body doesn't have the shape `to_wire_request` would
    have produced: a missing or non-mapping `questions`, a question with an
    unknown `type`, a question missing `instructions`/`criteria`, or a `criteria`
    container of the wrong shape for its question type.
    """
    if not isinstance(body, Mapping):
        raise ValueError("request body must be a mapping")

    if "state" not in body:
        raise ValueError("request body is missing 'state'")
    state = body["state"]

    model = body.get("model")
    if model is not None and not isinstance(model, str):
        raise ValueError("request 'model' must be a string")

    if "questions" not in body:
        raise ValueError("request body is missing 'questions'")
    raw_questions = body["questions"]
    if not isinstance(raw_questions, Mapping):
        raise ValueError("request 'questions' must be a mapping")
    if not raw_questions:
        raise ValueError("request 'questions' must not be empty")

    questions = {name: _parse_wire_question(name, wire) for name, wire in raw_questions.items()}

    try:
        return Request(state=state, questions=questions, model=model)
    except ValueError as exc:
        raise ValueError(f"invalid request: {exc}") from exc


def _get_field(a: Mapping[str, Any], key: str, name: str) -> Any:
    """Fetch `a[key]`, raising BadResponseError (never KeyError) if it is absent."""
    try:
        return a[key]
    except KeyError:
        raise BadResponseError(
            backend="wire", message=f"question '{name}' answer is missing field '{key}'"
        ) from None


def _numeric(value: Any, name: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BadResponseError(
            backend="wire",
            message=f"question '{name}' field '{field}' must be numeric, got {value!r}",
        )
    return float(value)


def _numeric_mapping(value: Any, name: str, field: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise BadResponseError(
            backend="wire", message=f"question '{name}' field '{field}' must be a mapping"
        )
    return {k: _numeric(v, name, f"{field}.{k}") for k, v in value.items()}


def _indexed_numeric_mapping(value: Any, name: str, field: str, n: int) -> list[float]:
    """Parse a `{"0": ..., "1": ..., ...}`-style mapping into an ordered `list[float]`.

    TypeSafe's `SystemOneResponse.ScoreAnswer.probabilities` keys each score
    level by its stringified index rather than using a JSON array (its sibling
    `legend` field uses the same index keys, but this function only parses
    `probabilities`; `from_wire_answers` never reads `legend` -- levels come
    from the question's own `criteria`). A missing index defaults to 0.0, and
    a key that isn't one of the expected indices is ignored, so a sparse or
    over-complete mapping from the backend is tolerated.
    """
    if not isinstance(value, Mapping):
        raise BadResponseError(
            backend="wire", message=f"question '{name}' field '{field}' must be a mapping"
        )
    result = []
    for i in range(n):
        key = str(i)
        if key in value:
            result.append(_numeric(value[key], name, f"{field}.{i}"))
        else:
            result.append(0.0)
    return result


def from_wire_answers(answers: dict[str, Any], req: Request) -> dict[str, Answer]:
    """Parse a backend's wire answers into decide Answer objects, validated against `req`."""
    unknown = set(answers) - set(req.questions)
    if unknown:
        raise BadResponseError(
            backend="wire", message=f"unknown questions in answer: {sorted(unknown)}"
        )

    result: dict[str, Answer] = {}
    for name, question in req.questions.items():
        if name not in answers:
            raise BadResponseError(backend="wire", message=f"missing answer for question '{name}'")
        a = answers[name]
        if not isinstance(a, Mapping):
            raise BadResponseError(
                backend="wire", message=f"answer for question '{name}' must be a mapping"
            )
        atype = a.get("type")

        if isinstance(question, Choice):
            if atype != "choice":
                raise BadResponseError(
                    backend="wire",
                    message=f"question '{name}' expected type 'choice', got {atype!r}",
                )
            probs = _numeric_mapping(_get_field(a, "probabilities", name), name, "probabilities")
            choice = a.get("choice")
            if choice is None:
                choice = max(probs, key=probs.get)
            elif choice not in question.criteria:
                raise BadResponseError(
                    backend="wire",
                    message=f"question '{name}' choice {choice!r} is not among the candidates",
                )
            result[name] = ChoiceAnswer(choice, probs)

        elif isinstance(question, Score):
            if atype != "score":
                raise BadResponseError(
                    backend="wire",
                    message=f"question '{name}' expected type 'score', got {atype!r}",
                )
            probs = _indexed_numeric_mapping(
                _get_field(a, "probabilities", name), name, "probabilities", len(question.criteria)
            )
            score = a.get("score")
            if score is None:
                score = sum(i * p for i, p in enumerate(probs))
            else:
                score = _numeric(score, name, "score")
            levels = [render_content(level) for level in question.criteria]
            result[name] = ScoreAnswer(score, probs, levels)

        elif isinstance(question, Noul):
            if atype != "noul":
                raise BadResponseError(
                    backend="wire",
                    message=f"question '{name}' expected type 'noul', got {atype!r}",
                )
            result[name] = NoulAnswer(_numeric(_get_field(a, "noul", name), name, "noul"))

        else:
            raise TypeError(f"unknown question type: {type(question)!r}")

    return result


def to_wire_answers(resp: Response) -> dict[str, Any]:
    """Render a Response's answers back into TypeSafe's wire JSON shape.

    `confidence` (Choice, Score) and `legend` (Score) are required by TypeSafe's
    `SystemOneResponse.Answer` schema but aren't stored on `ChoiceAnswer`/
    `ScoreAnswer`, so they're derived here: a Choice's confidence is the
    probability of its own `choice` (which may differ from the probability
    argmax -- `from_wire_answers` accepts an explicit `choice` field even when
    it isn't the top-probability candidate); a Score's confidence is its top
    level probability, in the same "max probability" spirit as
    `Gate.confidence`'s Choice/Noul derivation.
    """
    wire: dict[str, Any] = {}
    for name, answer in resp.answers.items():
        if isinstance(answer, ChoiceAnswer):
            if not answer.probabilities:
                raise BadResponseError(
                    backend="wire", message=f"answer '{name}' has no probabilities"
                )
            if answer.choice not in answer.probabilities:
                raise BadResponseError(
                    backend="wire",
                    message=(
                        f"answer '{name}' choice {answer.choice!r} is not among "
                        "its own probabilities"
                    ),
                )
            wire[name] = {
                "type": "choice",
                "choice": answer.choice,
                "confidence": answer.probabilities[answer.choice],
                "probabilities": dict(answer.probabilities),
            }
        elif isinstance(answer, ScoreAnswer):
            if not answer.probabilities:
                raise BadResponseError(
                    backend="wire", message=f"answer '{name}' has no probabilities"
                )
            wire[name] = {
                "type": "score",
                "score": answer.score,
                "confidence": max(answer.probabilities),
                "legend": {str(i): level for i, level in enumerate(answer.levels)},
                "probabilities": {str(i): p for i, p in enumerate(answer.probabilities)},
            }
        elif isinstance(answer, NoulAnswer):
            wire[name] = {"type": "noul", "noul": answer.noul}
        else:
            raise TypeError(f"unknown answer type: {type(answer)!r}")
    return wire
