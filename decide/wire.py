"""Codec between decide's Request/Response types and TypeSafe's wire JSON shape."""

from __future__ import annotations

from typing import Any

from decide.errors import BadResponseError
from decide.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
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
        atype = a.get("type")

        if isinstance(question, Choice):
            if atype != "choice":
                raise BadResponseError(
                    backend="wire",
                    message=f"question '{name}' expected type 'choice', got {atype!r}",
                )
            probs = dict(a["probabilities"])
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
            probs = list(a["probabilities"])
            score = a.get("score")
            if score is None:
                score = sum(i * p for i, p in enumerate(probs))
            levels = [render_content(level) for level in question.criteria]
            result[name] = ScoreAnswer(score, probs, levels)

        elif isinstance(question, Noul):
            if atype != "noul":
                raise BadResponseError(
                    backend="wire",
                    message=f"question '{name}' expected type 'noul', got {atype!r}",
                )
            result[name] = NoulAnswer(a["noul"])

        else:
            raise TypeError(f"unknown question type: {type(question)!r}")

    return result


def to_wire_answers(resp: Response) -> dict[str, Any]:
    """Render a Response's answers back into TypeSafe's wire JSON shape."""
    wire: dict[str, Any] = {}
    for name, answer in resp.answers.items():
        if isinstance(answer, ChoiceAnswer):
            wire[name] = {
                "type": "choice",
                "choice": answer.choice,
                "probabilities": dict(answer.probabilities),
            }
        elif isinstance(answer, ScoreAnswer):
            wire[name] = {
                "type": "score",
                "score": answer.score,
                "probabilities": list(answer.probabilities),
            }
        elif isinstance(answer, NoulAnswer):
            wire[name] = {"type": "noul", "noul": answer.noul}
        else:
            raise TypeError(f"unknown answer type: {type(answer)!r}")
    return wire
