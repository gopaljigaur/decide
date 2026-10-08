"""OpenAI Decisions HTTP backend (`POST /v1/decisions`, public beta)."""

from __future__ import annotations

from typing import Any

import httpx

from decide.backends._http import HttpClientMixin, raise_for_status
from decide.backends.base import BaseBackend, Capabilities
from decide.errors import BadResponseError
from decide.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Request,
    Score,
    ScoreAnswer,
    render_content,
)

# Answer `type` per question class, as the API names them.
_ANSWER_TYPES = {Choice: "choice", Score: "score", Noul: "predicate"}


def _number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _question(name: str, question: Question) -> dict[str, Any]:
    instructions = render_content(question.instructions)
    extra: dict[str, Any]
    if isinstance(question, Choice):
        kind = "choice"
        extra = {
            "choices": [
                {"value": k} if v is None else {"value": k, "description": render_content(v)}
                for k, v in question.criteria.items()
            ]
        }
    elif isinstance(question, Score):
        kind = "score"
        extra = {"levels": [{"label": render_content(level)} for level in question.criteria]}
    elif isinstance(question, Noul):
        kind = "predicate"
        extra = {}
        # The API has no slot for the true/false descriptions, so they join the instructions.
        lines = [
            f"If {key}: {render_content(question.criteria[key])}"
            for key in ("true", "false")
            if question.criteria and key in question.criteria
        ]
        if lines:
            instructions += "\n\n" + "\n".join(lines)
    else:
        raise TypeError(f"unknown question type: {type(question)!r}")
    return {"name": name, "type": kind, "instructions": instructions, **extra}


class OpenAIDecisionsBackend(HttpClientMixin, BaseBackend):
    name = "openai_decisions"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    PATH = "/decisions"
    DEFAULT_MODEL = "gpt-6-luna"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self.model = model
        self.timeout = timeout
        self._transport = transport
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities(batch=False, local=False)

    def _resolved_model(self, request: Request) -> str:
        return request.model or self.model or self.DEFAULT_MODEL

    def _body(self, request: Request) -> dict[str, Any]:
        return {
            "model": self._resolved_model(request),
            "input": render_content(request.state),
            "questions": [_question(n, q) for n, q in request.questions.items()],
        }

    def _bad(self, message: str) -> BadResponseError:
        return BadResponseError(self.name, message)

    def _parse_answer(self, name: str, question: Question, answer: dict[str, Any]) -> Answer:
        kind = answer.get("type")
        if kind == "refusal":
            raise self._bad(f"question {name!r}: the model refused to answer")
        expected = _ANSWER_TYPES[type(question)]
        if kind != expected:
            raise self._bad(f"question {name!r}: expected a {expected!r} answer, got {kind!r}")
        if isinstance(question, Noul):
            if not _number(answer.get("probability")):
                raise self._bad(f"question {name!r}: missing or non-numeric 'probability'")
            return NoulAnswer(float(answer["probability"]))
        entries = answer.get("probabilities")
        if not isinstance(entries, list) or not entries:
            raise self._bad(f"question {name!r}: missing or empty 'probabilities'")
        if not all(isinstance(e, dict) and _number(e.get("probability")) for e in entries):
            raise self._bad(
                f"question {name!r}: 'probabilities' entries need a numeric probability"
            )
        if isinstance(question, Choice):
            choice = answer.get("choice")
            if not isinstance(choice, str):
                raise self._bad(f"question {name!r}: missing or non-string 'choice'")
            if not all(isinstance(e.get("value"), str) for e in entries):
                raise self._bad(f"question {name!r}: 'probabilities' entries need a string value")
            return ChoiceAnswer(choice, {e["value"]: float(e["probability"]) for e in entries})
        score = answer.get("score")
        if not _number(score):
            raise self._bad(f"question {name!r}: missing or non-numeric 'score'")
        levels = [render_content(level) for level in question.criteria]
        by_index = {e.get("value"): float(e["probability"]) for e in entries}
        if len(by_index) != len(entries) or set(by_index) != set(range(len(levels))):
            raise self._bad(
                f"question {name!r}: 'probabilities' must cover each of the {len(levels)} "
                "levels exactly once, by zero-based index"
            )
        return ScoreAnswer(float(score), [by_index[i] for i in range(len(levels))], levels)

    def _handle_response(
        self, response: httpx.Response, request: Request
    ) -> tuple[dict[str, Answer], Any, str]:
        raise_for_status(self.name, response)
        try:
            data = response.json()
        except ValueError as exc:
            raise BadResponseError(self.name, "response body was not valid JSON", exc) from exc
        if not isinstance(data, dict) or not isinstance(data.get("answers"), list):
            raise self._bad("response JSON is missing an 'answers' array")
        by_name: dict[str, dict[str, Any]] = {}
        for answer in data["answers"]:
            name = answer.get("name") if isinstance(answer, dict) else None
            if not isinstance(name, str):
                raise self._bad("an answer is not an object with a string 'name'")
            if name not in request.questions:
                raise self._bad(f"answer for unknown question {name!r}")
            if name in by_name:
                raise self._bad(f"more than one answer for question {name!r}")
            by_name[name] = answer
        answers: dict[str, Answer] = {}
        for name, question in request.questions.items():
            if name not in by_name:
                raise self._bad(f"missing answer for question {name!r}")
            answers[name] = self._parse_answer(name, question, by_name[name])
        reported = data.get("model")
        model = reported if isinstance(reported, str) and reported else None
        return answers, data, model or self._resolved_model(request)

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = self._post_json(self.PATH, json=self._body(request), headers=headers)
        return self._handle_response(response, request)

    async def _adecide(self, request: Request) -> tuple[dict[str, Answer], Any, str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = await self._apost_json(self.PATH, json=self._body(request), headers=headers)
        return self._handle_response(response, request)
