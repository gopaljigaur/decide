"""LLM emulation backend, for any OpenAI-compatible chat-completions server.

This backend does not call a model purpose-built for structured decisions
(unlike `typesafe`, `openrouter` or `laya`). It asks a general chat model to
self-report a JSON-encoded probability distribution over each question's
answers via ordinary prompting, then parses whatever comes back. This makes
it the **least trustworthy backend** in `decide`: there is no guarantee the
model attends to every candidate, keeps its probabilities well-calibrated,
or even returns valid JSON. Every probability in its answers -- Choice
probabilities, Score level probabilities, Noul confidence -- is the LLM's
own self-reported estimate, not a measured or calibrated confidence, and
should be treated accordingly (e.g. as a fallback behind a `Gate` with other
backends, never as ground truth on its own).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from decide.backends._http import HttpClientMixin, raise_for_status
from decide.backends.base import BaseBackend, Capabilities
from decide.errors import BadResponseError
from decide.types import Answer, Choice, Noul, Request, Score, render_content
from decide.wire import from_wire_answers

SYSTEM_PROMPT = """You answer structured questions about a piece of state by \
estimating probabilities.

Respond with ONLY a single JSON object and nothing else: no prose, no \
markdown code fences, no commentary. The object's keys are exactly the \
question names given in the user message. The value for each key depends on \
that question's type:

- choice: {"probabilities": {"<candidate>": <p>, ...}} with one entry for \
every candidate listed for that question (and no others), where the values \
are your confidence that each candidate is correct and sum to 1.
- score: {"probabilities": [<p for level 0>, <p for level 1>, ...]} one \
probability per level, in the order the levels were given, summing to 1.
- noul: {"noul": <p>} a single probability in [0, 1] that the answer is yes.

Every probability is your own self-reported confidence estimate. It is not \
a guarantee of correctness. Do not add any keys beyond what is described \
above."""


def build_prompt(request: Request) -> str:
    """Render a Request's state and questions into the LLM user message."""
    lines = [f"State:\n{render_content(request.state)}", "", "Questions:"]
    for name, question in request.questions.items():
        if isinstance(question, Choice):
            lines.append(f"- {name} (choice): {render_content(question.instructions)}")
            lines.append("  Candidates:")
            for candidate, description in question.criteria.items():
                if description is None:
                    lines.append(f"    - {candidate}")
                else:
                    lines.append(f"    - {candidate}: {render_content(description)}")
        elif isinstance(question, Score):
            lines.append(f"- {name} (score): {render_content(question.instructions)}")
            lines.append("  Levels, lowest to highest:")
            for i, level in enumerate(question.criteria):
                lines.append(f"    {i}. {render_content(level)}")
        elif isinstance(question, Noul):
            lines.append(f"- {name} (noul): {render_content(question.instructions)}")
            if question.criteria:
                for key, description in question.criteria.items():
                    lines.append(f"    {key}: {render_content(description)}")
        else:
            raise TypeError(f"unknown question type: {type(question)!r}")
    return "\n".join(lines)


def _clamp01(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _normalised_choice_probs(raw: Any, candidates: list[str]) -> dict[str, float]:
    """Clamp to [0, 1], drop unknown candidates, fill missing with 0.0, renormalise to sum 1."""
    values = (
        {c: _clamp01(raw.get(c, 0.0)) for c in candidates}
        if isinstance(raw, dict)
        else {c: 0.0 for c in candidates}
    )
    total = sum(values.values())
    if total > 0:
        return {c: v / total for c, v in values.items()}
    return {c: 1.0 / len(candidates) for c in candidates}


def _normalised_score_probs(raw: Any, n: int) -> dict[str, float]:
    """Clamp a list of `n` probabilities to [0, 1], renormalise to sum 1, and index by string."""
    values = [_clamp01(raw[i]) if isinstance(raw, list) and i < len(raw) else 0.0 for i in range(n)]
    total = sum(values)
    values = [v / total for v in values] if total > 0 else [1.0 / n] * n
    return {str(i): v for i, v in enumerate(values)}


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    return text.strip()


def _to_wire_answers(data: dict[str, Any], request: Request) -> dict[str, Any]:
    wire: dict[str, Any] = {}
    for name, question in request.questions.items():
        raw = data.get(name)
        raw = raw if isinstance(raw, dict) else {}
        if isinstance(question, Choice):
            candidates = list(question.criteria)
            wire[name] = {
                "type": "choice",
                "probabilities": _normalised_choice_probs(raw.get("probabilities"), candidates),
            }
        elif isinstance(question, Score):
            wire[name] = {
                "type": "score",
                "probabilities": _normalised_score_probs(
                    raw.get("probabilities"), len(question.criteria)
                ),
            }
        elif isinstance(question, Noul):
            wire[name] = {"type": "noul", "noul": _clamp01(raw.get("noul", 0.0))}
        else:
            raise TypeError(f"unknown question type: {type(question)!r}")
    return wire


class LLMBackend(HttpClientMixin, BaseBackend):
    """Emulates a decision backend on top of any OpenAI-compatible chat-completions server.

    See the module docstring: this is the least trustworthy backend `decide`
    ships, since it prompts a general chat model rather than calling a
    purpose-built classifier, and every probability in its answers is the
    LLM's own self-reported estimate.
    """

    name = "llm"
    PATH = "/chat/completions"

    def __init__(
        self,
        *,
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
        temperature: float = 0.0,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self._transport = transport
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities(local=False)

    @property
    def _headers(self) -> dict[str, str]:
        if self.api_key is None:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _resolved_model(self, request: Request) -> str:
        return request.model or self.model

    def _messages(self, request: Request) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(request)},
        ]

    def _body(self, request: Request, *, response_format: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._resolved_model(request),
            "messages": self._messages(request),
            "temperature": self.temperature,
        }
        if response_format:
            body["response_format"] = {"type": "json_object"}
        return body

    @staticmethod
    def _rejects_response_format(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        return "response_format" in response.text.lower()

    def _parse_answers(
        self, response: httpx.Response, request: Request
    ) -> tuple[dict[str, Answer], Any, str | None]:
        try:
            data = response.json()
        except ValueError as exc:
            raise BadResponseError(self.name, "response body was not valid JSON", exc) from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BadResponseError(
                self.name, "response JSON is missing choices[0].message.content"
            ) from exc
        try:
            model_json = json.loads(_strip_fences(content))
        except (ValueError, TypeError) as exc:
            raise BadResponseError(self.name, "model content was not valid JSON", exc) from exc
        if not isinstance(model_json, dict):
            raise BadResponseError(self.name, "model content must be a JSON object")
        wire = _to_wire_answers(model_json, request)
        answers = from_wire_answers(wire, request)
        response_model = data.get("model")
        model = response_model if isinstance(response_model, str) and response_model else None
        return answers, data, model

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any, str | None]:
        response = self._post_json(
            self.PATH, json=self._body(request, response_format=True), headers=self._headers
        )
        if self._rejects_response_format(response):
            response = self._post_json(
                self.PATH, json=self._body(request, response_format=False), headers=self._headers
            )
        raise_for_status(self.name, response)
        return self._parse_answers(response, request)

    async def _adecide(self, request: Request) -> tuple[dict[str, Answer], Any, str | None]:
        response = await self._apost_json(
            self.PATH, json=self._body(request, response_format=True), headers=self._headers
        )
        if self._rejects_response_format(response):
            response = await self._apost_json(
                self.PATH, json=self._body(request, response_format=False), headers=self._headers
            )
        raise_for_status(self.name, response)
        return self._parse_answers(response, request)
