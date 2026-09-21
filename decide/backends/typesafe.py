"""TypeSafe (Jev) HTTP backend."""

from __future__ import annotations

from typing import Any

import httpx

from decide.backends._http import HttpClientMixin, raise_for_status
from decide.backends.base import BaseBackend, Capabilities
from decide.errors import BadResponseError
from decide.types import Answer, Request
from decide.wire import from_wire_answers, to_wire_request


class TypeSafeBackend(HttpClientMixin, BaseBackend):
    name = "typesafe"
    DEFAULT_BASE_URL = "https://api.typesafe.ai"
    PATH = "/v1/systemone"
    DEFAULT_MODEL = "jev-latest"

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

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _resolved_model(self, request: Request) -> str:
        return request.model or self.model or self.DEFAULT_MODEL

    def _body(self, request: Request) -> dict[str, Any]:
        """Render the wire request body with `model` resolved to
        `request.model or self.model or self.DEFAULT_MODEL`."""
        wire = to_wire_request(request)
        wire["model"] = self._resolved_model(request)
        return wire

    def _handle_response(
        self, response: httpx.Response, request: Request
    ) -> tuple[dict[str, Answer], Any, str]:
        raise_for_status(self.name, response)
        try:
            data = response.json()
        except ValueError as exc:
            raise BadResponseError(self.name, "response body was not valid JSON", exc) from exc
        if not isinstance(data, dict) or "answers" not in data:
            raise BadResponseError(self.name, "response JSON is missing 'answers'")
        answers = from_wire_answers(data["answers"], request)
        # Report the model that actually answered when the backend tells us
        # (SystemOneResponse's `model` "may differ from the alias supplied in
        # the request"), else fall back to the one we requested.
        response_model = data.get("model")
        model = response_model if isinstance(response_model, str) and response_model else None
        return answers, data, model or self._resolved_model(request)

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any, str]:
        body = self._body(request)
        response = self._post_json(self.PATH, json=body, headers=self._headers)
        return self._handle_response(response, request)

    async def _adecide(self, request: Request) -> tuple[dict[str, Answer], Any, str]:
        body = self._body(request)
        response = await self._apost_json(self.PATH, json=body, headers=self._headers)
        return self._handle_response(response, request)
