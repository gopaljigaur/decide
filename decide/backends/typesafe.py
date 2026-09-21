"""TypeSafe (Jev) HTTP backend."""

from __future__ import annotations

import time
from typing import Any

import httpx

from decide.backends.base import BaseBackend, Capabilities
from decide.errors import (
    AuthError,
    BackendConnectionError,
    BackendError,
    BadResponseError,
    RateLimitError,
)
from decide.types import Answer, Meta, Request, Response
from decide.wire import from_wire_answers, to_wire_request


class TypeSafeBackend(BaseBackend):
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
        transport: httpx.BaseTransport | None = None,
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

    def _client_(self) -> httpx.Client:
        """Lazily create and reuse one `httpx.Client` for this backend instance."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.base_url, timeout=self.timeout, transport=self._transport
            )
        return self._client

    def _aclient_(self) -> httpx.AsyncClient:
        """Lazily create and reuse one `httpx.AsyncClient` for this backend instance."""
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, transport=self._transport
            )
        return self._aclient

    def _body(self, request: Request) -> dict[str, Any]:
        """Render the wire request body with `model` resolved to
        `request.model or self.model or self.DEFAULT_MODEL`."""
        wire = to_wire_request(request)
        wire["model"] = request.model or self.model or self.DEFAULT_MODEL
        return wire

    def _handle_response(
        self, response: httpx.Response, request: Request
    ) -> tuple[dict[str, Answer], Any]:
        if response.status_code in (401, 403):
            raise AuthError(self.name, f"authentication failed (HTTP {response.status_code})")
        if response.status_code == 429:
            raise RateLimitError(self.name, "rate limited (HTTP 429)")
        if response.status_code >= 400:
            raise BackendError(self.name, f"HTTP {response.status_code}: {response.text[:200]}")
        try:
            data = response.json()
        except ValueError as exc:
            raise BadResponseError(self.name, "response body was not valid JSON", exc) from exc
        if not isinstance(data, dict) or "answers" not in data:
            raise BadResponseError(self.name, "response JSON is missing 'answers'")
        answers = from_wire_answers(data["answers"], request)
        return answers, data

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any]:
        body = self._body(request)
        try:
            response = self._client_().post(self.PATH, json=body, headers=self._headers)
        except httpx.TransportError as exc:
            raise BackendConnectionError(
                self.name, f"could not reach {self.name}: {exc}", exc
            ) from exc
        return self._handle_response(response, request)

    async def adecide(self, request: Request) -> Response:
        start = time.perf_counter()
        body = self._body(request)
        try:
            response = await self._aclient_().post(self.PATH, json=body, headers=self._headers)
        except httpx.TransportError as exc:
            raise BackendConnectionError(
                self.name, f"could not reach {self.name}: {exc}", exc
            ) from exc
        answers, raw = self._handle_response(response, request)
        latency_ms = (time.perf_counter() - start) * 1000
        return Response(
            answers=answers,
            meta=Meta(
                backend=self.name, model=self.model, latency_ms=latency_ms, route=[], raw=raw
            ),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None
