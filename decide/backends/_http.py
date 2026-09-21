"""Shared plumbing for httpx-based backends.

`HttpClientMixin` gives a backend a lazily-created, reused `httpx.Client` /
`httpx.AsyncClient` pair (plus `close()`/`aclose()`), and `raise_for_status`
maps a non-2xx `httpx.Response` to the right `decide.errors.BackendError`
subclass. Both `typesafe.py` and `llm.py` build on these instead of
duplicating the client lifecycle and status-code mapping.
"""

from __future__ import annotations

from typing import Any

import httpx

from decide.errors import AuthError, BackendConnectionError, BackendError, RateLimitError


class HttpClientMixin:
    """Lazy `httpx.Client`/`httpx.AsyncClient` lifecycle shared by HTTP backends.

    Subclasses must set `self.name: str`, `self.base_url: str`,
    `self.timeout: float`,
    `self._transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None`,
    `self._client: httpx.Client | None = None` and
    `self._aclient: httpx.AsyncClient | None = None` in `__init__`.
    """

    name: str
    base_url: str
    timeout: float
    _transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None
    _client: httpx.Client | None
    _aclient: httpx.AsyncClient | None

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

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    def _post_json(
        self, path: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """POST `json` to `path` on the sync client, mapping a transport failure to
        `BackendConnectionError`."""
        try:
            return self._client_().post(path, json=json, headers=headers)
        except httpx.TransportError as exc:
            raise BackendConnectionError(
                self.name, f"could not reach {self.name}: {exc}", exc
            ) from exc

    async def _apost_json(
        self, path: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """POST `json` to `path` on the async client, mapping a transport failure to
        `BackendConnectionError`."""
        try:
            return await self._aclient_().post(path, json=json, headers=headers)
        except httpx.TransportError as exc:
            raise BackendConnectionError(
                self.name, f"could not reach {self.name}: {exc}", exc
            ) from exc


def raise_for_status(backend: str, response: httpx.Response) -> None:
    """Raise the appropriate `BackendError` subclass for a non-2xx response.

    No-op for a successful response. 401/403 -> `AuthError`, 429 ->
    `RateLimitError`, any other 4xx/5xx -> `BackendError` carrying a short
    excerpt of the response body.
    """
    if response.status_code in (401, 403):
        raise AuthError(backend, f"authentication failed (HTTP {response.status_code})")
    if response.status_code == 429:
        raise RateLimitError(backend, "rate limited (HTTP 429)")
    if response.status_code >= 400:
        raise BackendError(backend, f"HTTP {response.status_code}: {response.text[:200]}")
