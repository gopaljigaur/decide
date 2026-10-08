"""Drop-in replacements for `typesafe_sdk.TypeSafeClient` and `AsyncTypeSafeClient`.

Swap the import and the calls keep working; add `fallback=` and the hosted API
is backed by a chain of other backends. Needs `typesafe-sdk` installed, which is
imported lazily so `import decide` never requires it.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from decide.backends import load as load_backend
from decide.backends.base import Backend
from decide.backends.typesafe import TypeSafeBackend
from decide.client import AsyncClient, Client
from decide.errors import (
    AllBackendsFailed,
    BackendConnectionError,
    BackendError,
    BadResponseError,
    ConfigError,
)
from decide.gate import Gate
from decide.types import Choice, Content, Noul, Question, Request, Response, Score
from decide.wire import _parse_wire_question, to_wire_answers

_API_KEY_ENV = "TYPESAFE_API_KEY"
_INSTALL_HINT = (
    "typesafe-sdk is required for decide.compat; install it with: pip install typesafe-sdk"
)


@dataclass(frozen=True)
class DecideInfo:
    """Which backend answered, and how the chain got there."""

    backend: str
    model: str | None
    latency_ms: float
    route: list[str] = field(default_factory=list)


def _sdk() -> Any:
    try:
        import typesafe_sdk
    except ImportError as exc:
        raise ConfigError(_INSTALL_HINT) from exc
    return typesafe_sdk


_response_class: type | None = None


def _decide_response_class() -> type:
    """Build `DecideResponse` on first use, since its base lives in the optional SDK."""
    global _response_class
    if _response_class is None:
        from pydantic import PrivateAttr

        base = _sdk().SystemOneResponse

        class DecideResponse(base):  # type: ignore[valid-type, misc]
            """The vendor `SystemOneResponse`, plus `.decide` describing the route taken."""

            # A private attribute keeps `model_dump()` and validation identical to the vendor model.
            _decide_info: DecideInfo | None = PrivateAttr(default=None)

            @property
            def decide(self) -> DecideInfo | None:
                return self._decide_info

        _response_class = DecideResponse
    return _response_class


def __getattr__(name: str) -> Any:
    if name == "DecideResponse":
        return _decide_response_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _to_question(q: Any) -> Question:
    """Convert a vendor question, a raw wire dict or a decide question to a decide question."""
    if isinstance(q, Choice | Score | Noul):
        return q
    if isinstance(q, Mapping):
        return _parse_wire_question("question", q)
    sdk = _sdk()
    instructions: Content = q.instructions if q.instructions is not None else ""
    if isinstance(q, sdk.Choice):
        return Choice(instructions, dict(q.criteria))
    if isinstance(q, sdk.Score):
        return Score(instructions, list(q.criteria))
    if isinstance(q, sdk.Noul):
        criteria = {k: v for k, v in (q.criteria or {}).items() if v is not None}
        return Noul(instructions, criteria or None)
    raise TypeError(f"unsupported question type: {type(q)!r}")


def _to_response(resp: Response) -> Any:
    raw = resp.meta.raw
    usage = raw.get("usage") if isinstance(raw, Mapping) else None
    if not isinstance(usage, Mapping):
        usage = {"input_tokens": 0, "output_tokens": 0}
    payload = {
        "model": resp.meta.model or resp.meta.backend,
        "answers": to_wire_answers(resp),
        "usage": dict(usage),
    }
    cls = _decide_response_class()
    # The vendor model is strict, which rejects the wire's string score indices.
    out = cls.model_validate(payload, strict=False)
    out._decide_info = DecideInfo(
        backend=resp.meta.backend,
        model=resp.meta.model,
        latency_ms=resp.meta.latency_ms,
        route=list(resp.meta.route),
    )
    return out


def _reject_unsupported(**options: Any) -> None:
    for name, value in options.items():
        if value is not None:
            raise NotImplementedError(f"{name}= is not supported by decide.compat yet")


def _vendor_error(exc: BackendError, hosted: TypeSafeBackend) -> Exception:
    """Map a backend failure to the vendor exception a `typesafe_sdk` call would have raised."""
    sdk = _sdk()
    endpoint = f"POST {hosted.base_url}{hosted.PATH}"
    if isinstance(exc, BackendConnectionError):
        if isinstance(exc.cause, httpx.TimeoutException):
            return sdk.TypeSafeAPITimeoutError(hosted.timeout)
        return sdk.TypeSafeAPIConnectionError(exc.message)
    try:
        import httpx2

        headers: Any = httpx2.Headers(exc.headers or {})
    except ImportError:
        headers = dict(exc.headers or {})
    if isinstance(exc, BadResponseError):
        return sdk.TypeSafeAPIResponseValidationError(
            200, exc.body, headers, "answers", endpoint=endpoint
        )
    status = exc.status or 0
    cls: type[Exception]
    if status == 401:
        cls = sdk.TypeSafeAuthenticationError
    elif status == 403:
        cls = sdk.TypeSafePermissionDeniedError
    elif status == 429:
        cls = sdk.TypeSafeRateLimitError
    elif status == 400:
        cls = sdk.TypeSafeBadRequestError
    elif status == 404:
        cls = sdk.TypeSafeNotFoundError
    elif status == 422:
        cls = sdk.TypeSafeUnprocessableEntityError
    elif status >= 500:
        cls = sdk.TypeSafeInternalServerError
    else:
        cls = sdk.TypeSafeAPIError
    return cls(status, exc.body, headers, endpoint=endpoint)


_failure_classes: dict[type, type] = {}


def _failure_class(vendor_cls: type) -> type:
    """A class that is both `AllBackendsFailed` and the vendor error, built per vendor class."""
    cls = _failure_classes.get(vendor_cls)
    if cls is None:

        def __init__(self: Any, failed: AllBackendsFailed, cause: Exception) -> None:
            # The vendor constructors take HTTP arguments, so copy their state instead.
            Exception.__init__(self, str(failed))
            self.__dict__.update(cause.__dict__)
            self.route = failed.route
            self.errors = failed.errors
            self.partial = failed.partial
            self.failed = failed.failed

        def __str__(self: Any) -> str:
            return f"all backends failed: {self.route}; hosted: {vendor_cls.__str__(self)}"

        cls = type(
            "AllBackendsFailed",
            (AllBackendsFailed, vendor_cls),
            {"__init__": __init__, "__str__": __str__, "__module__": __name__},
        )
        _failure_classes[vendor_cls] = cls
    return cls


def _translate(
    exc: BackendError | AllBackendsFailed, hosted: TypeSafeBackend, chained: bool
) -> Exception:
    """Build the exception to raise for a failed call, keeping vendor `except` clauses working."""
    if isinstance(exc, AllBackendsFailed):
        cause = _vendor_error(exc.errors[0], hosted) if exc.errors else None
        if not chained and cause is not None:
            cause.__cause__ = exc.errors[0]
            return cause
        if cause is None:
            cause = _sdk().TypeSafeAPIError(0, None, {})
        combined = _failure_class(type(cause))(exc, cause)
        combined.__cause__ = cause
        return combined
    vendor = _vendor_error(exc, hosted)
    vendor.__cause__ = exc
    return vendor


class _RetryingBackend:
    """Retries the hosted tier per a vendor `RetryPolicy` before the chain moves on."""

    def __init__(self, inner: TypeSafeBackend, policy: Any) -> None:
        self._inner = inner
        self._policy = policy
        self.name = inner.name

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _delay(self, attempt: int) -> float:
        p = self._policy
        delay = min(p.backoff_max, p.backoff_initial * 2 ** (attempt - 1))
        return delay - delay * p.backoff_jitter * random.random()

    def _retryable(self, exc: BackendError) -> bool:
        p = self._policy
        if isinstance(exc, BackendConnectionError):
            if isinstance(exc.cause, httpx.TimeoutException):
                return bool(p.api_timeout_error)
            return bool(p.api_connection_error)
        return exc.status in p.http_statuses

    def decide(self, request: Request) -> Response:
        attempt = 0
        while True:
            try:
                return self._inner.decide(request)
            except BackendError as exc:
                attempt += 1
                if attempt > self._policy.max_retries or not self._retryable(exc):
                    raise
                time.sleep(self._delay(attempt))

    async def adecide(self, request: Request) -> Response:
        attempt = 0
        while True:
            try:
                return await self._inner.adecide(request)
            except BackendError as exc:
                attempt += 1
                if attempt > self._policy.max_retries or not self._retryable(exc):
                    raise
                await asyncio.sleep(self._delay(attempt))


def _build_chain(
    *,
    api_key: str | None,
    model: str | None,
    retry: Any,
    timeout: Any,
    headers: Mapping[str, str] | None,
    transport: Any,
    http_client: Any,
    base_url: str | None,
    fallback: Sequence[Backend | str] | None,
) -> tuple[list[Backend], TypeSafeBackend]:
    sdk = _sdk()
    _reject_unsupported(http_client=http_client)
    if retry is not None and not isinstance(retry, sdk.RetryPolicy):
        raise TypeError("retry= must be a typesafe_sdk.RetryPolicy")
    key = api_key or os.environ.get(_API_KEY_ENV)
    if not key:
        raise ConfigError(f"missing api_key: pass api_key= or set {_API_KEY_ENV}")
    kwargs: dict[str, Any] = {}
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, int | float):
            raise NotImplementedError("timeout= must be a number of seconds in decide.compat")
        kwargs["timeout"] = float(timeout)
    hosted = TypeSafeBackend(
        key, base_url=base_url, model=model, headers=headers, transport=transport, **kwargs
    )
    first: Any = hosted if retry is None else _RetryingBackend(hosted, retry)
    extra = [load_backend(b) if isinstance(b, str) else b for b in fallback or ()]
    return [first, *extra], hosted


def _prepare(questions: Mapping[str, Any], per_call: Mapping[str, Any]) -> dict[str, Question]:
    _reject_unsupported(**per_call)
    return {name: _to_question(q) for name, q in questions.items()}


_RESPONSE_MODEL_MSG = (
    "response_model= is not supported by decide.compat: answers always come back as "
    "SystemOneResponse"
)


class TypeSafeClient:
    """Same constructor and `system_one` as `typesafe_sdk.TypeSafeClient`, with fallback.

    Hosted failures raise the vendor exceptions (`TypeSafeAPIError` and friends). When a
    fallback is configured and every tier fails, the error is also an `AllBackendsFailed`.

    `retry=` takes a `typesafe_sdk.RetryPolicy` and retries the hosted tier before the chain
    moves on. It honours `max_retries`, `backoff_initial`, `backoff_max`, `backoff_jitter`,
    `http_statuses`, `api_connection_error` and `api_timeout_error`. It ignores
    `respect_retry_after`, `exceptions`, `predicate` and `timeout`. Without `retry=` there are
    no retries. `http_client=` is not supported.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        retry: Any = None,
        timeout: Any = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
        http_client: Any = None,
        base_url: str | None = None,
        fallback: Sequence[Backend | str] | None = None,
        gate: Gate | None = None,
    ) -> None:
        backends, self._hosted = _build_chain(
            api_key=api_key,
            model=model,
            retry=retry,
            timeout=timeout,
            headers=headers,
            transport=transport,
            http_client=http_client,
            base_url=base_url,
            fallback=fallback,
        )
        self._client = Client(backends, policy=gate)

    def system_one(
        self,
        state: Content,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        retry: Any = None,
        timeout: Any = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        response_model: Any = None,
    ) -> Any:
        if response_model is not None:
            raise NotImplementedError(_RESPONSE_MODEL_MSG)
        converted = _prepare(
            questions,
            {
                "retry": retry,
                "timeout": timeout,
                "extra_headers": extra_headers,
                "extra_body": extra_body,
            },
        )
        try:
            response = self._client.decide(state, converted, model=model)
        except (BackendError, AllBackendsFailed) as exc:
            err = _translate(exc, self._hosted, len(self._client.backends) > 1)
            raise err from err.__cause__
        return _to_response(response)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> TypeSafeClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncTypeSafeClient:
    """Same constructor and `system_one` as `typesafe_sdk.AsyncTypeSafeClient`, with fallback."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        retry: Any = None,
        timeout: Any = None,
        headers: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        http_client: Any = None,
        base_url: str | None = None,
        fallback: Sequence[Backend | str] | None = None,
        gate: Gate | None = None,
    ) -> None:
        backends, self._hosted = _build_chain(
            api_key=api_key,
            model=model,
            retry=retry,
            timeout=timeout,
            headers=headers,
            transport=transport,
            http_client=http_client,
            base_url=base_url,
            fallback=fallback,
        )
        self._client = AsyncClient(backends, policy=gate)

    async def system_one(
        self,
        state: Content,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
        retry: Any = None,
        timeout: Any = None,
        extra_headers: Mapping[str, str] | None = None,
        extra_body: Mapping[str, Any] | None = None,
        response_model: Any = None,
    ) -> Any:
        if response_model is not None:
            raise NotImplementedError(_RESPONSE_MODEL_MSG)
        converted = _prepare(
            questions,
            {
                "retry": retry,
                "timeout": timeout,
                "extra_headers": extra_headers,
                "extra_body": extra_body,
            },
        )
        try:
            response = await self._client.decide(state, converted, model=model)
        except (BackendError, AllBackendsFailed) as exc:
            err = _translate(exc, self._hosted, len(self._client.backends) > 1)
            raise err from err.__cause__
        return _to_response(response)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncTypeSafeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = ["AsyncTypeSafeClient", "DecideInfo", "TypeSafeClient"]
