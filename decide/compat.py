"""Drop-in replacements for `typesafe_sdk.TypeSafeClient` and `AsyncTypeSafeClient`.

Swap the import and the calls keep working; add `fallback=` and the hosted API
is backed by a chain of other backends. Needs `typesafe-sdk` installed, which is
imported lazily so `import decide` never requires it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from decide.backends import load as load_backend
from decide.backends.base import Backend
from decide.backends.typesafe import TypeSafeBackend
from decide.client import AsyncClient, Client
from decide.errors import ConfigError
from decide.gate import Gate
from decide.types import Choice, Content, Noul, Question, Response, Score
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
) -> list[Backend]:
    _sdk()
    _reject_unsupported(retry=retry, http_client=http_client)
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
    extra = [load_backend(b) if isinstance(b, str) else b for b in fallback or ()]
    return [hosted, *extra]


def _prepare(questions: Mapping[str, Any], per_call: Mapping[str, Any]) -> dict[str, Question]:
    _reject_unsupported(**per_call)
    return {name: _to_question(q) for name, q in questions.items()}


_RESPONSE_MODEL_MSG = (
    "response_model= is not supported by decide.compat: answers always come back as "
    "SystemOneResponse"
)


class TypeSafeClient:
    """Same constructor and `system_one` as `typesafe_sdk.TypeSafeClient`, with fallback."""

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
        backends = _build_chain(
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
        return _to_response(self._client.decide(state, converted, model=model))

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
        backends = _build_chain(
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
        return _to_response(await self._client.decide(state, converted, model=model))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> AsyncTypeSafeClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


__all__ = ["AsyncTypeSafeClient", "DecideInfo", "TypeSafeClient"]
