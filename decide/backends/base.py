from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from decide.types import Answer, Meta, Request, Response


@dataclass(frozen=True)
class Capabilities:
    batch: bool = False
    local: bool = False
    max_choice_options: int | None = None


@runtime_checkable
class Backend(Protocol):
    name: str

    def capabilities(self) -> Capabilities: ...

    def decide(self, request: Request) -> Response: ...

    async def adecide(self, request: Request) -> Response: ...

    def decide_batch(self, requests: Sequence[Request]) -> list[Response]: ...


class BaseBackend:
    """Common scaffolding for backend implementations.

    Subclasses implement `_decide` (and, for a real async transport rather than
    a thread-pooled sync call, `_adecide`) and may override `capabilities` and
    `decide_batch` for backend-specific behavior (e.g. real batching).

    `_decide`/`_adecide` return `(answers, raw, model)`: `model` is the model
    name actually used to answer, or `None` to mean "report `self.model`" (the
    common case for backends with a single fixed or instance-configured
    model). This lets a backend that resolves its model per-request (e.g. from
    `request.model` or a value the response itself reports) surface that in
    `Meta.model` instead of `decide()`/`adecide()` always reporting the
    backend's static default.
    """

    name = "base"
    model: str | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def decide(self, request: Request) -> Response:
        start = time.perf_counter()
        answers, raw, model = self._decide(request)
        latency_ms = (time.perf_counter() - start) * 1000
        return Response(
            answers=answers,
            meta=Meta(
                backend=self.name,
                model=model if model is not None else self.model,
                latency_ms=latency_ms,
                route=[],
                raw=raw,
            ),
        )

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any, str | None]:
        raise NotImplementedError

    async def adecide(self, request: Request) -> Response:
        start = time.perf_counter()
        answers, raw, model = await self._adecide(request)
        latency_ms = (time.perf_counter() - start) * 1000
        return Response(
            answers=answers,
            meta=Meta(
                backend=self.name,
                model=model if model is not None else self.model,
                latency_ms=latency_ms,
                route=[],
                raw=raw,
            ),
        )

    async def _adecide(self, request: Request) -> tuple[dict[str, Answer], Any, str | None]:
        return await asyncio.to_thread(self._decide, request)

    def decide_batch(self, requests: Sequence[Request]) -> list[Response]:
        return [self.decide(r) for r in requests]
