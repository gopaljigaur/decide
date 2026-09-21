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

    Subclasses implement `_decide` and may override `capabilities`, `adecide`
    and `decide_batch` for backend-specific behavior (e.g. real batching).
    """

    name = "base"
    model: str | None = None

    def capabilities(self) -> Capabilities:
        return Capabilities()

    def decide(self, request: Request) -> Response:
        start = time.perf_counter()
        answers, raw = self._decide(request)
        latency_ms = (time.perf_counter() - start) * 1000
        return Response(
            answers=answers,
            meta=Meta(
                backend=self.name,
                model=self.model,
                latency_ms=latency_ms,
                route=[],
                raw=raw,
            ),
        )

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any]:
        raise NotImplementedError

    async def adecide(self, request: Request) -> Response:
        return await asyncio.to_thread(self.decide, request)

    def decide_batch(self, requests: Sequence[Request]) -> list[Response]:
        return [self.decide(r) for r in requests]
