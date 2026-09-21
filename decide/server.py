"""Jev/TypeSafe-compatible HTTP server exposing a `Client` over `POST /v1/systemone`.

Requires the `pydecide[server]` extra (FastAPI). FastAPI is imported lazily
inside `create_app` so the core library has no hard dependency on it.
"""

# NOTE: no `from __future__ import annotations` here. FastAPI's route dependency
# injection resolves parameter annotations with `typing.get_type_hints`, which
# looks names up in the function's `__globals__` (the module namespace) -- not
# in the locals of the enclosing `create_app` call. Since FastAPI is imported
# lazily inside `create_app` (see its docstring), a stringified annotation
# (postponed evaluation) referring to that locally-imported `Request` name
# would fail to resolve and silently be treated as a plain (non-special)
# type, so FastAPI would try to bind it as a query parameter instead of the
# request object. Keeping annotations eagerly evaluated avoids that.

import hmac
import json
import logging
from typing import TYPE_CHECKING, Any

from decide.client import Client
from decide.errors import AllBackendsFailed, ConfigError, DecideError
from decide.wire import parse_wire_request, to_wire_answers

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def _error_response(status_code: int, message: str, error_type: str, **extra: Any) -> Any:
    from fastapi.responses import JSONResponse

    body: dict[str, Any] = {"message": message, "type": error_type}
    body.update(extra)
    return JSONResponse(status_code=status_code, content={"error": body})


def create_app(client: Client, *, api_key: str | None = None) -> "FastAPI":
    """Build a FastAPI app exposing `client` as a Jev/TypeSafe-compatible HTTP server.

    - `POST /v1/systemone`: TypeSafe `SystemOneRequest` in, `SystemOneResponse`-shaped
      JSON out (plus a `decide` extension carrying our own backend/route metadata).
    - `GET /health`: liveness plus the configured backend names.
    - `GET /v1/models`: an OpenAI-style listing of the configured backend names.

    If `api_key` is set, `POST /v1/systemone` requires `Authorization: Bearer <api_key>`.
    """
    try:
        from fastapi import FastAPI
        from fastapi import Request as HttpRequest
        from fastapi.concurrency import run_in_threadpool
    except ImportError as exc:
        raise ConfigError(
            "the server needs FastAPI; install it with the 'pydecide[server]' extra"
        ) from exc

    app = FastAPI()

    def _authorized(request: HttpRequest) -> bool:
        if api_key is None:
            return True
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer":
            return False
        return hmac.compare_digest(token.encode("utf-8"), api_key.encode("utf-8"))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "backends": [b.name for b in client.backends]}

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": b.name, "object": "model"} for b in client.backends],
        }

    @app.post("/v1/systemone")
    async def systemone(request: HttpRequest) -> Any:
        if not _authorized(request):
            return _error_response(401, "invalid api key", "authentication_error")

        raw_body = await request.body()
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            return _error_response(400, f"invalid JSON body: {exc}", "invalid_request_error")

        try:
            wire_request = parse_wire_request(body)
        except ValueError as exc:
            return _error_response(400, str(exc), "invalid_request_error")

        try:
            response = await run_in_threadpool(
                client.decide,
                wire_request.state,
                wire_request.questions,
                model=wire_request.model,
            )
            model_name = (
                response.meta.model if response.meta.model is not None else response.meta.backend
            )
            return {
                "model": model_name,
                "answers": to_wire_answers(response),
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "decide": {
                    "backend": response.meta.backend,
                    "latency_ms": response.meta.latency_ms,
                    "route": response.meta.route,
                },
            }
        except AllBackendsFailed as exc:
            return _error_response(502, str(exc), "backend_error", route=exc.route)
        except DecideError as exc:
            return _error_response(500, str(exc), "backend_error")
        except Exception:
            logger.exception("unexpected error handling POST /v1/systemone")
            return _error_response(500, "internal error", "server_error")

    return app
