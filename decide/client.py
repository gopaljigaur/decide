from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

from decide.backends import available
from decide.backends import load as load_backend
from decide.backends.base import Backend
from decide.errors import AllBackendsFailed, BackendError, ConfigError
from decide.gate import Gate
from decide.types import Content, Question, Request, Response

_LAYA_MODEL_ENV = "DECIDE_LOCAL_MODEL"

# Backends that `_kwargs_for` cannot build from environment variables, with the
# reason to show instead of the generic "unknown backend" message.
_ENV_UNSUPPORTED_BACKENDS: dict[str, str] = {
    "crossencoder": (
        "cannot be configured from the environment; construct it directly "
        "and pass it to Client([...])"
    ),
}


def _decision(
    backend: Backend,
    response: Response | None,
    error: BackendError | None,
    policy: Gate,
    route: list[str],
    errors: list[BackendError],
    best: tuple[float, Response, str] | None,
) -> tuple[Response | None, tuple[float, Response, str] | None]:
    """Update `route`/`errors`/`best` in place for one backend's outcome.

    Returns `(response_to_return, best)`: `response_to_return` is set when the
    gate passed and the chain should stop immediately.
    """
    if error is not None:
        route.append(f"{backend.name}:error")
        errors.append(error)
        return None, best

    assert response is not None
    ok, why = policy.passes(response)
    if ok:
        route.append(f"{backend.name}:ok")
        return response.with_meta(route=list(route)), best

    route.append(f"{backend.name}:{why}")
    conf = min(policy.confidence(response).values(), default=1.0)
    if best is None or conf > best[0]:
        best = (conf, response, backend.name)
    return None, best


def _finish(
    route: list[str], errors: list[BackendError], best: tuple[float, Response, str] | None
) -> Response:
    if best is not None:
        route.append(f"{best[2]}:accepted_low_confidence")
        return best[1].with_meta(route=list(route))
    raise AllBackendsFailed(list(route), errors)


def _apply_decisions(
    backend: Backend,
    pending: Sequence[int],
    sub_responses: Sequence[Response | None],
    sub_errors: Sequence[BackendError | None],
    policy: Gate,
    routes: list[list[str]],
    errors_by_state: list[list[BackendError]],
    bests: list[tuple[float, Response, str] | None],
    results: list[Response | None],
) -> list[int]:
    """Apply one backend's outcomes for a batch round to the running per-state state.

    `pending` and `sub_responses`/`sub_errors` are parallel: `pending[i]` is
    the input index that `sub_responses[i]`/`sub_errors[i]` belongs to.
    `pending` may be a prefix of the round's original pending indices (a
    non-batching send loop with `policy.on_error == "raise"` stops sending
    once it hits an error, so only that prefix has answers to apply).

    Mutates `routes`, `errors_by_state`, `bests` and `results` in place for
    every index in `pending`. Returns the input indices that are still
    pending after this backend (gate did not pass and there was no error).
    Raises the first `BackendError` immediately when `policy.on_error ==
    "raise"`, matching `Client._run_chain`/`AsyncClient._run_chain`.
    """
    still_pending: list[int] = []
    for idx, resp, err in zip(pending, sub_responses, sub_errors, strict=True):
        result, bests[idx] = _decision(
            backend, resp, err, policy, routes[idx], errors_by_state[idx], bests[idx]
        )
        if err is not None and policy.on_error == "raise":
            raise err
        if result is not None:
            results[idx] = result
        else:
            still_pending.append(idx)
    return still_pending


def _finish_batch(
    pending: Sequence[int],
    routes: list[list[str]],
    errors_by_state: list[list[BackendError]],
    bests: list[tuple[float, Response, str] | None],
    results: list[Response | None],
) -> list[Response]:
    """Resolve every state still pending once all backends have run.

    A pending state with a recorded `best` (some backend produced a response
    that just missed the confidence gate) resolves via `_finish`
    (`accepted_low_confidence`). A pending state with no `best` at all
    (every backend errored, none produced any response) is unroutable.

    If any state is unroutable, raises `AllBackendsFailed` with `partial`
    (every state that did resolve, keyed by input index) and `failed` (the
    route so far for every unroutable state, keyed by input index) so a
    caller can recover the completed siblings instead of losing them.
    """
    unroutable: dict[int, list[str]] = {}
    for idx in pending:
        if bests[idx] is None:
            unroutable[idx] = list(routes[idx])
            continue
        results[idx] = _finish(routes[idx], errors_by_state[idx], bests[idx])

    if unroutable:
        partial = {i: r for i, r in enumerate(results) if r is not None}
        route = [step for idx in unroutable for step in unroutable[idx]]
        errors = [err for idx in unroutable for err in errors_by_state[idx]]
        raise AllBackendsFailed(route, errors, partial=partial, failed=unroutable)

    return [r for r in results if r is not None]


class Client:
    def __init__(self, backends: Sequence[Backend], policy: Gate | None = None) -> None:
        self.backends = list(backends)
        self.policy = policy if policy is not None else Gate()

    def decide(
        self, state: Content, questions: Mapping[str, Question], *, model: str | None = None
    ) -> Response:
        request = Request(state=state, questions=questions, model=model)
        return self._run_chain(request)

    def _run_chain(self, request: Request) -> Response:
        route: list[str] = []
        errors: list[BackendError] = []
        best: tuple[float, Response, str] | None = None
        for backend in self.backends:
            try:
                response = backend.decide(request)
            except BackendError as exc:
                result, best = _decision(backend, None, exc, self.policy, route, errors, best)
                if self.policy.on_error == "raise":
                    raise
                continue
            result, best = _decision(backend, response, None, self.policy, route, errors, best)
            if result is not None:
                return result
        return _finish(route, errors, best)

    def decide_batch(
        self,
        states: Sequence[Content],
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> list[Response]:
        """Run the fallback chain over each state, preserving input order.

        Runs each backend over the whole remaining sub-batch: uses
        `backend.decide_batch` when `capabilities().batch` is true, else
        loops `decide` per request (stopping early if `policy.on_error ==
        "raise"` hits an error). The gate is applied per state; states that
        fail move to the next backend. On success, returns one `Response`
        per input state, in input order, each carrying its own route.

        If any state cannot be resolved by any backend, raises
        `AllBackendsFailed` instead of returning partial results silently;
        see its docstring for `partial`/`failed`.
        """
        requests = [Request(state=s, questions=questions, model=model) for s in states]
        n = len(requests)
        results: list[Response | None] = [None] * n
        routes: list[list[str]] = [[] for _ in range(n)]
        errors_by_state: list[list[BackendError]] = [[] for _ in range(n)]
        bests: list[tuple[float, Response, str] | None] = [None] * n
        pending = list(range(n))

        for backend in self.backends:
            if not pending:
                break
            sub_requests = [requests[i] for i in pending]
            sub_responses: list[Response | None]
            sub_errors: list[BackendError | None]
            if backend.capabilities().batch:
                try:
                    sub_responses = list(backend.decide_batch(sub_requests))
                    sub_errors = [None] * len(sub_requests)
                except BackendError as exc:
                    sub_responses = [None] * len(sub_requests)
                    sub_errors = [exc] * len(sub_requests)
                round_pending = pending
            else:
                sub_responses = []
                sub_errors = []
                for req in sub_requests:
                    try:
                        sub_responses.append(backend.decide(req))
                        sub_errors.append(None)
                    except BackendError as exc:
                        sub_responses.append(None)
                        sub_errors.append(exc)
                        if self.policy.on_error == "raise":
                            break
                round_pending = pending[: len(sub_responses)]

            pending = _apply_decisions(
                backend,
                round_pending,
                sub_responses,
                sub_errors,
                self.policy,
                routes,
                errors_by_state,
                bests,
                results,
            )

        return _finish_batch(pending, routes, errors_by_state, bests, results)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, policy: Gate | None = None) -> Client:
        backend_names, kwargs_by_name = _resolve_env_backends(env)
        backends = [load_backend(name, **kwargs_by_name[name]) for name in backend_names]
        resolved_policy = policy if policy is not None else _policy_from_env(env)
        return cls(backends, policy=resolved_policy)

    def close(self) -> None:
        for backend in self.backends:
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class AsyncClient:
    def __init__(self, backends: Sequence[Backend], policy: Gate | None = None) -> None:
        self.backends = list(backends)
        self.policy = policy if policy is not None else Gate()

    async def decide(
        self, state: Content, questions: Mapping[str, Question], *, model: str | None = None
    ) -> Response:
        request = Request(state=state, questions=questions, model=model)
        return await self._run_chain(request)

    async def _run_chain(self, request: Request) -> Response:
        route: list[str] = []
        errors: list[BackendError] = []
        best: tuple[float, Response, str] | None = None
        for backend in self.backends:
            try:
                response = await backend.adecide(request)
            except BackendError as exc:
                result, best = _decision(backend, None, exc, self.policy, route, errors, best)
                if self.policy.on_error == "raise":
                    raise
                continue
            result, best = _decision(backend, response, None, self.policy, route, errors, best)
            if result is not None:
                return result
        return _finish(route, errors, best)

    async def decide_batch(
        self,
        states: Sequence[Content],
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
    ) -> list[Response]:
        """Run the fallback chain over each state concurrently, preserving input order.

        Runs each backend over the whole remaining sub-batch by calling
        `adecide` per request concurrently with `asyncio.gather` (backends
        have no async batch path in v1). The gate is applied per state;
        states that fail move to the next backend. On success, returns one
        `Response` per input state, in input order, each carrying its own
        route.

        If any state cannot be resolved by any backend, raises
        `AllBackendsFailed` instead of returning partial results silently;
        see its docstring for `partial`/`failed`.
        """
        requests = [Request(state=s, questions=questions, model=model) for s in states]
        n = len(requests)
        results: list[Response | None] = [None] * n
        routes: list[list[str]] = [[] for _ in range(n)]
        errors_by_state: list[list[BackendError]] = [[] for _ in range(n)]
        bests: list[tuple[float, Response, str] | None] = [None] * n
        pending = list(range(n))

        for backend in self.backends:
            if not pending:
                break
            sub_requests = [requests[i] for i in pending]

            async def _run_one(
                req: Request, backend: Backend = backend
            ) -> tuple[Response | None, BackendError | None]:
                try:
                    return await backend.adecide(req), None
                except BackendError as exc:
                    return None, exc

            outcomes = await asyncio.gather(*(_run_one(req) for req in sub_requests))
            sub_responses = [resp for resp, _ in outcomes]
            sub_errors = [err for _, err in outcomes]

            pending = _apply_decisions(
                backend,
                pending,
                sub_responses,
                sub_errors,
                self.policy,
                routes,
                errors_by_state,
                bests,
                results,
            )

        return _finish_batch(pending, routes, errors_by_state, bests, results)

    @classmethod
    def from_env(
        cls, env: Mapping[str, str] | None = None, policy: Gate | None = None
    ) -> AsyncClient:
        backend_names, kwargs_by_name = _resolve_env_backends(env)
        backends = [load_backend(name, **kwargs_by_name[name]) for name in backend_names]
        resolved_policy = policy if policy is not None else _policy_from_env(env)
        return cls(backends, policy=resolved_policy)

    async def aclose(self) -> None:
        for backend in self.backends:
            aclose = getattr(backend, "aclose", None)
            if callable(aclose):
                await aclose()
                continue
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _policy_from_env(env: Mapping[str, str] | None) -> Gate:
    env = env or {}
    min_confidence = env.get("DECIDE_MIN_CONFIDENCE")
    if min_confidence is not None:
        try:
            value = float(min_confidence)
        except ValueError as exc:
            raise ConfigError(
                f"invalid DECIDE_MIN_CONFIDENCE {min_confidence!r}: must be a float"
            ) from exc
        return Gate(min_confidence=value)
    return Gate()


def _kwargs_for(name: str, env: Mapping[str, str]) -> dict[str, Any]:
    if name == "typesafe":
        if "TYPESAFE_API_KEY" not in env:
            raise ConfigError("missing TYPESAFE_API_KEY for backend 'typesafe'")
        kwargs: dict[str, Any] = {"api_key": env["TYPESAFE_API_KEY"]}
        base_url = env.get("TYPESAFE_BASE_URL")
        if base_url is not None:
            kwargs["base_url"] = base_url
        return kwargs
    if name == "openrouter":
        if "OPENROUTER_API_KEY" not in env:
            raise ConfigError("missing OPENROUTER_API_KEY for backend 'openrouter'")
        return {"api_key": env["OPENROUTER_API_KEY"]}
    if name == "llm":
        kwargs = {
            "base_url": env.get("DECIDE_LLM_BASE_URL"),
            "api_key": env.get("OPENAI_API_KEY"),
            "model": env.get("DECIDE_LLM_MODEL", "gpt-4o-mini"),
        }
        return {k: v for k, v in kwargs.items() if v is not None}
    if name in ("laya", "laya_mlx"):
        if _LAYA_MODEL_ENV not in env:
            raise ConfigError(f"missing {_LAYA_MODEL_ENV} for backend {name!r}")
        return {"model": env[_LAYA_MODEL_ENV]}
    if name in _ENV_UNSUPPORTED_BACKENDS:
        raise ConfigError(f"backend {name!r} {_ENV_UNSUPPORTED_BACKENDS[name]}")
    raise ConfigError(f"unknown backend {name!r}")


def _resolve_env_backends(
    env: Mapping[str, str] | None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    env = env or {}

    explicit = env.get("DECIDE_BACKENDS")
    if explicit is not None and explicit.strip():
        names = [n.strip() for n in explicit.split(",") if n.strip()]
        kwargs_by_name = {name: _kwargs_for(name, env) for name in names}
        return names, kwargs_by_name

    availability = available()

    names = []
    kwargs_by_name = {}

    if _LAYA_MODEL_ENV in env:
        if availability.get("laya_mlx"):
            names.append("laya_mlx")
            kwargs_by_name["laya_mlx"] = _kwargs_for("laya_mlx", env)
        elif availability.get("laya"):
            names.append("laya")
            kwargs_by_name["laya"] = _kwargs_for("laya", env)

    if "TYPESAFE_API_KEY" in env:
        names.append("typesafe")
        kwargs_by_name["typesafe"] = _kwargs_for("typesafe", env)

    if "OPENROUTER_API_KEY" in env:
        names.append("openrouter")
        kwargs_by_name["openrouter"] = _kwargs_for("openrouter", env)

    if "DECIDE_LLM_BASE_URL" in env or "OPENAI_API_KEY" in env:
        names.append("llm")
        kwargs_by_name["llm"] = _kwargs_for("llm", env)

    if not names:
        raise ConfigError(
            "no backend configured; checked DECIDE_LOCAL_MODEL, TYPESAFE_API_KEY, "
            "OPENROUTER_API_KEY, DECIDE_LLM_BASE_URL, OPENAI_API_KEY"
        )

    return names, kwargs_by_name
