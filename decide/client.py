from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from decide.backends import available
from decide.backends import load as load_backend
from decide.backends.base import Backend
from decide.config import DEFAULT_CONFIG, Config
from decide.errors import AllBackendsFailed, BackendError, ConfigError, LowConfidence
from decide.gate import Gate
from decide.recording import (
    Recorder,
    decision_row,
    new_decision_id,
    safe_record,
)
from decide.types import Content, Question, Request, Response

_LAYA_MODEL_ENV = "DECIDE_LOCAL_MODEL"


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
    route: list[str],
    errors: list[BackendError],
    best: tuple[float, Response, str] | None,
    on_low_confidence: Literal["accept", "raise"] = "accept",
) -> Response:
    if best is not None:
        if on_low_confidence == "raise":
            route.append(f"{best[2]}:low_confidence")
            raise LowConfidence(list(route), best[1].with_meta(route=list(route)))
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
    on_low_confidence: Literal["accept", "raise"] = "accept",
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
        results[idx] = _finish(routes[idx], errors_by_state[idx], bests[idx], on_low_confidence)

    if unroutable:
        partial = {i: r for i, r in enumerate(results) if r is not None}
        route = [step for idx in unroutable for step in unroutable[idx]]
        errors = [err for idx in unroutable for err in errors_by_state[idx]]
        raise AllBackendsFailed(route, errors, partial=partial, failed=unroutable)

    return [r for r in results if r is not None]


class Client:
    def __init__(
        self,
        backends: Sequence[Backend] | None = None,
        policy: Gate | None = None,
        *,
        on_low_confidence: Literal["accept", "raise"] | None = None,
        config: Config | str | Path | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        """Build from backend instances, a Config, or a JSON path.

        With neither source, read decide.json in the current directory or discover
        an installed local backend. Explicit policy and abstention arguments override
        config defaults. Construction never prompts.
        """
        if backends is not None and config is not None:
            raise ConfigError("pass backends or config, not both")
        resolved = _read_config(config) if backends is None else Config()
        self.backends = (
            list(backends) if backends is not None else _configured_backends(resolved, os.environ)
        )
        self.policy = policy if policy is not None else Gate(resolved.min_confidence)
        self.on_low_confidence = (
            on_low_confidence if on_low_confidence is not None else resolved.on_low_confidence
        )
        self.recorder = recorder
        if _local_only(os.environ):
            _refuse_hosted(self.backends)

    def decide(
        self,
        state: Content,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        decision_id: str | None = None,
    ) -> Response:
        """`decision_id` tags the recorded row, so an outcome can be attached to it later."""
        request = Request(state=state, questions=questions, model=model)
        response = self._run_chain(request)
        if self.recorder is not None:
            safe_record(
                self.recorder,
                decision_row(decision_id or new_decision_id(), request, response),
            )
        return response

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
        return _finish(route, errors, best, self.on_low_confidence)

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

        return _finish_batch(
            pending, routes, errors_by_state, bests, results, self.on_low_confidence
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        policy: Gate | None = None,
        *,
        on_low_confidence: Literal["accept", "raise"] | None = None,
        config: Config | str | Path | None = None,
    ) -> Client:
        """Build a `Client` from an environment mapping.

        Omitted `env` reads `os.environ` and the current directory's decide.json.
        An explicit mapping (including `{}`) skips file discovery unless `config`
        is supplied. DECIDE_BACKENDS overrides the saved chain; API keys only
        authenticate explicitly selected backends.
        """
        resolved_env = env if env is not None else os.environ
        resolved = _read_config(config) if config is not None or env is None else Config()
        backends = _configured_backends(resolved, resolved_env, env_selection=True)
        resolved_policy = policy if policy is not None else _policy_from_env(resolved_env, resolved)
        return cls(
            backends,
            policy=resolved_policy,
            on_low_confidence=(
                on_low_confidence if on_low_confidence is not None else resolved.on_low_confidence
            ),
        )

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
    def __init__(
        self,
        backends: Sequence[Backend] | None = None,
        policy: Gate | None = None,
        *,
        on_low_confidence: Literal["accept", "raise"] | None = None,
        config: Config | str | Path | None = None,
        recorder: Recorder | None = None,
    ) -> None:
        """Build from backend instances, a Config, or a JSON path.

        With neither source, read decide.json in the current directory or discover
        an installed local backend. Explicit policy and abstention arguments override
        config defaults. Construction never prompts.
        """
        if backends is not None and config is not None:
            raise ConfigError("pass backends or config, not both")
        resolved = _read_config(config) if backends is None else Config()
        self.backends = (
            list(backends) if backends is not None else _configured_backends(resolved, os.environ)
        )
        self.policy = policy if policy is not None else Gate(resolved.min_confidence)
        self.on_low_confidence = (
            on_low_confidence if on_low_confidence is not None else resolved.on_low_confidence
        )
        self.recorder = recorder
        if _local_only(os.environ):
            _refuse_hosted(self.backends)

    async def decide(
        self,
        state: Content,
        questions: Mapping[str, Question],
        *,
        model: str | None = None,
        decision_id: str | None = None,
    ) -> Response:
        """`decision_id` tags the recorded row, so an outcome can be attached to it later."""
        request = Request(state=state, questions=questions, model=model)
        response = await self._run_chain(request)
        if self.recorder is not None:
            safe_record(
                self.recorder,
                decision_row(decision_id or new_decision_id(), request, response),
            )
        return response

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
        return _finish(route, errors, best, self.on_low_confidence)

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

        return _finish_batch(
            pending, routes, errors_by_state, bests, results, self.on_low_confidence
        )

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        policy: Gate | None = None,
        *,
        on_low_confidence: Literal["accept", "raise"] | None = None,
        config: Config | str | Path | None = None,
    ) -> AsyncClient:
        """Build an `AsyncClient` from an environment mapping.

        Omitted `env` reads `os.environ` and the current directory's decide.json.
        An explicit mapping (including `{}`) skips file discovery unless `config`
        is supplied. DECIDE_BACKENDS overrides the saved chain; API keys only
        authenticate explicitly selected backends.
        """
        resolved_env = env if env is not None else os.environ
        resolved = _read_config(config) if config is not None or env is None else Config()
        backends = _configured_backends(resolved, resolved_env, env_selection=True)
        resolved_policy = policy if policy is not None else _policy_from_env(resolved_env, resolved)
        return cls(
            backends,
            policy=resolved_policy,
            on_low_confidence=(
                on_low_confidence if on_low_confidence is not None else resolved.on_low_confidence
            ),
        )

    async def aclose(self) -> None:
        for backend in self.backends:
            aclose = getattr(backend, "aclose", None)
            if callable(aclose):
                await aclose()
            close = getattr(backend, "close", None)
            if callable(close):
                close()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()


def _policy_from_env(env: Mapping[str, str] | None, config: Config | None = None) -> Gate:
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
    return Gate(config.min_confidence if config is not None else 0.0)


def _kwargs_for(
    name: str, env: Mapping[str, str], options: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    credentials = {
        "typesafe": "TYPESAFE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "openai_decisions": "OPENAI_API_KEY",
    }
    if name == "typesafe":
        kwargs: dict[str, Any] = {
            "api_key": env.get("TYPESAFE_API_KEY"),
            "base_url": env.get("TYPESAFE_BASE_URL"),
        }
    elif name == "openrouter":
        kwargs = {"api_key": env.get("OPENROUTER_API_KEY")}
    elif name == "openai_decisions":
        kwargs = {
            "api_key": env.get("OPENAI_API_KEY"),
            "model": env.get("DECIDE_OPENAI_DECISIONS_MODEL"),
        }
    elif name == "llm":
        kwargs = {
            "base_url": env.get("DECIDE_LLM_BASE_URL"),
            "api_key": env.get("OPENAI_API_KEY"),
            "model": env.get("DECIDE_LLM_MODEL", "gpt-4o-mini"),
        }
    elif name in ("laya", "laya_mlx"):
        kwargs = _auto_local_kwargs(env)
    elif name == "crossencoder":
        kwargs = {}
    else:
        raise ConfigError(f"unknown backend {name!r}")
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    kwargs.update(options or {})
    if name in credentials and "api_key" not in kwargs:
        raise ConfigError(f"missing {credentials[name]} for backend {name!r}")
    if name == "crossencoder" and not kwargs.get("model"):
        raise ConfigError(
            "crossencoder requires a model in Config.options; it cannot be configured "
            "from the environment alone"
        )
    return kwargs


def _auto_local_kwargs(env: Mapping[str, str]) -> dict[str, Any]:
    """kwargs for `laya`/`laya_mlx`: `model=` only when `DECIDE_LOCAL_MODEL` is set, so the
    backend's own constructor default applies otherwise."""
    if _LAYA_MODEL_ENV in env:
        return {"model": env[_LAYA_MODEL_ENV]}
    return {}


def _refuse_hosted(backends: Sequence[Backend]) -> None:
    """Raise unless every backend answers in process.

    A backend that does not say it is local is treated as hosted, so a custom one has to
    opt in through `Capabilities(local=True)`.
    """
    hosted = [b.name for b in backends if not b.capabilities().local]
    if hosted:
        raise ConfigError(
            f"local only is on and these backends send the state to a server: {hosted}"
        )


_LOCAL_ONLY_ENV = "DECIDE_LOCAL_ONLY"

# Backends that answer in process. Anything else sends the state to a server.
_LOCAL_BACKENDS = frozenset({"laya", "laya_mlx", "crossencoder"})


def _local_only(env: Mapping[str, str] | None) -> bool:
    """`DECIDE_LOCAL_ONLY` set to anything but 0."""
    value = (env or {}).get(_LOCAL_ONLY_ENV, "")
    return bool(value) and value != "0"


def _resolve_env_backends(
    env: Mapping[str, str] | None,
    local_only: bool = False,
    options: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    env = env or {}

    explicit = env.get("DECIDE_BACKENDS")
    if explicit is not None and explicit.strip():
        seen: set[str] = set()
        names = []
        for n in explicit.split(","):
            n = n.strip()
            if n and n not in seen:
                seen.add(n)
                names.append(n)
        if not names:
            raise ConfigError("DECIDE_BACKENDS must contain at least one backend name")
        if local_only:
            hosted = [n for n in names if n not in _LOCAL_BACKENDS]
            if hosted:
                raise ConfigError(
                    f"DECIDE_BACKENDS names {hosted} while local only is on; "
                    f"local backends are {sorted(_LOCAL_BACKENDS)}"
                )
        kwargs_by_name = {name: _kwargs_for(name, env, (options or {}).get(name)) for name in names}
        return names, kwargs_by_name

    availability = available()

    names = []
    kwargs_by_name = {}

    if availability.get("laya_mlx"):
        names.append("laya_mlx")
        kwargs_by_name["laya_mlx"] = _auto_local_kwargs(env)
    elif availability.get("laya"):
        names.append("laya")
        kwargs_by_name["laya"] = _auto_local_kwargs(env)

    if local_only:
        if not names:
            raise ConfigError(
                "local only is on and no local backend is installed; install one with "
                'uv tool install "pydecide[mlx]"  (Apple Silicon) or "pydecide[laya]"'
            )
        return names, kwargs_by_name

    if not names:
        raise ConfigError(
            "no local backend installed and no backend explicitly selected. Run decide init.\n"
            'Install a local model backend: uv tool install "pydecide[mlx]"  (Apple Silicon) '
            'or "pydecide[laya]"\n'
            "For hosted inference, run decide init or select DECIDE_BACKENDS explicitly; "
            "credentials such as TYPESAFE_API_KEY never select a backend."
        )

    return names, kwargs_by_name


def _read_config(config: Config | str | Path | None) -> Config:
    if isinstance(config, Config):
        return config
    if config is not None:
        return Config.load(config)
    return Config.load() if DEFAULT_CONFIG.exists() else Config()


def _configured_backends(
    config: Config, env: Mapping[str, str], *, env_selection: bool = False
) -> list[Backend]:
    # Constructors honor the supplied config. from_env retains explicit env overrides.
    effective_env = dict(env)
    if not env_selection or not effective_env.get("DECIDE_BACKENDS", "").strip():
        effective_env["DECIDE_BACKENDS"] = ",".join(config.backends)
    names, kwargs = _resolve_env_backends(effective_env, _local_only(env), config.options)
    try:
        backends = [load_backend(name, **kwargs[name]) for name in names]
    except TypeError as exc:
        raise ConfigError(f"invalid backend options: {exc}") from exc
    if _local_only(env):
        _refuse_hosted(backends)
    return backends
