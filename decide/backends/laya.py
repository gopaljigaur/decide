"""laya backend: local structured judgments with a laya RL-agent decision model (PyTorch).

`laya.Agent.predict` (aliased `system_one`) accepts almost exactly the question dict
`decide.wire.to_wire_request(...)["questions"]` produces -- `{"type", "instructions", "criteria"}`
per question, `criteria` a dict for `choice`, a list for `score`, omitted or a dict for `noul` --
and returns TypeSafe wire-shaped answers with `Score.probabilities` already index-keyed as
`{"0": ..., "1": ...}` (see laya 0.3.5's `laya/agent.py::Agent.system_one` and laya-mlx 0.1.0's
`laya_mlx/agent.py::Agent.system_one`, which mirrors it). So this backend is close to a
pass-through: no question/answer reshaping is needed beyond what `decide.wire` already does.
`request.state` (str, dict or list) is passed to `agent.predict` unrendered, since `Agent.predict`
accepts the same shapes `decide.types.Content` does.

Neither package exposes a batch `predict` today, so `capabilities().batch` and `decide_batch`
check for an `agent.predict_batch(states, questions)` attribute (one raw response per request,
in order) and fall back to `BaseBackend`'s per-request loop when it is absent.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from decide.backends.base import BaseBackend, Capabilities
from decide.errors import BackendError, BadResponseError, ConfigError
from decide.types import Answer, Meta, Request, Response
from decide.wire import from_wire_answers, to_wire_request


def _model_label(model: str, subfolder: str | None) -> str:
    """`Meta.model`: the requested model id, plus '/subfolder' when one was given."""
    return model if subfolder is None else f"{model}/{subfolder}"


def _normalize_score_probabilities(answers: Any) -> Any:
    """Defensive: convert a `score` answer's `probabilities` from a list to an index-keyed dict.

    The real laya/laya-mlx `predict` already returns index-keyed `{"0": p0, "1": p1, ...}`
    mappings, matching what `decide.wire.from_wire_answers` requires, so this is a no-op for the
    packages as they ship today. Kept defensively in case a future version -- or a hand-rolled
    `agent` -- returns a plain list instead.
    """
    if not isinstance(answers, Mapping):
        return answers
    fixed: dict[str, Any] = {}
    for name, a in answers.items():
        if (
            isinstance(a, Mapping)
            and a.get("type") == "score"
            and isinstance(a.get("probabilities"), Sequence)
            and not isinstance(a.get("probabilities"), (Mapping, str, bytes))
        ):
            a = dict(a)
            a["probabilities"] = {str(i): p for i, p in enumerate(a["probabilities"])}
        fixed[name] = a
    return fixed


def _predict(agent: Any, backend_name: str, request: Request) -> Any:
    """Call `agent.predict(state, questions)`, wrapping any raised exception in `BackendError`."""
    questions = to_wire_request(request)["questions"]
    try:
        return agent.predict(request.state, questions)
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(backend_name, f"agent.predict failed: {exc}", exc) from exc


def _parse(agent_response: Any, backend_name: str, request: Request) -> dict[str, Answer]:
    """Parse an agent's raw response (`{"answers": {...}, ...}`) into decide `Answer`s."""
    if not isinstance(agent_response, Mapping) or "answers" not in agent_response:
        raise BadResponseError(
            backend=backend_name,
            message=f"expected a mapping with an 'answers' key, got {agent_response!r}",
        )
    answers_wire = _normalize_score_probabilities(agent_response["answers"])
    return from_wire_answers(answers_wire, request)


def _decide_with_agent(
    agent: Any, backend_name: str, request: Request
) -> tuple[dict[str, Answer], Any, None]:
    """Shared `_decide` body for `LayaBackend` and `LayaMLXBackend`."""
    raw = _predict(agent, backend_name, request)
    answers = _parse(raw, backend_name, request)
    return answers, raw, None


def _batch_with_agent(
    agent: Any, backend_name: str, model: str, requests: Sequence[Request]
) -> list[Response]:
    """Shared `decide_batch` body: calls `agent.predict_batch(states, questions)` once."""
    start = time.perf_counter()
    states = [r.state for r in requests]
    questions = [to_wire_request(r)["questions"] for r in requests]
    try:
        raw_list = agent.predict_batch(states, questions)
    except BackendError:
        raise
    except Exception as exc:
        raise BackendError(backend_name, f"agent.predict_batch failed: {exc}", exc) from exc

    raw_list = list(raw_list)
    if len(raw_list) != len(requests):
        raise BadResponseError(
            backend=backend_name,
            message=(
                f"predict_batch must return one response per request "
                f"({len(requests)}), got {len(raw_list)}"
            ),
        )

    latency_ms = (time.perf_counter() - start) * 1000
    responses = []
    for request, raw in zip(requests, raw_list, strict=True):
        answers = _parse(raw, backend_name, request)
        responses.append(
            Response(
                answers=answers,
                meta=Meta(
                    backend=backend_name, model=model, latency_ms=latency_ms, route=[], raw=raw
                ),
            )
        )
    return responses


class LayaBackend(BaseBackend):
    """Answer `Choice`/`Score`/`Noul` questions locally with a laya RL-agent decision model.

    Args:
        model: A Hugging Face repo id (or local path) to load lazily with `laya.load` (requires
            the `pydecide[laya]` extra), unless `agent` is given.
        subfolder: Selects one checkpoint out of a repo bundling several (e.g. `"multilingual"`).
            Included in `Meta.model` as `f"{model}/{subfolder}"`.
        device: Forwarded to `laya.load(...)`.
        agent: A pre-loaded `laya.Agent` (or any object with a matching `predict`), used as is --
            `model`/`subfolder`/`device` are then only used to compute `Meta.model`.

    Raises:
        ConfigError: `agent` is `None` and `laya` cannot be imported.
    """

    name = "laya"

    def __init__(
        self,
        model: str = "convaiinnovations/laya",
        *,
        subfolder: str | None = None,
        device: str | None = None,
        agent: Any = None,
    ) -> None:
        if agent is None:
            try:
                import laya
            except ImportError as exc:
                raise ConfigError(
                    "the laya backend needs the 'laya' package; install it with the "
                    "'pydecide[laya]' extra"
                ) from exc
            agent = laya.load(model, subfolder=subfolder, device=device)
        self.agent = agent
        self.model = _model_label(model, subfolder)

    def capabilities(self) -> Capabilities:
        return Capabilities(local=True, batch=hasattr(self.agent, "predict_batch"))

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any, None]:
        return _decide_with_agent(self.agent, self.name, request)

    def decide_batch(self, requests: Sequence[Request]) -> list[Response]:
        if not hasattr(self.agent, "predict_batch"):
            return super().decide_batch(requests)
        return _batch_with_agent(self.agent, self.name, self.model, requests)
