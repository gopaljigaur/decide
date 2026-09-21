"""laya-mlx backend: local structured judgments with a laya RL-agent decision model, on MLX
(Apple Silicon).

`laya_mlx.Agent.predict` mirrors upstream laya's contract (see `decide.backends.laya`'s module
docstring): same question dict shape, same TypeSafe wire-shaped, index-keyed-probabilities answer
shape. This module reuses `laya.py`'s shared `_decide_with_agent`/`_batch_with_agent`/
`_model_label` helpers rather than duplicating them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from decide.backends.base import BaseBackend, Capabilities
from decide.backends.laya import _batch_with_agent, _decide_with_agent, _model_label
from decide.errors import ConfigError
from decide.types import Answer, Request, Response


class LayaMLXBackend(BaseBackend):
    """Answer `Choice`/`Score`/`Noul` questions locally with a laya RL-agent decision model,
    running on MLX (Apple Silicon). See `decide.backends.laya.LayaBackend` for the shared
    question/answer contract.

    Args:
        model: A Hugging Face repo id (or local path) to load lazily with `laya_mlx.load`
            (requires the `pydecide[mlx]` extra), unless `agent` is given. Defaults to
            `"aac6fef/laya-multilingual-mlx"`, an MLX-ready checkpoint (`laya_mlx.load`'s own
            default, `"convaiinnovations/laya"`, is laya's *PyTorch* repo id -- MLX loads its
            `model.safetensors` fine, but the multilingual checkpoint is the more broadly useful
            default for this backend).
        dtype: Forwarded to `laya_mlx.load(...)` (one of `"float32"`, `"float16"`, `"bfloat16"`);
            `None` leaves `laya_mlx.Agent`'s own default (`"float16"`) in place.
        agent: A pre-loaded `laya_mlx.Agent` (or any object with a matching `predict`), used as
            is -- `model`/`dtype` are then only used to compute `Meta.model`.

    Raises:
        ConfigError: `agent` is `None` and `laya_mlx` cannot be imported.
    """

    name = "laya_mlx"

    def __init__(
        self,
        model: str = "aac6fef/laya-multilingual-mlx",
        *,
        dtype: str | None = None,
        agent: Any = None,
    ) -> None:
        if agent is None:
            try:
                import laya_mlx
            except ImportError as exc:
                raise ConfigError(
                    "the laya_mlx backend needs the 'laya-mlx' package; install it with the "
                    "'pydecide[mlx]' extra"
                ) from exc
            kwargs: dict[str, Any] = {}
            if dtype is not None:
                kwargs["dtype"] = dtype
            agent = laya_mlx.load(model, **kwargs)
        self.agent = agent
        self.model = _model_label(model, None)

    def capabilities(self) -> Capabilities:
        return Capabilities(local=True, batch=hasattr(self.agent, "predict_batch"))

    def _decide(self, request: Request) -> tuple[dict[str, Answer], Any, None]:
        return _decide_with_agent(self.agent, self.name, request)

    def decide_batch(self, requests: Sequence[Request]) -> list[Response]:
        if not hasattr(self.agent, "predict_batch"):
            return super().decide_batch(requests)
        return _batch_with_agent(self.agent, self.name, self.model, requests)
