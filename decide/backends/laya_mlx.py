"""laya-mlx backend: local structured judgments with a laya RL-agent decision model, on MLX
(Apple Silicon).

`laya_mlx.Agent.predict` mirrors upstream laya's contract (see `decide.backends.laya`'s module
docstring): same question dict shape, same TypeSafe wire-shaped, index-keyed-probabilities answer
shape. `LayaMLXBackend` subclasses `LayaBackend` and inherits `capabilities`/`_decide`/
`decide_batch` unchanged -- only construction differs (a different loader function and kwargs), so
only `__init__` and `name` are overridden here.
"""

from __future__ import annotations

from typing import Any

from decide.backends.laya import LayaBackend, _model_label
from decide.errors import ConfigError


class LayaMLXBackend(LayaBackend):
    """Answer `Choice`/`Score`/`Noul` questions locally with a laya RL-agent decision model,
    running on MLX (Apple Silicon). See `decide.backends.laya.LayaBackend` for the shared
    question/answer contract, `capabilities`, `_decide` and `decide_batch`.

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
        # Deliberately does not call LayaBackend.__init__: the loader function, its kwargs
        # (`dtype` instead of `subfolder`/`device`) and the import-error extra name all differ.
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
