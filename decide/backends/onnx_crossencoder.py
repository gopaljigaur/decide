"""ONNX CrossEncoder backend: serve a fine-tuned cross-encoder on CPU without torch.

Reads a bundle directory holding `model.onnx`, `tokenizer.json` and `config.json`
(`{"max_length": int}`). Templates and aggregation are inherited from `CrossEncoderBackend`, so
scores match the sentence-transformers model the bundle was exported from.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from decide.backends.crossencoder import CrossEncoderBackend, GenericTemplate, Template
from decide.errors import BackendError, ConfigError


class OnnxCrossEncoderBackend(CrossEncoderBackend):
    """`CrossEncoderBackend` that runs an ONNX bundle with onnxruntime and tokenizers.

    Args:
        model_dir: Directory with `model.onnx`, `tokenizer.json` and `config.json`.
        template: See `CrossEncoderBackend`.
        temperature: See `CrossEncoderBackend`.
        batch_size: Pairs per onnxruntime call.
        threads: Intra-op and inter-op thread count for the session.

    Raises:
        ValueError: `temperature` is not a positive, finite number.
        ConfigError: onnxruntime or tokenizers is missing, or a bundle file is absent.
    """

    name = "onnx-crossencoder"

    def __init__(
        self,
        model_dir: str | os.PathLike,
        *,
        template: Template | None = None,
        temperature: float = 1.0,
        batch_size: int = 32,
        threads: int = 1,
    ) -> None:
        if not (math.isfinite(temperature) and temperature > 0):
            raise ValueError(f"temperature must be a positive finite number, got {temperature!r}")
        try:
            import numpy
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise ConfigError(
                "the onnx-crossencoder backend needs onnxruntime, tokenizers and numpy; "
                "install them with the 'pydecide[onnx]' extra"
            ) from exc

        root = Path(model_dir)
        paths = {n: root / n for n in ("model.onnx", "tokenizer.json", "config.json")}
        for path in paths.values():
            if not path.is_file():
                raise ConfigError(f"onnx bundle file not found: {path}")
        max_length = int(json.loads(paths["config.json"].read_text())["max_length"])

        self._np = numpy
        self._tokenizer = Tokenizer.from_file(str(paths["tokenizer.json"]))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding()
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = threads
        self._session = onnxruntime.InferenceSession(
            str(paths["model.onnx"]), options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {i.name for i in self._session.get_inputs()}

        self.model = str(root)
        self._model = None
        self.template = template if template is not None else GenericTemplate()
        self.temperature = temperature
        self.batch_size = batch_size

    def _predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        try:
            for start in range(0, len(pairs), self.batch_size):
                encodings = self._tokenizer.encode_batch(pairs[start : start + self.batch_size])
                columns = {
                    "input_ids": [e.ids for e in encodings],
                    "attention_mask": [e.attention_mask for e in encodings],
                    "token_type_ids": [e.type_ids for e in encodings],
                }
                feed = {
                    k: self._np.asarray(v, dtype=self._np.int64)
                    for k, v in columns.items()
                    if k in self._input_names
                }
                logits = self._session.run(None, feed)[0]
                scores.extend(float(row[0]) for row in logits)
        except Exception as exc:
            raise BackendError(self.name, str(exc), exc) from exc
        return scores
