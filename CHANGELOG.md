# Changelog

## 0.1.1 (2026-09-22)

- `decide ask` question names are now optional: an unnamed `--choice`,
  `--score` or `--noul` is auto-named `choice`/`score`/`noul`, with a second
  unnamed flag of the same type becoming `choice2`/`score2`/`noul2`; explicit
  `NAME=` names still work and can be mixed with unnamed flags. Duplicate
  explicit names are now a usage error instead of silently overwriting each
  other.
- FastAPI and uvicorn moved into the base install, so `pip install pydecide`
  is enough to run `decide serve`; `server` remains as an empty extra for
  compatibility with `pip install "pydecide[server]"`.
- `decide backends` now shows an `INSTALL` column with the exact
  `pip install "pydecide[...]"` command for every backend that isn't
  installed.

## 0.1.0 (2026-09-22)

- Add `Choice`, `Score`, `Noul`, `Request`, `Response` and their answer types
  as frozen, validated dataclasses (`decide/types.py`).
- Add `Client`/`AsyncClient` with an ordered backend fallback chain,
  confidence gating via `Gate`, `Client.from_env`/`AsyncClient.from_env`
  environment-based configuration, and `decide_batch` with partial-failure
  reporting through `AllBackendsFailed`.
- Add six backends: `typesafe` (TypeSafe/Jev HTTP API), `openrouter`
  (OpenRouter Decisions endpoint), `laya` (local PyTorch), `laya_mlx`
  (local MLX, Apple Silicon), `crossencoder` (any sentence-transformers
  CrossEncoder, with generic and KaLM-Jev templates), and `llm` (JSON-prompted
  fallback over any OpenAI-compatible chat-completions server).
- Add a Jev/TypeSafe wire-compatible HTTP server (`decide serve`, the
  `server` extra) exposing `POST /v1/systemone`, `GET /health` and
  `GET /v1/models` over any configured backend chain.
- Add a command line interface: `decide ask`, `decide backends`, `decide serve`.

### Known limitations

- LLM emulation (`llm` backend) probabilities are self-reported by the
  model, not measured or calibrated.
- CrossEncoder (`crossencoder` backend) outputs are normalized scores
  (softmax/sigmoid over raw logits), not calibrated probabilities.
- `AsyncClient.decide_batch` calls each backend's `adecide` concurrently
  per request; it does not use a backend's batch path.
- The `mlx` extra (`laya_mlx` backend) needs Python 3.11 or newer.
