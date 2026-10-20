# Changelog

## 0.2.18 (2026-10-20)

Packaging and types, after an audit of the whole repository.

- `py.typed` ships, so a consumer's type checker reads the annotations that
  were already there. `mypy` is clean and runs in CI, which it did not before;
  it found 28 errors, nearly all from `criteria` being declared
  `Mapping | Sequence[str]` while `__post_init__` normalizes it to a mapping.
- The base install is `httpx` alone. FastAPI and uvicorn moved to the `server`
  extra, where `decide serve` already told people to look. A client no longer
  installs a web server it never runs.
- Every README link is absolute, so the PyPI page renders: its diagram was a
  broken image and every link to `docs/`, the changelog and the licence was a
  404, because PyPI resolves relative paths against `pypi.org`.
- Removed `Capabilities.max_choice_options`, which nothing set and nothing
  read, and an unused helper in `decide.recording`.

## 0.2.6 to 0.2.17 (15 to 19 October 2026)

Four changes, released in more steps than they deserved. What they amount to:

- **The gate reads the confidence the backend reports** (0.2.10). Every answer
  now carries `confidence`, which is what the backend sent or `None`, and the
  gate prefers it over the top probability. The two differ: von reports 0.877
  where its top probability is 0.9077, and laya has reported 0.5909 against
  0.918. Backends that report nothing keep the computed fallback.
- **A `Noul` can name its two options** (0.2.8, 0.2.9). `labels={"true":
  "positive", "false": "negative"}` renames what the model scores. The default
  pair carries no task meaning, and the published laya checkpoints answer a
  constant 0.000 at confidence 1.0 to a question containing the word
  "positive" until the options are named. A backend that cannot carry the
  names refuses the question instead of dropping them.
- **Local only is a deployment switch** (0.2.12 to 0.2.15). `DECIDE_LOCAL_ONLY=1`
  refuses every backend that would send the state to a server, whatever keys
  are in the environment, checked on what `from_env` builds and on what any
  `Client` accepts. `on_low_confidence="raise"` abstains with `LowConfidence`
  instead of accepting an answer that missed the gate.
- **Decisions can be recorded and a threshold fitted from them** (0.2.16).
  `Client(recorder=JsonlRecorder(path))` writes a row per decision,
  `record_outcome` attaches what turned out to be right, and `decide fit`
  prints what each threshold would have accepted and how often it was right.
  The calibration module could always fit a threshold; nothing produced the
  rows to fit it on.
- The `laya_mlx` default checkpoint also changed to the English one
  (0.2.6, 0.2.7), chosen on calibration, not accuracy.

## 0.2.5 (2026-10-15)

- Add `decide.policies` with four alternatives to thresholding the top
  probability: `CalibratedGate` rescales by a fitted temperature,
  `EntropyGate` reads the whole distribution, `MarginGate` reads the gap
  between first and second place, and `ConformalGate` accepts only when one
  candidate survives a fitted cut. Each is a drop-in for `Gate`, which is
  unchanged, and all of them leave Score answers ungated.
- Add `decide.calibration` with `fit_temperature` and `fit_conformal_quantile`,
  which set those parameters from answers whose correct label is known, plus a
  `Calibration` record that saves and loads as JSON. Pure Python, so the core
  install gains no dependency.
- The `llm` backend now sends `max_tokens`, defaulting to 512. Without a cap, a
  model that loses the JSON format generates until its context runs out, and a
  hosted API bills for all of it. Pass `None` for the old behaviour.
- Document running the `typesafe` backend against other servers that speak
  `POST /v1/systemone`, with a table of the ones actually tested.
- Split the reference sections of the README into `docs/`.

## 0.2.4 (2026-10-13)

- Fix `decide.__version__`, which 0.2.3 left reading `0.2.2`. A test now keeps it
  in step with the version in `pyproject.toml`.

## 0.2.3 (2026-10-13)

- Packaging metadata only, no code changes. The PyPI summary and keywords now
  name the OpenAI Decisions backend, laya-mlx and the CrossEncoder reranker,
  the classifiers declare the development status, audience and FastAPI, and
  the project links include the changelog.
- README: fixed a wrapped line that rendered as a stray list item, and replaced
  four spaced hyphens with colons.

## 0.2.2 (2026-10-08)

- Add `decide.compat`, with `TypeSafeClient` and `AsyncTypeSafeClient` as drop-in replacements for the typesafe-sdk clients that accept `fallback=` and `gate=` and return the SDK's `SystemOneResponse` plus a `.decide` route record.
- Raise the vendor exceptions from `decide.compat`: a failed hosted call raises the matching `TypeSafeAPIError` subclass (`TypeSafeInternalServerError`, `TypeSafeAuthenticationError`, `TypeSafeAPIConnectionError` and so on), so existing `except` clauses keep catching. With a fallback and every tier failing, the error is both a `TypeSafeError` and an `AllBackendsFailed`.
- Support `retry=` (a `typesafe_sdk.RetryPolicy`) in `decide.compat` for the hosted tier; `http_client=` still raises `NotImplementedError`.
- Add `BackendError.status`, `body` and `headers`, set for HTTP failures.
- Add the `openai_decisions` backend for the OpenAI Decisions API (public beta), auto-selected from `OPENAI_API_KEY` unless `DECIDE_OPENAI_DECISIONS=0`, with `DECIDE_OPENAI_DECISIONS_MODEL` to choose the model.

## 0.2.1 (2026-09-23)

- Remove internal development notes from the repository and the source distribution.

## 0.2.0 (2026-09-23)

- `Choice.criteria` and `Noul.criteria` now also accept a list of strings
  (candidate names for `Choice`, true then false for `Noul`), normalized to
  the same mapping shape as before; `parse_wire_request` and the server
  accept the equivalent wire shapes for free. No breaking changes: the
  mapping shape still works exactly as before, and `to_wire_request` still
  emits the mapping form.
- `Gate` gained `per_type`, an optional `{"choice"/"score"/"noul":
  threshold}` mapping. Per question, the most specific threshold wins:
  `per_question`, then `per_type`, then `min_confidence`. No breaking
  changes: `per_type` defaults to `None` and behavior is unchanged unless
  it's set.
- README: reframed the opening around the fallback chain, noted that laya
  now ships its own `laya-serve` server, and added a "Choosing a threshold"
  section with measured Jev accuracy-vs-threshold data from Jevals to guide
  `min_confidence`/`per_type` choices.

## 0.1.3 (2026-09-22)

- The CLI silences model loader progress output by default; `--verbose` on `ask` and `serve` shows it again. `--json` output is only JSON on stdout.

## 0.1.2 (2026-09-22)

- `Client.from_env`/`AsyncClient.from_env` now auto-select an installed
  local backend (`laya_mlx` or `laya`) with zero configuration: no
  environment variable is required, and `DECIDE_LOCAL_MODEL` becomes an
  optional override of the backend's own default model instead of a
  requirement to opt in. A local backend, when importable, is tried before
  the hosted backends as a fallback chain; `decide backends` reports its
  `CONFIGURED` column as `default` in this state.
- The `ConfigError` raised when nothing is configured now ends with
  concrete next steps: which extra to install for a local backend, or which
  environment variable to set for a hosted one.

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
