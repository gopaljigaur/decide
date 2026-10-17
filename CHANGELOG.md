# Changelog

## Unreleased

- The gate now reads the backend's own `confidence` when it reports one,
  instead of always computing the top probability itself. The two differ: on
  one ag_news story von reports 0.877 against a top probability of 0.9077, and
  laya reports 0.906 against 0.9779; on a two option choice laya has reported
  0.5909 against 0.918. Every `Choice`, `Score` and `Noul` answer now carries
  `confidence`, which is what the backend sent or `None`. Backends that report
  nothing, such as the reranker, keep the computed fallback.

## 0.2.9 (2026-10-17)

- Correct what 0.2.6 said about the multilingual checkpoint. Its 0.527 on
  yes/no questions was mostly the unnamed options: the same 150 rows with the
  two options named score 0.853, against the English checkpoint's 0.887. The
  English checkpoint stays the default, on calibration (0.049 against 0.103),
  not on "barely above chance". `docs/backends.md` has both columns and the
  third checkpoint.
- Re-measured the README and `docs/` numbers on the current release of every
  dependency, including laya-mlx 0.4.0, which answers identically to 0.1.0 and
  is slower here: one Choice question goes from 15.6 ms to 20.9 ms.

## 0.2.8 (2026-10-17)

- `laya_mlx` now reads the installed laya-mlx version for the `noul_labels`
  capability. laya-mlx renders a `labels` override since 0.3.0, and 0.2.7
  refused labelled questions on every version of it. Checked against laya-mlx
  0.4.0: the question that answers a constant 0.000 answers 0.979 with the
  labels named, matching the PyTorch backend exactly.

## 0.2.7 (2026-10-17)

- `Noul` takes `labels`, which renames the two options the model scores, for
  example `labels={"true": "positive", "false": "negative"}`. The default
  `true`/`false` pair carries no task meaning: the published laya checkpoints
  answer a constant 0.000 at confidence 1.0 to a question containing the word
  "positive" until the labels are named, and naming them takes that question
  from 0.000 to 0.979 on a positive review.
- A backend that would answer while dropping those labels now refuses the
  question, so the chain hands it to the next backend instead of returning a
  confident wrong answer. `laya_mlx` and `crossencoder` cannot render them;
  `laya` can since laya 0.3.8; the HTTP backends send them on; the `llm`
  backend names them in its prompt. `Capabilities` carries the new
  `noul_labels` flag, and `docs/backends.md` has the table.

## 0.2.6 (2026-10-17)

- `laya_mlx` now defaults to `aac6fef/laya-mlx`, the English checkpoint, instead
  of the multilingual one. Measured on 150 rows each of ag_news and sst2, the
  multilingual checkpoint answers 0.527 of yes/no questions correctly against
  the English one's 0.833, and its expected calibration error on them is 0.373
  against 0.053. A gate reads that confidence, so the best calibrated checkpoint
  is the right default. Pass `model="aac6fef/laya-multilingual-mlx"` for input
  that is not English. The numbers are in `docs/backends.md`.
- Naming `laya` or `laya_mlx` in `DECIDE_BACKENDS` no longer requires
  `DECIDE_LOCAL_MODEL`. It behaves like auto-detection and uses the backend's
  own default checkpoint; set the variable to override it.
- README, `docs/server.md` and `docs/typesafe-sdk.md` carry re-measured output
  from the new default: one Choice question now takes 15.6 ms on an M4 instead
  of 8.6 ms, since the English checkpoint is 421M against 322M.

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
