# pydecide

[![CI](https://github.com/gopaljigaur/decide/actions/workflows/ci.yml/badge.svg)](https://github.com/gopaljigaur/decide/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pydecide)](https://pypi.org/project/pydecide/)

`decide` is one Python client for typed decisions - Choice, Score and Noul -
over any "System One" decision model: TypeSafe's hosted Jev, OpenRouter's
Decisions endpoint, the open-weight laya family (PyTorch and MLX), any
sentence-transformers CrossEncoder, and a JSON-prompted LLM fallback. A
`Client` takes an ordered list of backends and a confidence policy: if a
backend errors or answers with low confidence, the next backend is tried.
The library also ships a small HTTP server that speaks TypeSafe's wire
protocol, so existing TypeSafe clients can point at a local model instead.

## Install

```bash
pip install pydecide
```

Local model backends and the server are optional extras:

```bash
pip install "pydecide[server]"   # decide serve (FastAPI + uvicorn)
pip install "pydecide[laya]"     # laya backend (PyTorch)
pip install "pydecide[mlx]"      # laya_mlx backend (Apple Silicon; needs Python 3.11+)
pip install "pydecide[st]"       # crossencoder backend (sentence-transformers)
pip install "pydecide[all]"      # everything above
```

The `mlx` extra depends on `laya-mlx`, which requires Python 3.11 or newer;
on 3.10 the extra installs nothing and the `laya_mlx` backend is
unavailable.

## Quickstart: the fallback chain

`Client.from_env()` builds a backend chain from whatever is installed and
configured in the environment mapping you pass it. It does not read
`os.environ` implicitly, so pass it explicitly (as the CLI does):

```python
import os
from decide import Client, Choice, Score, Noul

client = Client.from_env(env=os.environ)

r = client.decide(
    state={"ticket": "I was charged twice, please refund."},
    questions={
        "team": Choice(
            "Which team handles this?",
            {
                "billing": "charges, refunds",
                "engineering": "bugs and outages",
                "sales": "pricing and upgrades",
            },
        ),
        "severity": Score("How severe is this?", ["minor", "degraded", "blocked"]),
        "refund": Noul("Does the customer ask for money back?"),
    },
)

r.choices["team"].choice  # "billing"
r.choices["team"].probabilities  # {"billing": 1.0, "engineering": 0.0, "sales": 0.0}
r.scores["severity"].score  # 1.5522  (expected level index, 0..len(levels)-1)
r.scores["severity"].probabilities  # [0.0197, 0.4083, 0.5719]
r.nouls["refund"].noul  # 0.9461
r.meta.backend  # "laya_mlx"
r.meta.latency_ms  # 38.1
r.meta.route  # ["laya_mlx:ok"]
```

This was run against the `laya_mlx` backend
(`DECIDE_LOCAL_MODEL=aac6fef/laya-multilingual-mlx`); every value above is
the real output of that run, not illustrative.

`from_env` picks the chain in this order, using whatever is both installed
and configured: `laya_mlx` or `laya` (if `DECIDE_LOCAL_MODEL` is set),
`typesafe` (if `TYPESAFE_API_KEY` is set), `openrouter` (if
`OPENROUTER_API_KEY` is set), `llm` (if `DECIDE_LLM_BASE_URL` or
`OPENAI_API_KEY` is set). Set `DECIDE_BACKENDS="laya,typesafe"` to override
the order explicitly. If nothing is configured, `from_env` raises
`ConfigError` naming every variable it checked.

`AsyncClient` has the same surface, `await`ed: `await client.decide(...)`,
`await client.decide_batch(...)`, `AsyncClient.from_env(...)`.

## Backends

| Module | Backend name | Extra | Notes |
|---|---|---|---|
| `typesafe.py` | `typesafe` | none (httpx only) | `POST {base_url}/v1/systemone`, bearer auth. Default base URL `https://api.typesafe.ai`, default model `jev-latest`. |
| `openrouter.py` | `openrouter` | none | Same wire shape as `typesafe`, `POST https://openrouter.ai/api/alpha/decisions`, default model `typesafe/jev-latest`. |
| `laya.py` | `laya` | `pydecide[laya]` | Local PyTorch `laya.Agent`, default model `convaiinnovations/laya` when constructed directly. |
| `laya_mlx.py` | `laya_mlx` | `pydecide[mlx]` (Python 3.11+) | Local MLX `laya_mlx.Agent` (Apple Silicon), default model `aac6fef/laya-multilingual-mlx` when constructed directly. |
| `crossencoder.py` | `crossencoder` | `pydecide[st]` | Local `sentence_transformers.CrossEncoder`. Not configurable from the environment; construct it directly and pass it to `Client([...])`. |
| `llm.py` | `llm` | none (httpx only) | Any OpenAI-compatible chat-completions server, default base URL `https://api.openai.com/v1`, default model `gpt-4o-mini`. The least trustworthy backend (see below). |

`crossencoder` is built by hand, for example:

```python
from decide import Client, Choice
from decide.backends.crossencoder import CrossEncoderBackend

backend = CrossEncoderBackend("cross-encoder/ms-marco-MiniLM-L6-v2")
client = Client([backend])

r = client.decide(
    "I was charged twice for the same order last week, please refund the duplicate charge.",
    {
        "team": Choice(
            "Which team should handle this?",
            {
                "billing": "Charges, invoices, payment problems, refunds",
                "eng": "Bugs, crashes, broken features",
                "shipping": "Delivery status, delays, lost packages",
            },
        )
    },
)
r.choices["team"].choice  # "billing"
r.choices["team"].probabilities  # {"billing": 0.947, "eng": 0.019, "shipping": 0.034}
```

### Environment variables

| Variable | Backend | Meaning |
|---|---|---|
| `DECIDE_LOCAL_MODEL` | `laya`, `laya_mlx` | Hugging Face repo id (or local path) of the model to load. Required to auto-select either backend. |
| `TYPESAFE_API_KEY` | `typesafe` | API key, sent as `Authorization: Bearer`. Required to auto-select `typesafe`. |
| `TYPESAFE_BASE_URL` | `typesafe` | Overrides the default `https://api.typesafe.ai`. |
| `OPENROUTER_API_KEY` | `openrouter` | API key, sent as `Authorization: Bearer`. Required to auto-select `openrouter`. |
| `DECIDE_LLM_BASE_URL` | `llm` | Base URL of an OpenAI-compatible chat-completions server. Setting it (or `OPENAI_API_KEY`) auto-selects `llm`. |
| `OPENAI_API_KEY` | `llm` | API key, sent as `Authorization: Bearer`, if the server needs one. |
| `DECIDE_LLM_MODEL` | `llm` | Model name to request; defaults to `gpt-4o-mini`. |
| `DECIDE_BACKENDS` | `Client.from_env` | Comma-separated backend names, overriding auto-detection entirely. |
| `DECIDE_MIN_CONFIDENCE` | `Client.from_env` (`Gate`) | Float threshold for the default `Gate` built by `from_env`, when no explicit `policy` is passed. |

## Gating and fallback

```python
from decide import Gate

Gate(
    min_confidence=0.0,  # top probability of a Choice, max(noul, 1-noul) for a Noul
    per_question=None,  # optional {"question_name": threshold} overrides
    on_error="next",  # or "raise" to stop the chain on the first BackendError
)
```

For each backend in order: call it. On `BackendError` with `on_error="next"`,
append `"<name>:error"` to `meta.route` and try the next backend (with
`on_error="raise"`, the error propagates immediately instead). If the
response fails the gate, append `"<name>:low_confidence:<question>=<value><threshold>"`
and try the next backend. If it passes, append `"<name>:ok"` and return.
Score answers are never gated (there is no single confidence number for an
expected value over levels).

If every backend is exhausted without a passing response, the best response
seen so far (highest minimum confidence across its gated answers) is
returned, with `meta.route[-1] == "<name>:accepted_low_confidence"`. If no
backend produced any response at all, `Client.decide` raises
`AllBackendsFailed(route, errors)`.

Route strings you will see in `meta.route`:

- `"<name>:ok"` - the backend answered and passed the gate.
- `"<name>:error"` - the backend raised a `BackendError`.
- `"<name>:low_confidence:<question>=<value><threshold>"` - the backend
  answered but at least one gated question fell below its threshold.
- `"<name>:accepted_low_confidence"` - appended once, at the end of the
  route, when no backend passed the gate and the best low-confidence
  response was returned instead.

`decide_batch`/`AsyncClient.decide_batch` run the same chain per input
state (using a backend's real batch path when `capabilities().batch` is
true), preserving input order; a state that clears the gate on an earlier
backend is not sent to later ones. If any state in the batch is left
unrouted, `AllBackendsFailed` carries `partial` (every `Response` that did
resolve, keyed by input index) and `failed` (the route so far for every
state that did not), so the resolved siblings are not silently lost.

## Server: point TypeSafe's SDK at a local model

Requires the `server` extra:

```bash
pip install "pydecide[server]"
DECIDE_LOCAL_MODEL=aac6fef/laya-multilingual-mlx decide serve --backends laya_mlx --port 8811
```

`GET /health` reports liveness and the configured backend names;
`GET /v1/models` lists them in an OpenAI-style shape; `POST /v1/systemone`
takes a TypeSafe `SystemOneRequest` body and returns a
`SystemOneResponse`-shaped body plus a `decide` extension carrying our own
backend/route metadata. Errors come back as
`{"error": {"message": ..., "type": ...}}` with a matching HTTP status.

A raw request against the running server above:

```bash
curl -s -X POST http://127.0.0.1:8811/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": {"ticket": "I was charged twice, please refund."},
    "questions": {
      "team": {"type": "choice", "instructions": "Which team handles this?",
               "criteria": {"billing": "charges, refunds", "engineering": "bugs and outages", "sales": "pricing and upgrades"}},
      "severity": {"type": "score", "instructions": "How severe is this?", "criteria": ["minor", "degraded", "blocked"]},
      "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"}
    }
  }'
```

```json
{
  "model": "aac6fef/laya-multilingual-mlx",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 1.0,
              "probabilities": {"billing": 1.0, "engineering": 0.0, "sales": 0.0}},
    "severity": {"type": "score", "score": 1.5522, "confidence": 0.5719,
                 "legend": {"0": "minor", "1": "degraded", "2": "blocked"},
                 "probabilities": {"0": 0.0197, "1": 0.4083, "2": 0.5719}},
    "refund": {"type": "noul", "noul": 0.9461}
  },
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "decide": {"backend": "laya_mlx", "latency_ms": 50.5, "route": ["laya_mlx:ok"]}
}
```

And the same request through TypeSafe's own SDK, pointed at the local
server (no API key is needed since this server has none configured, but the
SDK requires a non-empty string):

```python
from typesafe_sdk import TypeSafeClient, Choice, Score, Noul

client = TypeSafeClient(api_key="anything", base_url="http://127.0.0.1:8811")

resp = client.system_one(
    state={"ticket": "I was charged twice, please refund."},
    questions={
        "team": Choice(
            instructions="Which team handles this?",
            criteria={
                "billing": "charges, refunds",
                "engineering": "bugs and outages",
                "sales": "pricing and upgrades",
            },
        ),
        "severity": Score(
            instructions="How severe is this?", criteria=["minor", "degraded", "blocked"]
        ),
        "refund": Noul(instructions="Does the customer ask for money back?"),
    },
)
```

```text
model='aac6fef/laya-multilingual-mlx' usage=Usage(input_tokens=0, output_tokens=0)
answers={'team': ChoiceAnswer(type='choice', choice='billing', confidence=1.0,
  probabilities={'billing': 1.0, 'engineering': 0.0, 'sales': 0.0}),
 'severity': ScoreAnswer(type='score', score=1.5522, confidence=0.5719,
  legend={0: 'minor', 1: 'degraded', 2: 'blocked'},
  probabilities={0: 0.0197, 1: 0.4083, 2: 0.5719}),
 'refund': NoulAnswer(type='noul', noul=0.9461)}
```

TypeSafe's SDK parsed our server's response without modification: it is a
genuine `SystemOneResponse`, not a hand-shaped dict.

## CLI

```bash
decide ask "I was charged twice, please refund." \
  --choice "team=billing,engineering,sales" \
  --score "severity=minor,degraded,blocked" \
  --noul "refund=Does the customer ask for money back?"
```

```text
NAME      TYPE    ANSWER    PROBABILITIES
team      choice  billing   billing=0.93 engineering=0.02 sales=0.05
severity  score   degraded  1.09 (degraded)
refund    noul    true      0.87
backend=laya_mlx route=laya_mlx:ok latency=36.9ms
```

`--json` prints a machine-readable payload instead of the table.
`--min-confidence` and `--model` are also available on `ask`.

```bash
decide backends
```

```text
typesafe  installed=yes  configured=no
openrouter  installed=yes  configured=no
laya  installed=yes  configured=yes
laya_mlx  installed=yes  configured=yes
crossencoder  installed=yes  configured=n/a
llm  installed=yes  configured=no
```

`decide serve --backends a,b --host 127.0.0.1 --port 8811 [--api-key TOKEN] [--min-confidence FLOAT]`
runs the HTTP server described above.

## What the probabilities mean

Every probability in a `ChoiceAnswer`, `ScoreAnswer` or `NoulAnswer` is
whatever the backend reported; `decide` does not calibrate, smooth or
verify it. What that means differs by backend: `typesafe`, `openrouter` and
`laya`/`laya_mlx` are purpose-built decision models, but their outputs are
still self-reported by the model and not audited by this library. The
`crossencoder` backend turns a relevance reranker's raw logits into a
softmax or sigmoid - the result is a *normalized score* forced to distribute
mass over the supplied candidates, not a calibrated probability; a `Choice`
will still pick a winner even when every candidate is a bad fit, and a
`Noul` of 0.9 does not mean the condition holds 90% of the time. The `llm`
backend is the least trustworthy of all: it prompts a general chat model to
estimate its own confidence in JSON, with no guarantee the model attends to
every candidate or keeps its numbers well calibrated. Treat all of this
accordingly - as a signal to gate and fall back on, not as ground truth.

## Status

`pydecide` is at 0.1.0. The public API may still change before a 1.0
release.

## License

MIT, see [LICENSE](LICENSE).
