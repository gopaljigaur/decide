# Backends

| Module | Backend name | Extra | Notes |
|---|---|---|---|
| `typesafe.py` | `typesafe` | none (httpx only) | `POST {base_url}/v1/systemone`, bearer auth. Default base URL `https://api.typesafe.ai`, default model `jev-latest`. |
| `openrouter.py` | `openrouter` | none | Same wire shape as `typesafe`, `POST https://openrouter.ai/api/alpha/decisions`, default model `typesafe/jev-latest`. |
| `openai_decisions.py` | `openai_decisions` | none (httpx only) | OpenAI Decisions API (public beta), `POST https://api.openai.com/v1/decisions`, bearer auth, default model `gpt-6-luna`. A `Noul` is a `predicate` question; a `Noul`'s true/false descriptions are appended to its instructions, since the API has no field for them. A refusal raises `BadResponseError`. |
| `laya.py` | `laya` | `pydecide[laya]` | Local PyTorch `laya.Agent`, default model `convaiinnovations/laya` when constructed directly. |
| `laya_mlx.py` | `laya_mlx` | `pydecide[mlx]` (Python 3.11+) | Local MLX `laya_mlx.Agent` (Apple Silicon), default model `aac6fef/laya-mlx` when constructed directly. |
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

### Environment variables: overrides and hosted-backend keys

None of these are required to get started. See
[Quickstart](../README.md#quickstart-zero-configuration) in the README. They
either override a local backend that `from_env` already auto-selects once
it's installed, or supply the API key a hosted backend needs to be
auto-selected at all.

| Variable | Backend | Meaning |
|---|---|---|
| `DECIDE_LOCAL_MODEL` | `laya`, `laya_mlx` | Overrides the default Hugging Face repo id (or local path) of the model to load. Optional: omit it and the installed local backend auto-selects with its own default model. |
| `TYPESAFE_API_KEY` | `typesafe` | API key, sent as `Authorization: Bearer`. Required to auto-select `typesafe`. |
| `TYPESAFE_BASE_URL` | `typesafe` | Overrides the default `https://api.typesafe.ai`. |
| `OPENROUTER_API_KEY` | `openrouter` | API key, sent as `Authorization: Bearer`. Required to auto-select `openrouter`. |
| `OPENAI_API_KEY` | `openai_decisions`, `llm` | API key, sent as `Authorization: Bearer`. Setting it auto-selects `openai_decisions` and `llm`; the `llm` backend sends it only if the server needs one. |
| `DECIDE_OPENAI_DECISIONS` | `openai_decisions` | Set to `0` to stop `OPENAI_API_KEY` from auto-selecting `openai_decisions`. |
| `DECIDE_OPENAI_DECISIONS_MODEL` | `openai_decisions` | Model to request; defaults to `gpt-6-luna`. |
| `DECIDE_LLM_BASE_URL` | `llm` | Base URL of an OpenAI-compatible chat-completions server. Setting it (or `OPENAI_API_KEY`) auto-selects `llm`. |
| `DECIDE_LLM_MODEL` | `llm` | Model name to request; defaults to `gpt-4o-mini`. |
| `DECIDE_BACKENDS` | `Client.from_env` | Comma-separated backend names, overriding auto-detection entirely. |
| `DECIDE_MIN_CONFIDENCE` | `Client.from_env` (`Gate`) | Float threshold for the default `Gate` built by `from_env`, when no explicit `policy` is passed. |
| `DECIDE_API_KEY` | `decide serve` | Bearer token required to call the server, when `--api-key` is not passed. The flag takes precedence over this variable. |

### Noul labels

`Noul("Is the sentiment positive?", labels={"true": "positive", "false": "negative"})`
renames the two options the model scores. The default `true`/`false` pair carries
no task meaning, and the published laya checkpoints answer a constant 0.000 at
confidence 1.0 to a `noul` question containing the word "positive" until the
labels are named ([laya#1244](https://github.com/NandhaKishorM/laya/issues/1244)).
Naming them takes that question from 0.000 to 0.979 on a positive review.

Not every backend can carry them. One that would answer while dropping them
refuses the question instead, so a chain hands it to the next backend and the
route records the refusal:

| Backend | `labels` |
|---|---|
| `laya` | rendered, with laya 0.3.8 or newer |
| `laya_mlx` | rendered, with laya-mlx 0.3.0 or newer |
| `typesafe`, `openrouter`, `openai_decisions` | sent to the server, which decides |
| `llm` | named in the prompt |
| `crossencoder` | refused: a reranker has no slot for them |

### Which laya checkpoint

Three are published and they are not interchangeable. Measured on 150 rows of
ag_news (four-way `Choice`) and 150 of sst2 (yes/no `Noul`), MLX on an M4, ECE
scored on the confidence the checkpoint reports, with the method and per-row
output in
[system-one-bench](https://github.com/gopaljigaur/system-one-bench):

| Checkpoint | ag_news | ECE | sst2 | ECE | sst2, options named | ECE | p50 |
|---|---|---|---|---|---|---|---|
| `aac6fef/laya-mlx` (English, the default) | 0.953 | 0.106 | 0.833 | **0.053** | 0.887 | **0.049** | 39.0 ms |
| `aac6fef/laya-typed-decisions-mlx` | 0.953 | 0.365 | **0.847** | 0.143 | **0.907** | 0.091 | 37.8 ms |
| `aac6fef/laya-multilingual-mlx` | 0.920 | **0.047** | 0.527 | 0.373 | 0.853 | 0.103 | **15.0 ms** |

English is the default: it is the best calibrated of the three on both yes/no
tasks, which is what the gate reads, and it ties for the best accuracy on the
four-way choice. The multilingual checkpoint is the best calibrated on that
choice task and the fastest, and it is the only one trained for other languages,
so pass it per backend for input that is not English; its 0.527 on unnamed
yes/no options is mostly the names, since the same rows with the options named
score 0.853. typed-decisions answers yes/no two points better than English and
is three times worse calibrated on it. The `laya` PyTorch backend's default,
`convaiinnovations/laya`, is the same English weights and scores the same.

### Other servers that speak the same protocol

A lot of decision models serve `POST /v1/systemone`. The `typesafe` backend
speaks that protocol and takes a `base_url`, so pointing it at one of them
needs no new code:

```python
from decide import Client
from decide.backends.typesafe import TypeSafeBackend

client = Client([TypeSafeBackend(api_key="unused", base_url="http://localhost:8000")])
```

Or from the shell, with no code at all:

```bash
TYPESAFE_API_KEY=unused TYPESAFE_BASE_URL=http://localhost:8000 \
  DECIDE_BACKENDS=typesafe \
  decide ask "I was charged twice" --choice billing,engineering
```

`DECIDE_BACKENDS` matters here: with a local backend installed, `from_env`
puts it first and the server never gets the question.

The table lists what has actually been run, not everything that claims
compatibility. The `model` column is what the server reported back in
`response.meta.model`.

| Server | Version tested | Model reported | Result |
|---|---|---|---|
| [Von](https://github.com/wfzyx/von) | 1.3.0 | `von-1.3.0` | Choice, Noul and Score all answered |
| [`laya-serve`](https://github.com/NandhaKishorM/laya) | laya 0.4.7 | `laya-rl-agent` | same, on `/v1/systemone` |
| `decide serve` | this package | the chain's backend | covered by the test suite |

Things that cost time while testing these, in case they bite you elsewhere:

- The first request loads the model, which took 9 seconds on CPU here. The
  backend's default timeout is 30 seconds, and a cold server plus a slow disk
  can exceed it. Raise `timeout` for the first call, or warm the server first.
- Servers that need no key still need `api_key` passed, since the backend
  sends an `Authorization` header unconditionally. Any string will do.
- `laya-serve` takes no flags. It reads `LAYA_PORT`, `LAYA_HOST` and the rest
  from the environment, and `--help` starts the server instead of printing
  help, binding `0.0.0.0:8000` by default.

Back to the [README](../README.md).
