# Gating and fallback

```python
from decide import Gate

Gate(
    min_confidence=0.0,  # the backend's own confidence, see below
    per_question=None,  # optional {"question_name": threshold} overrides
    per_type=None,  # optional {"choice"/"score"/"noul": threshold} overrides
    on_error="next",  # or "raise" to stop the chain on the first BackendError
)
```

## Running local only, with no escalation

A chain escalates by sending the state to the next backend, and the next
backend may be a hosted API. To keep every decision on the machine that makes
it, build the chain from local backends and tell the client to abstain instead
of accepting an answer that missed the gate:

```python
from decide import Client, Gate, LowConfidence
from decide.backends.laya_mlx import LayaMLXBackend

client = Client(
    [LayaMLXBackend()],
    Gate(min_confidence=0.8),
    on_low_confidence="raise",
)

try:
    r = client.decide(state, questions)
except LowConfidence as abstained:
    ...  # abstained.response is the answer that missed, abstained.route is why
```

`on_low_confidence` defaults to `"accept"`, which returns the best answer seen
with `"<name>:accepted_low_confidence"` on the route. With `"raise"` a low answer
is an outcome you handle instead of a reason to call out.

A chain with one backend has nowhere to hand off to, so a low answer ends the
chain either way: accepted with `"<name>:accepted_low_confidence"`, or raised as
`LowConfidence`. Escalation is what the next backend is for, and there is no
next backend.

### Enforcing it for a whole deployment

The default chain discovers only installed local backends. Hosted inference
requires an explicit choice in `decide init`, a config file, Python code, or
`DECIDE_BACKENDS`. API keys alone do not add backends.

`DECIDE_LOCAL_ONLY=1` additionally enforces local inference across configuration
sources. Selecting a server backend in the config or `DECIDE_BACKENDS` raises
`ConfigError`; passing a server backend directly to `Client(...)` raises too.
Custom backends must declare `Capabilities(local=True)` to pass this check.

## Which number the gate reads

The backend's own `confidence` when it reports one, and the top probability of
a `Choice` or `max(noul, 1 - noul)` of a `Noul` when it does not.

Those are not the same number. On one ag_news story von reports 0.877 while its
top probability is 0.9077, and laya reports 0.906 against 0.9779. On a two
option choice laya has reported 0.5909 against a top probability of 0.918. A
model that states a confidence means that number, so the gate uses it; the
computed fallback is for backends that report nothing, such as the reranker.

`answer.confidence` holds what the backend sent, or `None`, so a policy of your
own can read either.

For a question with a threshold set at more than one level, the most
specific one wins: `per_question[name]`, else `per_type[<the question's
type>]`, else `min_confidence`. Score answers are never gated regardless of
which of these sets a threshold for `"score"`.

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

A chain of the hosted Jev model, then OpenAI Decisions, then a local model is
three backends in order. The chain below is illustrative, and was not run
against the live APIs:

```python
from decide import Client
from decide.backends import load

client = Client(
    [
        load("typesafe", api_key="..."),
        load("openai_decisions", api_key="..."),
        load("laya_mlx"),
    ]
)
```

Route strings you will see in `meta.route`:

- `"<name>:ok"`: the backend answered and passed the gate.
- `"<name>:error"`: the backend raised a `BackendError`.
- `"<name>:low_confidence:<question>=<value><threshold>"`: the backend
  answered but at least one gated question fell below its threshold.
- `"<name>:accepted_low_confidence"`: appended once, at the end of the
  route, when no backend passed the gate and the best low-confidence
  response was returned instead.

`Client.decide_batch`/`AsyncClient.decide_batch` run the same chain per
input state, preserving input order; a state that clears the gate on an
earlier backend is not sent to later ones. `Client.decide_batch` uses a
backend's real batch path when `capabilities().batch` is true, looping
`decide` per request otherwise. `AsyncClient.decide_batch` always loops
`adecide` per state, concurrently via `asyncio.gather`; it does not use a
backend's batch path in v1. If any state in the batch is left unrouted,
`AllBackendsFailed` carries `partial` (every `Response` that did resolve,
keyed by input index) and `failed` (the route so far for every state that
did not), so the resolved siblings are not silently lost.

### Choosing a threshold

[Jevals](https://jevals.com/choice/), an independent Jev benchmark, measured
Jev's own accuracy against the confidence threshold it would take to reach
it, over 1,500 decisions per task:

Banking77, a `Choice` task (overall accuracy 79.7%):

| Threshold | Coverage | Accuracy |
|---|---|---|
| 0.55 | 93% | 83.4% |
| 0.65 | 88% | 85.1% |
| 0.85 | 76% | 88.6% |
| 0.96 | 59% | 94.3% |

PubMedQA, a yes/no `Noul` task (overall accuracy 91.3%):

| Threshold | Coverage | Accuracy |
|---|---|---|
| 0.55 | 98% | 92.0% |
| 0.65 | 94% | 92.7% |
| 0.85 | 69% | 96.4% |
| 0.91 | 49% | 98.6% |

Two things follow from this. First, thresholds below about 0.85 filter
almost nothing, because most of the accuracy gain from gating shows up only once
the threshold climbs well past the model's overall accuracy. Second, Jev
clips its probabilities to the 0.01-0.99 range, so a yes/no answer rarely
exceeds 0.98; a single `min_confidence` applied to both `Choice` and `Noul`
answers therefore behaves very differently across the two. A 0.96
threshold still keeps 59% of choice answers but only about 11% of yes/no
answers. Use `per_type` to set separate thresholds:

```python
from decide import Gate

Gate(per_type={"choice": 0.85, "noul": 0.95})
```

### Gating on something other than the top probability

`Gate` escalates when the winning probability falls below a number you pick.
That is the cascade baseline from
[FrugalGPT](https://arxiv.org/abs/2305.05176), and it has two known limits.
The probability it reads is usually not calibrated, so 0.85 does not mean
right 85% of the time. And the top probability alone discards the shape of
the rest of the distribution: `{a: 0.51, b: 0.02, c: 0.02, d: 0.45}` and
`{a: 0.51, b: 0.17, c: 0.16, d: 0.16}` are the same number to `Gate` and are
not the same answer.

Four alternatives ship in `decide.policies`. Each is a drop-in for `Gate` in
`Client(policy=...)`, and each skips `Score` answers for the reason above.

| Policy | Escalates when | Parameter |
|---|---|---|
| `EntropyGate` | the distribution is spread out | `max_entropy`, normalized to [0, 1] |
| `MarginGate` | first and second place are close | `min_margin` |
| `CalibratedGate` | a rescaled probability is below the threshold | `min_confidence` plus a fitted `temperature` |
| `ConformalGate` | more than one candidate survives the cut | `quantile`, fitted for a coverage rate |

```python
from decide import Client, EntropyGate

# A three-way near-tie escalates even though its winner clears 0.5.
client = Client(backends, policy=EntropyGate(max_entropy=0.75))
```

The reason string names which rule fired, and it reaches
`response.meta.route` the same way `Gate`'s does:

```
['crossencoder:high_entropy:team=0.89>0.75', 'laya:ok']
```

#### Fitting the parameters instead of guessing them

`CalibratedGate` and `ConformalGate` take numbers you fit on answers whose
correct label you already know, with no dependency beyond the standard
library:

```python
from decide import CalibratedGate, ConformalGate, fit_conformal_quantile, fit_temperature

# rows: (the probabilities a backend reported, the index of the right label)
rows = [([0.95, 0.05], 0), ([0.88, 0.12], 1), ...]

temperature = fit_temperature(rows)  # above 1.0 means it was overconfident
quantile = fit_conformal_quantile(rows, alpha=0.1)  # for 90% coverage

client = Client(backends, policy=CalibratedGate(min_confidence=0.85, temperature=temperature))
client = Client(backends, policy=ConformalGate(quantile=quantile))
```

`fit_temperature` returns exactly 1.0 for a backend that is already honest,
and above 1.0 for one that is not: rows claiming 0.95 while being right 70%
of the time fit a temperature of 3.48, which pulls every such answer under an
0.9 threshold that `Gate` would have waved through.

`fit_conformal_quantile` gives the stronger promise. Asking for 95, 90 and 80
percent coverage on synthetic rows returned 94.5, 89.6 and 80.3 percent on
held-out data, with the mean prediction set shrinking from 2.26 candidates to
1.44 as the target loosens. The guarantee is exchangeability, so it holds for
inputs that look like the ones you fitted on and says nothing about a
different workload.

Save a fit and load it next to the code that uses it:

```python
from decide import Calibration

Calibration(temperatures={"team": 3.48}, quantiles={"team": 0.82}).save("calibration.json")
loaded = Calibration.load("calibration.json")
```

Back to the [README](../README.md).
