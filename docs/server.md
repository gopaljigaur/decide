# Server: point TypeSafe's SDK at a local model

laya ships its own `laya-serve` command as of 0.3.7, so if you only run
laya behind a single endpoint, that's the more direct option. Reach for
`decide serve` instead when you want to chain several backends, local
or hosted, behind one TypeSafe-compatible endpoint, with the same
fallback and gating behavior as the Python client.

```bash
decide serve --backends laya_mlx --port 8811
```

`GET /health` reports liveness and the configured backend names;
`GET /v1/models` lists them in an OpenAI-style shape; `POST /v1/systemone`
takes a TypeSafe `SystemOneRequest` body and returns a
`SystemOneResponse`-shaped body plus a `decide` extension carrying our own
backend/route metadata. Errors come back as
`{"error": {"message": ..., "type": ...}}` with a matching HTTP status.

If `--api-key`/`DECIDE_API_KEY` is set, requests must send a matching
`Authorization: Bearer <token>` header; the key must be ASCII, since HTTP
header bytes are latin-1 decoded by the server before comparison.

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
  "model": "aac6fef/laya-mlx",
  "answers": {
    "team": {"type": "choice", "choice": "billing", "confidence": 0.9763,
              "probabilities": {"billing": 0.9763, "engineering": 0.0107, "sales": 0.013}},
    "severity": {"type": "score", "score": 1.6284, "confidence": 0.7086,
                 "legend": {"0": "minor", "1": "degraded", "2": "blocked"},
                 "probabilities": {"0": 0.0802, "1": 0.2112, "2": 0.7086}},
    "refund": {"type": "noul", "noul": 0.8919}
  },
  "usage": {"input_tokens": 0, "output_tokens": 0},
  "decide": {"backend": "laya_mlx", "latency_ms": 69.0, "route": ["laya_mlx:ok"]}
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
model='aac6fef/laya-mlx' usage=Usage(input_tokens=0, output_tokens=0)
answers={'team': ChoiceAnswer(type='choice', choice='billing', confidence=0.9763,
  probabilities={'billing': 0.9763, 'engineering': 0.0107, 'sales': 0.013}),
 'severity': ScoreAnswer(type='score', score=1.6284, confidence=0.7086,
  legend={0: 'minor', 1: 'degraded', 2: 'blocked'},
  probabilities={0: 0.0802, 1: 0.2112, 2: 0.7086}),
 'refund': NoulAnswer(type='noul', noul=0.8919)}
```

TypeSafe's SDK parsed our server's response without modification: it is a
genuine `SystemOneResponse`, not a hand-shaped dict.

Back to the [README](../README.md).
