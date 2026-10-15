# Already using typesafe-sdk?

Swap the import. Your questions, your `system_one` calls and the response
object stay the same, and with no other change the hosted API is the only
backend, exactly as before.

```python
from decide import TypeSafeClient  # was: from typesafe_sdk import TypeSafeClient
```

Add `fallback=` and the call keeps answering when the hosted API does not.
Here the hosted tier is a transport that always returns 503, so the local
model takes over:

```python
import httpx
from decide import TypeSafeClient
from typesafe_sdk import Choice

hosted_down = httpx.MockTransport(lambda request: httpx.Response(503, text="down"))
client = TypeSafeClient(api_key="sk-demo", transport=hosted_down, fallback=["laya_mlx"])
result = client.system_one(
    "I was charged twice for the same order, please refund the duplicate.",
    {
        "team": Choice(
            instructions="Which team owns this?", criteria={"billing": None, "engineering": None}
        )
    },
)
print(result.choices["team"].choice, round(result.choices["team"].confidence, 3))
print(result.decide.backend, result.decide.model)
print(result.decide.route)
```

```
billing 0.946
laya_mlx aac6fef/laya-mlx
['typesafe:error', 'laya_mlx:ok']
```

The result is a subclass of the SDK's own `SystemOneResponse`, so existing code
that reads `answers`, `choices` or `usage` keeps working; `result.decide` adds
the backend, model, latency and route. Questions can be the SDK's `Choice`,
`Score` and `Noul`, decide's own, or a mix. `typesafe-sdk` is not a dependency
of `pydecide`; install it yourself to use this layer. `AsyncTypeSafeClient`
works the same way.

Errors keep the vendor types: a failed hosted call raises the matching
`TypeSafeAPIError` subclass (`TypeSafeInternalServerError` for a 503,
`TypeSafeAPIConnectionError` for a transport failure, and so on), so existing
`except` clauses still catch it. With a `fallback` and every tier failing, the
error is both a `TypeSafeError` and a `decide.AllBackendsFailed`.

`retry=` takes a `typesafe_sdk.RetryPolicy` and retries the hosted tier before
the chain moves on. It honours `max_retries`, `backoff_initial`, `backoff_max`,
`backoff_jitter`, `http_statuses`, `api_connection_error` and
`api_timeout_error`; it ignores `respect_retry_after`, `exceptions`,
`predicate` and `timeout`. Not supported yet: `response_model=`, `http_client=`,
and per-call `retry=`, `extra_headers=`, `extra_body=` and `timeout=` (each
raises `NotImplementedError`).

Back to the [README](../README.md).
