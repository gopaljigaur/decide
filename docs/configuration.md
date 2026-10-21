# Configuration

Run `decide init` in your project directory. It asks which backends to try, in
order, their optional models and server URLs, and the minimum confidence for
accepting an answer. The default backend is local. Choosing a server backend
explicitly allows it to receive your state when the chain reaches it.

Setup saves `decide.json` after confirmation. Existing files require an extra
confirmation before replacement. It does not install dependencies, load models,
make inference requests, or save credentials. A missing extra is shown beside
the backend. Set API keys in the environment before using the selected servers.

## Python and files

```python
from decide import Client, AsyncClient, Config

config = Config(
    backends=["laya_mlx", "typesafe"],
    options={"typesafe": {"model": "jev-latest"}},
    min_confidence=0.85,
    on_low_confidence="raise",
)
client = Client(config=config)
async_client = AsyncClient(config="decide.json")
```

`Client()` and `AsyncClient()` read `decide.json` from the current directory.
Without a file, they discover `laya_mlx`, or `laya` when MLX is not installed.
They never prompt. An empty `backends` list also requests local discovery.

The file contains the same fields:

```json
{
  "backends": ["laya_mlx", "typesafe"],
  "options": {"typesafe": {"model": "jev-latest"}},
  "min_confidence": 0.85,
  "on_low_confidence": "raise"
}
```

`options` contains backend constructor arguments. A crossencoder requires a
`model` entry. `Config.load(path)` and `config.save(path)` also work directly;
`save` refuses an existing file unless `overwrite=True` is explicit. JSON works
on every supported Python version without an additional parser dependency.

## Overrides

- `Client(config=...)` uses that config's backend list and gate defaults.
  Explicit `policy=` and `on_low_confidence=` arguments override those defaults.
  Backend options override legacy environment settings; credentials are read
  from the environment unless explicitly supplied in the backend options.
- `Client([backend, ...])` continues to work. Passing both backend instances and
  `config=` raises `ConfigError`.
- `Client.from_env()` and `AsyncClient.from_env()` read the default file and
  retain legacy explicit overrides: `DECIDE_BACKENDS` replaces the chain and
  `DECIDE_MIN_CONFIDENCE` replaces its threshold. A supplied `env={...}` skips
  default file discovery; pass `config=` too to combine them.
- CLI `ask` and `serve` load the default file, or the path supplied through
  `--config`. `--min-confidence` overrides the saved threshold. `serve
  --backends` overrides the chain. Legacy environment overrides also apply.
- `DECIDE_LOCAL_ONLY=1` enforces local inference even with an explicit config.

API keys and base URLs never select backends. For migration from 0.2.18, run
`decide init` to explicitly select any hosted fallbacks you used previously, or
set `DECIDE_BACKENDS` to that chain. No additional environment switch is needed.

Back to the [README](../README.md).
