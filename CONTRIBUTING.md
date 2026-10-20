# Contributing

## Running it

```bash
uv sync --all-extras    # or `uv sync` for the client alone
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

CI runs those four on Python 3.10 through 3.14. All four have to pass.

The local backends need their extras: `laya` and `mlx` pull PyTorch and MLX,
about 2 GB. The suite passes without them; the tests that need a real model are
skipped.

## The bar for a change

- A behaviour change comes with a test that fails without it.
- A number in the README or in `docs/` is measured on a machine, and the text
  says which machine and which version. No estimates.
- Public API gets a docstring saying what it does and, where the reason is not
  in the code, why. Short.
- `py.typed` ships, so annotations are part of the contract. `uv run mypy` has
  to stay clean.

## Backends

A backend answers `Choice`, `Score` and `Noul` and reports `Capabilities`. If it
would answer a question while silently dropping part of it, say so in
`Capabilities` instead, and the chain hands that question to the next backend.
`decide/backends/base.py` is the contract and `docs/backends.md` the tour.

## Releases

See [RELEASING.md](RELEASING.md). Short version: a release is a finished change,
not a commit.
