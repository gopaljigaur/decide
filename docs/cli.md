# CLI

`--choice`, `--score` and `--noul` question names are optional. An unnamed
flag is auto-named after the flag itself (`choice`, `score`, `noul`); a
second unnamed flag of the same type becomes `choice2`, `score2`, `noul2`,
and so on:

```bash
decide ask "I had a rough day" --noul "Is the writer doing well?" --choice good,bad
```

```text
NAME    TYPE    ANSWER  PROBABILITIES
choice  choice  bad     good=0.07 bad=0.93
noul    noul    false   0.02
backend=laya_mlx route=laya_mlx:ok latency=65.3ms
```

Name a question explicitly with `NAME=...` to control what shows up in the
table and in `--json`; named and unnamed questions can be mixed on the same
command line:

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
backend=laya_mlx route=laya_mlx:ok latency=34.8ms
```

`--json` prints a machine-readable payload instead of the table, and writes
only that JSON to stdout, so `decide ask ... --json | jq .` is safe to use
in scripts. `--min-confidence` and `--model` are also available on `ask`.
By default, `ask` and `serve` set `HF_HUB_DISABLE_PROGRESS_BARS=1` (unless
it's already set) so a local backend's model-loading progress bars don't
mix into that output; pass `--verbose` to see them. When shown, that
progress goes to stderr, never stdout.

```bash
decide backends
```

```text
NAME              INSTALLED  CONFIGURED  INSTALL
typesafe          yes        no
openrouter        yes        no
openai_decisions  yes        no
laya              yes        default
laya_mlx          yes        default
crossencoder      yes        n/a
llm               yes        no
```

Every backend with `INSTALLED` `no` gets an `INSTALL` column naming the
exact command to add it, e.g. `pip install "pydecide[laya]"`. For `laya`
and `laya_mlx`, `CONFIGURED` reads `default` when installed with no
`DECIDE_LOCAL_MODEL` override (it will be auto-selected with its own
default model), `yes` when `DECIDE_LOCAL_MODEL` is set, and `no` only when
the backend isn't installed at all.

`decide serve --backends a,b --host 127.0.0.1 --port 8811 [--api-key TOKEN] [--min-confidence FLOAT]`
runs the HTTP server described above.

Back to the [README](../README.md).
