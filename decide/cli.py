"""Command line interface for decide (`decide ask`, `decide backends`, `decide serve`)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence

from decide.backends import REGISTRY, available
from decide.client import Client
from decide.errors import ConfigError, DecideError
from decide.gate import Gate
from decide.types import (
    Answer,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Response,
    Score,
    ScoreAnswer,
)
from decide.wire import to_wire_answers

# Env vars that make a backend "configured" for `decide backends`. A backend
# absent from this mapping (currently only `crossencoder`) is not
# environment-configurable and always reports "n/a".
_ENV_VARS: dict[str, tuple[str, ...]] = {
    "typesafe": ("TYPESAFE_API_KEY",),
    "openrouter": ("OPENROUTER_API_KEY",),
    "llm": ("DECIDE_LLM_BASE_URL", "OPENAI_API_KEY"),
    "laya": ("DECIDE_LOCAL_MODEL",),
    "laya_mlx": ("DECIDE_LOCAL_MODEL",),
}


def make_client(env: Mapping[str, str] | None, policy: Gate | None) -> Client:
    """Build a `Client` from an environment mapping and policy.

    A thin wrapper around `Client.from_env` so tests can monkeypatch client
    construction (e.g. `decide.cli.make_client`) without touching real
    backends or the real environment.
    """
    return Client.from_env(env=env, policy=policy)


def run_server(app: object, host: str, port: int) -> None:
    """Run `app` with uvicorn. A seam so tests can monkeypatch it without importing uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise ConfigError(
            "the server needs uvicorn; install it with the 'pydecide[server]' extra"
        ) from exc
    uvicorn.run(app, host=host, port=port)


def _parse_name_value(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {raw!r}")
    name, _, value = raw.partition("=")
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, got {raw!r}")
    return name, value


def _build_parser() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser(prog="decide", description="One CLI for every decision model.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Ask one or more questions about some state text")
    ask.add_argument("text", metavar="TEXT", help="The state/content to decide about")
    ask.add_argument(
        "--choice",
        metavar="NAME=a,b,c",
        action="append",
        default=[],
        type=_parse_name_value,
        help="A Choice question: comma-separated candidate names",
    )
    ask.add_argument(
        "--score",
        metavar="NAME=lo,mid,hi",
        action="append",
        default=[],
        type=_parse_name_value,
        help="A Score question: comma-separated levels, lowest to highest",
    )
    ask.add_argument(
        "--noul",
        metavar="NAME=question text",
        action="append",
        default=[],
        type=_parse_name_value,
        help="A Noul (yes/no) question: its instruction text",
    )
    ask.add_argument(
        "--instructions",
        metavar="TEXT",
        default=None,
        help="Default instructions for --choice/--score questions that don't carry their own",
    )
    ask.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    ask.add_argument("--min-confidence", type=float, default=None, metavar="FLOAT")
    ask.add_argument("--model", default=None, metavar="NAME")

    subparsers.add_parser("backends", help="List available and configured backends")

    serve = subparsers.add_parser("serve", help="Run the decide HTTP server")
    serve.add_argument(
        "--backends", default=None, metavar="a,b", help="Comma-separated backend names"
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8811)
    serve.add_argument(
        "--api-key",
        default=None,
        metavar="TOKEN",
        help="Bearer token required to call the server. Falls back to the DECIDE_API_KEY "
        "environment variable when omitted; this flag takes precedence over it.",
    )
    serve.add_argument("--min-confidence", type=float, default=0.0, metavar="FLOAT")

    return parser, ask


def _build_questions(
    args: argparse.Namespace, ask_parser: argparse.ArgumentParser
) -> dict[str, Question]:
    questions: dict[str, Question] = {}
    try:
        for name, value in args.choice:
            candidates = [c.strip() for c in value.split(",") if c.strip()]
            instructions = (
                args.instructions if args.instructions is not None else f"Answer for: {name}"
            )
            questions[name] = Choice(instructions, {c: None for c in candidates})
        for name, value in args.score:
            levels = [c.strip() for c in value.split(",") if c.strip()]
            instructions = (
                args.instructions if args.instructions is not None else f"Answer for: {name}"
            )
            questions[name] = Score(instructions, levels)
        for name, value in args.noul:
            instructions = (
                value.strip()
                if value.strip()
                else (args.instructions if args.instructions is not None else f"Answer for: {name}")
            )
            questions[name] = Noul(instructions)
    except ValueError as exc:
        ask_parser.error(str(exc))
    return questions


def _format_row(name: str, answer: Answer) -> tuple[str, str, str, str]:
    if isinstance(answer, ChoiceAnswer):
        probabilities = " ".join(f"{k}={v:.2f}" for k, v in answer.probabilities.items())
        return name, "choice", answer.choice, probabilities
    if isinstance(answer, ScoreAnswer):
        idx = max(0, min(round(answer.score), len(answer.levels) - 1))
        nearest = answer.levels[idx]
        return name, "score", nearest, f"{answer.score:.2f} ({nearest})"
    if isinstance(answer, NoulAnswer):
        decision = "true" if answer.noul >= 0.5 else "false"
        return name, "noul", decision, f"{answer.noul:.2f}"
    raise TypeError(f"unknown answer type: {type(answer)!r}")


def _padded_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render a header + rows as aligned columns, with trailing whitespace stripped per line."""
    widths = [max([len(header[i]), *(len(row[i]) for row in rows)]) for i in range(len(header))]

    def _fmt(row: Sequence[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip()

    return [_fmt(header), *(_fmt(row) for row in rows)]


def _print_table(response: Response) -> None:
    header = ("NAME", "TYPE", "ANSWER", "PROBABILITIES")
    rows = [_format_row(name, answer) for name, answer in response.answers.items()]
    for line in _padded_table(header, rows):
        print(line)

    route = ",".join(response.meta.route)
    print(f"backend={response.meta.backend} route={route} latency={response.meta.latency_ms:.1f}ms")


def _print_json(response: Response) -> None:
    payload = {
        "answers": to_wire_answers(response),
        "meta": {
            "backend": response.meta.backend,
            "model": response.meta.model,
            "latency_ms": response.meta.latency_ms,
            "route": response.meta.route,
        },
    }
    print(json.dumps(payload))


def _cmd_ask(args: argparse.Namespace, ask_parser: argparse.ArgumentParser) -> int:
    if not (args.choice or args.score or args.noul):
        ask_parser.error("at least one question is required: use --choice, --score, and/or --noul")

    questions = _build_questions(args, ask_parser)

    policy = Gate(min_confidence=args.min_confidence) if args.min_confidence is not None else None
    # `Client.from_env` reads `os.environ` itself when `env` is omitted/None.
    client = make_client(None, policy)
    try:
        response = client.decide(args.text, questions, model=args.model)
    finally:
        client.close()

    if args.json:
        _print_json(response)
    else:
        _print_table(response)
    return 0


def _backend_status(env: Mapping[str, str]) -> list[tuple[str, bool, str]]:
    """Report `(name, installed, configured)` for every registered backend."""
    installed_by_name = available()
    rows = []
    for name in REGISTRY:
        installed = installed_by_name.get(name, False)
        env_vars = _ENV_VARS.get(name)
        if env_vars is None:
            configured = "n/a"
        else:
            configured = "yes" if any(v in env for v in env_vars) else "no"
        rows.append((name, installed, configured))
    return rows


def _cmd_backends(args: argparse.Namespace) -> int:
    header = ("NAME", "INSTALLED", "CONFIGURED")
    rows = [
        (name, "yes" if installed else "no", configured)
        for name, installed, configured in _backend_status(os.environ)
    ]
    for line in _padded_table(header, rows):
        print(line)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    env = dict(os.environ)
    if args.backends:
        env["DECIDE_BACKENDS"] = args.backends
    policy = Gate(min_confidence=args.min_confidence)
    client = make_client(env, policy)

    from decide.server import create_app

    api_key = args.api_key if args.api_key is not None else env.get("DECIDE_API_KEY")
    app = create_app(client, api_key=api_key)
    run_server(app, args.host, args.port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser, ask_parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        if args.command == "ask":
            return _cmd_ask(args, ask_parser)
        if args.command == "backends":
            return _cmd_backends(args)
        if args.command == "serve":
            return _cmd_serve(args)
        parser.error(f"unknown command {args.command!r}")
        return 2  # unreachable: parser.error always raises SystemExit
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 2
    except DecideError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
