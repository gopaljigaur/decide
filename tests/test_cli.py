import json
import os

import decide.cli as cli
from decide.client import Client
from decide.errors import BackendError, ConfigError
from tests.conftest import FakeBackend


def _patch_make_client(monkeypatch, client):
    monkeypatch.setattr(cli, "make_client", lambda env, policy: client)


def test_ask_plain_output_has_table_and_footer(monkeypatch, capsys):
    client = Client([FakeBackend("fake")])
    _patch_make_client(monkeypatch, client)

    code = cli.main(
        ["ask", "charged twice", "--choice", "team=billing,eng", "--noul", "refund=Refund?"]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "billing" in out
    assert "backend=fake" in out
    assert "NAME" in out and "TYPE" in out and "ANSWER" in out and "PROBABILITIES" in out


def test_ask_json_output_is_valid_and_has_choice(monkeypatch, capsys):
    client = Client([FakeBackend("fake")])
    _patch_make_client(monkeypatch, client)

    code = cli.main(["ask", "charged twice", "--choice", "team=billing,eng", "--json"])

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answers"]["team"]["choice"] == "billing"
    assert payload["meta"]["backend"] == "fake"
    assert payload["meta"]["route"] == ["fake:ok"]


def test_ask_score_and_noul_rows_render(monkeypatch, capsys):
    client = Client([FakeBackend("fake")])
    _patch_make_client(monkeypatch, client)

    code = cli.main(
        [
            "ask",
            "ticket text",
            "--score",
            "sev=minor,major,blocked",
            "--noul",
            "refund=Did they ask for a refund?",
        ]
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "sev" in out
    assert "refund" in out


def test_ask_requires_at_least_one_question(monkeypatch, capsys):
    code = cli.main(["ask", "x"])

    assert code == 2
    err = capsys.readouterr().err
    assert err.strip() != ""


def test_ask_choice_flag_without_equals_is_usage_error(capsys):
    code = cli.main(["ask", "x", "--choice", "team-no-equals"])

    assert code == 2
    err = capsys.readouterr().err
    assert err.strip() != ""


def test_ask_instructions_default_used_for_choice_and_score(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--choice", "team=billing,eng"])

    assert captured["questions"]["team"].instructions == "Answer for: team"


def test_ask_instructions_flag_overrides_default(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--choice", "team=billing,eng", "--instructions", "Pick a team"])

    assert captured["questions"]["team"].instructions == "Pick a team"


def test_ask_noul_uses_its_own_text_as_instructions(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--noul", "refund=Did they ask for a refund?"])

    assert captured["questions"]["refund"].instructions == "Did they ask for a refund?"


def test_ask_min_confidence_builds_gate(monkeypatch):
    captured = {}

    def _capture_make_client(env, policy):
        captured["policy"] = policy
        return Client([FakeBackend()])

    monkeypatch.setattr(cli, "make_client", _capture_make_client)

    cli.main(["ask", "x", "--choice", "team=billing,eng", "--min-confidence", "0.75"])

    assert captured["policy"] is not None
    assert captured["policy"].min_confidence == 0.75


def test_ask_config_error_exits_1_with_message_on_stderr(monkeypatch, capsys):
    def _raise(env, policy):
        raise ConfigError("no backend configured")

    monkeypatch.setattr(cli, "make_client", _raise)

    code = cli.main(["ask", "x", "--choice", "team=billing,eng"])

    assert code == 1
    err = capsys.readouterr().err
    assert "no backend configured" in err


def test_ask_all_backends_failed_exits_1(monkeypatch, capsys):
    failing = FakeBackend("a", fail=BackendError("a", "boom"))
    client = Client([failing])
    _patch_make_client(monkeypatch, client)

    code = cli.main(["ask", "x", "--choice", "team=billing,eng"])

    assert code == 1
    err = capsys.readouterr().err
    assert err.strip() != ""


def test_backends_lists_registry_names(capsys):
    code = cli.main(["backends"])

    assert code == 0
    out = capsys.readouterr().out
    assert "typesafe" in out
    assert "llm" in out
    assert "NAME" in out and "INSTALLED" in out and "CONFIGURED" in out


def test_backends_crossencoder_is_configured_n_a(capsys):
    cli.main(["backends"])

    out = capsys.readouterr().out
    crossencoder_line = next(line for line in out.splitlines() if line.startswith("crossencoder"))
    assert "n/a" in crossencoder_line.split()


def test_backends_output_is_padded_and_aligned_with_no_trailing_whitespace(capsys):
    code = cli.main(["backends"])

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines
    for line in lines:
        assert line == line.rstrip(), f"line has trailing whitespace: {line!r}"

    rows = cli._backend_status(os.environ)
    name_width = max(len("NAME"), *(len(name) for name, _, _ in rows))
    header, *data_rows = lines
    assert header.startswith("NAME".ljust(name_width) + "  ")
    for line, (name, _installed, _configured) in zip(data_rows, rows, strict=True):
        assert line.startswith(name.ljust(name_width) + "  ")


def test_ask_table_rows_have_no_trailing_whitespace(monkeypatch, capsys):
    client = Client([FakeBackend("fake")])
    _patch_make_client(monkeypatch, client)

    code = cli.main(
        ["ask", "charged twice", "--choice", "team=billing,eng", "--noul", "refund=Refund?"]
    )

    assert code == 0
    for line in capsys.readouterr().out.splitlines():
        assert line == line.rstrip(), f"line has trailing whitespace: {line!r}"


def test_backend_status_uses_given_env_mapping():
    rows = cli._backend_status({"TYPESAFE_API_KEY": "secret"})
    row_by_name = {name: (installed, configured) for name, installed, configured in rows}

    assert row_by_name["typesafe"][1] == "yes"
    assert row_by_name["openrouter"][1] == "no"
    assert row_by_name["crossencoder"][1] == "n/a"


def test_serve_calls_run_server_seam(monkeypatch):
    captured = {}

    def _fake_make_client(env, policy):
        captured["env"] = env
        captured["policy"] = policy
        return Client([FakeBackend()])

    def _fake_run_server(app, host, port):
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(cli, "make_client", _fake_make_client)
    monkeypatch.setattr(cli, "run_server", _fake_run_server)

    code = cli.main(
        [
            "serve",
            "--backends",
            "a,b",
            "--host",
            "127.0.0.1",
            "--port",
            "8811",
            "--api-key",
            "TOKEN",
            "--min-confidence",
            "0.0",
        ]
    )

    assert code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8811
    assert captured["env"]["DECIDE_BACKENDS"] == "a,b"
    assert captured["policy"].min_confidence == 0.0
    assert captured["app"] is not None


def test_serve_api_key_flag_wins_over_env(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "make_client", lambda env, policy: Client([FakeBackend()]))
    monkeypatch.setattr(cli, "run_server", lambda app, host, port: None)
    monkeypatch.setattr(
        "decide.server.create_app",
        lambda client, api_key=None: captured.update(api_key=api_key) or object(),
    )
    monkeypatch.setenv("DECIDE_API_KEY", "ENV_TOKEN")

    code = cli.main(["serve", "--api-key", "FLAG_TOKEN"])

    assert code == 0
    assert captured["api_key"] == "FLAG_TOKEN"


def test_serve_falls_back_to_env_api_key_when_flag_absent(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli, "make_client", lambda env, policy: Client([FakeBackend()]))
    monkeypatch.setattr(cli, "run_server", lambda app, host, port: None)
    monkeypatch.setattr(
        "decide.server.create_app",
        lambda client, api_key=None: captured.update(api_key=api_key) or object(),
    )
    monkeypatch.setenv("DECIDE_API_KEY", "ENV_TOKEN")

    code = cli.main(["serve"])

    assert code == 0
    assert captured["api_key"] == "ENV_TOKEN"


def test_no_command_is_usage_error(capsys):
    code = cli.main([])

    assert code == 2
    assert capsys.readouterr().err.strip() != ""
