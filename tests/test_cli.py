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


def test_ask_choice_flag_without_equals_is_unnamed_single_candidate(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    code = cli.main(["ask", "x", "--choice", "team-no-equals"])

    assert code == 0
    assert set(captured["questions"]) == {"choice"}
    assert list(captured["questions"]["choice"].criteria) == ["team-no-equals"]


def test_ask_unnamed_choice_is_named_choice(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--choice", "good,bad"])

    assert set(captured["questions"]) == {"choice"}
    assert set(captured["questions"]["choice"].criteria) == {"good", "bad"}


def test_ask_unnamed_score_is_named_score(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--score", "lo,mid,hi"])

    assert set(captured["questions"]) == {"score"}
    assert list(captured["questions"]["score"].criteria) == ["lo", "mid", "hi"]


def test_ask_unnamed_noul_is_named_noul(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--noul", "Is the writer doing well?"])

    assert set(captured["questions"]) == {"noul"}
    assert captured["questions"]["noul"].instructions == "Is the writer doing well?"


def test_ask_two_unnamed_nouls_get_noul_and_noul2(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(
        [
            "ask",
            "x",
            "--noul",
            "Is the writer doing well?",
            "--noul",
            "Did they ask for a refund?",
        ]
    )

    assert set(captured["questions"]) == {"noul", "noul2"}
    assert captured["questions"]["noul"].instructions == "Is the writer doing well?"
    assert captured["questions"]["noul2"].instructions == "Did they ask for a refund?"


def test_ask_mixed_named_and_unnamed_questions(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--choice", "team=billing,eng", "--noul", "Is the writer doing well?"])

    assert set(captured["questions"]) == {"team", "noul"}
    assert captured["questions"]["noul"].instructions == "Is the writer doing well?"


def test_ask_noul_text_with_equals_and_spaces_is_unnamed(monkeypatch):
    captured = {}

    def _fabricate(request):
        captured["questions"] = dict(request.questions)
        return {name: FakeBackend()._fabricate(q) for name, q in request.questions.items()}

    client = Client([FakeBackend("fake", answers=_fabricate)])
    _patch_make_client(monkeypatch, client)

    cli.main(["ask", "x", "--noul", "Is the writer = doing well?"])

    assert set(captured["questions"]) == {"noul"}
    assert captured["questions"]["noul"].instructions == "Is the writer = doing well?"


def test_ask_duplicate_explicit_names_is_usage_error(capsys):
    code = cli.main(["ask", "x", "--choice", "team=billing,eng", "--score", "team=lo,mid,hi"])

    assert code == 2
    err = capsys.readouterr().err
    assert "team" in err


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
    assert "NAME" in out and "INSTALLED" in out and "CONFIGURED" in out and "INSTALL" in out


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
    name_width = max(len("NAME"), *(len(name) for name, _, _, _ in rows))
    header, *data_rows = lines
    assert header.startswith("NAME".ljust(name_width) + "  ")
    for line, (name, _installed, _configured, _hint) in zip(data_rows, rows, strict=True):
        assert line.startswith(name.ljust(name_width) + "  ")


def test_install_hint_for_extra_backed_backend():
    assert cli._install_hint("laya") == 'pip install "pydecide[laya]"'
    assert cli._install_hint("laya_mlx") == 'pip install "pydecide[mlx]"'
    assert cli._install_hint("crossencoder") == 'pip install "pydecide[st]"'


def test_install_hint_for_http_backend_is_empty():
    assert cli._install_hint("typesafe") == ""
    assert cli._install_hint("openrouter") == ""
    assert cli._install_hint("llm") == ""


def test_backends_appends_install_hint_for_not_installed_backends(monkeypatch, capsys):
    monkeypatch.setattr(cli, "available", lambda: dict.fromkeys(cli.REGISTRY, False))

    code = cli.main(["backends"])

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    rows = {line.split()[0]: line for line in lines[1:]}
    assert 'pip install "pydecide[laya]"' in rows["laya"]
    assert 'pip install "pydecide[mlx]"' in rows["laya_mlx"]
    assert 'pip install "pydecide[st]"' in rows["crossencoder"]
    assert "pip install" not in rows["typesafe"]
    assert "pip install" not in rows["openrouter"]
    assert "pip install" not in rows["llm"]


def test_backends_omits_install_hint_for_installed_backends(monkeypatch, capsys):
    monkeypatch.setattr(cli, "available", lambda: dict.fromkeys(cli.REGISTRY, True))

    code = cli.main(["backends"])

    assert code == 0
    lines = capsys.readouterr().out.splitlines()
    for line in lines[1:]:
        assert "pip install" not in line


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
    row_by_name = {
        name: (installed, configured, hint) for name, installed, configured, hint in rows
    }

    assert row_by_name["typesafe"][1] == "yes"
    assert row_by_name["openrouter"][1] == "no"
    assert row_by_name["crossencoder"][1] == "n/a"


def test_backend_status_local_backend_default_when_installed_and_no_override(monkeypatch):
    monkeypatch.setattr(cli, "available", lambda: dict.fromkeys(cli.REGISTRY, True))

    rows = cli._backend_status({})

    row_by_name = {name: configured for name, _installed, configured, _hint in rows}
    assert row_by_name["laya"] == "default"
    assert row_by_name["laya_mlx"] == "default"


def test_backend_status_local_backend_yes_when_decide_local_model_set(monkeypatch):
    monkeypatch.setattr(cli, "available", lambda: dict.fromkeys(cli.REGISTRY, True))

    rows = cli._backend_status({"DECIDE_LOCAL_MODEL": "m1"})

    row_by_name = {name: configured for name, _installed, configured, _hint in rows}
    assert row_by_name["laya"] == "yes"
    assert row_by_name["laya_mlx"] == "yes"


def test_backend_status_local_backend_no_when_not_installed(monkeypatch):
    monkeypatch.setattr(cli, "available", lambda: dict.fromkeys(cli.REGISTRY, False))

    rows = cli._backend_status({"DECIDE_LOCAL_MODEL": "m1"})

    row_by_name = {name: configured for name, _installed, configured, _hint in rows}
    assert row_by_name["laya"] == "no"
    assert row_by_name["laya_mlx"] == "no"


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


# --- HF_HUB_DISABLE_PROGRESS_BARS ------------------------------------------------------


def test_ask_sets_hf_hub_disable_progress_bars_when_unset(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    _patch_make_client(monkeypatch, Client([FakeBackend("fake")]))

    cli.main(["ask", "x", "--choice", "team=billing,eng"])

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


def test_ask_leaves_existing_hf_hub_disable_progress_bars_untouched(monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "0")
    _patch_make_client(monkeypatch, Client([FakeBackend("fake")]))

    cli.main(["ask", "x", "--choice", "team=billing,eng"])

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "0"


def test_ask_verbose_does_not_set_hf_hub_disable_progress_bars(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    _patch_make_client(monkeypatch, Client([FakeBackend("fake")]))

    cli.main(["ask", "x", "--choice", "team=billing,eng", "--verbose"])

    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ


def test_serve_sets_hf_hub_disable_progress_bars_when_unset(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.setattr(cli, "make_client", lambda env, policy: Client([FakeBackend()]))
    monkeypatch.setattr(cli, "run_server", lambda app, host, port: None)

    cli.main(["serve"])

    assert os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"


def test_serve_verbose_does_not_set_hf_hub_disable_progress_bars(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_PROGRESS_BARS", raising=False)
    monkeypatch.setattr(cli, "make_client", lambda env, policy: Client([FakeBackend()]))
    monkeypatch.setattr(cli, "run_server", lambda app, host, port: None)

    cli.main(["serve", "--verbose"])

    assert "HF_HUB_DISABLE_PROGRESS_BARS" not in os.environ


def test_no_command_is_usage_error(capsys):
    code = cli.main([])

    assert code == 2
    assert capsys.readouterr().err.strip() != ""
