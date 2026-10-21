import json

import pytest

import decide.cli as cli
import decide.client as client_module
from decide import AsyncClient, Choice, Client, Config, ConfigError, LowConfidence
from decide.backends.base import Capabilities
from tests.conftest import FakeBackend


@pytest.fixture
def loader(monkeypatch):
    made = {}

    def load(name, **kwargs):
        made[name] = kwargs
        backend = FakeBackend(name, confidence=0.6)
        if name in {"laya", "laya_mlx", "crossencoder"}:
            backend.capabilities = lambda: Capabilities(local=True)
        return backend

    monkeypatch.setattr(client_module, "load_backend", load)
    monkeypatch.setattr(client_module, "available", lambda: {"laya": True})
    for key in ("DECIDE_BACKENDS", "DECIDE_LOCAL_ONLY", "DECIDE_MIN_CONFIDENCE"):
        monkeypatch.delenv(key, raising=False)
    return made


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
def test_config_object_and_file_share_backend_options(tmp_path, monkeypatch, loader, client_type):
    monkeypatch.setenv("TYPESAFE_API_KEY", "credential")
    config = Config(
        ["laya", "typesafe"],
        options={"laya": {"model": "my-model"}, "typesafe": {"base_url": "https://example.test"}},
        min_confidence=0.85,
        on_low_confidence="raise",
    )
    path = tmp_path / "decide.json"
    config.save(path)
    for source in (config, path, str(path)):
        client = client_type(config=source)
        assert [b.name for b in client.backends] == ["laya", "typesafe"]
        assert loader == {
            "laya": {"model": "my-model"},
            "typesafe": {"api_key": "credential", "base_url": "https://example.test"},
        }
        assert client.policy.min_confidence == 0.85
        assert client.on_low_confidence == "raise"


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
def test_default_file_loaded_without_prompting(tmp_path, monkeypatch, loader, client_type):
    monkeypatch.chdir(tmp_path)
    Config(["crossencoder"], options={"crossencoder": {"model": "reranker"}}).save()
    assert client_type().backends[0].name == "crossencoder"
    assert client_type.from_env().backends[0].name == "crossencoder"
    assert loader == {"crossencoder": {"model": "reranker"}}
    # Supplying an isolated environment also isolates it from the working directory's config.
    assert client_type.from_env(env={}).backends[0].name == "laya"


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
def test_constructor_rejects_competing_sources(loader, client_type):
    with pytest.raises(ConfigError, match="not both"):
        client_type([FakeBackend()], config=Config(["laya"]))


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
def test_keys_and_urls_never_choose_hosted_backends(loader, client_type):
    client = client_type.from_env(
        env={
            "TYPESAFE_API_KEY": "a",
            "OPENROUTER_API_KEY": "b",
            "OPENAI_API_KEY": "c",
            "DECIDE_LLM_BASE_URL": "https://example.test",
        }
    )
    assert [b.name for b in client.backends] == ["laya"]
    assert set(loader) == {"laya"}


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
def test_local_only_applies_to_config_before_loading(monkeypatch, loader, client_type):
    monkeypatch.setenv("DECIDE_LOCAL_ONLY", "1")
    with pytest.raises(ConfigError, match="local only"):
        client_type(config=Config(["typesafe"]))
    assert not loader


def test_saved_abstention_applies_to_real_decision(loader):
    client = Client(config=Config(["laya"], min_confidence=0.9, on_low_confidence="raise"))
    with pytest.raises(LowConfidence):
        client.decide("state", {"team": Choice("Choose", ["a", "b"])})


@pytest.mark.parametrize(
    "data",
    [
        [],
        {"backends": "laya"},
        {"backends": ["unknown"]},
        {"backends": [7]},
        {"options": []},
        {"options": {"laya": {}}},
        {"min_confidence": "0.5"},
        {"min_confidence": True},
        {"min_confidence": -1},
        {"min_confidence": float("nan")},
        {"on_low_confidence": "maybe"},
        {"typo": True},
    ],
)
def test_invalid_files_raise_config_error(tmp_path, data):
    path = tmp_path / "decide.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError):
        Config.load(path)


def test_save_preserves_existing_file_and_reports_bad_json(tmp_path):
    path = tmp_path / "decide.json"
    path.write_text("old contents")
    with pytest.raises(ConfigError, match="cannot write"):
        Config(["laya"]).save(path)
    assert path.read_text() == "old contents"
    with pytest.raises(ConfigError, match="cannot read"):
        Config.load(path)


def interactive(monkeypatch, answers):
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    responses = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt: next(responses))
    monkeypatch.setattr(cli, "available", lambda: {"laya": True})
    monkeypatch.delenv("DECIDE_LOCAL_ONLY", raising=False)


def test_interactive_local_setup_used_by_cli(tmp_path, monkeypatch, capsys, loader):
    monkeypatch.chdir(tmp_path)
    interactive(monkeypatch, ["", "", "0.8", "y"])
    assert cli.main(["init"]) == 0
    config = Config.load()
    assert config.backends == ("laya",)
    assert config.min_confidence == 0.8
    assert "api_key" not in (tmp_path / "decide.json").read_text()
    client = cli.make_client(None, None)
    assert client.backends[0].name == "laya"
    assert client.policy.min_confidence == 0.8
    assert "Saved decide.json" in capsys.readouterr().out


def test_interactive_hosted_setup_requires_explicit_selection(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "do-not-write")
    interactive(monkeypatch, ["laya,typesafe", "", "", "", "0.85", "y"])
    assert cli.main(["init"]) == 0
    assert Config.load().backends == ("laya", "typesafe")
    assert "do-not-write" not in (tmp_path / "decide.json").read_text()
    assert "States may be sent to servers by: typesafe" in capsys.readouterr().out


def test_interactive_retries_invalid_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    interactive(monkeypatch, ["nope", "laya", "", "bad", "nan", "1.5", "0.7", "y"])
    assert cli.main(["init"]) == 0
    assert Config.load().min_confidence == 0.7


@pytest.mark.parametrize("answers", [["n"], ["y", "laya", "", "0", "n"]])
def test_interactive_cancellation_preserves_file(tmp_path, monkeypatch, answers):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "decide.json"
    path.write_text("original")
    interactive(monkeypatch, answers)
    assert cli.main(["init"]) == 0
    assert path.read_text() == "original"


def test_interactive_eof_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    interactive(monkeypatch, [])

    def end_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_input)
    assert cli.main(["init"]) == 0
    assert not (tmp_path / "decide.json").exists()


def test_init_rejects_noninteractive_input(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    assert cli.main(["init"]) == 1
    assert "interactive terminal" in capsys.readouterr().err
    assert not (tmp_path / "decide.json").exists()


@pytest.mark.parametrize("client_type", [Client, AsyncClient])
def test_explicit_config_takes_precedence_over_env_selection(monkeypatch, loader, client_type):
    monkeypatch.setenv("DECIDE_BACKENDS", "typesafe")
    monkeypatch.setenv("TYPESAFE_API_KEY", "key")
    assert client_type(config=Config(["laya"])).backends[0].name == "laya"
    assert set(loader) == {"laya"}


def test_from_env_explicit_selection_overrides_saved_chain(loader):
    client = Client.from_env(
        env={"DECIDE_BACKENDS": "typesafe", "TYPESAFE_API_KEY": "key"},
        config=Config(["laya"]),
    )
    assert client.backends[0].name == "typesafe"


def test_custom_cli_config_used_for_ask(tmp_path, monkeypatch, loader, capsys):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "custom.json"
    Config(["laya"], options={"laya": {"model": "selected"}}).save(path)
    assert cli.main(["ask", "state", "--choice", "a,b", "--config", str(path)]) == 0
    assert loader == {"laya": {"model": "selected"}}
    assert "backend=laya" in capsys.readouterr().out


def test_server_loads_saved_threshold(tmp_path, monkeypatch, loader):
    monkeypatch.chdir(tmp_path)
    Config(["laya"], min_confidence=0.9).save()
    captured = {}
    import decide.server as server

    def create_app(client, **kwargs):
        captured["client"] = client
        return object()

    monkeypatch.setattr(server, "create_app", create_app)
    monkeypatch.setattr(cli, "run_server", lambda *args: None)
    assert cli.main(["serve"]) == 0
    assert captured["client"].policy.min_confidence == 0.9


def test_no_local_install_with_keys_has_setup_instructions(monkeypatch, loader):
    monkeypatch.setattr(client_module, "available", lambda: {})
    with pytest.raises(ConfigError, match="decide init"):
        Client.from_env(env={"OPENAI_API_KEY": "key"})
    assert not loader


def test_interactive_chain_escalates_only_after_saved_selection(
    tmp_path, monkeypatch, capsys, loader
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "key")
    interactive(monkeypatch, ["laya,typesafe", "", "", "", "0.85", "y"])
    assert cli.main(["init"]) == 0
    client = Client(config="decide.json")
    client.backends[1].confidence = 0.95
    response = client.decide("state", {"team": Choice("Choose", ["a", "b"])})
    assert response.meta.backend == "typesafe"
    assert response.meta.route == ["laya:low_confidence:team=0.60<0.85", "typesafe:ok"]


async def test_async_client_applies_saved_abstention(tmp_path, loader):
    path = tmp_path / "async.json"
    Config(["laya"], min_confidence=0.9, on_low_confidence="raise").save(path)
    async with AsyncClient(config=path) as client:
        with pytest.raises(LowConfidence):
            await client.decide("state", {"team": Choice("Choose", ["a", "b"])})
