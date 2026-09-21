import pytest

from decide.backends import REGISTRY, available, load
from decide.errors import ConfigError


def test_load_unknown_backend_raises_config_error():
    with pytest.raises(ConfigError):
        load("nope")


def test_available_returns_all_registered_backends():
    result = available()
    assert set(result.keys()) == set(REGISTRY.keys())
    assert set(REGISTRY.keys()) == {
        "typesafe",
        "openrouter",
        "laya",
        "laya_mlx",
        "crossencoder",
        "llm",
    }
    assert all(isinstance(v, bool) for v in result.values())
