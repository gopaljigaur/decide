"""OpenRouter HTTP backend, served via the same TypeSafe-shaped wire protocol."""

from __future__ import annotations

from decide.backends.typesafe import TypeSafeBackend


class OpenRouterBackend(TypeSafeBackend):
    name = "openrouter"
    DEFAULT_BASE_URL = "https://openrouter.ai/api"
    PATH = "/alpha/decisions"
    DEFAULT_MODEL = "typesafe/jev-latest"
