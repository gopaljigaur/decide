import json

import httpx

from decide.backends.openrouter import OpenRouterBackend
from decide.types import Noul, Request


def test_openrouter_url_key_and_default_model():
    seen = {}

    def handler(r):
        seen["url"] = str(r.url)
        seen["model"] = json.loads(r.content)["model"]
        seen["auth"] = r.headers["authorization"]
        return httpx.Response(200, json={"answers": {"q": {"type": "noul", "noul": 0.3}}})

    b = OpenRouterBackend(api_key="or-key", transport=httpx.MockTransport(handler))
    r = b.decide(Request("s", {"q": Noul("x")}))
    assert seen["url"] == "https://openrouter.ai/api/alpha/decisions"
    assert seen["model"] == "typesafe/jev-latest" and seen["auth"] == "Bearer or-key"
    assert r.meta.backend == "openrouter" and r.nouls["q"].noul == 0.3
