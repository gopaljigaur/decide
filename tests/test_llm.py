import json

import httpx
import pytest

from decide.backends.llm import LLMBackend, build_prompt
from decide.errors import BadResponseError
from decide.types import Choice, Noul, Request, Score

REQ = Request(
    {"ticket": "charged twice"},
    {
        "team": Choice("Which team?", {"billing": "money", "eng": None}),
        "sev": Score("Severity?", ["minor", "blocked"]),
        "refund": Noul("Refund?"),
    },
)


def _chat(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def test_prompt_mentions_every_candidate_and_level():
    p = build_prompt(REQ)
    for s in [
        "charged twice",
        "team",
        "billing",
        "money",
        "eng",
        "sev",
        "minor",
        "blocked",
        "refund",
        "Refund?",
    ]:
        assert s in p


def test_parses_and_normalises_model_json():
    seen = {}

    def handler(r):
        seen["body"] = json.loads(r.content)
        return httpx.Response(
            200,
            json=_chat(
                json.dumps(
                    {
                        "team": {
                            "probabilities": {"billing": 0.8, "eng": 0.4}
                        },  # sums to 1.2 -> renormalise
                        "sev": {"probabilities": [0.25, 0.75]},
                        "refund": {"noul": 0.9},
                    }
                )
            ),
        )

    b = LLMBackend(api_key="k", transport=httpx.MockTransport(handler))
    r = b.decide(REQ)
    assert seen["body"]["model"] == "gpt-4o-mini" and seen["body"]["response_format"] == {
        "type": "json_object"
    }
    assert (
        abs(sum(r.choices["team"].probabilities.values()) - 1) < 1e-9
        and r.choices["team"].choice == "billing"
    )
    assert r.scores["sev"].score == 0.75 and r.nouls["refund"].noul == 0.9


def test_fenced_json_and_missing_candidates_handled():
    content = (
        "```json\n"
        + json.dumps(
            {
                "team": {"probabilities": {"billing": 1.0}},
                "sev": {"probabilities": [1, 0]},
                "refund": {"noul": 0.1},
            }
        )
        + "\n```"
    )
    r = LLMBackend(
        api_key="k",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_chat(content))),
    ).decide(REQ)
    assert r.choices["team"].probabilities == {"billing": 1.0, "eng": 0.0}


def test_garbage_raises_bad_response():
    with pytest.raises(BadResponseError):
        LLMBackend(
            api_key="k",
            transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_chat("not json"))),
        ).decide(REQ)


def test_no_api_key_sends_no_authorization_header():
    seen = {}

    def handler(r):
        seen["has_auth"] = "authorization" in {k.lower() for k in r.headers.keys()}
        return httpx.Response(
            200,
            json=_chat(
                json.dumps(
                    {
                        "team": {"probabilities": {"billing": 1.0, "eng": 0.0}},
                        "sev": {"probabilities": [1, 0]},
                        "refund": {"noul": 0.1},
                    }
                )
            ),
        )

    LLMBackend(transport=httpx.MockTransport(handler)).decide(REQ)
    assert seen["has_auth"] is False


def test_retries_once_without_response_format_on_400():
    calls = []

    def handler(r):
        body = json.loads(r.content)
        calls.append(body)
        if "response_format" in body:
            return httpx.Response(
                400, json={"error": {"message": "Unrecognized request argument: response_format"}}
            )
        return httpx.Response(
            200,
            json=_chat(
                json.dumps(
                    {
                        "team": {"probabilities": {"billing": 1.0, "eng": 0.0}},
                        "sev": {"probabilities": [1, 0]},
                        "refund": {"noul": 0.1},
                    }
                )
            ),
        )

    r = LLMBackend(api_key="k", transport=httpx.MockTransport(handler)).decide(REQ)
    assert len(calls) == 2
    assert "response_format" in calls[0] and "response_format" not in calls[1]
    assert r.choices["team"].choice == "billing"


async def test_async_path():
    def handler(r):
        return httpx.Response(
            200,
            json=_chat(
                json.dumps(
                    {
                        "team": {"probabilities": {"billing": 1.0, "eng": 0.0}},
                        "sev": {"probabilities": [1, 0]},
                        "refund": {"noul": 0.1},
                    }
                )
            ),
        )

    b = LLMBackend(api_key="k", transport=httpx.MockTransport(handler))
    r = await b.adecide(REQ)
    assert r.nouls["refund"].noul == 0.1


def test_capabilities_reports_not_local():
    assert LLMBackend(api_key="k").capabilities().local is False
