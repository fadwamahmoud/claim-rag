import json

import httpx
import pytest

from api_client import BackendClient, BackendError


def client_for(handler) -> BackendClient:
    return BackendClient("http://backend", transport=httpx.MockTransport(handler))


def test_chat_parses_reply_and_sends_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "response": "Covered up to $25,000.",
                "sources": ["sample_policy.md — Section 1: Home Water Damage Coverage"],
                "tool_calls": [{"name": "search_policy", "args": {"query": "pipe"}, "output": "...", "status": "success"}],
            },
        )

    reply = client_for(handler).chat("usr_123", "Is a burst pipe covered?")
    assert seen == {"path": "/api/v1/chat", "body": {"user_id": "usr_123", "message": "Is a burst pipe covered?"}}
    assert reply.response == "Covered up to $25,000."
    assert reply.sources[0].startswith("sample_policy.md")
    assert reply.tool_calls[0]["name"] == "search_policy"


def test_chat_formats_validation_errors():
    def handler(request):
        return httpx.Response(
            422, json={"detail": [{"loc": ["body", "user_id"], "msg": "String should match pattern"}]}
        )

    with pytest.raises(BackendError, match="Invalid request: user_id: String should match pattern"):
        client_for(handler).chat("bad id", "hi")


def test_chat_surfaces_backend_detail_on_502():
    def handler(request):
        return httpx.Response(502, json={"detail": "The assistant is temporarily unavailable."})

    with pytest.raises(BackendError, match="temporarily unavailable"):
        client_for(handler).chat("usr_1", "hi")


def test_chat_reports_unreachable_backend():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(BackendError, match="Can't reach the backend"):
        client_for(handler).chat("usr_1", "hi")


@pytest.mark.parametrize(
    "response, healthy",
    [
        (httpx.Response(200, json={"status": "healthy"}), True),
        (httpx.Response(500), False),
        (httpx.Response(200, text="not json"), False),
    ],
)
def test_health(response, healthy):
    assert client_for(lambda request: response).health() is healthy
