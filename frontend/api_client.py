"""HTTP client for the OmniCare backend. Kept free of Streamlit so it can be unit-tested."""

from dataclasses import dataclass, field
from typing import Any

import httpx


class BackendError(Exception):
    """A failure the UI should show to the user as-is."""


@dataclass
class ChatReply:
    response: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


class BackendClient:
    def __init__(self, base_url: str, timeout: float = 60.0, transport: httpx.BaseTransport | None = None):
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport)

    def health(self) -> bool:
        try:
            resp = self._http.get("/api/v1/health", timeout=3.0)
            return resp.status_code == 200 and resp.json().get("status") == "healthy"
        except (httpx.HTTPError, ValueError):
            return False

    def chat(self, user_id: str, message: str) -> ChatReply:
        try:
            resp = self._http.post("/api/v1/chat", json={"user_id": user_id, "message": message})
        except httpx.TimeoutException:
            raise BackendError("The assistant took too long to respond. Please try again.") from None
        except httpx.HTTPError:
            raise BackendError("Can't reach the backend. Is it running?") from None

        if resp.status_code == 422:
            raise BackendError(f"Invalid request: {_detail(resp)}")
        if resp.status_code >= 400:
            raise BackendError(_detail(resp) or f"Backend error (HTTP {resp.status_code}).")

        body = resp.json()
        return ChatReply(
            response=body.get("response", ""),
            sources=body.get("sources", []),
            tool_calls=body.get("tool_calls", []),
        )


def _detail(resp: httpx.Response) -> str:
    """Turn FastAPI's `detail` (a string or a list of validation errors) into one line."""
    try:
        detail = resp.json().get("detail")
    except ValueError:
        return resp.text
    if isinstance(detail, list):
        return "; ".join(f"{'.'.join(map(str, d.get('loc', [])[1:]))}: {d.get('msg')}" for d in detail)
    return str(detail or "")
