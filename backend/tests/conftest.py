import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from app.agent.graph import ClaimsAgent
from app.agent.tools import build_tools
from app.config import DATA_DIR
from app.main import create_app
from app.rag.embeddings import HashingEmbeddings
from app.rag.store import PolicyStore
from app.tools.claims import ClaimsRepository


class ScriptedChatModel(BaseChatModel):
    """Fake LLM that replays pre-written AIMessages (incl. tool calls), in order."""

    script: list[AIMessage]
    _cursor: int = PrivateAttr(default=0)
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        reply = self.script[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def claims_path(tmp_path: Path) -> Path:
    path = tmp_path / "mock_claims.json"
    shutil.copy(DATA_DIR / "mock_claims.json", path)
    return path


@pytest.fixture
def claims_repo(claims_path: Path) -> ClaimsRepository:
    return ClaimsRepository(claims_path)


@pytest.fixture
def policy_store() -> PolicyStore:
    store = PolicyStore(HashingEmbeddings())
    store.ingest(DATA_DIR)
    return store


@pytest.fixture
def make_agent(policy_store, claims_repo):
    def _make(*script: AIMessage) -> tuple[ClaimsAgent, ScriptedChatModel]:
        llm = ScriptedChatModel(script=list(script), seen=[])
        return ClaimsAgent(llm, build_tools(policy_store, claims_repo)), llm

    return _make


@pytest.fixture
def make_client(make_agent):
    def _make(*script: AIMessage) -> TestClient:
        agent, _ = make_agent(*script)
        return TestClient(create_app(agent=agent))

    return _make


def read_claims(path: Path) -> list[dict]:
    return json.loads(path.read_text())
