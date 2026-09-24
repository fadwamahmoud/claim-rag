"""LangGraph workflow: guardrail -> agent <-> tools.

    START ─▶ guard ──blocked──▶ END (refusal)
               │
               ▼
             agent ──no tool calls──▶ END
               ▲  │
               │  ▼ tool calls
              tools
"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.safety.guardrails import REFUSAL_MESSAGE, check_user_input, normalize

MAX_HISTORY_MESSAGES = 20
RECURSION_LIMIT = 12


class AgentState(MessagesState):
    blocked: bool


@dataclass
class ChatResult:
    response: str
    sources: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    blocked: bool = False


def build_graph(llm: BaseChatModel, tools: list[BaseTool], checkpointer: BaseCheckpointSaver | None = None):
    llm_with_tools = llm.bind_tools(tools)

    def guard(state: AgentState) -> dict:
        last = state["messages"][-1]
        verdict = check_user_input(last.content if isinstance(last.content, str) else str(last.content))
        if verdict.allowed:
            return {"blocked": False}
        return {"blocked": True, "messages": [AIMessage(REFUSAL_MESSAGE, name="guardrail")]}

    def route_after_guard(state: AgentState) -> Literal["agent", "__end__"]:
        return END if state.get("blocked") else "agent"

    def agent(state: AgentState) -> dict:
        history = _trim_history(state["messages"])
        reply = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT), *history])
        return {"messages": [reply]}

    graph = StateGraph(AgentState)
    graph.add_node("guard", guard)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", route_after_guard)
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def _trim_history(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep the last N messages, never starting on an orphaned ToolMessage."""
    trimmed = messages[-MAX_HISTORY_MESSAGES:]
    while trimmed and not isinstance(trimmed[0], HumanMessage):
        trimmed = trimmed[1:]
    return trimmed or messages[-1:]


class ClaimsAgent:
    """Thin facade the API layer talks to. One conversation thread per user_id."""

    def __init__(self, llm: BaseChatModel, tools: list[BaseTool], checkpointer: BaseCheckpointSaver | None = None):
        self.graph = build_graph(llm, tools, checkpointer)

    async def chat(self, user_id: str, message: str) -> ChatResult:
        config = {"configurable": {"thread_id": user_id}, "recursion_limit": RECURSION_LIMIT}
        state = await self.graph.ainvoke({"messages": [HumanMessage(normalize(message))]}, config)
        return _to_result(state)


def _to_result(state: dict) -> ChatResult:
    messages: list[BaseMessage] = state["messages"]
    # Only report what happened during this turn (after the latest user message).
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    turn = messages[start + 1 :]

    outputs = {m.tool_call_id: m for m in turn if isinstance(m, ToolMessage)}
    sources: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for msg in turn:
        if not isinstance(msg, AIMessage):
            continue
        for call in msg.tool_calls:
            tool_msg = outputs.get(call["id"])
            tool_calls.append(
                {
                    "name": call["name"],
                    "args": call["args"],
                    "output": _parse_output(tool_msg.content) if tool_msg else None,
                    "status": getattr(tool_msg, "status", "success") if tool_msg else "missing",
                }
            )
            if call["name"] == "search_policy" and tool_msg and tool_msg.artifact:
                for hit in tool_msg.artifact:
                    if hit["citation"] not in sources:
                        sources.append(hit["citation"])

    final = turn[-1] if turn else None
    text = final.text if isinstance(final, AIMessage) else ""
    return ChatResult(
        response=text,
        sources=sources,
        tool_calls=tool_calls,
        blocked=bool(state.get("blocked")),
    )


def _parse_output(content: Any) -> Any:
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content
