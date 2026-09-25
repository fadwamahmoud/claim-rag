"""Streamlit chat UI for the OmniCare claims assistant."""

import json
import os
import secrets

import streamlit as st

from api_client import BackendClient, BackendError

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
EXAMPLES = [
    "Is water damage from a burst pipe covered?",
    "Do I need a receipt for a $3,000 necklace?",
    "What's the status of claim CLM-9014?",
    "I want to file a claim for a stolen laptop.",
]
TOOL_LABELS = {
    "search_policy": "🔎 Searched policy",
    "get_claim_status": "📋 Looked up claim",
    "submit_claim": "📝 Submitted claim",
}

st.set_page_config(page_title="OmniCare Assistant", page_icon="🛡️")


@st.cache_resource
def get_client(base_url: str) -> BackendClient:
    return BackendClient(base_url)


def new_user_id() -> str:
    return f"usr_{secrets.token_hex(3)}"


def init_state() -> None:
    st.session_state.setdefault("user_id", new_user_id())
    st.session_state.setdefault("messages", [])  # [{"role", "content", "sources", "tool_calls"}]
    st.session_state.setdefault("pending", None)  # prompt queued by an example button


def render_sources(sources: list[str]) -> None:
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            for src in sources:
                st.markdown(f"- {src}")


def render_tool_calls(tool_calls: list[dict]) -> None:
    for call in tool_calls:
        label = TOOL_LABELS.get(call["name"], f"🔧 {call['name']}")
        output = call.get("output")
        failed = call.get("status") != "success" or (isinstance(output, dict) and output.get("ok") is False)
        with st.expander(f"{label}{' — failed' if failed else ''}"):
            st.caption("Arguments")
            st.code(json.dumps(call.get("args", {}), indent=2), language="json")
            st.caption("Result")
            if isinstance(output, (dict, list)):
                st.code(json.dumps(output, indent=2), language="json")
            else:
                st.text(str(output))


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_tool_calls(msg.get("tool_calls", []))
            render_sources(msg.get("sources", []))


def sidebar(client: BackendClient) -> None:
    with st.sidebar:
        st.header("🛡️ OmniCare")
        if client.health():
            st.success("Backend online", icon="✅")
        else:
            st.error(f"Backend unreachable at {BACKEND_URL}", icon="⚠️")

        st.text_input(
            "User ID",
            key="user_id",
            help="The backend keeps a separate conversation memory per user ID.",
        )
        st.button("New conversation", on_click=reset_conversation, use_container_width=True)

        st.subheader("Try asking")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True):
                st.session_state.pending = example


def reset_conversation() -> None:
    # Runs as a callback, before widgets are drawn, so it may change the `user_id` widget's value.
    # Server-side memory is keyed by user ID, so a fresh ID starts a clean thread.
    st.session_state.user_id = new_user_id()
    st.session_state.messages = []


def ask(client: BackendClient, prompt: str) -> None:
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking…"):
                reply = client.chat(st.session_state.user_id, prompt)
        except BackendError as exc:
            st.error(str(exc))
            return  # errors are not stored in history
    msg = {"role": "assistant", "content": reply.response, "sources": reply.sources, "tool_calls": reply.tool_calls}
    st.session_state.messages.append(msg)
    st.rerun()  # redraw from history so the reply renders in its final form


def main() -> None:
    init_state()
    client = get_client(BACKEND_URL)
    sidebar(client)

    st.title("OmniCare Claims Assistant")
    st.caption("Ask about your coverage, check a claim's status, or file a new claim.")

    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Type your question…")
    if st.session_state.pending:
        prompt, st.session_state.pending = st.session_state.pending, None
    if prompt and prompt.strip():
        ask(client, prompt.strip())


main()
