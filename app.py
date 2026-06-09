"""Streamlit UI for the Albert School student assistant.

A chat interface anyone can use without reading the code. It demonstrates the
"Going further" extensions visibly:

- **Persistence**: the conversation lives in the SQLite checkpointer, keyed by a
  thread id kept in the URL, so a page reload restores the same conversation.
- **Streaming**: each turn streams the agent's steps (which tool is being
  called, with what arguments; when a tool returns) into a live trace panel.
- **Safety / observability**: a recursion limit is passed on every run, and the
  same structured events shown here are written to ``albert_agent.log``.

Run with:  ``streamlit run app.py``
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from albert_agent import config, rag
from albert_agent.graph import get_agent

st.set_page_config(page_title="Albert Student Assistant", page_icon="🎓", layout="centered")


# --------------------------------------------------------------------------- #
# Cached resources (built once per process, not per rerun)
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_agent():
    """Compile the graph with the on-disk checkpointer exactly once."""
    return get_agent()


@st.cache_resource
def warm_documents() -> str:
    """Build/load the RAG index once; report its status for the sidebar."""
    try:
        return f"{rag.warmup()} chunks indexed"
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({exc})"


app = load_agent()


# --------------------------------------------------------------------------- #
# Per-conversation thread id, persisted in the URL so reloads resume it
# --------------------------------------------------------------------------- #
if "thread_id" not in st.query_params:
    st.query_params["thread_id"] = str(uuid.uuid4())
thread_id = st.query_params["thread_id"]
run_config = {
    "configurable": {"thread_id": thread_id},
    "recursion_limit": config.GRAPH_RECURSION_LIMIT,
}


def history_from_checkpointer() -> list[tuple]:
    """Rebuild the visible chat (user + final answers) from saved state.

    Called after a reload, when Streamlit's session is fresh but the
    checkpointer still holds the conversation.
    """
    state = app.get_state(run_config)
    messages = state.values.get("messages", []) if state.values else []
    visible = []
    for m in messages:
        if isinstance(m, HumanMessage):
            visible.append(("user", m.content, []))
        elif isinstance(m, AIMessage) and m.content and not m.tool_calls:
            visible.append(("assistant", m.content, []))
    return visible


if "history" not in st.session_state:
    st.session_state.history = history_from_checkpointer()


# --------------------------------------------------------------------------- #
# Streaming: run one turn and narrate each step into a live trace
# --------------------------------------------------------------------------- #
def _fmt_args(args: dict) -> str:
    shown = {k: v for k, v in (args or {}).items() if v not in (None, "")}
    return f"({', '.join(f'{k}={v!r}' for k, v in shown.items())})" if shown else ""


def run_and_stream(user_text: str) -> tuple[str, list[str]]:
    """Stream the graph for one user message, returning (answer, trace_lines)."""
    trace: list[str] = []
    with st.status("Thinking…", expanded=True) as status:
        for chunk in app.stream(
            {"messages": [HumanMessage(content=user_text)]},
            config=run_config,
            stream_mode="updates",
        ):
            for node, update in chunk.items():
                messages = update.get("messages", []) if isinstance(update, dict) else []
                if node == "agent":
                    ai = messages[-1] if messages else None
                    calls = getattr(ai, "tool_calls", []) if ai else []
                    for call in calls:
                        line = f"🔧 Calling **{call['name']}**{_fmt_args(call['args'])}"
                        trace.append(line)
                        st.write(line)
                    if not calls:
                        trace.append("✍️ Composing the answer")
                        st.write("✍️ Composing the answer")
                elif node == "tools":
                    for m in messages:
                        ok = not str(m.content).startswith("⚠️")
                        line = f"{'✅' if ok else '⚠️'} **{m.name}** returned"
                        trace.append(line)
                        st.write(line)
                elif node == "fallback":
                    trace.append("🛑 Step limit reached — stopping safely")
                    st.write("🛑 Step limit reached — stopping safely")
        status.update(label="Done", state="complete", expanded=False)

    answer = app.get_state(run_config).values["messages"][-1].content
    return answer, trace


def render_message(role: str, content: str, trace: list[str]) -> None:
    with st.chat_message(role):
        if role == "assistant" and trace:
            with st.expander("🧭 Agent trace", expanded=False):
                for line in trace:
                    st.markdown(line)
        st.markdown(content)


# --------------------------------------------------------------------------- #
# Header + existing history
# --------------------------------------------------------------------------- #
st.title("🎓 Albert Student Assistant")
st.caption(
    "Ask about your program, a course, your attendance, or your grades. "
    "LangGraph · Groq · live intranet API + RAG over your documents."
)

for entry in st.session_state.history:
    render_message(entry[0], entry[1], entry[2] if len(entry) > 2 else [])


# --------------------------------------------------------------------------- #
# Chat input
# --------------------------------------------------------------------------- #
prompt = st.chat_input("e.g. What's my attendance rate, and my average per teaching unit?")
if prompt:
    st.session_state.history.append(("user", prompt, []))
    render_message("user", prompt, [])
    try:
        answer, trace = run_and_stream(prompt)
    except Exception as exc:  # noqa: BLE001 - never let one bad turn break the app
        answer, trace = (
            f"⚠️ Something went wrong while answering: {exc}", ["error"],
        )
    st.session_state.history.append(("assistant", answer, trace))
    render_message("assistant", answer, trace)


# --------------------------------------------------------------------------- #
# Sidebar: session, examples, status
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Conversation")
    st.caption("Persisted by thread id (in the URL) — reload the page to test it.")
    st.code(thread_id, language="text")
    if st.button("🆕 New conversation", use_container_width=True):
        st.query_params["thread_id"] = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.markdown("### Try asking")
    for example in [
        "Quels cours est-ce que je suis ce semestre ?",
        "Quelles sont les modalités d'évaluation du cours d'IA générative ?",
        "Quel est mon taux de présence, et où suis-je le moins assidu ?",
        "Quelle est ma moyenne par Teaching Unit ?",
        "Quelles étaient mes notes en année 1 d'après mon relevé ?",
    ]:
        st.markdown(f"- {example}")

    st.markdown("### Status")
    st.caption(f"Model: `{config.MODEL_NAME}`")
    st.caption(f"Fallback: `{config.FALLBACK_MODEL}`")
    st.caption(f"Documents: {warm_documents()}")
    st.caption(f"Max tool loops/turn: {config.MAX_TOOL_ITERATIONS}")
