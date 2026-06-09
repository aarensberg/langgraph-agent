"""Streamlit UI for the Albert School student assistant.

A chat interface anyone can use without reading the code. It demonstrates the
"Going further" extensions visibly:

- **Persistence**: the conversation lives in the SQLite checkpointer, keyed by a
  thread id kept in the URL, so a page reload restores the same conversation —
  including a pending approval request.
- **Streaming**: each turn streams the agent's steps (which tool is being
  called, with what arguments; when a tool returns; when it pauses for consent)
  into a live trace panel.
- **Human-in-the-loop**: before the agent reads the private Gmail inbox, the
  graph interrupts and this UI asks the student to Allow or Deny; the decision is
  resumed back into the graph. The gate is toggleable in the sidebar.
- **Safety / observability**: a recursion limit is passed on every run, and the
  same structured events shown here are written to ``albert_agent.log``.

Run with:  ``streamlit run app.py``
"""

from __future__ import annotations

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from albert_agent import config, google_client, rag
from albert_agent.graph import get_agent

st.set_page_config(
    page_title="Albert Student Assistant", page_icon="🎓", layout="centered"
)


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

# The HITL gate is on by default; the sidebar toggle flips it per run.
if "gate" not in st.session_state:
    st.session_state.gate = config.REQUIRE_EMAIL_APPROVAL


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


def pending_interrupt_payload():
    """Return the payload of a pending approval interrupt, if the graph is paused.

    Lets a page reload during an approval request restore the prompt instead of
    losing it — the interrupt lives in the checkpointer, not the browser session.
    """
    try:
        state = app.get_state(run_config)
    except Exception:  # noqa: BLE001
        return None
    for task in getattr(state, "tasks", []) or []:
        for itr in getattr(task, "interrupts", []) or []:
            return itr.value
    return None


if "history" not in st.session_state:
    st.session_state.history = history_from_checkpointer()
    # If we reloaded mid-approval, restore the pending request.
    payload = pending_interrupt_payload()
    if payload is not None:
        st.session_state.pending = {"payload": payload, "trace": []}


# --------------------------------------------------------------------------- #
# Streaming: run one (possibly partial) turn and narrate each step
# --------------------------------------------------------------------------- #
def _fmt_args(args: dict) -> str:
    shown = {k: v for k, v in (args or {}).items() if v not in (None, "")}
    return f"({', '.join(f'{k}={v!r}' for k, v in shown.items())})" if shown else ""


def stream_turn(input_obj) -> tuple[str | None, list[str], dict | None]:
    """Stream the graph until it finishes OR pauses for approval.

    ``input_obj`` is the initial ``{"messages": [...]}`` for a new turn, or a
    ``Command(resume=...)`` to continue after an approval. Returns
    ``(answer, trace, interrupt_payload)``: if the graph paused, ``answer`` is
    ``None`` and ``interrupt_payload`` carries the pending email actions.
    """
    trace: list[str] = []
    interrupt_payload: dict | None = None
    with st.status("Thinking…", expanded=True) as status:
        for chunk in app.stream(input_obj, config=run_config, stream_mode="updates"):
            if "__interrupt__" in chunk:
                interrupt_payload = chunk["__interrupt__"][0].value
                line = "⏸️ Pausing for your approval to read your email"
                trace.append(line)
                st.write(line)
                continue
            for node, update in chunk.items():
                messages = (
                    update.get("messages", []) if isinstance(update, dict) else []
                )
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
                elif node == "approval":
                    decided = update.get("approved") if isinstance(update, dict) else None
                    line = ("🔓 Email access approved" if decided == "approve"
                            else "🚫 Email access denied")
                    trace.append(line)
                    st.write(line)
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

    if interrupt_payload is not None:
        return None, trace, interrupt_payload
    answer = app.get_state(run_config).values["messages"][-1].content
    return answer, trace, None


def render_message(role: str, content: str, trace: list[str]) -> None:
    with st.chat_message(role):
        if role == "assistant" and trace:
            with st.expander("🧭 Agent trace", expanded=False):
                for line in trace:
                    st.markdown(line)
        st.markdown(content)


def take_turn(input_obj, prior_trace: list[str] | None = None) -> None:
    """Run a turn, then either commit the answer or stash a pending approval."""
    try:
        answer, trace, payload = stream_turn(input_obj)
    except Exception as exc:  # noqa: BLE001 - never let one bad turn break the app
        st.session_state.history.append(
            ("assistant", f"⚠️ Something went wrong while answering: {exc}", ["error"])
        )
        return
    trace = (prior_trace or []) + trace
    if payload is not None:  # graph paused for approval
        st.session_state.pending = {"payload": payload, "trace": trace}
    else:
        st.session_state.history.append(("assistant", answer, trace))


# --------------------------------------------------------------------------- #
# Header + existing history
# --------------------------------------------------------------------------- #
st.title("🎓 Albert Student Assistant")
st.caption(
    "Ask about your program, a course, your attendance, your grades — or your "
    "school mail and calendar. LangGraph · Groq · live intranet API + RAG + "
    "read-only Gmail/Calendar."
)

for entry in st.session_state.history:
    render_message(entry[0], entry[1], entry[2] if len(entry) > 2 else [])


# --------------------------------------------------------------------------- #
# Pending approval (human-in-the-loop): render the Allow / Deny card
# --------------------------------------------------------------------------- #
if st.session_state.get("pending"):
    payload = st.session_state.pending["payload"]
    with st.chat_message("assistant"):
        st.warning(
            "The assistant needs your permission to read your **email** to answer "
            "this. It will only **read** the messages below — nothing is sent or "
            "changed."
        )
        for item in payload.get("pending", []):
            args = {k: v for k, v in (item.get("args") or {}).items() if v not in (None, "")}
            label = "Search the inbox" if item["tool"] == "search_emails" else "Open an email"
            st.markdown(f"- **{label}** {_fmt_args(args)}")
        col_allow, col_deny = st.columns(2)
        allow = col_allow.button("✅ Allow", use_container_width=True, type="primary")
        deny = col_deny.button("🚫 Deny", use_container_width=True)
    if allow or deny:
        prior = st.session_state.pending["trace"]
        st.session_state.pending = None
        take_turn(Command(resume="approve" if allow else "deny"), prior_trace=prior)
        st.rerun()


# --------------------------------------------------------------------------- #
# Chat input (disabled while an approval is pending)
# --------------------------------------------------------------------------- #
if st.session_state.get("pending"):
    st.chat_input("Respond to the approval request above to continue…", disabled=True)
else:
    prompt = st.chat_input(
        "e.g. What's my attendance rate? Any email from my professor this week?"
    )
    if prompt:
        st.session_state.history.append(("user", prompt, []))
        render_message("user", prompt, [])
        take_turn({"messages": [HumanMessage(content=prompt)],
                   "require_approval": st.session_state.gate})
        st.rerun()


# --------------------------------------------------------------------------- #
# Sidebar: session, privacy gate, Google connection, examples, status
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Conversation")
    st.caption("Persisted by thread id (in the URL) — reload the page to test it.")
    st.code(thread_id, language="text")
    if st.button("🆕 New conversation", use_container_width=True):
        st.query_params["thread_id"] = str(uuid.uuid4())
        st.session_state.history = []
        st.session_state.pending = None
        st.rerun()

    st.markdown("### Privacy")
    st.session_state.gate = st.checkbox(
        "Ask before reading my emails",
        value=st.session_state.gate,
        help="When on, the agent pauses for your approval before any Gmail tool "
             "runs (human-in-the-loop). Calendar and school data are never gated.",
    )

    st.markdown("### Google account (read-only)")
    connected = google_client.is_connected()
    st.caption("🟢 Connected" if connected else "⚪ Not connected")
    if not connected:
        if st.button("🔗 Connect Google", use_container_width=True):
            with st.spinner("Opening your browser for consent…"):
                try:
                    google_client.authenticate()
                    google_client._gmail.cache_clear()
                    google_client._calendar.cache_clear()
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not connect: {exc}")
        st.caption("Or run once: `python -m albert_agent.google_auth`")
    else:
        if st.button("🔌 Disconnect", use_container_width=True):
            config.GOOGLE_TOKEN_FILE.unlink(missing_ok=True)
            google_client._gmail.cache_clear()
            google_client._calendar.cache_clear()
            st.rerun()

    st.markdown("### Try asking")
    for example in [
        "What classes am I taking this semester?",
        "How is the generative AI course graded?",
        "What is my attendance rate, and where am I least consistent?",
        "What is my average score per teaching unit?",
        "Any email from my professor about the exam?",
        "What's on my calendar this week?",
    ]:
        st.markdown(f"- {example}")

    st.markdown("### Status")
    st.caption(f"Model: `{config.MODEL_NAME}`")
    st.caption(f"Fallback chain ({len(config.FALLBACK_MODELS)}): "
               f"{', '.join(config.FALLBACK_MODELS)}")
    st.caption(f"Documents: {warm_documents()}")
    st.caption(f"Max tool loops/turn: {config.MAX_TOOL_ITERATIONS}")
