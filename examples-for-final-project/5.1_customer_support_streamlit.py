"""
Customer Support Agent - Streamlit interface

Integrates every LangGraph concept introduced in Unit 4 on top of the
basic agent from 5_customer_support_agent.py:

    1. State with add_messages (chat history reducer)
    2. Classifier node (does the WORK of categorisation)
    3. Router function with add_conditional_edges (decides the NEXT NODE)
    4. Specialist nodes per category (billing / technical / general)
    5. Persistent checkpointer (SqliteSaver) so each user's conversation
       survives across page reloads and process restarts
    6. interrupt_before on the "human" node (human-in-the-loop)
    7. graph.stream(stream_mode="updates") so the user can see the agent's
       decisions in real time
    8. Streamlit chat UI with a HITL form when the agent escalates

Run with:
    streamlit run customer_support_streamlit.py
"""

import os
import sqlite3 # to store the checkpointer
import uuid # to generte universally unique identifiers
from typing import TypedDict, Annotated, Literal

import streamlit as st
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver #pip install langgraph-checkpoint-sqlite
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI


# =============================================================================
# 1. LLM
# =============================================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


# =============================================================================
# 2. State
# =============================================================================
class CustomerSupportState(TypedDict):
    messages: Annotated[list, add_messages]   # chat history (LangGraph reducer)
    category: str                              # "billing" | "technical" | "general" | "human"


# =============================================================================
# 3. Nodes
# =============================================================================
def classify(state: CustomerSupportState) -> dict:
    """Decide which specialist should handle the message. The node does the WORK
    of categorising; it writes the result into the state and stops there."""
    last_user_msg = state["messages"][-1].content
    prompt = f"""
Classify the following user question into ONE of these categories:
billing, technical, general, human.

Use "human" ONLY when the question is sensitive, asks for a refund over a
threshold, or explicitly requests speaking to a person.

Question: {last_user_msg}

Respond with ONLY the category word, in lowercase.
"""
    response = llm.invoke(prompt)
    category = response.content.strip().lower()
    if category not in {"billing", "technical", "general", "human"}:
        category = "general"
    return {"category": category}


def billing_specialist(state: CustomerSupportState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a billing specialist. Answer briefly and clearly about "
            "invoices, payments, refunds, and subscriptions."
        )),
        *state["messages"],
    ])
    return {"messages": [response]}


def tech_specialist(state: CustomerSupportState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a technical specialist. Help with login issues, bugs, "
            "and product features. Be concrete."
        )),
        *state["messages"],
    ])
    return {"messages": [response]}


def general_specialist(state: CustomerSupportState) -> dict:
    response = llm.invoke([
        SystemMessage(content=(
            "You are a general customer-support assistant. Be friendly and "
            "helpful."
        )),
        *state["messages"],
    ])
    return {"messages": [response]}


def human_review(state: CustomerSupportState) -> dict:
    """Runs only after the human reviewer typed their answer. By then the
    answer is already the last AIMessage in state['messages'] (Streamlit
    inserted it before resuming). This node is just the exit point."""
    return {}


# =============================================================================
# 4. Router (control flow only, no work)
# =============================================================================
def route(state: CustomerSupportState) -> Literal["billing", "technical", "general", "human"]:
    """Reads the State and returns the next node name. Does no LLM call."""
    return state["category"]


# =============================================================================
# 5. Graph
# =============================================================================
def build_graph():
    graph = StateGraph(CustomerSupportState)

    graph.add_node("classify",  classify)
    graph.add_node("billing",   billing_specialist)
    graph.add_node("technical", tech_specialist)
    graph.add_node("general",   general_specialist)
    graph.add_node("human",     human_review)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route)

    graph.add_edge("billing",   END)
    graph.add_edge("technical", END)
    graph.add_edge("general",   END)
    graph.add_edge("human",     END)

    return graph


# =============================================================================
# 6. Compile with persistent checkpointer + interrupt_before
#
# We use SqliteSaver so each user's conversation survives reloads. We
# interrupt_before the "human" node so Streamlit can show a HITL form.
# =============================================================================
@st.cache_resource
def get_compiled_app():
    """Compile the graph once per process. Streamlit caches the result."""
    conn = sqlite3.connect("support.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_graph()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human"],
    )


# =============================================================================
# 7. Streamlit UI
# =============================================================================
st.set_page_config(page_title="Customer Support Agent", page_icon="💬", layout="centered")
st.title("💬 Customer Support Agent")
st.caption("LangGraph + Streamlit · classifier · router · checkpointer · interrupt · streaming")

# Per-session thread id, so each browser tab is an independent conversation
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "history" not in st.session_state:
    st.session_state.history = []   # list of (role, content) tuples for display

app = get_compiled_app()
config = {"configurable": {"thread_id": st.session_state.thread_id}}


# -----------------------------------------------------------------------------
# Helper: render the chat so far
# -----------------------------------------------------------------------------
for entry in st.session_state.history:
    role, content = entry[0], entry[1]
    trace = entry[2] if len(entry) > 2 else []
    with st.chat_message(role):
        if role == "assistant" and trace:
            with st.expander("🧭 Agent trace", expanded=True):
                for line in trace:
                    st.markdown(line)
        st.markdown(content)


# -----------------------------------------------------------------------------
# Helper: stream the graph and display progress per node
# -----------------------------------------------------------------------------
def run_and_stream(input_messages):
    """Invoke the graph with streaming, narrating each step to the user.
    Returns (final_state, trace_lines) so the per-turn trace can be persisted
    in st.session_state.history and re-rendered on every script rerun."""
    final_state = None
    trace_lines = []
    with st.status("Agent is thinking...", expanded=True) as status:
        for chunk in app.stream(input_messages, config=config, stream_mode="updates"):
            # chunk is a dict {node_name: partial_update}
            for node_name, update in chunk.items():
                if node_name == "classify":
                    line = f"🔎 Classifier decided: **{update.get('category', '?')}**"
                elif node_name in ("billing", "technical", "general"):
                    line = f"🧑‍💼 Routed to **{node_name}** specialist"
                elif node_name == "human":
                    line = "👤 Human reviewer posted the response"
                else:
                    line = f"• {node_name}"
                trace_lines.append(line)
                st.write(line)
        # After streaming, fetch the final state from the checkpointer
        final_state = app.get_state(config)
        # Keep the status panel expanded so the trace stays visible
        status.update(label="Done", state="complete", expanded=True)
    return final_state, trace_lines


# -----------------------------------------------------------------------------
# Helper: check if the graph is paused waiting for human input
# -----------------------------------------------------------------------------
def is_waiting_for_human():
    state = app.get_state(config)
    return state.next == ("human",)


# -----------------------------------------------------------------------------
# Human-in-the-loop form (shown only when the graph paused before "human")
# -----------------------------------------------------------------------------
if is_waiting_for_human():
    st.warning("⚠️ This question needs a human reviewer.")
    with st.form("hitl_form", clear_on_submit=True):
        human_text = st.text_area("Your response as the reviewer:", height=120)
        col1, col2 = st.columns(2)
        approve = col1.form_submit_button("Send to user", type="primary")
        reject  = col2.form_submit_button("Reject and let the LLM retry")

    if approve and human_text.strip():
        # Push the human's reply into messages and resume the graph
        ai_reply = AIMessage(content=human_text.strip())
        app.update_state(config, {"messages": [ai_reply]})
        final, trace = run_and_stream(None)   # None resumes from where it stopped
        st.session_state.history.append(("assistant", human_text.strip(), trace))
        st.rerun()

    if reject:
        # Re-route as general specialist by overwriting the category
        app.update_state(config, {"category": "general"})
        final, trace = run_and_stream(None)
        last_msg = final.values["messages"][-1].content
        st.session_state.history.append(("assistant", last_msg, trace))
        st.rerun()


# -----------------------------------------------------------------------------
# Main chat input
# -----------------------------------------------------------------------------
user_input = st.chat_input("Ask anything about your account, a bug, or general info...")

if user_input:
    # Show the user's message immediately
    st.session_state.history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Run the graph
    final, trace = run_and_stream({"messages": [HumanMessage(content=user_input)]})

    # If the agent did NOT pause for a human, post the assistant's reply
    if not is_waiting_for_human():
        last_msg = final.values["messages"][-1].content
        st.session_state.history.append(("assistant", last_msg, trace))
        with st.chat_message("assistant"):
            with st.expander("🧭 Agent trace", expanded=True):
                for line in trace:
                    st.markdown(line)
            st.markdown(last_msg)
    else:
        st.rerun()   # re-render to show the HITL form


# -----------------------------------------------------------------------------
# Sidebar: thread state and reset
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Session")
    st.code(st.session_state.thread_id, language="text")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.markdown("### Resume a session")
    resume_id = st.text_input("Paste a thread_id", key="resume_input")
    if st.button("Resume") and resume_id.strip():
        st.session_state.thread_id = resume_id.strip()
        state = app.get_state({"configurable": {"thread_id": st.session_state.thread_id}})
        msgs = state.values.get("messages", []) if state.values else []
        st.session_state.history = [
            ("user" if isinstance(m, HumanMessage) else "assistant", m.content)
            for m in msgs
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("### Concepts demonstrated")
    st.markdown(
        "- State with `add_messages`\n"
        "- Classifier node (work)\n"
        "- Router function (control flow)\n"
        "- `add_conditional_edges`\n"
        "- `SqliteSaver` checkpointer\n"
        "- `interrupt_before=['human']`\n"
        "- `graph.stream(stream_mode='updates')`\n"
        "- `app.update_state(...)` to resume after HITL"
    )
