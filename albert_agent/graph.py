"""The LangGraph agent: state, nodes, the conditional edge, and compilation.

The shape is the canonical ReAct loop, built explicitly as a ``StateGraph`` so
every part is visible and testable:

    START -> guard -> agent -> route_after_agent -+- "tools"    -> tools -> (back to agent)
                       ^                          |
                       +------ (tools loop) ------+- "fallback" -> fallback -> END
                                                  +- "end"      -> END

``route_after_agent`` is the one meaningful conditional edge and it routes three
different ways: to the tools when the model asked for them and the loop budget
is intact, to ``fallback`` when that budget is exhausted (the safety harness),
or straight to END when the model produced a final answer. The per-turn loop
counter lives in the state and is reset by ``guard`` at the start of every turn.
"""

from __future__ import annotations

import re
import sqlite3
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from . import config
from .observability import log_event, setup_langsmith
from .tools import TOOLS

# --------------------------------------------------------------------------- #
# System prompt (documented in the README; it is part of the design)
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
You are the Albert School student assistant. You help one signed-in student get \
clear, accurate answers about their own studies: their program and current \
courses, a course's assessment/topics/documents, their attendance, and their \
grades.

How to work:
- Always ground answers in tool results. Never invent grades, rates, dates, \
course names or documents. If you don't have it, say so.
- Use your tools to fetch program/course info, attendance rates, grade \
averages, and the official PDF documents; use the calculator for any arithmetic.
- Mind the two grade sources, they are different: the live API grades are on a \
0-100 scale and reflect the current standing; the transcripts in the PDF \
documents are the official historical record on the French /20 scale with \
letter grades and ECTS. Do not mix them, and search the documents when the user \
asks about a past year, a transcript, or their enrollment certificate.
- You may call several tools at once when a question needs more than one \
(e.g. attendance and grades together), but do not call the same tool twice with \
the same arguments.
- After your tools return, write a complete, direct answer for the student using \
their results — give the actual numbers and facts. Never reply with meta-comments \
about the tool calls, and never mention internal tool or function names; describe \
what you can do in plain words.
- Be concise and factual. Reply in the same language the student used \
(French or English).
"""


# --------------------------------------------------------------------------- #
# State — only the two fields the graph reads and writes
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # chat history + tool results (reducer)
    iterations: int                          # tool-loop counter for this turn


# --------------------------------------------------------------------------- #
# Model (memoised: bind the tools once per model)
# --------------------------------------------------------------------------- #
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@lru_cache(maxsize=8)
def _bound_llm(model_name: str):
    """Build a tool-bound Groq model, memoised per model name."""
    llm = ChatGroq(
        model=model_name,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,  # caps each turn — part of the safety harness
    )
    return llm.bind_tools(TOOLS)


def get_llm():
    """Return the primary tool-bound model."""
    config.require("GROQ_API_KEY")
    return _bound_llm(config.MODEL_NAME)


def _model_chain() -> list[str]:
    """The primary model followed by the distinct fallback models, in order."""
    chain = [config.MODEL_NAME]
    for name in config.FALLBACK_MODELS:
        if name not in chain:
            chain.append(name)
    return chain


def _clean(response):
    """Strip <think>…</think> reasoning that some models emit into their content."""
    content = getattr(response, "content", None)
    if isinstance(content, str) and "<think>" in content:
        response.content = _THINK_RE.sub("", content).strip()
    return response


def invoke_model(messages):
    """Invoke the model chain, returning the first model that answers.

    Groq's free tier caps tokens-per-day *per model*, so a single model can run
    out mid-session. Keeping the agent available is exactly the failure the safety
    harness exists for, so we walk the fallback chain: on *any* model error we log
    it and try the next, and only raise once every model in the chain has failed
    (the UI then shows a graceful message). The assistant therefore keeps working
    as long as one Groq model still has quota.
    """
    config.require("GROQ_API_KEY")
    chain = _model_chain()
    last_exc = None
    for index, model_name in enumerate(chain):
        try:
            response = _bound_llm(model_name).invoke(messages)
        except Exception as exc:  # noqa: BLE001 - try the next model, whatever broke
            last_exc = exc
            log_event("llm_skip", model=model_name, error=type(exc).__name__)
            continue
        if index > 0:
            log_event("llm_fallback", used=model_name, primary=chain[0])
        return _clean(response)
    raise RuntimeError(
        f"All {len(chain)} Groq models are unavailable "
        f"(last error: {type(last_exc).__name__}). Please try again shortly."
    ) from last_exc


# --------------------------------------------------------------------------- #
# Nodes (each is State -> partial State)
# --------------------------------------------------------------------------- #
def guard(state: AgentState) -> dict:
    """Entry node: reset the per-turn loop counter.

    The checkpointer persists ``iterations`` across turns, so without this reset
    the second question in a thread would start already over budget. ``guard``
    runs once per user turn (it is the entry point) and zeroes it.
    """
    log_event("turn_start", history_len=len(state["messages"]))
    return {"iterations": 0}


def agent(state: AgentState) -> dict:
    """Call the LLM. It either answers directly or asks for tool calls."""
    response = invoke_model([SystemMessage(content=SYSTEM_PROMPT)] + state["messages"])
    calls = [c["name"] for c in getattr(response, "tool_calls", [])]
    log_event("agent", tool_calls=calls or "none",
              iteration=state.get("iterations", 0) + 1)
    return {"messages": [response], "iterations": state.get("iterations", 0) + 1}


def tools_node(state: AgentState) -> dict:
    """Run every tool the model requested (supports parallel calls).

    Each call is executed defensively: tools already return graceful "⚠️ ..."
    strings on failure, and this extra guard catches anything that still slips
    through, so one bad tool call never takes down the graph.
    """
    registry = {t.name: t for t in TOOLS}
    last = state["messages"][-1]
    results = []
    seen: dict[str, str] = {}  # cache identical (name, args) calls within a turn
    for call in last.tool_calls:
        key = f"{call['name']}:{call['args']}"
        if key in seen:  # model asked for the same thing twice -> reuse, don't redo
            content = seen[key]
        else:
            tool = registry.get(call["name"])
            if tool is None:
                content = f"⚠️ Unknown tool '{call['name']}'."
            else:
                try:
                    content = str(tool.invoke(call["args"]))
                except Exception as exc:  # noqa: BLE001 - backstop around the tool
                    content = f"⚠️ Tool '{call['name']}' failed: {exc}"
            seen[key] = content
        results.append(
            ToolMessage(content=content, tool_call_id=call["id"], name=call["name"])
        )
    return {"messages": results}


def fallback(state: AgentState) -> dict:
    """Safety exit when the tool-loop budget is exhausted."""
    log_event("fallback", reason="max_iterations", limit=config.MAX_TOOL_ITERATIONS)
    return {
        "messages": [
            AIMessage(
                content=(
                    "I reached my internal step limit for this question and "
                    "stopped to avoid looping. Could you narrow it down or "
                    "rephrase? I can help with your program, a course, your "
                    "attendance, or your grades."
                )
            )
        ]
    }


# --------------------------------------------------------------------------- #
# The conditional edge — routes three ways
# --------------------------------------------------------------------------- #
def route_after_agent(state: AgentState) -> Literal["tools", "fallback", "end"]:
    """Decide what happens after the model spoke, based on the State."""
    last = state["messages"][-1]
    if not getattr(last, "tool_calls", None):
        log_event("route", decision="end")
        return "end"  # the model gave a final answer
    if state.get("iterations", 0) >= config.MAX_TOOL_ITERATIONS:
        log_event("route", decision="fallback", iterations=state["iterations"])
        return "fallback"  # safety: too many tool loops this turn
    log_event("route", decision="tools", iterations=state["iterations"])
    return "tools"  # run the requested tool(s), then loop back to the model


# --------------------------------------------------------------------------- #
# Build & compile
# --------------------------------------------------------------------------- #
def build_graph() -> StateGraph:
    """Wire the nodes and edges (no checkpointer yet)."""
    graph = StateGraph(AgentState)
    graph.add_node("guard", guard)
    graph.add_node("agent", agent)
    graph.add_node("tools", tools_node)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "guard")
    graph.add_edge("guard", "agent")
    graph.add_conditional_edges(
        "agent",
        route_after_agent,
        {"tools": "tools", "fallback": "fallback", "end": END},
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("fallback", END)
    return graph


def get_agent(checkpointer=None):
    """Compile the graph with a checkpointer for persistence.

    Defaults to a SQLite checkpointer on disk so a conversation survives a page
    reload or a server restart. Pass an explicit checkpointer (e.g. an
    in-memory one) for tests.
    """
    setup_langsmith()  # enable tracing if a LangSmith key is configured
    if checkpointer is None:
        conn = sqlite3.connect(str(config.CHECKPOINT_DB), check_same_thread=False)
        checkpointer = SqliteSaver(conn)
    return build_graph().compile(checkpointer=checkpointer)
