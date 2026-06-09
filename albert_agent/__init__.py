"""Albert School student assistant — a LangGraph tool-calling agent.

The package is organised in layers, from the outside in:

- ``config``        : environment + tunable constants (single source of truth)
- ``observability`` : structured logging and optional LangSmith tracing
- ``api_client``    : thin authenticated wrapper over the Albert intranet API
- ``aggregations``  : pandas helpers that derive rates/averages from raw records
- ``rag``           : Chroma retriever over the official PDF documents
- ``tools``         : the LangChain tools the agent is allowed to call
- ``graph``         : the StateGraph (state, nodes, conditional edge, checkpointer)

The Streamlit UI in ``app.py`` only ever touches ``graph.get_agent`` and the
public helpers it re-exports, never the lower layers directly.
"""

__all__ = ["config"]
