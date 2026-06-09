"""Central configuration: environment variables and tunable constants.

Everything the rest of the package needs to be told (secrets, model choice,
safety limits, on-disk locations) lives here so there is a single place to
look. Secrets are read from a ``.env`` file via ``python-dotenv``; they are
never hard-coded and never logged.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env once, at import time. Real environment variables win over the file.
load_dotenv()

# --------------------------------------------------------------------------- #
# Filesystem layout (all paths are absolute, derived from this file)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_PDF_DIR = PROJECT_ROOT / "rag"                 # the 4 source PDFs
CHROMA_DIR = PROJECT_ROOT / ".chroma"              # persisted vector store
CHECKPOINT_DB = PROJECT_ROOT / ".checkpoints.sqlite"  # conversation persistence
LOG_FILE = PROJECT_ROOT / "albert_agent.log"       # structured run log

# --------------------------------------------------------------------------- #
# Secrets (read lazily where used; may be ``None`` until validated)
# --------------------------------------------------------------------------- #
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALBERT_PERSONAL_TOKEN = os.getenv("ALBERT_PERSONAL_TOKEN")
# ⚠️ Two DISTINCT student identifiers — do not mix them up:
#   USER_ID    keys the /student/.../course-module-instances and /attendance endpoints
#   STUDENT_ID keys the /student-exam-grade/student/... (grades) endpoint
ALBERT_USER_ID = os.getenv("ALBERT_USER_ID")
ALBERT_STUDENT_ID = os.getenv("ALBERT_STUDENT_ID")

# --------------------------------------------------------------------------- #
# Albert intranet API
# --------------------------------------------------------------------------- #
API_BASE_URL = "https://api-inside.albertschool.com"
REQUEST_TIMEOUT = 20  # seconds, per HTTP call

# --------------------------------------------------------------------------- #
# LLM (Groq)
# --------------------------------------------------------------------------- #
# llama-3.3-70b-versatile is strong at routing/tool-use and supports parallel
# tool calls; the 8b instant model is kept as a documented cheaper fallback.
MODEL_NAME = os.getenv("ALBERT_AGENT_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.0          # deterministic: this is an information assistant
MAX_TOKENS = 1024          # cap each LLM turn (part of the safety harness)

# --------------------------------------------------------------------------- #
# Safety harness
# --------------------------------------------------------------------------- #
# Hard ceiling on agent<->tools loops within a single user turn. The graph's
# conditional edge routes to the ``fallback`` node once this is exceeded, so a
# model that keeps calling tools forever can never run away.
MAX_TOOL_ITERATIONS = 6
# Backstop handed to LangGraph itself in case the structural cap is bypassed.
GRAPH_RECURSION_LIMIT = 25

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
GRADE_SCALE = 100          # the live API grades are out of 100
GRADE_SCALE_FR = 20        # also shown on the French /20 scale for readability
# The PDF documents are short and tabular (1-2 pages each); large chunks keep a
# transcript's teaching-unit table intact instead of fragmenting it.
RAG_CHUNK_SIZE = 2000
RAG_CHUNK_OVERLAP = 200
RAG_TOP_K = 4              # chunks returned per retrieval
RAG_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # documents are in English


class ConfigError(RuntimeError):
    """Raised when a required secret is missing from the environment."""


def require(*names: str) -> None:
    """Fail fast with a clear message if any required env var is unset.

    Called by the layers that actually need a given secret (the API client,
    the LLM factory) rather than at import time, so that e.g. building the RAG
    index does not demand the Albert token.
    """
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Add them to the .env file at the project root."
        )
