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
# Google OAuth: the downloaded client-id file, and the cached user token.
# Both are secrets and are gitignored; token.json is minted on first consent.
GOOGLE_CREDENTIALS_FILE = Path(
    os.getenv("GOOGLE_CREDENTIALS_FILE", PROJECT_ROOT / "credentials.json")
)
GOOGLE_TOKEN_FILE = Path(
    os.getenv("GOOGLE_TOKEN_FILE", PROJECT_ROOT / "token.json")
)

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
# Primary model: llama-3.3-70b-versatile is strong at routing/tool-use and
# supports parallel tool calls.
MODEL_NAME = os.getenv("ALBERT_AGENT_MODEL", "llama-3.3-70b-versatile")

# Fallback CHAIN, tried in order when a model is rate-limited or unavailable.
# Groq's free tier caps tokens-per-day *per model*, so one model running out does
# not affect the others — walking this chain keeps the agent available as long as
# any one model has quota. Ordered capability-first so quality degrades
# gracefully. Every entry supports the LOCAL tool-calling the agent depends on;
# `groq/compound` and `groq/compound-mini` are deliberately excluded because the
# Groq "Supported Models" table marks them "Local Tool Use: No".
FALLBACK_MODELS = [
    "openai/gpt-oss-120b",                        # strong, separate quota
    "qwen/qwen3-32b",                             # strong, parallel tools
    "meta-llama/llama-4-scout-17b-16e-instruct",  # parallel tools
    "openai/gpt-oss-20b",                         # smaller but capable
    "llama-3.1-8b-instant",                       # fast/cheap last resort
]

TEMPERATURE = 0.0          # deterministic: this is an information assistant
MAX_TOKENS = 1024          # cap each LLM turn (part of the safety harness)

# --------------------------------------------------------------------------- #
# Google Workspace (Gmail + Calendar) — READ-ONLY by design
# --------------------------------------------------------------------------- #
# The agent only ever READS the student's mail and calendar; the scopes below are
# the read-only variants, so even a compromised token cannot send mail or alter
# events. Changing these requires deleting token.json and re-consenting.
GOOGLE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
]
GMAIL_MAX_RESULTS = 5          # default number of emails a search returns
GMAIL_BODY_MAX_CHARS = 4000    # truncate a single email's body fed to the model
CALENDAR_MAX_RESULTS = 10      # default number of events a lookup returns
CALENDAR_DEFAULT_DAYS_AHEAD = 7  # default window when the user gives no end date

# --------------------------------------------------------------------------- #
# Safety harness
# --------------------------------------------------------------------------- #
# Human-in-the-loop: reading the inbox is the one genuinely private action, so by
# default the graph pauses (interrupt) for the student's approval before any
# email tool runs. The UI exposes a toggle; the eval flips it per case.
REQUIRE_EMAIL_APPROVAL = True
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
