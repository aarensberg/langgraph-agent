"""Read-only client for the student's Gmail and Google Calendar.

This is the V2 window onto the student's *personal* Google data, alongside the
Albert intranet API (:mod:`api_client`) and the document RAG (:mod:`rag`). It is
deliberately the mirror image of :mod:`api_client`: one place owns authentication
and transport, and every public ``search_*`` / ``list_*`` / ``get_*`` hands back
clean Python dicts so the tools layer never touches the raw Google payloads.

Two safety properties are baked in here, not bolted on by the caller:

- **Read-only scopes.** Only ``gmail.readonly`` and ``calendar.readonly`` are
  requested (see :data:`config.GOOGLE_SCOPES`), so the agent physically cannot
  send mail or change the calendar even if it tried.
- **OAuth, cached.** The first consent runs the installed-app flow once and
  caches a refresh token in ``token.json``; afterwards the token refreshes
  silently. Tools call the *non-interactive* accessor and degrade to a graceful
  "not connected" message rather than ever popping a browser mid-conversation.

All functions raise :class:`GoogleError` on any failure (auth, network, API) so
the tools can turn it into a single ``⚠️ …`` string.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, time as dtime
from functools import lru_cache
from html.parser import HTMLParser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from . import config
from .observability import log_event


class GoogleError(RuntimeError):
    """Any failure talking to Google (missing auth, network, or API error)."""


# --------------------------------------------------------------------------- #
# Authentication (OAuth installed-app flow, with a cached refresh token)
# --------------------------------------------------------------------------- #
def _save(creds: Credentials) -> None:
    """Persist the credentials (incl. refresh token) to ``token.json``."""
    config.GOOGLE_TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")


def _credentials_from_disk() -> Credentials | None:
    """Load cached credentials, refreshing silently if needed. Never prompts.

    Returns valid :class:`Credentials`, or ``None`` if there is no usable token
    (no file, or a refresh that failed). A browser is never opened here.
    """
    if not config.GOOGLE_TOKEN_FILE.exists():
        return None
    creds = Credentials.from_authorized_user_file(
        str(config.GOOGLE_TOKEN_FILE), config.GOOGLE_SCOPES
    )
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            _save(creds)
        except Exception as exc:  # noqa: BLE001 - stale/revoked token -> reconnect
            log_event("google_auth", status="refresh_failed", error=type(exc).__name__)
            return None
    return creds if creds and creds.valid else None


def is_connected() -> bool:
    """True iff a usable token exists (used by the UI/eval; never prompts)."""
    try:
        return _credentials_from_disk() is not None
    except Exception:  # noqa: BLE001 - any problem means "not connected"
        return False


def authenticate() -> Credentials:
    """Interactively connect a Google account, caching the token for reuse.

    Returns immediately if a valid cached token already exists; otherwise runs
    the installed-app consent flow (opens a browser, spins a temporary local
    server to catch the redirect) and saves ``token.json``. Used by the one-time
    CLI (``python -m albert_agent.google_auth``) and the UI "Connect" button.
    """
    creds = _credentials_from_disk()
    if creds:
        return creds
    if not config.GOOGLE_CREDENTIALS_FILE.exists():
        raise GoogleError(
            f"OAuth client file not found at {config.GOOGLE_CREDENTIALS_FILE}. "
            "Download an OAuth 2.0 client id ('Desktop app') from Google Cloud "
            "Console and save it there as credentials.json."
        )
    flow = InstalledAppFlow.from_client_secrets_file(
        str(config.GOOGLE_CREDENTIALS_FILE), config.GOOGLE_SCOPES
    )
    creds = flow.run_local_server(port=0)
    _save(creds)
    log_event("google_auth", status="connected", scopes=len(config.GOOGLE_SCOPES))
    return creds


def _require_credentials() -> Credentials:
    """Return valid credentials for a tool call, or raise a clear error."""
    creds = _credentials_from_disk()
    if creds is None:
        raise GoogleError(
            "Google account not connected. Connect it once with "
            "`python -m albert_agent.google_auth` (or the 'Connect Google' "
            "button in the app), then try again."
        )
    return creds


@lru_cache(maxsize=1)
def _gmail():
    return build("gmail", "v1", credentials=_require_credentials(),
                 cache_discovery=False)


@lru_cache(maxsize=1)
def _calendar():
    return build("calendar", "v3", credentials=_require_credentials(),
                 cache_discovery=False)


# --------------------------------------------------------------------------- #
# Gmail (read-only)
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    """Minimal HTML -> text fallback for emails with no text/plain part."""

    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join(" ".join(self._chunks).split())


def _b64(data: str) -> str:
    """Decode a base64url Gmail body part to text."""
    return base64.urlsafe_b64decode(data.encode("utf-8")).decode("utf-8", "replace")


def _header(headers: list[dict], name: str) -> str:
    """Case-insensitively read one header value from a Gmail header list."""
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def _extract_body(payload: dict) -> str:
    """Walk a Gmail MIME payload and return the best plain-text body."""
    plain, html = [], []

    def walk(part: dict) -> None:
        mime = part.get("mimeType", "")
        data = part.get("body", {}).get("data")
        if data and mime == "text/plain":
            plain.append(_b64(data))
        elif data and mime == "text/html":
            html.append(_b64(data))
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    if plain:
        return "\n".join(plain).strip()
    if html:
        parser = _TextExtractor()
        parser.feed("\n".join(html))
        return parser.text.strip()
    return ""


def search_messages(query: str = "", max_results: int | None = None) -> list[dict]:
    """Return recent messages matching a Gmail query, newest first.

    ``query`` uses Gmail's own search syntax (e.g. ``from:prof subject:exam``,
    ``is:unread``, ``newer_than:7d``). Each item is
    ``{id, from, subject, date, snippet}``.
    """
    limit = max_results or config.GMAIL_MAX_RESULTS
    try:
        service = _gmail()
        listed = (
            service.users().messages()
            .list(userId="me", q=query or "", maxResults=limit)
            .execute()
        )
        out = []
        for ref in listed.get("messages", []):
            msg = (
                service.users().messages()
                .get(userId="me", id=ref["id"], format="metadata",
                     metadataHeaders=["From", "Subject", "Date"])
                .execute()
            )
            headers = msg.get("payload", {}).get("headers", [])
            out.append({
                "id": msg["id"],
                "from": _header(headers, "From"),
                "subject": _header(headers, "Subject") or "(no subject)",
                "date": _header(headers, "Date"),
                "snippet": (msg.get("snippet") or "").strip(),
            })
        return out
    except HttpError as exc:
        raise GoogleError(f"Gmail API error: {exc.reason}") from exc
    except GoogleError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GoogleError(f"Could not search Gmail: {exc}") from exc


def get_message(message_id: str) -> dict:
    """Return one full message: ``{id, from, to, subject, date, body}``.

    The body is the decoded plain-text content (HTML is stripped as a fallback),
    truncated to :data:`config.GMAIL_BODY_MAX_CHARS`.
    """
    try:
        msg = (
            _gmail().users().messages()
            .get(userId="me", id=message_id, format="full")
            .execute()
        )
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])
        body = _extract_body(payload) or (msg.get("snippet") or "").strip()
        if len(body) > config.GMAIL_BODY_MAX_CHARS:
            body = body[: config.GMAIL_BODY_MAX_CHARS] + "\n…[truncated]"
        return {
            "id": msg["id"],
            "from": _header(headers, "From"),
            "to": _header(headers, "To"),
            "subject": _header(headers, "Subject") or "(no subject)",
            "date": _header(headers, "Date"),
            "body": body,
        }
    except HttpError as exc:
        raise GoogleError(f"Gmail API error: {exc.reason}") from exc
    except GoogleError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GoogleError(f"Could not read the email: {exc}") from exc


# --------------------------------------------------------------------------- #
# Google Calendar (read-only)
# --------------------------------------------------------------------------- #
def _parse_dt(value: str, *, end_of_day: bool) -> datetime:
    """Coerce a date or datetime string to a timezone-aware ``datetime``.

    Accepts ISO dates ("2026-06-12") and full ISO datetimes. A bare date is
    expanded to the start (or, for ``end_of_day``, the end) of that day, and a
    naive value is interpreted in the machine's local timezone.
    """
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise GoogleError(
            f"Could not understand the date '{value}'. Use ISO format like "
            "2026-06-12 or 2026-06-12T09:00."
        ) from exc
    if parsed.tzinfo is None and len(value.strip()) <= 10:  # date only
        edge = dtime.max if end_of_day else dtime.min
        parsed = datetime.combine(parsed.date(), edge)
    return parsed.astimezone()  # make tz-aware (assumes local if naive)


def _event_when(slot: dict) -> str:
    """Render a Calendar start/end slot, which is either a date or a dateTime."""
    if "dateTime" in slot:
        dt = datetime.fromisoformat(slot["dateTime"].replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    if "date" in slot:
        return f"{slot['date']} (all day)"
    return "?"


def list_events(
    start: str | None = None,
    end: str | None = None,
    query: str | None = None,
    max_results: int | None = None,
) -> list[dict]:
    """Return calendar events in a time window, ordered by start time.

    Defaults to the next :data:`config.CALENDAR_DEFAULT_DAYS_AHEAD` days from now.
    Each item is ``{summary, start, end, location}``.
    """
    now = datetime.now().astimezone()
    time_min = _parse_dt(start, end_of_day=False) if start else now
    time_max = (
        _parse_dt(end, end_of_day=True) if end
        else time_min + timedelta(days=config.CALENDAR_DEFAULT_DAYS_AHEAD)
    )
    try:
        events = (
            _calendar().events()
            .list(
                calendarId="primary",
                timeMin=time_min.isoformat(),
                timeMax=time_max.isoformat(),
                singleEvents=True,
                orderBy="startTime",
                q=query or None,
                maxResults=max_results or config.CALENDAR_MAX_RESULTS,
            )
            .execute()
        )
        return [
            {
                "summary": ev.get("summary", "(no title)"),
                "start": _event_when(ev.get("start", {})),
                "end": _event_when(ev.get("end", {})),
                "location": ev.get("location", ""),
            }
            for ev in events.get("items", [])
        ]
    except HttpError as exc:
        raise GoogleError(f"Calendar API error: {exc.reason}") from exc
    except GoogleError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GoogleError(f"Could not read the calendar: {exc}") from exc
