"""Client for the student's Gmail and Google Calendar.

This is the V2 window onto the student's *personal* Google data, alongside the
Albert intranet API (:mod:`api_client`) and the document RAG (:mod:`rag`). One
place owns authentication and transport, and every public ``search_*`` /
``list_*`` / ``get_*`` / ``create_*`` / ``send_*`` hands back clean Python so the
tools layer never touches the raw Google payloads.

The agent READS mail and calendar freely and can also WRITE: draft an email, send
an email, and create a calendar event. The safety boundary is **not** the scope
(these are write-capable) but the **human-in-the-loop gate** in the graph — every
outward-facing write (send mail, create event) is approved by the student before
it runs; drafting (saved, never sent) and all reads are ungated.

- **OAuth, cached.** The first consent runs the installed-app flow once and
  caches a refresh token in ``token.json``; afterwards the token refreshes
  silently. Tools call the *non-interactive* accessor and degrade to a graceful
  "not connected" message rather than ever popping a browser mid-conversation. A
  token minted with fewer scopes than the app now needs is treated as unusable,
  so the UI prompts a one-time re-consent instead of failing mid-call.

All functions raise :class:`GoogleError` on any failure (auth, network, API) so
the tools can turn it into a single ``⚠️ …`` string.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, time as dtime
from email.message import EmailMessage
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


def _token_scopes() -> set[str]:
    """Scopes actually GRANTED in the cached token (not the ones we now request).

    Read straight from ``token.json`` because a loaded ``Credentials`` object
    reports the *requested* scopes, not what the user consented to.
    """
    try:
        return set(json.loads(config.GOOGLE_TOKEN_FILE.read_text()).get("scopes", []))
    except Exception:  # noqa: BLE001 - unreadable/absent token -> no scopes
        return set()


def _credentials_from_disk() -> Credentials | None:
    """Load cached credentials, refreshing silently if needed. Never prompts.

    Returns valid :class:`Credentials`, or ``None`` if there is no usable token
    (no file, a refresh that failed, or a token granted fewer scopes than the app
    now needs — e.g. a read-only token from before write access was added). A
    browser is never opened here.
    """
    if not config.GOOGLE_TOKEN_FILE.exists():
        return None
    if not set(config.GOOGLE_SCOPES).issubset(_token_scopes()):
        log_event("google_auth", status="insufficient_scopes")
        return None  # token predates a scope we now require -> force re-consent
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
# Gmail (write): draft is ungated; send is gated by the approval node
# --------------------------------------------------------------------------- #
def _build_raw(to: str, subject: str, body: str) -> str:
    """Build a base64url-encoded MIME message for the Gmail API."""
    message = EmailMessage()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)
    return base64.urlsafe_b64encode(message.as_bytes()).decode()


def create_draft(to: str, subject: str, body: str) -> dict:
    """Save an email as a Gmail draft (NOT sent). Returns ``{id, to, subject}``."""
    try:
        draft = (
            _gmail().users().drafts()
            .create(userId="me",
                    body={"message": {"raw": _build_raw(to, subject, body)}})
            .execute()
        )
        return {"id": draft.get("id"), "to": to, "subject": subject}
    except HttpError as exc:
        raise GoogleError(f"Gmail API error: {exc.reason}") from exc
    except GoogleError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GoogleError(f"Could not create the draft: {exc}") from exc


def send_message(to: str, subject: str, body: str) -> dict:
    """Send an email immediately. Returns ``{id, to, subject}``."""
    try:
        sent = (
            _gmail().users().messages()
            .send(userId="me", body={"raw": _build_raw(to, subject, body)})
            .execute()
        )
        return {"id": sent.get("id"), "to": to, "subject": subject}
    except HttpError as exc:
        raise GoogleError(f"Gmail API error: {exc.reason}") from exc
    except GoogleError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GoogleError(f"Could not send the email: {exc}") from exc


# --------------------------------------------------------------------------- #
# Google Calendar (read + create)
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


def _event_time_field(value: str) -> dict:
    """Turn an ISO date/datetime into a Calendar start/end field.

    A bare date ("2026-06-20") yields an all-day field (``{"date": ...}``); a
    datetime yields a timed field (``{"dateTime": ...}``) in the local timezone.
    """
    raw = value.strip()
    if len(raw) <= 10:  # date only -> all-day
        return {"date": raw}
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.astimezone()  # interpret naive datetimes as local time
    return {"dateTime": dt.isoformat()}


def create_event(
    summary: str,
    start: str,
    end: str | None = None,
    description: str | None = None,
    location: str | None = None,
) -> dict:
    """Create a primary-calendar event. Returns ``{summary, start, end, link}``.

    ``start``/``end`` are ISO dates or datetimes. When ``end`` is omitted, a timed
    event lasts :data:`config.CALENDAR_DEFAULT_EVENT_HOURS` and an all-day event
    covers a single day (Google treats ``end.date`` as exclusive).
    """
    try:
        start_field = _event_time_field(start)
        if end:
            end_field = _event_time_field(end)
        elif "dateTime" in start_field:
            started = datetime.fromisoformat(start_field["dateTime"])
            ended = started + timedelta(hours=config.CALENDAR_DEFAULT_EVENT_HOURS)
            end_field = {"dateTime": ended.isoformat()}
        else:  # all-day: end is the next day
            next_day = datetime.fromisoformat(start_field["date"]).date() + timedelta(days=1)
            end_field = {"date": next_day.isoformat()}

        body = {"summary": summary, "start": start_field, "end": end_field}
        if description:
            body["description"] = description
        if location:
            body["location"] = location

        created = _calendar().events().insert(calendarId="primary", body=body).execute()
        return {
            "summary": created.get("summary", summary),
            "start": _event_when(created.get("start", {})),
            "end": _event_when(created.get("end", {})),
            "link": created.get("htmlLink", ""),
        }
    except HttpError as exc:
        raise GoogleError(f"Calendar API error: {exc.reason}") from exc
    except GoogleError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise GoogleError(f"Could not create the event: {exc}") from exc
