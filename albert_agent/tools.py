"""The tools the agent is allowed to call.

Twelve tools, each with a single responsibility and its own error handling. The
four Albert-API tools do all the data wrangling in Python (via
:mod:`aggregations`) and hand the model clean, already-computed text — the LLM
never sees raw JSON and is never asked to do arithmetic. The RAG tool answers
from the official PDFs, the calculator covers hypotheticals, and six Google tools
bring in the student's mail and schedule — three that read, three that write.

Tool boundaries (what each one owns):
    list_my_program         -> identity + this semester's course list
    get_course_details      -> one course: assessment, topics, documents
    get_attendance_summary  -> attendance rate (overall / by course / by month)
    get_grades_summary      -> averages (overall / by teaching unit / by course)
    search_school_documents -> the enrollment certificate & historical transcripts
    calculator              -> safe arithmetic on numbers the user/agent supplies
    search_emails           -> recent/matching Gmail messages (read)
    read_email              -> the full body of one email (read)
    get_calendar_events     -> Google Calendar events in a time window (read)
    draft_email             -> save an email draft, never sent (write, ungated)
    send_email              -> send an email (write, sensitive)
    create_calendar_event   -> create a calendar event (write, sensitive)

Every tool returns a string. On failure it returns a message starting with
"⚠️" rather than raising, so the agent can tell the user something useful and
carry on instead of crashing the graph.

:data:`SENSITIVE_TOOLS` lists the two outward-facing WRITES — ``send_email`` and
``create_calendar_event`` — because they change things in the real world: the
graph pauses for the student's approval before they run (see ``human_approval``
in :mod:`albert_agent.graph`). Reads and drafting an email are never gated.
"""

from __future__ import annotations

import ast
import operator
from typing import Callable

from langchain_core.tools import tool

from . import aggregations as agg
from . import api_client, config, google_client, rag
from .api_client import AlbertAPIError
from .google_client import GoogleError
from .observability import log_tool_call


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _fr(score_100: float | None) -> str:
    """Render a /100 figure with its /20 equivalent, e.g. ``79.5/100 (15.9/20)``."""
    if score_100 is None:
        return "n/a"
    return f"{score_100:.1f}/100 ({score_100 * config.GRADE_SCALE_FR / config.GRADE_SCALE:.1f}/20)"


def _resolve_course(course_name: str) -> tuple[dict | None, list[dict], list[str]]:
    """Find a course by fuzzy name, preferring the current term.

    Returns ``(match, candidates, current_term_names)``. If exactly one course
    matches, ``match`` is set; if several match, ``candidates`` lists them; if
    none match, both are empty and ``current_term_names`` helps the caller
    suggest valid options.
    """
    courses = api_client.fetch_courses()
    year, semester = agg.current_term(courses)
    current = agg.filter_courses_for_term(courses, year, semester)
    current_names = sorted(
        c["course_module_instance_name"].strip() for c in current
    )

    needle = course_name.strip().lower()

    def matches(pool: list[dict]) -> list[dict]:
        return [c for c in pool if needle in c["course_module_instance_name"].strip().lower()]

    hits = matches(current) or matches(courses)
    if len(hits) == 1:
        return hits[0], [], current_names
    return None, hits, current_names


def _program_version_id() -> str | None:
    """Resolve the student's program_version_id (needed for syllabus lookups).

    The profile carries the program *name* but not its version id, and its
    academic_program_id does not match the public catalog, so we match by name.
    Returns ``None`` if no catalog entry matches (caller degrades gracefully).
    """
    profile = api_client.fetch_profile()
    name = profile.get("academic_program_name")
    for program in api_client.fetch_programs():
        if program.get("name") == name:
            return program.get("program_version_id")
    return None


# --------------------------------------------------------------------------- #
# Tool 1 — program / identity
# --------------------------------------------------------------------------- #
@tool
def list_my_program() -> str:
    """List the student's program enrolment and the courses they take THIS semester.

    Use this for questions like "what program am I in", "which campus", "what
    courses am I taking this semester", "how many courses do I have". Returns the
    program name, level, track, campus and enrolment status, then one line per
    current-semester course (name, code, ECTS, teaching unit, teacher).
    """
    try:
        with log_tool_call("list_my_program", {}):
            profile = api_client.fetch_profile()
            courses = api_client.fetch_courses()
            year, semester = agg.current_term(courses)
            current = agg.filter_courses_for_term(courses, year, semester)

            header = (
                f"Student: {profile.get('first_name')} {profile.get('last_name')}\n"
                f"Program: {profile.get('academic_program_name')} "
                f"({profile.get('academic_program_level')}, "
                f"track: {profile.get('academic_program_track')})\n"
                f"Campus: {profile.get('campus_city')} | "
                f"Status: {profile.get('enrollment_status')}\n"
                f"Current term: {year} {semester} — {len(current)} courses\n"
            )
            lines = []
            for c in sorted(current, key=lambda x: x["course_module_instance_name"].strip()):
                teacher = c.get("teacher_name") or (
                    f"{c.get('teacher_first_name') or ''} {c.get('teacher_last_name') or ''}".strip()
                ) or "TBA"
                lines.append(
                    f"- {c['course_module_instance_name'].strip()} "
                    f"[{c.get('course_module_instance_code')}] · "
                    f"{c.get('ects')} ECTS · {c.get('teaching_unit_instance_name')} · {teacher}"
                )
            return header + "\n".join(lines)
    except AlbertAPIError as exc:
        return f"⚠️ Could not load your program: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while listing your program: {exc}"


# --------------------------------------------------------------------------- #
# Tool 2 — course details (assessment, topics, documents)
# --------------------------------------------------------------------------- #
@tool
def get_course_details(course_name: str) -> str:
    """Get details about ONE course: assessment rules, topics and documents.

    Use for "how is <course> graded / what's the assessment", "what topics does
    <course> cover", "what documents are available for <course>". ``course_name``
    can be partial (e.g. "generative AI"); if it matches several courses the tool
    asks you to disambiguate. Returns the teacher and ECTS, the assessment formula
    and weights, the syllabus topics and learning outcomes, and the list of
    documents the teacher uploaded.
    """
    try:
        with log_tool_call("get_course_details", {"course_name": course_name}):
            match, candidates, current_names = _resolve_course(course_name)
            if not match:
                if candidates:
                    names = ", ".join(c["course_module_instance_name"].strip() for c in candidates)
                    return f"Several courses match '{course_name}': {names}. Which one?"
                return (
                    f"No course matches '{course_name}'. Your current courses are: "
                    + ", ".join(current_names)
                )

            course_id = match["id"]
            name = match["course_module_instance_name"].strip()
            teacher = match.get("teacher_name") or (
                f"{match.get('teacher_first_name') or ''} {match.get('teacher_last_name') or ''}".strip()
            ) or "TBA"
            out = [
                f"Course: {name} [{match.get('course_module_instance_code')}]",
                f"Teacher: {teacher} | {match.get('ects')} ECTS | "
                f"{match.get('duration_hours')}h | {match.get('semester')} {match.get('academic_year')}",
            ]

            # The course LIST omits course_module_version_id; only the by-id
            # endpoint carries it. Fetch the instance (best effort) to unlock the
            # syllabus lookup.
            cmv_id = match.get("course_module_version_id")
            if not cmv_id:
                try:
                    cmv_id = api_client.fetch_course_instance(course_id).get(
                        "course_module_version_id")
                except AlbertAPIError:
                    cmv_id = None

            # Syllabus (assessment + topics + outcomes) — best effort.
            pv_id = _program_version_id()
            if pv_id and cmv_id:
                try:
                    syl = api_client.fetch_syllabus(pv_id, cmv_id)
                    evaluation = (syl.get("content", {}).get("teacher") or {}).get("evaluation") or {}
                    if evaluation.get("formula"):
                        out.append(f"\nAssessment: {evaluation['formula']}")
                    for m in evaluation.get("methods") or []:
                        out.append(f"  · {m.get('code')} ({m.get('weight')}%): "
                                   f"{m.get('name')} — {m.get('type')}")
                    generated = syl.get("content", {}).get("generated") or {}
                    topics = generated.get("topics") or []
                    if topics:
                        out.append("\nTopics:")
                        out += [f"  {i}. {t}" for i, t in enumerate(topics, 1)]
                    outcomes = generated.get("learning_outcomes") or []
                    if outcomes:
                        out.append("\nLearning outcomes:")
                        out += [f"  · {o}" for o in outcomes]
                except AlbertAPIError:
                    out.append("\n(Syllabus details unavailable.)")

            # Documents.
            try:
                docs = api_client.fetch_course_documents(course_id)
                if docs:
                    out.append(f"\nDocuments ({len(docs)}):")
                    out += [f"  · {d.get('document_name')} "
                            f"({d.get('original_filename')})" for d in docs]
                else:
                    out.append("\nNo documents uploaded for this course.")
            except AlbertAPIError:
                out.append("\n(Document list unavailable.)")

            return "\n".join(out)
    except AlbertAPIError as exc:
        return f"⚠️ Could not load course details: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while loading course details: {exc}"


# --------------------------------------------------------------------------- #
# Tool 3 — attendance
# --------------------------------------------------------------------------- #
@tool
def get_attendance_summary(course_name: str | None = None, month: str | None = None) -> str:
    """Report the student's attendance rate, overall or filtered.

    Use for "what's my attendance rate", "how many classes did I miss", "my
    attendance in <course>", "my attendance in March". Leave both arguments empty
    for the overall rate plus a per-course and per-month breakdown. Set
    ``course_name`` (partial ok) to focus on one course, or ``month`` (e.g.
    "2026-03" or "March 2026") to focus on one month.
    """
    try:
        with log_tool_call("get_attendance_summary",
                           {"course_name": course_name, "month": month}):
            records = api_client.fetch_attendance()
            s = agg.attendance_summary(records, course=course_name, month=month)
            if s["rate"] is None:
                return f"No attendance records found for {s['scope']}."

            out = [
                f"Attendance ({s['scope']}): {s['rate']}% "
                f"— present at {s['present']}/{s['sessions']} sessions "
                f"({s['absent']} absence(s))."
            ]
            for block in s.get("by_course", []):
                if block["absent"]:  # only flag courses with absences
                    out.append(f"  · {block['course']}: {block['rate']}% "
                               f"({block['absent']} absence(s)/{block['sessions']})")
            if not course_name and not month and s.get("by_month"):
                out.append("By month: " + ", ".join(
                    f"{b['month']} {b['rate']}%" for b in s["by_month"]))
            return "\n".join(out)
    except AlbertAPIError as exc:
        return f"⚠️ Could not load attendance: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while computing attendance: {exc}"


# --------------------------------------------------------------------------- #
# Tool 4 — grades
# --------------------------------------------------------------------------- #
@tool
def get_grades_summary(group_by: str = "overall", course_name: str | None = None) -> str:
    """Compute the student's average grade across exams currently on record.

    Use for "what's my average", "my average per teaching unit", "my grade in
    <course>". ``group_by`` is one of "overall", "teaching_unit" or "course".
    Set ``course_name`` (partial ok) to focus on one course. Grades are on the
    0-100 scale (the /20 equivalent is shown too). Averages are weighted by each
    exam's coefficient; only graded exams that count toward the average are used.
    """
    try:
        with log_tool_call("get_grades_summary",
                           {"group_by": group_by, "course_name": course_name}):
            grades = api_client.fetch_grades()
            courses = api_client.fetch_courses()
            s = agg.grades_summary(grades, courses, course=course_name)
            if s["overall"] is None:
                return f"No graded exams found{f' for {course_name}' if course_name else ''}."

            scope = f" in courses matching '{course_name}'" if course_name else ""
            out = [
                f"Overall average{scope}: {_fr(s['overall'])} "
                f"(across {s['n_courses']} course(s), {s['n_exams']} exam(s))."
            ]
            if s["overall_ects_weighted"] is not None:
                out.append(f"ECTS-weighted average: {_fr(s['overall_ects_weighted'])}.")

            if group_by == "teaching_unit" or not course_name:
                out.append("By teaching unit:")
                out += [f"  · {u['unit']}: {_fr(u['average'])} ({u['courses']} course(s))"
                        for u in s["by_unit"]]
            if group_by == "course" or course_name:
                out.append("By course:")
                out += [f"  · {c['course']} ({c['unit']}): {_fr(c['average'])}"
                        for c in s["by_course"]]
            return "\n".join(out)
    except AlbertAPIError as exc:
        return f"⚠️ Could not load grades: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while computing grades: {exc}"


# --------------------------------------------------------------------------- #
# Tool 5 — RAG over the official documents
# --------------------------------------------------------------------------- #
@tool
def search_school_documents(query: str) -> str:
    """Search the student's official PDF documents (certificate + transcripts).

    Use this for anything in the paper record the live API does not cover: the
    enrollment certificate, and the YEARLY/SEMESTRIAL transcripts with their
    official /20 grades, letter grades, ECTS and PASS/FAIL situations per
    teaching unit. Prefer this over the grades tool when the user explicitly
    asks about a transcript, a past year, the /20 scale, or their certificate.
    Returns the most relevant excerpts with their source filename.
    """
    try:
        with log_tool_call("search_school_documents", {"query": query}):
            hits = rag.search_documents(query)
            if not hits:
                return "No relevant passages found in the documents."
            blocks = []
            for h in hits:
                page = f", page {h['page'] + 1}" if isinstance(h["page"], int) else ""
                blocks.append(f"[{h['source']}{page}]\n{h['content']}")
            return "\n\n---\n\n".join(blocks)
    except FileNotFoundError as exc:
        return f"⚠️ Document store unavailable: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while searching documents: {exc}"


# --------------------------------------------------------------------------- #
# Tool 6 — calculator (safe arithmetic)
# --------------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS: dict[str, Callable] = {
    "round": round, "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
}


def _safe_eval(node: ast.AST):
    """Evaluate a parsed arithmetic expression, rejecting anything unsafe."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_safe_eval(e) for e in node.elts]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("unsupported or unsafe expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression and return the numeric result.

    Use this whenever a question needs a computation the data tools do not return
    directly — weighted averages of grades the user gives you, "what do I need on
    the final to reach 14/20", unit conversions, percentages. Supports + - * / //
    % ** parentheses and the functions round, abs, min, max, sum, len. Example:
    ``round((85*0.4 + 78*0.6), 2)``.
    """
    try:
        with log_tool_call("calculator", {"expression": expression}):
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            return f"{expression} = {result}"
    except Exception as exc:  # noqa: BLE001
        return (f"⚠️ Could not evaluate '{expression}': {exc}. "
                "Only arithmetic with numbers is allowed.")


# --------------------------------------------------------------------------- #
# Tool 7 — Gmail search (read; ungated)
# --------------------------------------------------------------------------- #
@tool
def search_emails(query: str = "", max_results: int = 5) -> str:
    """Search the student's Gmail inbox and list matching messages (newest first).

    Use for "any email from my professor?", "did I get the exam schedule?",
    "unread mail about the project". ``query`` uses Gmail search syntax —
    ``from:`` ``subject:`` ``is:unread`` ``newer_than:7d`` ``has:attachment`` —
    and can be combined (e.g. ``from:dupont subject:exam newer_than:14d``). Leave
    it empty for the most recent messages. Returns each message's sender, subject,
    date, a snippet, and an id you can pass to ``read_email`` for the full text.
    """
    try:
        with log_tool_call("search_emails", {"query": query, "max_results": max_results}):
            messages = google_client.search_messages(query, max_results)
            if not messages:
                scope = f" matching '{query}'" if query else ""
                return f"No emails found{scope}."
            out = [f"{len(messages)} email(s)"
                   + (f" matching '{query}'" if query else " (most recent)") + ":"]
            for m in messages:
                out.append(
                    f"\n• From: {m['from']}\n  Subject: {m['subject']}\n"
                    f"  Date: {m['date']}\n  Preview: {m['snippet']}\n  id: {m['id']}"
                )
            return "\n".join(out)
    except GoogleError as exc:
        return f"⚠️ Could not search your email: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while searching your email: {exc}"


# --------------------------------------------------------------------------- #
# Tool 8 — read one email in full (read; ungated)
# --------------------------------------------------------------------------- #
@tool
def read_email(message_id: str) -> str:
    """Read the full body of ONE email, identified by the id from ``search_emails``.

    Use when a snippet is not enough and the student wants the actual contents of
    a specific message ("what exactly does that email say?"). ``message_id`` must
    be an id returned by ``search_emails``. Returns the sender, recipient, subject,
    date and the full (plain-text) body.
    """
    try:
        with log_tool_call("read_email", {"message_id": message_id}):
            m = google_client.get_message(message_id)
            return (
                f"From: {m['from']}\nTo: {m['to']}\nSubject: {m['subject']}\n"
                f"Date: {m['date']}\n\n{m['body']}"
            )
    except GoogleError as exc:
        return f"⚠️ Could not read that email: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while reading the email: {exc}"


# --------------------------------------------------------------------------- #
# Tool 9 — Google Calendar lookup (read; ungated)
# --------------------------------------------------------------------------- #
@tool
def get_calendar_events(
    start_date: str | None = None,
    end_date: str | None = None,
    query: str | None = None,
    max_results: int = 10,
) -> str:
    """List the student's Google Calendar events in a date window.

    Use for "what's on my schedule this week", "any classes tomorrow", "when is
    my next exam". ``start_date``/``end_date`` are ISO dates ("2026-06-12") or
    datetimes; resolve relative dates from today's date (given in your system
    prompt). Omit them for the next week. ``query`` filters by text (e.g. "exam").
    Returns each event's title, start, end and location.
    """
    try:
        with log_tool_call("get_calendar_events",
                           {"start_date": start_date, "end_date": end_date,
                            "query": query}):
            events = google_client.list_events(start_date, end_date, query, max_results)
            if not events:
                scope = f" matching '{query}'" if query else ""
                return f"No calendar events found{scope} in that period."
            out = [f"{len(events)} event(s):"]
            for e in events:
                loc = f" @ {e['location']}" if e["location"] else ""
                out.append(f"• {e['start']} → {e['end']}: {e['summary']}{loc}")
            return "\n".join(out)
    except GoogleError as exc:
        return f"⚠️ Could not read your calendar: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while reading your calendar: {exc}"


# --------------------------------------------------------------------------- #
# Tool 10 — draft an email (write; UNGATED — a draft is saved, never sent)
# --------------------------------------------------------------------------- #
@tool
def draft_email(to: str, subject: str, body: str) -> str:
    """Compose an email and save it as a Gmail DRAFT (it is NOT sent).

    Use this to prepare a message for the student to review ("write an email to my
    professor about my absence", "draft a reply"). The draft lands in their Drafts
    folder and nothing leaves the mailbox, so this needs no approval. When the
    student is happy and asks to send, use ``send_email`` (which does ask for
    confirmation). ``to`` is the recipient address. Returns the draft's recipient
    and subject.
    """
    try:
        with log_tool_call("draft_email", {"to": to, "subject": subject}):
            d = google_client.create_draft(to, subject, body)
            return (f"Draft saved (not sent) — to: {d['to']}, subject: "
                    f"\"{d['subject']}\". Ask me to send it when you're ready.")
    except GoogleError as exc:
        return f"⚠️ Could not create the draft: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while drafting the email: {exc}"


# --------------------------------------------------------------------------- #
# Tool 11 — send an email (SENSITIVE: gated by human approval)
# --------------------------------------------------------------------------- #
@tool
def send_email(to: str, subject: str, body: str) -> str:
    """SEND an email from the student's account (the app gets their approval).

    Call this whenever the student asks to send a message ("email my professor
    that I'll be absent", "send it") — do NOT ask for confirmation yourself first.
    The graph automatically pauses and asks the student to approve before the email
    is actually sent, so calling this tool is how you request that approval; if
    they decline, do not retry. ``to`` is the recipient address. Returns a
    confirmation once the email is sent.
    """
    try:
        with log_tool_call("send_email", {"to": to, "subject": subject}):
            m = google_client.send_message(to, subject, body)
            return f"✅ Email sent to {m['to']} (subject \"{m['subject']}\")."
    except GoogleError as exc:
        return f"⚠️ Could not send the email: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while sending the email: {exc}"


# --------------------------------------------------------------------------- #
# Tool 12 — create a calendar event (SENSITIVE: gated by human approval)
# --------------------------------------------------------------------------- #
@tool
def create_calendar_event(
    summary: str,
    start: str,
    end: str | None = None,
    description: str | None = None,
    location: str | None = None,
) -> str:
    """Create an event in the student's Google Calendar (the app gets approval).

    Call this whenever the student asks to add or schedule something ("add X to my
    calendar", "schedule a revision session tomorrow 2-4pm") — do NOT ask for
    confirmation yourself first. ``start``/``end`` are ISO dates ("2026-06-20") or
    datetimes ("2026-06-20T14:00"); resolve relative dates from today's date in
    your system prompt. If ``end`` is omitted the event lasts one hour (or one full
    day for an all-day event). The graph automatically pauses for the student's
    approval before the event is created. Returns the created event.
    """
    try:
        with log_tool_call("create_calendar_event",
                           {"summary": summary, "start": start, "end": end}):
            e = google_client.create_event(summary, start, end, description, location)
            link = f" ({e['link']})" if e.get("link") else ""
            return (f"✅ Event created: \"{e['summary']}\", "
                    f"{e['start']} → {e['end']}.{link}")
    except GoogleError as exc:
        return f"⚠️ Could not create the event: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"⚠️ Unexpected error while creating the event: {exc}"


# The agent's full toolbox, in a stable order.
TOOLS = [
    list_my_program,
    get_course_details,
    get_attendance_summary,
    get_grades_summary,
    search_school_documents,
    calculator,
    search_emails,
    read_email,
    get_calendar_events,
    draft_email,
    send_email,
    create_calendar_event,
]

# Tools whose execution the graph gates behind explicit human approval, because
# they perform an outward-facing WRITE on the student's behalf — sending an email
# or creating a calendar event. The conditional edge routes to the
# ``human_approval`` node whenever the model requests one of these. Reads and
# drafting an email (saved, never sent) are NOT gated.
SENSITIVE_TOOLS = {"send_email", "create_calendar_event"}
