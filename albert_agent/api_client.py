"""Thin authenticated wrapper over the Albert intranet API.

One private ``_get`` does auth, timeout and error handling; every public
``fetch_*`` is a one-liner that knows its endpoint and — crucially — which of
the two student identifiers that endpoint expects. Structural data that does
not change within a session (profile, catalog, course list, syllabus) is
memoised; the "live" lists a student re-checks (attendance, grades, documents)
are always fetched fresh.

All functions raise :class:`AlbertAPIError` on any failure so callers (the
tools) can convert it into a single graceful message.
"""

from __future__ import annotations

from functools import lru_cache

import requests

from . import config
from .observability import log_event


class AlbertAPIError(RuntimeError):
    """Any failure talking to the Albert intranet API (HTTP, network, JSON)."""


def _get(path: str) -> dict | list:
    """GET ``BASE_URL + path`` with bearer auth, returning parsed JSON.

    Raises :class:`AlbertAPIError` on a network problem, a non-2xx status, or a
    body that is not valid JSON — the three ways this call can fail.
    """
    config.require("ALBERT_PERSONAL_TOKEN")
    url = config.API_BASE_URL + path
    headers = {"Authorization": f"Bearer {config.ALBERT_PERSONAL_TOKEN}"}
    try:
        response = requests.get(url, headers=headers, timeout=config.REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        log_event("api_error", path=path, kind="network", error=str(exc))
        raise AlbertAPIError(f"Network error calling {path}: {exc}") from exc

    if not response.ok:
        log_event("api_error", path=path, kind="http", status=response.status_code)
        raise AlbertAPIError(
            f"Albert API returned HTTP {response.status_code} for {path}."
        )
    try:
        return response.json()
    except ValueError as exc:
        log_event("api_error", path=path, kind="json")
        raise AlbertAPIError(f"Albert API sent a non-JSON response for {path}.") from exc


# --------------------------------------------------------------------------- #
# Stable structural data (memoised for the process lifetime)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def fetch_profile() -> dict:
    """The signed-in student's profile and program enrolment."""
    return _get("/user/user-profile")


@lru_cache(maxsize=1)
def fetch_programs() -> list:
    """The public catalog of programs (used to resolve a program_version_id)."""
    return _get("/public/programs")


@lru_cache(maxsize=1)
def fetch_courses() -> list:
    """Every course-module-instance the student is enrolled in (all terms).

    Endpoint keyed by ALBERT_USER_ID (NOT the student_id).
    """
    config.require("ALBERT_USER_ID")
    return _get(f"/student/{config.ALBERT_USER_ID}/course-module-instances")


@lru_cache(maxsize=64)
def fetch_course_instance(course_id: int) -> dict:
    """Details of one course-module-instance by its numeric ``id``."""
    return _get(f"/course/course-module-instance/by-id/{course_id}")


@lru_cache(maxsize=64)
def fetch_syllabus(program_version_id: str, course_module_version_id: int) -> dict:
    """The published syllabus (assessment, topics, learning outcomes)."""
    return _get(
        f"/public/program-versions/{program_version_id}"
        f"/course-module-versions/{course_module_version_id}"
    )


# --------------------------------------------------------------------------- #
# Live data (always fetched fresh)
# --------------------------------------------------------------------------- #
def fetch_attendance() -> list:
    """Per-session attendance records. Keyed by ALBERT_USER_ID."""
    config.require("ALBERT_USER_ID")
    return _get(f"/attendance/user/{config.ALBERT_USER_ID}")


def fetch_grades() -> list:
    """Per-exam grade records. Keyed by ALBERT_STUDENT_ID (NOT the user_id)."""
    config.require("ALBERT_STUDENT_ID")
    return _get(f"/student-exam-grade/student/{config.ALBERT_STUDENT_ID}")


def fetch_course_documents(course_id: int) -> list:
    """All academic documents for a course, transparently de-paginated."""
    documents: list = []
    page = 1
    while True:
        payload = _get(
            f"/course/academic-documents/by-course-module-instance/{course_id}"
            f"?page={page}&limit=50&include_archived=false&include_sessions=true"
        )
        # Endpoint returns {documents: [...], total, page, total_pages}.
        if isinstance(payload, dict):
            documents.extend(payload.get("documents", []))
            if page >= int(payload.get("total_pages", 1)):
                break
            page += 1
        else:  # unexpected shape — return whatever we have
            break
    return documents
