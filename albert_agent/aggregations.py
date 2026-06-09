"""Derive rates and averages from raw API records with pandas.

The Albert API returns *records*, never summaries: attendance is one row per
session, grades one row per exam. The numbers a student actually asks for
(attendance rate, course/UE/global averages) are computed here, in Python,
where the rules are explicit and testable — not left to the LLM, which must not
be trusted to do arithmetic over JSON.

Conventions:
- Grades are on a 0-100 scale. Only ``grade_status == "NUMERIC"`` rows with a
  non-null grade and ``counts_in_average`` count; PASS/FAIL rows are ignored.
- A course's final mark is the mean of its exam grades weighted by each exam's
  ``coefficient``; the overall average is the (unweighted) mean of course
  finals, with an ECTS-weighted figure reported alongside.
- Teaching unit comes from the course list (course_module_id -> unit), since it
  is absent from the grade records themselves.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Coarse fallback when a course_module_id is not found in the course list.
_CODE_PREFIX_TO_UNIT = {
    "MAT": "Mathematics",
    "DAT": "Data",
    "BUS": "Business",
    "ECO": "Business",
    "FIN": "Business",
    "MAR": "Business",
    "STR": "Business",
    "HUM": "Humanities and soft skills",
    "INT": "Internship",
}


def _get(d: Any, *keys: str, default=None):
    """Safely walk nested dicts: ``_get(rec, 'a', 'b')`` ~ ``rec['a']['b']``."""
    cur = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


# --------------------------------------------------------------------------- #
# Term handling
# --------------------------------------------------------------------------- #
def current_term(courses: list[dict]) -> tuple[int, str]:
    """Return the latest ``(academic_year, semester)`` present in the courses.

    academic_year is an int like 2526 (2025-26); semesters sort as plain
    strings ("S2" > "S1"), so the max pair is the current term.
    """
    terms = {
        (int(c["academic_year"]), str(c["semester"]))
        for c in courses
        if c.get("academic_year") and c.get("semester")
    }
    if not terms:
        raise ValueError("No (academic_year, semester) found in course list.")
    return max(terms)


def filter_courses_for_term(courses: list[dict], year: int, semester: str) -> list[dict]:
    """Subset of courses belonging to one ``(year, semester)``."""
    return [
        c
        for c in courses
        if int(c.get("academic_year", 0)) == year and str(c.get("semester")) == semester
    ]


def build_unit_map(courses: list[dict]) -> dict[int, str]:
    """Map ``course_module_id -> teaching_unit_instance_name`` from the courses."""
    return {
        int(c["course_module_id"]): c["teaching_unit_instance_name"]
        for c in courses
        if c.get("course_module_id") and c.get("teaching_unit_instance_name")
    }


def _unit_for(cm_id, code: str | None, unit_map: dict[int, str]) -> str:
    """Resolve a teaching unit, preferring the course map over the code prefix."""
    if cm_id is not None and int(cm_id) in unit_map:
        return unit_map[int(cm_id)]
    if code:
        for prefix, unit in _CODE_PREFIX_TO_UNIT.items():
            if str(code).upper().lstrip("0123456789_").startswith(prefix):
                return unit
    return "Unknown"


# --------------------------------------------------------------------------- #
# Attendance
# --------------------------------------------------------------------------- #
def _attendance_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        start = _get(r, "course_instance_session", "session_start_datetime_utc")
        rows.append(
            {
                "course": str(
                    _get(r, "course_module_instance", "course_module_instance_name")
                    or "Unknown"
                ).strip(),
                "present": bool(r.get("present")),
                "start": pd.to_datetime(start, errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _rate_block(df: pd.DataFrame) -> dict:
    n = len(df)
    present = int(df["present"].sum())
    return {
        "rate": round(100 * present / n, 1) if n else None,
        "sessions": n,
        "present": present,
        "absent": n - present,
    }


def attendance_summary(
    records: list[dict],
    course: str | None = None,
    month: str | None = None,
) -> dict:
    """Attendance rate for the requested scope, plus per-course/per-month splits.

    ``course`` is matched case-insensitively as a substring; ``month`` accepts
    anything pandas can parse to a year-month (e.g. "2026-03", "March 2026").
    """
    df = _attendance_frame(records)
    if df.empty:
        return {"scope": "overall", "rate": None, "sessions": 0,
                "present": 0, "absent": 0, "by_course": [], "by_month": []}

    scope = "overall"
    if course:
        mask = df["course"].str.contains(course, case=False, na=False)
        df = df[mask]
        scope = f"course '{course}'"
    if month:
        ts = pd.to_datetime(month, errors="coerce")
        if pd.notna(ts):
            df = df[(df["start"].dt.year == ts.year) & (df["start"].dt.month == ts.month)]
            scope = f"{scope}, {ts.strftime('%B %Y')}" if course else ts.strftime("%B %Y")

    result = {"scope": scope, **_rate_block(df)}
    if df.empty:
        return result

    by_course = (
        df.groupby("course")
        .apply(_rate_block, include_groups=False)
        .to_dict()
    )
    result["by_course"] = [
        {"course": c, **vals} for c, vals in sorted(by_course.items())
    ]
    monthly = df.dropna(subset=["start"]).copy()
    monthly["label"] = monthly["start"].dt.strftime("%Y-%m")
    by_month = monthly.groupby("label").apply(_rate_block, include_groups=False).to_dict()
    result["by_month"] = [{"month": m, **vals} for m, vals in sorted(by_month.items())]
    return result


# --------------------------------------------------------------------------- #
# Grades
# --------------------------------------------------------------------------- #
def _grades_frame(grades: list[dict], unit_map: dict[int, str]) -> pd.DataFrame:
    rows = []
    for g in grades:
        cm = _get(g, "exam_paper", "course_module") or {}
        cm_id = cm.get("course_module_id")
        code = cm.get("course_module_code")
        # Reliable weight is the exam_paper coefficient; fall back to top-level.
        coeff = _get(g, "exam_paper", "coefficient")
        if coeff is None:
            coeff = g.get("coefficient")
        rows.append(
            {
                "course": cm.get("course_module_name") or "Unknown",
                "code": code,
                "unit": _unit_for(cm_id, code, unit_map),
                "grade": g.get("grade"),
                "coeff": float(coeff) if coeff not in (None, 0) else 1.0,
                "ects": _get(g, "enrollment", "source_cmi", "ects"),
                "numeric": g.get("grade_status") == "NUMERIC" and g.get("grade") is not None,
                "counts": bool(g.get("counts_in_average")),
            }
        )
    return pd.DataFrame(rows)


def _weighted(group: pd.DataFrame) -> float:
    """Coefficient-weighted mean of the grades in one course."""
    w = group["coeff"].sum()
    return round(float((group["grade"] * group["coeff"]).sum() / w), 2) if w else float("nan")


def grades_summary(
    grades: list[dict],
    courses: list[dict],
    course: str | None = None,
) -> dict:
    """Compute course / teaching-unit / overall averages on the 0-100 scale.

    Returns a structured dict the grades tool formats for the LLM. When
    ``course`` is given, only matching courses are kept (case-insensitive
    substring), so the same function answers "my average in X".
    """
    unit_map = build_unit_map(courses)
    df = _grades_frame(grades, unit_map)
    df = df[df["numeric"] & df["counts"]]
    if course:
        df = df[df["course"].str.contains(course, case=False, na=False)]
    if df.empty:
        return {"overall": None, "overall_ects_weighted": None,
                "by_course": [], "by_unit": [], "n_courses": 0, "n_exams": 0}

    finals = (
        df.groupby(["course", "unit"])
        .apply(lambda g: pd.Series({"final": _weighted(g), "n_exams": len(g),
                                    "ects": g["ects"].dropna().max()}),
               include_groups=False)
        .reset_index()
    )

    overall = round(float(finals["final"].mean()), 2)
    # ECTS-weight only when EVERY counted course carries a positive credit value;
    # otherwise the figure would silently cover a subset and mislead, so omit it.
    ects = finals["ects"]
    if ects.notna().all() and (ects > 0).all():
        overall_ects = round(float((finals["final"] * ects).sum() / ects.sum()), 2)
    else:
        overall_ects = None
    by_unit = (
        finals.groupby("unit")["final"]
        .agg(["mean", "count"]).reset_index()
        .sort_values("unit")
    )

    return {
        "overall": overall,
        "overall_ects_weighted": overall_ects,
        "n_courses": int(len(finals)),
        "n_exams": int(len(df)),
        "by_course": [
            {"course": r.course, "unit": r.unit, "average": round(r.final, 2),
             "exams": int(r.n_exams)}
            for r in finals.sort_values("final", ascending=False).itertuples()
        ],
        "by_unit": [
            {"unit": r["unit"], "average": round(float(r["mean"]), 2), "courses": int(r["count"])}
            for _, r in by_unit.iterrows()
        ],
    }
