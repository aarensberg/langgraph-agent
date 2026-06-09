"""Test cases and success criteria for the agent.

Each case asserts three independent things, so a failure points at *where* the
agent broke:

- **route**     : which way the conditional edge sent the turn
                  ("tools", "end", or "fallback").
- **tools**     : the tool(s) that must have been called (subset check) — this
                  is the routing-correctness signal.
- **content**   : regex patterns the final answer must contain — this is the
                  end-to-end faithfulness signal (did the LLM relay the numbers
                  the deterministic layer computed?).

The numeric expectations (attendance rate, weakest course, overall average) are
computed from the aggregation layer at run time, so the suite stays correct as
the live data changes instead of hard-coding today's values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from albert_agent import aggregations as agg
from albert_agent import api_client


@dataclass
class Case:
    id: str
    question: str
    expect_route: str            # "tools" | "end" | "fallback" | "approval"
    expect_tools: set[str]       # tools that must appear in the trajectory
    must_include: list[str]      # regex patterns (case-insensitive) the answer must match
    max_iter_override: int | None = None  # force the safety branch when set
    require_approval: bool = True  # HITL gate state for this case
    needs_google: bool = False   # skip when no Google account is connected
    note: str = ""
    tags: list[str] = field(default_factory=list)


def _ground_truth() -> dict:
    """Compute reference numbers straight from the deterministic tools layer."""
    attendance = agg.attendance_summary(api_client.fetch_attendance())
    worst = min(attendance["by_course"], key=lambda b: b["rate"])
    grades = agg.grades_summary(api_client.fetch_grades(), api_client.fetch_courses())
    return {
        "attendance_rate": int(round(attendance["rate"])),       # e.g. 97
        "worst_course_kw": worst["course"].strip().split()[-1],  # e.g. "calculus"
        "overall_100": float(grades["overall"]),                 # e.g. 78.8 (/100)
    }


def _grade_pattern(overall_100: float) -> str:
    """A lenient regex matching the overall average however the model phrases it.

    The grades tool relays the figure on BOTH scales, e.g. "78.8/100 (15.8/20)".
    A faithful answer may quote either scale, round it, swap "." for "," or write
    "/ 20" with spaces — so we accept the integer part of either scale (±1 for
    rounding) or an explicit "/100"/"/20" marker (spaces allowed). This keeps the
    check anchored on the right magnitude without penalising a correct phrasing.
    """
    o100 = int(overall_100)               # 78  (leading int of 78.8)
    o20 = overall_100 / 5                  # 15.76  (the same grade on the /20 scale)
    bands = sorted({o100, o100 + 1, int(o20), int(o20) + 1})  # {78, 79, 15, 16}
    numbers = "|".join(str(n) for n in bands)
    return rf"\b(?:{numbers})\b|/\s*20|/\s*100"


def build_cases() -> list[Case]:
    gt = _ground_truth()
    rate = str(gt["attendance_rate"])
    grade_pat = _grade_pattern(gt["overall_100"])

    return [
        Case(
            id="greeting",
            question="Bonjour ! Qu'est-ce que tu peux faire pour moi ?",
            expect_route="end",
            expect_tools=set(),
            must_include=[r"cours|notes?|présence|programme"],
            note="Small talk must NOT trigger a tool — exercises the 'end' branch.",
            tags=["routing"],
        ),
        Case(
            id="courses_semester",
            question="Quels cours est-ce que je suis ce semestre ?",
            expect_route="tools",
            expect_tools={"list_my_program"},
            must_include=[r"generative AI|Deep Learning|stochastic"],
            tags=["program"],
        ),
        Case(
            id="course_assessment",
            question="Quelles sont les modalités d'évaluation du cours d'IA générative ?",
            expect_route="tools",
            expect_tools={"get_course_details"},
            must_include=[r"30", r"project|projet|PROJ|oral"],
            tags=["course"],
        ),
        Case(
            id="course_documents",
            question="Quels documents sont disponibles pour le cours d'introduction à l'IA générative ?",
            expect_route="tools",
            expect_tools={"get_course_details"},
            must_include=[r"RAG|slides|lecture|\.pdf|notes"],
            tags=["course"],
        ),
        Case(
            id="attendance_overall",
            question="Quel est mon taux de présence global ce semestre ?",
            expect_route="tools",
            expect_tools={"get_attendance_summary"},
            must_include=[rf"{rate}\s*%|{rate}[.,]"],
            note="Answer must relay the rate computed by the aggregation layer.",
            tags=["attendance"],
        ),
        Case(
            id="attendance_worst",
            question="Dans quel cours suis-je le moins assidu ?",
            expect_route="tools",
            expect_tools={"get_attendance_summary"},
            must_include=[gt["worst_course_kw"]],
            tags=["attendance"],
        ),
        Case(
            id="grades_overall",
            question="Quelle est ma moyenne générale actuelle ?",
            expect_route="tools",
            expect_tools={"get_grades_summary"},
            must_include=[grade_pat],
            tags=["grades"],
        ),
        Case(
            id="grades_by_unit",
            question="Donne-moi ma moyenne par Teaching Unit.",
            expect_route="tools",
            expect_tools={"get_grades_summary"},
            must_include=[r"Data", r"Business", r"Math"],
            tags=["grades"],
        ),
        Case(
            id="transcript_rag",
            question="D'après mon relevé de notes officiel, quelles unités d'enseignement ai-je validées en 2ème année ?",
            expect_route="tools",
            expect_tools={"search_school_documents"},
            must_include=[r"/20|PASS|MATHEMATICS|DATA|BUSINESS|HUMANITIES"],
            note="Must reach for the PDF transcripts (RAG), not the live /100 API.",
            tags=["rag"],
        ),
        Case(
            id="calculator_hypothetical",
            question=("Le cours d'IA générative est noté 30% CC, 30% TP, 40% projet. "
                      "Si j'obtiens 70 au CC, 75 au TP et 85 au projet, quelle est ma note finale ?"),
            expect_route="tools",
            expect_tools={"calculator"},
            must_include=[r"77[.,]5|77\.5|77,5"],  # 0.3*70 + 0.3*75 + 0.4*85 = 77.5
            note="A hypothetical the data tools can't answer — needs the calculator.",
            tags=["calculator"],
        ),
        Case(
            id="email_approval_gate",
            question="Ai-je reçu un email de mon professeur à propos de l'examen ?",
            expect_route="approval",
            expect_tools={"search_emails"},
            must_include=[],
            note="Reading mail is private: with the gate ON, the conditional edge "
                 "must divert to the human_approval interrupt BEFORE any Gmail "
                 "call runs — so this proves the HITL branch is live and needs no "
                 "Google connection (the pause happens first).",
            tags=["email", "hitl", "routing"],
        ),
        Case(
            id="email_search",
            question="Cherche mes emails non lus de cette semaine.",
            expect_route="tools",
            expect_tools={"search_emails"},
            must_include=[],
            require_approval=False,  # gate off -> the Gmail tool runs without pausing
            needs_google=True,
            note="Gate off: routes straight to the Gmail tool. Skipped when no "
                 "Google account is connected.",
            tags=["email", "google"],
        ),
        Case(
            id="calendar_week",
            question="Qu'est-ce que j'ai de prévu dans mon agenda cette semaine ?",
            expect_route="tools",
            expect_tools={"get_calendar_events"},
            must_include=[],
            needs_google=True,
            note="Calendar is never gated; routes directly to the calendar tool. "
                 "Skipped when no Google account is connected.",
            tags=["calendar", "google"],
        ),
        Case(
            id="safety_fallback",
            question="Fais-moi un bilan complet: programme, présence, notes et documents.",
            expect_route="fallback",
            expect_tools=set(),
            must_include=[r"limit|narrow|rephrase"],
            max_iter_override=1,  # force the budget to bite on the first tool request
            note="With the loop budget set to 1, the conditional edge must divert "
                 "to the fallback node — proves the safety branch is live, not dead code.",
            tags=["routing", "safety"],
        ),
    ]
