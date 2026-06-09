"""Run the evaluation suite and report the agent's success rate.

Usage (from the project root):

    python -m evaluation.run_eval            # run every case
    python -m evaluation.run_eval --tag rag  # only cases tagged "rag"

For each case it streams one fresh conversation through the compiled graph,
records the route taken and the tools called, checks the three success criteria
(route / tools / content), and prints a per-case verdict, the overall success
rate, and a breakdown of failure modes.
"""

from __future__ import annotations

import argparse
import re
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from albert_agent import config, google_client
from albert_agent.graph import get_agent
from evaluation.cases import Case, build_cases


def run_case(app, case: Case) -> dict:
    """Execute one case, returning the observed route, tools, answer and verdict."""
    # The safety case forces a tiny loop budget so the fallback branch triggers.
    original_budget = config.MAX_TOOL_ITERATIONS
    if case.max_iter_override is not None:
        config.MAX_TOOL_ITERATIONS = case.max_iter_override

    run_config = {
        "configurable": {"thread_id": f"eval-{case.id}-{uuid.uuid4()}"},
        "recursion_limit": config.GRAPH_RECURSION_LIMIT,
    }
    tools_called: list[str] = []
    nodes_seen: set[str] = set()
    interrupted = False
    error = None
    try:
        for chunk in app.stream(
            {"messages": [HumanMessage(content=case.question)],
             "require_approval": case.require_approval},
            config=run_config,
            stream_mode="updates",
        ):
            if "__interrupt__" in chunk:  # graph paused at the HITL gate
                interrupted = True
                continue
            for node, update in chunk.items():
                nodes_seen.add(node)
                if node == "agent":
                    msgs = update.get("messages", []) if isinstance(update, dict) else []
                    if msgs:
                        tools_called += [c["name"] for c in getattr(msgs[-1], "tool_calls", [])]
        # When paused for approval there is no final answer yet — that IS the result.
        answer = "" if interrupted else app.get_state(run_config).values["messages"][-1].content
    except Exception as exc:  # noqa: BLE001
        error, answer = str(exc), ""
    finally:
        config.MAX_TOOL_ITERATIONS = original_budget

    route = ("approval" if interrupted
             else "fallback" if "fallback" in nodes_seen
             else "tools" if "tools" in nodes_seen else "end")

    route_ok = route == case.expect_route
    tools_ok = case.expect_tools.issubset(set(tools_called))
    content_ok = all(re.search(p, answer, re.I) for p in case.must_include)
    passed = bool(route_ok and tools_ok and content_ok and not error)

    failures = []
    if error:
        failures.append(f"error: {error[:80]}")
    if not route_ok:
        failures.append(f"route={route}≠{case.expect_route}")
    if not tools_ok:
        missing = case.expect_tools - set(tools_called)
        failures.append(f"missing tools={sorted(missing)}")
    if not content_ok:
        missing = [p for p in case.must_include if not re.search(p, answer, re.I)]
        failures.append(f"answer missing={missing}")

    return {
        "case": case, "passed": passed, "route": route,
        "tools": tools_called, "answer": answer, "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Albert student agent.")
    parser.add_argument("--tag", help="only run cases with this tag")
    args = parser.parse_args()

    cases = build_cases()
    if args.tag:
        cases = [c for c in cases if args.tag in c.tags]

    app = get_agent(checkpointer=InMemorySaver())  # isolated, no on-disk state
    google_ok = google_client.is_connected()
    print(f"Model: {config.MODEL_NAME} | fallback chain: {', '.join(config.FALLBACK_MODELS)}")
    print(f"Google connected: {google_ok}")
    print(f"Running {len(cases)} cases\n" + "=" * 72)

    results = []
    skipped = 0
    for case in cases:
        if case.needs_google and not google_ok:
            skipped += 1
            print(f"[SKIP] {case.id:<22} (needs a connected Google account)")
            continue
        res = run_case(app, case)
        results.append(res)
        status = "PASS" if res["passed"] else "FAIL"
        tools = ",".join(dict.fromkeys(res["tools"])) or "—"
        print(f"[{status}] {case.id:<22} route={res['route']:<8} tools={tools}")
        if not res["passed"]:
            print(f"        ↳ {'; '.join(res['failures'])}")

    passed = sum(r["passed"] for r in results)
    total = len(results)
    print("=" * 72)
    rate = f"{100 * passed / total:.0f}%" if total else "n/a"
    print(f"SUCCESS RATE: {passed}/{total} = {rate}"
          + (f"  ({skipped} skipped: Google not connected)" if skipped else ""))

    failed = [r for r in results if not r["passed"]]
    if failed:
        print("\nFAILURE MODES:")
        for r in failed:
            print(f"  · {r['case'].id}: {'; '.join(r['failures'])}")


if __name__ == "__main__":
    main()
