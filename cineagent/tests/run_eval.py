"""
Eval runner for CineAgent.

Executes every case in EVAL_CASES against the system under test,
checks the assertions in each case's `expect` dict, and reports
category-bucketed pass rates with named failures.

Design notes:
  - The runner is assertion-driven: it checks whichever keys are present
    in `expect`, ignoring the rest. This is what lets the same runner
    handle retrieval cases today and agent-loop cases later without a rewrite.
  - A case passes only if ALL its assertions pass. Any single failed
    assertion fails the case, and the specific failure is recorded.
  - The "system under test" is injected as a function. Today it's Tool 1
    directly (Option A). Later it becomes the agent loop — same runner.
"""

import time
from collections import defaultdict

from cineagent.tools.cinema_search import search_cinema_knowledge
from cineagent.tests.eval_cases import EVAL_CASES


# ---------------------------------------------------------------
# Assertion checkers.
# Each returns (passed: bool, detail: str). detail explains a failure.
# ---------------------------------------------------------------
def _check_status(result: dict, expected: str):
    actual = result.get("status")
    if actual == expected:
        return True, ""
    return False, f"status: expected '{expected}', got '{actual}'"


def _check_min_results(result: dict, minimum: int):
    results = result.get("results", [])
    count = len(results)
    if count >= minimum:
        return True, ""
    return False, f"min_results: expected >={minimum}, got {count}"


def _check_source_contains_any(result: dict, substrings: list):
    results = result.get("results", [])
    sources = [r.get("source", "") for r in results]
    for sub in substrings:
        if any(sub in src for src in sources):
            return True, ""
    return False, (
        f"source_contains_any: none of {substrings} found in sources {sources}"
    )


def _check_source_contains_all(result: dict, substrings: list):
    results = result.get("results", [])
    sources = [r.get("source", "") for r in results]
    missing = [
        sub for sub in substrings
        if not any(sub in src for src in sources)
    ]
    if not missing:
        return True, ""
    return False, (
        f"source_contains_all: missing {missing} from sources {sources}"
    )


# Map assertion keys to their checker functions.
ASSERTION_CHECKERS = {
    "status": _check_status,
    "min_results": _check_min_results,
    "source_contains_any": _check_source_contains_any,
    "source_contains_all": _check_source_contains_all,
}


# ---------------------------------------------------------------
# Run a single case.
# ---------------------------------------------------------------
def run_case(case: dict, system_fn) -> dict:
    """
    Execute one case against system_fn and check all its assertions.

    Returns a result dict:
      {
        "id": ...,
        "category": ...,
        "passed": bool,
        "failures": [list of failure detail strings],
        "note": ...,
      }
    """
    query = case["query"]

    # Run the system under test. Catch crashes so one bad case
    # doesn't kill the whole run — a crash is itself a failure.
    try:
        result = system_fn(query)
    except Exception as e:
        return {
            "id": case["id"],
            "category": case["category"],
            "passed": False,
            "failures": [f"CRASHED: {type(e).__name__}: {e}"],
            "note": case.get("note", ""),
        }

    # Check every assertion present in the case's expect dict.
    failures = []
    for key, expected_value in case["expect"].items():
        checker = ASSERTION_CHECKERS.get(key)
        if checker is None:
            failures.append(f"UNKNOWN assertion key: '{key}'")
            continue
        passed, detail = checker(result, expected_value)
        if not passed:
            failures.append(detail)

    return {
        "id": case["id"],
        "category": case["category"],
        "passed": len(failures) == 0,
        "failures": failures,
        "note": case.get("note", ""),
    }


# ---------------------------------------------------------------
# Run the full suite and report.
# ---------------------------------------------------------------
def run_eval(system_fn=search_cinema_knowledge, cases=EVAL_CASES):
    print("=" * 64)
    print(f"CineAgent Eval Run — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System under test: {system_fn.__name__}")
    print(f"Total cases: {len(cases)}")
    print("=" * 64)

    start = time.time()
    results = [run_case(c, system_fn) for c in cases]
    elapsed = time.time() - start

    # Aggregate
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    # Per-category tallies
    by_category = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        by_category[r["category"]]["total"] += 1
        if r["passed"]:
            by_category[r["category"]]["passed"] += 1

    # ---- Report ----
    print(f"\nOverall: {len(passed)}/{len(results)} passed "
          f"({100 * len(passed) / len(results):.0f}%)\n")

    print("Pass rate by category:")
    for cat in sorted(by_category.keys()):
        p = by_category[cat]["passed"]
        t = by_category[cat]["total"]
        print(f"  {cat:20s} {p}/{t}  ({100 * p / t:.0f}%)")

    if failed:
        print(f"\n{'-' * 64}")
        print(f"FAILURES ({len(failed)}):")
        print("-" * 64)
        for r in failed:
            print(f"\n  [{r['id']}] ({r['category']})")
            for f in r["failures"]:
                print(f"      ✗ {f}")
            if r["note"]:
                print(f"      note: {r['note']}")
    else:
        print("\n✅ All cases passed.")

    print(f"\n{'-' * 64}")
    print(f"Run took {elapsed:.1f}s "
          f"({elapsed / len(results):.2f}s per case).")
    print("=" * 64)

    return results


if __name__ == "__main__":
    run_eval()