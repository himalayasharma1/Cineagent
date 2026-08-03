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

# Maps a case's "tool" field to the function that executes it.
# Cases without a "tool" field default to Tool 1 (backwards compatible).

from collections import defaultdict

from cineagent.tools.cinema_search import search_cinema_knowledge
from cineagent.tests.eval_cases import EVAL_CASES
from cineagent.tools.film_details import get_film_details   # NEW
from cineagent.tools.streaming_lookup import streaming_lookup   # NEW

# ---------------------------------------------------------------
# Assertion checkers.
# Each returns (passed: bool, detail: str). detail explains a failure.
# ---------------------------------------------------------------


TOOL_REGISTRY = {
    "search_cinema_knowledge": search_cinema_knowledge,
    "get_film_details": get_film_details,
    "streaming_lookup": streaming_lookup,
    "streaming_no_avail_probe": _streaming_no_availability_probe, 
}


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
def _check_film_field_contains(result: dict, field_expectations: dict):
    """
    For Tool 2 (get_film_details): check that fields inside result["film"]
    contain expected substrings. Value-drift-safe: use for stable fields
    only (director, title, original_language), never ratings/vote counts.

    field_expectations: {"director": "Bong", "original_language": "ko"}
    """
    film = result.get("film")
    if film is None:
        return False, f"film_field_contains: no 'film' in result (status={result.get('status')})"

    failures = []
    for field, expected_substring in field_expectations.items():
        actual = film.get(field)
        actual_str = str(actual) if actual is not None else ""
        if expected_substring.lower() not in actual_str.lower():
            failures.append(f"{field}: expected substring '{expected_substring}', got '{actual}'")

    if failures:
        return False, "film_field_contains: " + "; ".join(failures)
    return True, ""


def _check_film_year(result: dict, expected_year):
    """Check the film's release_year matches (stable field)."""
    film = result.get("film")
    if film is None:
        return False, f"film_year: no 'film' in result (status={result.get('status')})"
    actual = film.get("release_year")
    if str(actual) == str(expected_year):
        return True, ""
    return False, f"film_year: expected {expected_year}, got {actual}"

def _check_top_field_contains(result: dict, field_expectations: dict):
    """
    Check that top-level result fields contain expected substrings.
    Used for Tool 3 (streaming_lookup), whose fields (title, country)
    sit at the top level rather than nested. Drift-safe: use for stable
    fields (title, year, country) only, never provider lists.
    """
    failures = []
    for field, expected_substring in field_expectations.items():
        actual = result.get(field)
        actual_str = str(actual) if actual is not None else ""
        if expected_substring.lower() not in actual_str.lower():
            failures.append(f"{field}: expected substring '{expected_substring}', got '{actual}'")
    if failures:
        return False, "top_field_contains: " + "; ".join(failures)
    return True, ""

# Map assertion keys to their checker functions.
ASSERTION_CHECKERS = {
    "status": _check_status,
    "min_results": _check_min_results,
    "source_contains_any": _check_source_contains_any,
    "source_contains_all": _check_source_contains_all,
    "film_field_contains": _check_film_field_contains,
    "film_year": _check_film_year,
    "top_field_contains": _check_top_field_contains,   # NEW
}

# ---------------------------------------------------------------
# Run a single case.
# ---------------------------------------------------------------
def run_case(case: dict) -> dict:
    """
    Execute one case against the tool named in case["tool"]
    (defaulting to search_cinema_knowledge) and check its assertions.
    """
    query = case["query"]

    # Pick the tool this case targets. Default to Tool 1 for older cases.
    tool_name = case.get("tool", "search_cinema_knowledge")
    system_fn = TOOL_REGISTRY.get(tool_name)
    if system_fn is None:
        return {
            "id": case["id"],
            "category": case["category"],
            "passed": False,
            "failures": [f"UNKNOWN tool: '{tool_name}'"],
            "note": case.get("note", ""),
        }

    # Tool 2 takes an optional `year` kwarg; pass it through if present.
    call_kwargs = {}
    if "year" in case:
        call_kwargs["year"] = case["year"]

    try:
        result = system_fn(query, **call_kwargs)
    except Exception as e:
        return {
            "id": case["id"],
            "category": case["category"],
            "passed": False,
            "failures": [f"CRASHED: {type(e).__name__}: {e}"],
            "note": case.get("note", ""),
        }

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
def run_eval(cases=EVAL_CASES):
    print("=" * 64)
    print(f"CineAgent Eval Run — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total cases: {len(cases)}")
    print("=" * 64)

    start = time.time()
    results = [run_case(c) for c in cases]
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

