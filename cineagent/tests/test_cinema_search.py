"""
Test for Tool 1: search_cinema_knowledge.

These are NOT unit tests with mocks. They hit the real ChromaDB on disk.
We want to verify the tool behaves correctly against the actual corpus,
because that's the only thing the agent will ever see in production.
"""

from cineagent.tools.cinema_search import search_cinema_knowledge


def test_known_topic():
    """A query we KNOW exists in the corpus should return results."""
    print("\n--- Test 1: Known topic ---")
    result = search_cinema_knowledge("French New Wave directors")
    print(f"Status: {result['status']}")
    if result["status"] == "ok":
        print(f"Got {len(result['results'])} results")
        for i, r in enumerate(result["results"][:2], 1):
            print(f"  {i}. [{r['source']}] dist={r['distance']}")
            print(f"     {r['text'][:120]}...")
    else:
        print(f"Message: {result['message']}")
    print()


def test_out_of_domain():
    """A query OUTSIDE the cinema corpus should return no_results."""
    print("--- Test 2: Out of domain (should return no_results) ---")
    result = search_cinema_knowledge("how to fix a leaking kitchen faucet")
    print(f"Status: {result['status']}")
    if result["status"] == "no_results":
        print(f"  ✓ Correctly returned no_results")
        print(f"  Message: {result['message']}")
    else:
        print(f"  ⚠ Expected no_results but got: {result['status']}")
        if result["status"] == "ok":
            print(f"  Top result distance: {result['results'][0]['distance']}")
            print(f"  This means the threshold may need tuning.")
    print()


def test_empty_query():
    """An empty query should return an error, not crash."""
    print("--- Test 3: Empty query (should return error) ---")
    result = search_cinema_knowledge("")
    print(f"Status: {result['status']}")
    print(f"Message: {result.get('message', '(none)')}")
    print()


def test_specific_film():
    """A query about a specific well-known film."""
    print("--- Test 4: Specific film query ---")
    result = search_cinema_knowledge("Akira Kurosawa Seven Samurai")
    print(f"Status: {result['status']}")
    if result["status"] == "ok":
        print(f"Got {len(result['results'])} results")
        sources = set(r["source"] for r in result["results"])
        print(f"Sources hit: {sources}")
    print()


if __name__ == "__main__":
    test_known_topic()
    test_out_of_domain()
    test_empty_query()
    test_specific_film()
    print("All tests complete.")