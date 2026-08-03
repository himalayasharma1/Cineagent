"""
Test the dispatcher in isolation — no model involved.
Verifies valid calls route correctly and invalid calls return
structured errors instead of raising.
"""

from cineagent.agent.dispatcher import dispatch_tool


def show(label, result):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print("-" * 60)
    print(f"Status: {result.get('status')}")
    if result.get("status") == "error":
        print(f"  error_type: {result.get('error_type')}")
        print(f"  message: {result.get('message')}")
    else:
        # Just confirm it returned something tool-shaped
        print(f"  keys: {list(result.keys())}")


if __name__ == "__main__":
    # --- Valid calls: should route to the real tools ---
    show("Valid: search_cinema_knowledge",
         dispatch_tool("search_cinema_knowledge", {"query": "French New Wave"}))

    show("Valid: get_film_details",
         dispatch_tool("get_film_details", {"title": "Parasite"}))

    show("Valid: streaming_lookup with country",
         dispatch_tool("streaming_lookup", {"title": "Oppenheimer", "country": "IN"}))

    # --- Invalid calls: should return structured errors, NOT raise ---
    show("Invalid: unknown tool",
         dispatch_tool("get_weather", {"city": "Tokyo"}))

    show("Invalid: missing required arg",
         dispatch_tool("get_film_details", {}))

    show("Invalid: arguments not a dict",
         dispatch_tool("get_film_details", "Parasite"))

    # --- Edge: hallucinated extra arg should be filtered, call succeeds ---
    show("Edge: extra hallucinated arg (should be filtered)",
         dispatch_tool("get_film_details", {"title": "The Godfather", "director": "hallucinated"}))

    print(f"\n{'=' * 60}")
    print("All dispatcher tests complete.")