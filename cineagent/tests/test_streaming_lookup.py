"""
Standalone test for Tool 3: streaming_lookup.
Hits the REAL TMDB watch-providers API.

Note: streaming availability is LIVE and REGIONAL — results change over
time and by country. Tests assert on structure and status, not on
specific providers (those drift). We verify the tool behaves correctly,
not that a specific film is on a specific service today.
"""

from cineagent.tools.streaming_lookup import streaming_lookup


def show(label, result):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print("-" * 60)
    print(f"Status: {result['status']}")
    if result["status"] == "ok":
        print(f"  Title:     {result['title']} ({result['year']})")
        print(f"  Country:   {result['country']}")
        print(f"  Streaming: {result['streaming']}")
        print(f"  Rent:      {result['rent']}")
        print(f"  Buy:       {result['buy']}")
        print(f"  Link:      {result['tmdb_watch_link']}")
    else:
        print(f"  Message: {result.get('message', '(none)')}")


if __name__ == "__main__":
    # 1. Popular recent film — likely available somewhere in IN
    show("Parasite (IN)", streaming_lookup("Parasite"))

    # 2. Same film, different country — availability differs by region
    show("Parasite (US)", streaming_lookup("Parasite", country="US"))

    # 3. Very popular film — almost certainly available
    show("Oppenheimer (IN)", streaming_lookup("Oppenheimer"))

    # 4. Old/obscure film — may exist but have no_availability
    show("The Cabinet of Dr. Caligari (IN)", streaming_lookup("The Cabinet of Dr. Caligari"))

    # 5. Nonsense title — should be no_results
    show("Nonsense title", streaming_lookup("qwertyuiop asdfghjkl zxcvbnm"))

    # 6. Empty title — should be error
    show("Empty title", streaming_lookup(""))

    print(f"\n{'=' * 60}")
    print("All tests complete.")