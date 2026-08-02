"""
Standalone test for Tool 2: get_film_details.

Hits the REAL TMDB API. Verifies:
  - A well-known film returns correctly-shaped details
  - The year disambiguation parameter works
  - An ambiguous title surfaces alternatives
  - A nonsense title returns no_results (not a crash)
  - Empty input returns error
"""

from cineagent.tools.film_details import get_film_details
import json


def show(label, result):
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print("-" * 60)
    print(f"Status: {result['status']}")
    if result["status"] == "ok":
        film = result["film"]
        print(f"  Title:     {film['title']} ({film['release_year']})")
        print(f"  Director:  {film['director']}")
        print(f"  Runtime:   {film['runtime_minutes']} min")
        print(f"  Genres:    {film['genres']}")
        print(f"  Rating:    {film['tmdb_rating']}")
        print(f"  Language:  {film['original_language']}")
        print(f"  Cast:      {film['cast']}")
        print(f"  Overview:  {film['overview'][:100]}..." if film['overview'] else "  Overview:  (none)")
        if result["other_matches"]:
            print(f"  Other matches: {result['other_matches']}")
    else:
        print(f"  Message: {result.get('message', '(none)')}")


if __name__ == "__main__":
    # 1. Well-known, unambiguous film
    show("Parasite (well-known film)", get_film_details("Parasite"))

    # 2. Well-known film with director we can verify
    show("The Godfather", get_film_details("The Godfather"))

    # 3. Ambiguous title WITHOUT year — should return top match + alternatives
    show("Drive (ambiguous, no year)", get_film_details("Drive"))

    # 4. Ambiguous title WITH year — should disambiguate to 2011 Refn film
    show("Drive (with year=2011)", get_film_details("Drive", year=2011))

    # 5. Indian film — tests non-Hollywood coverage
    show("Sholay (Indian classic)", get_film_details("Sholay"))

    # 6. Nonsense title — should return no_results, NOT crash
    show("Nonsense title", get_film_details("qwertyuiop asdfghjkl zxcvbnm"))

    # 7. Empty input — should return error
    show("Empty title", get_film_details(""))

    print(f"\n{'=' * 60}")
    print("All tests complete.")