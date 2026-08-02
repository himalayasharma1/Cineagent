"""
Tool 2: get_film_details

Fetches current film metadata from TMDB (The Movie Database).
Given a film title (and optionally a year for disambiguation), returns
a curated set of fields: director, cast, runtime, genres, rating, plot.

Unlike Tool 1 (search_cinema_knowledge), this tool makes an external
network call. Its core design commitments:
  - Loads the API key from .env, never hardcoded.
  - Shapes TMDB's large response down to 9 useful fields.
  - Handles ambiguity via an optional `year` param + an alternatives list.
  - NEVER raises on network/API failure — converts everything to a
    structured {"status": "error"} dict so the agent can adapt.
"""

import os
import requests
from dotenv import load_dotenv

# Load .env once at import time so os.getenv can see TMDB_API_KEY.
load_dotenv()

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Network timeout in seconds. If TMDB doesn't respond in this window,
# we treat it as a failure rather than hanging the agent indefinitely.
REQUEST_TIMEOUT = 8

# How many top-billed cast members to include. A film can credit 100+
# actors; the agent almost never needs beyond the leads. Keeps context lean.
CAST_LIMIT = 5

# How many alternative matches to surface for disambiguation.
MAX_ALTERNATIVES = 4


# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------
def _extract_year(release_date: str):
    """TMDB gives release_date as 'YYYY-MM-DD'. Pull the year, safely."""
    if release_date and len(release_date) >= 4:
        return release_date[:4]
    return None


def _shape_film(details: dict, credits: dict) -> dict:
    """
    Reduce TMDB's large movie + credits payload to our 9 curated fields.
    `details` is the /movie/{id} response; `credits` is /movie/{id}/credits.
    """
    # Director: find the crew member whose job is 'Director'.
    director = None
    for person in credits.get("crew", []):
        if person.get("job") == "Director":
            director = person.get("name")
            break

    # Top-billed cast, capped at CAST_LIMIT.
    cast = [
        member.get("name")
        for member in credits.get("cast", [])[:CAST_LIMIT]
    ]

    # Genres come as a list of {id, name} dicts; we want just names.
    genres = [g.get("name") for g in details.get("genres", [])]

    return {
        "title": details.get("title"),
        "release_year": _extract_year(details.get("release_date")),
        "director": director,
        "runtime_minutes": details.get("runtime"),
        "genres": genres,
        "overview": details.get("overview"),
        "tmdb_rating": details.get("vote_average"),
        "cast": cast,
        "original_language": details.get("original_language"),
    }


# ---------------------------------------------------------------
# The tool itself
# ---------------------------------------------------------------
def get_film_details(title: str, year: int = None) -> dict:
    """
    Fetch curated metadata for a film from TMDB.

    Args:
        title: Film title to search for.
        year:  Optional release year to disambiguate same-title films.

    Returns one of:

      Success:
        {
          "status": "ok",
          "film": { ...9 fields... },
          "other_matches": [ {"title": ..., "year": ...}, ... ]  # may be empty
        }

      Not found (valid query, no TMDB match):
        {"status": "no_results", "message": ...}

      Error (empty input, missing key, or network/API failure):
        {"status": "error", "message": ...}
    """
    # --- Input validation ---
    if not title or not title.strip():
        return {"status": "error", "message": "Title is empty."}

    if not TMDB_API_KEY:
        return {
            "status": "error",
            "message": "TMDB_API_KEY not found. Check your .env file.",
        }

    # --- Step 1: search for the film by title ---
    try:
        search_params = {"api_key": TMDB_API_KEY, "query": title}
        if year:
            search_params["year"] = year

        search_resp = requests.get(
            f"{TMDB_BASE_URL}/search/movie",
            params=search_params,
            timeout=REQUEST_TIMEOUT,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "TMDB request timed out."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"TMDB search failed: {e}"}

    results = search_data.get("results", [])
    if not results:
        return {
            "status": "no_results",
            "message": f"No film found matching '{title}'"
                       + (f" ({year})" if year else "") + ".",
        }

    # TMDB returns results sorted by popularity. Take the top as our match.
    top = results[0]
    film_id = top["id"]

    # Build the alternatives list (other matches) for disambiguation.
    other_matches = [
        {"title": r.get("title"), "year": _extract_year(r.get("release_date"))}
        for r in results[1 : 1 + MAX_ALTERNATIVES]
    ]

    # --- Step 2: fetch full details + credits for the chosen film ---
    try:
        details_resp = requests.get(
            f"{TMDB_BASE_URL}/movie/{film_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        details_resp.raise_for_status()
        details = details_resp.json()

        credits_resp = requests.get(
            f"{TMDB_BASE_URL}/movie/{film_id}/credits",
            params={"api_key": TMDB_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        credits_resp.raise_for_status()
        credits = credits_resp.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "TMDB details request timed out."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"TMDB details fetch failed: {e}"}

    return {
        "status": "ok",
        "film": _shape_film(details, credits),
        "other_matches": other_matches,
    }