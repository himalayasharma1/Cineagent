"""
Tool 3: streaming_lookup

Finds where a film is available to stream, rent, or buy in a given country,
using TMDB's watch-providers endpoint (which is sourced from JustWatch).

Reuses the same TMDB key as Tool 2 — no new vendor, no new secret.

Design commitments:
  - Self-contained: does its own title→ID search (Approach A), consistent
    with the other tools' independence.
  - Country-aware: streaming availability is regional, so country is a
    first-class parameter (defaults to India).
  - Four states: ok / no_availability / no_results / error.
    Crucially distinguishes "film exists but streams nowhere here"
    (no_availability) from "film not found at all" (no_results), so the
    agent can phrase the two situations differently.
  - Never raises on network failure — structured error dicts only.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# Configuration (shared conventions with Tool 2)
# ---------------------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
REQUEST_TIMEOUT = 8

# Default country for availability. ISO 3166-1 alpha-2 code.
# India, matching the primary use case.
DEFAULT_COUNTRY = "IN"

# ---------------------------------------------------------------
# Country normalization.
# The model (and users) say "India", "USA", "UK" — but TMDB keys
# availability by ISO 3166-1 alpha-2 codes ("IN", "US", "GB").
# We normalize common names/variants to codes. Unknown input is
# rejected LOUDLY (error status) rather than silently guessed —
# a visible error gets fixed; a silent wrong answer gets shipped.
# ---------------------------------------------------------------

# The set of valid 2-letter codes we accept directly (expand as needed).
_VALID_ISO_CODES = {
    "IN", "US", "GB", "CA", "AU", "FR", "DE", "JP", "KR", "IT",
    "ES", "BR", "MX", "NL", "SE", "AQ",
}

# Common names / variants -> ISO code.
_COUNTRY_NAME_TO_ISO = {
    "india": "IN",
    "united states": "US", "usa": "US", "us": "US", "america": "US",
    "united kingdom": "GB", "uk": "GB", "britain": "GB", "england": "GB",
    "canada": "CA",
    "australia": "AU",
    "france": "FR",
    "germany": "DE",
    "japan": "JP",
    "south korea": "KR", "korea": "KR",
    "italy": "IT",
    "spain": "ES",
    "brazil": "BR",
    "mexico": "MX",
    "netherlands": "NL",
    "sweden": "SE",
}


def _normalize_country(country: str):
    """
    Normalize a country string to an ISO 3166-1 alpha-2 code.

    Returns (code, None) on success, or (None, error_message) if the
    input can't be recognized — so the caller can fail loudly rather
    than silently querying a nonexistent country key.
    """
    if not country or not country.strip():
        return DEFAULT_COUNTRY, None

    raw = country.strip()

    # Already a valid 2-letter code?
    if raw.upper() in _VALID_ISO_CODES:
        return raw.upper(), None

    # A known country name / variant?
    mapped = _COUNTRY_NAME_TO_ISO.get(raw.lower())
    if mapped:
        return mapped, None

    # Unrecognized — fail loudly.
    return None, (
        f"Unrecognized country '{country}'. Use an ISO 3166-1 alpha-2 "
        f"code (e.g. 'IN', 'US') or a common country name."
    )
# ---------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------
def _extract_year(release_date: str):
    if release_date and len(release_date) >= 4:
        return release_date[:4]
    return None


def _search_film_id(title: str, year: int = None):
    """
    Self-contained title→ID search (Approach A).
    Returns (film_id, resolved_title, resolved_year) on success,
    or None if no film matches.
    Raises requests exceptions on network failure (caller handles).
    """
    params = {"api_key": TMDB_API_KEY, "query": title}
    if year:
        params["year"] = year

    resp = requests.get(
        f"{TMDB_BASE_URL}/search/movie",
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])

    if not results:
        return None

    top = results[0]
    return (
        top["id"],
        top.get("title"),
        _extract_year(top.get("release_date")),
    )


def _shape_providers(country_block: dict) -> dict:
    """
    Reduce TMDB's watch-providers country block to the three offer types
    we care about, as lists of provider names.

    TMDB structures availability as:
      flatrate -> subscription streaming (Netflix, Prime, etc.)
      rent     -> rental services
      buy      -> purchase services
    Each is a list of {provider_name, ...} dicts. We keep just names.
    """
    def names(offer_type):
        return [p.get("provider_name") for p in country_block.get(offer_type, [])]

    return {
        "streaming": names("flatrate"),
        "rent": names("rent"),
        "buy": names("buy"),
    }


# ---------------------------------------------------------------
# The tool itself
# ---------------------------------------------------------------
def streaming_lookup(title: str, country: str = DEFAULT_COUNTRY) -> dict:
    """
    Find where a film can be watched in a given country.

    Args:
        title:   Film title to look up.
        country: ISO 3166-1 alpha-2 country code (default "IN").

    Returns one of:

      Available:
        {
          "status": "ok",
          "title": <resolved title>,
          "year": <resolved year>,
          "country": <country code>,
          "streaming": [provider names],   # subscription
          "rent": [provider names],
          "buy": [provider names],
          "tmdb_watch_link": <url or None>
        }

      Film found, but no availability in this country:
        {"status": "no_availability", "title": ..., "country": ...,
         "message": ...}

      Film not found at all:
        {"status": "no_results", "message": ...}

      Error (empty input, missing key, network/API failure):
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

    country, country_error = _normalize_country(country)
    if country_error:
        return {"status": "error", "message": country_error}

    # --- Step 1: resolve title -> film ID ---
    try:
        found = _search_film_id(title)
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "TMDB request timed out."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"TMDB search failed: {e}"}

    if found is None:
        return {
            "status": "no_results",
            "message": f"No film found matching '{title}'.",
        }

    film_id, resolved_title, resolved_year = found

    # --- Step 2: fetch watch providers ---
    try:
        resp = requests.get(
            f"{TMDB_BASE_URL}/movie/{film_id}/watch/providers",
            params={"api_key": TMDB_API_KEY},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        providers_data = resp.json()
    except requests.exceptions.Timeout:
        return {"status": "error", "message": "TMDB providers request timed out."}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"TMDB providers fetch failed: {e}"}

    # TMDB returns availability keyed by country under "results".
    country_block = providers_data.get("results", {}).get(country)

    # Film exists, but no availability data for this country.
    if not country_block:
        return {
            "status": "no_availability",
            "title": resolved_title,
            "year": resolved_year,
            "country": country,
            "message": (
                f"'{resolved_title}' is not currently available to stream, "
                f"rent, or buy in {country}."
            ),
        }

    shaped = _shape_providers(country_block)

    # A country block can exist but still have no actual offers
    # (edge case). Treat empty-across-the-board as no_availability too.
    if not (shaped["streaming"] or shaped["rent"] or shaped["buy"]):
        return {
            "status": "no_availability",
            "title": resolved_title,
            "year": resolved_year,
            "country": country,
            "message": (
                f"'{resolved_title}' has no streaming, rental, or purchase "
                f"options in {country} right now."
            ),
        }

    return {
        "status": "ok",
        "title": resolved_title,
        "year": resolved_year,
        "country": country,
        "streaming": shaped["streaming"],
        "rent": shaped["rent"],
        "buy": shaped["buy"],
        # TMDB provides a link to its own watch page (no deep links).
        "tmdb_watch_link": country_block.get("link"),
    }