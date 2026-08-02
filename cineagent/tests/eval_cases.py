"""
Eval cases for CineAgent — Retrieval layer (Tool 1: search_cinema_knowledge).

Each case is a dict with:
  - id:       unique identifier
  - query:    the input to search_cinema_knowledge
  - category: grouping for aggregate pass rates
  - expect:   dict of assertions the runner checks
  - note:     plain-English statement of what failure this case catches

The `expect` dict is intentionally extensible. Today it holds retrieval
assertions (status, min_results, source_contains_any). When the agent loop
exists, agent-level cases add keys like expected_tools / max_iterations to
the SAME format — the runner checks whichever assertions are present.

Assertion keys currently supported:
  - status:              exact match on returned status ("ok"/"no_results"/"error")
  - min_results:         at least this many results returned
  - source_contains_any: at least one result's source contains one of these substrings
  - source_contains_all: results collectively cover all these substrings (for cross-corpus)
"""

"""

KNOWN EXPECTED FAILURES (documented, not bugs):
  - ood_002 (Greta Gerwig): tool-layer limitation, deferred to agent layer.
    See the case note. Expected steady-state score at tool layer: 14/15.
"""

EVAL_CASES = [
    # ---------------------------------------------------------------
    # wiki_known — retrieval against the Wikipedia actor_/director_ set
    # ---------------------------------------------------------------
    {
        "id": "wiki_001",
        "query": "Christopher Nolan's approach to nonlinear storytelling",
        "category": "wiki_known",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Christopher_Nolan"],
        },
        "note": "Core Wikipedia director, well-covered (30 chunks). Baseline recall.",
    },
    {
        "id": "wiki_002",
        "query": "Shah Rukh Khan career and notable films",
        "category": "wiki_known",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Shah_Rukh_Khan"],
        },
        "note": "Indian cinema, well-covered actor (42 chunks). Tests non-Hollywood recall.",
    },
    {
        "id": "wiki_003",
        "query": "Martin Scorsese and his collaborations with Robert De Niro",
        "category": "wiki_known",
        "expect": {
            "status": "ok",
            "min_results": 2,
            "source_contains_any": ["Scorsese", "De_Niro"],
        },
        "note": "Query spans two well-covered sources. Should retrieve from either/both.",
    },
    {
        "id": "wiki_004",
        "query": "Imtiaz Ali films",
        "category": "wiki_known",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Imtiaz_Ali"],
        },
        "note": "THIN source (6 chunks). Stress-tests recall when coverage is sparse.",
    },

    # ---------------------------------------------------------------
    # senses_known — retrieval against the art-house essay set
    # ---------------------------------------------------------------
    {
        "id": "senses_001",
        "query": "Pedro Almodóvar's use of melodrama and color",
        "category": "senses_known",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Almodóvar"],
        },
        "note": "Art-house essay retrieval. Tests the Senses sub-corpus is reachable.",
    },
    {
        "id": "senses_002",
        "query": "Chantal Akerman's minimalist style and long takes",
        "category": "senses_known",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Akerman"],
        },
        "note": "Essayistic/critical query. Different text character than Wikipedia.",
    },
    {
        "id": "senses_003",
        "query": "Wong Kar-wai mood and cinematography in his romances",
        "category": "senses_known",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Wong_Kar-wai"],
        },
        "note": "Present ONLY in Senses (no Wikipedia file). Confirms Senses-only recall.",
    },

    # ---------------------------------------------------------------
    # cross_corpus — figures present in BOTH sub-corpora
    # ---------------------------------------------------------------
    {
        "id": "cross_001",
        "query": "Guru Dutt filmmaking style and legacy",
        "category": "cross_corpus",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Guru_Dutt", "Dutt_Guru"],
        },
        "note": "Present in BOTH (Wikipedia 12 chunks + Senses 14). Either source is a valid hit.",
    },
    {
        "id": "cross_002",
        "query": "Mrinal Sen's contribution to parallel cinema",
        "category": "cross_corpus",
        "expect": {
            "status": "ok",
            "min_results": 1,
            "source_contains_any": ["Mrinal_Sen", "Sen_Mrinal"],
        },
        "note": "Dual-source, but Senses (28 chunks) far richer than Wikipedia (7). Likely Senses-dominant retrieval.",
    },

    # ---------------------------------------------------------------
    # out_of_domain — cinema-adjacent but genuinely ABSENT
    # ---------------------------------------------------------------
    {
        "id": "ood_001",
        "query": "Bong Joon-ho and Parasite",
        "category": "out_of_domain",
        "expect": {
            "status": "no_results",
        },
        "note": "Famous director, but NO Korean cinema in corpus. Guardrail must reject despite fame.",
    },
    {
        "id": "ood_002",
        "query": "Greta Gerwig directing style",
        "category": "out_of_domain",
        "expect": {
            "status": "no_results",
        },
        "note": (
            "KNOWN EXPECTED FAILURE at tool layer. Query matches the *concept* "
            "'female director' — retrieves Arzner/Akerman chunks at dist ~0.51, "
            "inside the in-domain range. No threshold can reject this without "
            "harming recall (in-domain queries score 0.42-0.55). Deferred to the "
            "agent layer: the LLM reads retrieved chunks, sees they're about "
            "Arzner not Gerwig, and judges them irrelevant. Documented, not hidden."
        ),
    },
    {
        "id": "ood_003",
        "query": "how to change a car tyre",
        "category": "out_of_domain",
        "expect": {
            "status": "no_results",
        },
        "note": "Completely non-cinema. Hard negative — should be trivially rejected.",
    },

    # ---------------------------------------------------------------
    # edge_case — malformed or degenerate input
    # ---------------------------------------------------------------
    {
        "id": "edge_001",
        "query": "",
        "category": "edge_case",
        "expect": {
            "status": "error",
        },
        "note": "Empty query. Must return error, not crash or retrieve garbage.",
    },
    {
        "id": "edge_002",
        "query": "   ",
        "category": "edge_case",
        "expect": {
            "status": "error",
        },
        "note": "Whitespace-only. Tests the .strip() check in the tool's validation.",
    },
    {
        "id": "edge_003",
        "query": "asdkfj qwerty zxcvbn",
        "category": "edge_case",
        "expect": {
            "status": "no_results",
        },
        "note": "Gibberish (non-empty). Should retrieve nothing above threshold, not error.",
    },
# ===============================================================
    # TOOL 2 — get_film_details (TMDB, live API)
    # Assertions on STABLE fields only (status, director, year,
    # language). Never on ratings/vote counts — those drift.
    # ===============================================================

    {
        "id": "film_001",
        "query": "Parasite",
        "tool": "get_film_details",
        "category": "film_single_known",
        "expect": {
            "status": "ok",
            "film_field_contains": {"director": "Bong", "original_language": "ko"},
            "film_year": "2019",
        },
        "note": "Well-known film. Notably ABSENT from local corpus — proves Tool 2 fetches what Tool 1 can't.",
    },
    {
        "id": "film_002",
        "query": "The Godfather",
        "tool": "get_film_details",
        "category": "film_single_known",
        "expect": {
            "status": "ok",
            "film_field_contains": {"director": "Coppola"},
            "film_year": "1972",
        },
        "note": "Director extraction from credits endpoint. Stable fields only.",
    },
    {
        "id": "film_003",
        "query": "Sholay",
        "tool": "get_film_details",
        "category": "film_single_known",
        "expect": {
            "status": "ok",
            "film_field_contains": {"director": "Sippy", "original_language": "hi"},
            "film_year": "1975",
        },
        "note": "Non-Hollywood coverage. Tests TMDB handles Indian cinema.",
    },

    # --- Disambiguation ---
    {
        "id": "film_disambig_001",
        "query": "Drive",
        "tool": "get_film_details",
        "year": 2011,
        "category": "film_disambiguation",
        "expect": {
            "status": "ok",
            "film_field_contains": {"director": "Refn"},
            "film_year": "2011",
        },
        "note": "Ambiguous title + year param. Must resolve to the 2011 Refn film.",
    },

    # --- Failure modes ---
    {
        "id": "film_ood_001",
        "query": "qwertyuiop asdfghjkl zxcvbnm",
        "tool": "get_film_details",
        "category": "film_no_results",
        "expect": {
            "status": "no_results",
        },
        "note": "Gibberish title. TMDB returns nothing — tool must report no_results, not crash.",
    },
    {
        "id": "film_edge_001",
        "query": "",
        "tool": "get_film_details",
        "category": "film_edge_case",
        "expect": {
            "status": "error",
        },
        "note": "Empty title. Input validation before any network call.",
    },
    
    
    # ===============================================================
    # TOOL 3 — streaming_lookup (TMDB watch-providers, live API)
    # Assertions on STABLE fields only (status, title, year, country).
    # NEVER on specific providers — those drift as licensing changes.
    # ===============================================================

    {
        "id": "stream_001",
        "query": "Oppenheimer",
        "tool": "streaming_lookup",
        "country": "IN",
        "category": "stream_available",
        "expect": {
            "status": "ok",
            "top_field_contains": {"title": "Oppenheimer", "country": "IN"},
        },
        "note": "Major recent film, widely available in IN. Asserts status + stable fields, not providers.",
    },
    {
        "id": "stream_002",
        "query": "Parasite",
        "tool": "streaming_lookup",
        "country": "US",
        "category": "stream_available",
        "expect": {
            "status": "ok",
            "top_field_contains": {"title": "Parasite", "country": "US"},
        },
        "note": "Country-awareness: same film, US region. Verifies country param flows through.",
    },

    # --- no_availability (deterministic-ish probe against a no-market country) ---
    {
        "id": "stream_no_avail_001",
        "query": "Oppenheimer",
        "tool": "streaming_no_avail_probe",
        "category": "stream_no_availability",
        "expect": {
            "status": "no_availability",
        },
        "note": (
            "Verifies the no_availability branch: a real film queried against a "
            "country with no streaming market (AQ) reliably has no provider block. "
            "Deviates from live-eval default because this state can't be triggered "
            "reliably against a normal-market country without flakiness."
        ),
    },

    # --- Failure modes ---
    {
        "id": "stream_no_results_001",
        "query": "qwertyuiop asdfghjkl zxcvbnm",
        "tool": "streaming_lookup",
        "category": "stream_no_results",
        "expect": {
            "status": "no_results",
        },
        "note": "Gibberish title. Film search fails → no_results, distinct from no_availability.",
    },
    {
        "id": "stream_edge_001",
        "query": "",
        "tool": "streaming_lookup",
        "category": "stream_edge_case",
        "expect": {
            "status": "error",
        },
        "note": "Empty title. Input validation before any network call.",
    },]