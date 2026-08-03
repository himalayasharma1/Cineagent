"""
Tool schemas + dispatcher for the CineAgent ReAct loop.

Two responsibilities:
  1. TOOL_SCHEMAS — machine-readable descriptions of each tool, injected
     into the system prompt so the model knows what it can call and how.
     These descriptions ARE prompt engineering: the model selects tools
     based on them, so they're written for the model's benefit.
  2. dispatch_tool() — takes a parsed tool name + arguments dict, validates
     them, calls the real Python function, and returns its result. Never
     raises on a bad tool call — returns a structured error the loop can
     feed back to the model so it can recover.
"""

from cineagent.tools.cinema_search import search_cinema_knowledge
from cineagent.tools.film_details import get_film_details
from cineagent.tools.streaming_lookup import streaming_lookup


# ---------------------------------------------------------------
# Tool schemas — what the model sees.
# Descriptions are written to help the model choose CORRECTLY:
# each says what the tool is for AND, implicitly, when NOT to use it.
# ---------------------------------------------------------------
TOOL_SCHEMAS = [
    {
        "name": "search_cinema_knowledge",
        "description": (
            "Search a curated local knowledge base of world cinema for "
            "background, analysis, and critical writing about directors, "
            "actors, films, movements, and styles. Use this for questions "
            "about a filmmaker's approach, themes, legacy, or how films "
            "relate to each other. This is BACKGROUND KNOWLEDGE, not "
            "current facts — do NOT use it for a film's runtime, cast, "
            "release year, rating, or where to stream it."
        ),
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "The topic or question to search for.",
            }
        },
    },
    {
        "name": "get_film_details",
        "description": (
            "Get current factual metadata about a specific film from an "
            "external database: director, cast, runtime, genres, rating, "
            "plot summary, and release year. Use this for factual questions "
            "about a specific, named film. Do NOT use it for critical "
            "analysis or for streaming availability."
        ),
        "parameters": {
            "title": {
                "type": "string",
                "required": True,
                "description": "The film's title.",
            },
            "year": {
                "type": "integer",
                "required": False,
                "description": "Optional release year to disambiguate "
                               "same-title films.",
            },
        },
    },
    {
        "name": "streaming_lookup",
        "description": (
            "Find where a specific film can currently be streamed, rented, "
            "or bought, in a given country. Use this ONLY for questions "
            "about watching or availability ('where can I watch X', 'is X "
            "on Netflix'). Do NOT use it for film facts or analysis."
        ),
        "parameters": {
            "title": {
                "type": "string",
                "required": True,
                "description": "The film's title.",
            },
            "country": {
                "type": "string",
                "required": False,
                "description": "ISO 3166-1 alpha-2 country code "
                               "(e.g. 'IN', 'US'). Defaults to 'IN'.",
            },
        },
    },
]

# Map tool names to their functions. The dispatcher uses this.
_TOOL_FUNCTIONS = {
    "search_cinema_knowledge": search_cinema_knowledge,
    "get_film_details": get_film_details,
    "streaming_lookup": streaming_lookup,
}

# Map tool names to their required + optional parameter names,
# derived from the schemas, for validation.
_TOOL_PARAMS = {
    schema["name"]: schema["parameters"]
    for schema in TOOL_SCHEMAS
}


def dispatch_tool(tool_name: str, arguments: dict) -> dict:
    """
    Execute a tool call parsed from the model's output.

    Args:
        tool_name: name the model asked to call.
        arguments: dict of arguments the model supplied.

    Returns:
        A dict. On a valid call, the tool's own result dict. On an invalid
        call (unknown tool, missing required arg, wrong type), a structured
        error dict — NEVER raises. The loop feeds this back to the model so
        it can correct itself.
    """
    # --- Unknown tool ---
    if tool_name not in _TOOL_FUNCTIONS:
        available = ", ".join(_TOOL_FUNCTIONS.keys())
        return {
            "status": "error",
            "error_type": "unknown_tool",
            "message": (
                f"No tool named '{tool_name}'. "
                f"Available tools: {available}."
            ),
        }

    if not isinstance(arguments, dict):
        return {
            "status": "error",
            "error_type": "bad_arguments",
            "message": f"Arguments for '{tool_name}' must be an object, "
                       f"got {type(arguments).__name__}.",
        }

    # --- Validate required arguments are present ---
    param_spec = _TOOL_PARAMS[tool_name]
    for param_name, spec in param_spec.items():
        if spec.get("required") and param_name not in arguments:
            return {
                "status": "error",
                "error_type": "missing_argument",
                "message": (
                    f"Tool '{tool_name}' requires argument "
                    f"'{param_name}' but it was not provided."
                ),
            }

    # --- Drop any arguments the tool doesn't accept ---
    # (Small models sometimes hallucinate extra args; we filter rather
    # than crash by passing unexpected kwargs.)
    accepted = {
        k: v for k, v in arguments.items()
        if k in param_spec
    }

    # --- Call the real tool. Wrap so a tool bug can't kill the loop. ---
    try:
        return _TOOL_FUNCTIONS[tool_name](**accepted)
    except Exception as e:
        return {
            "status": "error",
            "error_type": "tool_exception",
            "message": f"Tool '{tool_name}' raised {type(e).__name__}: {e}",
        }