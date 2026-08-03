"""
Parser for the CineAgent ReAct loop.

Qwen3 emits text that mixes free-form reasoning with zero or more
structured tool calls in <tool_call>...</tool_call> tags (JSON inside).
This module turns that raw text into a structured decision the loop
can act on.
"""

import re
import json

# Matches <tool_call> ... </tool_call>, capturing the inside, across newlines.
_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


def parse_model_output(text: str) -> dict:
    """
    Parse one turn of model output.

    Returns a dict with keys: action ("act"|"answer"), reasoning (str),
    tool_calls (list), final_answer (str or None).

    Logic:
      - One or more <tool_call> blocks present -> action "act".
      - None present -> action "answer"; the whole text is the final answer.
    """
    if text is None:
        text = ""

    matches = _TOOL_CALL_PATTERN.findall(text)

    # Reasoning = everything OUTSIDE the tool_call tags.
    reasoning = _TOOL_CALL_PATTERN.sub("", text).strip()

    # No tool calls -> the model is answering.
    if not matches:
        return {
            "action": "answer",
            "reasoning": reasoning,
            "tool_calls": [],
            "final_answer": reasoning,
        }

    # One or more tool calls -> parse each.
    tool_calls = []
    for raw in matches:
        parsed = _parse_single_tool_call(raw)
        tool_calls.append(parsed)

    return {
        "action": "act",
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "final_answer": None,
    }


def _parse_single_tool_call(raw: str) -> dict:
    """
    Parse the JSON inside one <tool_call> block.

    Returns {"name", "arguments"} on success, or {"parse_error", "raw"}
    on malformed JSON / missing fields, so the loop can feed the error
    back to the model.
    """
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        return {"parse_error": f"Invalid JSON in tool_call: {e}", "raw": raw}

    if not isinstance(obj, dict):
        return {"parse_error": "tool_call is not a JSON object", "raw": raw}

    name = obj.get("name")
    if not name or not isinstance(name, str):
        return {"parse_error": "tool_call missing 'name' or 'name' is not a string", "raw": raw}

    arguments = obj.get("arguments", {})
    if not isinstance(arguments, dict):
        return {"parse_error": "tool_call 'arguments' must be an object", "raw": raw}

    return {"name": name, "arguments": arguments}