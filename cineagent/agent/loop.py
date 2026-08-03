"""
The CineAgent ReAct loop executor.

Ties together the system prompt, the parser, and the dispatcher into a
reasoning loop:

  build messages -> call Qwen3 -> parse -> (answer? stop) or (tool? dispatch,
  feed observation back) -> repeat, bounded by stop conditions.

Stop conditions (all three, per design):
  - natural:        model emits a final answer instead of a tool call
  - max_iterations: hard cap, prevents runaway loops
  - no_progress:    same tool + same args twice in a row -> model is stuck

Returns a rich trace, not just the answer, so every decision is legible
(for debugging, evals, and observability).
"""

import json
from llama_cpp import Llama

from cineagent.agent.prompt import build_system_prompt
from cineagent.agent.parser import parse_model_output
from cineagent.agent.dispatcher import dispatch_tool

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
MODEL_PATH = "./models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
MAX_ITERATIONS = 5

# How much of a tool observation to feed back to the model.
# Keeps context lean across iterations (Decision 3: compact + truncate).
SNIPPET_CHARS = 200          # per retrieved chunk from Tool 1
MAX_OBSERVATION_CHARS = 1500  # hard cap on any single observation string

# Lazy-loaded model (loaded once per process, not per call).
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=8192,
            n_gpu_layers=-1,
            verbose=False,
        )
    return _llm


# ---------------------------------------------------------------
# Observation shaping — compact what a tool returns before feeding back.
# ---------------------------------------------------------------
def _shape_observation(tool_name: str, result: dict) -> str:
    """
    Turn a tool's result dict into a compact string for the model.
    Tool 1 (retrieval) can be verbose, so truncate chunk text.
    Tools 2 & 3 are already compact — pass through as JSON.
    """
    status = result.get("status")

    # Errors and empty states: feed the status + message back plainly.
    if status in ("error", "no_results", "no_availability"):
        msg = result.get("message", "")
        return json.dumps({"status": status, "message": msg})

    if tool_name == "search_cinema_knowledge":
        # Truncate each chunk's text to a snippet.
        compact = {"status": status, "results": []}
        for r in result.get("results", []):
            compact["results"].append({
                "source": r.get("source"),
                "snippet": (r.get("text", "")[:SNIPPET_CHARS]),
            })
        obs = json.dumps(compact)
    else:
        # Tools 2 & 3 are already compact.
        obs = json.dumps(result)

    # Hard cap as a final safety net.
    if len(obs) > MAX_OBSERVATION_CHARS:
        obs = obs[:MAX_OBSERVATION_CHARS] + "...(truncated)"
    return obs


# ---------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------
def run_agent(user_query: str, verbose: bool = True) -> dict:
    """
    Run the ReAct loop for one user query.

    Returns:
      {
        "answer": <final answer string, or None if it never produced one>,
        "trace": [ {iteration, reasoning, tool_call, observation}, ... ],
        "stop_reason": "natural" | "max_iterations" | "no_progress",
        "iterations": <count>,
      }
    """
    llm = _get_llm()

    # A1: alternating-role message list. Grows as the loop proceeds.
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_query},
    ]

    trace = []
    last_tool_signature = None  # for no_progress detection
    stop_reason = None
    answer = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        # --- Call the model ---
        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )
        raw_output = response["choices"][0]["message"]["content"] or ""

        parsed = parse_model_output(raw_output)

        if verbose:
            print(f"\n--- Iteration {iteration} ---")
            print(f"Reasoning: {parsed['reasoning'][:200]}")

        # --- Case 1: model gave a final answer -> natural termination ---
        if parsed["action"] == "answer":
            answer = parsed["final_answer"]
            stop_reason = "natural"
            trace.append({
                "iteration": iteration,
                "reasoning": parsed["reasoning"],
                "tool_call": None,
                "observation": None,
            })
            if verbose:
                print(f"Final answer: {answer[:200]}")
            break

        # --- Case 2: model called a tool ---
        # v1 executes ONE tool per turn (Option A). If the model emitted
        # multiple, we take the first and ignore the rest this turn.
        first_call = parsed["tool_calls"][0]

        # Handle a parse error in the tool call: feed it back so the
        # model can correct itself.
        if "parse_error" in first_call:
            observation = json.dumps({
                "status": "error",
                "message": f"Your tool call could not be parsed: "
                           f"{first_call['parse_error']}. "
                           f"Please emit a valid tool call.",
            })
            tool_call_record = {"parse_error": first_call["parse_error"]}
        else:
            tool_name = first_call["name"]
            arguments = first_call["arguments"]

            # no_progress detection: same tool + same args as last turn.
            signature = (tool_name, json.dumps(arguments, sort_keys=True))
            if signature == last_tool_signature:
                stop_reason = "no_progress"
                trace.append({
                    "iteration": iteration,
                    "reasoning": parsed["reasoning"],
                    "tool_call": {"name": tool_name, "arguments": arguments},
                    "observation": "STOPPED: repeated identical tool call.",
                })
                if verbose:
                    print(f"Stopping: no progress (repeated {tool_name}).")
                break
            last_tool_signature = signature

            # Dispatch the tool.
            result = dispatch_tool(tool_name, arguments)
            observation = _shape_observation(tool_name, result)
            tool_call_record = {"name": tool_name, "arguments": arguments}

            if verbose:
                print(f"Tool call: {tool_name}({arguments})")
                print(f"Observation: {observation[:200]}")

        # Record this iteration in the trace.
        trace.append({
            "iteration": iteration,
            "reasoning": parsed["reasoning"],
            "tool_call": tool_call_record,
            "observation": observation,
        })

        # A1: append assistant's action and the observation to messages.
        messages.append({"role": "assistant", "content": raw_output})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}",
        })

    # --- Loop ended. If we ran out of iterations without an answer: ---
    if stop_reason is None:
        stop_reason = "max_iterations"

    return {
        "answer": answer,
        "trace": trace,
        "stop_reason": stop_reason,
        "iterations": len(trace),
    }