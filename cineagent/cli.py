"""
Interactive CLI for CineAgent.

Type a question, watch the agent reason through it — every Thought,
Tool call, and Observation — then see the final answer. Type 'quit' to exit.

This is the demo surface: it makes the reasoning LOOP visible, which is
the whole point of an agent (vs. a single-shot pipeline that only shows
an answer).
"""

import os
import sys
import warnings
import contextlib

# Quiet down noisy third-party warnings before anything imports them.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")


# --- Simple ANSI colors for a readable trace (works in most terminals) ---
class C:
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    GREY = "\033[90m"
    RESET = "\033[0m"


@contextlib.contextmanager
def _suppress_output():
    """Temporarily silence stdout/stderr (for noisy model/embedder loads)."""
    with open(os.devnull, "w") as devnull:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err


def _warmup():
    """
    Pre-load the model and embedder BEFORE the prompt appears, so the
    first query doesn't dump load noise mid-conversation. We suppress
    the load logs for a clean demo start.
    """
    print(f"{C.DIM}Starting CineAgent… (loading local model){C.RESET}")
    with _suppress_output():
        # Importing here (after warning filters) and touching the model
        # + embedder forces them to load now, quietly.
        from cineagent.agent.loop import _get_llm
        from cineagent.tools.cinema_search import _get_embedder, _get_collection
        _get_llm()
        _get_embedder()
        _get_collection()


def print_banner():
    print(f"\n{C.BOLD}{C.CYAN}CineAgent{C.RESET} — ask me about world cinema.")
    print(f"{C.DIM}Films, directors, styles, and where to watch. "
          f"Type 'quit' to exit.{C.RESET}\n")


def print_trace_step(step):
    """Pretty-print one iteration of the agent's reasoning."""
    n = step["iteration"]
    reasoning = step.get("reasoning") or ""
    tool_call = step.get("tool_call")
    observation = step.get("observation")

    print(f"  {C.GREY}┌─ step {n}{C.RESET}")

    if reasoning:
        r = reasoning if len(reasoning) < 300 else reasoning[:300] + "…"
        print(f"  {C.GREY}│{C.RESET} {C.DIM}thought:{C.RESET} {r}")

    if tool_call:
        if "parse_error" in tool_call:
            print(f"  {C.GREY}│{C.RESET} {C.YELLOW}tool call could not be "
                  f"parsed — retrying{C.RESET}")
        else:
            name = tool_call.get("name")
            args = tool_call.get("arguments")
            print(f"  {C.GREY}│{C.RESET} {C.YELLOW}action:{C.RESET} "
                  f"{name}({args})")

    if observation:
        obs = observation if len(observation) < 200 else observation[:200] + "…"
        print(f"  {C.GREY}│{C.RESET} {C.DIM}observation:{C.RESET} {obs}")

    print(f"  {C.GREY}└─{C.RESET}")


def main():
    _warmup()          # load quietly, before the banner
    print_banner()

    # Import here so the warmup's suppression has already run.
    from cineagent.agent.loop import run_agent

    while True:
        try:
            query = input(f"{C.BOLD}You:{C.RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C.DIM}Goodbye.{C.RESET}")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print(f"{C.DIM}Goodbye.{C.RESET}")
            break

        print(f"\n{C.DIM}thinking…{C.RESET}\n")

        # Suppress any residual per-call load noise (e.g. embedder on first
        # retrieval) so the trace stays clean.
        with _suppress_output():
            result = run_agent(query, verbose=False)

        for step in result["trace"]:
            print_trace_step(step)

        print(f"\n{C.BOLD}{C.GREEN}CineAgent:{C.RESET} {result['answer']}\n")
        print(f"{C.GREY}[{result['stop_reason']}, "
              f"{result['iterations']} step(s)]{C.RESET}\n")
        print(f"{C.GREY}{'─' * 60}{C.RESET}\n")


if __name__ == "__main__":
    main()