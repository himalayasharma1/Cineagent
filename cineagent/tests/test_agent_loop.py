"""
First end-to-end run of the CineAgent ReAct loop against Qwen3.
This is the first time the full agent runs. Expect the unexpected —
a 4B model on a loop may surprise us. The trace tells us what happened.
"""

from cineagent.agent.loop import run_agent


def show(query):
    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)
    result = run_agent(query, verbose=True)
    print("\n" + "-" * 70)
    print(f"STOP REASON: {result['stop_reason']}")
    print(f"ITERATIONS: {result['iterations']}")
    print(f"\nFINAL ANSWER:\n{result['answer']}")
    print("-" * 70)


if __name__ == "__main__":
    # Start with the SIMPLEST possible case: one tool, one clear answer.
    show("Who directed Parasite and how long is it?")