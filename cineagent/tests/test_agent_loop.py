"""
Stress test of the CineAgent ReAct loop against Qwen3.
Run ONE query at a time by uncommenting it. We isolate each capability.
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
    # --- Test 1: KNOWLEDGE query -> should pick search_cinema_knowledge ---
    #show("What is Christopher Nolan's approach to storytelling?")

    # --- Test 2: STREAMING query -> should pick streaming_lookup ---
     #show("Where can I watch Oppenheimer in India?")

    # --- Test 3: MULTI-TOOL sequential -> should call TWO tools across turns ---
    #show("Is Parasite streaming in India, and what is it about?")

    # --- Test 4: GERWIG relevance judgment -> should NOT pretend ---
    #show("Tell me about Greta Gerwig's directing style.")
    # --- Test 5: capability-gap probe -> tool can't do "all countries" ---
    show("Which countries is Oppenheimer available to stream in?")