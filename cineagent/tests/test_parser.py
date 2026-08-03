"""
Test the parser in isolation — no model involved.
Feeds sample model-output strings and verifies correct extraction.
"""

from cineagent.agent.parser import parse_model_output


def show(label, raw_output):
    result = parse_model_output(raw_output)
    print(f"\n{'=' * 60}")
    print(f"TEST: {label}")
    print("-" * 60)
    print(f"Action: {result['action']}")
    print(f"Reasoning: {result['reasoning'][:80]!r}")
    if result["action"] == "act":
        for i, tc in enumerate(result["tool_calls"], 1):
            if "parse_error" in tc:
                print(f"  Tool call {i}: PARSE ERROR — {tc['parse_error']}")
            else:
                print(f"  Tool call {i}: {tc['name']}({tc['arguments']})")
    else:
        print(f"Final answer: {result['final_answer'][:80]!r}")


if __name__ == "__main__":
    # 1. Reasoning + one clean tool call (the normal case)
    show("Reasoning + one tool call",
         'I should look up this film\'s details.\n'
         '<tool_call>\n{"name": "get_film_details", "arguments": {"title": "Parasite"}}\n</tool_call>')

    # 2. No tool call -> answer
    show("No tool call (final answer)",
         "Parasite was directed by Bong Joon-ho and won Best Picture in 2020.")

    # 3. Two tool calls in one output
    show("Two tool calls",
         'Need details and streaming.\n'
         '<tool_call>{"name": "get_film_details", "arguments": {"title": "Drive"}}</tool_call>\n'
         '<tool_call>{"name": "streaming_lookup", "arguments": {"title": "Drive", "country": "IN"}}</tool_call>')

    # 4. Malformed JSON inside tool_call
    show("Malformed JSON in tool_call",
         '<tool_call>{"name": "get_film_details", "arguments": {"title": "Parasite",}}</tool_call>')

    # 5. tool_call missing name
    show("tool_call missing name",
         '<tool_call>{"arguments": {"title": "Parasite"}}</tool_call>')

    # 6. Tool call with no arguments field
    show("tool_call with no arguments",
         '<tool_call>{"name": "search_cinema_knowledge"}</tool_call>')

    # 7. Empty output
    show("Empty output", "")

    print(f"\n{'=' * 60}")
    print("All parser tests complete.")