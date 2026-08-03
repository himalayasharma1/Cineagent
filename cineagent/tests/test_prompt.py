"""
Test that the system prompt builds correctly — no model involved.
Verifies the tool descriptions inject and the literal braces render.
"""

from cineagent.agent.prompt import build_system_prompt


if __name__ == "__main__":
    prompt = build_system_prompt()
    print(prompt)
    print("\n" + "=" * 60)
    print("SANITY CHECKS:")
    print("=" * 60)

    # The three tool names should appear
    for tool in ["search_cinema_knowledge", "get_film_details", "streaming_lookup"]:
        present = tool in prompt
        print(f"  {'✓' if present else '✗'} mentions {tool}")

    # Literal braces should render (the tool_call format needs real { })
    has_literal_braces = '{"name":' in prompt
    print(f"  {'✓' if has_literal_braces else '✗'} tool_call format has literal braces")

    # The placeholder should be GONE (fully substituted)
    no_placeholder = "{tool_descriptions}" not in prompt
    print(f"  {'✓' if no_placeholder else '✗'} no unfilled placeholder remains")

    print(f"\n  Prompt length: {len(prompt)} chars")