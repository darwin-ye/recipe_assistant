#!/usr/bin/env python3
"""Demo the LLM-enhanced intent recognition"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def demo_llm_enhanced():
    agent = SimpleReActAgent(llm)

    print("🚀 LLM-Enhanced Intent Recognition Demo")
    print("=" * 60)
    print("Testing queries that would previously fail:")
    print()

    test_queries = [
        # The original failing case
        "show me the recipe I have the most",

        # Other challenging cases now handled by LLM
        "what do I cook most often",
        "my most made dish",
        "the recipe I prepare frequently",
        "how often do I use chicken",
        "frequency of beef in my recipes",
    ]

    for query in test_queries:
        print(f"🗣️  User: {query}")
        print("─" * 40)

        # Run the full agent
        result = agent.run(query)

        # Show just the key result line
        if "Most Frequent Recipe Analysis" in result:
            key_line = [line for line in result.split('\n') if 'Most frequent recipe:' in line][0]
            print(f"✅ {key_line}")
        elif "Recipe Count for" in result:
            key_line = [line for line in result.split('\n') if 'Found' in line and 'recipe' in line][0]
            print(f"✅ {key_line}")
        else:
            print(f"Result: {result.split(chr(10))[0]}")
        print()

    print("🎯 Key Innovation: Hybrid Rule-Based + LLM System")
    print("• Fast rule-based matching for common patterns")
    print("• Smart LLM fallback for complex/edge cases")
    print("• Best of both worlds: speed + flexibility")

if __name__ == "__main__":
    demo_llm_enhanced()