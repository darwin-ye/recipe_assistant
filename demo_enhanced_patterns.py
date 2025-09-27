#!/usr/bin/env python3
"""Demo the enhanced semantic understanding"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def demo_enhanced_patterns():
    agent = SimpleReActAgent(llm)

    # Test various ways to ask for the most frequent recipe
    queries = [
        "show me the recipe I always have",
        "what's my go-to recipe?",
        "my favorite recipe",
        "what do I usually cook?",
        "recipe I always make",
        "my regular recipe",
        "my signature recipe",
    ]

    print("🎬 Enhanced Semantic Understanding Demo")
    print("=" * 60)
    print("Testing different ways to ask for the most frequent recipe:")
    print()

    for query in queries:
        print(f"🗣️  User: {query}")
        print("─" * 40)

        intent, params = agent.detect_intent(query)
        if intent == "analytics_frequent":
            print("✅ Correctly detected as: Most Frequent Recipe Analysis")
            result = agent.run(query)
            # Show just the first line of the result
            first_line = result.split('\n')[2] if len(result.split('\n')) > 2 else result
            print(f"📊 Result: {first_line}")
        else:
            print(f"❌ Incorrectly detected as: {intent}")
        print()

if __name__ == "__main__":
    demo_enhanced_patterns()