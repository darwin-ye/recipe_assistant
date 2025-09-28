#!/usr/bin/env python3
"""Test natural language flexibility between approaches"""

from llm import llm
from simple_react_agent import SimpleReActAgent
from llm_intent_classifier import LLMIntentClassifier

def test_flexibility():
    print("🗣️ Natural Language Flexibility Test")
    print("=" * 50)

    react_agent = SimpleReActAgent(llm)
    llm_classifier = LLMIntentClassifier(llm)

    # Different ways to ask for the same thing
    test_groups = {
        "Most Frequent Recipe (should all be analytics_frequent)": [
            "show me the recipe I always have",
            "what's my go-to recipe?",
            "my most frequent recipe",
            "what do I cook most often",
            "my signature dish",
            "recipe I make all the time"
        ],
        "Recipe Creation (should all be create_recipe)": [
            "create a chicken recipe",
            "make something with beef",
            "give me a salmon recipe",
            "cook up something new",
            "I want a new recipe"
        ]
    }

    for group, queries in test_groups.items():
        print(f"\n📋 {group}:")
        print("-" * 40)

        react_results = []
        llm_results = []

        for query in queries:
            # Test both
            react_intent, _ = react_agent.detect_intent(query)
            llm_result = llm_classifier.classify_intent(query)

            react_results.append(react_intent)
            llm_results.append(llm_result.intent)

            print(f"'{query}'")
            print(f"  ReAct: {react_intent}")
            print(f"  LLM:   {llm_result.intent} (conf: {llm_result.confidence:.2f})")

        # Check consistency
        react_consistent = len(set(react_results)) == 1
        llm_consistent = len(set(llm_results)) == 1

        print(f"\nConsistency:")
        print(f"  ReAct: {'✅ Consistent' if react_consistent else '❌ Inconsistent'} ({set(react_results)})")
        print(f"  LLM:   {'✅ Consistent' if llm_consistent else '❌ Inconsistent'} ({set(llm_results)})")

if __name__ == "__main__":
    test_flexibility()