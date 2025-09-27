#!/usr/bin/env python3
"""Test the enhanced intent patterns"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_enhanced_patterns():
    agent = SimpleReActAgent(llm)

    test_cases = [
        # Enhanced frequent recipe patterns
        ("show me the recipe I always have", "analytics_frequent", {}),
        ("recipe I always make", "analytics_frequent", {}),
        ("what's my go-to recipe", "analytics_frequent", {}),
        ("my favorite recipe", "analytics_frequent", {}),
        ("what do I usually cook", "analytics_frequent", {}),
        ("my regular recipe", "analytics_frequent", {}),
        ("my staple recipe", "analytics_frequent", {}),
        ("my signature recipe", "analytics_frequent", {}),
        ("the recipe I always use", "analytics_frequent", {}),
        ("what recipe do I always make", "analytics_frequent", {}),

        # Enhanced counting patterns
        ("how often do I use chicken", "analytics_count", {"ingredient": "chicken"}),
        ("how much do I cook with beef", "analytics_count", {"ingredient": "beef"}),
        ("frequency of salmon in my recipes", "analytics_count", {"ingredient": "salmon"}),

        # Original patterns should still work
        ("most frequent recipe", "analytics_frequent", {}),
        ("how many chicken recipes", "analytics_count", {"ingredient": "chicken"}),
    ]

    print("🧪 Testing Enhanced Intent Patterns")
    print("=" * 60)

    passed = 0
    failed = 0

    for input_text, expected_intent, expected_params in test_cases:
        actual_intent, actual_params = agent.detect_intent(input_text)

        # Check intent
        intent_match = actual_intent == expected_intent

        # Check parameters for count intent
        params_match = True
        if expected_intent == "analytics_count":
            ingredient_match = actual_params.get("ingredient") == expected_params.get("ingredient")
            params_match = ingredient_match

        if intent_match and params_match:
            print(f"✅ '{input_text}' -> {actual_intent}")
            if actual_params:
                print(f"    Params: {actual_params}")
            passed += 1
        else:
            print(f"❌ '{input_text}' -> Expected: {expected_intent} {expected_params}")
            print(f"    Got: {actual_intent} {actual_params}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")

    # Test the specific user query
    print(f"\n🎯 Testing User's Original Query:")
    test_input = "show me the recipe I always have"
    result = agent.run(test_input)

    print(f"Query: '{test_input}'")
    if "Most Frequent Recipe Analysis" in result:
        print("✅ Successfully recognized as frequent recipe request!")
    else:
        print("❌ Did not recognize as frequent recipe request")
        print(f"Result: {result[:100]}...")

    return failed == 0

if __name__ == "__main__":
    test_enhanced_patterns()