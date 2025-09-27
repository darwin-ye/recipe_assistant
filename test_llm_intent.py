#!/usr/bin/env python3
"""Test LLM-enhanced intent detection for edge cases"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_llm_intent_enhancement():
    agent = SimpleReActAgent(llm)

    # Test cases that should now work with LLM fallback
    edge_cases = [
        # Original failing case
        ("show me the recipe I have the most", "analytics_frequent"),

        # Additional challenging cases
        ("what do I cook most often", "analytics_frequent"),
        ("my most made dish", "analytics_frequent"),
        ("the recipe I prepare frequently", "analytics_frequent"),
        ("what's the dish I make repeatedly", "analytics_frequent"),
        ("recipe I cook all the time", "analytics_frequent"),

        # Analytics count edge cases
        ("how often do I use chicken", "analytics_count"),
        ("frequency of beef in my recipes", "analytics_count"),
        ("how much do I cook with salmon", "analytics_count"),
    ]

    print("🧪 Testing LLM-Enhanced Intent Detection")
    print("=" * 60)

    passed = 0
    failed = 0

    for input_text, expected_intent in edge_cases:
        print(f"\n🗣️  Testing: '{input_text}'")

        # Test just the intent detection
        actual_intent, actual_params = agent.detect_intent(input_text)

        if actual_intent == expected_intent:
            print(f"✅ Correctly detected as: {actual_intent}")
            if actual_params:
                print(f"   Params: {actual_params}")
            passed += 1
        else:
            print(f"❌ Expected: {expected_intent}, Got: {actual_intent}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed")

    # Test the original user query that was failing
    print(f"\n🎯 Testing Original Failing Query:")
    original_query = "show me the recipe I have the most"
    print(f"Query: '{original_query}'")

    result = agent.run(original_query)
    if "Most Frequent Recipe Analysis" in result:
        print("✅ Now successfully handles the original failing case!")
        print(f"Preview: {result.split(chr(10))[2]}")  # Show result preview
    else:
        print("❌ Still not working correctly")
        print(f"Result: {result[:100]}...")

    return failed == 0

if __name__ == "__main__":
    test_llm_intent_enhancement()