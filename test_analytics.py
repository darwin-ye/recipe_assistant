#!/usr/bin/env python3
"""Test the new analytics features"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_analytics_features():
    agent = SimpleReActAgent(llm)

    test_cases = [
        # Test most frequent recipe
        ("show me the most frequent recipe", "analytics_frequent", {}),
        ("what is the most common recipe", "analytics_frequent", {}),
        ("most popular recipe", "analytics_frequent", {}),

        # Test ingredient counting
        ("how many recipes with chicken", "analytics_count", {"ingredient": "chicken"}),
        ("count recipes with salmon", "analytics_count", {"ingredient": "salmon"}),
        ("how many chicken recipes", "analytics_count", {"ingredient": "chicken"}),
        ("count recipes containing tomatoes", "analytics_count", {"ingredient": "tomatoes"}),
    ]

    print("🧪 Testing Analytics Features")
    print("=" * 50)

    passed = 0
    failed = 0

    for input_text, expected_intent, expected_params in test_cases:
        actual_intent, actual_params = agent.detect_intent(input_text)

        # Check intent
        intent_match = actual_intent == expected_intent

        # Check parameters
        params_match = True
        for key in expected_params:
            if key in actual_params:
                if key == "ingredient":
                    # Check if the extracted ingredient matches expectation
                    params_match = actual_params[key] == expected_params[key]
                else:
                    params_match = True
            else:
                params_match = False

        if intent_match and params_match:
            print(f"✅ '{input_text}' -> {actual_intent} {actual_params}")
            passed += 1
        else:
            print(f"❌ '{input_text}' -> Expected: {expected_intent} {expected_params}, Got: {actual_intent} {actual_params}")
            failed += 1

    print(f"\n📊 Intent Detection Results: {passed} passed, {failed} failed")

    # Test actual execution
    print(f"\n🔬 Testing Execution:")

    test_executions = [
        "show me the most frequent recipe",
        "how many recipes with chicken",
        "count recipes with salmon"
    ]

    for test_input in test_executions:
        print(f"\nTesting: '{test_input}'")
        try:
            result = agent.run(test_input)
            if result and len(result) > 50:  # Should have substantial response
                print(f"✅ Execution successful ({len(result)} chars)")
                # Show first line of result
                first_line = result.split('\n')[0][:100]
                print(f"   Preview: {first_line}...")
            else:
                print(f"❌ Execution returned short response: {result}")
        except Exception as e:
            print(f"❌ Execution error: {str(e)}")

    return failed == 0

if __name__ == "__main__":
    test_analytics_features()