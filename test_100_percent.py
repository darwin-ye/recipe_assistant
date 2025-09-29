#!/usr/bin/env python3
"""
Test the enhanced JSON parsing to achieve 100% success rate
Tests edge cases that previously caused failures
"""

import json
from llm_intent_classifier import LLMIntentClassifier
from llm import llm

def test_extreme_edge_cases():
    """Test the most challenging JSON parsing cases"""

    classifier = LLMIntentClassifier(llm)

    # Edge cases that typically cause JSON parsing failures
    test_cases = [
        # Incomplete JSON
        '{"intent": "create_recipe", "confidence": 0.9,',

        # Malformed quotes
        '{"intent": create_recipe, "confidence": 0.9}',

        # Mixed content with JSON
        'Here is the analysis: {"intent": "create_recipe", "confidence": 0.9} Hope this helps!',

        # Missing closing braces
        '{"intent": "create_recipe", "entities": {"ingredients": ["chicken"]',

        # Extra commas
        '{"intent": "create_recipe", "confidence": 0.9,}',

        # Template placeholders
        '{"intent": "create_recipe", "confidence": null_or_number, "entities": {}}',

        # Completely broken JSON
        'intent: create_recipe\nconfidence: 0.9\nentities: ingredients: chicken',

        # Empty or null responses
        '',
        'null',
        '{}',

        # Non-JSON text
        'I cannot determine the intent for this query.',

        # Partial JSON with text
        'Based on analysis: {"intent": "create_recipe" and the confidence is high.',

        # Multiple JSON-like objects
        '{"intent": "help"} and also {"intent": "create_recipe"}',

        # Escape character issues
        '{"intent": "create_recipe", "reasoning": "User wants to create a "special" recipe"}',

        # Unicode and special characters
        '{"intent": "créer_recette", "confidence": 0.9}',

        # Very long incomplete JSON
        '{"intent": "create_recipe", "entities": {"ingredients": ["chicken", "tomatoes", "onions", "garlic",' * 10,
    ]

    print("🔬 Testing Enhanced JSON Parsing for 100% Success Rate")
    print("=" * 70)

    success_count = 0
    total_tests = len(test_cases)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i:2d}/{total_tests}: {test_case[:50]}{'...' if len(test_case) > 50 else ''}")

        try:
            # Test the enhanced parsing
            result = classifier._parse_json_with_fallback(test_case)

            # Verify we got a valid dict with required fields
            if isinstance(result, dict) and "intent" in result:
                print(f"✅ SUCCESS: {result['intent']} (confidence: {result.get('confidence', 'N/A')})")
                success_count += 1
            else:
                print(f"❌ FAILED: Invalid result type or missing fields")

        except Exception as e:
            print(f"❌ EXCEPTION: {e}")

    success_rate = (success_count / total_tests) * 100
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Success Rate: {success_count}/{total_tests} = {success_rate:.1f}%")

    if success_rate == 100.0:
        print("🎉 ACHIEVED 100% SUCCESS RATE!")
    else:
        print(f"⚠️  Still need to fix {total_tests - success_count} cases")

    return success_rate

def test_real_world_scenarios():
    """Test with realistic LLM responses that might cause issues"""

    classifier = LLMIntentClassifier(llm)

    # Simulate real LLM responses that could be problematic
    real_world_cases = [
        # LLM provides explanation before JSON
        '''Based on the user's query, I can analyze this as follows:

{
    "intent": "create_recipe",
    "confidence": 0.9,
    "entities": {
        "ingredients": ["chicken", "pasta"]
    }
}

This analysis shows the user wants to create a recipe.''',

        # LLM uses markdown formatting
        '''```json
{
    "intent": "scale_recipe",
    "confidence": 0.8
}
```''',

        # LLM provides partial response
        '''I understand this is a recipe creation request. The structured analysis would be:
{
    "intent": "create_recipe"
    // Analysis incomplete due to processing limits''',

        # LLM uses different field names
        '''{"user_intent": "create_recipe", "certainty": 0.9, "extracted_entities": {}}''',

        # LLM provides multiple JSON objects
        '''First analysis: {"intent": "help", "confidence": 0.3}
Better analysis: {"intent": "create_recipe", "confidence": 0.9}''',
    ]

    print("\n🌍 Testing Real-World LLM Response Scenarios")
    print("=" * 50)

    success_count = 0
    total_tests = len(real_world_cases)

    for i, test_case in enumerate(real_world_cases, 1):
        print(f"\nReal-world Test {i}/{total_tests}:")
        print(f"Input: {test_case[:100]}...")

        try:
            result = classifier._parse_json_with_fallback(test_case)

            if isinstance(result, dict) and "intent" in result:
                print(f"✅ Extracted: {result['intent']}")
                success_count += 1
            else:
                print(f"❌ Failed to extract valid intent")

        except Exception as e:
            print(f"❌ Exception: {e}")

    real_world_rate = (success_count / total_tests) * 100
    print(f"\n🌍 Real-world Success Rate: {success_count}/{total_tests} = {real_world_rate:.1f}%")

    return real_world_rate

if __name__ == "__main__":
    edge_case_rate = test_extreme_edge_cases()
    real_world_rate = test_real_world_scenarios()

    overall_rate = (edge_case_rate + real_world_rate) / 2
    print(f"\n🏆 OVERALL SUCCESS RATE: {overall_rate:.1f}%")

    if overall_rate >= 99.5:
        print("🎉 EXCELLENT! Nearly perfect JSON parsing achieved!")
    elif overall_rate >= 95:
        print("✅ VERY GOOD! Strong JSON parsing performance!")
    else:
        print("⚠️  Needs improvement to reach 100% target")