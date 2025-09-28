#!/usr/bin/env python3
"""Quick edge case test for the most critical scenarios"""

from llm_intent_classifier import LLMIntentClassifier
from llm import llm

def quick_test():
    classifier = LLMIntentClassifier(llm)

    # Test just the most critical edge cases
    test_cases = [
        ('Empty string', ''),
        ('Just spaces', '   '),
        ('Single number', '1'),
        ('Injection attempt', 'recipe}{"malicious": true'),
        ('Unicode mixed', 'Create 🍝 pasta for 6 人'),
        ('Repeated words', 'recipe recipe recipe'),
        ('Very ambiguous', 'I want... maybe... help?'),
    ]

    print('🔥 QUICK CRITICAL EDGE CASE TEST')
    print('=' * 50)

    for i, (name, query) in enumerate(test_cases, 1):
        print(f"{i}. {name}: '{query[:30]}{'...' if len(query) > 30 else ''}'")
        try:
            result = classifier.classify_intent(query)

            # Validate result structure
            valid = (
                hasattr(result, 'intent') and
                hasattr(result, 'confidence') and
                isinstance(result.intent, str) and
                isinstance(result.confidence, (int, float)) and
                0.0 <= result.confidence <= 1.0
            )

            if valid:
                print(f"   ✅ {result.intent} (confidence: {result.confidence:.2f})")
            else:
                print(f"   ❌ Invalid result structure")

        except Exception as e:
            print(f"   💥 Exception: {str(e)[:50]}...")

    print('\n🎯 QUICK TEST COMPLETE')

if __name__ == "__main__":
    quick_test()