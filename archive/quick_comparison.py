#!/usr/bin/env python3
"""Quick comparison between ReAct and LLM approaches"""

import time
from llm import llm
from simple_react_agent import SimpleReActAgent
from llm_intent_classifier import LLMIntentClassifier

def quick_test():
    print("🚀 Quick Comparison: ReAct vs Pure LLM")
    print("=" * 50)

    # Initialize
    react_agent = SimpleReActAgent(llm)
    llm_classifier = LLMIntentClassifier(llm)

    # Key test cases
    test_cases = [
        ("create a chicken pasta recipe", "create_recipe"),
        ("show me the recipe I always have", "analytics_frequent"),
        ("what's my go-to recipe?", "analytics_frequent"),
        ("how many chicken recipes do I have?", "analytics_count"),
        ("2", "numbered_reference"),
        ("scale this to 8 people", "scale_recipe"),
    ]

    react_correct = 0
    llm_correct = 0

    print("Testing critical queries:\n")

    for query, expected in test_cases:
        print(f"Query: '{query}'")
        print(f"Expected: {expected}")

        # Test ReAct
        react_intent, _ = react_agent.detect_intent(query)
        react_match = react_intent == expected
        if react_match:
            react_correct += 1

        # Test LLM
        start = time.time()
        llm_result = llm_classifier.classify_intent(query)
        llm_time = time.time() - start
        llm_match = llm_result.intent == expected
        if llm_match:
            llm_correct += 1

        print(f"ReAct: {react_intent} {'✅' if react_match else '❌'}")
        print(f"LLM:   {llm_result.intent} {'✅' if llm_match else '❌'} ({llm_time:.2f}s, conf: {llm_result.confidence:.2f})")
        print()

    total = len(test_cases)
    print("📊 RESULTS:")
    print(f"ReAct Accuracy: {react_correct}/{total} ({react_correct/total*100:.1f}%)")
    print(f"LLM Accuracy:   {llm_correct}/{total} ({llm_correct/total*100:.1f}%)")

    if llm_correct > react_correct:
        print("\n🏆 LLM approach wins on accuracy!")
    elif react_correct > llm_correct:
        print("\n🏆 ReAct approach wins on accuracy!")
    else:
        print("\n🤝 Tie on accuracy!")

if __name__ == "__main__":
    quick_test()