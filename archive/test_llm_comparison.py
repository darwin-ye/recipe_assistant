#!/usr/bin/env python3
"""
Test and compare the different recipe assistant approaches:
1. ReAct (Hybrid rule-based + LLM)
2. Pure LLM approach
3. Performance and accuracy comparison
"""

import time
from typing import List, Dict, Any

from llm import llm
from simple_react_agent import SimpleReActAgent  # ReAct approach
from llm_agent import LLMRecipeAgent            # Pure LLM approach
from llm_intent_classifier import LLMIntentClassifier


def test_intent_classification_comparison():
    """Compare intent classification between ReAct and LLM approaches"""

    print("🧪 Intent Classification Comparison")
    print("=" * 60)

    # Initialize agents
    react_agent = SimpleReActAgent(llm)
    llm_classifier = LLMIntentClassifier(llm)

    # Test queries with expected intents
    test_cases = [
        ("create a chicken pasta recipe", "create_recipe"),
        ("show me recent recipes", "get_recent"),
        ("find pasta recipes", "search"),
        ("what's my most frequent recipe?", "analytics_frequent"),
        ("how many chicken recipes do I have?", "analytics_count"),
        ("scale this to 8 people", "scale_recipe"),
        ("2", "numbered_reference"),
        ("give me the salmon recipe", "get_details"),
        ("show me the recipe I always have", "analytics_frequent"),
        ("help", "help"),
        # Edge cases
        ("what do I cook most often", "analytics_frequent"),
        ("my go-to recipe", "analytics_frequent"),
        ("recipe I make all the time", "analytics_frequent"),
        ("count recipes with beef", "analytics_count"),
        ("make it for 6 people", "scale_recipe")
    ]

    react_correct = 0
    llm_correct = 0
    react_times = []
    llm_times = []

    print(f"Testing {len(test_cases)} queries...\n")

    for query, expected_intent in test_cases:
        print(f"Query: '{query}'")
        print(f"Expected: {expected_intent}")

        # Test ReAct approach
        start_time = time.time()
        react_intent, react_params = react_agent.detect_intent(query)
        react_time = time.time() - start_time
        react_times.append(react_time)

        # Test LLM approach
        start_time = time.time()
        llm_result = llm_classifier.classify_intent(query)
        llm_time = time.time() - start_time
        llm_times.append(llm_time)

        # Check accuracy
        react_match = react_intent == expected_intent
        llm_match = llm_result.intent == expected_intent

        if react_match:
            react_correct += 1
        if llm_match:
            llm_correct += 1

        print(f"ReAct: {react_intent} {'✅' if react_match else '❌'} ({react_time:.3f}s)")
        print(f"LLM:   {llm_result.intent} {'✅' if llm_match else '❌'} ({llm_time:.3f}s, conf: {llm_result.confidence:.2f})")
        print()

    # Summary
    total_tests = len(test_cases)
    react_accuracy = (react_correct / total_tests) * 100
    llm_accuracy = (llm_correct / total_tests) * 100
    avg_react_time = sum(react_times) / len(react_times)
    avg_llm_time = sum(llm_times) / len(llm_times)

    print("📊 RESULTS SUMMARY")
    print("=" * 40)
    print(f"ReAct Accuracy:  {react_correct}/{total_tests} ({react_accuracy:.1f}%)")
    print(f"LLM Accuracy:    {llm_correct}/{total_tests} ({llm_accuracy:.1f}%)")
    print(f"ReAct Avg Time:  {avg_react_time:.3f}s")
    print(f"LLM Avg Time:    {avg_llm_time:.3f}s")
    print()

    if llm_accuracy > react_accuracy:
        print("🏆 LLM approach wins on accuracy!")
    elif react_accuracy > llm_accuracy:
        print("🏆 ReAct approach wins on accuracy!")
    else:
        print("🤝 Tie on accuracy!")

    if avg_llm_time < avg_react_time:
        print("⚡ LLM approach is faster!")
    else:
        print("⚡ ReAct approach is faster!")


def test_natural_language_flexibility():
    """Test how well each approach handles natural language variations"""

    print("\n🗣️ Natural Language Flexibility Test")
    print("=" * 60)

    # Initialize agents
    react_agent = SimpleReActAgent(llm)
    llm_classifier = LLMIntentClassifier(llm)

    # Variations of the same intent
    test_groups = {
        "Most Frequent Recipe": [
            "show me the recipe I always have",
            "what's my go-to recipe?",
            "my most frequent recipe",
            "what do I cook most often",
            "my signature dish",
            "recipe I make all the time",
            "what's my usual recipe"
        ],
        "Create Recipe": [
            "create a chicken pasta recipe",
            "make something with beef",
            "give me a salmon recipe",
            "cook up something with vegetables",
            "I want to make a new recipe",
            "help me create a dish with tomatoes"
        ],
        "Recipe Counting": [
            "how many chicken recipes do I have?",
            "count recipes with beef",
            "how often do I use tomatoes?",
            "frequency of pasta in my recipes",
            "how much do I cook with salmon?"
        ]
    }

    for group_name, variations in test_groups.items():
        print(f"\n📋 {group_name} Variations:")
        print("-" * 40)

        react_consistent = 0
        llm_consistent = 0
        first_react_intent = None
        first_llm_intent = None

        for i, query in enumerate(variations):
            # Test both approaches
            react_intent, _ = react_agent.detect_intent(query)
            llm_result = llm_classifier.classify_intent(query)

            if i == 0:
                first_react_intent = react_intent
                first_llm_intent = llm_result.intent

            # Check consistency
            react_match = react_intent == first_react_intent
            llm_match = llm_result.intent == first_llm_intent

            if react_match:
                react_consistent += 1
            if llm_match:
                llm_consistent += 1

            print(f"'{query}'")
            print(f"  ReAct: {react_intent} {'✅' if react_match else '❌'}")
            print(f"  LLM:   {llm_result.intent} {'✅' if llm_match else '❌'} (conf: {llm_result.confidence:.2f})")

        total_variations = len(variations)
        react_consistency = (react_consistent / total_variations) * 100
        llm_consistency = (llm_consistent / total_variations) * 100

        print(f"\nConsistency: ReAct {react_consistency:.1f}% | LLM {llm_consistency:.1f}%")


def test_edge_cases():
    """Test challenging edge cases and ambiguous queries"""

    print("\n🎯 Edge Cases Test")
    print("=" * 60)

    # Initialize agents
    react_agent = SimpleReActAgent(llm)
    llm_classifier = LLMIntentClassifier(llm)

    edge_cases = [
        "show me my cooking",  # Ambiguous
        "something with chicken",  # Incomplete
        "recipe",  # Too vague
        "make it bigger",  # Missing context
        "what's good?",  # Very vague
        "I'm hungry",  # Implicit request
        "pasta pasta pasta",  # Repetitive
        "chicken beef salmon",  # Multiple ingredients
        "create find show",  # Mixed commands
        "123abc",  # Invalid number
    ]

    print("Testing ambiguous and challenging queries:")
    print()

    for query in edge_cases:
        react_intent, _ = react_agent.detect_intent(query)
        llm_result = llm_classifier.classify_intent(query)

        print(f"'{query}'")
        print(f"  ReAct: {react_intent}")
        print(f"  LLM:   {llm_result.intent} (conf: {llm_result.confidence:.2f})")
        print(f"  LLM Reasoning: {llm_result.reasoning}")
        print()


def benchmark_performance():
    """Benchmark performance with repeated queries"""

    print("\n⚡ Performance Benchmark")
    print("=" * 60)

    # Initialize agents
    react_agent = SimpleReActAgent(llm)
    llm_classifier = LLMIntentClassifier(llm)

    # Common queries for benchmarking
    benchmark_queries = [
        "create a chicken recipe",
        "show recent recipes",
        "what's my frequent recipe",
        "find pasta dishes",
        "2"
    ]

    iterations = 5  # Number of times to run each query

    print(f"Running {len(benchmark_queries)} queries {iterations} times each...\n")

    react_total_time = 0
    llm_total_time = 0

    for query in benchmark_queries:
        print(f"Benchmarking: '{query}'")

        # Benchmark ReAct
        react_times = []
        for _ in range(iterations):
            start_time = time.time()
            react_agent.detect_intent(query)
            react_times.append(time.time() - start_time)

        # Benchmark LLM
        llm_times = []
        for _ in range(iterations):
            start_time = time.time()
            llm_classifier.classify_intent(query)
            llm_times.append(time.time() - start_time)

        react_avg = sum(react_times) / len(react_times)
        llm_avg = sum(llm_times) / len(llm_times)

        react_total_time += react_avg
        llm_total_time += llm_avg

        print(f"  ReAct: {react_avg:.3f}s avg")
        print(f"  LLM:   {llm_avg:.3f}s avg")

        if llm_avg < react_avg:
            print("  🏆 LLM wins")
        else:
            print("  🏆 ReAct wins")
        print()

    print("📊 Overall Performance:")
    print(f"ReAct Total: {react_total_time:.3f}s")
    print(f"LLM Total:   {llm_total_time:.3f}s")

    if llm_total_time < react_total_time:
        speedup = (react_total_time - llm_total_time) / react_total_time * 100
        print(f"🚀 LLM is {speedup:.1f}% faster overall!")
    else:
        slowdown = (llm_total_time - react_total_time) / react_total_time * 100
        print(f"🐌 LLM is {slowdown:.1f}% slower overall")


def main():
    """Run all comparison tests"""

    print("🚀 Recipe Assistant Approach Comparison")
    print("=" * 80)
    print("Comparing ReAct (hybrid) vs Pure LLM approaches")
    print("=" * 80)

    try:
        # Run all tests
        test_intent_classification_comparison()
        test_natural_language_flexibility()
        test_edge_cases()
        benchmark_performance()

        print("\n🎉 Comparison complete!")
        print("\n💡 Recommendations:")
        print("- LLM approach: Better for natural language flexibility and accuracy")
        print("- ReAct approach: Faster for simple patterns, more predictable")
        print("- Consider using LLM approach for production due to better user experience")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()