# test_simple_react.py - Comprehensive testing for simplified ReAct system

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_intent_detection():
    """Test intent detection with various inputs"""
    agent = SimpleReActAgent(llm)

    test_cases = [
        # CREATE RECIPE tests
        ("create a shrimp recipe", "create_recipe", {"ingredients": "shrimp"}),
        ("make a pasta dish with tomatoes", "create_recipe", {"ingredients": "tomatoes"}),
        ("generate a recipe using chicken and garlic", "create_recipe", {"ingredients": "chicken garlic"}),
        ("new recipe", "create_recipe", {"ingredients": ""}),
        ("give me a recipe for beef", "create_recipe", {"ingredients": "beef"}),
        ("provide a recipe", "create_recipe", {"ingredients": ""}),
        ("suggest a chicken recipe", "create_recipe", {"ingredients": "chicken"}),
        ("give me a salmon recipe", "create_recipe", {"ingredients": "salmon"}),
        ("give me an apple recipe", "create_recipe", {"ingredients": "apple"}),

        # SHOW RECENT tests
        ("show me recent recipes", "show_recent", {"limit": 5}),
        ("list my latest 10 recipes", "show_recent", {"limit": 10}),
        ("display newest recipes", "show_recent", {"limit": 5}),

        # GET DETAILS tests (historical recipes)
        ("show me the cooking steps for the recipe I made before", "get_details", {"recipe_name": ""}),
        ("how to cook the pasta I had earlier", "get_details", {"recipe_name": ""}),
        ("cooking directions for my previous recipe", "get_details", {"recipe_name": ""}),
        ("recipe I used to have", "get_details", {"recipe_name": ""}),
        ("give me a previous recipe", "get_details", {"recipe_name": ""}),
        ("give me the previous recipe", "get_details", {"recipe_name": ""}),
        ("show me last recipe", "get_details", {"recipe_name": ""}),
        ("the one before", "get_details", {"recipe_name": ""}),

        # SEARCH tests
        ("find chicken recipes", "search", {"query": "chicken"}),
        ("show me beef recipes", "search", {"query": "beef"}),

        # SCALE tests
        ("scale recipe for 8 people", "scale", {"servings": 8}),
        ("adjust recipe to 6 servings", "scale", {"servings": 6}),

        # NUMBERED RECIPE REFERENCE tests
        ("2", "get_recipe_by_number", {"recipe_number": 2}),
        ("5", "get_recipe_by_number", {"recipe_number": 5}),
        ("10", "get_recipe_by_number", {"recipe_number": 10}),

        # HELP/AMBIGUOUS tests
        ("help", "help", {}),
        ("yes", "help", {}),
        ("ok", "help", {}),
        ("", "help", {}),

        # DEFAULT to search
        ("chicken", "search", {"query": "chicken"}),
        ("random text", "search", {"query": "random text"}),
    ]

    print("🧪 Testing Intent Detection")
    print("=" * 50)

    passed = 0
    failed = 0

    for input_text, expected_intent, expected_params in test_cases:
        actual_intent, actual_params = agent.detect_intent(input_text)

        # Check intent
        intent_match = actual_intent == expected_intent

        # Check key parameters (flexible on exact values)
        params_match = True
        for key in expected_params:
            if key in actual_params:
                # For ingredients, check if key words are present (flexible matching)
                if key == "ingredients" and expected_params[key]:
                    expected_words = expected_params[key].lower().split()
                    actual_text = actual_params[key].lower()
                    params_match = all(word in actual_text for word in expected_words)
                # For other params, check exact match or reasonable default
                elif key == "limit":
                    params_match = actual_params[key] == expected_params[key]
                else:
                    params_match = True  # Be flexible on other params
            else:
                params_match = key in actual_params

        if intent_match and params_match:
            print(f"✅ '{input_text}' -> {actual_intent} {actual_params}")
            passed += 1
        else:
            print(f"❌ '{input_text}' -> Expected: {expected_intent} {expected_params}, Got: {actual_intent} {actual_params}")
            failed += 1

    print(f"\n📊 Intent Detection Results: {passed} passed, {failed} failed")
    return failed == 0

def test_edge_cases():
    """Test edge cases and error handling"""
    agent = SimpleReActAgent(llm)

    print("\n🧪 Testing Edge Cases")
    print("=" * 50)

    edge_cases = [
        # Empty and whitespace
        ("", "help"),
        ("   ", "help"),
        ("\n\t", "help"),

        # Very long input
        ("create a recipe with " + "very " * 100 + "long ingredients", "create_recipe"),

        # Mixed case
        ("CREATE A RECIPE WITH CHICKEN", "create_recipe"),
        ("Show Me Recent Recipes", "show_recent"),

        # Numbers in different formats
        ("show me 15 recent recipes", "show_recent"),
        ("scale to twenty servings", "scale"),

        # Typos and variations
        ("crete a recpie", "search"),  # Should fallback to search
        ("show recetn recipes", "search"),  # Should fallback to search
    ]

    passed = 0
    failed = 0

    for input_text, expected_intent in edge_cases:
        try:
            actual_intent, _ = agent.detect_intent(input_text)
            if actual_intent == expected_intent:
                print(f"✅ Edge case: '{input_text[:50]}...' -> {actual_intent}")
                passed += 1
            else:
                print(f"❌ Edge case: '{input_text[:50]}...' -> Expected: {expected_intent}, Got: {actual_intent}")
                failed += 1
        except Exception as e:
            print(f"❌ Edge case error: '{input_text[:50]}...' -> {str(e)}")
            failed += 1

    print(f"\n📊 Edge Case Results: {passed} passed, {failed} failed")
    return failed == 0

def test_execution_safety():
    """Test that execution methods don't crash"""
    agent = SimpleReActAgent(llm)

    print("\n🧪 Testing Execution Safety")
    print("=" * 50)

    test_executions = [
        ("help", {}),
        ("show_recent", {"limit": 5}),
        ("search", {"query": "chicken"}),
        ("create_recipe", {"ingredients": ""}),  # Should ask for ingredients
        ("get_details", {"recipe_name": ""}),    # Should ask for recipe name
        ("get_recipe_by_number", {"recipe_number": 2}),  # Should work if recipes exist
    ]

    passed = 0
    failed = 0

    for intent, params in test_executions:
        try:
            result = agent.execute_intent(intent, params)
            if isinstance(result, str) and len(result) > 0:
                print(f"✅ Execution: {intent} -> Response received ({len(result)} chars)")
                passed += 1
            else:
                print(f"❌ Execution: {intent} -> Empty or invalid response")
                failed += 1
        except Exception as e:
            print(f"❌ Execution error: {intent} -> {str(e)}")
            failed += 1

    print(f"\n📊 Execution Safety Results: {passed} passed, {failed} failed")
    return failed == 0

def run_comprehensive_test():
    """Run all tests"""
    print("🚀 Starting Comprehensive ReAct Testing")
    print("=" * 60)

    test1 = test_intent_detection()
    test2 = test_edge_cases()
    test3 = test_execution_safety()

    print("\n" + "=" * 60)
    print("📋 Final Results:")
    print(f"Intent Detection: {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"Edge Cases: {'✅ PASS' if test2 else '❌ FAIL'}")
    print(f"Execution Safety: {'✅ PASS' if test3 else '❌ FAIL'}")

    overall = test1 and test2 and test3
    print(f"\n🎯 Overall: {'✅ ALL TESTS PASSED' if overall else '❌ SOME TESTS FAILED'}")

    return overall

if __name__ == "__main__":
    run_comprehensive_test()