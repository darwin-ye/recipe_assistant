#!/usr/bin/env python3
"""
Comprehensive test suite for the structured JSON multi-action ReAct agent
"""

from llm import llm
from multi_action_react_agent import MultiActionReActAgent

def test_2_action_queries():
    """Test 2-action queries that should work perfectly"""

    print("🎯 TESTING 2-ACTION QUERIES")
    print("=" * 60)

    agent = MultiActionReActAgent(llm)

    test_cases = [
        "Find my most frequent recipe and scale it to 8 people",
        "Create a pasta recipe then get its details",
        "Search for chicken recipes and count how many there are",
        "Get recent recipes then scale the first one to 6 servings"
    ]

    results = []

    for i, query in enumerate(test_cases, 1):
        print(f"\n📝 Test 2.{i}: {query}")
        print("-" * 40)

        try:
            result = agent.process_request(query)

            # Check success indicators
            steps_completed = result.count("**Step")
            has_success_msg = "🎉 All requested actions have been completed successfully!" in result
            has_details = len(result) > 500  # Substantial output

            success = steps_completed >= 2 and has_success_msg
            results.append({
                "query": query,
                "success": success,
                "steps": steps_completed,
                "detailed": has_details,
                "issues": []
            })

            print(f"✅ Steps: {steps_completed} | Success: {success} | Detailed: {has_details}")

        except Exception as e:
            results.append({
                "query": query,
                "success": False,
                "steps": 0,
                "detailed": False,
                "issues": [str(e)]
            })
            print(f"❌ Error: {e}")

    return results

def test_3_action_queries():
    """Test 3-action queries - the main improvement"""

    print("\n🎯 TESTING 3-ACTION QUERIES (KEY IMPROVEMENT)")
    print("=" * 60)

    agent = MultiActionReActAgent(llm)

    test_cases = [
        "Find my most frequent recipe, display nutrient facts, and then scale it to serve 10 people",
        "Create a beef recipe, get its details, then scale it to 8 servings",
        "Search for pasta recipes, get details of the first one, then count ingredients",
        "Get recent recipes, show details of the latest one, then scale it to 4 people"
    ]

    results = []

    for i, query in enumerate(test_cases, 1):
        print(f"\n📝 Test 3.{i}: {query}")
        print("-" * 40)

        try:
            result = agent.process_request(query)

            # Check for 3-action completion
            steps_completed = result.count("**Step")
            has_success_msg = "🎉 All requested actions have been completed successfully!" in result
            has_recipe_details = "**Main Ingredients:**" in result or "**" in result
            has_scaling_details = "servings" in result.lower() and ("×" in result or "scaled" in result.lower())

            success = steps_completed >= 3 and has_success_msg
            results.append({
                "query": query,
                "success": success,
                "steps": steps_completed,
                "has_details": has_recipe_details,
                "has_scaling": has_scaling_details,
                "issues": []
            })

            print(f"✅ Steps: {steps_completed} | Success: {success}")
            print(f"   Recipe Details: {has_recipe_details} | Scaling: {has_scaling_details}")

        except Exception as e:
            results.append({
                "query": query,
                "success": False,
                "steps": 0,
                "has_details": False,
                "has_scaling": False,
                "issues": [str(e)]
            })
            print(f"❌ Error: {e}")

    return results

def test_4_action_stress_tests():
    """Test 4+ action queries for stress testing"""

    print("\n🎯 TESTING 4-ACTION STRESS TESTS")
    print("=" * 60)

    agent = MultiActionReActAgent(llm)

    test_cases = [
        "Search for chicken recipes, get details of the first one, create a similar recipe, then scale it to 12 people",
        "Get recent recipes, count those with tomatoes, find my most frequent recipe, then scale it to 15 people"
    ]

    results = []

    for i, query in enumerate(test_cases, 1):
        print(f"\n📝 Stress Test {i}: {query}")
        print("-" * 40)

        try:
            result = agent.process_request(query)

            steps_completed = result.count("**Step")
            has_success_msg = "🎉 All requested actions have been completed successfully!" in result

            success = steps_completed >= 4 and has_success_msg
            results.append({
                "query": query,
                "success": success,
                "steps": steps_completed,
                "issues": []
            })

            print(f"✅ Steps: {steps_completed} | Success: {success}")

        except Exception as e:
            results.append({
                "query": query,
                "success": False,
                "steps": 0,
                "issues": [str(e)]
            })
            print(f"❌ Error: {e}")

    return results

def test_parameter_extraction():
    """Test parameter extraction capabilities"""

    print("\n🎯 TESTING PARAMETER EXTRACTION")
    print("=" * 60)

    agent = MultiActionReActAgent(llm)

    test_cases = [
        ("Create a chicken and tomato recipe then scale it to 6 servings", ["chicken", "tomato"], "6"),
        ("Find recipes with beef, get details, then scale to 8 people", ["beef"], "8"),
        ("Make a pasta dish with garlic and scale to 12 servings", ["pasta", "garlic"], "12")
    ]

    results = []

    for i, (query, expected_ingredients, expected_servings) in enumerate(test_cases, 1):
        print(f"\n📝 Param Test {i}: {query}")
        print(f"   Expected: {expected_ingredients}, {expected_servings} servings")
        print("-" * 40)

        try:
            result = agent.process_request(query)

            # Check if ingredients and servings appear in result
            ingredients_found = any(ing.lower() in result.lower() for ing in expected_ingredients)
            servings_found = expected_servings in result

            success = ingredients_found and servings_found
            results.append({
                "query": query,
                "success": success,
                "ingredients_found": ingredients_found,
                "servings_found": servings_found,
                "issues": []
            })

            print(f"✅ Ingredients: {ingredients_found} | Servings: {servings_found}")

        except Exception as e:
            results.append({
                "query": query,
                "success": False,
                "ingredients_found": False,
                "servings_found": False,
                "issues": [str(e)]
            })
            print(f"❌ Error: {e}")

    return results

def run_comprehensive_tests():
    """Run all tests and provide summary"""

    print("🚀 COMPREHENSIVE STRUCTURED JSON MULTI-ACTION AGENT TESTING")
    print("=" * 80)

    # Run all test categories
    results_2_action = test_2_action_queries()
    results_3_action = test_3_action_queries()
    results_4_action = test_4_action_stress_tests()
    results_params = test_parameter_extraction()

    # Calculate overall statistics
    all_results = results_2_action + results_3_action + results_4_action + results_params
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r["success"])
    success_rate = (successful_tests / total_tests) * 100

    print(f"\n📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Success Rate: {success_rate:.1f}%")

    # Category breakdown
    print(f"\n📋 Category Breakdown:")
    print(f"2-Action Tests: {sum(1 for r in results_2_action if r['success'])}/{len(results_2_action)}")
    print(f"3-Action Tests: {sum(1 for r in results_3_action if r['success'])}/{len(results_3_action)}")
    print(f"4-Action Tests: {sum(1 for r in results_4_action if r['success'])}/{len(results_4_action)}")
    print(f"Parameter Tests: {sum(1 for r in results_params if r['success'])}/{len(results_params)}")

    # Identify issues for fixing
    issues_found = []
    for result in all_results:
        if not result["success"]:
            issues_found.extend(result["issues"])

    if issues_found:
        print(f"\n⚠️ Issues Found (need fixing):")
        for issue in set(issues_found):
            print(f"  • {issue}")
    else:
        print(f"\n🎉 No critical issues found!")

    return {
        "success_rate": success_rate,
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "issues": issues_found,
        "detailed_results": {
            "2_action": results_2_action,
            "3_action": results_3_action,
            "4_action": results_4_action,
            "parameters": results_params
        }
    }

if __name__ == "__main__":
    test_results = run_comprehensive_tests()

    if test_results["success_rate"] >= 80:
        print(f"\n🎉 EXCELLENT! Agent performing at {test_results['success_rate']:.1f}%")
    elif test_results["success_rate"] >= 60:
        print(f"\n✅ GOOD! Agent performing at {test_results['success_rate']:.1f}%")
    else:
        print(f"\n⚠️ NEEDS IMPROVEMENT! Agent at {test_results['success_rate']:.1f}%")

    print("\n🔧 Ready to fix any identified issues!")