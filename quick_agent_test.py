#!/usr/bin/env python3
"""
Quick focused test to identify specific issues
"""

from llm import llm
from multi_action_react_agent import MultiActionReActAgent

def test_basic_functionality():
    """Test one simple case to check basic functionality"""

    print("🔧 QUICK AGENT TEST - Basic Functionality")
    print("=" * 50)

    agent = MultiActionReActAgent(llm)

    # Test the simplest 2-action case
    query = "Find my most frequent recipe and scale it to 8 people"

    print(f"Testing: {query}")
    print("-" * 30)

    try:
        result = agent.process_request(query)

        # Basic checks
        steps = result.count("**Step")
        has_success = "🎉 All requested actions have been completed successfully!" in result
        has_recipe_name = "Crisp Korean-Style Beef Tacos" in result or "recipe" in result.lower()

        print(f"✅ Steps completed: {steps}")
        print(f"✅ Success message: {has_success}")
        print(f"✅ Recipe mentioned: {has_recipe_name}")

        if steps >= 2 and has_success:
            print(f"\n🎉 BASIC FUNCTIONALITY WORKING!")
            return True
        else:
            print(f"\n⚠️ Issues detected:")
            if steps < 2:
                print(f"  • Expected 2+ steps, got {steps}")
            if not has_success:
                print(f"  • Missing success message")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_3_action_specific():
    """Test the specific 3-action query that was problematic"""

    print(f"\n🎯 TESTING SPECIFIC 3-ACTION QUERY")
    print("=" * 50)

    agent = MultiActionReActAgent(llm)

    query = "Find my most frequent recipe, display nutrient facts, and then scale it to serve 10 people"

    print(f"Testing: {query}")
    print("-" * 30)

    try:
        result = agent.process_request(query)

        # Check for improvements
        steps = result.count("**Step")
        has_recipe_details = "**Main Ingredients:**" in result
        has_nutrition = "nutritional" in result.lower() or "nutrient" in result.lower()
        has_scaling = "scaled" in result.lower() and "10" in result

        print(f"✅ Steps completed: {steps}")
        print(f"✅ Recipe details: {has_recipe_details}")
        print(f"✅ Nutrition info: {has_nutrition}")
        print(f"✅ Scaling to 10: {has_scaling}")

        if steps >= 3 and (has_recipe_details or has_nutrition):
            print(f"\n🎉 3-ACTION QUERY WORKING!")
            return True
        else:
            print(f"\n⚠️ Issues with 3-action:")
            if steps < 3:
                print(f"  • Expected 3+ steps, got {steps}")
            if not has_recipe_details:
                print(f"  • Missing detailed recipe info")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 QUICK AGENT TESTING")
    print("=" * 60)

    basic_works = test_basic_functionality()
    three_action_works = test_3_action_specific()

    print(f"\n📊 QUICK TEST SUMMARY:")
    print(f"Basic 2-action: {'✅ PASS' if basic_works else '❌ FAIL'}")
    print(f"Complex 3-action: {'✅ PASS' if three_action_works else '❌ FAIL'}")

    if basic_works and three_action_works:
        print(f"\n🎉 ALL TESTS PASSING! Agent is working well.")
    elif basic_works:
        print(f"\n⚠️ Basic functionality works, but 3-action needs improvement.")
    else:
        print(f"\n🔧 Need to fix basic functionality first.")

    print(f"\nReady to fix any identified issues!")