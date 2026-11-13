#!/usr/bin/env python3
"""
Test just the create recipe functionality
"""

from llm import llm
from multi_action_react_agent import MultiActionReActAgent

def test_create_recipe_only():
    """Test just the _create_recipe method directly"""

    print("🔧 TESTING CREATE RECIPE ONLY")
    print("=" * 40)

    agent = MultiActionReActAgent(llm)

    # Test create recipe directly
    ingredients = ["pasta", "tomatoes"]
    recipe_type = "Italian"

    print(f"Testing create_recipe with {ingredients}")
    print("-" * 30)

    try:
        result = agent._create_recipe(ingredients, recipe_type)
        print(f"✅ Create recipe result: {result[:200]}...")

    except Exception as e:
        print(f"❌ Create recipe error: {e}")
        import traceback
        traceback.print_exc()

def test_create_recipe_query():
    """Test create recipe through normal query"""

    print("\n🔧 TESTING CREATE RECIPE QUERY")
    print("=" * 40)

    agent = MultiActionReActAgent(llm)

    # Simple create recipe query
    query = "Create a pasta recipe"

    print(f"Testing: {query}")
    print("-" * 30)

    try:
        result = agent.process_request(query)
        print(f"✅ Query result: {result[:200]}...")

    except Exception as e:
        print(f"❌ Query error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_create_recipe_only()
    test_create_recipe_query()