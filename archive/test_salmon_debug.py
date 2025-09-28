#!/usr/bin/env python3
"""Debug the salmon recipe issue specifically"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_salmon_issue():
    agent = SimpleReActAgent(llm)

    # Test the exact user input
    user_input = "give me the previous salmon recipe"
    print(f"Testing input: '{user_input}'")
    print("=" * 50)

    # Check intent detection
    intent, params = agent.detect_intent(user_input)
    print(f"Intent: {intent}")
    print(f"Params: {params}")
    print(f"Recipe name extracted: '{params.get('recipe_name', '')}'")

    # Test the recipe name extraction manually
    recipe_name = agent._extract_recipe_name(user_input)
    print(f"Manual recipe name extraction: '{recipe_name}'")

    # Check what recipes are available
    recipes = agent.db.get_recent_recipes(10)
    print(f"\nAvailable recipes ({len(recipes)}):")
    for i, recipe in enumerate(recipes, 1):
        print(f"{i}. {recipe.title}")
        print(f"   Main ingredients: {recipe.main_ingredients}")

        # Check if salmon matches
        salmon_in_title = "salmon" in recipe.title.lower()
        salmon_in_ingredients = "salmon" in [ing.lower() for ing in recipe.main_ingredients]
        print(f"   Salmon in title: {salmon_in_title}, in ingredients: {salmon_in_ingredients}")
        print()

if __name__ == "__main__":
    test_salmon_issue()