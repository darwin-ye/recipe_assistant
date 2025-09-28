#!/usr/bin/env python3
"""Test the get_recipe_details function to find the bug"""

from react_agent import get_recipe_details
from recipe_models import RecipeDatabase

def test_recipe_details_bug():
    db = RecipeDatabase()

    # Test the exact recipe name we expect
    recipe_name = "Pan-Seared Salmon Cakes with Lemon-Dill Sauce"
    print(f"Testing get_recipe_details with: '{recipe_name}'")
    print("=" * 50)

    # Manually check what the function is doing
    print("Manual search through recipes:")
    for recipe_id, recipe in db.recipes.items():
        match = recipe_name.lower() in recipe.title.lower()
        print(f"Recipe: '{recipe.title}' | Match: {match}")
        if match:
            print(f"  -> This would be returned first!")
            break

    print("\nActual function result:")
    result = get_recipe_details(recipe_name)
    # Just print the first few lines to see which recipe it returned
    lines = result.split('\n')
    print(f"Returned recipe title: {lines[0] if lines else 'No result'}")

if __name__ == "__main__":
    test_recipe_details_bug()