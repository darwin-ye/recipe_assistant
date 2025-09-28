#!/usr/bin/env python3
"""Debug what recipes we have"""

from recipe_models import RecipeDatabase

def debug_recipes():
    db = RecipeDatabase()
    recipes = db.get_recent_recipes(10)

    print("Available recipes:")
    print("=" * 50)
    for i, recipe in enumerate(recipes, 1):
        print(f"{i}. {recipe.title}")
        print(f"   Main ingredients: {recipe.main_ingredients}")
        print(f"   Created: {recipe.created_at}")
        print()

if __name__ == "__main__":
    debug_recipes()