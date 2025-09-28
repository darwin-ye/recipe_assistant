#!/usr/bin/env python3
"""Remove corrupted recipes with invalid ingredients"""

from recipe_models import RecipeDatabase

def clean_corrupted_recipes():
    db = RecipeDatabase()

    print("Before cleanup:")
    print(f"Total recipes: {len(db.recipes)}")

    # Find recipes with suspicious ingredients
    corrupted_recipes = []
    for recipe_id, recipe in db.recipes.items():
        invalid_ingredients = []
        for ingredient in recipe.main_ingredients:
            # Check for words that are clearly not ingredients
            if ingredient.lower() in ['previous', 'last', 'recent', 'earlier', 'past', 'before']:
                invalid_ingredients.append(ingredient)

        if invalid_ingredients:
            print(f"\nFound corrupted recipe:")
            print(f"  ID: {recipe_id}")
            print(f"  Title: {recipe.title}")
            print(f"  Invalid ingredients: {invalid_ingredients}")
            corrupted_recipes.append(recipe_id)

    if corrupted_recipes:
        print(f"\nRemoving {len(corrupted_recipes)} corrupted recipes...")
        for recipe_id in corrupted_recipes:
            del db.recipes[recipe_id]

        # Save the cleaned database
        db.save_recipes()
        print("✅ Database cleaned and saved")
    else:
        print("✅ No corrupted recipes found")

    print(f"\nAfter cleanup:")
    print(f"Total recipes: {len(db.recipes)}")

if __name__ == "__main__":
    clean_corrupted_recipes()