# simple_react_agent.py - Simplified, Reliable ReAct Implementation
"""
A much simpler ReAct implementation that focuses on reliability over complexity.
Uses deterministic intent detection and clear fallback patterns.
"""

import re
from typing import List, Any, Optional, Tuple
from recipe_models import RecipeDatabase

class SimpleReActAgent:
    """Simplified ReAct agent with deterministic intent detection"""

    def __init__(self, llm):
        self.llm = llm
        self.db = RecipeDatabase()
        self.current_recipe = None

    def detect_intent(self, user_input: str) -> Tuple[str, dict]:
        """Deterministic intent detection with clear rules"""
        text = user_input.lower().strip()
        import re

        # Intent 1: ANALYTICS - MOST FREQUENT RECIPE (highest priority for analytics)
        frequent_keywords = ["most frequent", "most common", "most popular", "frequently used", "most used"]

        # Enhanced patterns for "recipe I always have/make/use"
        always_patterns = [
            r"recipe.*?(?:i|we).*?always.*?(?:have|make|use|cook)",
            r"(?:i|we).*?always.*?(?:have|make|use|cook).*?recipe",
            r"recipe.*?(?:i|we).*?always",
            r"always.*?(?:making|cooking|using)(?!.*?new|.*?with)",  # Avoid conflicts with creation
            r"go[- ]?to recipe",
            r"favorite recipe",
            r"usual recipe",
            r"regular recipe",
            r"staple recipe",
            r"signature recipe",
            r"what.*?(?:i|we).*?usually.*?(?:cook|make)",
            r"what.*?recipe.*?(?:i|we).*?always",
            r"(?:recipe|dish).*?(?:i|we).*?always.*?(?:make|have|use)"
        ]

        # Check for enhanced patterns
        if (any(keyword in text for keyword in frequent_keywords) or
            any(re.search(pattern, text, re.IGNORECASE) for pattern in always_patterns)):
            return "analytics_frequent", {}

        # Intent 2: ANALYTICS - COUNT RECIPES BY INGREDIENT
        count_keywords = ["count", "how many"]
        ingredient_keywords = ["recipes with", "recipes containing", "recipes that have", "with"]

        # Enhanced patterns for counting
        count_patterns = [
            r"how many (\w+) recipes",
            r"count.*?(\w+).*?recipes",
            r"how often.*?(?:use|cook|make).*?(\w+)",
            r"how much.*?(\w+).*?(?:recipes|cooking)",
            r"frequency.*?of.*?(\w+)"
        ]

        count_pattern_match = any(re.search(pattern, text, re.IGNORECASE) for pattern in count_patterns)

        if (any(keyword in text for keyword in count_keywords) and
            (any(keyword in text for keyword in ingredient_keywords) or count_pattern_match)):
            # Extract the ingredient to count
            ingredient = self._extract_count_ingredient(text)
            return "analytics_count", {"ingredient": ingredient}

        # Intent 3: CREATE RECIPE (moved after analytics)
        create_keywords = ["create", "make", "new recipe", "generate", "cook up", "come up with"]
        provide_keywords = ["give me a recipe", "give me recipe", "provide a recipe", "suggest a recipe", "recommend a recipe", "suggest a"]

        # Check for historical reference patterns first (higher priority than creation)
        give_me_history_pattern = r"give me (?:a |an |the )?(?:previous|last|recent|earlier|past|before)"
        give_me_history_match = re.search(give_me_history_pattern, text)

        # Also check for "give me a [ingredient] recipe" pattern
        give_me_pattern = r"give me (?:a |an )?(\w+) recipe"
        give_me_match = re.search(give_me_pattern, text)

        # If it's a historical reference, don't treat as recipe creation
        if give_me_history_match:
            # This will be handled later in the get_details intent
            pass
        elif (any(keyword in text for keyword in create_keywords) or
              any(keyword in text for keyword in provide_keywords) or
              give_me_match):
            # Extract ingredients if provided
            ingredients = self._extract_ingredients(text)
            # If we matched the "give me a [ingredient] recipe" pattern, use that ingredient
            if give_me_match and not ingredients:
                ingredients = give_me_match.group(1)
            return "create_recipe", {"ingredients": ingredients}

        # Intent 4: SHOW RECENT RECIPES
        recent_keywords = ["recent", "latest", "last", "newest"]
        show_keywords = ["show", "list", "display", "see"]
        # Check for "show me last recipe" (singular) - should be treated as historical reference
        show_last_singular = "show me last recipe" in text or "show last recipe" in text
        # Don't conflict with historical references - need both recent and show keywords
        if (any(r in text for r in recent_keywords) and any(s in text for s in show_keywords) and
            not give_me_history_match and not show_last_singular):
            # Extract number if provided
            limit = self._extract_number(text, default=5)
            return "show_recent", {"limit": limit}

        # Intent 5: SCALE RECIPE (moved up for higher priority)
        scale_keywords = ["scale", "adjust", "resize", "bigger", "smaller", "servings"]
        people_keywords = ["people", "person", "persons"]

        if any(keyword in text for keyword in scale_keywords) or any(keyword in text for keyword in people_keywords):
            # Look for number before "people" or "servings"
            people_match = re.search(r'(\d+)\s*(?:people|person|persons|servings)', text)
            if people_match:
                servings = int(people_match.group(1))
                return "scale", {"servings": servings}
            else:
                servings = self._extract_number(text, default=4)
                return "scale", {"servings": servings}

        # Intent 6: NUMBERED RECIPE REFERENCE (from recent list)
        # Check if input is just a number (1-10) - likely referring to recent recipes list
        if text.strip().isdigit() and 1 <= int(text.strip()) <= 10:
            recipe_number = int(text.strip())
            return "get_recipe_by_number", {"recipe_number": recipe_number}

        # Intent 7: GET RECIPE DETAILS/INSTRUCTIONS (check first for higher priority)
        detail_keywords = ["steps", "instructions", "how to cook", "cooking directions", "directions"]
        history_keywords = ["previous", "before", "past", "earlier", "used to have", "had before", "made before", "cooked before"]

        # Use the already defined give_me_history_match from above
        # Also check for "show me last recipe" pattern
        show_last_pattern = show_last_singular

        # Check for cooking instructions or historical recipe requests
        if (any(keyword in text for keyword in detail_keywords) or
            any(keyword in text for keyword in history_keywords) or
            give_me_history_match or show_last_pattern):
            recipe_name = self._extract_recipe_name(text)
            return "get_details", {"recipe_name": recipe_name}

        # Intent 8: SEARCH RECIPES (be more selective to allow LLM fallback)
        search_keywords = ["find", "search", "look for"]
        explicit_search = any(keyword in text for keyword in search_keywords)

        # Only use search if it's clearly a search query, not analytics
        show_me_analytics = ("show me" in text and
                           any(word in text for word in ["most", "frequent", "often", "always", "usually"]))

        if explicit_search and not show_me_analytics:
            query = self._extract_search_query(text)
            return "search", {"query": query}

        # Intent 9: AMBIGUOUS/HELP
        ambiguous = ["yes", "ok", "okay", "help", "what can you do", ""]
        if text in ambiguous:
            return "help", {}

        # LLM-powered fallback for complex queries
        llm_intent = self._classify_intent_with_llm(text)
        if llm_intent:
            return llm_intent

        # Default: treat as search
        return "search", {"query": text}

    def _extract_ingredients(self, text: str) -> str:
        """Extract ingredients from text"""
        # Look for patterns like "with X", "using X", "for X"
        patterns = [
            r"with\s+(.+?)(?:\s+recipe|$)",
            r"using\s+(.+?)(?:\s+recipe|$)",
            r"from\s+(.+?)(?:\s+recipe|$)",
            r"recipe\s+for\s+(.+?)(?:\s|$)",
            r"for\s+(.+?)(?:\s+recipe|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                ingredients_text = match.group(1).strip()
                # Clean up common non-ingredient words
                ingredients_text = re.sub(r'\b(and|or|also|plus)\b', ',', ingredients_text)
                return ingredients_text

        # Look for common ingredients mentioned
        common_ingredients = [
            "chicken", "beef", "pork", "fish", "salmon", "shrimp", "pasta",
            "rice", "tomato", "tomatoes", "onion", "garlic", "cheese", "eggs"
        ]

        found_ingredients = [ing for ing in common_ingredients if ing in text]
        if found_ingredients:
            return ", ".join(found_ingredients)

        return ""

    def _extract_number(self, text: str, default: int = 5) -> int:
        """Extract number from text"""
        numbers = re.findall(r'\b(\d+)\b', text)
        if numbers:
            return int(numbers[0])
        return default

    def _extract_recipe_name(self, text: str) -> str:
        """Extract recipe name from text"""
        # Get recent recipes to match against
        recent_recipes = self.db.get_recent_recipes(10)

        # Check for historical reference keywords that mean "most recent"
        historical_refs = ["previous", "last", "recent", "earlier", "past", "before", "the one before"]

        # First, check if this is a specific ingredient request with historical reference
        # Pattern: "previous/last [ingredient] recipe" or "show me the last [ingredient] recipe"
        for ref in historical_refs:
            pattern = rf"{ref}\s+(\w+)\s+recipe"
            match = re.search(pattern, text.lower())
            if match:
                ingredient = match.group(1)
                # Find the most recent recipe containing this ingredient
                for recipe in recent_recipes:
                    if ingredient.lower() in recipe.title.lower() or ingredient.lower() in [ing.lower() for ing in recipe.main_ingredients]:
                        return recipe.title
                # If no match found, continue with general logic

        # Also check for "show me the [historical] [ingredient] recipe" pattern
        show_pattern = r"show.*?(?:the\s+)?(previous|last|recent|earlier|past)\s+(\w+)\s+recipe"
        show_match = re.search(show_pattern, text.lower())
        if show_match:
            ingredient = show_match.group(2)
            # Find the most recent recipe containing this ingredient
            for recipe in recent_recipes:
                if ingredient.lower() in recipe.title.lower() or ingredient.lower() in [ing.lower() for ing in recipe.main_ingredients]:
                    return recipe.title

        # Look for specific recipe names mentioned in text
        for recipe in recent_recipes:
            if recipe.title.lower() in text:
                return recipe.title

        # Check for pure historical reference (no specific ingredient)
        if any(ref in text.lower() for ref in historical_refs):
            if recent_recipes:
                return recent_recipes[0].title
            return ""

        # Look for "for X" pattern
        for_match = re.search(r"for\s+(.+?)(?:\s|$)", text)
        if for_match:
            return for_match.group(1).strip()

        # Return most recent recipe as fallback
        if recent_recipes:
            return recent_recipes[0].title

        return ""

    def _extract_search_query(self, text: str) -> str:
        """Extract search query from text"""
        # Remove common command words
        clean_text = text
        remove_words = ["show me", "find", "search", "look for", "recipes", "recipe"]
        for word in remove_words:
            clean_text = clean_text.replace(word, "").strip()

        return clean_text if clean_text else "chicken"

    def _extract_count_ingredient(self, text: str) -> str:
        """Extract ingredient from count queries"""
        # Enhanced patterns for counting recipes
        patterns = [
            r"count.*?recipes with (\w+)",
            r"how many.*?recipes.*?with (\w+)",
            r"how many.*?(\w+) recipes",
            r"count.*?(\w+) recipes",
            r"recipes.*?containing (\w+)",
            r"recipes.*?that have (\w+)",
            r"how often.*?(?:use|cook|make).*?(\w+)",
            r"how much.*?(\w+).*?(?:recipes|cooking)",
            r"frequency.*?of.*?(\w+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                ingredient = match.group(1)
                # Clean up common non-ingredient words
                if ingredient not in ["recipes", "cooking", "often", "much", "many"]:
                    return ingredient

        # Look for common ingredients mentioned anywhere in text
        common_ingredients = [
            "chicken", "beef", "pork", "fish", "salmon", "shrimp", "pasta",
            "rice", "tomato", "tomatoes", "onion", "garlic", "cheese", "eggs",
            "lamb", "turkey", "duck", "vegetables", "carrots", "potatoes",
            "mushrooms", "peppers", "spinach", "broccoli", "beans", "lentils"
        ]

        # Find ingredients with word boundaries to avoid partial matches
        for ingredient in common_ingredients:
            if re.search(rf"\b{ingredient}\b", text.lower()):
                return ingredient

        return ""

    def _classify_intent_with_llm(self, text: str):
        """Use LLM to classify intent for queries that don't match rule-based patterns"""
        try:
            prompt = f"""You are an intent classifier for a recipe assistant. Analyze this user query and determine the most likely intent.

User query: "{text}"

Available intents:
1. analytics_frequent - User wants to see their most frequent/common/favorite recipe
   Examples: "recipe I have the most", "what do I cook most often", "my most made dish"

2. analytics_count - User wants to count recipes by ingredient
   Examples: "how often do I use chicken", "frequency of beef recipes"

3. create_recipe - User wants to create a new recipe
   Examples: "make a new dish", "create something with chicken"

4. get_details - User wants details of a specific recipe
   Examples: "show me instructions for pasta dish"

5. show_recent - User wants to see recent recipes
   Examples: "my latest recipes", "recent dishes I made"

IMPORTANT: Only respond if you're confident (>80%) that the query matches analytics_frequent or analytics_count. For other intents or unclear queries, respond with "UNKNOWN".

For analytics_frequent patterns, look for concepts like:
- Most frequent/common/popular recipe
- Recipe they have/make/cook the most
- Favorite/go-to/signature recipe
- What they usually/typically cook

For analytics_count patterns, look for:
- Counting/frequency questions about ingredients
- "How often/much do I use X"
- "How many times do I cook with X"

Respond with EXACTLY one of:
- analytics_frequent
- analytics_count:[ingredient]
- UNKNOWN

Response:"""

            response = self.llm.invoke(prompt).content.strip().lower()

            # Look for intent keywords in the response
            if "analytics_frequent" in response:
                return "analytics_frequent", {}
            elif "analytics_count:" in response:
                # Extract ingredient after analytics_count:
                lines = response.split('\n')
                for line in lines:
                    if "analytics_count:" in line:
                        ingredient = line.split("analytics_count:", 1)[1].strip()
                        # Clean up any extra text after the ingredient name
                        ingredient = ingredient.split()[0] if ingredient else ""
                        return "analytics_count", {"ingredient": ingredient}
            else:
                return None

        except Exception as e:
            print(f"LLM intent classification error: {e}")
            return None

    def execute_intent(self, intent: str, params: dict) -> str:
        """Execute the detected intent"""
        try:
            if intent == "create_recipe":
                return self._create_recipe(params["ingredients"])

            elif intent == "show_recent":
                return self._show_recent_recipes(params["limit"])

            elif intent == "get_details":
                return self._get_recipe_details(params["recipe_name"])

            elif intent == "get_recipe_by_number":
                return self._get_recipe_by_number(params["recipe_number"])

            elif intent == "search":
                return self._search_recipes(params["query"])

            elif intent == "scale":
                return self._scale_recipe(params["servings"])

            elif intent == "analytics_frequent":
                return self._show_most_frequent_recipe()

            elif intent == "analytics_count":
                return self._count_recipes_by_ingredient(params["ingredient"])

            elif intent == "help":
                return self._show_help()

            else:
                return "I'm not sure what you'd like to do. Try asking for help!"

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}. Please try again or ask for help."

    def _create_recipe(self, ingredients: str) -> str:
        """Create a new recipe"""
        if not ingredients:
            return """To create a recipe, I need to know what ingredients you'd like to use.

For example, try:
• "Create a recipe with chicken and tomatoes"
• "Make a pasta dish"
• "Generate a recipe using beef and potatoes"

What ingredients would you like me to use?"""

        try:
            from react_agent import create_new_recipe
            result = create_new_recipe.invoke({"ingredients": ingredients, "dietary_needs": ""})

            # Extract the recipe ID and load the actual recipe object
            # Look for the recipe ID in the result (it's usually at the end)
            import re
            id_match = re.search(r'Recipe saved to database with ID:\s*(\w+)', result)
            if id_match:
                recipe_id = id_match.group(1)
                # Force database reload to get the newly created recipe
                self.db.load_recipes()
                # Load the recipe from database and set as current
                recipe = self.db.get_recipe(recipe_id)
                if recipe:
                    self.current_recipe = recipe
            else:
                # Try to get the most recent recipe as fallback
                recent_recipes = self.db.get_recent_recipes(1)
                if recent_recipes:
                    self.current_recipe = recent_recipes[0]

            return result
        except Exception as e:
            return f"Error creating recipe: {str(e)}"

    def _show_recent_recipes(self, limit: int) -> str:
        """Show recent recipes"""
        try:
            recipes = self.db.get_recent_recipes(limit)
            if not recipes:
                return "No recipes found in your database."

            result = f"Here are your {len(recipes)} most recent recipes:\n\n"
            for i, recipe in enumerate(recipes, 1):
                result += f"{i}. {recipe.title}\n"
                result += f"   Created: {recipe.created_at.strftime('%Y-%m-%d')} | Serves: {recipe.servings}\n"
                result += f"   Main ingredients: {', '.join(recipe.main_ingredients)}\n\n"

            result += "Would you like to see the full details for any of these recipes? Just ask!"
            return result

        except Exception as e:
            return f"Error retrieving recipes: {str(e)}"

    def _get_recipe_details(self, recipe_name: str) -> str:
        """Get full recipe details"""
        if not recipe_name:
            return "Which recipe would you like to see the details for? Please specify the recipe name."

        try:
            from react_agent import get_recipe_details
            result = get_recipe_details.invoke(recipe_name)
            return result
        except Exception as e:
            return f"Error getting recipe details: {str(e)}"

    def _get_recipe_by_number(self, recipe_number: int) -> str:
        """Get recipe details by number from recent list"""
        try:
            # Get recent recipes (limit to 10 to match number detection)
            recent_recipes = self.db.get_recent_recipes(10)

            if not recent_recipes:
                return "No recent recipes found."

            if recipe_number > len(recent_recipes):
                return f"Recipe number {recipe_number} not found. I only have {len(recent_recipes)} recent recipes."

            # Get the recipe (convert to 0-indexed)
            selected_recipe = recent_recipes[recipe_number - 1]

            # Return the full recipe details
            return selected_recipe.to_display_string()

        except Exception as e:
            return f"Error getting recipe by number: {str(e)}"

    def _search_recipes(self, query: str) -> str:
        """Search for recipes"""
        try:
            from react_agent import search_recipe_database
            result = search_recipe_database.invoke(query)
            return result
        except Exception as e:
            return f"Error searching recipes: {str(e)}"

    def _scale_recipe(self, servings: int) -> str:
        """Scale a recipe using current recipe context or most recent"""
        # First try to use current recipe from conversation context
        recipe_to_scale = self.current_recipe

        # If no current recipe, try to get the most recent one
        if not recipe_to_scale:
            recent_recipes = self.db.get_recent_recipes(1)
            if not recent_recipes:
                return "No recent recipe available to scale. Please create or select a recipe first."
            recipe_to_scale = recent_recipes[0]

        # Use the existing scaling logic from react_agent.py
        try:
            from react_agent import scale_current_recipe
            # Scale the recipe
            original_servings = recipe_to_scale.servings
            scaling_factor = servings / original_servings

            scaled_ingredients = []
            for ing in recipe_to_scale.all_ingredients:
                if ing.amount and ing.name and not ing.name.endswith(':**'):
                    try:
                        # Extract numeric value
                        import re
                        amount_str = str(ing.amount)
                        numbers = re.findall(r'\d+\.?\d*', amount_str)
                        if numbers:
                            amount = float(numbers[0])
                            scaled_amount = amount * scaling_factor

                            # Format nicely
                            if scaled_amount.is_integer():
                                scaled_amount = int(scaled_amount)
                            else:
                                scaled_amount = round(scaled_amount, 2)

                            unit = f" {ing.unit}" if ing.unit else ""
                            notes = f" ({ing.notes})" if ing.notes else ""
                            scaled_ingredients.append(f"• {scaled_amount}{unit} {ing.name}{notes}")
                        else:
                            scaled_ingredients.append(f"• {ing.name} (adjust to taste)")
                    except:
                        scaled_ingredients.append(f"• {ing.name} (adjust to taste)")
                elif ing.name and not ing.name.endswith(':**'):
                    scaled_ingredients.append(f"• {ing.name}")

            result = f"""✅ Scaled '{recipe_to_scale.title}' from {original_servings} to {servings} servings:

**Scaled Ingredients:**
{chr(10).join(scaled_ingredients)}

**Instructions remain the same:**
{chr(10).join([f"{i+1}. {instr}" for i, instr in enumerate(recipe_to_scale.instructions) if not instr.endswith(':**')])}

**Note:** Cooking times may need slight adjustments for larger quantities."""

            return result

        except Exception as e:
            return f"Error scaling recipe: {str(e)}"

    def _show_most_frequent_recipe(self) -> str:
        """Show the most frequent recipe title in the database"""
        try:
            recipes = self.db.get_recent_recipes(100)  # Get all recipes
            if not recipes:
                return "No recipes found in your database."

            # Count recipe titles
            title_counts = {}
            for recipe in recipes:
                title = recipe.title
                title_counts[title] = title_counts.get(title, 0) + 1

            # Find the most frequent
            if not title_counts:
                return "No recipes found to analyze."

            most_frequent_title = max(title_counts, key=title_counts.get)
            count = title_counts[most_frequent_title]

            # Get one instance of the most frequent recipe for details
            most_frequent_recipe = None
            for recipe in recipes:
                if recipe.title == most_frequent_title:
                    most_frequent_recipe = recipe
                    break

            if count == 1:
                return f"All recipes appear only once. Here's a recent one:\n\n**{most_frequent_title}**\nServes: {most_frequent_recipe.servings} | Main ingredients: {', '.join(most_frequent_recipe.main_ingredients)}"
            else:
                return f"""📊 **Most Frequent Recipe Analysis**

**Most frequent recipe:** {most_frequent_title}
**Appears:** {count} times in your database
**Serves:** {most_frequent_recipe.servings}
**Main ingredients:** {', '.join(most_frequent_recipe.main_ingredients)}

This recipe appears more often than others, suggesting it's one of your favorites!"""

        except Exception as e:
            return f"Error analyzing recipe frequency: {str(e)}"

    def _count_recipes_by_ingredient(self, ingredient: str) -> str:
        """Count how many recipes contain a specific ingredient"""
        if not ingredient:
            return "Please specify an ingredient to count. For example: 'How many recipes with chicken?'"

        try:
            recipes = self.db.get_recent_recipes(100)  # Get all recipes
            if not recipes:
                return "No recipes found in your database."

            matching_recipes = []
            ingredient_lower = ingredient.lower()

            for recipe in recipes:
                # Check both title and main ingredients
                title_match = ingredient_lower in recipe.title.lower()
                ingredient_match = any(ingredient_lower in ing.lower() for ing in recipe.main_ingredients)

                if title_match or ingredient_match:
                    matching_recipes.append(recipe)

            count = len(matching_recipes)

            if count == 0:
                return f"No recipes found containing '{ingredient}'. Try searching for a different ingredient."
            elif count == 1:
                recipe = matching_recipes[0]
                return f"""📊 **Recipe Count for '{ingredient.title()}'**

Found **1 recipe** containing '{ingredient}':

• {recipe.title}
  Serves: {recipe.servings} | Created: {recipe.created_at.strftime('%Y-%m-%d')}"""
            else:
                result = f"""📊 **Recipe Count for '{ingredient.title()}'**

Found **{count} recipes** containing '{ingredient}':

"""
                # Show up to 10 recipes
                for i, recipe in enumerate(matching_recipes[:10], 1):
                    result += f"{i}. {recipe.title}\n"
                    result += f"   Serves: {recipe.servings} | Created: {recipe.created_at.strftime('%Y-%m-%d')}\n\n"

                if count > 10:
                    result += f"... and {count - 10} more recipes.\n\n"

                result += f"That's {count}/{len(recipes)} recipes ({count/len(recipes)*100:.1f}%) of your collection!"
                return result

        except Exception as e:
            return f"Error counting recipes with '{ingredient}': {str(e)}"

    def _show_help(self) -> str:
        """Show help information"""
        return """🍳 I can help you with recipes! Here's what you can ask:

**Create Recipes:**
• "Create a recipe with chicken and tomatoes"
• "Make a pasta dish"
• "Generate a beef recipe"

**Browse Recipes:**
• "Show me recent recipes"
• "List my latest 10 recipes"

**Get Recipe Details:**
• "Show me the cooking steps for [recipe name]"
• "How do I make [recipe name]?"
• After showing recent recipes, just type a number (1-10) to see full details

**Search Recipes:**
• "Find chicken recipes"
• "Search for pasta dishes"

**Analytics:**
• "Show me the most frequent recipe"
• "How many recipes with chicken?"
• "Count recipes containing salmon"

**Other:**
• "Scale recipe for 8 people"
• "Help" (shows this message)

What would you like to do?"""

    def run(self, user_input: str, current_recipe=None) -> str:
        """Main entry point - detect intent and execute"""
        print(f"🔍 Processing: '{user_input}'")

        # Store current recipe context for scaling
        self.current_recipe = current_recipe

        # Detect intent
        intent, params = self.detect_intent(user_input)
        print(f"🎯 Detected intent: {intent} with params: {params}")

        # Execute intent
        result = self.execute_intent(intent, params)
        return result