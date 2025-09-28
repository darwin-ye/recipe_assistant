# llm_agent.py - Pure LLM-Based Recipe Agent
"""
A streamlined recipe agent that uses pure LLM intent classification
and direct function dispatch. Eliminates the complexity of regex patterns
and provides more natural, flexible user interactions.
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from recipe_models import RecipeDatabase, create_recipe_from_llm_response
from llm_intent_classifier import LLMIntentClassifier, IntentResult
from llm import generate_recipe_with_llm


@dataclass
class AgentResponse:
    """Structured response from the LLM agent"""
    content: str
    updated_context: Dict[str, Any]
    success: bool
    action_taken: str


class LLMRecipeAgent:
    """Pure LLM-based recipe agent with simplified architecture"""

    def __init__(self, llm):
        self.llm = llm
        self.db = RecipeDatabase()
        self.intent_classifier = LLMIntentClassifier(llm)
        self.context = {}

    def process_input(self, user_input: str, current_recipe: Optional[Any] = None) -> AgentResponse:
        """
        Process user input using pure LLM understanding and return natural response

        Args:
            user_input: User's natural language input
            current_recipe: Optional current recipe context

        Returns:
            AgentResponse with natural language content and updated context
        """

        # Build context for intent classification
        context = {
            "current_recipe": current_recipe,
            "last_action": self.context.get("last_action"),
            "recent_recipes": self.db.get_recent_recipes(5) if self.db.recipes else []
        }

        # Classify intent using LLM
        intent_result = self.intent_classifier.classify_intent(user_input, context)

        print(f"🧠 LLM Intent: {intent_result.intent} (confidence: {intent_result.confidence:.2f})")
        if intent_result.parameters:
            print(f"   Parameters: {intent_result.parameters}")

        # Dispatch to appropriate handler
        response = self._dispatch_intent(intent_result, user_input, current_recipe)

        # Update context
        self.context["last_action"] = intent_result.intent
        self.context["last_input"] = user_input

        return response

    def _dispatch_intent(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Dispatch intent to appropriate handler function"""

        intent_handlers = {
            "create_recipe": self._handle_create_recipe,
            "search_recipes": self._handle_search_recipes,
            "get_recent": self._handle_get_recent,
            "get_details": self._handle_get_details,
            "analytics_frequent": self._handle_analytics_frequent,
            "analytics_count": self._handle_analytics_count,
            "scale_recipe": self._handle_scale_recipe,
            "numbered_reference": self._handle_numbered_reference,
            "help": self._handle_help
        }

        handler = intent_handlers.get(intent_result.intent, self._handle_help)

        try:
            return handler(intent_result, user_input, current_recipe)
        except Exception as e:
            error_message = f"I encountered an error while processing your request: {str(e)}"
            return AgentResponse(
                content=self._generate_natural_response(error_message, user_input),
                updated_context={"error": str(e)},
                success=False,
                action_taken="error"
            )

    def _handle_create_recipe(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle recipe creation requests"""

        # Extract ingredients and dietary needs
        ingredients = intent_result.parameters.get("ingredients", "")
        dietary_needs = intent_result.parameters.get("dietary_needs", "")

        # If no ingredients specified, ask for them
        if not ingredients:
            # Use LLM to extract ingredients from the original input
            ingredients = self._extract_ingredients_with_llm(user_input)

        if not ingredients:
            response_text = "I'd love to create a recipe for you! What ingredients would you like me to work with?"
            return AgentResponse(
                content=response_text,
                updated_context={"awaiting": "ingredients"},
                success=True,
                action_taken="request_ingredients"
            )

        try:
            # Generate recipe using LLM
            print(f"🍳 Generating recipe with: {ingredients}")
            llm_response = generate_recipe_with_llm(ingredients, dietary_needs)

            # Convert to structured format
            recipe = create_recipe_from_llm_response(llm_response, ingredients, dietary_needs)

            # Save to database
            recipe_id = self.db.add_recipe(recipe)

            # Generate natural response
            success_message = f"✅ I've created a delicious recipe for you!\n\n{recipe.to_display_string()}\n\n📝 Recipe saved to your collection."

            return AgentResponse(
                content=success_message,
                updated_context={"current_recipe": recipe, "recipe_id": recipe_id},
                success=True,
                action_taken="create_recipe"
            )

        except Exception as e:
            error_message = f"I had trouble creating that recipe. Could you try with different ingredients or be more specific?"
            return AgentResponse(
                content=error_message,
                updated_context={"error": str(e)},
                success=False,
                action_taken="create_recipe_failed"
            )

    def _handle_search_recipes(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle recipe search requests"""

        query = intent_result.parameters.get("query", user_input)

        if not self.db.recipes:
            response_text = "You don't have any recipes saved yet. Would you like me to create your first recipe?"
            return AgentResponse(
                content=response_text,
                updated_context={"suggestion": "create_first_recipe"},
                success=True,
                action_taken="no_recipes_found"
            )

        # Search recipes
        results = self.db.search(query, top_k=5)

        if not results:
            response_text = f"I couldn't find any recipes matching '{query}'. Would you like me to create a new recipe with those ingredients instead?"
            return AgentResponse(
                content=response_text,
                updated_context={"suggestion": "create_instead", "query": query},
                success=True,
                action_taken="no_search_results"
            )

        # Format results naturally
        response_parts = [f"I found {len(results)} recipes matching '{query}':\n"]
        for i, (recipe_id, recipe, score) in enumerate(results[:3], 1):
            response_parts.append(f"{i}. **{recipe.title}** (match: {score:.0%})")
            response_parts.append(f"   Serves {recipe.servings} | Main ingredients: {', '.join(recipe.main_ingredients[:3])}")

        response_parts.append("\nWhich recipe would you like to see? Just tell me the number or name!")

        return AgentResponse(
            content="\n".join(response_parts),
            updated_context={"search_results": results},
            success=True,
            action_taken="search_recipes"
        )

    def _handle_get_recent(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle requests for recent recipes"""

        limit = intent_result.parameters.get("limit", 5)
        recent_recipes = self.db.get_recent_recipes(limit)

        if not recent_recipes:
            response_text = "You haven't created any recipes yet. Let's make your first one! What ingredients do you have?"
            return AgentResponse(
                content=response_text,
                updated_context={"suggestion": "create_first_recipe"},
                success=True,
                action_taken="no_recent_recipes"
            )

        # Format recent recipes naturally
        response_parts = [f"Here are your {len(recent_recipes)} most recent recipes:\n"]
        for i, recipe in enumerate(recent_recipes, 1):
            response_parts.append(f"{i}. **{recipe.title}**")
            response_parts.append(f"   Created: {recipe.created_at.strftime('%Y-%m-%d')} | Serves: {recipe.servings}")

        response_parts.append("\nWhich one would you like to see? Just tell me the number!")

        return AgentResponse(
            content="\n".join(response_parts),
            updated_context={"recent_recipes": recent_recipes},
            success=True,
            action_taken="get_recent"
        )

    def _handle_get_details(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle requests for specific recipe details"""

        recipe_name = intent_result.parameters.get("recipe_name", "")

        # If no specific recipe name, try to find from current context or recent recipes
        if not recipe_name and current_recipe:
            recipe = current_recipe
        else:
            # Search for the recipe by name
            recipe = self._find_recipe_by_name(recipe_name or user_input)

        if not recipe:
            response_text = "I couldn't find that specific recipe. Could you be more specific about which recipe you'd like to see?"
            return AgentResponse(
                content=response_text,
                updated_context={"need_clarification": True},
                success=False,
                action_taken="recipe_not_found"
            )

        # Show full recipe details
        response_text = f"Here's your recipe:\n\n{recipe.to_display_string()}"

        return AgentResponse(
            content=response_text,
            updated_context={"current_recipe": recipe},
            success=True,
            action_taken="get_details"
        )

    def _handle_analytics_frequent(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle requests for most frequent recipe analysis"""

        if not self.db.recipes:
            response_text = "You don't have any recipes yet to analyze. Let's create some recipes first!"
            return AgentResponse(
                content=response_text,
                updated_context={"suggestion": "create_recipes"},
                success=True,
                action_taken="no_recipes_for_analytics"
            )

        # Get frequency analysis
        most_frequent = self.db.get_most_frequent_recipe()

        if not most_frequent:
            response_text = "I couldn't determine your most frequent recipe yet. Create a few more recipes to see patterns!"
            return AgentResponse(
                content=response_text,
                updated_context={"suggestion": "create_more_recipes"},
                success=True,
                action_taken="insufficient_data"
            )

        recipe, count = most_frequent
        response_text = f"📊 **Your Most Frequent Recipe**\n\nYou've created '{recipe.title}' {count} time{'s' if count > 1 else ''}!\n\nThis seems to be your go-to recipe. Would you like to see the details again?"

        return AgentResponse(
            content=response_text,
            updated_context={"current_recipe": recipe, "frequency_count": count},
            success=True,
            action_taken="analytics_frequent"
        )

    def _handle_analytics_count(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle requests for ingredient frequency counting"""

        ingredient = intent_result.parameters.get("ingredient", "")

        if not ingredient:
            # Try to extract ingredient from user input
            ingredient = self._extract_ingredient_for_counting(user_input)

        if not ingredient:
            response_text = "Which ingredient would you like me to count in your recipes?"
            return AgentResponse(
                content=response_text,
                updated_context={"awaiting": "ingredient_for_counting"},
                success=True,
                action_taken="request_ingredient"
            )

        # Count recipes with this ingredient
        count = self.db.count_recipes_with_ingredient(ingredient)
        total_recipes = len(self.db.recipes)

        if count == 0:
            response_text = f"You don't have any recipes with {ingredient} yet. Would you like me to create one?"
        else:
            percentage = (count / total_recipes) * 100 if total_recipes > 0 else 0
            response_text = f"📊 **Ingredient Analysis: {ingredient.title()}**\n\nFound in {count} out of {total_recipes} recipes ({percentage:.1f}%)\n\nThis ingredient appears in {self._frequency_description(percentage)} of your recipes."

        return AgentResponse(
            content=response_text,
            updated_context={"analyzed_ingredient": ingredient, "count": count},
            success=True,
            action_taken="analytics_count"
        )

    def _handle_scale_recipe(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle recipe scaling requests"""

        target_servings = intent_result.parameters.get("target_servings")

        if not target_servings:
            # Try to extract servings number from user input
            target_servings = self._extract_servings_number(user_input)

        if not target_servings or not current_recipe:
            response_text = "I need to know which recipe to scale and for how many people. Could you be more specific?"
            return AgentResponse(
                content=response_text,
                updated_context={"need_clarification": True},
                success=False,
                action_taken="scale_clarification_needed"
            )

        try:
            # Scale the recipe
            scaled_result = self._scale_recipe(current_recipe, int(target_servings))
            response_text = f"📏 **Scaled Recipe**\n\n{scaled_result}"

            return AgentResponse(
                content=response_text,
                updated_context={"scaled_servings": target_servings},
                success=True,
                action_taken="scale_recipe"
            )

        except Exception as e:
            response_text = f"I had trouble scaling that recipe. Could you try again with a specific number of people?"
            return AgentResponse(
                content=response_text,
                updated_context={"error": str(e)},
                success=False,
                action_taken="scale_failed"
            )

    def _handle_numbered_reference(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle numbered recipe selections"""

        number = intent_result.parameters.get("number", 0)

        # Get the appropriate list based on context
        recipes_list = self.context.get("recent_recipes") or self.context.get("search_results")

        if not recipes_list:
            response_text = "I don't have a numbered list to reference right now. Could you tell me what you're looking for?"
            return AgentResponse(
                content=response_text,
                updated_context={"need_clarification": True},
                success=False,
                action_taken="no_numbered_list"
            )

        if number < 1 or number > len(recipes_list):
            response_text = f"Please choose a number between 1 and {len(recipes_list)}."
            return AgentResponse(
                content=response_text,
                updated_context={"need_valid_number": True},
                success=False,
                action_taken="invalid_number"
            )

        # Get the selected recipe
        if isinstance(recipes_list[0], tuple):  # Search results format
            _, recipe, _ = recipes_list[number - 1]
        else:  # Recent recipes format
            recipe = recipes_list[number - 1]

        response_text = f"Here's your selected recipe:\n\n{recipe.to_display_string()}"

        return AgentResponse(
            content=response_text,
            updated_context={"current_recipe": recipe},
            success=True,
            action_taken="numbered_selection"
        )

    def _handle_help(self, intent_result: IntentResult, user_input: str, current_recipe: Optional[Any]) -> AgentResponse:
        """Handle help requests and unclear inputs"""

        help_text = """🍳 **AI Recipe Assistant - What I Can Do**

**Create Recipes:**
- "Create a chicken pasta recipe"
- "Make something with beef and vegetables"
- "Give me a salmon recipe for 4 people"

**Find Recipes:**
- "Show me recent recipes"
- "Find pasta recipes"
- "Search for chicken dishes"

**Recipe Analytics:**
- "What's my most frequent recipe?"
- "How many chicken recipes do I have?"
- "What do I cook most often?"

**Recipe Details:**
- "Show me the salmon recipe"
- "Scale this to 8 people"
- Just say a number (1, 2, 3) to select from lists

**Just talk naturally!** I understand many ways of asking for things. What would you like to do?"""

        return AgentResponse(
            content=help_text,
            updated_context={"showed_help": True},
            success=True,
            action_taken="help"
        )

    # Helper methods

    def _extract_ingredients_with_llm(self, user_input: str) -> str:
        """Use LLM to extract ingredients from user input"""
        prompt = f"""Extract the main ingredients mentioned in this request: "{user_input}"

Return only the ingredients as a comma-separated list, or empty string if no specific ingredients are mentioned.

Examples:
"make a chicken pasta recipe" → "chicken, pasta"
"create something with beef and vegetables" → "beef, vegetables"
"new recipe" → ""

Ingredients:"""

        try:
            response = self.llm.invoke(prompt)
            ingredients = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            return ingredients if ingredients and ingredients != '""' else ""
        except:
            return ""

    def _extract_ingredient_for_counting(self, user_input: str) -> str:
        """Extract ingredient name for counting from user input"""
        import re

        # Simple patterns to extract ingredient
        patterns = [
            r"how many (\w+) recipes",
            r"count.*?(\w+).*?recipes",
            r"recipes with (\w+)",
            r"(\w+) recipes"
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                return match.group(1)

        return ""

    def _extract_servings_number(self, user_input: str) -> Optional[int]:
        """Extract number of servings from user input"""
        import re

        numbers = re.findall(r'\d+', user_input)
        if numbers:
            return int(numbers[0])

        # Check for words like "double", "triple", etc.
        if "double" in user_input.lower():
            return 2
        elif "triple" in user_input.lower():
            return 3

        return None

    def _find_recipe_by_name(self, query: str) -> Optional[Any]:
        """Find recipe by name or partial match"""
        for recipe in self.db.recipes.values():
            if query.lower() in recipe.title.lower():
                return recipe
        return None

    def _scale_recipe(self, recipe: Any, target_servings: int) -> str:
        """Scale recipe ingredients to target servings"""
        original_servings = recipe.servings
        scaling_factor = target_servings / original_servings

        scaled_ingredients = []
        for ing in recipe.all_ingredients:
            if ing.amount:
                try:
                    import re
                    numbers = re.findall(r'\d+\.?\d*', str(ing.amount))
                    if numbers:
                        amount = float(numbers[0])
                        scaled_amount = amount * scaling_factor

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
            else:
                scaled_ingredients.append(f"• {ing.name}")

        result = f"""**{recipe.title}** (scaled from {original_servings} to {target_servings} servings)

**Ingredients:**
{chr(10).join(scaled_ingredients)}

**Instructions:**
{chr(10).join([f"{i+1}. {instr}" for i, instr in enumerate(recipe.instructions)])}"""

        return result

    def _frequency_description(self, percentage: float) -> str:
        """Convert percentage to descriptive text"""
        if percentage >= 50:
            return "most"
        elif percentage >= 25:
            return "many"
        elif percentage >= 10:
            return "some"
        else:
            return "a few"

    def _generate_natural_response(self, content: str, user_input: str) -> str:
        """Use LLM to generate more natural responses (optional enhancement)"""
        # For now, return content as-is, but this could be enhanced
        # to make responses even more natural and conversational
        return content


# Example usage
if __name__ == "__main__":
    from llm import llm

    # Create agent
    agent = LLMRecipeAgent(llm)

    # Test queries
    test_queries = [
        "create a chicken pasta recipe",
        "show me recent recipes",
        "what's my most frequent recipe?",
        "2",
        "help"
    ]

    print("🧪 Testing LLM Recipe Agent")
    print("=" * 50)

    for query in test_queries:
        print(f"\nUser: {query}")
        response = agent.process_input(query)
        print(f"Agent: {response.content[:200]}...")
        print(f"Success: {response.success}, Action: {response.action_taken}")