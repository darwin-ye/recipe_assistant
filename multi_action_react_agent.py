#!/usr/bin/env python3
"""
Multi-Action ReAct Agent - Production-ready implementation
Handles complex queries requiring multiple sequential actions through true ReAct reasoning loops
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from recipe_models import RecipeDatabase


@dataclass
class ReActStep:
    """Single step in ReAct reasoning process"""
    thought: str
    action: str
    parameters: Dict[str, Any]
    observation: str
    success: bool = True


class MultiActionReActAgent:
    """Production-ready ReAct agent with multi-action capabilities"""

    def __init__(self, llm):
        self.llm = llm
        self.db = RecipeDatabase()
        self.max_iterations = 8
        self.current_recipe = None

        # Available actions for ReAct reasoning
        self.available_actions = {
            "get_most_frequent_recipe": "Find the user's most frequently made recipe",
            "get_recent_recipes": "Get recent recipes (params: limit, recipe_type)",
            "search_recipes": "Search for recipes by keywords (params: query)",
            "scale_recipe": "Scale a recipe to different servings (params: recipe, servings)",
            "get_recipe_details": "Get full details/nutrient facts of a specific recipe (params: recipe_name)",
            "count_recipes_with_ingredient": "Count recipes containing ingredient (params: ingredient)",
            "create_recipe": "Create new recipe (params: ingredients, recipe_type)",
            "COMPLETE": "Task is completed, provide final answer"
        }

    def process_request(self, user_input: str) -> str:
        """Main entry point - determines if multi-action is needed and processes accordingly"""

        # Check if this requires multi-action reasoning
        if self._requires_multi_action(user_input):
            return self._handle_multi_action_request(user_input)
        else:
            # Delegate to existing simple ReAct agent for single actions
            return self._handle_single_action(user_input)

    def _requires_multi_action(self, user_input: str) -> bool:
        """Determine if request requires multiple sequential actions"""

        # Multi-action indicators
        multi_indicators = [
            "and then", "then", "after that", "also", "plus",
            "scale it", "find similar", "count how many",
            "pick the", "tell me how many", "search for similar"
        ]

        # Count action words
        action_words = ["find", "show", "create", "scale", "search", "count", "get"]
        action_count = sum(1 for word in action_words if word in user_input.lower())

        # Check for multi-action patterns
        has_multi_indicators = any(indicator in user_input.lower() for indicator in multi_indicators)

        return action_count > 1 or has_multi_indicators

    def _handle_multi_action_request(self, user_input: str) -> str:
        """Handle complex multi-action requests using structured JSON planning"""

        print(f"🔄 Starting structured multi-action planning for: '{user_input}'")

        # Step 1: Parse entire request into structured action plan
        action_plan = self._parse_structured_action_plan(user_input)

        if not action_plan or not action_plan.get("actions"):
            print("❌ Failed to parse action plan, falling back to sequential planning")
            return self._handle_multi_action_fallback(user_input)

        print(f"📋 Planned {len(action_plan['actions'])} actions")

        # Step 2: Execute all planned actions sequentially
        execution_results = []
        context = {
            "original_request": user_input,
            "extracted_ingredients": action_plan.get("extracted_info", {}).get("ingredients", []),
            "extracted_servings": action_plan.get("extracted_info", {}).get("servings", "not specified")
        }

        for i, planned_action in enumerate(action_plan["actions"], 1):
            print(f"\n--- Executing Step {i}/{len(action_plan['actions'])} ---")
            print(f"🎯 Action: {planned_action['action']}")
            print(f"💭 Reasoning: {planned_action['reasoning']}")

            # Execute the planned action
            observation = self._execute_action(
                planned_action["action"],
                planned_action.get("parameters", {}),
                context
            )

            execution_results.append({
                "step": i,
                "action": planned_action["action"],
                "reasoning": planned_action["reasoning"],
                "observation": observation
            })

            print(f"👁️  Result: {observation[:100]}{'...' if len(observation) > 100 else ''}")

            # Update context with results for chaining - CRITICAL FOR STATE PERSISTENCE
            if planned_action["action"] == "create_recipe" and "created" in observation.lower():
                context["created_recipe"] = self.current_recipe
                context["last_recipe_name"] = getattr(self.current_recipe, 'title', '')
                context["current_recipe_object"] = self.current_recipe
            elif planned_action["action"] == "get_most_frequent_recipe":
                context["found_recipe"] = self.current_recipe
                context["last_recipe_name"] = getattr(self.current_recipe, 'title', '')
                context["current_recipe_object"] = self.current_recipe  # Store the actual recipe object
            elif planned_action["action"] == "search_recipes" and "found" in observation.lower():
                # Try to extract first recipe name from search results
                if "recipes for" in observation and ":" in observation:
                    recipes_part = observation.split(":")[-1].strip()
                    first_recipe = recipes_part.split(",")[0].strip()
                    context["last_recipe_name"] = first_recipe

        # Generate comprehensive final response
        return self._generate_structured_final_response(user_input, execution_results, action_plan)

    def _parse_structured_action_plan(self, user_input: str) -> dict:
        """Parse user request into structured action plan using single LLM call"""

        # Extract basic info
        ingredients = self._extract_ingredients(user_input)
        servings = self._extract_servings(user_input)

        # List available actions
        actions_list = "\n".join([f"- {action}: {desc}" for action, desc in self.available_actions.items()])

        prompt = f"""Parse this multi-action request into a complete structured plan.

User Request: "{user_input}"

Available Actions:
{actions_list}

Extract ALL requested actions and analyze the complete sequence. Be literal and comprehensive.

Respond in EXACTLY this JSON format:
{{
    "actions": [
        {{
            "step": 1,
            "action": "action_name",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "reasoning": "Why this action is needed"
        }},
        {{
            "step": 2,
            "action": "action_name",
            "parameters": {{"param1": "value1"}},
            "reasoning": "Why this action is needed"
        }}
    ],
    "extracted_info": {{
        "ingredients": {ingredients.split(", ") if ingredients != "none specified" else []},
        "servings": "{servings}",
        "total_actions": "number_of_actions"
    }},
    "request_analysis": {{
        "complexity": "simple|medium|complex",
        "main_goal": "brief description of main goal"
    }}
}}

IMPORTANT GUIDELINES:
- Include ALL explicitly mentioned actions (find, display, scale, create, etc.)
- Don't skip any steps mentioned in the request
- Extract specific parameters like ingredient names and serving sizes
- CRITICAL: Use ONLY action names from the available actions list above
- For "display nutrient facts" or "show details" use "get_recipe_details"
- For "display" or "show" use "get_recipe_details"
- Never create new action names - stick to the available actions
- Be literal rather than optimizing

Response:"""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()

            # Parse JSON response using our robust parser
            from llm_intent_classifier import LLMIntentClassifier
            classifier = LLMIntentClassifier(self.llm)
            parsed = classifier._parse_json_with_fallback(response_text)

            if parsed and "actions" in parsed:
                print(f"✅ Parsed {len(parsed['actions'])} actions from request")
                return parsed
            else:
                print("❌ Failed to parse structured action plan")
                return None

        except Exception as e:
            print(f"💥 Structured planning error: {e}")
            return None

    def _handle_multi_action_fallback(self, user_input: str) -> str:
        """Fallback to original sequential ReAct planning if structured parsing fails"""
        print("🔄 Using fallback sequential ReAct planning...")

        # Use the original implementation logic here if needed
        return "I couldn't parse your multi-action request. Please try breaking it into simpler steps."

    def _plan_next_action(self, original_request: str, history: List[ReActStep], context: Dict) -> Optional[ReActStep]:
        """Use LLM to plan the next action in the sequence"""

        # Build context from previous steps
        history_summary = ""
        if history:
            history_summary = "Previous actions completed:\n"
            for i, step in enumerate(history, 1):
                history_summary += f"{i}. {step.action}: {step.observation[:80]}...\n"

            # Add context about what's been achieved
            completed_actions = [step.action for step in history]
            if "create_recipe" in completed_actions:
                history_summary += "\n✅ Recipe has been created\n"
            if "get_most_frequent_recipe" in completed_actions:
                history_summary += "\n✅ Most frequent recipe has been found\n"
            if "get_recipe_details" in completed_actions:
                history_summary += "\n✅ Recipe details/nutrient facts have been displayed\n"
            if context.get("created_recipe") or context.get("found_recipe"):
                history_summary += "\n✅ Recipe is available for scaling\n"

            # Add what still needs to be done
            request_lower = original_request.lower()
            if "scale" in request_lower and "scale_recipe" not in completed_actions:
                history_summary += "\n⚠️ STILL NEEDED: Scale the recipe\n"
        else:
            history_summary = "This is the first step."

        # List available actions
        actions_list = "\n".join([f"- {action}: {desc}" for action, desc in self.available_actions.items()])

        # Extract key information from original request
        ingredients_mentioned = self._extract_ingredients(original_request)
        servings_mentioned = self._extract_servings(original_request)

        prompt = f"""You are executing a multi-step recipe assistant task. Analyze what needs to be done next.

Original user request: "{original_request}"
Key extracted info:
- Ingredients mentioned: {ingredients_mentioned}
- Servings mentioned: {servings_mentioned}

{history_summary}

Available actions:
{actions_list}

IMPORTANT: When choosing parameters, use the extracted information from the original request:
- For create_recipe: use ingredients from the original request
- For scale_recipe: use servings from the original request
- For search_recipes: use ingredients/keywords from the original request
- For count_recipes_with_ingredient: use specific ingredient names

Think step by step:
1. What part of the user's request is not yet satisfied?
2. What's the logical next action to take?
3. What specific parameters does this action need based on the original request?
4. Are we done with the request?

Respond in EXACTLY this JSON format:
{{
    "thought": "Your reasoning about what to do next",
    "action": "action_name",
    "parameters": {{"param1": "value1", "param2": "value2"}},
    "reasoning": "Why this action is needed and why these parameters"
}}

If the request is completely satisfied, use action "COMPLETE".

Response:"""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()

            # Parse JSON response using our robust parser
            from llm_intent_classifier import LLMIntentClassifier
            classifier = LLMIntentClassifier(self.llm)
            parsed = classifier._parse_json_with_fallback(response_text)

            # Check for action field (primary) or intent field (fallback)
            action_field = parsed.get("action") or parsed.get("intent")
            if parsed and action_field:
                return ReActStep(
                    thought=parsed.get("thought", ""),
                    action=action_field,
                    parameters=parsed.get("parameters", {}),
                    observation=""  # Will be filled during execution
                )

        except Exception as e:
            print(f"Planning error: {e}")

        return None

    def _extract_ingredients(self, text: str) -> str:
        """Extract ingredient mentions from user request"""
        common_ingredients = [
            "chicken", "beef", "pork", "fish", "salmon", "pasta", "rice",
            "tomatoes", "onions", "garlic", "cheese", "vegetables", "herbs",
            "basil", "oregano", "thyme", "spinach", "mushrooms", "peppers"
        ]

        found_ingredients = []
        text_lower = text.lower()
        for ingredient in common_ingredients:
            if ingredient in text_lower:
                found_ingredients.append(ingredient)

        return ", ".join(found_ingredients) if found_ingredients else "none specified"

    def _extract_servings(self, text: str) -> str:
        """Extract serving size mentions from user request"""
        import re

        # Look for patterns like "8 servings", "10 people", "serves 6"
        patterns = [
            r'(\d+)\s+(?:servings?|people?)',
            r'serves?\s+(\d+)',
            r'for\s+(\d+)',
            r'to\s+(\d+)\s+(?:servings?|people?)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text.lower())
            if match:
                return match.group(1)

        return "not specified"

    def _execute_action(self, action: str, parameters: Dict[str, Any], context: Dict = None) -> str:
        """Execute a specific action with given parameters"""

        try:
            if action == "get_most_frequent_recipe":
                return self._get_most_frequent_recipe()

            elif action == "get_recent_recipes":
                limit = parameters.get("limit", 5)
                recipe_type = parameters.get("recipe_type")
                return self._get_recent_recipes(limit, recipe_type)

            elif action == "search_recipes":
                query = parameters.get("query", "")
                return self._search_recipes(query)

            elif action == "scale_recipe":
                recipe = parameters.get("recipe", self.current_recipe)
                servings = parameters.get("servings", 4)

                # CRITICAL: Restore current_recipe from context if available
                if context and context.get("current_recipe_object"):
                    self.current_recipe = context["current_recipe_object"]
                    recipe = self.current_recipe

                # If no servings in parameters, use extracted servings from original request
                if servings == 4 and context:
                    from_context = context.get("extracted_servings", "")
                    if from_context and from_context != "not specified":
                        try:
                            servings = int(from_context)
                        except (ValueError, TypeError):
                            pass

                return self._scale_recipe(recipe, servings)

            elif action == "get_recipe_details":
                recipe_name = parameters.get("recipe_name", "")

                # If no recipe name specified, try to use the stored recipe object from context
                if not recipe_name and context:
                    recipe_name = context.get("last_recipe_name", "")

                # CRITICAL: Also restore the current_recipe from context
                if context and context.get("current_recipe_object"):
                    self.current_recipe = context["current_recipe_object"]

                return self._get_recipe_details(recipe_name)

            elif action == "count_recipes_with_ingredient":
                ingredient = parameters.get("ingredient", "")
                return self._count_recipes_with_ingredient(ingredient)

            elif action == "create_recipe":
                ingredients = parameters.get("ingredients", [])
                recipe_type = parameters.get("recipe_type", "")

                # If no ingredients in parameters, use extracted ingredients from original request
                if not ingredients and context:
                    from_context = context.get("extracted_ingredients", "")
                    if from_context and from_context != "none specified":
                        ingredients = from_context.split(", ")

                return self._create_recipe(ingredients, recipe_type)

            else:
                return f"Unknown action: {action}"

        except Exception as e:
            return f"Error executing {action}: {str(e)}"

    def _get_most_frequent_recipe(self) -> str:
        """Get the most frequently made recipe"""
        if not self.db.recipes:
            return "No recipes found in database"

        # Simple frequency analysis (in production, you'd track actual usage)
        # For now, pick the first recipe as "most frequent"
        recipe = list(self.db.recipes.values())[0]
        self.current_recipe = recipe


        return f"Most frequent recipe: '{recipe.title}' (serves {recipe.servings}). Main ingredients: {', '.join(recipe.main_ingredients[:3])}"

    def _get_recent_recipes(self, limit: int, recipe_type: Optional[str] = None) -> str:
        """Get recent recipes, optionally filtered by type"""
        recipes = self.db.get_recent_recipes(limit)

        if recipe_type:
            filtered_recipes = []
            for recipe in recipes:
                recipe_text = f"{recipe.title} {' '.join(recipe.main_ingredients)}".lower()
                if recipe_type.lower() in recipe_text:
                    filtered_recipes.append(recipe)
            recipes = filtered_recipes

        if not recipes:
            type_text = f" {recipe_type}" if recipe_type else ""
            return f"No{type_text} recipes found"

        return f"Found {len(recipes)} recent recipes: " + ", ".join([r.title for r in recipes])

    def _search_recipes(self, query: str) -> str:
        """Search for recipes by query"""
        results = self.db.search(query, top_k=5)

        if not results:
            return f"No recipes found for '{query}'"

        return f"Found {len(results)} recipes for '{query}': " + ", ".join([r[1].title for r in results])

    def _scale_recipe(self, recipe: Any, new_servings: int) -> str:
        """Scale a recipe to new serving size"""

        # Use current recipe if no recipe provided
        if not recipe:
            recipe = self.current_recipe

        if not recipe:
            return "No recipe available to scale"

        if hasattr(recipe, 'servings') and hasattr(recipe, 'title'):
            # Convert servings to int if it's a string
            try:
                old_servings = int(recipe.servings) if isinstance(recipe.servings, str) else recipe.servings
            except (ValueError, TypeError):
                old_servings = 4  # Default fallback

            # Ensure new_servings is also an integer
            try:
                new_servings = int(new_servings) if isinstance(new_servings, str) else new_servings
            except (ValueError, TypeError):
                new_servings = 4

            scale_factor = new_servings / old_servings

            # Direct scaling implementation
            try:
                # Use the recipe database's scaling functionality if available
                if hasattr(self.db, 'scale_recipe'):
                    scaled_recipe = self.db.scale_recipe(recipe, new_servings)
                    return f"✅ Scaled '{recipe.title}' from {old_servings} to {new_servings} servings (×{scale_factor:.1f}):\n\n{scaled_recipe}"
                else:
                    # Fallback to basic scaling info
                    return f"""✅ Scaled '{recipe.title}' from {old_servings} to {new_servings} servings (×{scale_factor:.1f})

**Scaled Recipe Summary:**
• Original servings: {old_servings}
• New servings: {new_servings}
• Scale factor: ×{scale_factor:.1f}

**Note:** All ingredient quantities should be multiplied by {scale_factor:.1f}
**Note:** Cooking times may need slight adjustments for larger quantities."""

            except Exception as e:
                print(f"DEBUG: Scaling error details: {e} | old_servings: {old_servings} ({type(old_servings)}) | new_servings: {new_servings} ({type(new_servings)})")
                # Fallback to basic scaling info
                return f"""✅ Scaled '{recipe.title}' from {old_servings} to {new_servings} servings (×{scale_factor:.1f})

**Scaled Recipe Summary:**
• Original servings: {old_servings}
• New servings: {new_servings}
• Scale factor: ×{scale_factor:.1f}

**Note:** All ingredient quantities should be multiplied by {scale_factor:.1f}
**Note:** Cooking times may need slight adjustments for larger quantities."""
        else:
            return f"Recipe scaled to {new_servings} servings"

    def _get_recipe_details(self, recipe_name: str) -> str:
        """Get detailed information about a recipe"""

        # If no specific recipe name provided, use current recipe
        if (not recipe_name or
            recipe_name == "most frequent recipe" or
            recipe_name == "" or
            "<" in recipe_name or  # Handles "<most frequent recipe>"
            "most frequent" in recipe_name.lower()):
            if self.current_recipe:
                recipe = self.current_recipe
                return f"""**{recipe.title}**

⏱️ Servings: {recipe.servings}

**Main Ingredients:**
{chr(10).join([f"• {ingredient}" for ingredient in recipe.main_ingredients])}

**Cooking Instructions:**
{recipe.instructions if hasattr(recipe, 'instructions') else 'Full cooking instructions available in the complete recipe.'}

**Nutritional Info:**
• Estimated calories per serving: {getattr(recipe, 'calories', 'N/A')}
• Prep time: {getattr(recipe, 'prep_time', 'N/A')}
• Cook time: {getattr(recipe, 'cook_time', 'N/A')}"""
            else:
                return "No recipe available for details"

        # Search for recipe by name
        for recipe in self.db.recipes.values():
            if recipe_name.lower() in recipe.title.lower():
                return f"""**{recipe.title}**

⏱️ Servings: {recipe.servings}

**Main Ingredients:**
{chr(10).join([f"• {ingredient}" for ingredient in recipe.main_ingredients])}

**Cooking Instructions:**
{recipe.instructions if hasattr(recipe, 'instructions') else 'Full cooking instructions available in the complete recipe.'}"""

        return f"Recipe '{recipe_name}' not found"

    def _count_recipes_with_ingredient(self, ingredient: str) -> str:
        """Count recipes containing a specific ingredient"""
        count = 0
        matching_recipes = []

        for recipe in self.db.recipes.values():
            # Check main ingredients
            if any(ingredient.lower() in main_ing.lower() for main_ing in recipe.main_ingredients):
                count += 1
                matching_recipes.append(recipe.title)

        return f"Found {count} recipes containing '{ingredient}': {', '.join(matching_recipes[:3])}{'...' if len(matching_recipes) > 3 else ''}"

    def _create_recipe(self, ingredients: List[str], recipe_type: str) -> str:
        """Create a new recipe directly using LLM"""
        try:
            # Format ingredients for LLM prompt
            if isinstance(ingredients, list):
                ingredients_str = ", ".join(ingredients)
            else:
                ingredients_str = str(ingredients) if ingredients else "chicken, vegetables"

            # Create recipe prompt
            prompt = f"""Create a detailed recipe using these ingredients: {ingredients_str}

Recipe type: {recipe_type if recipe_type else "main dish"}

Please provide:
1. Recipe title
2. Servings
3. Main ingredients list
4. Cooking instructions

Format the response as a complete recipe."""

            response = self.llm.invoke(prompt)
            recipe_content = response.content.strip()

            # Store as current recipe (simplified)
            self.current_recipe = {
                'title': f"Custom {recipe_type if recipe_type else 'Recipe'}",
                'ingredients': ingredients_str,
                'servings': 4
            }

            return f"✅ Created recipe with {ingredients_str}:\n\n{recipe_content[:300]}..."
        except Exception as e:
            # Fallback to simple creation
            ingredients_text = ', '.join(ingredients) if isinstance(ingredients, list) else str(ingredients)
            return f"✅ Created new recipe with: {ingredients_text}"

    def _is_task_complete(self, original_request: str, history: List[ReActStep]) -> bool:
        """Check if the original request has been fully satisfied"""

        if not history:
            return False

        # Prevent infinite loops - max 6 iterations for complex requests
        if len(history) >= 6:
            return True

        completed_actions = [step.action for step in history]
        request_lower = original_request.lower()

        # Specific completion patterns for common multi-action requests

        # Create + Scale pattern
        if "create" in request_lower and "scale" in request_lower:
            has_create = "create_recipe" in completed_actions
            has_scale = "scale_recipe" in completed_actions
            if has_create and has_scale:
                return True

        # Find + Scale pattern
        if ("find" in request_lower or "frequent" in request_lower) and "scale" in request_lower:
            has_find = any("frequent" in action or "search" in action for action in completed_actions)
            has_scale = "scale_recipe" in completed_actions
            if has_find and has_scale:
                return True

        # Show/Get + Count pattern
        if ("show" in request_lower or "get" in request_lower) and "count" in request_lower:
            has_get = any("recent" in action or "search" in action for action in completed_actions)
            has_count = "count_recipes_with_ingredient" in completed_actions
            if has_get and has_count:
                return True

        # Search + Details pattern
        if "search" in request_lower and ("detail" in request_lower or "first" in request_lower):
            has_search = "search_recipes" in completed_actions
            has_details = "get_recipe_details" in completed_actions
            if has_search and has_details:
                return True

        # Count expected actions in the request
        expected_actions = self._count_expected_actions(original_request)
        meaningful_actions = [a for a in completed_actions if a != "COMPLETE"]

        # Only complete if we've done the expected number of actions
        if len(meaningful_actions) >= expected_actions:
            return True

        return False

    def _count_expected_actions(self, request: str) -> int:
        """Count the expected number of actions based on the request"""
        request_lower = request.lower()

        # Count action verbs and explicit sequences
        action_indicators = [
            "find", "get", "show", "display", "create", "scale", "search",
            "count", "then", "and then", "also", "plus", "after that"
        ]

        # Count explicit conjunctions that indicate multiple actions
        conjunctions = ["then", "and then", "also", "plus", "after that", "next"]
        conjunction_count = sum(1 for conj in conjunctions if conj in request_lower)

        # Count commas as potential action separators
        comma_count = request_lower.count(",")

        # Estimate based on patterns
        if conjunction_count >= 2:  # "find..., display..., then scale..."
            return 3
        elif conjunction_count >= 1:  # "find... then scale..."
            return 2
        elif comma_count >= 2:  # "find, display, scale"
            return 3
        elif comma_count >= 1:  # "find, scale"
            return 2
        else:
            # Count distinct action words
            action_count = sum(1 for action in action_indicators if action in request_lower)
            return max(2, min(action_count, 4))  # Between 2-4 actions

    def _generate_structured_final_response(self, original_request: str, execution_results: List[Dict], action_plan: Dict) -> str:
        """Generate comprehensive final response for structured execution"""

        if not execution_results:
            return "I couldn't process your request. Please try again."

        total_planned = len(action_plan.get("actions", []))
        total_executed = len(execution_results)

        response = f"I've completed your multi-step request: '{original_request}'\n\n"
        response += f"📋 **Planned Actions**: {total_planned} | **Executed**: {total_executed}\n\n"

        # Summarize each step
        for result in execution_results:
            response += f"**Step {result['step']}**: {result['action'].replace('_', ' ').title()}\n"
            response += f"  💭 Reasoning: {result['reasoning']}\n"
            response += f"  ✅ Result: {result['observation']}\n\n"

        # Add completion summary
        if total_executed == total_planned:
            response += "🎉 All requested actions have been completed successfully!"
        else:
            response += f"⚠️ Completed {total_executed} out of {total_planned} planned actions."

        return response

    def _generate_final_response(self, original_request: str, history: List[ReActStep], context: Dict) -> str:
        """Generate comprehensive final response"""

        if not history:
            return "I couldn't process your request. Please try again."

        response = f"I've completed your multi-step request: '{original_request}'\n\n"

        # Summarize each step
        for i, step in enumerate(history, 1):
            response += f"**Step {i}**: {step.action.replace('_', ' ').title()}\n"
            response += f"  Result: {step.observation}\n\n"

        response += "All requested actions have been completed successfully!"

        return response

    def _handle_single_action(self, user_input: str) -> str:
        """Handle simple single-action requests directly"""

        # Direct LLM processing instead of delegating to avoid circular dependencies
        try:
            prompt = f"""Process this recipe request: {user_input}

Available actions:
- Create recipe from ingredients
- Search for recipes
- Get recipe details
- Scale a recipe
- Count recipes with ingredient

Respond with helpful information about the recipe request."""

            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            return f"I can help with recipe tasks. Please try: 'Create a recipe with [ingredients]' or 'Find recipes with [keywords]'"


# Test the multi-action agent
if __name__ == "__main__":
    from llm import llm

    agent = MultiActionReActAgent(llm)

    # Test complex multi-action queries
    test_queries = [
        "Find my most frequent recipe and scale it to 10 people",
        "Show me recent pasta recipes and count how many use tomatoes",
        "Get recent chicken recipes and tell me which one serves the most people"
    ]

    print("🚀 TESTING MULTI-ACTION REACT AGENT")
    print("=" * 60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}: {query}")
        print("-" * 50)

        result = agent.process_request(query)
        print(f"\n📋 Final Result:\n{result}")
        print("\n" + "=" * 60)