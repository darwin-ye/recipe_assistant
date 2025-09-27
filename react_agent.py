# react_agent.py - ReAct implementation for Recipe Assistant
"""
ReAct (Reasoning and Acting) agent for enhanced recipe assistance.
This adds tool-using capabilities with explicit reasoning steps.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
import json
import requests
from datetime import datetime
import re

# ========================================================================
# 1. DEFINE TOOLS FOR THE REACT AGENT
# ========================================================================

@tool
def search_online_recipes(query: str) -> str:
    """
    Search for recipes online based on ingredients or dish names.
    Returns recipe suggestions from the web.
    """
    # Simulated web search - in production, you'd use a real API
    # You could integrate with APIs like Spoonacular, Edamam, or TheMealDB
    try:
        # Example with a free API (TheMealDB)
        base_url = "https://www.themealdb.com/api/json/v1/1/search.php"
        response = requests.get(base_url, params={"s": query}, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("meals"):
                meal = data["meals"][0]
                return f"Found recipe: {meal['strMeal']}. Category: {meal['strCategory']}. Instructions preview: {meal['strInstructions'][:200]}..."
            else:
                return f"No online recipes found for '{query}'. Try different keywords."
        else:
            return "Unable to search online recipes at this time."
    except Exception as e:
        return f"Search failed: {str(e)}. Using local knowledge instead."

@tool
def calculate_recipe_scaling(original_servings: int, desired_servings: int, ingredients: str) -> str:
    """
    Calculate scaled ingredient amounts when adjusting serving sizes.

    Args:
        original_servings: Original number of servings
        desired_servings: Desired number of servings
        ingredients: JSON string of ingredients with amounts
    """
    try:
        scaling_factor = desired_servings / original_servings
        ingredients_list = json.loads(ingredients)

        scaled = []
        for ing in ingredients_list:
            if 'amount' in ing and ing['amount']:
                try:
                    # Extract numeric value
                    amount_str = ing['amount']
                    amount = float(re.findall(r'\d+\.?\d*', amount_str)[0])
                    scaled_amount = amount * scaling_factor

                    # Format nicely
                    if scaled_amount.is_integer():
                        scaled_amount = int(scaled_amount)
                    else:
                        scaled_amount = round(scaled_amount, 2)

                    scaled.append(f"• {scaled_amount} {ing.get('unit', '')} {ing['name']}")
                except:
                    scaled.append(f"• {ing['name']} (adjust to taste)")
            else:
                scaled.append(f"• {ing['name']}")

        return f"Scaled recipe from {original_servings} to {desired_servings} servings:\n" + "\n".join(scaled)
    except Exception as e:
        return f"Error scaling recipe: {str(e)}"

@tool
def scale_current_recipe(desired_servings: int) -> str:
    """
    Scale the current recipe to a different number of servings.
    This tool should be used when there's a current recipe in context.

    Args:
        desired_servings: Desired number of servings
    """
    # This will be replaced at runtime with actual recipe data
    return f"To use this tool, ensure there's a current recipe in context with serving information."

@tool
def find_ingredient_substitutes(ingredient: str, dietary_restriction: str = "") -> str:
    """
    Find substitutes for ingredients based on dietary restrictions or availability.
    
    Args:
        ingredient: The ingredient to substitute
        dietary_restriction: Optional dietary restriction (vegan, gluten-free, etc.)
    """
    # Common substitutions database
    substitutes = {
        "butter": {
            "vegan": ["coconut oil", "olive oil", "vegan butter", "avocado"],
            "healthy": ["greek yogurt", "applesauce", "mashed banana"],
            "general": ["margarine", "oil", "shortening"]
        },
        "eggs": {
            "vegan": ["flax eggs (1 tbsp ground flax + 3 tbsp water per egg)", 
                     "chia eggs", "mashed banana", "applesauce"],
            "general": ["egg substitute", "silken tofu"]
        },
        "milk": {
            "vegan": ["almond milk", "soy milk", "oat milk", "coconut milk"],
            "lactose-free": ["lactose-free milk", "almond milk", "soy milk"],
            "general": ["water + butter", "evaporated milk", "cream"]
        },
        "flour": {
            "gluten-free": ["almond flour", "rice flour", "oat flour", "coconut flour"],
            "general": ["whole wheat flour", "bread flour", "cake flour"]
        },
        "sugar": {
            "healthy": ["honey", "maple syrup", "stevia", "dates"],
            "general": ["brown sugar", "powdered sugar", "molasses"]
        }
    }
    
    ingredient_lower = ingredient.lower()
    restriction_lower = dietary_restriction.lower()
    
    # Find substitutes
    for key, subs in substitutes.items():
        if key in ingredient_lower:
            if restriction_lower and restriction_lower in subs:
                options = subs[restriction_lower]
            else:
                options = subs.get("general", [])
            
            if options:
                return f"Substitutes for {ingredient}:\n" + "\n".join([f"• {opt}" for opt in options])
    
    # Generic response if no specific substitute found
    return f"No specific substitutes found for {ingredient}. Consider using similar ingredients or omitting if optional."

@tool
def estimate_cooking_time(dish_type: str, cooking_method: str, main_protein: str = "") -> str:
    """
    Estimate cooking time based on dish type and method.
    
    Args:
        dish_type: Type of dish (stir-fry, roast, soup, etc.)
        cooking_method: Cooking method (oven, stovetop, slow-cooker, etc.)
        main_protein: Optional main protein to consider
    """
    cooking_times = {
        "stir-fry": {"prep": 15, "cook": 10, "total": 25},
        "roast": {"prep": 20, "cook": 60, "total": 80},
        "soup": {"prep": 20, "cook": 30, "total": 50},
        "salad": {"prep": 15, "cook": 0, "total": 15},
        "pasta": {"prep": 10, "cook": 20, "total": 30},
        "grill": {"prep": 15, "cook": 15, "total": 30},
        "bake": {"prep": 20, "cook": 35, "total": 55},
        "slow-cooker": {"prep": 15, "cook": 240, "total": 255}
    }
    
    # Protein cooking adjustments
    protein_adjustments = {
        "chicken": 0,
        "beef": 5,
        "pork": 5,
        "fish": -5,
        "tofu": -10,
        "beans": 10
    }
    
    dish_lower = dish_type.lower()
    times = cooking_times.get(dish_lower, {"prep": 20, "cook": 30, "total": 50})
    
    # Adjust for protein
    if main_protein:
        protein_lower = main_protein.lower()
        for protein, adjustment in protein_adjustments.items():
            if protein in protein_lower:
                times["cook"] += adjustment
                times["total"] += adjustment
                break
    
    return (f"Estimated times for {dish_type}:\n"
            f"• Prep time: {times['prep']} minutes\n"
            f"• Cooking time: {times['cook']} minutes\n"
            f"• Total time: {times['total']} minutes")

@tool
def search_recipe_database(query: str) -> str:
    """
    Search the local recipe database using semantic search.

    Args:
        query: Search query for recipes (ingredients, dish names, etc.)
    """
    from recipe_models import RecipeDatabase

    db = RecipeDatabase()
    results = db.search(query, top_k=5)

    if results:
        output = [f"Found {len(results)} matching recipes:\n"]
        for i, (recipe_id, recipe, score) in enumerate(results, 1):
            output.append(f"{i}. {recipe.title} (match: {score:.0%})")
            output.append(f"   Serves: {recipe.servings} | Ingredients: {', '.join(recipe.main_ingredients[:3])}")
        return "\n".join(output)
    else:
        return f"No recipes found matching '{query}'"

@tool
def get_recipe_details(recipe_title: str) -> str:
    """
    Get full details of a specific recipe by title.

    Args:
        recipe_title: Title of the recipe to retrieve
    """
    from recipe_models import RecipeDatabase

    db = RecipeDatabase()

    # Find recipe by title (case-insensitive partial match)
    for recipe in db.recipes.values():
        if recipe_title.lower() in recipe.title.lower():
            return recipe.to_display_string()

    return f"Recipe '{recipe_title}' not found in database"

@tool
def get_recent_recipes(limit: int = 5) -> str:
    """
    Get the most recently created recipes.

    Args:
        limit: Number of recent recipes to return (default 5)
    """
    from recipe_models import RecipeDatabase

    db = RecipeDatabase()
    recent = db.get_recent_recipes(limit)

    if recent:
        output = [f"Here are your {len(recent)} most recent recipes:\n"]
        for i, recipe in enumerate(recent, 1):
            output.append(f"{i}. {recipe.title}")
            output.append(f"   Created: {recipe.created_at.strftime('%Y-%m-%d')} | Serves: {recipe.servings}")
        return "\n".join(output)
    else:
        return "No recipes in database yet"

@tool
def create_new_recipe(ingredients: str, dietary_needs: str = "") -> str:
    """
    Create a new recipe using AI generation.

    Args:
        ingredients: Main ingredients to use
        dietary_needs: Any dietary restrictions or preferences
    """
    try:
        from llm import generate_recipe_with_llm
        from recipe_models import create_recipe_from_llm_response, RecipeDatabase

        # Generate recipe using LLM
        llm_response = generate_recipe_with_llm(ingredients, dietary_needs)

        # Convert to structured format
        recipe = create_recipe_from_llm_response(llm_response, ingredients, dietary_needs)

        # Save to database
        db = RecipeDatabase()
        recipe_id = db.add_recipe(recipe)

        # Return the full recipe details including cooking instructions
        recipe_display = recipe.to_display_string()

        return f"✅ Created and saved new recipe!\n\n{recipe_display}\n\n📝 Recipe saved to database with ID: {recipe_id}"

    except Exception as e:
        return f"Error creating recipe: {str(e)}"

@tool
def get_nutrition_info(recipe_title: str) -> str:
    """
    Get nutritional information for a recipe.

    Args:
        recipe_title: Title of the recipe to analyze
    """
    from recipe_models import RecipeDatabase, add_nutrition_to_recipe
    from llm import generate_nutrition_info

    db = RecipeDatabase()

    # Find recipe
    for recipe_id, recipe in db.recipes.items():
        if recipe_title.lower() in recipe.title.lower():

            # Check if nutrition already exists
            if recipe.nutrition and recipe.nutrition.calories:
                return recipe.to_nutrition_string()
            else:
                # Generate nutrition info
                recipe_text = recipe.raw_text or recipe.to_display_string()
                nutrition_text = generate_nutrition_info(recipe_text)

                # Add to recipe and save
                updated_recipe = add_nutrition_to_recipe(recipe, nutrition_text)
                db.recipes[recipe_id] = updated_recipe
                db.save_recipes()

                return updated_recipe.to_nutrition_string()

    return f"Recipe '{recipe_title}' not found"

@tool
def find_similar_recipes(recipe_title: str) -> str:
    """
    Find recipes similar to the specified recipe.

    Args:
        recipe_title: Title of the recipe to find similar ones for
    """
    from recipe_models import RecipeDatabase

    db = RecipeDatabase()

    # Find the target recipe
    target_recipe = None
    for recipe in db.recipes.values():
        if recipe_title.lower() in recipe.title.lower():
            target_recipe = recipe
            break

    if not target_recipe:
        return f"Recipe '{recipe_title}' not found"

    # Find similar recipes
    similar = db.search_engine.find_similar_recipes(target_recipe, db.recipes, top_k=3)

    if similar:
        output = [f"Recipes similar to '{target_recipe.title}':\n"]
        for i, (recipe_id, recipe, score) in enumerate(similar, 1):
            output.append(f"{i}. {recipe.title} (similarity: {score:.0%})")
            output.append(f"   Serves: {recipe.servings} | Main ingredients: {', '.join(recipe.main_ingredients[:3])}")
        return "\n".join(output)
    else:
        return f"No similar recipes found for '{recipe_title}'"

# ========================================================================
# 2. REACT AGENT STATE AND PROMPTS
# ========================================================================

REACT_SYSTEM_PROMPT = """You are a helpful recipe assistant that uses ReAct (Reasoning and Acting) to handle all user requests naturally.

CRITICAL: When you need a tool, you MUST use this EXACT format:
ACTION: tool_name(parameter_name=value)

Example: ACTION: search_recipe_database(query=chicken)

**THESE ARE THE ONLY VALID TOOLS - DO NOT USE ANY OTHERS:**

**Recipe Database Operations:**
- search_recipe_database(query): Search recipes by ingredients, name, or description
- get_recipe_details(recipe_title): Get full recipe details
- get_recent_recipes(limit): Show recent recipes
- create_new_recipe(ingredients, dietary_needs): Create new recipe from ingredients and show full details
- get_nutrition_info(recipe_title): Get nutritional analysis
- find_similar_recipes(recipe_title): Find similar recipes

**Recipe Modification:**
- scale_current_recipe(desired_servings): Scale current recipe for different servings
- find_ingredient_substitutes(ingredient, dietary_restriction): Find ingredient substitutes

**External & Analysis:**
- search_online_recipes(query): Search web for recipes
- estimate_cooking_time(dish_type, cooking_method): Estimate cooking times

**CRITICAL INTENT DETECTION RULES:**

1. **CREATE/NEW RECIPE REQUESTS:**
   - Keywords: "create", "make", "new", "generate", "come up with"
   - ALWAYS use create_new_recipe IMMEDIATELY - DO NOT search first
   - If user gives ingredients, use them: create_new_recipe(ingredients=user_ingredients, dietary_needs=any_restrictions)
   - If no specific ingredients, ask user what they want: create_new_recipe(ingredients=, dietary_needs=)

2. **RECIPE INSTRUCTIONS/STEPS:**
   - Keywords: "steps", "instructions", "how to cook", "how to make", "directions"
   - Use get_recipe_details(recipe_title=specific_recipe_name)

3. **SEARCH EXISTING RECIPES:**
   - Keywords: "find", "search", "show me", "look for"
   - Use search_recipe_database(query=search_terms)

4. **AMBIGUOUS INPUTS:**
   - For unclear requests like "yes", "ok", "help": provide helpful guidance instead of using tools
   - Explain what you can do and ask specific questions
   - Examples: "I can help you create recipes, search existing ones, or get cooking instructions. What would you like to do?"
   - Do NOT guess or use random tools

**Parameter Validation Rules:**
- create_new_recipe: If ingredients empty, gather user requirements first
- get_recipe_details: Must have valid recipe title
- search_recipe_database: Must have meaningful search query

**Important Instructions:**
1. Always start with "THOUGHT:" to analyze what the user needs
2. Use "ACTION: tool_name(parameter=value)" when you need a tool
3. Do NOT make up tool results - wait for real OBSERVATION
4. End with "FINAL ANSWER:" followed by your complete helpful response
5. NEVER use tools not in the list above

EXAMPLES:
User: "Create a shrimp recipe"
CORRECT: ACTION: create_new_recipe(ingredients=shrimp, dietary_needs=)

User: "create a new recipe"
CORRECT: ACTION: create_new_recipe(ingredients=, dietary_needs=)

User: "show me the cooking steps"
CORRECT: Ask user which recipe they want steps for, then use get_recipe_details

Think step by step and use tools appropriately."""

# ========================================================================
# 3. REACT AGENT IMPLEMENTATION
# ========================================================================

class ReActAgent:
    """ReAct agent for recipe assistance with reasoning and tool use"""
    
    def __init__(self, llm):
        self.llm = llm
        self.tools = [
            # Recipe database operations
            search_recipe_database,
            get_recipe_details,
            get_recent_recipes,
            create_new_recipe,
            get_nutrition_info,
            find_similar_recipes,

            # Recipe modification
            scale_current_recipe,
            calculate_recipe_scaling,
            find_ingredient_substitutes,

            # External and analysis
            search_online_recipes,
            estimate_cooking_time
        ]
        self.current_recipe = None
        
        # Create the ReAct prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", REACT_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Bind tools to LLM (for Ollama, we'll use a simpler approach)
        self.setup_agent()
    
    def setup_agent(self):
        """Setup the ReAct agent with tool handling"""
        # For Ollama/Llama3, we need a different approach than OpenAI functions
        # We'll use a more explicit ReAct loop
        pass
    
    def parse_action(self, text: str) -> tuple:
        """Parse action from LLM response with improved reliability"""
        # Multiple patterns to catch different formats
        patterns = [
            r'ACTION:\s*(\w+)\((.*?)\)',           # ACTION: tool_name(args)
            r'ACTION:\s*(\w+)\s*\((.*?)\)',        # ACTION: tool_name (args)
            r'I will use (\w+)\((.*?)\)',          # I will use tool_name(args)
            r'Let me (\w+)\((.*?)\)',              # Let me tool_name(args)
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                tool_name = match.group(1)
                args_str = match.group(2)

                # Parse arguments more robustly
                args = {}
                if args_str.strip():
                    # Handle simple cases first
                    if '=' in args_str:
                        # Parse key=value pairs
                        arg_parts = [part.strip() for part in args_str.split(',')]
                        for part in arg_parts:
                            if '=' in part:
                                key, value = part.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")
                                args[key] = value
                    else:
                        # Assume it's a single query parameter
                        args['query'] = args_str.strip().strip('"').strip("'")

                print(f"🎯 Parsed action: {tool_name} with args: {args}")
                return tool_name, args

        # If no action found, try to infer from intent with priority for create actions
        text_lower = text.lower()

        # PRIORITY 1: Creation keywords (always check first)
        if any(word in text_lower for word in ["create", "make", "new recipe", "new", "generate", "cook"]):
            print("🎨 Detected CREATE intent - forcing create_new_recipe")
            # Extract ingredients
            if "with" in text_lower:
                ingredients_part = text_lower.split("with")[1].strip()
                return "create_new_recipe", {"ingredients": ingredients_part, "dietary_needs": ""}
            elif "pasta" in text_lower and "tomato" in text_lower:
                return "create_new_recipe", {"ingredients": "pasta, tomatoes", "dietary_needs": ""}
            else:
                # Extract ingredients from the whole query
                words = text_lower.split()
                ingredients = [word for word in words if word in ["pasta", "chicken", "beef", "tomato", "tomatoes", "rice", "fish"]]
                return "create_new_recipe", {"ingredients": ", ".join(ingredients) or "pasta", "dietary_needs": ""}

        # PRIORITY 2: Recipe details/instructions keywords
        elif any(word in text_lower for word in ["steps", "instructions", "how to cook", "recipe information", "cooking"]):
            print("📋 Detected INSTRUCTIONS intent - getting recipe details")
            # Try to find the most recent recipe or mentioned recipe
            return "get_recipe_details", {"recipe_title": "Pasta alla Pescatora"}

        # PRIORITY 3: Search keywords (only if not a create request)
        elif any(word in text_lower for word in ["search", "find", "show me", "list"]):
            if "chicken" in text_lower:
                return "search_recipe_database", {"query": "chicken"}
            elif "recent" in text_lower:
                return "get_recent_recipes", {"limit": 5}

        # PRIORITY 4: Handle ambiguous inputs
        elif text_lower.strip() in ["yes", "ok", "okay", "sure", "help", "what can you do", ""]:
            print("❓ Detected ambiguous input - providing clarification")
            # Don't return a tool, let it fall through to explain capabilities
            return None, None

        return None, None
    
    def execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool by name with arguments"""

        # Validate parameters before execution
        validation_error = self._validate_tool_parameters(tool_name, args)
        if validation_error:
            return validation_error

        # Handle scale_current_recipe specially with current recipe context
        if tool_name == "scale_current_recipe":
            if not self.current_recipe:
                return "No current recipe available to scale. Please select a recipe first."

            try:
                desired_servings = int(args.get('desired_servings', 0))
                if desired_servings <= 0:
                    return "Please provide a valid number of servings (greater than 0)."

                return self._scale_recipe(self.current_recipe, desired_servings)
            except Exception as e:
                return f"Error scaling current recipe: {str(e)}"

        # Handle other tools normally
        tool_map = {
            # Recipe database operations
            "search_recipe_database": search_recipe_database,
            "get_recipe_details": get_recipe_details,
            "get_recent_recipes": get_recent_recipes,
            "create_new_recipe": create_new_recipe,
            "get_nutrition_info": get_nutrition_info,
            "find_similar_recipes": find_similar_recipes,

            # Recipe modification
            "calculate_recipe_scaling": calculate_recipe_scaling,
            "find_ingredient_substitutes": find_ingredient_substitutes,

            # External and analysis
            "search_online_recipes": search_online_recipes,
            "estimate_cooking_time": estimate_cooking_time
        }

        if tool_name in tool_map:
            try:
                tool_func = tool_map[tool_name]
                # Use invoke method for LangChain tools
                if hasattr(tool_func, 'invoke'):
                    if tool_name == 'create_new_recipe':
                        # Handle create_new_recipe specially - it expects two separate parameters
                        ingredients = args.get('ingredients', '')
                        dietary_needs = args.get('dietary_needs', '')
                        return tool_func.invoke({"ingredients": ingredients, "dietary_needs": dietary_needs})
                    elif tool_name == 'get_recipe_details':
                        # Handle get_recipe_details specially
                        recipe_title = args.get('recipe_title', '')
                        return tool_func.invoke(recipe_title)
                    elif len(args) == 1 and 'query' in args:
                        return tool_func.invoke(args['query'])
                    elif len(args) == 0:
                        return tool_func.invoke("")
                    else:
                        return tool_func.invoke(args)
                else:
                    # Direct function call for non-LangChain tools
                    return tool_func(**args)
            except Exception as e:
                print(f"DEBUG: Tool execution error for {tool_name}: {e}")
                return f"Error executing {tool_name}: {str(e)}"
        else:
            return f"Unknown tool: {tool_name}"

    def _scale_recipe(self, recipe, desired_servings: int) -> str:
        """Scale a recipe to desired servings"""
        original_servings = recipe.servings
        scaling_factor = desired_servings / original_servings

        scaled_ingredients = []
        for ing in recipe.all_ingredients:
            if ing.amount:
                try:
                    # Extract numeric value
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
            else:
                scaled_ingredients.append(f"• {ing.name}")

        result = f"""Scaled '{recipe.title}' from {original_servings} to {desired_servings} servings:

**Scaled Ingredients:**
{chr(10).join(scaled_ingredients)}

**Instructions remain the same:**
{chr(10).join([f"{i+1}. {instr}" for i, instr in enumerate(recipe.instructions)])}

**Note:** Cooking times may need slight adjustments for larger quantities."""

        return result

    def _validate_tool_parameters(self, tool_name: str, args: dict) -> str:
        """Validate tool parameters before execution"""

        # Validation for create_new_recipe
        if tool_name == "create_new_recipe":
            ingredients = args.get('ingredients', '').strip()
            if not ingredients:
                return "To create a recipe, I need ingredients. What ingredients would you like to use? For example: 'chicken, tomatoes, pasta' or 'beef, onions, potatoes'"

        # Validation for get_recipe_details
        elif tool_name == "get_recipe_details":
            recipe_title = args.get('recipe_title', '').strip()
            if not recipe_title:
                return "To show recipe details, I need the recipe name. Which recipe would you like to see the full instructions for?"

        # Validation for search_recipe_database
        elif tool_name == "search_recipe_database":
            query = args.get('query', '').strip()
            if not query or len(query) < 2:
                return "To search recipes, I need specific search terms. What are you looking for? (ingredients, dish name, cuisine type, etc.)"

        # Validation for get_nutrition_info
        elif tool_name == "get_nutrition_info":
            recipe_title = args.get('recipe_title', '').strip()
            if not recipe_title:
                return "To get nutrition information, I need a specific recipe name. Which recipe's nutrition would you like to see?"

        # Validation for find_similar_recipes
        elif tool_name == "find_similar_recipes":
            recipe_title = args.get('recipe_title', '').strip()
            if not recipe_title:
                return "To find similar recipes, I need a reference recipe name. Which recipe should I use as a reference?"

        # Validation for find_ingredient_substitutes
        elif tool_name == "find_ingredient_substitutes":
            ingredient = args.get('ingredient', '').strip()
            if not ingredient:
                return "To find substitutes, I need to know which ingredient you want to substitute. What ingredient are you looking to replace?"

        # Validation for search_online_recipes
        elif tool_name == "search_online_recipes":
            query = args.get('query', '').strip()
            if not query or len(query) < 2:
                return "To search online recipes, I need specific search terms. What type of recipe are you looking for?"

        # Validation for estimate_cooking_time
        elif tool_name == "estimate_cooking_time":
            dish_type = args.get('dish_type', '').strip()
            if not dish_type:
                return "To estimate cooking time, I need to know what dish you're making. What type of dish is it?"

        return None  # No validation errors

    def run(self, user_input: str, chat_history: List = None, current_recipe: Any = None) -> str:
        """Run the ReAct loop for a user query"""
        if chat_history is None:
            chat_history = []

        # Set current recipe for tool access
        self.current_recipe = current_recipe

        # Build context with current recipe info
        context_info = ""
        if current_recipe:
            context_info = f"""
Current Recipe Context:
- Title: {current_recipe.title}
- Servings: {current_recipe.servings}
- Main ingredients: {', '.join(current_recipe.main_ingredients)}
- Has {len(current_recipe.all_ingredients)} total ingredients
- Has {len(current_recipe.instructions)} instructions
"""

        # Build the ReAct prompt with more explicit instructions
        react_prompt = f"""
{REACT_SYSTEM_PROMPT}

{context_info}

User Query: {user_input}

You MUST follow this format:
1. Start with "THOUGHT:" and analyze what the user needs
2. If you need a tool, write "ACTION: tool_name(parameter=value)"
3. Wait for "OBSERVATION:" with tool results
4. Continue thinking or provide "FINAL ANSWER:"

Let's solve this step by step.

THOUGHT: """
        
        # Run the ReAct loop (maximum 5 iterations)
        max_iterations = 5

        print(f"🔍 Starting ReAct reasoning for: '{user_input[:50]}...'")

        for i in range(max_iterations):
            print(f"\n🔄 ReAct Step {i+1}/{max_iterations}")

            # Get LLM response
            response = self.llm.invoke(react_prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Show the LLM's thinking
            print(f"💭 AI Thinking: {response_text[:150]}...")

            # Check if we have a final answer
            if "FINAL ANSWER:" in response_text:
                final_answer = response_text.split("FINAL ANSWER:")[-1].strip()
                print(f"✅ Final Answer Ready: {final_answer[:100]}...")
                return final_answer

            # Parse action if present
            tool_name, args = self.parse_action(response_text)

            if tool_name:
                print(f"🔧 Using Tool: {tool_name} with parameters {args}")

                # Execute tool
                observation = self.execute_tool(tool_name, args)
                print(f"📊 Tool Result: {observation[:100]}...")

                # Add to conversation
                react_prompt += response_text + f"\n\nOBSERVATION: {observation}\n\nTHOUGHT: "
            else:
                print("⚠️  No tool action detected, continuing reasoning...")
                react_prompt += response_text + "\n\nTHOUGHT: "

            # Check if we should continue
            if "no more tools needed" in response_text.lower() or "final answer" in response_text.lower():
                print("🏁 Reasoning complete, preparing final answer...")
                break
        
        # If no final answer was provided, force one
        print("⚠️  ReAct loop completed without final answer, forcing completion...")

        # Get a final response from LLM
        final_prompt = react_prompt + "\n\nPlease provide your FINAL ANSWER now based on all the above reasoning:\n\nFINAL ANSWER: "
        final_response = self.llm.invoke(final_prompt)
        final_text = final_response.content if hasattr(final_response, 'content') else str(final_response)

        if "FINAL ANSWER:" in final_text:
            return final_text.split("FINAL ANSWER:")[-1].strip()
        else:
            return final_text.strip()
    
    def synthesize_answer(self, user_input: str, thoughts: List[str]) -> str:
        """Synthesize a final answer from the thought process"""
        synthesis_prompt = f"""
Based on the following reasoning process, provide a clear, helpful answer to the user's question.

User Question: {user_input}

Reasoning Process:
{' '.join(thoughts)}

Provide a concise, practical answer:
"""
        
        response = self.llm.invoke(synthesis_prompt)
        return response.content if hasattr(response, 'content') else str(response)

# ========================================================================
# 4. INTEGRATION WITH MAIN RECIPE ASSISTANT
# ========================================================================

def create_react_recipe_node(llm):
    """Create a ReAct node for the recipe assistant"""
    
    react_agent = ReActAgent(llm)
    
    def react_node(state):
        """ReAct node that can be added to the LangGraph workflow"""
        last_message = state["messages"][-1].content if state["messages"] else ""
        
        # Check if this is a complex query that needs ReAct
        complex_keywords = [
            "how long", "substitute", "replace", "instead of",
            "scale", "adjust servings", "online recipe",
            "cooking time", "find recipe"
        ]
        
        needs_react = any(keyword in last_message.lower() for keyword in complex_keywords)
        
        if needs_react:
            print("\n🤔 Using ReAct reasoning to answer your question...")
            result = react_agent.run(last_message)
            
            response = f"Based on my analysis:\n\n{result}"
        else:
            response = "This query doesn't require complex reasoning. Please use the standard recipe features."
        
        print(f"\nAI: {response}")
        ai_msg = AIMessage(content=response)
        
        return {
            "messages": [ai_msg],
            "next_action": "continue"
        }
    
    return react_node

# ========================================================================
# 5. EXAMPLE USAGE
# ========================================================================

if __name__ == "__main__":
    from llm import llm
    
    # Create ReAct agent
    agent = ReActAgent(llm)
    
    # Test queries
    test_queries = [
        "I need to make beef stir-fry for 8 people instead of 4. Help me scale the recipe.",
        "What can I use instead of eggs in baking if I'm vegan?",
        "How long does it take to make a chicken roast?",
        "Find me an online recipe for pad thai"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"USER: {query}")
        print(f"{'='*60}")
        
        response = agent.run(query)
        print(f"\nASSISTANT: {response}")