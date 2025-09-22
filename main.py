# main.py - Clean Simplified Version
from typing import TypedDict, Literal, Annotated, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import re

from llm import generate_recipe_with_llm, generate_nutrition_info
from prompts import (
    INGREDIENTS_PROMPT,
    DIETARY_NEEDS_PROMPT,
    ANOTHER_RECIPE_PROMPT,
    NUTRITION_PROMPT_TEMPLATE,
)

# ========================================================================
# 1. SIMPLIFIED STATE - Only what we actually need
# ========================================================================
class RecipeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # Recipe storage: ingredient -> recipe content
    recipes: Dict[str, str]
    # Current conversation context
    last_recipe: Optional[str]
    # What the user wants to do next
    next_action: Optional[str]

# ========================================================================
# 2. RECIPE STORAGE UTILITIES
# ========================================================================
class RecipeManager:
    @staticmethod
    def normalize_ingredient(ingredient: str) -> str:
        """Normalize ingredient names for consistent storage"""
        return ingredient.lower().strip()
    
    @staticmethod
    def store_recipe(state: RecipeState, ingredient: str, recipe: str) -> None:
        """Store a recipe with normalized ingredient key"""
        key = RecipeManager.normalize_ingredient(ingredient)
        if "recipes" not in state:
            state["recipes"] = {}
        state["recipes"][key] = recipe
        state["last_recipe"] = recipe
        print(f"DEBUG: Stored recipe for '{key}'")
    
    @staticmethod
    def find_recipe(state: RecipeState, user_input: str) -> Optional[tuple[str, str]]:
        """Find a recipe based on user input. Returns (ingredient, recipe) or None"""
        recipes = state.get("recipes", {})
        user_input = user_input.lower()
        
        # Direct ingredient match
        for ingredient in recipes.keys():
            if ingredient in user_input:
                print(f"DEBUG: Found recipe for '{ingredient}'")
                return ingredient, recipes[ingredient]
        
        # Pattern matching for recipe requests
        patterns = [
            r'(?:recipe|dish|meal).*?(beef|lamb|chicken|pork|fish|salmon|pasta)',
            r'(beef|lamb|chicken|pork|fish|salmon|pasta).*?(?:recipe|dish|meal)',
            r'(?:the|my|that)\s+(beef|lamb|chicken|pork|fish|salmon|pasta)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, user_input)
            for match in matches:
                normalized = RecipeManager.normalize_ingredient(match)
                if normalized in recipes:
                    print(f"DEBUG: Pattern matched '{normalized}'")
                    return normalized, recipes[normalized]
        
        print("DEBUG: No recipe found")
        return None

# ========================================================================
# 3. INTENT CLASSIFICATION
# ========================================================================
def classify_intent_with_llm(user_input: str, state: RecipeState) -> str:
    """Use LLM to classify user intent - much more robust than keyword matching"""
    
    recipes = state.get("recipes", {})
    available_ingredients = list(recipes.keys())
    has_recipes = len(recipes) > 0
    has_last_recipe = bool(state.get("last_recipe"))
    
    # Build context for the LLM
    context = f"""
Available recipes in storage: {available_ingredients if available_ingredients else "None"}
Has previous/last recipe available: {has_last_recipe}
User request: "{user_input}"

Classify the user's intent into ONE of these categories:

1. "show_previous" - User wants to see the most recent/last/previous recipe
   Examples: "previous recipe", "last one", "show me the recent recipe", "what was that recipe"

2. "show_specific" - User wants a specific recipe from history by ingredient
   Examples: "beef recipe", "show me the chicken dish", "I want that lamb recipe from before"

3. "nutrition" - User wants nutritional information about a recipe
   Examples: "nutrition info", "calories", "how healthy is it", "nutritional breakdown"

4. "new_recipe" - User wants to create a new recipe (DEFAULT)
   Examples: "new recipe", "cook something", "make a dish", "create recipe"

Respond with ONLY the category name (show_previous, show_specific, nutrition, or new_recipe).
"""
    
    try:
        print(f"DEBUG: Asking LLM to classify: '{user_input}'")
        
        from llm import llm  # Import the LLM instance
        response = llm.invoke(context)
        intent = response.content.strip().lower()
        
        # Ensure we get a valid intent
        valid_intents = ["show_previous", "show_specific", "nutrition", "new_recipe"]
        if intent not in valid_intents:
            print(f"DEBUG: LLM returned invalid intent '{intent}', defaulting to new_recipe")
            intent = "new_recipe"
        
        print(f"DEBUG: LLM classified intent as: '{intent}'")
        return intent
        
    except Exception as e:
        print(f"DEBUG: Error in LLM intent classification: {e}, falling back to keyword matching")
        # Fallback to simple keyword matching if LLM fails
        return classify_intent_fallback(user_input, state)

def classify_intent_fallback(user_input: str, state: RecipeState) -> str:
    """Fallback keyword-based classification if LLM fails"""
    user_input = user_input.lower().strip()
    
    # Previous/last recipe
    previous_keywords = ["previous", "last", "recent", "earlier", "before"]
    if any(word in user_input for word in previous_keywords):
        return "show_previous"
    
    # Specific recipe lookup
    recipe_found = RecipeManager.find_recipe(state, user_input)
    if recipe_found:
        return "show_specific"
    
    # Nutrition info
    nutrition_keywords = ["nutrition", "calories", "protein", "nutrients", "nutritional"]
    if any(word in user_input for word in nutrition_keywords):
        return "nutrition"
    
    # Default to new recipe
    return "new_recipe"

# ========================================================================
# 4. MAIN CONVERSATION NODE
# ========================================================================
def conversation_node(state: RecipeState):
    """Main conversation handler - processes user input and decides what to do"""
    
    # Get user input
    print("\nAI: Hello! I can help you with recipes. What would you like to do?")
    print("- Ask for a new recipe")
    print("- Get a previous recipe")
    print("- Get nutritional info")
    user_input = input("User: ")
    
    # Classify what the user wants using LLM
    intent = classify_intent_with_llm(user_input, state)
    print(f"DEBUG: LLM classified intent as '{intent}'")
    
    # Add user message to conversation
    user_msg = HumanMessage(content=user_input)
    
    return {
        "messages": [user_msg],
        "next_action": intent
    }

# ========================================================================
# 5. ACTION HANDLERS
# ========================================================================
def show_previous_recipe_node(state: RecipeState):
    """Show the last generated recipe"""
    last_recipe = state.get("last_recipe")
    
    print(f"DEBUG: Looking for last_recipe in state")
    print(f"DEBUG: last_recipe exists: {bool(last_recipe)}")
    print(f"DEBUG: recipes in storage: {list(state.get('recipes', {}).keys())}")
    
    if not last_recipe:
        # Try to get the most recent recipe from storage
        recipes = state.get("recipes", {})
        if recipes:
            # Get the last added recipe (this is a simple approach)
            last_key = list(recipes.keys())[-1]
            last_recipe = recipes[last_key]
            print(f"DEBUG: Retrieved last recipe for '{last_key}' from storage")
        else:
            response = "I don't have any previous recipes to show. Let's create a new one!"
            next_action = "new_recipe"
            ai_msg = AIMessage(content=response)
            print(f"\nAI: {response}")
            return {
                "messages": [ai_msg],
                "next_action": next_action
            }
    
    response = f"Here's your previous recipe:\n\n{last_recipe}\n\nWould you like nutritional info? (yes/no)"
    next_action = "await_nutrition_response"
    
    ai_msg = AIMessage(content=response)
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    user_msg = HumanMessage(content=user_input)
    next_action = "nutrition" if "yes" in user_input.lower() else "continue"
    
    return {
        "messages": [ai_msg, user_msg],
        "last_recipe": last_recipe,  # Make sure this is set for nutrition
        "next_action": next_action
    }

def show_specific_recipe_node(state: RecipeState):
    """Show a specific recipe from history using LLM to find the right one"""
    last_message = state["messages"][-1].content if state["messages"] else ""
    recipes = state.get("recipes", {})
    
    if not recipes:
        response = "I don't have any recipes stored yet. Let's create a new one!"
        next_action = "new_recipe"
        ai_msg = AIMessage(content=response)
        print(f"\nAI: {response}")
        return {
            "messages": [ai_msg],
            "next_action": next_action
        }
    
    # Use LLM to find the best matching recipe
    recipe_context = f"""
User request: "{last_message}"
Available recipes: {list(recipes.keys())}

The user is asking for a specific recipe from their history. Which recipe key from the available recipes best matches their request?

Respond with ONLY the exact recipe key from the list, or "none" if no good match exists.
"""
    
    try:
        from llm import llm
        response = llm.invoke(recipe_context)
        matched_key = response.content.strip().lower()
        
        if matched_key in recipes:
            recipe = recipes[matched_key]
            state["last_recipe"] = recipe  # Update for potential nutrition requests
            
            # Display the core ingredient name nicely
            display_ingredient = matched_key.replace(" recipe", "").replace(" dish", "").replace(" meal", "").strip()
            response = f"Here's your {display_ingredient} recipe:\n\n{recipe}\n\nWould you like nutritional info? (yes/no)"
            print(f"\nAI: {response}")
            user_input = input("User: ")
            user_msg = HumanMessage(content=user_input)
            next_action = "nutrition" if "yes" in user_input.lower() else "continue"
            
            ai_msg = AIMessage(content=response)
            return {
                "messages": [ai_msg, user_msg],
                "last_recipe": recipe,
                "recipes": recipes,
                "next_action": next_action
            }
        else:
            print(f"DEBUG: LLM couldn't match '{matched_key}' to available recipes")
            
    except Exception as e:
        print(f"DEBUG: Error in LLM recipe matching: {e}")
    
    # Fallback: try the original RecipeManager approach
    recipe_result = RecipeManager.find_recipe(state, last_message)
    if recipe_result:
        ingredient, recipe = recipe_result
        state["last_recipe"] = recipe
        response = f"Here's your {ingredient} recipe:\n\n{recipe}\n\nWould you like nutritional info? (yes/no)"
        print(f"\nAI: {response}")
        user_input = input("User: ")
        user_msg = HumanMessage(content=user_input)
        next_action = "nutrition" if "yes" in user_input.lower() else "continue"
        
        ai_msg = AIMessage(content=response)
        return {
            "messages": [ai_msg, user_msg],
            "last_recipe": recipe,
            "recipes": recipes,
            "next_action": next_action
        }
    
    # No match found
    response = "I couldn't find that recipe. Let me help you create a new one!"
    next_action = "new_recipe"
    ai_msg = AIMessage(content=response)
    print(f"\nAI: {response}")
    return {
        "messages": [ai_msg],
        "recipes": recipes,
        "next_action": next_action
    }

def create_new_recipe_node(state: RecipeState):
    """Create a new recipe through ingredient and dietary needs gathering"""
    
    # Step 1: Get ingredients
    print(f"\nAI: {INGREDIENTS_PROMPT}")
    ingredients = input("User: ")
    
    # Step 2: Get dietary needs
    print(f"\nAI: {DIETARY_NEEDS_PROMPT}")
    dietary_needs = input("User: ")
    
    # Step 3: Generate recipe
    print("\nAI: Let me create a recipe for you...")
    recipe = generate_recipe_with_llm(ingredients, dietary_needs)
    
    # Step 4: Store the recipe and update state
    if "recipes" not in state:
        state["recipes"] = {}
    
    key = RecipeManager.normalize_ingredient(ingredients)
    state["recipes"][key] = recipe
    print(f"DEBUG: Stored recipe for '{key}'")
    print(f"DEBUG: Total recipes in storage: {len(state['recipes'])}")
    
    # Step 5: Show recipe and ask about nutrition
    response = f"Here's your recipe:\n\n{recipe}\n\nWould you like nutritional information? (yes/no)"
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    
    # Create messages for this interaction
    ingredient_msg = HumanMessage(content=ingredients)
    dietary_msg = HumanMessage(content=dietary_needs) 
    ai_msg = AIMessage(content=response)
    user_response_msg = HumanMessage(content=user_input)
    
    next_action = "nutrition" if "yes" in user_input.lower() else "continue"
    
    return {
        "messages": [ingredient_msg, dietary_msg, ai_msg, user_response_msg],
        "recipes": state["recipes"],
        "last_recipe": recipe,  # Critical: Set this for nutrition analysis
        "next_action": next_action
    }

def provide_nutrition_node(state: RecipeState):
    """Provide nutritional information for the current recipe"""
    last_recipe = state.get("last_recipe", "")
    
    print(f"DEBUG: Nutrition node - last_recipe exists: {bool(last_recipe)}")
    print(f"DEBUG: Nutrition node - recipes in state: {list(state.get('recipes', {}).keys())}")
    
    if not last_recipe:
        # Try to get from recipes if last_recipe is missing
        recipes = state.get("recipes", {})
        if recipes:
            last_key = list(recipes.keys())[-1]
            last_recipe = recipes[last_key]
            print(f"DEBUG: Using recipe for '{last_key}' for nutrition analysis")
        else:
            response = "I don't have a recipe to analyze for nutrition."
            ai_msg = AIMessage(content=response)
            print(f"\nAI: {response}")
            user_input = input("User: ")
            user_msg = HumanMessage(content=user_input)
            return {
                "messages": [ai_msg, user_msg],
                "next_action": "continue"
            }
    
    print("\nAI: Analyzing nutritional content...")
    nutrition_info = generate_nutrition_info(last_recipe)
    response = f"Nutritional Information:\n\n{nutrition_info}\n\n{ANOTHER_RECIPE_PROMPT}"
    
    print(f"\nAI: {response}")
    user_input = input("User: ")
    
    ai_msg = AIMessage(content=response)
    user_msg = HumanMessage(content=user_input)
    
    # The key fix: Always route back to continue (conversation) instead of trying to determine intent here
    next_action = "continue"
    
    return {
        "messages": [ai_msg, user_msg],
        "last_recipe": last_recipe,  # Preserve the recipe
        "recipes": state.get("recipes", {}),  # Preserve recipe storage
        "next_action": next_action
    }

def end_conversation_node(state: RecipeState):
    """End the conversation gracefully"""
    response = "Thanks for using the recipe assistant! Enjoy your cooking!"
    print(f"\nAI: {response}")
    
    ai_msg = AIMessage(content=response)
    return {
        "messages": [ai_msg],
        "next_action": "end"
    }

# ========================================================================
# 6. ROUTING FUNCTION
# ========================================================================
def route_next_action(state: RecipeState) -> Literal["show_previous", "show_specific", "new_recipe", "nutrition", "continue", "end"]:
    """Simple routing based on next_action"""
    next_action = state.get("next_action", "continue")
    print(f"DEBUG: Routing to '{next_action}'")
    return next_action

# ========================================================================
# 7. BUILD THE GRAPH
# ========================================================================
workflow = StateGraph(RecipeState)

# Add nodes
workflow.add_node("conversation", conversation_node)
workflow.add_node("show_previous", show_previous_recipe_node)
workflow.add_node("show_specific", show_specific_recipe_node) 
workflow.add_node("new_recipe", create_new_recipe_node)
workflow.add_node("nutrition", provide_nutrition_node)
workflow.add_node("end", end_conversation_node)

# Set entry point
workflow.set_entry_point("conversation")

# Add routing
workflow.add_conditional_edges(
    "conversation",
    route_next_action,
    {
        "show_previous": "show_previous",
        "show_specific": "show_specific",
        "new_recipe": "new_recipe",
        "nutrition": "nutrition",
        "continue": "conversation",
        "end": END
    }
)

# All action nodes route back to conversation or end
for node in ["show_previous", "show_specific", "new_recipe", "nutrition"]:
    workflow.add_conditional_edges(
        node,
        route_next_action,
        {
            "show_previous": "show_previous",
            "show_specific": "show_specific", 
            "new_recipe": "new_recipe",
            "nutrition": "nutrition",
            "continue": "conversation",
            "end": END
        }
    )

# Compile with checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ========================================================================
# 8. MAIN LOOP
# ========================================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "recipe_session"}}
    
    # Initialize state
    current_state = {
        "messages": [],
        "recipes": {},
        "last_recipe": None,
        "next_action": "continue"
    }
    
    print("🍳 Welcome to the Recipe Assistant!")
    print("=" * 40)
    
    try:
        while True:
            # Invoke the workflow with current state
            result = app.invoke(current_state, config=config)
            
            print(f"DEBUG: Result next_action: {result.get('next_action')}")
            print(f"DEBUG: Result has recipes: {bool(result.get('recipes'))}")
            print(f"DEBUG: Result has last_recipe: {bool(result.get('last_recipe'))}")
            
            if result is None or result.get("next_action") == "end":
                break
                
            # Update state for next iteration - preserve all important data
            current_state = {
                "messages": result.get("messages", []),
                "recipes": result.get("recipes", current_state.get("recipes", {})),
                "last_recipe": result.get("last_recipe", current_state.get("last_recipe")),
                "next_action": result.get("next_action", "continue")
            }
            
            print(f"DEBUG: Updated state - recipes: {list(current_state.get('recipes', {}).keys())}")
            print(f"DEBUG: Updated state - has last_recipe: {bool(current_state.get('last_recipe'))}")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye! Happy cooking!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Please restart the application.")