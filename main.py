# main.py - Complete Recipe Assistant with All Fixes Applied
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

# Import the new structured models
from recipe_models import (
    Recipe, 
    RecipeDatabase, 
    RecipeParser,
    create_recipe_from_llm_response,
    add_nutrition_to_recipe
)

# ========================================================================
# 1. UPDATED STATE - Now uses structured recipes
# ========================================================================
class RecipeState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # Current recipe being worked on
    current_recipe: Optional[Recipe]
    # Recipe database instance
    recipe_db: RecipeDatabase
    # What the user wants to do next
    next_action: Optional[str]
    # Last search results for reference
    last_search_results: Optional[List[tuple]]

# Initialize the database globally
recipe_database = RecipeDatabase()

# ========================================================================
# 2. INTENT CLASSIFICATION (FIXED for better LLM response handling)
# ========================================================================
def classify_intent_with_llm(user_input: str, state: RecipeState) -> str:
    """Use LLM to classify user intent - fixed to handle LLM response format"""
    
    recipe_db = state.get("recipe_db", recipe_database)
    has_recipes = len(recipe_db.recipes) > 0
    has_current_recipe = state.get("current_recipe") is not None
    
    # Simplified prompt for better LLM response
    context = f"""
User said: "{user_input}"

What does the user want to do? Choose ONE:
- search_recipes (find existing recipes)  
- show_recent (see recent/previous recipes)
- nutrition (get nutritional info)
- new_recipe (create new recipe)
- show_similar (find similar recipes)

Reply with ONLY the option name, nothing else.
"""
    
    try:
        print(f"DEBUG: Classifying intent for: '{user_input}'")
        
        from llm import llm
        response = llm.invoke(context)
        raw_intent = response.content.strip()
        
        # Clean up the LLM response - extract just the intent name
        # Handle various response formats
        intent = raw_intent.lower()
        
        # Remove quotes, numbers, periods, etc.
        intent = re.sub(r'["\'\d.\-\*]', '', intent).strip()
        
        # Extract intent name if it's in a sentence
        valid_intents = ["search_recipes", "show_recent", "nutrition", "new_recipe", "show_similar"]
        
        # Look for valid intent in the response
        for valid in valid_intents:
            if valid in intent:
                intent = valid
                break
        
        # If still not valid, use fallback
        if intent not in valid_intents:
            print(f"DEBUG: Extracted intent '{intent}' not valid, using fallback")
            intent = classify_intent_fallback(user_input, state)
        
        print(f"DEBUG: Classified as: '{intent}'")
        return intent
        
    except Exception as e:
        print(f"DEBUG: LLM classification failed: {e}")
        return classify_intent_fallback(user_input, state)

def classify_intent_fallback(user_input: str, state: RecipeState) -> str:
    """Enhanced fallback classification - better pattern matching"""
    user_input = user_input.lower().strip()
    
    # Search keywords
    search_keywords = ["find", "search", "show me", "recipes with", "dishes with", "what recipes"]
    if any(word in user_input for word in search_keywords):
        return "search_recipes"
    
    # Recent/history/previous keywords - ENHANCED
    recent_keywords = ["recent", "latest", "history", "previous", "past", "earlier", "last", "see recipes", "show recipes"]
    if any(word in user_input for word in recent_keywords):
        return "show_recent"
    
    # Similar recipes
    similar_keywords = ["similar", "like this", "more like", "same as"]
    if any(word in user_input for word in similar_keywords):
        return "show_similar"
    
    # Nutrition
    nutrition_keywords = ["nutrition", "calories", "protein", "healthy", "nutritional"]
    if any(word in user_input for word in nutrition_keywords):
        return "nutrition"
    
    # New recipe - be more specific
    new_keywords = ["new recipe", "create", "make", "cook", "generate"]
    if any(word in user_input for word in new_keywords):
        return "new_recipe"
    
    # Default to new recipe
    return "new_recipe"

# ========================================================================
# 3. MAIN CONVERSATION NODE
# ========================================================================
def conversation_node(state: RecipeState):
    """Main conversation handler"""
    
    # Ensure recipe_db is in state
    if "recipe_db" not in state:
        state["recipe_db"] = recipe_database
    
    recipe_count = len(state["recipe_db"].recipes)
    
    print("\n" + "="*50)
    print(f"AI: Hello! I have {recipe_count} recipes in my database.")
    print("What would you like to do?")
    print("- Search for recipes (e.g., 'find chicken recipes')")
    print("- Create a new recipe")
    print("- See recent recipes")
    if state.get("current_recipe"):
        print("- Get nutritional info for the current recipe")
        print("- Find similar recipes")
    
    user_input = input("\nUser: ")
    
    # Classify intent
    intent = classify_intent_with_llm(user_input, state)
    
    # Add user message
    user_msg = HumanMessage(content=user_input)
    
    return {
        "messages": [user_msg],
        "next_action": intent,
        "recipe_db": state["recipe_db"]
    }

# ========================================================================
# 4. SEARCH RECIPES NODE
# ========================================================================
def search_recipes_node(state: RecipeState):
    """Search for recipes using semantic search"""
    last_message = state["messages"][-1].content if state["messages"] else ""
    recipe_db = state.get("recipe_db", recipe_database)
    
    if not recipe_db.recipes:
        response = "No recipes in the database yet. Let's create your first recipe!"
        print(f"\nAI: {response}")
        ai_msg = AIMessage(content=response)
        return {
            "messages": [ai_msg],
            "next_action": "new_recipe",
            "recipe_db": recipe_db
        }
    
    # Perform semantic search
    print("\nAI: Searching for recipes...")
    results = recipe_db.search(last_message, top_k=5)
    
    if not results:
        response = "I couldn't find any recipes matching your search. Let's create a new one!"
        print(f"\nAI: {response}")
        user_input = input("User: ")
        user_msg = HumanMessage(content=user_input)
        ai_msg = AIMessage(content=response)
        return {
            "messages": [ai_msg, user_msg],
            "next_action": "new_recipe" if "yes" in user_input.lower() else "continue",
            "recipe_db": recipe_db
        }
    
    # Display search results
    response_parts = [f"Found {len(results)} matching recipes:\n"]
    for i, (recipe_id, recipe, score) in enumerate(results[:3]):  # Show top 3
        response_parts.append(f"\n{i+1}. **{recipe.title}**")
        if recipe.description:
            response_parts.append(f"   {recipe.description}")
        response_parts.append(f"   Main ingredients: {', '.join(recipe.main_ingredients[:3])}")
        if recipe.dietary_tags:
            response_parts.append(f"   Dietary: {', '.join(recipe.dietary_tags)}")
        response_parts.append(f"   Match score: {score:.0%}")
    
    response_parts.append("\n\nWhich recipe would you like to see? (Enter number, or 'new' for a new recipe)")
    response = "\n".join(response_parts)
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    user_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=response)
    
    # Process user selection
    current_recipe = None
    if user_input.strip().lower() == 'new':
        next_action = "new_recipe"
    else:
        try:
            selection = int(user_input.strip()) - 1
            if 0 <= selection < len(results):
                _, selected_recipe, _ = results[selection]
                current_recipe = selected_recipe
                
                # Display the selected recipe
                recipe_display = selected_recipe.to_display_string()
                print(f"\nAI: Here's your recipe:\n\n{recipe_display}")
                print("\nWould you like nutritional info, similar recipes, or something else?")
                
                followup_input = input("User: ")
                followup_msg = HumanMessage(content=followup_input)
                
                # Determine next action based on followup
                if "nutrition" in followup_input.lower():
                    next_action = "nutrition"
                elif "similar" in followup_input.lower():
                    next_action = "show_similar"
                else:
                    next_action = "continue"
                
                return {
                    "messages": [ai_msg, user_msg, followup_msg],
                    "current_recipe": selected_recipe,
                    "last_search_results": results,
                    "next_action": next_action,
                    "recipe_db": recipe_db
                }
            else:
                next_action = "continue"
        except (ValueError, IndexError):
            next_action = "continue"
    
    return {
        "messages": [ai_msg, user_msg],
        "current_recipe": current_recipe,
        "last_search_results": results,
        "next_action": next_action,
        "recipe_db": recipe_db
    }

# ========================================================================
# 5. SHOW RECENT RECIPES NODE
# ========================================================================
def show_recent_recipes_node(state: RecipeState):
    """Show recently created recipes"""
    recipe_db = state.get("recipe_db", recipe_database)
    
    recent_recipes = recipe_db.get_recent_recipes(limit=5)
    
    if not recent_recipes:
        response = "No recipes in the database yet. Let's create your first recipe!"
        print(f"\nAI: {response}")
        ai_msg = AIMessage(content=response)
        return {
            "messages": [ai_msg],
            "next_action": "new_recipe",
            "recipe_db": recipe_db
        }
    
    response_parts = [f"Here are your {len(recent_recipes)} most recent recipes:\n"]
    for i, recipe in enumerate(recent_recipes, 1):
        response_parts.append(f"\n{i}. **{recipe.title}**")
        response_parts.append(f"   Created: {recipe.created_at.strftime('%Y-%m-%d %H:%M')}")
        response_parts.append(f"   Main ingredients: {', '.join(recipe.main_ingredients[:3])}")
    
    response_parts.append("\n\nWhich recipe would you like to see? (Enter number)")
    response = "\n".join(response_parts)
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    user_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=response)
    
    try:
        selection = int(user_input.strip()) - 1
        if 0 <= selection < len(recent_recipes):
            selected_recipe = recent_recipes[selection]
            
            # Display the recipe
            recipe_display = selected_recipe.to_display_string()
            print(f"\nAI: Here's your recipe:\n\n{recipe_display}")
            print("\nWhat would you like to do next?")
            
            followup_input = input("User: ")
            followup_msg = HumanMessage(content=followup_input)
            
            return {
                "messages": [ai_msg, user_msg, followup_msg],
                "current_recipe": selected_recipe,
                "next_action": "continue",
                "recipe_db": recipe_db
            }
    except (ValueError, IndexError):
        pass
    
    return {
        "messages": [ai_msg, user_msg],
        "next_action": "continue",
        "recipe_db": recipe_db
    }

# ========================================================================
# 6. SHOW SIMILAR RECIPES NODE
# ========================================================================
def show_similar_recipes_node(state: RecipeState):
    """Find recipes similar to the current one"""
    recipe_db = state.get("recipe_db", recipe_database)
    current_recipe = state.get("current_recipe")
    
    if not current_recipe:
        response = "No current recipe selected. Please search for or create a recipe first."
        print(f"\nAI: {response}")
        ai_msg = AIMessage(content=response)
        return {
            "messages": [ai_msg],
            "next_action": "continue",
            "recipe_db": recipe_db
        }
    
    print(f"\nAI: Finding recipes similar to '{current_recipe.title}'...")
    
    similar = recipe_db.search_engine.find_similar_recipes(
        current_recipe, 
        recipe_db.recipes, 
        top_k=3
    )
    
    if not similar:
        response = f"No similar recipes found. Would you like to create a variation of '{current_recipe.title}'?"
        print(f"\nAI: {response}")
        user_input = input("User: ")
        user_msg = HumanMessage(content=user_input)
        ai_msg = AIMessage(content=response)
        
        next_action = "new_recipe" if "yes" in user_input.lower() else "continue"
        return {
            "messages": [ai_msg, user_msg],
            "current_recipe": current_recipe,
            "next_action": next_action,
            "recipe_db": recipe_db
        }
    
    response_parts = [f"Found {len(similar)} similar recipes:\n"]
    for i, (recipe_id, recipe, score) in enumerate(similar, 1):
        response_parts.append(f"\n{i}. **{recipe.title}**")
        if recipe.description:
            response_parts.append(f"   {recipe.description}")
        response_parts.append(f"   Similarity: {score:.0%}")
    
    response_parts.append("\n\nWhich recipe would you like to see? (Enter number)")
    response = "\n".join(response_parts)
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    user_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=response)
    
    try:
        selection = int(user_input.strip()) - 1
        if 0 <= selection < len(similar):
            _, selected_recipe, _ = similar[selection]
            
            recipe_display = selected_recipe.to_display_string()
            print(f"\nAI: Here's the recipe:\n\n{recipe_display}")
            
            return {
                "messages": [ai_msg, user_msg],
                "current_recipe": selected_recipe,
                "next_action": "continue",
                "recipe_db": recipe_db
            }
    except (ValueError, IndexError):
        pass
    
    return {
        "messages": [ai_msg, user_msg],
        "current_recipe": current_recipe,
        "next_action": "continue",
        "recipe_db": recipe_db
    }

# ========================================================================
# 7. CREATE NEW RECIPE NODE
# ========================================================================
def create_new_recipe_node(state: RecipeState):
    """Create a new recipe with structured format"""
    recipe_db = state.get("recipe_db", recipe_database)
    
    # Get ingredients
    print(f"\nAI: {INGREDIENTS_PROMPT}")
    ingredients = input("User: ")
    
    # Get dietary needs
    print(f"\nAI: {DIETARY_NEEDS_PROMPT}")
    dietary_needs = input("User: ")
    
    # Generate recipe
    print("\nAI: Creating your personalized recipe...")
    llm_response = generate_recipe_with_llm(ingredients, dietary_needs)
    
    # Convert to structured format
    recipe = create_recipe_from_llm_response(llm_response, ingredients, dietary_needs)
    
    # Save to database
    recipe_id = recipe_db.add_recipe(recipe)
    print(f"DEBUG: Saved recipe '{recipe.title}' with ID: {recipe_id}")
    
    # Display the recipe
    recipe_display = recipe.to_display_string()
    response = f"Here's your new recipe:\n\n{recipe_display}\n\nWould you like nutritional information? (yes/no)"
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    
    # Create messages
    ingredient_msg = HumanMessage(content=ingredients)
    dietary_msg = HumanMessage(content=dietary_needs)
    ai_msg = AIMessage(content=response)
    user_response_msg = HumanMessage(content=user_input)
    
    next_action = "nutrition" if "yes" in user_input.lower() else "continue"
    
    return {
        "messages": [ingredient_msg, dietary_msg, ai_msg, user_response_msg],
        "current_recipe": recipe,
        "recipe_db": recipe_db,
        "next_action": next_action
    }

# ========================================================================
# 8. PROVIDE NUTRITION NODE
# ========================================================================
def provide_nutrition_node(state: RecipeState):
    """Provide nutritional information for the current recipe"""
    recipe_db = state.get("recipe_db", recipe_database)
    current_recipe = state.get("current_recipe")
    
    if not current_recipe:
        response = "No recipe selected. Please select or create a recipe first."
        print(f"\nAI: {response}")
        ai_msg = AIMessage(content=response)
        return {
            "messages": [ai_msg],
            "next_action": "continue",
            "recipe_db": recipe_db
        }
    
    # Check if nutrition already exists
    if current_recipe.nutrition and current_recipe.nutrition.calories:
        nutrition_display = current_recipe.to_nutrition_string()
    else:
        # Generate nutrition info
        print("\nAI: Analyzing nutritional content...")
        
        # Use the raw text if available, otherwise use display string
        recipe_text = current_recipe.raw_text or current_recipe.to_display_string()
        nutrition_text = generate_nutrition_info(recipe_text)
        
        # Parse and add to recipe
        current_recipe = add_nutrition_to_recipe(current_recipe, nutrition_text)
        
        # Update in database
        if current_recipe.id in recipe_db.recipes:
            recipe_db.recipes[current_recipe.id] = current_recipe
            recipe_db.save_recipes()
        
        nutrition_display = current_recipe.to_nutrition_string()
    
    response = f"{nutrition_display}\n\n{ANOTHER_RECIPE_PROMPT}"
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    ai_msg = AIMessage(content=response)
    user_msg = HumanMessage(content=user_input)
    
    return {
        "messages": [ai_msg, user_msg],
        "current_recipe": current_recipe,
        "recipe_db": recipe_db,
        "next_action": "continue"
    }

# ========================================================================
# 9. END CONVERSATION NODE
# ========================================================================
def end_conversation_node(state: RecipeState):
    """End the conversation gracefully"""
    recipe_db = state.get("recipe_db", recipe_database)
    total_recipes = len(recipe_db.recipes)
    
    response = f"Thanks for using the Recipe Assistant! You now have {total_recipes} recipes in your collection. Enjoy cooking!"
    print(f"\nAI: {response}")
    
    ai_msg = AIMessage(content=response)
    return {
        "messages": [ai_msg],
        "next_action": "end"
    }

# ========================================================================
# 10. ROUTING FUNCTION
# ========================================================================
def route_next_action(state: RecipeState) -> Literal["search_recipes", "show_recent", "show_similar", "new_recipe", "nutrition", "continue", "end"]:
    """Route to the next action based on state"""
    next_action = state.get("next_action", "continue")
    print(f"DEBUG: Routing to '{next_action}'")
    
    # Check for exit keywords
    if state.get("messages"):
        last_message = state["messages"][-1].content.lower()
        if any(word in last_message for word in ["exit", "quit", "bye", "goodbye", "end"]):
            return "end"
    
    return next_action

# ========================================================================
# 11. BUILD THE GRAPH
# ========================================================================
workflow = StateGraph(RecipeState)

# Add all nodes
workflow.add_node("conversation", conversation_node)
workflow.add_node("search_recipes", search_recipes_node)
workflow.add_node("show_recent", show_recent_recipes_node)
workflow.add_node("show_similar", show_similar_recipes_node)
workflow.add_node("new_recipe", create_new_recipe_node)
workflow.add_node("nutrition", provide_nutrition_node)
workflow.add_node("end", end_conversation_node)

# Set entry point
workflow.set_entry_point("conversation")

# Add routing from conversation node
workflow.add_conditional_edges(
    "conversation",
    route_next_action,
    {
        "search_recipes": "search_recipes",
        "show_recent": "show_recent",
        "show_similar": "show_similar",
        "new_recipe": "new_recipe",
        "nutrition": "nutrition",
        "continue": "conversation",
        "end": "end"
    }
)

# All action nodes route back to conversation or end
for node in ["search_recipes", "show_recent", "show_similar", "new_recipe", "nutrition"]:
    workflow.add_conditional_edges(
        node,
        route_next_action,
        {
            "search_recipes": "search_recipes",
            "show_recent": "show_recent",
            "show_similar": "show_similar",
            "new_recipe": "new_recipe",
            "nutrition": "nutrition",
            "continue": "conversation",
            "end": "end"
        }
    )

# Compile with checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ========================================================================
# 12. MAIN LOOP
# ========================================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "recipe_session"}}
    
    # Initialize state with database
    current_state = {
        "messages": [],
        "current_recipe": None,
        "recipe_db": recipe_database,
        "next_action": "continue"
    }
    
    print("🍳 Welcome to the AI Recipe Assistant!")
    print("=" * 50)
    print(f"📚 Loaded {len(recipe_database.recipes)} recipes from your collection")
    print("=" * 50)
    
    try:
        while True:
            # Invoke the workflow
            result = app.invoke(current_state, config=config)
            
            if result is None or result.get("next_action") == "end":
                break
            
            # Update state for next iteration
            current_state = {
                "messages": result.get("messages", []),
                "current_recipe": result.get("current_recipe"),
                "recipe_db": result.get("recipe_db", recipe_database),
                "next_action": result.get("next_action", "continue"),
                "last_search_results": result.get("last_search_results")
            }
            
            print(f"DEBUG: Current recipe: {current_state['current_recipe'].title if current_state.get('current_recipe') else 'None'}")
            
    except KeyboardInterrupt:
        print("\n\n🍽️ Goodbye! Happy cooking!")
        print(f"📚 Your {len(recipe_database.recipes)} recipes have been saved.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Your recipes have been saved. Please restart the application.")