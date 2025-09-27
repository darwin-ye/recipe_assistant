# main.py - Complete Recipe Assistant with ReAct Integration
from typing import TypedDict, Literal, Annotated, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import re

from llm import generate_recipe_with_llm, generate_nutrition_info, llm
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

# Import ReAct agent
from react_agent import ReActAgent

# ========================================================================
# 1. UPDATED STATE - Now includes ReAct capability
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
    # ReAct agent for complex queries
    react_agent: Optional[Any]

# Initialize the database globally
recipe_database = RecipeDatabase()

# Initialize ReAct agent
react_agent = ReActAgent(llm)

# ========================================================================
# 2. ENHANCED INTENT CLASSIFICATION (includes ReAct detection)
# ========================================================================
def classify_intent_with_llm(user_input: str, state: RecipeState) -> str:
    """Enhanced intent classification that includes ReAct detection"""
    
    recipe_db = state.get("recipe_db", recipe_database)
    has_recipes = len(recipe_db.recipes) > 0
    has_current_recipe = state.get("current_recipe") is not None
    
    # Check for ReAct-appropriate queries first
    react_keywords = [
        "substitute", "replace", "instead of", "alternative",
        "scale", "adjust serving", "make for", "people",
        "how long", "cooking time", "estimate time",
        "online recipe", "search web", "find recipe online",
        "what if", "can i", "is it possible"
    ]
    
    if any(keyword in user_input.lower() for keyword in react_keywords):
        return "react_query"
    
    # Simplified prompt for better LLM response
    context = f"""
User said: "{user_input}"

What does the user want to do? Choose ONE:
- search_recipes (find existing recipes in database)  
- show_recent (see recent/previous recipes)
- nutrition (get nutritional info)
- new_recipe (create new recipe)
- show_similar (find similar recipes)
- react_query (complex question needing reasoning)

Reply with ONLY the option name, nothing else.
"""
    
    try:
        print(f"DEBUG: Classifying intent for: '{user_input}'")
        
        response = llm.invoke(context)
        raw_intent = response.content.strip()
        
        # Clean up the LLM response
        intent = raw_intent.lower()
        intent = re.sub(r'["\'\d.\-\*]', '', intent).strip()
        
        # Extract intent name if it's in a sentence
        valid_intents = ["search_recipes", "show_recent", "nutrition", "new_recipe", "show_similar", "react_query"]
        
        for valid in valid_intents:
            if valid in intent:
                intent = valid
                break
        
        if intent not in valid_intents:
            print(f"DEBUG: Extracted intent '{intent}' not valid, using fallback")
            intent = classify_intent_fallback(user_input, state)
        
        print(f"DEBUG: Classified as: '{intent}'")
        return intent
        
    except Exception as e:
        print(f"DEBUG: LLM classification failed: {e}")
        return classify_intent_fallback(user_input, state)

def classify_intent_fallback(user_input: str, state: RecipeState) -> str:
    """Enhanced fallback classification including ReAct"""
    user_input = user_input.lower().strip()
    
    # ReAct keywords - check first
    react_keywords = [
        "substitute", "replace", "instead", "alternative",
        "scale", "serving", "people", "portion",
        "how long", "time", "duration",
        "online", "web search"
    ]
    if any(word in user_input for word in react_keywords):
        return "react_query"
    
    # Search keywords
    search_keywords = ["find", "search", "show me", "recipes with", "dishes with", "what recipes"]
    if any(word in user_input for word in search_keywords):
        return "search_recipes"
    
    # Recent/history/previous keywords
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
    
    # New recipe
    new_keywords = ["new recipe", "create", "make", "cook", "generate"]
    if any(word in user_input for word in new_keywords):
        return "new_recipe"
    
    return "new_recipe"

# ========================================================================
# 3. MAIN CONVERSATION NODE (Updated with ReAct support)
# ========================================================================
def conversation_node(state: RecipeState):
    """Main conversation handler with ReAct support"""

    # Ensure recipe_db is in state
    if "recipe_db" not in state:
        state["recipe_db"] = recipe_database

    recipe_count = len(state["recipe_db"].recipes)
    current_recipe = state.get("current_recipe")

    # If there's a current recipe, handle simple questions about it
    if current_recipe:
        # Check if this is a simple question about the current recipe
        if state.get("messages"):
            last_message = state["messages"][-1].content.lower()

            # Handle simple serving questions
            if any(phrase in last_message for phrase in ["how many people", "serves", "serving size", "servings"]):
                serving_response = f"The '{current_recipe.title}' recipe serves {current_recipe.servings} people."
                print(f"\nAI: {serving_response}")

                # Ask for follow-up while preserving context
                user_input = input("What else would you like to know about this recipe? ")
                user_msg = HumanMessage(content=user_input)

                # Re-classify the new input
                intent = classify_intent_with_llm(user_input, state)

                return {
                    "messages": [AIMessage(content=serving_response), user_msg],
                    "current_recipe": current_recipe,
                    "next_action": intent,
                    "recipe_db": state["recipe_db"],
                    "react_agent": react_agent
                }

    print("\n" + "="*50)
    if current_recipe:
        print(f"AI: Current recipe: '{current_recipe.title}' (serves {current_recipe.servings})")
        print("What would you like to do?")
        print("- Get nutritional info for this recipe")
        print("- Find similar recipes")
        print("- Scale this recipe for different servings")
        print("- Search for other recipes")
        print("- Create a new recipe")
        print("- See recent recipes")
    else:
        print(f"AI: Hello! I have {recipe_count} recipes in my database.")
        print("What would you like to do?")
        print("- Search for recipes (e.g., 'find chicken recipes')")
        print("- Create a new recipe")
        print("- See recent recipes")
        print("- Ask complex questions (substitutions, scaling, timing)")

    user_input = input("\nUser: ")

    # Classify intent
    intent = classify_intent_with_llm(user_input, state)

    # Add user message
    user_msg = HumanMessage(content=user_input)

    return {
        "messages": [user_msg],
        "current_recipe": current_recipe,  # Preserve current recipe
        "next_action": intent,
        "recipe_db": state["recipe_db"],
        "react_agent": react_agent
    }

# ========================================================================
# 4. REACT QUERY NODE - Handle complex queries with reasoning
# ========================================================================
def react_query_node(state: RecipeState):
    """Handle complex queries using ReAct reasoning"""
    last_message = state["messages"][-1].content if state["messages"] else ""
    current_recipe = state.get("current_recipe")

    print("\n🤔 Let me think about this step by step...")
    print("-" * 50)

    # Handle common scaling requests directly
    if ("scale" in last_message.lower() or "serve" in last_message.lower() or "people" in last_message.lower()) and current_recipe:
        # Extract number from the message
        import re
        numbers = re.findall(r'\d+', last_message)
        if numbers:
            desired_servings = int(numbers[0])

            # Use the ReAct agent's scaling tool directly
            react_agent.current_recipe = current_recipe
            scaled_result = react_agent.execute_tool('scale_current_recipe', {'desired_servings': desired_servings})

            formatted_response = f"Here's your scaled recipe:\n\n{scaled_result}"
        else:
            formatted_response = f"I can help you scale '{current_recipe.title}' which currently serves {current_recipe.servings} people. How many people do you want to serve?"
    else:
        # Use the ReAct agent for other complex queries
        try:
            # Get chat history for context
            chat_history = []
            if current_recipe:
                chat_history.append(f"Current recipe: {current_recipe.title}")
                chat_history.append(f"Servings: {current_recipe.servings}")

            # Run ReAct agent with current recipe context
            response = react_agent.run(last_message, chat_history, current_recipe)

            # Format the response
            formatted_response = f"Based on my analysis:\n\n{response}"

            # Check if user wants to proceed with a recipe action
            if "would you like" in response.lower() or "shall i" in response.lower():
                formatted_response += "\n\nWhat would you like to do next?"

        except Exception as e:
            print(f"DEBUG: ReAct error: {e}")
            formatted_response = "I need to think about that more carefully. Could you rephrase your question?"
    
    print(f"\nAI: {formatted_response}")
    user_input = input("User: ")
    
    ai_msg = AIMessage(content=formatted_response)
    user_msg = HumanMessage(content=user_input)
    
    # Determine next action based on user response
    next_action = "continue"
    if "yes" in user_input.lower() and "recipe" in formatted_response.lower():
        next_action = "new_recipe"
    
    return {
        "messages": [ai_msg, user_msg],
        "next_action": next_action,
        "react_agent": react_agent,
        "recipe_db": state.get("recipe_db", recipe_database),
        "current_recipe": state.get("current_recipe")
    }

# ========================================================================
# 5. SEARCH RECIPES NODE
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
    
    response_parts = [f"Found {len(results)} matching recipes:\n"]
    for i, (recipe_id, recipe, score) in enumerate(results[:3]):
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
    
    current_recipe = None
    if user_input.strip().lower() == 'new':
        next_action = "new_recipe"
    else:
        try:
            selection = int(user_input.strip()) - 1
            if 0 <= selection < len(results):
                _, selected_recipe, _ = results[selection]
                current_recipe = selected_recipe
                
                recipe_display = selected_recipe.to_display_string()
                print(f"\nAI: Here's your recipe:\n\n{recipe_display}")
                print("\nWould you like nutritional info, similar recipes, or to scale this recipe?")
                
                followup_input = input("User: ")
                followup_msg = HumanMessage(content=followup_input)

                # Handle simple questions about the current recipe
                if any(phrase in followup_input.lower() for phrase in ["how many people", "serves", "serving size", "servings"]):
                    # Answer the serving question directly
                    serving_response = f"This recipe serves {selected_recipe.servings} people."
                    print(f"\nAI: {serving_response}")

                    # Ask for follow-up
                    final_input = input("What else would you like to know? ")
                    final_msg = HumanMessage(content=final_input)

                    return {
                        "messages": [ai_msg, user_msg, followup_msg, AIMessage(content=serving_response), final_msg],
                        "current_recipe": selected_recipe,
                        "last_search_results": results,
                        "next_action": "continue",
                        "recipe_db": recipe_db
                    }
                elif "nutrition" in followup_input.lower():
                    next_action = "nutrition"
                elif "similar" in followup_input.lower():
                    next_action = "show_similar"
                elif "scale" in followup_input.lower():
                    next_action = "react_query"
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
# 6. SHOW RECENT RECIPES NODE
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
# 7. SHOW SIMILAR RECIPES NODE
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
# 8. CREATE NEW RECIPE NODE
# ========================================================================
def create_new_recipe_node(state: RecipeState):
    """Create a new recipe with structured format"""
    recipe_db = state.get("recipe_db", recipe_database)
    
    print(f"\nAI: {INGREDIENTS_PROMPT}")
    ingredients = input("User: ")
    
    print(f"\nAI: {DIETARY_NEEDS_PROMPT}")
    dietary_needs = input("User: ")
    
    print("\nAI: Creating your personalized recipe...")
    llm_response = generate_recipe_with_llm(ingredients, dietary_needs)
    
    recipe = create_recipe_from_llm_response(llm_response, ingredients, dietary_needs)
    
    recipe_id = recipe_db.add_recipe(recipe)
    print(f"DEBUG: Saved recipe '{recipe.title}' with ID: {recipe_id}")
    
    recipe_display = recipe.to_display_string()
    response = f"Here's your new recipe:\n\n{recipe_display}\n\nWould you like nutritional information? (yes/no)"
    print(f"\nAI: {response}")
    
    user_input = input("User: ")
    
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
# 9. PROVIDE NUTRITION NODE
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
    
    if current_recipe.nutrition and current_recipe.nutrition.calories:
        nutrition_display = current_recipe.to_nutrition_string()
    else:
        print("\nAI: Analyzing nutritional content...")
        
        recipe_text = current_recipe.raw_text or current_recipe.to_display_string()
        nutrition_text = generate_nutrition_info(recipe_text)
        
        current_recipe = add_nutrition_to_recipe(current_recipe, nutrition_text)
        
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
# 10. END CONVERSATION NODE
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
# 11. ROUTING FUNCTION (Updated with react_query)
# ========================================================================
def route_next_action(state: RecipeState) -> Literal["search_recipes", "show_recent", "show_similar", "new_recipe", "nutrition", "react_query", "continue", "end"]:
    """Route to the next action based on state"""
    next_action = state.get("next_action", "continue")
    print(f"DEBUG: Routing to '{next_action}'")
    
    if state.get("messages"):
        last_message = state["messages"][-1].content.lower()
        if any(word in last_message for word in ["exit", "quit", "bye", "goodbye", "end"]):
            return "end"
    
    return next_action

# ========================================================================
# 12. BUILD THE GRAPH (Updated with ReAct node)
# ========================================================================
workflow = StateGraph(RecipeState)

# Add all nodes (including new react_query node)
workflow.add_node("conversation", conversation_node)
workflow.add_node("search_recipes", search_recipes_node)
workflow.add_node("show_recent", show_recent_recipes_node)
workflow.add_node("show_similar", show_similar_recipes_node)
workflow.add_node("new_recipe", create_new_recipe_node)
workflow.add_node("nutrition", provide_nutrition_node)
workflow.add_node("react_query", react_query_node)  # New ReAct node
workflow.add_node("end", end_conversation_node)

# Set entry point
workflow.set_entry_point("conversation")

# Add routing from conversation node (updated with react_query)
workflow.add_conditional_edges(
    "conversation",
    route_next_action,
    {
        "search_recipes": "search_recipes",
        "show_recent": "show_recent",
        "show_similar": "show_similar",
        "new_recipe": "new_recipe",
        "nutrition": "nutrition",
        "react_query": "react_query",  # New routing option
        "continue": "conversation",
        "end": "end"
    }
)

# All action nodes route back to conversation or end (including react_query)
for node in ["search_recipes", "show_recent", "show_similar", "new_recipe", "nutrition", "react_query"]:
    workflow.add_conditional_edges(
        node,
        route_next_action,
        {
            "search_recipes": "search_recipes",
            "show_recent": "show_recent",
            "show_similar": "show_similar",
            "new_recipe": "new_recipe",
            "nutrition": "nutrition",
            "react_query": "react_query",
            "continue": "conversation",
            "end": "end"
        }
    )

# Compile with checkpointer
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ========================================================================
# 13. MAIN LOOP
# ========================================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "recipe_session"}}
    
    # Initialize state with database and react agent
    current_state = {
        "messages": [],
        "current_recipe": None,
        "recipe_db": recipe_database,
        "next_action": "continue",
        "react_agent": react_agent
    }
    
    print("🍳 Welcome to the AI Recipe Assistant with ReAct!")
    print("=" * 50)
    print(f"📚 Loaded {len(recipe_database.recipes)} recipes from your collection")
    print("🤖 ReAct reasoning enabled for complex queries")
    print("=" * 50)
    
    try:
        while True:
            result = app.invoke(current_state, config=config)
            
            if result is None or result.get("next_action") == "end":
                break
            
            current_state = {
                "messages": result.get("messages", []),
                "current_recipe": result.get("current_recipe"),
                "recipe_db": result.get("recipe_db", recipe_database),
                "next_action": result.get("next_action", "continue"),
                "last_search_results": result.get("last_search_results"),
                "react_agent": result.get("react_agent", react_agent)
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