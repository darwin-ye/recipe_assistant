# main.py (updated)
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from llm import generate_recipe_with_llm
from prompts import (
    INGREDIENTS_PROMPT,
    DIETARY_NEEDS_PROMPT,
    ANOTHER_RECIPE_PROMPT,
    NUTRITION_PROMPT_TEMPLATE
)

# Define the state of our graph
class AgentState(TypedDict):
    ingredients: str
    dietary_needs: str
    recipe: str
    user_message: str

# Define the nodes (functions) of the graph

def gather_ingredients_node(state: AgentState):
    print("AI:", INGREDIENTS_PROMPT)
    user_input = input("User: ")
    return {"ingredients": user_input}

def check_dietary_needs_node(state: AgentState):
    print("AI:", DIETARY_NEEDS_PROMPT)
    user_input = input("User: ")
    return {"dietary_needs": user_input}

def generate_recipe_node(state: AgentState):
    ingredients = state.get("ingredients", "")
    dietary_needs = state.get("dietary_needs", "")
    recipe = generate_recipe_with_llm(ingredients, dietary_needs)
    
    print("\nAI: Here is a recipe for you:\n")
    print(recipe)
    print("\n" + "Would you like nutritional information for this recipe? (yes/no)")
    
    # Store the recipe in the state
    return {"recipe": recipe, "user_message": input("User: ")}

def get_nutrition_node(state: AgentState):
    """
    Calls the LLM to provide nutritional information.
    """
    recipe_text = state.get("recipe", "")
    
    # This is the new logic! We use a dedicated prompt for the LLM.
    nutritional_info_prompt = NUTRITION_PROMPT_TEMPLATE.format(recipe=recipe_text)
    
    # We call the LLM again, but this time for a different purpose.
    nutritional_info = generate_recipe_with_llm(nutritional_info_prompt, "")
    
    print("\nAI:", nutritional_info)
    print("\n" + ANOTHER_RECIPE_PROMPT)
    
    return {"user_message": input("User: ")}

def end_conversation_node(state: AgentState):
    print("AI: You're welcome! Enjoy your meal!")
    return state

# Define the conditional logic for transitions
def route_after_recipe(state: AgentState) -> Literal["nutrition_path", "end_path", "restart_path"]:
    user_input = state.get("user_message", "").strip().lower()

    if "yes" in user_input:
        return "nutrition_path"
    else:
        return "end_path"

def route_after_nutrition(state: AgentState) -> Literal["end_path", "restart_path"]:
    user_input = state.get("user_message", "").strip().lower()

    if "yes" in user_input:
        return "restart_path"
    else:
        return "end_path"

# Build the LangGraph
workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("gather_ingredients", gather_ingredients_node)
workflow.add_node("check_dietary_needs", check_dietary_needs_node)
workflow.add_node("generate_recipe", generate_recipe_node)
workflow.add_node("get_nutrition", get_nutrition_node)
workflow.add_node("end_conversation", end_conversation_node)

# Set the entry point
workflow.set_entry_point("gather_ingredients")

# Add edges to connect the nodes
workflow.add_edge("gather_ingredients", "check_dietary_needs")
workflow.add_edge("check_dietary_needs", "generate_recipe")

# Conditional edges after the recipe is generated
workflow.add_conditional_edges(
    "generate_recipe",
    route_after_recipe,
    {
        "nutrition_path": "get_nutrition",
        "end_path": END,
    }
)

# Conditional edges after nutritional info is provided
workflow.add_conditional_edges(
    "get_nutrition",
    route_after_nutrition,
    {
        "restart_path": "gather_ingredients",
        "end_path": END,
    }
)

# Compile the graph
app = workflow.compile()

# Run the agent in a loop
if __name__ == "__main__":
    current_state = {"ingredients": "", "dietary_needs": "", "user_message": "", "recipe": ""}

    while True:
        try:
            current_state = app.invoke(current_state)
            if current_state is None:
                break
        except Exception as e:
            print(f"An error occurred: {e}. Restarting the conversation.")
            current_state = {"ingredients": "", "dietary_needs": "", "user_message": "", "recipe": ""}