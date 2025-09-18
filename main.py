# main.py
from typing import TypedDict, Literal, Annotated, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages

from llm import generate_recipe_with_llm
from prompts import (
    INGREDIENTS_PROMPT,
    DIETARY_NEEDS_PROMPT,
    ANOTHER_RECIPE_PROMPT,
    NUTRITION_PROMPT_TEMPLATE,
)

# ========================================================================
# 1. Define the state with explicit message tracking
# ========================================================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    ingredients: Optional[str]
    dietary_needs: Optional[str]
    recipe: Optional[str]
    user_message: Optional[str]

# ========================================================================
# 2. Define nodes
# ========================================================================
def gather_ingredients_node(state: AgentState):
    ai_msg = AIMessage(content=INGREDIENTS_PROMPT)
    print("AI:", ai_msg.content)
    user_input = input("User: ")

    return {
        "ingredients": user_input,
        "messages": [ai_msg, HumanMessage(content=user_input)],
    }


def check_dietary_needs_node(state: AgentState):
    ai_msg = AIMessage(content=DIETARY_NEEDS_PROMPT)
    print("AI:", ai_msg.content)
    user_input = input("User: ")

    return {
        "dietary_needs": user_input,
        "messages": [ai_msg, HumanMessage(content=user_input)],
    }


def generate_recipe_node(state: AgentState):
    ingredients = state.get("ingredients", "")
    dietary_needs = state.get("dietary_needs", "")
    recipe = generate_recipe_with_llm(ingredients, dietary_needs)

    ai_msg = AIMessage(
        content=f"Here is a recipe for you:\n\n{recipe}\n\nWould you like nutritional information for this recipe? (yes/no)"
    )
    print("\nAI:", ai_msg.content)
    user_input = input("User: ")

    return {
        "recipe": recipe,
        "user_message": user_input,
        "messages": [ai_msg, HumanMessage(content=user_input)],
    }


def get_nutrition_node(state: AgentState):
    recipe_text = state.get("recipe", "")
    nutritional_info_prompt = NUTRITION_PROMPT_TEMPLATE.format(recipe=recipe_text)
    nutritional_info = generate_recipe_with_llm(nutritional_info_prompt, "")

    ai_msg = AIMessage(content=f"{nutritional_info}\n\n{ANOTHER_RECIPE_PROMPT}")
    print("\nAI:", ai_msg.content)
    user_input = input("User: ")

    return {
        "user_message": user_input,
        "messages": [ai_msg, HumanMessage(content=user_input)],
    }


def end_conversation_node(state: AgentState):
    ai_msg = AIMessage(content="You're welcome! Enjoy your meal!")
    print("AI:", ai_msg.content)
    return {"messages": [ai_msg]}

# ========================================================================
# 3. Routing functions
# ========================================================================
def route_after_recipe(
    state: AgentState,
) -> Literal["nutrition_path", "end_path", "restart_path"]:
    user_input = (state.get("user_message") or "").strip().lower()
    if "yes" in user_input:
        return "nutrition_path"
    else:
        return "end_path"


def route_after_nutrition(state: AgentState) -> Literal["end_path", "restart_path"]:
    user_input = (state.get("user_message") or "").strip().lower()
    if "yes" in user_input:
        return "restart_path"
    else:
        return "end_path"

# ========================================================================
# 4. Build the LangGraph
# ========================================================================
workflow = StateGraph(AgentState)

workflow.add_node("gather_ingredients", gather_ingredients_node)
workflow.add_node("check_dietary_needs", check_dietary_needs_node)
workflow.add_node("generate_recipe", generate_recipe_node)
workflow.add_node("get_nutrition", get_nutrition_node)
workflow.add_node("end_conversation", end_conversation_node)

workflow.set_entry_point("gather_ingredients")

workflow.add_edge("gather_ingredients", "check_dietary_needs")
workflow.add_edge("check_dietary_needs", "generate_recipe")

workflow.add_conditional_edges(
    "generate_recipe",
    route_after_recipe,
    {
        "nutrition_path": "get_nutrition",
        "end_path": END,
    },
)

workflow.add_conditional_edges(
    "get_nutrition",
    route_after_nutrition,
    {
        "restart_path": "gather_ingredients",
        "end_path": END,
    },
)

app = workflow.compile()

# ========================================================================
# 5. Run the agent loop
# ========================================================================
if __name__ == "__main__":
    current_state = {
        "messages": [],
        "ingredients": "",
        "dietary_needs": "",
        "recipe": "",
        "user_message": "",
    }

    while True:
        try:
            current_state = app.invoke(current_state)
            if current_state is None:
                break
        except Exception as e:
            print(f"An error occurred: {e}. Restarting the conversation.")
            current_state = {
                "messages": [],
                "ingredients": "",
                "dietary_needs": "",
                "recipe": "",
                "user_message": "",
            }
