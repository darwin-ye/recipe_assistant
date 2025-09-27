# main_react.py - ReAct-First Recipe Assistant
"""
A completely ReAct-driven recipe assistant that handles all user inputs
through natural conversation rather than menu-driven interactions.
"""

from typing import TypedDict, List, Optional, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated

from llm import llm
from recipe_models import RecipeDatabase
from react_agent import ReActAgent

# ========================================================================
# 1. SIMPLIFIED STATE
# ========================================================================
class ConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    conversation_context: Optional[str]  # Track general context
    last_action: Optional[str]  # Track what was last done

# Initialize components
recipe_database = RecipeDatabase()
react_agent = ReActAgent(llm)

# ========================================================================
# 2. SINGLE CONVERSATION NODE
# ========================================================================
def react_conversation_node(state: ConversationState):
    """Handle all user interactions through ReAct reasoning"""

    # Get user input
    if not state.get("messages"):
        # First interaction
        print("🍳 Welcome to your AI Recipe Assistant!")
        print("=" * 50)
        print(f"📚 I have {len(recipe_database.recipes)} recipes available")
        print("💬 What can I help you with? Just ask naturally!")
        print("   Examples:")
        print("   • 'Show me chicken recipes'")
        print("   • 'Create a pasta dish with tomatoes'")
        print("   • 'What did I cook recently?'")
        print("   • 'Scale the beef recipe for 8 people'")
        print("=" * 50)
    else:
        # Check for exit keywords
        last_message = state["messages"][-1].content.lower()
        if any(word in last_message for word in ["exit", "quit", "bye", "goodbye"]):
            print("\n🍽️ Thanks for using the Recipe Assistant! Happy cooking!")
            return {
                "messages": [AIMessage(content="Goodbye! Happy cooking!")],
                "conversation_context": "ended",
                "last_action": "end"
            }

    user_input = input("\n🧑‍🍳 You: ")

    # Process through ReAct
    print("\n🤔 Let me help you with that...")
    print("-" * 40)

    try:
        # Use ReAct agent to handle the request
        response = react_agent.run(user_input, [])

        # Clean up the response (remove any FINAL ANSWER: prefix if present)
        if "FINAL ANSWER:" in response:
            response = response.split("FINAL ANSWER:")[-1].strip()

        formatted_response = f"🍴 {response}"

    except Exception as e:
        print(f"DEBUG: ReAct error: {e}")
        formatted_response = "I'm having trouble processing that. Could you rephrase your request?"

    print(f"\n{formatted_response}")

    # Create messages
    user_msg = HumanMessage(content=user_input)
    ai_msg = AIMessage(content=formatted_response)

    return {
        "messages": [user_msg, ai_msg],
        "conversation_context": "active",
        "last_action": "react_response"
    }

# ========================================================================
# 3. ROUTING FUNCTION
# ========================================================================
def should_continue(state: ConversationState) -> str:
    """Determine if conversation should continue"""
    if state.get("conversation_context") == "ended":
        return "end"
    return "continue"

# ========================================================================
# 4. BUILD SIMPLE WORKFLOW
# ========================================================================
workflow = StateGraph(ConversationState)

# Single node handles everything
workflow.add_node("conversation", react_conversation_node)

# Set entry point
workflow.set_entry_point("conversation")

# Add self-loop for continuous conversation
workflow.add_conditional_edges(
    "conversation",
    should_continue,
    {
        "continue": "conversation",
        "end": "__end__"
    }
)

# Compile workflow
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# ========================================================================
# 5. MAIN LOOP
# ========================================================================
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "react_recipe_session"}}

    # Initialize empty state
    current_state = {
        "messages": [],
        "conversation_context": None,
        "last_action": None
    }

    try:
        while True:
            # Invoke the workflow
            result = app.invoke(current_state, config=config)

            # Check if we should end
            if (result is None or
                result.get("conversation_context") == "ended" or
                result.get("last_action") == "end"):
                break

            # Update state for next iteration
            current_state = {
                "messages": result.get("messages", []),
                "conversation_context": result.get("conversation_context"),
                "last_action": result.get("last_action")
            }

    except KeyboardInterrupt:
        print("\n\n🍽️ Goodbye! Happy cooking!")
        print(f"📚 Your {len(recipe_database.recipes)} recipes have been saved.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        print("Your recipes have been saved. Please restart the application.")