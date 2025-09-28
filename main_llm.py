# main_llm.py - Pure LLM Recipe Assistant Interface
"""
A pure LLM-driven recipe assistant that uses natural language understanding
for all user interactions. This provides the most flexible and intuitive
user experience by eliminating rule-based patterns and leveraging
the full power of LLM reasoning.
"""

from typing import TypedDict, List, Optional, Any, Dict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated

from llm import llm
from recipe_models import RecipeDatabase
from llm_agent import LLMRecipeAgent


# ========================================================================
# 1. SIMPLIFIED STATE FOR LLM APPROACH
# ========================================================================
class LLMConversationState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_recipe: Optional[Any]  # Current recipe context
    conversation_context: Dict[str, Any]  # General conversation context
    session_stats: Dict[str, int]  # Session statistics


# Initialize components
recipe_database = RecipeDatabase()
llm_agent = LLMRecipeAgent(llm)


# ========================================================================
# 2. PURE LLM CONVERSATION NODE
# ========================================================================
def llm_conversation_node(state: LLMConversationState):
    """Handle all user interactions through pure LLM reasoning"""

    # Initialize session stats if not present
    if "session_stats" not in state:
        state["session_stats"] = {
            "interactions": 0,
            "recipes_created": 0,
            "searches_performed": 0,
            "analytics_queries": 0
        }

    # Get user input
    if not state.get("messages"):
        # First interaction - show welcome
        print("🧠 Welcome to your LLM-Powered Recipe Assistant!")
        print("=" * 60)
        print(f"📊 Database: {len(recipe_database.recipes)} recipes available")
        print("🤖 Pure AI Understanding - Talk naturally!")
        print()
        print("💬 Examples:")
        print("   • 'Create a recipe with chicken and tomatoes'")
        print("   • 'What's my most frequent recipe?'")
        print("   • 'Show me recent recipes'")
        print("   • 'Find pasta dishes'")
        print("   • 'Scale this to 8 people'")
        print()
        print("🎯 Just talk naturally - I understand context and nuance!")
        print("=" * 60)
    else:
        # Check for exit keywords
        last_message = state["messages"][-1].content.lower()
        if any(word in last_message for word in ["exit", "quit", "bye", "goodbye", "stop"]):
            print("\n🍳 Thanks for cooking with me! Happy cooking!")

            # Show session summary
            stats = state["session_stats"]
            print(f"\n📊 Session Summary:")
            print(f"   • Interactions: {stats['interactions']}")
            print(f"   • Recipes created: {stats['recipes_created']}")
            print(f"   • Searches performed: {stats['searches_performed']}")
            print(f"   • Analytics queries: {stats['analytics_queries']}")

            return {
                "messages": [AIMessage(content="Goodbye! Happy cooking! 🍳")],
                "conversation_context": {"ended": True},
                "session_stats": stats
            }

    # Get user input
    user_input = input("\n🗣️  You: ").strip()

    if not user_input:
        return {
            "messages": [AIMessage(content="I'm here when you're ready to talk about recipes!")],
            "conversation_context": state.get("conversation_context", {}),
            "session_stats": state.get("session_stats", {})
        }

    print("\n🧠 Processing with pure LLM understanding...")
    print("-" * 50)

    try:
        # Process through LLM agent
        current_recipe = state.get("current_recipe")
        response = llm_agent.process_input(user_input, current_recipe)

        # Update session statistics
        stats = state.get("session_stats", {})
        stats["interactions"] = stats.get("interactions", 0) + 1

        # Update specific stats based on action taken
        action = response.action_taken
        if action == "create_recipe":
            stats["recipes_created"] = stats.get("recipes_created", 0) + 1
        elif action in ["search_recipes", "get_recent"]:
            stats["searches_performed"] = stats.get("searches_performed", 0) + 1
        elif action in ["analytics_frequent", "analytics_count"]:
            stats["analytics_queries"] = stats.get("analytics_queries", 0) + 1

        # Show response
        print(f"\n🤖 Assistant: {response.content}")

        # Show debug info if enabled
        if response.updated_context.get("error"):
            print(f"\n⚠️  Debug: {response.updated_context['error']}")

        # Create messages
        user_msg = HumanMessage(content=user_input)
        ai_msg = AIMessage(content=response.content)

        # Update context
        updated_context = state.get("conversation_context", {})
        updated_context.update(response.updated_context)

        # Update current recipe if changed
        new_current_recipe = response.updated_context.get("current_recipe", current_recipe)

        return {
            "messages": [user_msg, ai_msg],
            "current_recipe": new_current_recipe,
            "conversation_context": updated_context,
            "session_stats": stats
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        error_response = """I encountered an unexpected error. Let me try to help you differently.

🍳 **What I can do:**
- Create recipes from ingredients
- Search your recipe collection
- Show analytics and patterns
- Scale recipes for different servings

What would you like to try?"""

        return {
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content=error_response)
            ],
            "conversation_context": {"error": str(e)},
            "session_stats": state.get("session_stats", {})
        }


# ========================================================================
# 3. WORKFLOW SETUP
# ========================================================================
def create_llm_workflow():
    """Create the LLM-powered conversation workflow"""

    # Create workflow
    workflow = StateGraph(LLMConversationState)

    # Add the single conversation node
    workflow.add_node("conversation", llm_conversation_node)

    # Set entry point
    workflow.set_entry_point("conversation")

    # Add self-loop for continuous conversation
    workflow.add_edge("conversation", "conversation")

    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    return app


# ========================================================================
# 4. MAIN EXECUTION
# ========================================================================
def main():
    """Main execution function"""

    print("🚀 Starting LLM Recipe Assistant...")
    print("   Pure AI understanding with natural conversation")
    print()

    # Create workflow
    app = create_llm_workflow()

    # Configuration for persistence
    config = {"configurable": {"thread_id": "llm_recipe_session"}}

    # Initial state
    initial_state = {
        "messages": [],
        "current_recipe": None,
        "conversation_context": {},
        "session_stats": {
            "interactions": 0,
            "recipes_created": 0,
            "searches_performed": 0,
            "analytics_queries": 0
        }
    }

    # Run the conversation loop
    try:
        state = initial_state
        while True:
            # Process one interaction
            result = app.invoke(state, config)

            # Update state for next iteration
            state = result

            # Check if conversation ended
            if result.get("conversation_context", {}).get("ended"):
                break

    except KeyboardInterrupt:
        print("\n\n👋 Conversation interrupted. Thanks for using the LLM Recipe Assistant!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("The conversation has ended unexpectedly.")


# ========================================================================
# 5. UTILITY FUNCTIONS
# ========================================================================
def test_llm_components():
    """Test the LLM components independently"""

    print("🧪 Testing LLM Components")
    print("=" * 40)

    # Test intent classifier
    from llm_intent_classifier import LLMIntentClassifier
    classifier = LLMIntentClassifier(llm)

    test_queries = [
        "create a chicken pasta recipe",
        "show me recent recipes",
        "what's my go-to recipe?",
        "2",
        "help"
    ]

    print("\n1. Testing Intent Classification:")
    for query in test_queries:
        result = classifier.classify_intent(query)
        print(f"   '{query}' → {result.intent} ({result.confidence:.2f})")

    # Test agent
    print("\n2. Testing LLM Agent:")
    agent = LLMRecipeAgent(llm)

    sample_query = "help"
    response = agent.process_input(sample_query)
    print(f"   Query: '{sample_query}'")
    print(f"   Response: {response.content[:100]}...")
    print(f"   Success: {response.success}")

    print("\n✅ Component testing complete!")


def show_comparison():
    """Show comparison between different approaches"""

    print("📊 Recipe Assistant Approaches Comparison")
    print("=" * 50)
    print()
    print("🤖 **ReAct Mode** (python main.py)")
    print("   • Hybrid rule-based + LLM intent detection")
    print("   • 40+ regex patterns for common queries")
    print("   • LLM fallback for edge cases")
    print("   • Complex but battle-tested")
    print()
    print("📋 **Menu Mode** (python main_menu_driven.py)")
    print("   • Traditional guided interface")
    print("   • Step-by-step numbered menus")
    print("   • Predictable and structured")
    print("   • Great for new users")
    print()
    print("🧠 **LLM Mode** (python main_llm.py)")
    print("   • Pure LLM natural language understanding")
    print("   • No regex patterns or rules")
    print("   • Maximum flexibility and context awareness")
    print("   • Most natural conversations")
    print()
    print("Choose the mode that fits your preference!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_llm_components()
        elif sys.argv[1] == "compare":
            show_comparison()
        else:
            print("Usage: python main_llm.py [test|compare]")
    else:
        main()