#!/usr/bin/env python3
"""
Simplest possible test to identify the timeout issue
"""

from llm import llm
from multi_action_react_agent import MultiActionReActAgent

def test_one_simple_case():
    """Test a single simple case without create_recipe"""

    print("🔧 SIMPLE TEST - No Create Recipe")
    print("=" * 40)

    agent = MultiActionReActAgent(llm)

    # Test a non-create recipe action
    query = "Search for chicken recipes and count how many there are"

    print(f"Testing: {query}")
    print("-" * 30)

    try:
        result = agent.process_request(query)
        print(f"✅ Result length: {len(result)}")
        print(f"✅ Contains success: {'🎉 All requested actions have been completed successfully!' in result}")
        print(f"✅ First 200 chars: {result[:200]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_one_simple_case()