#!/usr/bin/env python3
"""
Debug version of focused test - test one case at a time
"""

from llm import llm
from multi_action_react_agent import MultiActionReActAgent

def test_one_case():
    """Test just one problematic case"""

    print("🔧 DEBUG FOCUSED TEST - One Case")
    print("=" * 40)

    agent = MultiActionReActAgent(llm)

    # Start with the simplest multi-action create case
    query = "Create a pasta recipe then get its details"

    print(f"Testing: {query}")
    print("-" * 30)

    try:
        result = agent.process_request(query)
        print(f"✅ Success! Result length: {len(result)}")
        print(f"✅ First 300 chars: {result[:300]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_one_case()