#!/usr/bin/env python3
"""
Focused test to identify specific issues without timeout
"""

from llm import llm
from multi_action_react_agent import MultiActionReActAgent

def test_single_cases():
    """Test individual problematic cases"""

    print("🎯 FOCUSED TESTING - Individual Cases")
    print("=" * 50)

    agent = MultiActionReActAgent(llm)

    # Test cases that might have issues
    test_cases = [
        "Create a pasta recipe then get its details",
        "Search for chicken recipes and count how many there are",
        "Create a beef recipe, get its details, then scale it to 8 servings"
    ]

    for i, query in enumerate(test_cases, 1):
        print(f"\nTest {i}: {query}")
        print("-" * 30)

        try:
            result = agent.process_request(query)

            steps = result.count("**Step")
            has_success = "🎉 All requested actions have been completed successfully!" in result

            print(f"✅ Steps: {steps} | Success: {has_success}")

            if not has_success:
                print("⚠️ Missing success message")
                print(f"Last 200 chars: {result[-200:]}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_single_cases()