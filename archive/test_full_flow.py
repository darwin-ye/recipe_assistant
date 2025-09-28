#!/usr/bin/env python3
"""Test the complete flow as the user experienced it"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_full_flow():
    agent = SimpleReActAgent(llm)

    # Test the exact input the user used
    user_input = "give me the previous salmon recipe"
    print(f"Testing full flow with input: '{user_input}'")
    print("=" * 50)

    # Run the full agent with the exact same method signature
    result = agent.run(user_input, current_recipe=None)

    # Print first few lines to see which recipe was returned
    lines = result.split('\n')
    print(f"Returned recipe title: {lines[0] if lines else 'No result'}")
    print(f"Full result length: {len(result)} characters")

if __name__ == "__main__":
    test_full_flow()