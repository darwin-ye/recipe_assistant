#!/usr/bin/env python3
"""Test the SimpleReActAgent _get_recipe_details method directly"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_agent_method():
    agent = SimpleReActAgent(llm)

    # Test the agent's method directly
    recipe_name = "Pan-Seared Salmon Cakes with Lemon-Dill Sauce"
    print(f"Testing agent._get_recipe_details with: '{recipe_name}'")
    print("=" * 50)

    result = agent._get_recipe_details(recipe_name)

    # Print first few lines to see which recipe was returned
    lines = result.split('\n')
    print(f"Returned recipe title: {lines[0] if lines else 'No result'}")
    print(f"Full result length: {len(result)} characters")

if __name__ == "__main__":
    test_agent_method()