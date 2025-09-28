#!/usr/bin/env python3
"""Test specific historical recipe requests"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def test_specific_historical():
    agent = SimpleReActAgent(llm)

    test_cases = [
        ("give me the previous salmon recipe", "Pan-Seared Salmon Cakes with Lemon-Dill Sauce"),
        ("show me the last chicken recipe", "Sun-Kissed Mediterranean Chicken Shawarma"),
        ("the previous lamb recipe", "Lamb Kofta Shawarma Wrap"),
        ("give me a previous recipe", "Sun-Kissed Mediterranean Chicken Shawarma"),  # Most recent
    ]

    print("Testing specific historical recipe requests:")
    print("=" * 50)

    for input_text, expected_title in test_cases:
        intent, params = agent.detect_intent(input_text)
        recipe_name = params.get("recipe_name", "")

        if recipe_name == expected_title:
            print(f"✅ '{input_text}' -> Found: {recipe_name}")
        else:
            print(f"❌ '{input_text}' -> Expected: {expected_title}, Got: {recipe_name}")

if __name__ == "__main__":
    test_specific_historical()