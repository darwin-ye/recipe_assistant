#!/usr/bin/env python3
"""Demo the new analytics features"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def demo_analytics():
    agent = SimpleReActAgent(llm)

    queries = [
        "show me the most frequent recipe",
        "how many recipes with chicken",
        "count recipes with salmon",
    ]

    print("🎬 Analytics Features Demo")
    print("=" * 60)

    for query in queries:
        print(f"\n🗣️  User: {query}")
        print("─" * 40)
        result = agent.run(query)
        print(f"{result}")
        print()

if __name__ == "__main__":
    demo_analytics()