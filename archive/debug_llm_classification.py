#!/usr/bin/env python3
"""Debug LLM classification directly"""

from simple_react_agent import SimpleReActAgent
from llm import llm

def debug_llm_classification():
    agent = SimpleReActAgent(llm)

    test_query = "show me the recipe I have the most"

    print(f"Testing query: '{test_query}'")
    print("=" * 50)

    # Test if it hits our rules-based patterns
    print("1. Testing rule-based patterns:")

    # Test individual pattern components
    text = test_query.lower().strip()

    # Check frequent keywords
    frequent_keywords = ["most frequent", "most common", "most popular", "frequently used", "most used"]
    freq_match = any(keyword in text for keyword in frequent_keywords)
    print(f"   Frequent keywords match: {freq_match}")

    # Check always patterns
    import re
    always_patterns = [
        r"recipe.*?(?:i|we).*?always.*?(?:have|make|use|cook)",
        r"(?:i|we).*?always.*?(?:have|make|use|cook).*?recipe",
        r"recipe.*?(?:i|we).*?always",
        r"always.*?(?:making|cooking|using)(?!.*?new|.*?with)",
        r"go[- ]?to recipe",
        r"favorite recipe",
        r"usual recipe",
        r"regular recipe",
        r"staple recipe",
        r"signature recipe",
        r"what.*?(?:i|we).*?usually.*?(?:cook|make)",
        r"what.*?recipe.*?(?:i|we).*?always",
        r"(?:recipe|dish).*?(?:i|we).*?always.*?(?:make|have|use)"
    ]

    always_match = any(re.search(pattern, text, re.IGNORECASE) for pattern in always_patterns)
    print(f"   Always patterns match: {always_match}")

    # Check recent recipes patterns
    recent_keywords = ["recent", "latest", "last", "newest"]
    show_keywords = ["show", "list", "display", "see"]
    recent_match = any(r in text for r in recent_keywords)
    show_match = any(s in text for s in show_keywords)
    print(f"   Recent keywords match: {recent_match}")
    print(f"   Show keywords match: {show_match}")
    print(f"   Would trigger recent recipes: {recent_match and show_match}")

    # Check search patterns
    search_keywords = ["find", "search", "look for"]
    explicit_search = any(keyword in text for keyword in search_keywords)
    show_me_analytics = ("show me" in text and
                       any(word in text for word in ["most", "frequent", "often", "always", "usually"]))
    print(f"   Explicit search: {explicit_search}")
    print(f"   Show me analytics: {show_me_analytics}")
    print(f"   Would trigger search: {explicit_search and not show_me_analytics}")

    print("\n2. Testing LLM classification directly:")
    llm_result = agent._classify_intent_with_llm(text)
    print(f"   LLM result: {llm_result}")

    print("\n3. Testing full intent detection:")
    intent, params = agent.detect_intent(test_query)
    print(f"   Final intent: {intent}, params: {params}")

if __name__ == "__main__":
    debug_llm_classification()