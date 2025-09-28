#!/usr/bin/env python3
"""
Improved structured JSON output with robust error handling and better prompting
"""

import json
import time
import re
from llm import llm


def improved_structured_approach(query):
    """Improved structured JSON approach with better prompting and error handling"""
    print(f"⚡ Improved Structured JSON Approach for: '{query}'")

    # More explicit and strict prompt
    prompt = f"""You are a recipe assistant analyzer. Analyze this query and return EXACTLY one valid JSON object with no additional text.

Query: "{query}"

Rules:
1. Return ONLY the JSON object, no other text
2. Use actual values, not template placeholders
3. For confidence, use a decimal number between 0.0 and 1.0
4. For servings, use an actual number or null (not "null_or_number")
5. All strings must be in quotes, all arrays properly formatted

{{
    "intent": "create_recipe",
    "entities": {{
        "ingredients": ["actual", "ingredient", "names"],
        "numbers": [6, 8],
        "servings": 6
    }},
    "confidence": 0.9,
    "reasoning": "actual explanation here"
}}

For query "{query}", return the JSON:"""

    start = time.time()
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    total_time = time.time() - start

    return parse_json_with_fallback(response_text, total_time, query)


def parse_json_with_fallback(response_text, total_time, query):
    """Robust JSON parsing with multiple fallback strategies"""

    try:
        # Strategy 1: Direct JSON parsing
        result = json.loads(response_text)
        print(f"   Intent: {result.get('intent', 'unknown')}")
        print(f"   Entities: {result.get('entities', {})}")
        print(f"   Confidence: {result.get('confidence', 'unknown')}")
        print(f"   Reasoning: {result.get('reasoning', 'none')}")
        print(f"   📊 Total: {total_time:.2f}s, 1 API call")
        return {"result": result, "time": total_time, "calls": 1, "success": True}

    except json.JSONDecodeError:
        print(f"   ⚠️  Direct parsing failed, trying extraction...")

        # Strategy 2: Extract JSON from mixed content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(0)
                result = json.loads(json_text)
                print(f"   Intent: {result.get('intent', 'unknown')}")
                print(f"   Entities: {result.get('entities', {})}")
                print(f"   Confidence: {result.get('confidence', 'unknown')}")
                print(f"   Reasoning: {result.get('reasoning', 'none')}")
                print(f"   📊 Total: {total_time:.2f}s, 1 API call (extracted)")
                return {"result": result, "time": total_time, "calls": 1, "success": True}
            except json.JSONDecodeError:
                pass

        # Strategy 3: Clean and repair common issues
        cleaned_text = clean_json_response(response_text)
        if cleaned_text:
            try:
                result = json.loads(cleaned_text)
                print(f"   Intent: {result.get('intent', 'unknown')}")
                print(f"   Entities: {result.get('entities', {})}")
                print(f"   Confidence: {result.get('confidence', 'unknown')}")
                print(f"   Reasoning: {result.get('reasoning', 'none')}")
                print(f"   📊 Total: {total_time:.2f}s, 1 API call (repaired)")
                return {"result": result, "time": total_time, "calls": 1, "success": True}
            except json.JSONDecodeError:
                pass

        # Strategy 4: Rule-based fallback
        print(f"   ❌ All JSON parsing failed, using rule-based fallback")
        print(f"   Raw response: {response_text[:100]}...")
        fallback_result = create_fallback_result(query)
        print(f"   Intent: {fallback_result['intent']}")
        print(f"   Entities: {fallback_result['entities']}")
        print(f"   Confidence: {fallback_result['confidence']}")
        print(f"   📊 Total: {total_time:.2f}s, 1 API call (fallback)")
        return {"result": fallback_result, "time": total_time, "calls": 1, "success": False}


def clean_json_response(response_text):
    """Clean common JSON formatting issues"""

    # Remove common prefixes
    text = response_text
    text = re.sub(r'^.*?Here is the JSON.*?:', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'^.*?JSON.*?:', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = text.strip()

    # Find JSON boundaries
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if not json_match:
        return None

    json_text = json_match.group(0)

    # Fix common template issues
    json_text = json_text.replace('null_or_number', 'null')
    json_text = json_text.replace('0.0_to_1.0', '0.5')
    json_text = re.sub(r'"[^"]*null_or_number[^"]*"', 'null', json_text)

    # Fix incomplete strings
    json_text = re.sub(r':\s*"[^"]*$', ': "incomplete"', json_text, flags=re.MULTILINE)

    return json_text


def create_fallback_result(query):
    """Create reasonable fallback result based on query analysis"""

    query_lower = query.lower()

    # Simple rule-based intent detection
    if any(word in query_lower for word in ['create', 'make', 'recipe for']):
        intent = 'create_recipe'
    elif any(word in query_lower for word in ['show', 'recent', 'list']):
        intent = 'search'
    elif 'frequent' in query_lower or 'most' in query_lower:
        intent = 'analytics_frequent'
    elif 'scale' in query_lower:
        intent = 'scale'
    else:
        intent = 'help'

    # Extract numbers
    numbers = re.findall(r'\d+', query)
    numbers = [int(n) for n in numbers]

    # Extract common ingredients
    ingredients = []
    for ingredient in ['chicken', 'pasta', 'beef', 'rice', 'tomato']:
        if ingredient in query_lower:
            ingredients.append(ingredient)

    return {
        "intent": intent,
        "entities": {
            "ingredients": ingredients,
            "numbers": numbers,
            "servings": numbers[0] if numbers else None
        },
        "confidence": 0.6,
        "reasoning": "Rule-based fallback analysis"
    }


def compare_all_approaches():
    """Compare original, improved, and traditional approaches"""

    test_queries = [
        "create a chicken pasta recipe for 6 people",
        "show me recent recipes",
        "what's my most frequent recipe?",
        "scale this to 8 people"
    ]

    print("🔬 COMPREHENSIVE APPROACH COMPARISON")
    print("=" * 80)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}/4: '{query}'")
        print("-" * 60)

        # Original structured approach (from previous demo)
        original_start = time.time()
        original_prompt = f"""Analyze this query and return ONLY a JSON object:

Query: "{query}"

{{
    "intent": "create_recipe|search|analytics|scale|help",
    "entities": {{
        "ingredients": ["list", "of", "ingredients"],
        "numbers": [1, 2, 3],
        "servings": null_or_number
    }},
    "confidence": 0.0_to_1.0,
    "reasoning": "brief explanation"
}}

JSON:"""

        original_response = llm.invoke(original_prompt)
        original_time = time.time() - original_start

        print(f"📊 Original Structured (1 call, {original_time:.2f}s):")
        try:
            original_result = json.loads(original_response.content.strip())
            print(f"   ✅ Success: {original_result.get('intent', 'unknown')}")
        except:
            print(f"   ❌ JSON parsing failed")

        # Improved structured approach
        improved_result = improved_structured_approach(query)

        print()


if __name__ == "__main__":
    print("🚀 IMPROVED STRUCTURED JSON DEMONSTRATION")
    print("This shows enhanced prompt engineering with robust error handling")
    print()

    compare_all_approaches()

    print("\n🎯 IMPROVEMENTS MADE:")
    print("1. More explicit JSON formatting instructions")
    print("2. Removed template placeholders from examples")
    print("3. Multi-strategy JSON parsing with fallbacks")
    print("4. Rule-based fallback for failed JSON parsing")
    print("5. JSON cleaning and repair mechanisms")