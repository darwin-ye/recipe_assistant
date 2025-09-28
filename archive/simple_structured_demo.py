#!/usr/bin/env python3
"""
Simple demo showing structured JSON output vs traditional multiple calls
"""

import json
import time
from llm import llm


def traditional_approach(query):
    """Traditional approach: Multiple separate LLM calls"""
    print(f"🔄 Traditional Approach for: '{query}'")

    # Call 1: Intent Classification
    start = time.time()
    intent_prompt = f"What is the user's intent in this query: '{query}'. Respond with just: create_recipe, search, analytics, scale, or help."
    intent_response = llm.invoke(intent_prompt)
    intent = intent_response.content.strip()
    call1_time = time.time() - start

    # Call 2: Entity Extraction
    start = time.time()
    entity_prompt = f"Extract ingredients and numbers from: '{query}'. List ingredients and numbers separately."
    entity_response = llm.invoke(entity_prompt)
    entities = entity_response.content.strip()
    call2_time = time.time() - start

    # Call 3: Confidence Assessment
    start = time.time()
    confidence_prompt = f"Rate confidence 0-1 for understanding this query: '{query}'"
    confidence_response = llm.invoke(confidence_prompt)
    confidence = confidence_response.content.strip()
    call3_time = time.time() - start

    total_time = call1_time + call2_time + call3_time

    print(f"   Call 1 (Intent): {intent} ({call1_time:.2f}s)")
    print(f"   Call 2 (Entities): {entities[:50]}... ({call2_time:.2f}s)")
    print(f"   Call 3 (Confidence): {confidence} ({call3_time:.2f}s)")
    print(f"   📊 Total: {total_time:.2f}s, 3 API calls")
    return {"intent": intent, "entities": entities, "confidence": confidence, "time": total_time, "calls": 3}


def structured_json_approach(query):
    """Structured JSON approach: Single comprehensive call"""
    print(f"⚡ Structured JSON Approach for: '{query}'")

    prompt = f"""Analyze this query and return ONLY a JSON object:

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

    start = time.time()
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    total_time = time.time() - start

    try:
        # Clean the response to get just JSON
        if '{' in response_text:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_text = response_text[json_start:json_end]
            result = json.loads(json_text)

            print(f"   Intent: {result.get('intent', 'unknown')}")
            print(f"   Entities: {result.get('entities', {})}")
            print(f"   Confidence: {result.get('confidence', 'unknown')}")
            print(f"   Reasoning: {result.get('reasoning', 'none')}")
            print(f"   📊 Total: {total_time:.2f}s, 1 API call")

            return {"result": result, "time": total_time, "calls": 1, "success": True}
        else:
            print(f"   ❌ No JSON found in response: {response_text[:100]}")
            return {"result": {}, "time": total_time, "calls": 1, "success": False}

    except json.JSONDecodeError as e:
        print(f"   ❌ JSON parsing error: {e}")
        print(f"   Raw response: {response_text[:200]}")
        return {"result": {}, "time": total_time, "calls": 1, "success": False}


def compare_approaches():
    """Compare both approaches with real examples"""

    test_queries = [
        "create a chicken pasta recipe for 6 people",
        "show me recent recipes",
        "what's my most frequent recipe?",
        "scale this to 8 people"
    ]

    print("🧪 STRUCTURED JSON vs TRADITIONAL COMPARISON")
    print("=" * 80)

    traditional_total_time = 0
    structured_total_time = 0
    traditional_total_calls = 0
    structured_total_calls = 0
    structured_successes = 0

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test {i}/4: '{query}'")
        print("-" * 60)

        # Traditional approach
        trad_result = traditional_approach(query)
        traditional_total_time += trad_result["time"]
        traditional_total_calls += trad_result["calls"]

        print()

        # Structured approach
        struct_result = structured_json_approach(query)
        structured_total_time += struct_result["time"]
        structured_total_calls += struct_result["calls"]
        if struct_result["success"]:
            structured_successes += 1

        print()

    # Summary
    print("=" * 80)
    print("📊 FINAL COMPARISON RESULTS")
    print("=" * 80)
    print(f"Traditional Approach:")
    print(f"   ⏱️  Total Time: {traditional_total_time:.2f}s")
    print(f"   📞 Total API Calls: {traditional_total_calls}")
    print(f"   💰 Cost: {traditional_total_calls}x base cost")
    print()
    print(f"Structured JSON Approach:")
    print(f"   ⏱️  Total Time: {structured_total_time:.2f}s")
    print(f"   📞 Total API Calls: {structured_total_calls}")
    print(f"   💰 Cost: {structured_total_calls}x base cost")
    print(f"   ✅ Success Rate: {structured_successes}/{len(test_queries)} ({structured_successes/len(test_queries)*100:.0f}%)")
    print()

    if structured_total_time > 0:
        time_savings = (traditional_total_time - structured_total_time) / traditional_total_time * 100
        cost_savings = (traditional_total_calls - structured_total_calls) / traditional_total_calls * 100

        print("🏆 STRUCTURED JSON ADVANTAGES:")
        print(f"   ⚡ {time_savings:.1f}% faster execution")
        print(f"   💸 {cost_savings:.1f}% lower API costs")
        print(f"   🎯 Single atomic operation (no state inconsistency)")
        print(f"   🧠 Better context understanding (all tasks see full picture)")
        print(f"   🔄 Easier error handling (one call to manage)")


def show_example_outputs():
    """Show concrete examples of what each approach returns"""

    print("\n📋 CONCRETE EXAMPLE OUTPUTS")
    print("=" * 60)

    query = "create a chicken pasta recipe for 6 people"
    print(f"Query: '{query}'")
    print()

    print("🔄 Traditional Approach Returns:")
    print("   Intent Call: 'create_recipe'")
    print("   Entity Call: 'Ingredients: chicken, pasta. Numbers: 6.'")
    print("   Confidence Call: '0.9'")
    print("   → 3 separate strings to parse and combine")
    print()

    print("⚡ Structured JSON Returns:")
    example_json = {
        "intent": "create_recipe",
        "entities": {
            "ingredients": ["chicken", "pasta"],
            "numbers": [6],
            "servings": 6
        },
        "confidence": 0.9,
        "reasoning": "Clear recipe creation request with specific ingredients and serving size"
    }
    print("   ", json.dumps(example_json, indent=3))
    print("   → Single structured object with all information")


if __name__ == "__main__":
    print("🚀 STRUCTURED JSON OUTPUT DEMONSTRATION")
    print("This shows advanced prompt engineering for multiple detection tasks")
    print()

    show_example_outputs()
    compare_approaches()

    print("\n💡 KEY TAKEAWAY:")
    print("Structured JSON output is a production-level prompt engineering technique")
    print("that optimizes performance, cost, and consistency for multi-task LLM analysis.")