#!/usr/bin/env python3
"""
Example of structured JSON output for multiple detection tasks in a single LLM call
This demonstrates advanced prompt engineering for the recipe assistant
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class StructuredAnalysis:
    """Structured result from multi-task LLM analysis"""
    intent: str
    entities: Dict[str, Any]
    confidence: float
    reasoning: str
    follow_up_needed: bool
    context_dependencies: List[str]
    suggested_actions: List[str]


class StructuredJSONAnalyzer:
    """Advanced analyzer using structured JSON output for multiple tasks"""

    def __init__(self, llm):
        self.llm = llm

    def analyze_query_comprehensive(self, user_input: str, context: Optional[Dict] = None) -> StructuredAnalysis:
        """
        Perform multiple detection tasks in a single LLM call using structured JSON output

        Tasks performed simultaneously:
        1. Intent classification
        2. Entity extraction (ingredients, servings, dietary needs)
        3. Confidence scoring
        4. Context dependency analysis
        5. Follow-up determination
        6. Action suggestion
        """

        # Build context information
        context_info = ""
        if context:
            if context.get("current_recipe"):
                recipe = context["current_recipe"]
                context_info += f"Current recipe: '{recipe.title}' (serves {recipe.servings})\n"
            if context.get("recent_recipes"):
                context_info += f"Recent recipes: {len(context['recent_recipes'])} available\n"

        prompt = f"""You are an advanced recipe assistant analyzer. Analyze this user query and return a comprehensive JSON analysis performing multiple detection tasks simultaneously.

User query: "{user_input}"

Context:
{context_info}

Perform ALL these tasks in a single analysis and return EXACTLY this JSON structure:

{{
    "intent": "one of: create_recipe|search_recipes|get_recent|get_details|analytics_frequent|analytics_count|scale_recipe|numbered_reference|help",
    "entities": {{
        "ingredients": ["list", "of", "mentioned", "ingredients"],
        "servings": null_or_number,
        "dietary_restrictions": ["vegetarian", "gluten-free", "etc"],
        "recipe_reference": "specific_recipe_name_if_mentioned",
        "numbers": [1, 2, 3],
        "time_references": ["recent", "last", "previous"],
        "measurement_units": ["cups", "tablespoons", "etc"]
    }},
    "confidence": 0.0_to_1.0,
    "reasoning": "brief explanation of why this intent was chosen and what entities were found",
    "follow_up_needed": true_or_false,
    "context_dependencies": ["current_recipe", "user_history", "recipe_database", "none"],
    "suggested_actions": ["action1", "action2", "action3"],
    "ambiguity_flags": {{
        "multiple_possible_intents": false,
        "unclear_entities": false,
        "missing_information": false
    }},
    "parameters_for_execution": {{
        "key1": "value1",
        "key2": "value2"
    }}
}}

IMPORTANT GUIDELINES:
1. Intent MUST be one of the specified options
2. Set confidence based on clarity of the query (0.9+ for clear, 0.7-0.8 for somewhat clear, <0.7 for ambiguous)
3. Extract ALL entities present, use null/empty for missing ones
4. Identify what context dependencies are needed to fulfill this request
5. Suggest 1-3 concrete actions the system should take
6. Flag any ambiguities or missing information
7. Provide parameters ready for function execution

Examples:

Query: "create a chicken pasta recipe for 6 people"
→ Intent: create_recipe, Entities: ingredients=["chicken","pasta"], servings=6

Query: "show me recipe 2"
→ Intent: numbered_reference, Entities: numbers=[2], Context: ["recipe_database"]

Query: "scale this to 8 people"
→ Intent: scale_recipe, Entities: servings=8, Context: ["current_recipe"]

Query: "what's my go-to recipe?"
→ Intent: analytics_frequent, Context: ["user_history"]

Return ONLY the JSON object, no additional text:"""

        try:
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse JSON response with robust error handling
            analysis_data = self._parse_json_with_fallback(response_text.strip())

            # Convert to structured object
            return StructuredAnalysis(
                intent=analysis_data.get("intent", "help"),
                entities=analysis_data.get("entities", {}),
                confidence=float(analysis_data.get("confidence", 0.5)),
                reasoning=analysis_data.get("reasoning", ""),
                follow_up_needed=analysis_data.get("follow_up_needed", False),
                context_dependencies=analysis_data.get("context_dependencies", []),
                suggested_actions=analysis_data.get("suggested_actions", [])
            )

        except Exception as e:
            print(f"Analysis error: {e}")
            return self._fallback_analysis(user_input)

    def _parse_json_with_fallback(self, response_text: str) -> dict:
        """Robust JSON parsing with multiple fallback strategies"""
        import re

        try:
            # Strategy 1: Direct JSON parsing
            parsed = json.loads(response_text)
            # Ensure we return a dict, not None or other types
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from mixed content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(0)
                return json.loads(json_text)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Clean and repair common issues
        try:
            cleaned_text = self._clean_json_response(response_text)
            if cleaned_text:
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    pass
        except:
            pass

        # Strategy 4: Return basic structure for rule-based fallback
        return {"intent": "help", "entities": {}, "confidence": 0.3, "reasoning": "JSON parsing failed"}

    def _clean_json_response(self, response_text: str) -> str:
        """Clean common JSON formatting issues"""
        import re

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

    def _fallback_analysis(self, user_input: str) -> StructuredAnalysis:
        """Fallback analysis when JSON parsing fails"""
        return StructuredAnalysis(
            intent="help",
            entities={},
            confidence=0.3,
            reasoning="Failed to parse LLM response",
            follow_up_needed=True,
            context_dependencies=["clarification"],
            suggested_actions=["ask_for_clarification"]
        )

    def compare_approaches(self, test_queries: List[str]):
        """Compare single JSON call vs multiple separate calls"""

        print("🔬 Comparing Structured JSON vs Multiple Calls")
        print("=" * 60)

        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 40)

            # Structured JSON approach (1 call)
            import time
            start_time = time.time()
            structured_result = self.analyze_query_comprehensive(query)
            structured_time = time.time() - start_time

            print(f"📊 Structured JSON (1 call, {structured_time:.2f}s):")
            print(f"   Intent: {structured_result.intent}")
            print(f"   Entities: {structured_result.entities}")
            print(f"   Confidence: {structured_result.confidence}")
            print(f"   Actions: {structured_result.suggested_actions}")

            # Multiple calls approach would be:
            # call 1: classify_intent(query)      ~2s
            # call 2: extract_entities(query)     ~2s
            # call 3: get_confidence(query)       ~2s
            # call 4: suggest_actions(query)      ~2s
            # Total: ~8s vs ~2s for structured approach

            print(f"⚡ Traditional Multiple Calls (estimated ~8s):")
            print(f"   Would require 4 separate LLM calls")
            print(f"   4x the cost and time")


# Example usage and benefits
if __name__ == "__main__":
    from llm import llm

    analyzer = StructuredJSONAnalyzer(llm)

    test_queries = [
        "create a chicken pasta recipe for 6 people",
        "show me recipe 2",
        "scale this to 8 people",
        "what's my most frequent recipe?",
        "how many beef recipes do I have?"
    ]

    print("🚀 Advanced Structured JSON Analysis")
    print("=" * 50)

    for query in test_queries:
        print(f"\nAnalyzing: '{query}'")
        result = analyzer.analyze_query_comprehensive(query)

        print(f"Intent: {result.intent}")
        print(f"Entities: {result.entities}")
        print(f"Confidence: {result.confidence}")
        print(f"Actions: {result.suggested_actions}")
        print(f"Context needed: {result.context_dependencies}")

    # Show comparison
    analyzer.compare_approaches(test_queries[:3])