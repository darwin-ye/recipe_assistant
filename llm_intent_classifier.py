# llm_intent_classifier.py - Pure LLM Intent Classification System
"""
A pure LLM-based intent classification system that replaces rule-based patterns
with natural language understanding. Provides more flexible and maintainable
intent detection for the recipe assistant.
"""

import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class IntentResult:
    """Structured result from LLM intent classification"""
    intent: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str


class LLMIntentClassifier:
    """Pure LLM-based intent classification system"""

    def __init__(self, llm):
        self.llm = llm

        # Define available intents with descriptions
        self.intent_definitions = {
            "create_recipe": {
                "description": "User wants to create a new recipe",
                "parameters": ["ingredients", "dietary_needs", "servings"],
                "examples": [
                    "create a chicken pasta recipe",
                    "make something with beef and vegetables",
                    "give me a salmon recipe",
                    "new recipe for 4 people"
                ]
            },
            "search_recipes": {
                "description": "User wants to search existing recipes",
                "parameters": ["query", "ingredients", "cuisine"],
                "examples": [
                    "find pasta recipes",
                    "search for chicken dishes",
                    "recipes with tomatoes",
                    "show me Italian recipes"
                ]
            },
            "get_recent": {
                "description": "User wants to see recent recipes",
                "parameters": ["limit"],
                "examples": [
                    "show me recent recipes",
                    "what did I make lately",
                    "my latest recipes",
                    "recent cooking history"
                ]
            },
            "get_details": {
                "description": "User wants detailed information about a specific recipe",
                "parameters": ["recipe_name", "recipe_reference"],
                "examples": [
                    "show me the salmon recipe",
                    "give me details for chicken pasta",
                    "how do I make the beef stew",
                    "recipe instructions for pizza"
                ]
            },
            "analytics_frequent": {
                "description": "User wants to know their most frequent/favorite recipe",
                "parameters": [],
                "examples": [
                    "what's my most frequent recipe",
                    "show me the recipe I always have",
                    "my go-to recipe",
                    "what do I cook most often"
                ]
            },
            "analytics_count": {
                "description": "User wants to count recipes by ingredient",
                "parameters": ["ingredient"],
                "examples": [
                    "how many chicken recipes do I have",
                    "count recipes with beef",
                    "how often do I use tomatoes",
                    "frequency of pasta in my recipes"
                ]
            },
            "scale_recipe": {
                "description": "User wants to scale a recipe for different servings",
                "parameters": ["target_servings", "recipe_reference"],
                "examples": [
                    "scale this to 8 people",
                    "make it for 6 servings",
                    "adjust the recipe for 2 people",
                    "double the recipe"
                ]
            },
            "numbered_reference": {
                "description": "User is selecting a recipe by number from a list",
                "parameters": ["number"],
                "examples": [
                    "2",
                    "number 3",
                    "recipe 1",
                    "show me #4"
                ]
            },
            "help": {
                "description": "User wants help or doesn't know what to do",
                "parameters": [],
                "examples": [
                    "help",
                    "what can you do",
                    "I don't know",
                    "options"
                ]
            }
        }

    def classify_intent(self, user_input: str, context: Optional[Dict] = None) -> IntentResult:
        """
        Classify user intent using pure LLM understanding

        Args:
            user_input: The user's natural language input
            context: Optional context (current recipe, previous actions, etc.)

        Returns:
            IntentResult with intent, parameters, confidence, and reasoning
        """

        # Build context information
        context_info = ""
        if context:
            if context.get("current_recipe"):
                recipe = context["current_recipe"]
                context_info += f"\nCurrent recipe: '{recipe.title}' (serves {recipe.servings})"

            if context.get("last_action"):
                context_info += f"\nLast action: {context['last_action']}"

            if context.get("recent_recipes"):
                context_info += f"\nRecent recipes available: {len(context['recent_recipes'])} recipes"

        # Create comprehensive prompt
        prompt = self._build_classification_prompt(user_input, context_info)

        try:
            # Get LLM response
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse the structured response
            return self._parse_llm_response(response_text, user_input)

        except Exception as e:
            print(f"LLM classification error: {e}")
            # Fallback to help intent
            return IntentResult(
                intent="help",
                parameters={},
                confidence=0.5,
                reasoning=f"Error in classification: {str(e)}"
            )

    def _build_classification_prompt(self, user_input: str, context_info: str) -> str:
        """Build a comprehensive prompt for intent classification"""

        # Create intent definitions section
        intents_section = ""
        for intent, definition in self.intent_definitions.items():
            intents_section += f"\n**{intent}**:\n"
            intents_section += f"  Description: {definition['description']}\n"
            intents_section += f"  Parameters: {', '.join(definition['parameters'])}\n"
            intents_section += f"  Examples: {', '.join(definition['examples'][:2])}\n"

        prompt = f"""You are an expert intent classifier for a recipe assistant. Analyze the user's input and determine their intent with high accuracy.

USER INPUT: "{user_input}"

CONTEXT:{context_info}

AVAILABLE INTENTS:{intents_section}

CLASSIFICATION RULES:
1. Choose the MOST SPECIFIC intent that matches the user's request
2. If the user gives just a number (1, 2, 3, etc.), it's "numbered_reference"
3. If asking about frequency/most common recipes, it's "analytics_frequent"
4. If asking to count recipes by ingredient, it's "analytics_count"
5. If mentioning scaling/servings/people, it's "scale_recipe"
6. If wanting to create/make/generate new recipes, it's "create_recipe"
7. If searching for existing recipes, it's "search_recipes"
8. If asking for recent/latest recipes, it's "get_recent"
9. If asking for details of a specific recipe, it's "get_details"
10. If unclear or asking for help, it's "help"

PARAMETER EXTRACTION:
- Extract relevant parameters from the user input
- For ingredients: extract food items mentioned
- For servings: extract numbers related to people/servings
- For recipe references: extract recipe names or descriptors

OUTPUT FORMAT (return EXACTLY this JSON structure):
{{
    "intent": "intent_name",
    "parameters": {{
        "param1": "value1",
        "param2": "value2"
    }},
    "confidence": 0.95,
    "reasoning": "Brief explanation of why this intent was chosen"
}}

IMPORTANT: Return ONLY the JSON structure, no additional text or formatting."""

        return prompt

    def _parse_llm_response(self, response_text: str, user_input: str) -> IntentResult:
        """Parse LLM response into structured IntentResult with robust error handling"""

        try:
            # Use improved JSON parsing with multiple fallback strategies
            result_data = self._parse_json_with_fallback(response_text.strip())

            # Validate and extract data
            intent = result_data.get("intent", "help")
            parameters = result_data.get("parameters", {})
            confidence = float(result_data.get("confidence", 0.8))
            reasoning = result_data.get("reasoning", "LLM classification")

            # Validate intent exists
            if intent not in self.intent_definitions:
                intent = "help"
                reasoning = f"Invalid intent '{intent}' returned by LLM"

            return IntentResult(
                intent=intent,
                parameters=parameters,
                confidence=confidence,
                reasoning=reasoning
            )

        except Exception as e:
            print(f"Response parsing error: {e}")
            return IntentResult(
                intent="help",
                parameters={},
                confidence=0.5,
                reasoning=f"Parsing error: {str(e)}"
            )

    def _fallback_parse(self, response_text: str, user_input: str) -> IntentResult:
        """Fallback parsing when JSON extraction fails"""

        response_lower = response_text.lower()

        # Try to extract intent from response text
        for intent in self.intent_definitions.keys():
            if intent in response_lower:
                return IntentResult(
                    intent=intent,
                    parameters={},
                    confidence=0.7,
                    reasoning="Fallback text parsing"
                )

        # Final fallback based on simple keywords in user input
        user_lower = user_input.lower().strip()

        if user_lower.isdigit():
            return IntentResult(
                intent="numbered_reference",
                parameters={"number": int(user_lower)},
                confidence=0.9,
                reasoning="Simple number detection"
            )
        elif any(word in user_lower for word in ["create", "make", "new", "generate"]):
            return IntentResult(
                intent="create_recipe",
                parameters={},
                confidence=0.8,
                reasoning="Create keyword detection"
            )
        elif any(word in user_lower for word in ["recent", "latest", "history"]):
            return IntentResult(
                intent="get_recent",
                parameters={},
                confidence=0.8,
                reasoning="Recent keyword detection"
            )
        else:
            return IntentResult(
                intent="help",
                parameters={},
                confidence=0.6,
                reasoning="Unable to determine intent"
            )

    def _parse_json_with_fallback(self, response_text: str) -> dict:
        """Robust JSON parsing with multiple fallback strategies - targeting 100% success"""
        import re

        try:
            # Strategy 1: Direct JSON parsing
            parsed = json.loads(response_text)
            # Ensure we return a dict, not None or other types
            if isinstance(parsed, dict):
                # Handle empty dict case
                if not parsed:
                    return {"intent": "help", "confidence": 0.3, "entities": {}}
                # Normalize field names
                return self._normalize_field_names(parsed)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract JSON from mixed content
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                json_text = json_match.group(0)
                parsed = json.loads(json_text)
                if isinstance(parsed, dict):
                    if not parsed:
                        return {"intent": "help", "confidence": 0.3, "entities": {}}
                    return self._normalize_field_names(parsed)
            except json.JSONDecodeError:
                pass

        # Strategy 3: Clean and repair common issues
        try:
            cleaned_text = self._clean_json_response(response_text)
            if cleaned_text:
                try:
                    parsed = json.loads(cleaned_text)
                    if isinstance(parsed, dict):
                        return self._normalize_field_names(parsed)
                except json.JSONDecodeError:
                    pass
        except:
            pass

        # Strategy 4: Advanced repair with bracket balancing
        try:
            repaired_text = self._advanced_json_repair(response_text)
            if repaired_text:
                parsed = json.loads(repaired_text)
                if isinstance(parsed, dict):
                    return self._normalize_field_names(parsed)
        except:
            pass

        # Strategy 5: Line-by-line reconstruction
        try:
            reconstructed = self._reconstruct_json_from_lines(response_text)
            if reconstructed:
                parsed = json.loads(reconstructed)
                if isinstance(parsed, dict):
                    return self._normalize_field_names(parsed)
        except:
            pass

        # Strategy 6: Pattern-based field extraction
        try:
            pattern_extracted = self._extract_fields_by_pattern(response_text)
            if pattern_extracted:
                return self._normalize_field_names(pattern_extracted)
        except:
            pass

        # Strategy 7: Return basic structure for rule-based fallback
        return {"intent": "help", "parameters": {}, "confidence": 0.3, "reasoning": "JSON parsing failed"}

    def _clean_json_response(self, response_text: str) -> str:
        """Clean common JSON formatting issues"""
        import re

        # Handle edge cases first
        if not response_text or not isinstance(response_text, str):
            return None

        # Remove common prefixes
        text = response_text
        text = re.sub(r'^.*?Here is the JSON.*?:', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r'^.*?JSON.*?:', '', text, flags=re.IGNORECASE | re.DOTALL)
        text = text.strip()

        # Handle common non-JSON responses
        if text.lower() in ['null', 'undefined', 'none', '']:
            return None

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

    def _advanced_json_repair(self, response_text: str) -> str:
        """Advanced JSON repair with bracket balancing and completion"""
        import re

        # Look for any JSON-like content more broadly
        json_match = re.search(r'\{.*', response_text, re.DOTALL)
        if not json_match:
            return None

        json_text = json_match.group(0)

        # Handle cases where JSON is cut off mid-word
        if '"intent": "create_recipe' in json_text and json_text.endswith(' and '):
            # Extract just the clean part
            clean_match = re.search(r'\{[^}]*"intent":\s*"[^"]*"[^}]*\}', json_text)
            if clean_match:
                json_text = clean_match.group(0)

        # Count brackets and fix imbalance
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')

        if open_braces > close_braces:
            # Add missing closing braces
            json_text += '}' * (open_braces - close_braces)

        # Fix unquoted values
        json_text = re.sub(r':\s*([^",}\]\s][^",}\]]*)\s*([,}])', r': "\1"\2', json_text)

        # Fix trailing commas
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)

        # Clean up any malformed trailing content
        json_text = re.sub(r'\s*(and|the)\s*$', '', json_text)

        return json_text

    def _reconstruct_json_from_lines(self, response_text: str) -> str:
        """Reconstruct JSON by parsing lines for key-value pairs"""
        import re

        lines = response_text.split('\n')
        json_obj = {}

        for line in lines:
            # Look for key: value patterns
            key_value_match = re.search(r'"?(\w+)"?\s*:\s*(.+)', line.strip())
            if key_value_match:
                key = key_value_match.group(1)
                value = key_value_match.group(2).strip()

                # Clean up value
                value = value.rstrip(',')

                # Try to parse as JSON value
                try:
                    if value.startswith('"') and value.endswith('"'):
                        json_obj[key] = value[1:-1]  # Remove quotes
                    elif value.lower() in ['true', 'false']:
                        json_obj[key] = value.lower() == 'true'
                    elif value.lower() == 'null':
                        json_obj[key] = None
                    elif value.startswith('[') and value.endswith(']'):
                        json_obj[key] = json.loads(value)
                    elif value.replace('.', '').isdigit():
                        json_obj[key] = float(value) if '.' in value else int(value)
                    else:
                        json_obj[key] = value.strip('"')
                except:
                    json_obj[key] = value.strip('"')

        return json.dumps(json_obj) if json_obj else None

    def _extract_fields_by_pattern(self, response_text: str) -> dict:
        """Extract fields using regex patterns as last resort"""
        import re

        result = {}

        # Extract intent
        intent_match = re.search(r'intent["\']?\s*:\s*["\']?(\w+)', response_text, re.IGNORECASE)
        if intent_match:
            result["intent"] = intent_match.group(1)

        # Extract confidence
        conf_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', response_text, re.IGNORECASE)
        if conf_match:
            result["confidence"] = float(conf_match.group(1))

        # Extract entities (look for common patterns)
        entities = {}

        # Look for ingredients
        ingredients_match = re.search(r'ingredients["\']?\s*:\s*\[([^\]]*)\]', response_text, re.IGNORECASE)
        if ingredients_match:
            ingredients_str = ingredients_match.group(1)
            entities["ingredients"] = [item.strip().strip('"\'') for item in ingredients_str.split(',') if item.strip()]

        # Look for servings
        servings_match = re.search(r'servings["\']?\s*:\s*(\d+)', response_text, re.IGNORECASE)
        if servings_match:
            entities["servings"] = int(servings_match.group(1))

        if entities:
            result["entities"] = entities

        # Ensure required fields exist
        if "intent" not in result:
            result["intent"] = "help"
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "entities" not in result:
            result["entities"] = {}

        return result

    def _normalize_field_names(self, parsed_dict: dict) -> dict:
        """Normalize different field names to standard format"""
        result = {}

        # Map alternative field names to standard names
        field_mappings = {
            "intent": ["intent", "user_intent", "action", "classification"],
            "confidence": ["confidence", "certainty", "score", "probability"],
            "entities": ["entities", "extracted_entities", "parameters", "data"],
            "reasoning": ["reasoning", "explanation", "rationale", "why"]
        }

        # Apply mappings
        for standard_field, alternatives in field_mappings.items():
            for alt_field in alternatives:
                if alt_field in parsed_dict:
                    result[standard_field] = parsed_dict[alt_field]
                    break

        # Copy any unmapped fields
        for key, value in parsed_dict.items():
            if key not in [alt for alts in field_mappings.values() for alt in alts]:
                result[key] = value

        # Ensure required fields exist
        if "intent" not in result:
            result["intent"] = "help"
        if "confidence" not in result:
            result["confidence"] = 0.5
        if "entities" not in result:
            result["entities"] = {}

        return result

    def get_example_queries(self) -> Dict[str, List[str]]:
        """Get example queries for each intent (useful for testing)"""
        examples = {}
        for intent, definition in self.intent_definitions.items():
            examples[intent] = definition["examples"]
        return examples

    def validate_intent_coverage(self, test_queries: List[str]) -> Dict[str, Any]:
        """Test the classifier with a set of queries and return coverage stats"""
        results = {}
        intent_counts = {}

        for query in test_queries:
            result = self.classify_intent(query)
            results[query] = result

            if result.intent in intent_counts:
                intent_counts[result.intent] += 1
            else:
                intent_counts[result.intent] = 1

        return {
            "total_queries": len(test_queries),
            "intent_distribution": intent_counts,
            "average_confidence": sum(r.confidence for r in results.values()) / len(results),
            "detailed_results": results
        }


# Example usage and testing
if __name__ == "__main__":
    from llm import llm

    # Create classifier
    classifier = LLMIntentClassifier(llm)

    # Test queries
    test_queries = [
        "create a chicken pasta recipe",
        "show me recent recipes",
        "find pasta recipes",
        "2",
        "what's my most frequent recipe",
        "how many chicken recipes do I have",
        "scale this to 8 people",
        "help",
        "give me the salmon recipe",
        "what do I cook most often"
    ]

    print("🧪 Testing LLM Intent Classifier")
    print("=" * 50)

    for query in test_queries:
        result = classifier.classify_intent(query)
        print(f"\nQuery: '{query}'")
        print(f"Intent: {result.intent}")
        print(f"Parameters: {result.parameters}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")