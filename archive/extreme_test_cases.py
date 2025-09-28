#!/usr/bin/env python3
"""
Extreme test cases for robust JSON parsing - testing every possible failure mode
"""

import json
import time
from llm import llm
from llm_intent_classifier import LLMIntentClassifier
from structured_json_example import StructuredJSONAnalyzer


class ExtremeTestSuite:
    """Comprehensive test suite for extreme edge cases"""

    def __init__(self):
        self.llm_classifier = LLMIntentClassifier(llm)
        self.structured_analyzer = StructuredJSONAnalyzer(llm)
        self.test_results = []

    def run_all_extreme_tests(self):
        """Run comprehensive extreme test suite"""
        print("🔥 EXTREME EDGE CASE TEST SUITE")
        print("=" * 80)
        print("Testing every possible failure mode and edge case...")
        print()

        # Test categories
        test_categories = [
            ("Malformed JSON Responses", self.test_malformed_json),
            ("Special Characters & Encoding", self.test_special_characters),
            ("Template Placeholder Issues", self.test_template_issues),
            ("Incomplete & Truncated Responses", self.test_incomplete_responses),
            ("Mixed Content & Prefixes", self.test_mixed_content),
            ("Unicode & International Characters", self.test_unicode_content),
            ("Large & Complex Responses", self.test_large_responses),
            ("Nested JSON & Arrays", self.test_nested_structures),
            ("Empty & Null Responses", self.test_empty_responses),
            ("Performance Under Stress", self.test_stress_performance)
        ]

        total_tests = 0
        total_failures = 0

        for category_name, test_function in test_categories:
            print(f"\n🧪 {category_name}")
            print("-" * 60)

            category_results = test_function()
            category_tests = len(category_results)
            category_failures = sum(1 for r in category_results if not r['success'])

            total_tests += category_tests
            total_failures += category_failures

            print(f"   Tests: {category_tests}, Failures: {category_failures}")
            if category_failures > 0:
                for result in category_results:
                    if not result['success']:
                        print(f"   ❌ {result['test_name']}: {result['error']}")

        # Final summary
        success_rate = ((total_tests - total_failures) / total_tests * 100) if total_tests > 0 else 0
        print("\n" + "=" * 80)
        print("🎯 EXTREME TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests: {total_tests}")
        print(f"Failures: {total_failures}")
        print(f"Success Rate: {success_rate:.1f}%")

        if success_rate >= 95:
            print("🏆 EXCELLENT: System handles extreme cases very well!")
        elif success_rate >= 85:
            print("✅ GOOD: System is robust with minor edge case issues")
        elif success_rate >= 70:
            print("⚠️  FAIR: System needs improvement for edge cases")
        else:
            print("❌ POOR: System struggles with edge cases")

        return self.test_results

    def test_malformed_json(self):
        """Test various malformed JSON responses"""
        test_cases = [
            # Missing closing brace
            '{"intent": "create_recipe", "confidence": 0.9',

            # Missing quotes around keys
            '{intent: "create_recipe", confidence: 0.9}',

            # Trailing commas
            '{"intent": "create_recipe", "confidence": 0.9,}',

            # Missing commas
            '{"intent": "create_recipe" "confidence": 0.9}',

            # Extra closing braces
            '{"intent": "create_recipe", "confidence": 0.9}}',

            # Mismatched brackets
            '{"intent": "create_recipe", "entities": [}',

            # Unescaped quotes
            '{"intent": "create_recipe", "reasoning": "This is a "test" recipe"}',

            # Invalid numbers
            '{"intent": "create_recipe", "confidence": 0.9.5}',

            # Missing opening brace
            '"intent": "create_recipe", "confidence": 0.9}',

            # Double quotes issues
            '{"intent": ""create_recipe"", "confidence": 0.9}'
        ]

        results = []
        for i, malformed_json in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(malformed_json)
                success = isinstance(result, dict) and 'intent' in result
                results.append({
                    'test_name': f'Malformed JSON {i+1}',
                    'success': success,
                    'error': None if success else 'Failed to parse malformed JSON'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Malformed JSON {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_special_characters(self):
        """Test responses with special characters"""
        test_cases = [
            # Special characters in strings
            '{"intent": "create_recipe", "reasoning": "Recipe with café & naïve ingredients"}',

            # Escape sequences
            '{"intent": "create_recipe", "reasoning": "Path: C:\\\\recipes\\\\pasta.txt"}',

            # Unicode escape sequences
            '{"intent": "create_recipe", "reasoning": "\\u0048\\u0065\\u006c\\u006c\\u006f"}',

            # Control characters
            '{"intent": "create_recipe", "reasoning": "Line 1\\nLine 2\\tTabbed"}',

            # Emoji and symbols
            '{"intent": "create_recipe", "reasoning": "🍝 Pasta recipe with ♥ love"}',

            # HTML entities (shouldn't be in JSON but test anyway)
            '{"intent": "create_recipe", "reasoning": "Recipe &amp; cooking &lt;tips&gt;"}',

            # SQL injection attempts
            '{"intent": "DROP TABLE; --", "reasoning": "\'; DROP TABLE recipes; --"}',

            # Script injection
            '{"intent": "<script>alert(\\"XSS\\")</script>", "reasoning": "javascript:void(0)"}',
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                success = isinstance(result, dict) and 'intent' in result
                results.append({
                    'test_name': f'Special Characters {i+1}',
                    'success': success,
                    'error': None if success else 'Failed to handle special characters'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Special Characters {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_template_issues(self):
        """Test template placeholder problems"""
        test_cases = [
            # Template placeholders that should be fixed
            '{"intent": "create_recipe", "confidence": 0.0_to_1.0, "servings": null_or_number}',

            # Mixed template and real values
            '{"intent": "create_recipe", "confidence": 0.8, "servings": null_or_number}',

            # Multiple template issues
            '{"intent": "intent_name", "confidence": 0.0_to_1.0, "servings": null_or_number}',

            # Template in arrays
            '{"intent": "create_recipe", "entities": {"ingredients": ["list", "of", "ingredients"]}}',

            # Complex template nesting
            '{"intent": "create_recipe", "entities": {"key1": "value1", "key2": null_or_number}}',
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                success = isinstance(result, dict) and 'intent' in result
                # Check if templates were properly cleaned
                json_str = json.dumps(result)
                has_templates = any(template in json_str for template in ['null_or_number', '0.0_to_1.0'])
                success = success and not has_templates

                results.append({
                    'test_name': f'Template Issue {i+1}',
                    'success': success,
                    'error': None if success else 'Template placeholders not cleaned'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Template Issue {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_incomplete_responses(self):
        """Test incomplete and truncated responses"""
        test_cases = [
            # Incomplete JSON
            '{"intent": "create_recipe", "confidence": 0.9, "reasoning": "This is incom',

            # Cut off mid-key
            '{"intent": "create_recipe", "confid',

            # Cut off mid-value
            '{"intent": "create_recipe", "confidence": 0.',

            # Empty braces
            '{}',

            # Just opening brace
            '{',

            # Incomplete array
            '{"intent": "create_recipe", "entities": {"ingredients": ["chicken", "pasta"',

            # Incomplete nested object
            '{"intent": "create_recipe", "entities": {"ingredients": {"main":',
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                success = isinstance(result, dict) and 'intent' in result
                results.append({
                    'test_name': f'Incomplete Response {i+1}',
                    'success': success,
                    'error': None if success else 'Failed to handle incomplete response'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Incomplete Response {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_mixed_content(self):
        """Test responses with mixed content and prefixes"""
        test_cases = [
            # Common LLM prefixes
            'Here is the JSON object:\n{"intent": "create_recipe", "confidence": 0.9}',

            # Multiple prefixes
            'Sure! Here is the JSON:\n\n{"intent": "create_recipe", "confidence": 0.9}',

            # Explanation after JSON
            '{"intent": "create_recipe", "confidence": 0.9}\n\nThis analysis shows...',

            # Code block formatting
            '```json\n{"intent": "create_recipe", "confidence": 0.9}\n```',

            # HTML-like formatting
            '<json>{"intent": "create_recipe", "confidence": 0.9}</json>',

            # Multiple JSON objects
            '{"intent": "help"} and also {"intent": "create_recipe", "confidence": 0.9}',

            # Markdown formatting
            '## Analysis\n\n```\n{"intent": "create_recipe", "confidence": 0.9}\n```\n\n## Explanation',
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                success = isinstance(result, dict) and 'intent' in result
                results.append({
                    'test_name': f'Mixed Content {i+1}',
                    'success': success,
                    'error': None if success else 'Failed to extract JSON from mixed content'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Mixed Content {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_unicode_content(self):
        """Test Unicode and international character handling"""
        test_cases = [
            # Chinese characters
            '{"intent": "create_recipe", "reasoning": "中文食谱分析"}',

            # Arabic text
            '{"intent": "create_recipe", "reasoning": "تحليل وصفة الطعام"}',

            # Russian text
            '{"intent": "create_recipe", "reasoning": "Анализ рецепта"}',

            # Japanese text
            '{"intent": "create_recipe", "reasoning": "レシピ分析"}',

            # Mixed Unicode
            '{"intent": "create_recipe", "reasoning": "Recipe: 🍝 Pasta αλά Française"}',

            # Unicode escape sequences
            '{"intent": "create_recipe", "reasoning": "\\u4e2d\\u6587"}',
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                success = isinstance(result, dict) and 'intent' in result
                results.append({
                    'test_name': f'Unicode Content {i+1}',
                    'success': success,
                    'error': None if success else 'Failed to handle Unicode content'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Unicode Content {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_large_responses(self):
        """Test large and complex JSON responses"""

        # Generate large JSON
        large_ingredients = [f"ingredient_{i}" for i in range(100)]
        large_json = json.dumps({
            "intent": "create_recipe",
            "entities": {
                "ingredients": large_ingredients,
                "numbers": list(range(50)),
                "servings": 100
            },
            "confidence": 0.9,
            "reasoning": "This is a very long reasoning text. " * 100
        })

        # Deeply nested JSON
        nested_json = '{"intent": "create_recipe", "entities": {"level1": {"level2": {"level3": {"level4": {"ingredients": ["deep", "nested", "structure"]}}}}}}'

        test_cases = [large_json, nested_json]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                start_time = time.time()
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                parse_time = time.time() - start_time

                success = isinstance(result, dict) and 'intent' in result and parse_time < 1.0
                results.append({
                    'test_name': f'Large Response {i+1} ({parse_time:.3f}s)',
                    'success': success,
                    'error': None if success else f'Parsing too slow: {parse_time:.3f}s'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Large Response {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_nested_structures(self):
        """Test complex nested JSON structures"""
        test_cases = [
            # Deep nesting
            '{"intent": "create_recipe", "entities": {"ingredients": {"vegetables": {"root": ["carrot", "potato"]}}}}',

            # Arrays within objects
            '{"intent": "create_recipe", "entities": {"cooking_steps": [{"step": 1, "action": "chop"}, {"step": 2, "action": "cook"}]}}',

            # Mixed types
            '{"intent": "create_recipe", "entities": {"mixed": [1, "string", {"nested": true}, [1, 2, 3]]}}',
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                success = isinstance(result, dict) and 'intent' in result
                results.append({
                    'test_name': f'Nested Structure {i+1}',
                    'success': success,
                    'error': None if success else 'Failed to handle nested structure'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Nested Structure {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_empty_responses(self):
        """Test empty and null responses"""
        test_cases = [
            '',  # Empty string
            '   ',  # Whitespace only
            'null',  # Null value
            'undefined',  # Undefined
            'None',  # Python None as string
            '[]',  # Empty array
            'No JSON found',  # Plain text
        ]

        results = []
        for i, test_case in enumerate(test_cases):
            try:
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                success = isinstance(result, dict) and 'intent' in result
                results.append({
                    'test_name': f'Empty Response {i+1}',
                    'success': success,
                    'error': None if success else 'Failed to handle empty response'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Empty Response {i+1}',
                    'success': False,
                    'error': str(e)
                })

        return results

    def test_stress_performance(self):
        """Test performance under stress conditions"""

        # Generate stress test cases
        stress_cases = []

        # Many repeated parsing attempts
        for _ in range(10):
            stress_cases.append('{"intent": "create_recipe", "confidence": 0.9}')

        # Large malformed JSON
        large_malformed = '{"intent": "create_recipe", "reasoning": "' + 'A' * 10000 + '"confidence": 0.9'
        stress_cases.append(large_malformed)

        results = []
        start_time = time.time()

        for i, test_case in enumerate(stress_cases):
            try:
                case_start = time.time()
                result = self.llm_classifier._parse_json_with_fallback(test_case)
                case_time = time.time() - case_start

                success = isinstance(result, dict) and 'intent' in result and case_time < 0.1
                results.append({
                    'test_name': f'Stress Test {i+1}',
                    'success': success,
                    'error': None if success else f'Too slow: {case_time:.3f}s'
                })
            except Exception as e:
                results.append({
                    'test_name': f'Stress Test {i+1}',
                    'success': False,
                    'error': str(e)
                })

        total_time = time.time() - start_time
        print(f"   Total stress test time: {total_time:.3f}s")

        return results


if __name__ == "__main__":
    print("🚀 Starting Extreme Edge Case Testing...")

    suite = ExtremeTestSuite()
    results = suite.run_all_extreme_tests()

    print("\n💡 TESTING COMPLETE")
    print("This comprehensive test validates the robustness of the improved JSON parsing system.")