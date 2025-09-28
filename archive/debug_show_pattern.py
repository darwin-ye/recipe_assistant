#!/usr/bin/env python3
"""Debug show pattern matching"""

import re

def debug_show_pattern():
    text = "show me the last chicken recipe"

    print(f"Testing text: '{text}'")
    print("=" * 50)

    # Test the pattern
    show_pattern = r"show.*?(?:the\s+)?(previous|last|recent|earlier|past)\s+(\w+)\s+recipe"
    show_match = re.search(show_pattern, text.lower())

    if show_match:
        print(f"✅ Pattern matched!")
        print(f"Historical ref: {show_match.group(1)}")
        print(f"Ingredient: {show_match.group(2)}")
    else:
        print("❌ Pattern did not match")

    # Test simpler patterns
    patterns = [
        r"show.*last.*chicken.*recipe",
        r"show.*last\s+(\w+)\s+recipe",
        r"last\s+(\w+)\s+recipe",
    ]

    for pattern in patterns:
        match = re.search(pattern, text.lower())
        print(f"Pattern '{pattern}': {'✅' if match else '❌'}")
        if match and match.groups():
            print(f"  Groups: {match.groups()}")

if __name__ == "__main__":
    debug_show_pattern()