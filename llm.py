# llm.py - Improved Version
import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from prompts import SYSTEM_PROMPT, RECIPE_PROMPT_TEMPLATE, NUTRITION_SYSTEM_PROMPT
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Initialize the Ollama model with better configuration
try:
    llm = ChatOllama(
        model="llama3",
        temperature=0.7,  # Add some creativity but keep it controlled
        timeout=60,       # 60 second timeout
        num_predict=1024  # Limit response length
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize LLM: {e}")
    raise

def generate_recipe_with_llm(ingredients: str, dietary_needs: str = "") -> str:
    """
    Generates a recipe using the local Ollama LLM with better error handling.
    """
    try:
        # Build the complete prompt
        full_prompt = SYSTEM_PROMPT + "\n\n" + RECIPE_PROMPT_TEMPLATE.format(
            ingredients=ingredients, 
            dietary_needs=dietary_needs if dietary_needs else "No specific dietary restrictions"
        )

        print(f"🤖 Generating recipe for: {ingredients}")
        start_time = time.time()
        
        # Invoke the LLM with the complete prompt
        response = llm.invoke(full_prompt)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Recipe generated in {elapsed_time:.1f} seconds")

        return response.content

    except Exception as e:
        print(f"❌ Error generating recipe: {e}")
        # Return a fallback recipe
        return f"""
**Simple {ingredients.title()} Recipe**

**Ingredients:**
* {ingredients}
* Salt and pepper to taste
* Olive oil
* Your favorite seasonings

**Instructions:**
1. Season the {ingredients} with salt and pepper.
2. Heat olive oil in a pan over medium heat.
3. Cook the {ingredients} until done to your preference.
4. Season with your favorite spices and serve.

*Note: This is a fallback recipe due to a technical issue. Please try again for a more detailed recipe.*
"""

def generate_nutrition_info(recipe: str) -> str:
    """
    Generates nutritional information for a given recipe.
    """
    try:
        full_prompt = NUTRITION_SYSTEM_PROMPT + f"\n\nRecipe to analyze:\n{recipe}\n\nProvide nutritional analysis:"
        
        print("🧮 Analyzing nutritional content...")
        start_time = time.time()
        
        response = llm.invoke(full_prompt)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Nutrition analysis completed in {elapsed_time:.1f} seconds")
        
        return response.content
        
    except Exception as e:
        print(f"❌ Error generating nutrition info: {e}")
        return """
**Nutritional Information**

*Unable to generate detailed nutritional analysis at this time.*

**General Guidelines:**
- This recipe provides a balanced mix of nutrients
- Portion sizes can be adjusted based on your dietary needs
- Consider consulting a nutritionist for specific dietary requirements

*Please try the nutrition analysis again later.*
"""

def get_agent_llm():
    """
    Returns the configured LLM instance for other uses.
    """
    return llm

# Health check function
def test_llm_connection():
    """Test if the LLM is working properly"""
    try:
        test_response = llm.invoke("Say 'LLM is working' if you can respond.")
        return "working" in test_response.content.lower()
    except:
        return False

if __name__ == "__main__":
    # Quick test when run directly
    print("Testing LLM connection...")
    if test_llm_connection():
        print("✅ LLM is working properly")
    else:
        print("❌ LLM connection failed")