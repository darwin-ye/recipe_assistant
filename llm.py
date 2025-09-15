# llm.py
import os
from langchain_ollama import ChatOllama
from prompts import SYSTEM_PROMPT, RECIPE_PROMPT_TEMPLATE
from dotenv import load_dotenv

# Load environment variables (useful if you expand the project)
load_dotenv()

# Initialize the Ollama model
# Ensure the 'llama3' model is pulled and Ollama is running in the background.
llm = ChatOllama(model="llama3")


def generate_recipe_with_llm(ingredients: str, dietary_needs: str) -> str:
    """
    Generates a recipe using the local Ollama LLM.
    """
    full_prompt = SYSTEM_PROMPT + RECIPE_PROMPT_TEMPLATE.format(
        ingredients=ingredients, dietary_needs=dietary_needs
    )

    # Invoke the LLM with the complete prompt
    response = llm.invoke(full_prompt)

    return response.content
