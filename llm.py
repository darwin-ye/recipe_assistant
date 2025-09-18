# llm.py
import os
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from prompts import SYSTEM_PROMPT, RECIPE_PROMPT_TEMPLATE
from dotenv import load_dotenv

# Load environment variables (useful if you expand the project)
load_dotenv()

# Initialize the Ollama model
# Ensure the 'llama3' model is pulled and Ollama is running in the background.
llm = ChatOllama(model="llama3")

# Define a tool for getting nutritional information
@tool
def get_nutrition_info(recipe: str) -> str:
    """
    Looks up nutritional information for a given recipe.
    """
    # In a real-world scenario, this would call a real API
    # For this example, we'll use the LLM to get the info.
    full_prompt = "Provide a simplified nutritional breakdown for the following recipe, including estimated calories, protein, fat, and carbs:\n\n" + recipe
    response = llm.invoke(full_prompt)
    return response.content

def get_agent_llm():
    """
    Initializes the LLM and binds the tools to it.
    """
    tools = [get_nutrition_info]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools

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