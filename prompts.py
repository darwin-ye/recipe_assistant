# prompts.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# A system prompt to set the context for the LLM
SYSTEM_PROMPT = """
You are a helpful culinary assistant. Your task is to generate a recipe based on the user's ingredients and dietary preferences.
Be creative and provide a complete recipe with a title, a list of ingredients, and step-by-step instructions.
Keep the recipe concise and easy to follow.
"""

# A prompt for generating the recipe
RECIPE_PROMPT_TEMPLATE = """
Based on the following information, generate a complete recipe.

Ingredients: {ingredients}
Dietary needs: {dietary_needs}

---
Recipe:
"""

# A prompt template for the ReAct agent
AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# A simple user-facing prompt to ask about ingredients
INGREDIENTS_PROMPT = "Hello! I can help you find a recipe. What ingredients do you have on hand? Please provide them as a comma-separated list."

# A prompt for dietary needs
DIETARY_NEEDS_PROMPT = "Great! Are there any dietary restrictions or preferences? (e.g., vegetarian, gluten-free, no nuts, etc.) If not, you can just say 'none'."

# A prompt to ask for another recipe
ANOTHER_RECIPE_PROMPT = (
    "Would you like another recipe based on different ingredients? (yes/no)"
)

NUTRITION_PROMPT_TEMPLATE = """
Based on the following recipe, provide a simplified nutritional breakdown.
Estimate the total calories, protein, fat, and carbohydrates.
Do not provide a full nutritional label. Keep the response concise.

Recipe:
{recipe}

---
Nutritional Breakdown:
"""