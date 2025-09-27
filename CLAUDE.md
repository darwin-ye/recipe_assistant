# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites and Setup

Before running the application, ensure Ollama is installed and running:
- Download from [ollama.com](https://ollama.com/)
- Pull the Llama 3 model: `ollama run llama3`

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
chmod +x setup.sh
./setup.sh
source langgraph_venv/bin/activate

# Install dependencies manually if needed
pip install -r requirements.txt
```

### Running the Application
```bash
# Activate environment and run
source langgraph_venv/bin/activate
python main.py
```

### Testing LLM Connection
```bash
python llm.py
```

## Architecture Overview

This is a conversational AI recipe assistant built with LangGraph that provides personalized recipe generation, semantic search, and nutritional analysis.

### Core Components

**State Machine (main.py)**: The application uses LangGraph's StateGraph with these main nodes:
- `conversation`: Main entry point and user interaction handler
- `search_recipes`: Semantic search through existing recipes
- `show_recent`: Display recently created recipes
- `show_similar`: Find recipes similar to current selection
- `new_recipe`: Create new recipes using LLM
- `nutrition`: Generate nutritional analysis
- `end`: Graceful conversation termination

**Recipe Models (recipe_models.py)**: Structured data models using Pydantic:
- `Recipe`: Main recipe structure with ingredients, instructions, metadata
- `Ingredient`: Structured ingredient with amount/unit/notes
- `NutritionInfo`: Nutritional breakdown per serving
- `RecipeDatabase`: JSON-based persistence with semantic search
- `SemanticRecipeSearch`: Embedding-based recipe similarity (falls back to keyword search if sentence-transformers unavailable)

**LLM Integration (llm.py)**: Handles Ollama ChatLLama integration:
- Recipe generation with structured prompts
- Nutritional analysis
- Connection health checks and fallback handling

**Prompts (prompts.py)**: Template system for consistent LLM interactions with specific formatting requirements for recipe parsing.

### Data Flow

1. User input is classified using LLM or keyword fallback to determine intent
2. Intent routes to appropriate node in the state graph
3. Recipe creation uses structured parsing to convert LLM text to `Recipe` objects
4. All recipes are persisted to `recipes_db.json` with semantic search indexing
5. State is maintained throughout conversation using LangGraph's MemorySaver checkpointer

### Key Features

- **Semantic Search**: Uses sentence-transformers for recipe similarity matching
- **Structured Parsing**: Converts LLM text to structured Recipe objects with title extraction, ingredient parsing, and instruction formatting
- **Persistent Storage**: JSON-based database that loads/saves automatically
- **Conversational Memory**: Maintains context across multiple turns
- **Nutritional Analysis**: On-demand LLM-generated nutrition information
- **Intent Classification**: Smart routing based on user input patterns

### Important Implementation Notes

- The system gracefully degrades when sentence-transformers is unavailable
- Recipe titles are extracted using multiple regex patterns to handle various LLM response formats
- The database auto-loads existing recipes on startup and saves after each addition
- All LLM calls include error handling with fallback responses
- The intent classification system uses both LLM and keyword-based fallbacks for reliability