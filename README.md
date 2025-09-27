# 🍜 AI Recipe Assistant

An intelligent, multi-turn conversational agent built with LangGraph that generates personalized recipes using advanced ReAct (Reasoning and Acting) architecture. The system combines natural language understanding, recipe database management, and analytics to provide a comprehensive culinary assistant experience.

<br>

## ✨ Features

### 🎯 Core Capabilities
- **AI Recipe Generation:** Creates personalized recipes from ingredients using local LLM
- **Smart Recipe Search:** Semantic search through your recipe database
- **Recipe Management:** Store, retrieve, and organize your recipe collection
- **Recipe Scaling:** Automatically adjust serving sizes with intelligent ingredient scaling
- **Dietary Customization:** Accommodates specific dietary requirements (gluten-free, vegetarian, keto, etc.)

### 🧠 Advanced Intelligence
- **Natural Language Understanding:** Understands various ways to express recipe requests
- **ReAct Architecture:** Combines reasoning with tool usage for complex queries
- **Intent Detection:** Hybrid rule-based + LLM system for accurate intent classification
- **Context Awareness:** Maintains recipe context across conversation turns

### 📊 Analytics & Insights
- **Recipe Analytics:** Discover your most frequently created recipes
- **Ingredient Analysis:** Track ingredient usage patterns and frequency
- **Historical Tracking:** Browse and analyze your recipe creation history
- **Usage Statistics:** Understand your cooking patterns and preferences

### 🔧 Technical Features
- **Local-First Architecture:** Runs entirely on your machine using Ollama for privacy
- **Stateful Conversations:** Remembers context across multiple interactions
- **Robust Error Handling:** Graceful handling of edge cases and user input variations
- **Comprehensive Testing:** Extensive test suite ensuring reliability

<br>

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

You need to have **Ollama** installed and running on your system.

1.  **Download Ollama:** Visit [ollama.com](https://ollama.com/) and download the application for your operating system (macOS, Windows, or Linux).
2.  **Run the LLM:** Once Ollama is installed, open your terminal and pull the Llama 3 model by running:
    ```bash
    ollama run llama3
    ```
    This will download and run the model. Keep Ollama running in the background.

### Installation

1.  Clone this repository to your local machine:
    ```bash
    git clone [https://github.com/darwin-ye/recipe_assistant.git](https://github.com/darwin-ye/recipe_assistant.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd recipe_assistant
    ```
3.  Run the setup script to create a Python virtual environment and install all dependencies:
    ```bash
    chmod +x setup.sh
    ./setup.sh
    ```
    This script will create a virtual environment named `langgraph_venv` and install `langgraph`, `langchain`, `langchain-ollama`, and other required packages.

### Usage

1.  Activate the virtual environment:
    ```bash
    source langgraph_venv/bin/activate
    ```
2.  Run the main application:
    ```bash
    python main.py
    ```
    The conversational agent will now be running in your terminal, guiding you through the recipe-finding process.

<br>

## 🧠 Workflow

The agent's logic is defined using a LangGraph state machine, which orchestrates a sequence of nodes to handle the conversation flow.

<img width="746" height="722" alt="Screenshot 2025-09-15 at 15 02 15" src="https://github.com/user-attachments/assets/2a8c6106-74da-4fc6-bec5-517c00fc69fe" />


<br>

## ⚙️ Technologies Used

- **Python:** The core programming language for the agent.
- **LangGraph:** The framework for building and orchestrating the agent's stateful workflow.
- **LangChain:** Provides the essential abstractions and integrations for connecting to the LLM.
- **Ollama:** A powerful tool for running open-source LLMs locally.
- **Llama 3:** The Large Language Model used as the agent's core reasoning engine.

<br>

## 🎮 Usage Examples

### Basic Recipe Creation
```
You: create a chicken pasta recipe
AI: [Generates complete recipe with ingredients and instructions]

You: scale it to 8 people
AI: [Automatically scales the current recipe]
```

### Natural Language Queries
```
You: show me the recipe I always have
AI: [Shows your most frequently created recipe]

You: how often do I use chicken?
AI: [Displays ingredient frequency analysis]

You: give me the previous salmon recipe
AI: [Retrieves specific historical recipe]
```

### Recipe Management
```
You: 2
AI: [Shows recipe #2 from recent list]

You: show recent recipes
AI: [Lists your most recent recipe creations]
```

## 🔧 Architecture

The system uses a sophisticated **ReAct (Reasoning and Acting)** architecture:

### SimpleReActAgent (Primary Interface)
- **Intent Detection:** Hybrid rule-based + LLM classification
- **Context Management:** Maintains recipe state across conversations
- **Analytics:** Real-time recipe frequency and ingredient analysis
- **Natural Language:** Understands various query formulations

### ReActAgent (Advanced Reasoning)
- **Tool Integration:** 11 specialized tools for complex operations
- **Multi-step Reasoning:** Handles complex queries requiring multiple steps
- **External APIs:** Web recipe search and nutritional analysis
- **Ingredient Substitutions:** Smart alternatives for dietary restrictions

### Database System
- **JSON Storage:** Persistent recipe storage with semantic search
- **Analytics Engine:** Real-time usage pattern analysis
- **Versioning:** Recipe history and modification tracking

## 📚 Documentation

- **[ReAct Architecture Documentation](REACT_DOCUMENTATION.md):** Comprehensive technical documentation
- **Testing:** Run `python test_simple_react.py` for intent detection tests
- **Demos:** Execute `python demo_analytics.py` for feature demonstrations

## 🗺️ Future Enhancements

### Immediate Roadmap
- **Pure LLM Intent Classification:** Replace hybrid system with full LLM understanding
- **Enhanced Context:** Multi-turn recipe modification conversations
- **Meal Planning:** Weekly meal planning with shopping list generation

### Advanced Features
- **Nutritional Optimization:** AI-powered nutritional balance suggestions
- **Seasonal Recommendations:** Recipe suggestions based on seasonal ingredients
- **Community Features:** Recipe sharing and rating system
- **GUI Interface:** Web-based or desktop application


