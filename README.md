# 🍜 AI Recipe Assistant

An intelligent, multi-turn conversational agent built with LangGraph that generates personalized recipes based on user-provided ingredients and dietary needs. The agent leverages a local Large Language Model (LLM) to provide detailed recipes and on-demand nutritional information.

<br>

## ✨ Features

- **Personalized Recipes:** Generates recipes tailored to the ingredients you have on hand.
- **Dietary Customization:** Accommodates specific dietary requirements (e.g., gluten-free, vegetarian, keto).
- **Nutritional Estimates:** Provides an on-the-spot nutritional breakdown for any generated recipe using the LLM's own knowledge.
- **Stateful Conversation:** Remembers your inputs across multiple turns, creating a seamless conversational experience.
- **Local-First Architecture:** Runs entirely on your local machine using Ollama, ensuring privacy and control without relying on external APIs.

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

!(https://i.imgur.com/B9B5o3p.png)

<br>

## ⚙️ Technologies Used

- **Python:** The core programming language for the agent.
- **LangGraph:** The framework for building and orchestrating the agent's stateful workflow.
- **LangChain:** Provides the essential abstractions and integrations for connecting to the LLM.
- **Ollama:** A powerful tool for running open-source LLMs locally.
- **Llama 3:** The Large Language Model used as the agent's core reasoning engine.

<br>

## 🗺️ Future Enhancements

- **Tool Use:** Integrate a tool for searching the web for ingredients or specific recipes.
- **Shopping List:** Add a feature to generate a shopping list from the final recipe.
- **GUI:** Build a simple web-based or desktop graphical user interface.
- **Multi-turn Refinement:** Allow the user to ask for changes to the recipe after it's generated (e.g., "make it spicier," "use less sugar").


