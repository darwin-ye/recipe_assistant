#!/bin/bash

# A simple script to set up the Python environment and install packages for the Recipe Assistant project.

echo "Creating Python virtual environment..."
python3 -m venv langgraph_venv

echo "Activating virtual environment..."
source langgraph_venv/bin/activate

echo "Installing required packages..."
pip install langgraph
pip install langchain
pip install langchain-community
pip install langchain-ollama
pip install python-dotenv

echo "Setup complete! You can now run the project by activating the environment with 'source langgraph_venv/bin/activate' and then running 'python main.py'."