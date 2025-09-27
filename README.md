# 🍳 AI Recipe Assistant

A sophisticated AI-powered recipe assistant that combines natural language understanding, intelligent reasoning, and comprehensive recipe management. Built with LangGraph and featuring both conversational AI (ReAct) and traditional menu-driven interfaces.

## 🎯 Overview

This project offers **two distinct interaction modes**:

1. **🤖 ReAct Mode** (`main.py`) - Natural language conversation with AI reasoning
2. **📋 Menu Mode** (`main_menu_driven.py`) - Traditional step-by-step guided interface

Both modes share the same powerful backend featuring recipe generation, database management, analytics, and intelligent search capabilities.

---

## ✨ Key Features

### 🧠 Intelligent AI Capabilities
- **Natural Language Understanding**: Interprets various ways to express recipe requests
- **Smart Intent Detection**: Hybrid rule-based + LLM system for accurate command interpretation
- **ReAct Architecture**: Reasoning and Acting framework for complex multi-step queries
- **Context Awareness**: Maintains conversation context across interactions

### 🍽️ Recipe Management
- **AI Recipe Generation**: Creates detailed recipes from ingredients using local LLM
- **Smart Database**: Persistent JSON storage with semantic search capabilities
- **Recipe Scaling**: Automatically adjusts ingredients for different serving sizes
- **Recipe History**: Tracks and retrieves previously created recipes

### 📊 Analytics & Insights
- **Frequency Analysis**: Discover your most frequently created recipes
- **Ingredient Tracking**: Count recipes by specific ingredients
- **Usage Patterns**: Understand your cooking habits and preferences
- **Historical Analysis**: Browse and analyze your recipe creation timeline

### 🔧 Technical Features
- **Local-First**: Runs entirely on your machine using Ollama (privacy-focused)
- **Dual Interfaces**: Choose between conversational AI or guided menus
- **Robust Error Handling**: Graceful fallbacks and comprehensive error management
- **Extensive Testing**: Comprehensive test suite ensuring reliability

---

## 🚀 Quick Start

### Prerequisites

**Ollama** must be installed and running on your system.

1. **Install Ollama**: Visit [ollama.com](https://ollama.com/) and download for your OS
2. **Download the model**:
   ```bash
   ollama run llama3
   ```
   Keep Ollama running in the background.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/darwin-ye/recipe_assistant.git
   cd recipe_assistant
   ```

2. **Set up the environment**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   This creates a virtual environment and installs all dependencies.

### Usage

**Activate the environment** (required for both modes):
```bash
source langgraph_venv/bin/activate
```

**Choose your preferred interface:**

**🤖 ReAct Mode** (Natural Conversation):
```bash
python main.py
```

**📋 Menu Mode** (Guided Interface):
```bash
python main_menu_driven.py
```

---

## 🤖 ReAct Mode (Natural Language Interface)

Experience the future of recipe assistance with natural conversation.

### What You Can Say

**Recipe Creation**:
```
"Create a chicken pasta recipe"
"Give me a salmon recipe"
"Make something with beef and vegetables"
"New recipe for 4 people with pasta"
```

**Recipe Discovery**:
```
"Show me recent recipes"
"Find pasta recipes"
"What's my most frequent recipe?"
"Search for chicken dishes"
```

**Analytics & Insights**:
```
"Show me the recipe I always have"
"What's my go-to recipe?"
"How often do I use chicken?"
"Count recipes with beef"
```

**Recipe Management**:
```
"Scale it to 8 people"           # After viewing a recipe
"2"                              # Select recipe #2 from a list
"Give me the previous salmon recipe"
"Show me recipe details"
```

### Features
- **Context Preservation**: Remembers current recipe for immediate scaling/modifications
- **Intelligent Fallback**: Falls back to menu options when natural language processing fails
- **Multi-turn Conversations**: Maintains context across multiple interactions
- **Smart Error Recovery**: Provides helpful suggestions when queries are unclear

---

## 📋 Menu Mode (Guided Interface)

Perfect for users who prefer structured, step-by-step interactions.

### Main Menu Options

1. **🔍 Search Recipes** - Find recipes in your database by ingredients or name
2. **📅 Recent Recipes** - Browse your most recently created recipes
3. **🆕 Create New Recipe** - Generate a new recipe with AI assistance
4. **🔬 Get Nutrition Info** - Analyze nutritional content of recipes
5. **🔗 Find Similar Recipes** - Discover recipes similar to existing ones
6. **❓ Complex Questions** - Access ReAct reasoning for advanced queries
7. **📊 Recipe Analytics** - View usage statistics and frequency analysis

### Workflow
- Clear numbered menus for easy navigation
- Step-by-step guidance through each process
- Comprehensive feedback and confirmation prompts
- Seamless integration with the same AI backend as ReAct mode

---

## 🏗️ Architecture

### Core Components

**SimpleReActAgent** (`simple_react_agent.py`)
- Primary AI interface with hybrid intent detection
- Supports 8 intent types with 40+ natural language patterns
- Context-aware recipe operations
- Real-time analytics processing

**ReActAgent** (`react_agent.py`)
- Advanced reasoning system with 11 specialized tools
- Multi-step problem solving capabilities
- External API integrations for web recipes and nutrition
- Complex query handling (substitutions, timing, scaling)

**Recipe Database** (`recipe_models.py`)
- JSON-based persistent storage
- TF-IDF semantic search engine
- Recipe frequency tracking and analytics
- Structured data models with validation

**LangGraph Workflow** (`main.py` / `main_menu_driven.py`)
- State management for conversations
- Memory persistence across sessions
- Flexible node-based processing

### Intent Detection System

**Hybrid Approach**:
1. **Rule-Based Patterns** (90% of queries): Fast regex matching for common requests
2. **LLM Fallback** (10% of queries): Advanced semantic understanding for edge cases

**Supported Intents**:
- `analytics_frequent` - Most frequent recipe analysis
- `analytics_count` - Ingredient frequency counting
- `create_recipe` - New recipe generation
- `get_recent` - Recent recipe retrieval
- `get_details` - Recipe detail viewing
- `scale_recipe` - Recipe scaling operations
- `search` - Database recipe search
- `numbered_reference` - Direct recipe selection

---

## 📚 Advanced Features

### Recipe Analytics
```python
# Examples of analytics queries
"Show me my most frequent recipe"      # → Frequency analysis
"How often do I use chicken?"          # → Ingredient counting
"What's my cooking pattern?"           # → Usage statistics
```

### Recipe Scaling
```python
# Intelligent ingredient scaling
"Scale this recipe to 8 people"        # → Proportional adjustments
"Make it for 2 instead of 4"          # → Automatic calculations
```

### Smart Search
```python
# Semantic search capabilities
"Find spicy pasta dishes"              # → Searches descriptions
"Recipes with tomatoes and cheese"     # → Ingredient matching
"Italian-style recipes"                # → Cuisine-based search
```

---

## 🧪 Testing & Development

### Run Tests
```bash
# Core functionality tests
python test_simple_react.py           # Intent detection (33 tests)
python test_analytics.py              # Analytics features

# Feature demonstrations
python demo_analytics.py              # Analytics showcase
python demo_enhanced_patterns.py      # Pattern matching demo
```

### Key Test Coverage
- ✅ 33 intent detection patterns
- ✅ Recipe creation and storage
- ✅ Analytics functionality
- ✅ Context preservation
- ✅ Error handling and fallbacks

---

## 📁 Project Structure

### Core Files (Required)
```
main.py                    # ReAct conversational interface
main_menu_driven.py        # Menu-driven interface
simple_react_agent.py      # Primary AI agent
react_agent.py             # Advanced ReAct tools
recipe_models.py           # Database and data models
llm.py                     # LLM integration
recipes_db.json           # Recipe database
```

### Optional Files
```
test_*.py                  # Test suites (development/validation)
demo_*.py                  # Feature demonstrations
debug_*.py                # Debugging utilities
REACT_DOCUMENTATION.md    # Technical architecture docs
```

---

## 🔮 Technology Stack

- **Python 3.8+** - Core programming language
- **LangGraph** - State machine and workflow orchestration
- **LangChain** - LLM abstractions and integrations
- **Ollama** - Local LLM execution platform
- **Llama 3** - Large language model for recipe generation
- **TF-IDF** - Semantic search and recipe matching
- **JSON** - Lightweight database storage

---

## 🛣️ Roadmap

### Immediate Enhancements
- **Pure LLM Intent Classification**: Replace rule-based patterns with full LLM understanding
- **Enhanced Context Management**: Multi-turn recipe modification conversations
- **Improved Analytics**: Nutritional trends and dietary pattern analysis

### Future Features
- **Meal Planning**: Weekly meal planning with automated shopping lists
- **Nutritional Optimization**: AI-powered nutritional balance suggestions
- **Community Features**: Recipe sharing and rating system
- **Web Interface**: Browser-based GUI for enhanced usability
- **Mobile App**: iOS/Android companion application

### Advanced Integrations
- **External APIs**: Integration with grocery delivery services
- **Smart Kitchen**: IoT device integration for cooking assistance
- **Voice Interface**: Voice-activated recipe assistance
- **Image Recognition**: Recipe generation from food photos

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Run tests**: `python test_simple_react.py`
4. **Commit changes**: `git commit -m 'Add amazing feature'`
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open a Pull Request**

### Development Guidelines
- Maintain test coverage for new features
- Follow existing code style and patterns
- Update documentation for significant changes
- Test both ReAct and Menu interfaces

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Ollama** - For providing excellent local LLM capabilities
- **LangChain/LangGraph** - For powerful agent frameworks
- **Meta** - For the Llama 3 language model
- **Open Source Community** - For the tools and inspiration

---

## 📞 Support

- **Documentation**: [REACT_DOCUMENTATION.md](REACT_DOCUMENTATION.md)
- **Issues**: [GitHub Issues](https://github.com/darwin-ye/recipe_assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/darwin-ye/recipe_assistant/discussions)

---

*Happy cooking with AI! 🍳✨*