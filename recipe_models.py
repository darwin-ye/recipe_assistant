# recipe_models.py
"""
Structured recipe models and semantic search functionality
"""

from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from datetime import datetime
import json
import os
import re
import numpy as np
from pathlib import Path

# You'll need to install: pip install sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Semantic search will fallback to keyword matching.")
    print("Install with: pip install sentence-transformers scikit-learn")

# ========================================================================
# 1. STRUCTURED RECIPE FORMAT
# ========================================================================

class Ingredient(BaseModel):
    """Structured ingredient with amount and unit"""
    name: str
    amount: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None  # e.g., "diced", "room temperature"
    
    def __str__(self):
        parts = []
        if self.amount:
            parts.append(self.amount)
        if self.unit:
            parts.append(self.unit)
        parts.append(self.name)
        if self.notes:
            parts.append(f"({self.notes})")
        return " ".join(parts)

class NutritionInfo(BaseModel):
    """Structured nutrition information per serving"""
    calories: Optional[int] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    fiber_g: Optional[float] = None
    sugar_g: Optional[float] = None
    sodium_mg: Optional[float] = None
    vitamins: Optional[Dict[str, str]] = Field(default_factory=dict)
    minerals: Optional[Dict[str, str]] = Field(default_factory=dict)
    health_notes: Optional[List[str]] = Field(default_factory=list)

class Recipe(BaseModel):
    """Structured recipe format"""
    id: Optional[str] = None
    title: str
    description: Optional[str] = None
    main_ingredients: List[str]  # Key ingredients for search
    all_ingredients: List[Ingredient] = Field(default_factory=list)
    instructions: List[str]
    prep_time_minutes: Optional[int] = None
    cook_time_minutes: Optional[int] = None
    total_time_minutes: Optional[int] = None
    servings: int = 4
    difficulty: Optional[str] = "medium"  # easy, medium, hard
    cuisine_type: Optional[str] = None
    meal_type: Optional[List[str]] = Field(default_factory=list)  # breakfast, lunch, dinner, snack
    dietary_tags: Optional[List[str]] = Field(default_factory=list)  # vegetarian, gluten-free, etc.
    nutrition: Optional[NutritionInfo] = None
    tips: Optional[List[str]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    raw_text: Optional[str] = None  # Store original LLM response
    
    def to_display_string(self) -> str:
        """Convert recipe to a nicely formatted display string"""
        output = []
        output.append(f"**{self.title}**")
        
        if self.description:
            output.append(f"\n{self.description}")
        
        # Time info
        time_info = []
        if self.prep_time_minutes:
            time_info.append(f"Prep: {self.prep_time_minutes} min")
        if self.cook_time_minutes:
            time_info.append(f"Cook: {self.cook_time_minutes} min")
        if time_info:
            output.append(f"\n⏱️ {' | '.join(time_info)} | Servings: {self.servings}")
        
        # Ingredients
        output.append("\n**Ingredients:**")
        if self.all_ingredients:
            for ing in self.all_ingredients:
                output.append(f"• {str(ing)}")
        else:
            # Fallback if we only have main ingredients
            for ing in self.main_ingredients:
                output.append(f"• {ing}")
        
        # Instructions
        output.append("\n**Instructions:**")
        for i, instruction in enumerate(self.instructions, 1):
            output.append(f"{i}. {instruction}")
        
        # Tips
        if self.tips:
            output.append("\n**Tips:**")
            for tip in self.tips:
                output.append(f"• {tip}")
        
        # Dietary tags
        if self.dietary_tags:
            output.append(f"\n**Dietary Info:** {', '.join(self.dietary_tags)}")
        
        return "\n".join(output)
    
    def to_nutrition_string(self) -> str:
        """Convert nutrition info to display string"""
        if not self.nutrition:
            return "Nutritional information not available."
        
        n = self.nutrition
        output = ["**Nutritional Information (per serving):**\n"]
        
        if n.calories:
            output.append(f"• Calories: {n.calories}")
        if n.protein_g:
            output.append(f"• Protein: {n.protein_g}g")
        if n.carbs_g:
            output.append(f"• Carbohydrates: {n.carbs_g}g")
        if n.fat_g:
            output.append(f"• Fat: {n.fat_g}g")
        if n.fiber_g:
            output.append(f"• Fiber: {n.fiber_g}g")
        if n.sodium_mg:
            output.append(f"• Sodium: {n.sodium_mg}mg")
        
        if n.health_notes:
            output.append("\n**Health Benefits:**")
            for note in n.health_notes:
                output.append(f"• {note}")
        
        return "\n".join(output)

# ========================================================================
# 2. RECIPE PARSER - Convert LLM text to structured format (UPDATED)
# ========================================================================

class RecipeParser:
    """Parse unstructured LLM recipe text into structured Recipe objects"""
    
    @staticmethod
    def parse_recipe_from_llm(recipe_text: str, main_ingredients: str) -> Recipe:
        """
        Parse LLM-generated recipe text into structured Recipe object.
        Enhanced to better extract titles and handle various formats.
        """
        lines = recipe_text.strip().split('\n')
        
        # Initialize recipe components
        title = None
        description = None
        ingredients = []
        instructions = []
        tips = []
        current_section = None
        
        # Try to extract title from various patterns
        title_patterns = [
            r'Recipe:\s*"([^"]+)"',     # Recipe: "Title"
            r'Recipe:\s*([^\n]+)',       # Recipe: Title
            r'\*\*([^*]+)\*\*',         # **Title**
            r'^#\s+(.+)$',              # # Title
            r'Title:\s*([^\n]+)',       # Title: Something
            r'^([A-Z][^.!?]+)$',       # Line starting with capital, no punctuation
        ]
        
        # Parse line by line
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            
            # Try to extract title if we don't have one yet
            if not title and i < 5:  # Look for title in first 5 lines
                for pattern in title_patterns:
                    match = re.search(pattern, line)
                    if match:
                        potential_title = match.group(1).strip()
                        # Avoid section headers as titles
                        section_words = ['ingredient', 'instruction', 'step', 'tip', 'note', 
                                       'serving', 'recipe:', 'directions', 'method']
                        if not any(word in potential_title.lower() for word in section_words):
                            title = potential_title
                            # Clean up title
                            title = title.replace('Recipe', '').replace('recipe', '').strip()
                            if title:
                                break
            
            # Detect sections
            section_markers = {
                'ingredients': ['ingredient', 'you will need', 'you\'ll need'],
                'instructions': ['instruction', 'step', 'method', 'direction', 'procedure'],
                'tips': ['tip', 'note', 'variation', 'suggestion']
            }
            
            lower_line = line_stripped.lower()
            for section, markers in section_markers.items():
                if any(marker in lower_line for marker in markers):
                    if line_stripped.startswith('**') or line_stripped.startswith('#'):
                        current_section = section
                        continue
            
            # Parse content based on current section
            if current_section == 'ingredients':
                # Remove bullet points and parse
                clean_line = re.sub(r'^[\s•*\-–]+', '', line_stripped).strip()
                if clean_line and len(clean_line) > 2:  # Avoid single characters
                    ing = RecipeParser._parse_ingredient_line(clean_line)
                    if ing.name and ing.name != main_ingredients:  # Don't duplicate main ingredient
                        ingredients.append(ing)
                    
            elif current_section == 'instructions':
                # Remove numbering and bullet points
                clean_line = re.sub(r'^[\d.)\s•*\-–]+', '', line_stripped).strip()
                if clean_line and len(clean_line) > 5:  # Avoid too short instructions
                    instructions.append(clean_line)
                    
            elif current_section == 'tips':
                clean_line = re.sub(r'^[\s•*\-–]+', '', line_stripped).strip()
                if clean_line:
                    tips.append(clean_line)
        
        # If still no title, create one from ingredients
        if not title or title.lower() == "untitled recipe":
            main_ing_list = [ing.strip() for ing in main_ingredients.split(',')]
            if main_ing_list:
                # Create a better title from the main ingredient
                main_ing = main_ing_list[0].title()
                # Try to guess a dish type based on common patterns
                if 'chicken' in main_ing.lower():
                    title = f"Savory {main_ing} Dish"
                elif 'beef' in main_ing.lower():
                    title = f"Hearty {main_ing} Recipe"
                elif 'fish' in main_ing.lower() or 'salmon' in main_ing.lower():
                    title = f"Delicious {main_ing} Dish"
                elif 'pasta' in main_ing.lower():
                    title = f"{main_ing} Delight"
                else:
                    title = f"Homemade {main_ing} Recipe"
        
        # Clean up the title
        if title:
            title = title.strip().strip('"').strip("'")
            # Remove redundant words
            title = re.sub(r'\b(Recipe|recipe|Dish|dish)\s+(Recipe|recipe|Dish|dish)\b', 'Recipe', title)
        
        # Parse main ingredients into list
        main_ing_list = [ing.strip() for ing in main_ingredients.split(',')]
        
        # If no ingredients were parsed, use the main ingredients
        if not ingredients:
            for ing in main_ing_list:
                ingredients.append(Ingredient(name=ing))
        
        # If no instructions were found, add a default
        if not instructions:
            instructions = [f"Prepare {main_ingredients} according to your preference"]
        
        # Create Recipe object
        recipe = Recipe(
            title=title if title else f"{main_ingredients.title()} Recipe",
            description=description,
            main_ingredients=main_ing_list,
            all_ingredients=ingredients,
            instructions=instructions,
            tips=tips,
            servings=4,  # Default
            raw_text=recipe_text
        )
        
        # Generate ID from title and timestamp
        safe_title = re.sub(r'[^a-z0-9_]', '_', title.lower() if title else 'recipe')
        safe_title = re.sub(r'_+', '_', safe_title).strip('_')
        recipe.id = f"{safe_title}_{int(datetime.now().timestamp())}"
        
        return recipe
    
    @staticmethod
    def _parse_ingredient_line(line: str) -> Ingredient:
        """Parse a single ingredient line into structured format"""
        # This is a simple parser - you might want to use regex for better parsing
        parts = line.split()
        
        # Try to identify amount and unit (simple heuristic)
        amount = None
        unit = None
        name_parts = []
        notes = None
        
        # Check for parenthetical notes
        if '(' in line and ')' in line:
            start = line.index('(')
            end = line.index(')')
            notes = line[start+1:end]
            line = line[:start] + line[end+1:]
            parts = line.split()
        
        # Common units
        units = ['cup', 'cups', 'tbsp', 'tablespoon', 'tablespoons', 'tsp', 'teaspoon', 
                'teaspoons', 'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds',
                'g', 'gram', 'grams', 'kg', 'kilogram', 'ml', 'liter', 'l',
                'clove', 'cloves', 'slice', 'slices', 'piece', 'pieces']
        
        for i, part in enumerate(parts):
            # First 1-2 parts might be amount/unit
            if i == 0 and any(char.isdigit() for char in part):
                amount = part
            elif i <= 1 and part.lower() in units:
                unit = part
            else:
                name_parts.append(part)
        
        name = ' '.join(name_parts) if name_parts else line
        
        return Ingredient(
            name=name,
            amount=amount,
            unit=unit,
            notes=notes
        )
    
    @staticmethod
    def parse_nutrition_from_llm(nutrition_text: str) -> NutritionInfo:
        """Parse LLM-generated nutrition text into structured format"""
        nutrition = NutritionInfo()
        
        lines = nutrition_text.lower().split('\n')
        for line in lines:
            # Parse calories
            if 'calorie' in line:
                numbers = re.findall(r'\d+', line)
                if numbers:
                    nutrition.calories = int(numbers[0])
            
            # Parse macros
            if 'protein' in line:
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    nutrition.protein_g = float(numbers[0])
            
            if 'carb' in line:
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    nutrition.carbs_g = float(numbers[0])
            
            if 'fat' in line and 'trans' not in line:
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    nutrition.fat_g = float(numbers[0])
            
            # Health notes - look for benefit statements
            if 'benefit' in line or 'rich in' in line or 'good source' in line:
                clean_line = line.strip('•*- ').strip()
                if clean_line:
                    nutrition.health_notes.append(clean_line)
        
        return nutrition

# ========================================================================
# 3. SEMANTIC SEARCH ENGINE
# ========================================================================

class SemanticRecipeSearch:
    """Semantic search for recipes using embeddings"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the semantic search engine"""
        if EMBEDDINGS_AVAILABLE:
            print("Loading embedding model...")
            self.model = SentenceTransformer(model_name)
            self.embeddings_cache = {}
            print("✅ Semantic search ready")
        else:
            self.model = None
            print("⚠️ Semantic search not available - using keyword fallback")
    
    def _get_recipe_text(self, recipe: Recipe) -> str:
        """Get searchable text representation of a recipe"""
        # Combine relevant fields for embedding
        text_parts = [
            recipe.title,
            recipe.description or "",
            ' '.join(recipe.main_ingredients),
            ' '.join(recipe.dietary_tags) if recipe.dietary_tags else "",
            recipe.cuisine_type or ""
        ]
        return ' '.join(text_parts).strip()
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text, using cache when possible"""
        if not self.model:
            return None
            
        if text not in self.embeddings_cache:
            self.embeddings_cache[text] = self.model.encode([text])[0]
        return self.embeddings_cache[text]
    
    def search_recipes(
        self, 
        query: str, 
        recipes: Dict[str, Recipe], 
        top_k: int = 5,
        min_similarity: float = 0.3
    ) -> List[Tuple[str, Recipe, float]]:
        """
        Search recipes using semantic similarity.
        Returns list of (recipe_id, recipe, similarity_score) tuples.
        """
        if not recipes:
            return []
        
        # Fallback to keyword search if embeddings not available
        if not self.model:
            return self._keyword_search(query, recipes, top_k)
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Calculate similarities
        results = []
        for recipe_id, recipe in recipes.items():
            recipe_text = self._get_recipe_text(recipe)
            recipe_embedding = self._get_embedding(recipe_text)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                recipe_embedding.reshape(1, -1)
            )[0][0]
            
            if similarity >= min_similarity:
                results.append((recipe_id, recipe, similarity))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def _keyword_search(
        self, 
        query: str, 
        recipes: Dict[str, Recipe], 
        top_k: int
    ) -> List[Tuple[str, Recipe, float]]:
        """Fallback keyword-based search"""
        query_words = set(query.lower().split())
        results = []
        
        for recipe_id, recipe in recipes.items():
            # Calculate simple matching score
            recipe_text = self._get_recipe_text(recipe).lower()
            recipe_words = set(recipe_text.split())
            
            # Count matching words
            matches = len(query_words & recipe_words)
            if matches > 0:
                # Simple scoring: ratio of matched words
                score = matches / len(query_words)
                results.append((recipe_id, recipe, score))
        
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]
    
    def find_similar_recipes(
        self, 
        recipe: Recipe, 
        all_recipes: Dict[str, Recipe], 
        top_k: int = 3
    ) -> List[Tuple[str, Recipe, float]]:
        """Find recipes similar to a given recipe"""
        recipe_text = self._get_recipe_text(recipe)
        
        # Remove the query recipe from results
        other_recipes = {k: v for k, v in all_recipes.items() if k != recipe.id}
        
        return self.search_recipes(recipe_text, other_recipes, top_k)

# ========================================================================
# 4. RECIPE DATABASE WITH PERSISTENCE
# ========================================================================

class RecipeDatabase:
    """Persistent recipe storage with JSON backend"""
    
    def __init__(self, db_file: str = "recipes_db.json", embeddings_file: str = "recipe_embeddings.json"):
        self.db_file = db_file
        self.embeddings_file = embeddings_file
        self.recipes: Dict[str, Recipe] = {}
        self.search_engine = SemanticRecipeSearch()
        self.load_recipes()
    
    def load_recipes(self):
        """Load recipes from JSON file"""
        if os.path.exists(self.db_file):
            try:
                with open(self.db_file, 'r') as f:
                    data = json.load(f)
                    for recipe_id, recipe_dict in data.items():
                        # Convert dict back to Recipe object
                        if 'created_at' in recipe_dict:
                            recipe_dict['created_at'] = datetime.fromisoformat(recipe_dict['created_at'])
                        
                        # Handle nested objects
                        if 'nutrition' in recipe_dict and recipe_dict['nutrition']:
                            recipe_dict['nutrition'] = NutritionInfo(**recipe_dict['nutrition'])
                        
                        if 'all_ingredients' in recipe_dict:
                            recipe_dict['all_ingredients'] = [
                                Ingredient(**ing) if isinstance(ing, dict) else ing 
                                for ing in recipe_dict['all_ingredients']
                            ]
                        
                        self.recipes[recipe_id] = Recipe(**recipe_dict)
                
                print(f"✅ Loaded {len(self.recipes)} recipes from database")
            except Exception as e:
                print(f"Error loading recipes: {e}")
                self.recipes = {}
    
    def save_recipes(self):
        """Save all recipes to JSON file"""
        try:
            # Convert recipes to JSON-serializable format
            data = {}
            for recipe_id, recipe in self.recipes.items():
                recipe_dict = recipe.model_dump()
                # Convert datetime to ISO format
                if 'created_at' in recipe_dict:
                    recipe_dict['created_at'] = recipe_dict['created_at'].isoformat()
                data[recipe_id] = recipe_dict
            
            # Write to file
            with open(self.db_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"✅ Saved {len(self.recipes)} recipes to database")
        except Exception as e:
            print(f"Error saving recipes: {e}")
    
    def add_recipe(self, recipe: Recipe) -> str:
        """Add a new recipe to the database"""
        if not recipe.id:
            # Generate ID if not present
            safe_title = re.sub(r'[^a-z0-9_]', '_', recipe.title.lower())
            safe_title = re.sub(r'_+', '_', safe_title).strip('_')
            recipe.id = f"{safe_title}_{int(datetime.now().timestamp())}"
        
        self.recipes[recipe.id] = recipe
        self.save_recipes()
        return recipe.id
    
    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Get a recipe by ID"""
        return self.recipes.get(recipe_id)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, Recipe, float]]:
        """Search recipes using semantic search"""
        return self.search_engine.search_recipes(query, self.recipes, top_k)
    
    def get_recent_recipes(self, limit: int = 10) -> List[Recipe]:
        """Get most recently created recipes"""
        sorted_recipes = sorted(
            self.recipes.values(), 
            key=lambda r: r.created_at, 
            reverse=True
        )
        return sorted_recipes[:limit]
    
    def get_by_dietary_tag(self, tag: str) -> List[Recipe]:
        """Get all recipes with a specific dietary tag"""
        return [
            recipe for recipe in self.recipes.values()
            if recipe.dietary_tags and tag.lower() in [t.lower() for t in recipe.dietary_tags]
        ]
    
    def get_by_main_ingredient(self, ingredient: str) -> List[Recipe]:
        """Get all recipes containing a main ingredient"""
        ingredient_lower = ingredient.lower()
        return [
            recipe for recipe in self.recipes.values()
            if any(ingredient_lower in ing.lower() for ing in recipe.main_ingredients)
        ]

# ========================================================================
# 5. INTEGRATION HELPERS
# ========================================================================

def create_recipe_from_llm_response(
    llm_response: str, 
    ingredients: str, 
    dietary_needs: str = ""
) -> Recipe:
    """Helper function to create a structured recipe from LLM response"""
    recipe = RecipeParser.parse_recipe_from_llm(llm_response, ingredients)
    
    # Add dietary tags if provided
    if dietary_needs:
        dietary_tags = []
        dietary_lower = dietary_needs.lower()
        
        # Common dietary patterns
        if 'vegetarian' in dietary_lower:
            dietary_tags.append('vegetarian')
        if 'vegan' in dietary_lower:
            dietary_tags.append('vegan')
        if 'gluten' in dietary_lower:
            dietary_tags.append('gluten-free')
        if 'dairy' in dietary_lower:
            dietary_tags.append('dairy-free')
        if 'keto' in dietary_lower or 'low-carb' in dietary_lower:
            dietary_tags.append('low-carb')
        if 'paleo' in dietary_lower:
            dietary_tags.append('paleo')
        
        recipe.dietary_tags = dietary_tags
    
    return recipe

def add_nutrition_to_recipe(recipe: Recipe, nutrition_text: str) -> Recipe:
    """Add parsed nutrition information to an existing recipe"""
    nutrition = RecipeParser.parse_nutrition_from_llm(nutrition_text)
    recipe.nutrition = nutrition
    return recipe

# ========================================================================
# 6. USAGE EXAMPLE
# ========================================================================

if __name__ == "__main__":
    # Example usage
    print("Testing Recipe Models and Search...")
    
    # Initialize database
    db = RecipeDatabase()
    
    # Create a sample recipe
    sample_recipe = Recipe(
        title="Spicy Garlic Chicken Stir-Fry",
        description="A quick and flavorful Asian-inspired dish",
        main_ingredients=["chicken", "garlic", "bell peppers"],
        all_ingredients=[
            Ingredient(name="chicken breast", amount="500", unit="g", notes="cut into strips"),
            Ingredient(name="garlic", amount="4", unit="cloves", notes="minced"),
            Ingredient(name="bell peppers", amount="2", notes="mixed colors, sliced"),
        ],
        instructions=[
            "Heat oil in a wok over high heat",
            "Add chicken and cook until golden",
            "Add garlic and peppers, stir-fry for 3 minutes",
            "Season with soy sauce and serve"
        ],
        prep_time_minutes=15,
        cook_time_minutes=10,
        servings=4,
        dietary_tags=["gluten-free", "dairy-free"],
        cuisine_type="Asian"
    )
    
    # Add to database
    recipe_id = db.add_recipe(sample_recipe)
    print(f"Added recipe with ID: {recipe_id}")
    
    # Test search
    results = db.search("garlic asian chicken")
    print(f"\nSearch results for 'garlic asian chicken':")
    for recipe_id, recipe, score in results:
        print(f"  - {recipe.title} (score: {score:.2f})")
    
    # Display recipe
    print("\n" + sample_recipe.to_display_string())