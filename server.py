"""FastAPI server for makros."""

import ast
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai

from models import Item, VALID_UNITS
from postgres_storage import PostgresStorage


# Pydantic models for API
class AnalyzeRequest(BaseModel):
    description: str


class AnalyzeResponse(BaseModel):
    items: list[dict]
    totals: dict


class LogMealRequest(BaseModel):
    description: str
    items: list[dict]
    totals: dict


class ItemCreate(BaseModel):
    name: str
    bar_code: Optional[str] = None
    description: Optional[str] = None
    unit_conversions: dict[str, float] = {}
    default_unit: str = 'g'
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None
    fiber: Optional[float] = None
    alcohol: Optional[float] = None


class ItemUpdate(BaseModel):
    name: Optional[str] = None
    bar_code: Optional[str] = None
    description: Optional[str] = None
    unit_conversions: Optional[dict[str, float]] = None
    default_unit: Optional[str] = None
    calories: Optional[float] = None
    protein: Optional[float] = None
    carbs: Optional[float] = None
    fat: Optional[float] = None
    fiber: Optional[float] = None
    alcohol: Optional[float] = None


# Initialize app
app = FastAPI(title="Makros API", description="Macro nutrition tracking API")

# Global storage
_storage: PostgresStorage = None
_genai_client: genai.Client = None

# Paths
PROJECT_ROOT = Path(__file__).parent
WEB_DIR = PROJECT_ROOT / "web"


def get_storage() -> PostgresStorage:
    """Get or create storage instance."""
    global _storage
    if _storage is None:
        _storage = PostgresStorage()
    return _storage


def get_genai_client() -> genai.Client:
    """Get or create Gemini client."""
    global _genai_client
    if _genai_client is not None:
        return _genai_client

    config_file = os.path.expanduser('~/.config/tongue/config.json')
    api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            api_key = config.get('gemini_api_key')

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    _genai_client = genai.Client(api_key=api_key)
    return _genai_client


@app.on_event("startup")
async def startup():
    """Initialize storage on startup."""
    storage = get_storage()
    print(f"Connected to database: {get_storage().db_url}")


@app.on_event("shutdown")
async def shutdown():
    """Close storage on shutdown."""
    if _storage:
        _get_storage().close()


# Mount static files
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
async def root():
    """Serve web app."""
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Makros API", "docs": "/docs"}


@app.get("/items")
async def items_page():
    """Serve items page."""
    items_path = WEB_DIR / "items.html"
    if items_path.exists():
        return FileResponse(items_path)
    return {"message": "Items page not found"}


# Helper functions for item matching
def normalize_macros(item_data: dict, amount: float) -> dict:
    """Normalize macros to per-unit values (divide by amount)."""
    if amount == 0:
        return {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0, 'fiber': 0, 'alcohol': 0}
    return {
        'calories': (item_data.get('calories') or 0) / amount,
        'protein': (item_data.get('protein') or 0) / amount,
        'carbs': (item_data.get('carbs') or 0) / amount,
        'fat': (item_data.get('fat') or 0) / amount,
        'fiber': (item_data.get('fiber') or 0) / amount,
        'alcohol': (item_data.get('alcohol') or 0) / amount,
    }


def macros_match(macros1: dict, macros2: dict, tolerance: float = 0.1) -> bool:
    """Check if two normalized macro profiles match within tolerance (10% by default)."""
    for key in ['calories', 'protein', 'carbs', 'fat']:
        v1 = macros1.get(key, 0)
        v2 = macros2.get(key, 0)
        # If both are near zero, consider them matching
        if abs(v1) < 0.1 and abs(v2) < 0.1:
            continue
        # Use relative tolerance for larger values
        max_val = max(abs(v1), abs(v2))
        if max_val > 0 and abs(v1 - v2) / max_val > tolerance:
            return False
    return True


def find_unique_name(storage, base_name: str) -> str:
    """Find a unique name by appending a number if needed."""
    # Check if base name with number pattern already exists
    name = base_name
    counter = 2
    while storage.get_item_by_name(name):
        name = f"{base_name} #{counter}"
        counter += 1
    return name


def process_analyzed_items(items: list[dict]) -> list[dict]:
    """Process analyzed items: check DB, save new items, handle duplicates."""
    storage = get_storage()
    processed_items = []

    for item_data in items:
        name = item_data.get('name', '').strip()
        amount = item_data.get('amount', 1)
        unit = item_data.get('unit', 'item')

        if not name:
            processed_items.append(item_data)
            continue

        # Normalize macros to per-unit values
        new_macros = normalize_macros(item_data, amount)

        # Check if item exists in DB
        existing_item = storage.get_item_by_name(name)

        if existing_item:
            # Compare macros (existing item stores per-unit values)
            existing_macros = {
                'calories': existing_item.calories or 0,
                'protein': existing_item.protein or 0,
                'carbs': existing_item.carbs or 0,
                'fat': existing_item.fat or 0,
                'fiber': existing_item.fiber or 0,
                'alcohol': existing_item.alcohol or 0,
            }

            if macros_match(new_macros, existing_macros):
                # Same item, use existing - keep the analyzed data as-is
                processed_items.append(item_data)
            else:
                # Different macros, create new item with unique name
                unique_name = find_unique_name(storage, name)
                new_item = Item(
                    name=unique_name,
                    default_unit=unit,
                    calories=new_macros['calories'],
                    protein=new_macros['protein'],
                    carbs=new_macros['carbs'],
                    fat=new_macros['fat'],
                    fiber=new_macros['fiber'],
                    alcohol=new_macros['alcohol'],
                )
                storage.add_item(new_item)
                # Update the item data with new name
                item_data['name'] = unique_name
                processed_items.append(item_data)
        else:
            # New item, add to DB
            new_item = Item(
                name=name,
                default_unit=unit,
                calories=new_macros['calories'],
                protein=new_macros['protein'],
                carbs=new_macros['carbs'],
                fat=new_macros['fat'],
                fiber=new_macros['fiber'],
                alcohol=new_macros['alcohol'],
            )
            storage.add_item(new_item)
            processed_items.append(item_data)

    return processed_items


# Meal analysis endpoints
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_meal(request: AnalyzeRequest):
    """Analyze a meal description and return nutritional breakdown."""
    client = get_genai_client()

    units_list = ', '.join(VALID_UNITS)
    prompt = f"""
Analyze this meal and provide a detailed nutritional breakdown:

Meal: "{request.description}"

For each ingredient/item in the meal, provide:
- name: ingredient name (string)
- amount: numeric quantity (float)
- unit: one of [{units_list}]
- calories: kcal (int)
- protein: grams (float)
- carbs: grams (float)
- fat: grams (float)
- fiber: grams (float)
- alcohol: grams (float) - only for alcoholic beverages, 0 otherwise

Respond with ONLY a Python dictionary in this exact format:
{{
    'items': [
        {{'name': 'ingredient name', 'amount': 100.0, 'unit': 'g', 'calories': 123, 'protein': 12.5, 'carbs': 10.0, 'fat': 8.0, 'fiber': 1.0, 'alcohol': 0.0}},
        ...
    ],
    'totals': {{'calories': 456, 'protein': 25.0, 'carbs': 30.0, 'fat': 20.0, 'fiber': 3.0, 'alcohol': 0.0}}
}}

Be accurate with portion sizes. Use standard nutritional databases as reference.
Return ONLY the dictionary, no other text or markdown.
"""

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )
    text = response.text.strip()
    text = text.replace('```python', '').replace('```', '').strip()

    try:
        result = ast.literal_eval(text)
        items = result.get('items', [])
        # Process items: check DB, save new items, handle name conflicts
        processed_items = process_analyzed_items(items)
        return AnalyzeResponse(
            items=processed_items,
            totals=result.get('totals', {})
        )
    except (SyntaxError, ValueError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {e}")


@app.post("/api/meals")
async def log_meal(request: LogMealRequest):
    """Log a meal to the database."""
    meal_id = get_storage().log_meal(request.description, request.items, request.totals)
    return {"id": meal_id, "message": "Meal logged successfully"}


@app.get("/api/meals")
async def get_meals(limit: int = 50, offset: int = 0, date: Optional[str] = None):
    """Get logged meals, optionally filtered by date."""
    dt = None
    if date:
        try:
            dt = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    meals = get_storage().get_meals(limit, offset, dt)
    return {"meals": meals}


@app.get("/api/meals/{meal_id}")
async def get_meal(meal_id: int):
    """Get a specific meal with items."""
    meal = get_storage().get_meal_with_items(meal_id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    return meal


@app.delete("/api/meals/{meal_id}")
async def delete_meal(meal_id: int):
    """Delete a meal."""
    if get_storage().delete_meal(meal_id):
        return {"message": "Meal deleted"}
    raise HTTPException(status_code=404, detail="Meal not found")


@app.post("/api/meals/{meal_id}/copy")
async def copy_meal(meal_id: int):
    """Copy a meal to the current day."""
    meal = get_storage().get_meal_with_items(meal_id)
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")

    # Prepare items and totals for logging
    items = [{
        'name': item['name'],
        'amount': item['amount'],
        'unit': item['unit'],
        'calories': item['calories'],
        'protein': item['protein'],
        'carbs': item['carbs'],
        'fat': item['fat'],
        'fiber': item.get('fiber', 0),
        'alcohol': item.get('alcohol', 0),
    } for item in meal.get('items', [])]

    totals = {
        'calories': meal['total_calories'],
        'protein': meal['total_protein'],
        'carbs': meal['total_carbs'],
        'fat': meal['total_fat'],
        'fiber': meal.get('total_fiber', 0),
        'alcohol': meal.get('total_alcohol', 0),
    }

    new_meal_id = get_storage().log_meal(meal['description'], items, totals)
    return {"id": new_meal_id, "message": "Meal copied successfully"}


@app.get("/api/daily")
async def get_daily_totals(date: Optional[str] = None):
    """Get daily nutrition totals."""
    dt = None
    if date:
        try:
            dt = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    return get_storage().get_daily_totals(dt)


@app.get("/api/daily/breakdown")
async def get_daily_breakdown(date: Optional[str] = None):
    """Get all meal items for a day."""
    dt = None
    if date:
        try:
            dt = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    items = get_storage().get_daily_breakdown(dt)
    return {"items": items}


# Item management endpoints
@app.get("/api/items")
async def list_items(limit: int = 100, offset: int = 0, search: Optional[str] = None):
    """List or search items."""
    if search:
        items = get_storage().search_items(search, limit)
    else:
        items = get_storage().list_items(limit, offset)
    return {"items": [i.to_dict() for i in items], "total": get_storage().count_items()}


class ItemSuggestRequest(BaseModel):
    name: str


@app.post("/api/items/suggest")
async def suggest_item_macros(request: ItemSuggestRequest):
    """Suggest macros for an item name using AI."""
    client = get_genai_client()

    prompt = f"""
Analyze this food item and provide nutritional information:

Item: "{request.name}"

You must determine if this is a recognizable food item with known nutritional values.
Provide nutrition BOTH per 100g AND per single item/piece (if applicable).

Respond with ONLY a Python dictionary in this exact format:
{{
    'confidence': 0.95,  # Float 0-1: How confident you are this is a real food with known nutrition
    'recognized': True,  # Boolean: Is this a recognizable food item?
    'recommended_unit': 'item',  # 'g' for bulk foods (rice, meat), 'item' for countable items (burger, apple, egg)
    'per_100g': {{
        'calories': 250,
        'protein': 10.0,
        'carbs': 30.0,
        'fat': 8.0,
        'fiber': 2.0,
        'alcohol': 0.0
    }},
    'per_item': {{  # Set to None if not applicable (e.g., rice, oil, ground meat)
        'calories': 550,
        'protein': 30.0,
        'carbs': 40.0,
        'fat': 25.0,
        'fiber': 2.0,
        'alcohol': 0.0,
        'weight_g': 220  # Typical weight of one item in grams
    }}
}}

Guidelines:
- recommended_unit should be 'item' for: burgers, sandwiches, eggs, fruits, cookies, slices, etc.
- recommended_unit should be 'g' for: rice, pasta, meat (raw/cooked), vegetables, liquids, etc.
- per_item should be None for bulk foods that don't have a standard piece size
- Confidence 0.9-1.0: Common foods, 0.7-0.9: Less common, 0.4-0.7: Vague, 0.0-0.4: Unknown

If confidence is below 0.5, set all nutritional values to 0.
Return ONLY the dictionary, no other text.
"""

    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        text = response.text.strip()
        text = text.replace('```python', '').replace('```', '').strip()
        result = ast.literal_eval(text)

        per_100g = result.get('per_100g', {})
        per_item = result.get('per_item')

        return {
            'confidence': result.get('confidence', 0),
            'recognized': result.get('recognized', False),
            'recommended_unit': result.get('recommended_unit', 'g'),
            'per_100g': {
                'calories': per_100g.get('calories', 0),
                'protein': per_100g.get('protein', 0),
                'carbs': per_100g.get('carbs', 0),
                'fat': per_100g.get('fat', 0),
                'fiber': per_100g.get('fiber', 0),
                'alcohol': per_100g.get('alcohol', 0),
            },
            'per_item': {
                'calories': per_item.get('calories', 0),
                'protein': per_item.get('protein', 0),
                'carbs': per_item.get('carbs', 0),
                'fat': per_item.get('fat', 0),
                'fiber': per_item.get('fiber', 0),
                'alcohol': per_item.get('alcohol', 0),
                'weight_g': per_item.get('weight_g', 0),
            } if per_item else None,
        }
    except Exception as e:
        # Return empty suggestion on error
        return {
            'confidence': 0,
            'recognized': False,
            'recommended_unit': 'g',
            'per_100g': {
                'calories': 0,
                'protein': 0,
                'carbs': 0,
                'fat': 0,
                'fiber': 0,
                'alcohol': 0,
            },
            'per_item': None,
        }


@app.post("/api/items")
async def create_item(request: ItemCreate):
    """Create a new item."""
    item = Item(
        name=request.name,
        bar_code=request.bar_code,
        description=request.description,
        unit_conversions=request.unit_conversions,
        default_unit=request.default_unit,
        calories=request.calories,
        protein=request.protein,
        carbs=request.carbs,
        fat=request.fat,
        fiber=request.fiber,
        alcohol=request.alcohol,
    )
    try:
        item = get_storage().add_item(item)
        return item.to_dict()
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(status_code=409, detail="Item with this name or barcode already exists")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/items/{item_id}")
async def get_item(item_id: int):
    """Get an item by ID."""
    item = get_storage().get_item_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item.to_dict()


@app.put("/api/items/{item_id}")
async def update_item(item_id: int, request: ItemUpdate):
    """Update an item."""
    item = get_storage().get_item_by_id(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if request.name is not None:
        item.name = request.name
    if request.bar_code is not None:
        item.bar_code = request.bar_code
    if request.description is not None:
        item.description = request.description
    if request.unit_conversions is not None:
        item.unit_conversions = request.unit_conversions
    if request.default_unit is not None:
        item.default_unit = request.default_unit
    if request.calories is not None:
        item.calories = request.calories
    if request.protein is not None:
        item.protein = request.protein
    if request.carbs is not None:
        item.carbs = request.carbs
    if request.fat is not None:
        item.fat = request.fat
    if request.fiber is not None:
        item.fiber = request.fiber
    if request.alcohol is not None:
        item.alcohol = request.alcohol

    get_storage().update_item(item)
    return item.to_dict()


@app.delete("/api/items/{item_id}")
async def delete_item(item_id: int):
    """Delete an item."""
    if get_storage().delete_item(item_id):
        return {"message": "Item deleted"}
    raise HTTPException(status_code=404, detail="Item not found")


@app.get("/api/items/barcode/{barcode}")
async def get_item_by_barcode(barcode: str):
    """Get an item by barcode."""
    item = get_storage().get_item_by_barcode(barcode)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    return item.to_dict()


def create_app():
    """Factory function for creating the app."""
    return app
