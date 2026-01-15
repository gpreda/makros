"""FastAPI server for makros."""

import ast
import json
import os
import time
import uuid
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


# Session management for logging
_user_sessions: dict[str, str] = {}
DEFAULT_USER = "default"


def get_session_id(user_id: str = DEFAULT_USER) -> str:
    """Get or create a session ID for a user."""
    if user_id not in _user_sessions:
        _user_sessions[user_id] = str(uuid.uuid4())[:8]
    return _user_sessions[user_id]


def log_event(event: str, user_id: str = DEFAULT_USER,
              ai_used: bool = False, model_name: str = None,
              model_tokens: int = None, model_ms: int = None, **data) -> None:
    """Log an event to the database."""
    storage = get_storage()
    if storage:
        session_id = get_session_id(user_id)
        storage.log_event(event, user_id, session_id,
                          ai_used=ai_used, model_name=model_name,
                          model_tokens=model_tokens, model_ms=model_ms, **data)


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
                try:
                    storage.add_item(new_item)
                    # Update the item data with new name
                    item_data['name'] = unique_name
                except Exception as e:
                    # If add fails (e.g., race condition), just use original name
                    print(f"Warning: Failed to add item '{unique_name}': {e}")
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
            try:
                storage.add_item(new_item)
            except Exception as e:
                # If add fails (e.g., item was just added), continue anyway
                print(f"Warning: Failed to add item '{name}': {e}")
            processed_items.append(item_data)

    return processed_items


# Meal analysis endpoints
@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_meal(request: AnalyzeRequest):
    """Analyze a meal description and return nutritional breakdown."""
    client = get_genai_client()

    units_list = ', '.join(VALID_UNITS)
    prompt = f"""
Analyze this meal and provide a nutritional breakdown:

Meal: "{request.description}"

IMPORTANT - Item granularity rules:
- Keep well-defined products as SINGLE items. Do NOT break them into components:
  - Restaurant/fast food items: "McDonald's Big Mac", "Starbucks latte", "Subway sandwich"
  - Branded products: "Snickers bar", "Coca-Cola", "Kind bar"
  - Complete dishes: "Caesar salad", "pepperoni pizza slice", "chicken burrito"
- Only break down into ingredients for homemade/generic descriptions:
  - "eggs with toast and butter" -> 3 separate items
  - "chicken rice and vegetables" -> 3 separate items

For each item, provide:
- name: item name (string) - use the product name for branded items
- amount: numeric quantity (float) - use 1 for single items like burgers
- unit: one of [{units_list}] - use 'item' for countable products
- calories: kcal (int)
- protein: grams (float)
- carbs: grams (float)
- fat: grams (float)
- fiber: grams (float)
- alcohol: grams (float) - only for alcoholic beverages, 0 otherwise

Respond with ONLY a Python dictionary in this exact format:
{{
    'items': [
        {{'name': 'McDonald\\'s Double Cheeseburger', 'amount': 1, 'unit': 'item', 'calories': 450, 'protein': 25.0, 'carbs': 34.0, 'fat': 24.0, 'fiber': 2.0, 'alcohol': 0.0}},
        ...
    ],
    'totals': {{'calories': 456, 'protein': 25.0, 'carbs': 30.0, 'fat': 20.0, 'fiber': 3.0, 'alcohol': 0.0}}
}}

Be accurate with portion sizes. Use standard nutritional databases as reference.
Return ONLY the dictionary, no other text or markdown.
"""

    model_name = 'gemini-2.0-flash'
    start_time = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )
    model_ms = int((time.time() - start_time) * 1000)

    text = response.text.strip()
    text = text.replace('```python', '').replace('```', '').strip()

    # Get token usage if available
    model_tokens = None
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        model_tokens = getattr(response.usage_metadata, 'total_token_count', None)

    try:
        result = ast.literal_eval(text)
        items = result.get('items', [])
        totals = result.get('totals', {})
        print(f"[DEBUG] Analyzed items: {items}")
        print(f"[DEBUG] Analyzed totals: {totals}")

        # Process items: check DB, save new items, handle name conflicts
        processed_items = process_analyzed_items(items)
        print(f"[DEBUG] Processed items: {processed_items}")

        # Log the analysis event with AI tracking
        log_event('meal.analyze',
                  ai_used=True,
                  model_name=model_name,
                  model_tokens=model_tokens,
                  model_ms=model_ms,
                  description=request.description,
                  item_count=len(processed_items),
                  calories=totals.get('calories', 0))

        return AnalyzeResponse(
            items=processed_items,
            totals=totals
        )
    except (SyntaxError, ValueError) as e:
        print(f"[ERROR] Failed to parse AI response: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {e}")


@app.post("/api/meals")
async def log_meal_endpoint(request: LogMealRequest):
    """Log a meal to the database."""
    print(f"[DEBUG] log_meal_endpoint called: description={request.description}, items={len(request.items)}, totals={request.totals}")
    try:
        storage = get_storage()
        today = datetime.now()
        today_str = today.strftime('%Y-%m-%d')

        print(f"[DEBUG] Server datetime.now() = {today}, date string = {today_str}")

        # Count meals before insert
        meals_before = storage.get_meals(limit=1000, date=today)
        count_before = len(meals_before)
        print(f"[DEBUG] Meals for today ({today_str}) BEFORE insert: {count_before}")

        meal_id = storage.log_meal(request.description, request.items, request.totals)
        print(f"[DEBUG] Meal logged successfully with id={meal_id}")

        # Count meals after insert
        meals_after = storage.get_meals(limit=1000, date=today)
        count_after = len(meals_after)
        print(f"[DEBUG] Meals for today AFTER insert: {count_after}")

        if count_after == count_before + 1:
            print(f"[DEBUG] COUNT VERIFIED: Meal count increased by 1")
        else:
            print(f"[ERROR] COUNT MISMATCH: Expected {count_before + 1}, got {count_after}")
            # Check what date the meal was logged with
            verification = storage.get_meal_with_items(meal_id)
            if verification:
                print(f"[DEBUG] Meal {meal_id} logged_at: {verification.get('logged_at')}")

        # Verify the meal was actually added to the database
        verification = storage.get_meal_with_items(meal_id)
        if verification:
            print(f"[DEBUG] VERIFIED: Meal {meal_id} exists in DB with {len(verification.get('items', []))} items, logged_at={verification.get('logged_at')}")
        else:
            print(f"[ERROR] VERIFICATION FAILED: Meal {meal_id} NOT FOUND in DB after insert!")

        # Log the meal logging event
        log_event('meal.log',
                  meal_id=meal_id,
                  description=request.description,
                  item_count=len(request.items),
                  calories=request.totals.get('calories', 0),
                  protein=request.totals.get('protein', 0),
                  carbs=request.totals.get('carbs', 0),
                  fat=request.totals.get('fat', 0))

        return {"id": meal_id, "message": "Meal logged successfully", "verified": verification is not None, "count_before": count_before, "count_after": count_after}
    except Exception as e:
        print(f"[ERROR] Error logging meal: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to log meal: {str(e)}")


@app.get("/api/meals")
async def get_meals(limit: int = 50, offset: int = 0, date: Optional[str] = None):
    """Get logged meals, optionally filtered by date."""
    dt = None
    if date:
        try:
            dt = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")

    print(f"[DEBUG] GET /api/meals called: date param={date}, parsed dt={dt}, server now={datetime.now()}")
    meals = get_storage().get_meals(limit, offset, dt)
    print(f"[DEBUG] GET /api/meals returning {len(meals)} meals")
    if meals:
        print(f"[DEBUG] First meal: id={meals[0].get('id')}, logged_at={meals[0].get('logged_at')}")
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
        log_event('meal.delete', meal_id=meal_id)
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

    # Log the copy event
    log_event('meal.copy',
              original_meal_id=meal_id,
              new_meal_id=new_meal_id,
              description=meal['description'],
              calories=totals.get('calories', 0))

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


# Weight endpoints
class WeightRequest(BaseModel):
    weight_lbs: float


@app.get("/api/weight")
async def get_weight(date: Optional[str] = None):
    """Get weight for a specific date."""
    dt = None
    if date:
        try:
            dt = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    weight = get_storage().get_weight(dt)
    return {"weight_lbs": weight}


@app.post("/api/weight")
async def set_weight(request: WeightRequest, date: Optional[str] = None):
    """Set weight for a specific date."""
    dt = None
    if date:
        try:
            dt = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    get_storage().set_weight(request.weight_lbs, dt)

    log_event('weight.set',
              weight_lbs=request.weight_lbs,
              date=date or datetime.now().strftime('%Y-%m-%d'))

    return {"message": "Weight saved", "weight_lbs": request.weight_lbs}


@app.delete("/api/weight")
async def delete_weight(date: Optional[str] = None):
    """Delete weight for a specific date."""
    dt = None
    if date:
        try:
            dt = datetime.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format")
    if get_storage().delete_weight(dt):
        log_event('weight.delete',
                  date=date or datetime.now().strftime('%Y-%m-%d'))
        return {"message": "Weight deleted"}
    raise HTTPException(status_code=404, detail="No weight found for this date")


@app.get("/api/weight/history")
async def get_weight_history(days: Optional[int] = None):
    """Get weight history. Pass days=30, 90, 365 or omit for all."""
    history = get_storage().get_weight_history(days)
    return {"history": history}


@app.get("/weight")
async def weight_page():
    """Serve weight chart page."""
    weight_path = WEB_DIR / "weight.html"
    if weight_path.exists():
        return FileResponse(weight_path)
    return {"message": "Weight page not found"}


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
        model_name = 'gemini-2.0-flash'
        start_time = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        model_ms = int((time.time() - start_time) * 1000)

        text = response.text.strip()
        text = text.replace('```python', '').replace('```', '').strip()
        result = ast.literal_eval(text)

        # Get token usage if available
        model_tokens = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            model_tokens = getattr(response.usage_metadata, 'total_token_count', None)

        per_100g = result.get('per_100g', {})
        per_item = result.get('per_item')
        confidence = result.get('confidence', 0)

        # Log the suggestion event with AI tracking
        log_event('item.suggest',
                  ai_used=True,
                  model_name=model_name,
                  model_tokens=model_tokens,
                  model_ms=model_ms,
                  name=request.name,
                  confidence=confidence,
                  recognized=result.get('recognized', False))

        return {
            'confidence': confidence,
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

        # Log the item creation event
        log_event('item.create',
                  item_id=item.id,
                  name=item.name,
                  default_unit=item.default_unit,
                  calories=item.calories)

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
        log_event('item.delete', item_id=item_id)
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
