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


class ItemUpdate(BaseModel):
    name: Optional[str] = None
    bar_code: Optional[str] = None
    description: Optional[str] = None
    unit_conversions: Optional[dict[str, float]] = None


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
        return AnalyzeResponse(
            items=result.get('items', []),
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


@app.post("/api/items")
async def create_item(request: ItemCreate):
    """Create a new item."""
    item = Item(
        name=request.name,
        bar_code=request.bar_code,
        description=request.description,
        unit_conversions=request.unit_conversions
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
