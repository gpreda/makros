"""PostgreSQL storage for makros."""

import json
import os
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from models import Item


class PostgresStorage:
    """PostgreSQL storage for items and meals."""

    def __init__(self, db_url: Optional[str] = None):
        """Initialize storage.

        Args:
            db_url: PostgreSQL connection URL. Defaults to DATABASE_URL env var
                    or postgresql://predator@localhost:5432/makros
        """
        self.db_url = db_url or os.environ.get(
            'DATABASE_URL',
            'postgresql://predator@localhost:5432/makros'
        )
        self._conn = None
        self._initialized = False

    @property
    def conn(self):
        """Lazy connection initialization."""
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.db_url)
            if not self._initialized:
                self._init_db()
                self._initialized = True
        return self._conn

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._conn.cursor() as cur:
            # Items table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id SERIAL PRIMARY KEY,
                    bar_code VARCHAR(255) UNIQUE,
                    name VARCHAR(255) NOT NULL UNIQUE,
                    description TEXT,
                    unit_conversions JSONB DEFAULT '{}',
                    default_unit VARCHAR(20) DEFAULT 'item',
                    calories REAL,
                    protein REAL,
                    carbs REAL,
                    fat REAL,
                    fiber REAL,
                    alcohol REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Migration: add nutrition columns if they don't exist
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'items' AND column_name = 'default_unit'
                    ) THEN
                        ALTER TABLE items ADD COLUMN default_unit VARCHAR(20) DEFAULT 'item';
                        ALTER TABLE items ADD COLUMN calories REAL;
                        ALTER TABLE items ADD COLUMN protein REAL;
                        ALTER TABLE items ADD COLUMN carbs REAL;
                        ALTER TABLE items ADD COLUMN fat REAL;
                        ALTER TABLE items ADD COLUMN fiber REAL;
                        ALTER TABLE items ADD COLUMN alcohol REAL;
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_name
                ON items(LOWER(name))
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_items_bar_code
                ON items(bar_code) WHERE bar_code IS NOT NULL
            """)

            # Meals table (logged meals)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meals (
                    id SERIAL PRIMARY KEY,
                    description TEXT NOT NULL,
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_calories INTEGER,
                    total_protein REAL,
                    total_carbs REAL,
                    total_fat REAL,
                    total_fiber REAL,
                    total_alcohol REAL DEFAULT 0
                )
            """)
            # Add alcohol column if it doesn't exist (migration)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'meals' AND column_name = 'total_alcohol'
                    ) THEN
                        ALTER TABLE meals ADD COLUMN total_alcohol REAL DEFAULT 0;
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meals_logged_at
                ON meals(logged_at)
            """)

            # Meal items (ingredients in a meal)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meal_items (
                    id SERIAL PRIMARY KEY,
                    meal_id INTEGER REFERENCES meals(id) ON DELETE CASCADE,
                    item_id INTEGER REFERENCES items(id) ON DELETE SET NULL,
                    name VARCHAR(255) NOT NULL,
                    amount REAL NOT NULL,
                    unit VARCHAR(20) NOT NULL,
                    calories INTEGER,
                    protein REAL,
                    carbs REAL,
                    fat REAL,
                    fiber REAL,
                    alcohol REAL DEFAULT 0
                )
            """)
            # Add alcohol column if it doesn't exist (migration)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'meal_items' AND column_name = 'alcohol'
                    ) THEN
                        ALTER TABLE meal_items ADD COLUMN alcohol REAL DEFAULT 0;
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meal_items_meal_id
                ON meal_items(meal_id)
            """)

        self._conn.commit()

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    # Item CRUD operations
    def add_item(self, item: Item) -> Item:
        """Add a new item. Returns item with assigned id."""
        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO items (bar_code, name, description, unit_conversions,
                                   default_unit, calories, protein, carbs, fat, fiber, alcohol)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (item.bar_code, item.name, item.description,
                  json.dumps(item.unit_conversions), item.default_unit,
                  item.calories, item.protein, item.carbs, item.fat,
                  item.fiber, item.alcohol))
            item.id = cur.fetchone()[0]
        self.conn.commit()
        return item

    def get_item_by_id(self, item_id: int) -> Optional[Item]:
        """Get item by id."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM items WHERE id = %s", (item_id,))
            row = cur.fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def get_item_by_name(self, name: str) -> Optional[Item]:
        """Get item by name (case-insensitive)."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM items WHERE LOWER(name) = LOWER(%s)",
                (name,)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def get_item_by_barcode(self, bar_code: str) -> Optional[Item]:
        """Get item by barcode."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM items WHERE bar_code = %s", (bar_code,))
            row = cur.fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def update_item(self, item: Item) -> bool:
        """Update an existing item. Returns True if updated."""
        if item.id is None:
            return False
        with self.conn.cursor() as cur:
            cur.execute("""
                UPDATE items
                SET bar_code = %s, name = %s, description = %s,
                    unit_conversions = %s, default_unit = %s,
                    calories = %s, protein = %s, carbs = %s,
                    fat = %s, fiber = %s, alcohol = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (item.bar_code, item.name, item.description,
                  json.dumps(item.unit_conversions), item.default_unit,
                  item.calories, item.protein, item.carbs, item.fat,
                  item.fiber, item.alcohol, item.id))
            updated = cur.rowcount > 0
        self.conn.commit()
        return updated

    def delete_item(self, item_id: int) -> bool:
        """Delete an item. Returns True if deleted."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM items WHERE id = %s", (item_id,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def get_item_by_name(self, name: str) -> Optional[Item]:
        """Get item by exact name (case-insensitive)."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM items WHERE LOWER(name) = LOWER(%s)",
                (name,)
            )
            row = cur.fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def search_items(self, query: str, limit: int = 20) -> list[Item]:
        """Search items by name (partial match)."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT * FROM items
                WHERE LOWER(name) LIKE LOWER(%s)
                ORDER BY name LIMIT %s
            """, (f'%{query}%', limit))
            return [self._row_to_item(row) for row in cur.fetchall()]

    def list_items(self, limit: int = 100, offset: int = 0) -> list[Item]:
        """List all items with pagination."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM items ORDER BY name LIMIT %s OFFSET %s",
                (limit, offset)
            )
            return [self._row_to_item(row) for row in cur.fetchall()]

    def count_items(self) -> int:
        """Get total number of items."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM items")
            return cur.fetchone()[0]

    def _row_to_item(self, row: dict) -> Item:
        """Convert database row to Item object."""
        return Item(
            id=row['id'],
            bar_code=row['bar_code'],
            name=row['name'],
            description=row['description'],
            unit_conversions=row['unit_conversions'] or {},
            default_unit=row.get('default_unit', 'item'),
            calories=row.get('calories'),
            protein=row.get('protein'),
            carbs=row.get('carbs'),
            fat=row.get('fat'),
            fiber=row.get('fiber'),
            alcohol=row.get('alcohol'),
        )

    # Meal operations
    def log_meal(self, description: str, items: list[dict], totals: dict) -> int:
        """Log a meal with its items. Returns meal id."""
        with self.conn.cursor() as cur:
            # Insert meal
            cur.execute("""
                INSERT INTO meals (description, total_calories, total_protein,
                                   total_carbs, total_fat, total_fiber, total_alcohol)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (description, totals.get('calories'), totals.get('protein'),
                  totals.get('carbs'), totals.get('fat'), totals.get('fiber'),
                  totals.get('alcohol', 0)))
            meal_id = cur.fetchone()[0]

            # Insert meal items
            for item in items:
                cur.execute("""
                    INSERT INTO meal_items (meal_id, name, amount, unit,
                                           calories, protein, carbs, fat, fiber, alcohol)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (meal_id, item.get('name'), item.get('amount'),
                      item.get('unit'), item.get('calories'),
                      item.get('protein'), item.get('carbs'),
                      item.get('fat'), item.get('fiber'), item.get('alcohol', 0)))

        self.conn.commit()
        return meal_id

    def get_meals(self, limit: int = 50, offset: int = 0,
                  date: Optional[datetime] = None) -> list[dict]:
        """Get logged meals with pagination, optionally filtered by date."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if date:
                cur.execute("""
                    SELECT * FROM meals
                    WHERE DATE(logged_at) = DATE(%s)
                    ORDER BY logged_at DESC
                    LIMIT %s OFFSET %s
                """, (date, limit, offset))
            else:
                cur.execute("""
                    SELECT * FROM meals
                    ORDER BY logged_at DESC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
            return [dict(row) for row in cur.fetchall()]

    def get_meal_with_items(self, meal_id: int) -> Optional[dict]:
        """Get a meal with all its items."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM meals WHERE id = %s", (meal_id,))
            meal = cur.fetchone()
            if not meal:
                return None

            cur.execute("""
                SELECT * FROM meal_items WHERE meal_id = %s ORDER BY id
            """, (meal_id,))
            items = [dict(row) for row in cur.fetchall()]

            return {**dict(meal), 'items': items}

    def get_daily_totals(self, date: Optional[datetime] = None) -> dict:
        """Get nutrition totals for a specific day."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COALESCE(SUM(total_calories), 0) as calories,
                    COALESCE(SUM(total_protein), 0) as protein,
                    COALESCE(SUM(total_carbs), 0) as carbs,
                    COALESCE(SUM(total_fat), 0) as fat,
                    COALESCE(SUM(total_fiber), 0) as fiber,
                    COALESCE(SUM(total_alcohol), 0) as alcohol,
                    COUNT(*) as meal_count
                FROM meals
                WHERE DATE(logged_at) = DATE(%s)
            """, (date,))
            return dict(cur.fetchone())

    def get_daily_breakdown(self, date: Optional[datetime] = None) -> list[dict]:
        """Get all meal items for a specific day."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT mi.name, mi.amount, mi.unit, mi.calories,
                       mi.protein, mi.carbs, mi.fat, mi.fiber, mi.alcohol
                FROM meal_items mi
                JOIN meals m ON mi.meal_id = m.id
                WHERE DATE(m.logged_at) = DATE(%s)
                ORDER BY m.logged_at, mi.id
            """, (date,))
            return [dict(row) for row in cur.fetchall()]

    def delete_meal(self, meal_id: int) -> bool:
        """Delete a meal. Returns True if deleted."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM meals WHERE id = %s", (meal_id,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted
