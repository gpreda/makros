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
            # Add name column for short meal names (migration)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'meals' AND column_name = 'name'
                    ) THEN
                        ALTER TABLE meals ADD COLUMN name VARCHAR(100);
                    END IF;
                END $$;
            """)
            # Add image column for meal photos (migration)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'meals' AND column_name = 'image_data'
                    ) THEN
                        ALTER TABLE meals ADD COLUMN image_data BYTEA;
                    END IF;
                END $$;
            """)
            # Add extended nutrition columns to meals (migration)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'meals' AND column_name = 'total_saturated_fat'
                    ) THEN
                        ALTER TABLE meals ADD COLUMN total_saturated_fat REAL DEFAULT 0;
                        ALTER TABLE meals ADD COLUMN total_trans_fat REAL DEFAULT 0;
                        ALTER TABLE meals ADD COLUMN total_cholesterol REAL DEFAULT 0;
                        ALTER TABLE meals ADD COLUMN total_sodium REAL DEFAULT 0;
                        ALTER TABLE meals ADD COLUMN total_potassium REAL DEFAULT 0;
                        ALTER TABLE meals ADD COLUMN total_added_sugar REAL DEFAULT 0;
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
            # Add extended nutrition columns to meal_items (migration)
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'meal_items' AND column_name = 'saturated_fat'
                    ) THEN
                        ALTER TABLE meal_items ADD COLUMN saturated_fat REAL DEFAULT 0;
                        ALTER TABLE meal_items ADD COLUMN trans_fat REAL DEFAULT 0;
                        ALTER TABLE meal_items ADD COLUMN cholesterol REAL DEFAULT 0;
                        ALTER TABLE meal_items ADD COLUMN sodium REAL DEFAULT 0;
                        ALTER TABLE meal_items ADD COLUMN potassium REAL DEFAULT 0;
                        ALTER TABLE meal_items ADD COLUMN added_sugar REAL DEFAULT 0;
                    END IF;
                END $$;
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meal_items_meal_id
                ON meal_items(meal_id)
            """)

            # Daily weights table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS daily_weights (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL UNIQUE,
                    weight_lbs REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_daily_weights_date
                ON daily_weights(date)
            """)

        self._conn.commit()

    def close(self):
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()

    # Item CRUD operations
    def add_item(self, item: Item) -> Item:
        """Add a new item. Returns item with assigned id."""
        try:
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
        except Exception as e:
            self.conn.rollback()
            raise

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
    def log_meal(self, description: str, items: list[dict], totals: dict,
                 logged_at: Optional[datetime] = None, name: Optional[str] = None,
                 image_data: Optional[bytes] = None) -> int:
        """Log a meal with its items. Returns meal id."""
        try:
            with self.conn.cursor() as cur:
                # Insert meal with explicit timestamp if provided
                if logged_at:
                    cur.execute("""
                        INSERT INTO meals (name, description, total_calories, total_protein,
                                           total_carbs, total_fat, total_fiber, total_alcohol,
                                           total_saturated_fat, total_trans_fat, total_cholesterol,
                                           total_sodium, total_potassium, total_added_sugar,
                                           logged_at, image_data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (name, description, totals.get('calories'), totals.get('protein'),
                          totals.get('carbs'), totals.get('fat'), totals.get('fiber'),
                          totals.get('alcohol', 0), totals.get('saturated_fat', 0),
                          totals.get('trans_fat', 0), totals.get('cholesterol', 0),
                          totals.get('sodium', 0), totals.get('potassium', 0),
                          totals.get('added_sugar', 0), logged_at, image_data))
                else:
                    cur.execute("""
                        INSERT INTO meals (name, description, total_calories, total_protein,
                                           total_carbs, total_fat, total_fiber, total_alcohol,
                                           total_saturated_fat, total_trans_fat, total_cholesterol,
                                           total_sodium, total_potassium, total_added_sugar, image_data)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (name, description, totals.get('calories'), totals.get('protein'),
                          totals.get('carbs'), totals.get('fat'), totals.get('fiber'),
                          totals.get('alcohol', 0), totals.get('saturated_fat', 0),
                          totals.get('trans_fat', 0), totals.get('cholesterol', 0),
                          totals.get('sodium', 0), totals.get('potassium', 0),
                          totals.get('added_sugar', 0), image_data))
                meal_id = cur.fetchone()[0]

                # Insert meal items
                for item in items:
                    cur.execute("""
                        INSERT INTO meal_items (meal_id, name, amount, unit,
                                               calories, protein, carbs, fat, fiber, alcohol,
                                               saturated_fat, trans_fat, cholesterol,
                                               sodium, potassium, added_sugar)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (meal_id, item.get('name'), item.get('amount'),
                          item.get('unit'), item.get('calories'),
                          item.get('protein'), item.get('carbs'),
                          item.get('fat'), item.get('fiber'), item.get('alcohol', 0),
                          item.get('saturated_fat', 0), item.get('trans_fat', 0),
                          item.get('cholesterol', 0), item.get('sodium', 0),
                          item.get('potassium', 0), item.get('added_sugar', 0)))

            self.conn.commit()
            return meal_id
        except Exception as e:
            self.conn.rollback()
            raise

    def get_meals(self, limit: int = 50, offset: int = 0,
                  date: Optional[datetime] = None) -> list[dict]:
        """Get logged meals with pagination, optionally filtered by date."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if date:
                cur.execute("""
                    SELECT id, name, description, logged_at, total_calories, total_protein,
                           total_carbs, total_fat, total_fiber, total_alcohol,
                           (image_data IS NOT NULL) as has_image
                    FROM meals
                    WHERE DATE(logged_at) = DATE(%s)
                    ORDER BY logged_at ASC
                    LIMIT %s OFFSET %s
                """, (date, limit, offset))
            else:
                cur.execute("""
                    SELECT id, name, description, logged_at, total_calories, total_protein,
                           total_carbs, total_fat, total_fiber, total_alcohol,
                           (image_data IS NOT NULL) as has_image
                    FROM meals
                    ORDER BY logged_at ASC
                    LIMIT %s OFFSET %s
                """, (limit, offset))
            return [dict(row) for row in cur.fetchall()]

    def get_meal_image(self, meal_id: int) -> Optional[bytes]:
        """Get image data for a meal."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT image_data FROM meals WHERE id = %s", (meal_id,))
            row = cur.fetchone()
            if row and row[0]:
                return bytes(row[0])
        return None

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
                    COALESCE(SUM(total_saturated_fat), 0) as saturated_fat,
                    COALESCE(SUM(total_trans_fat), 0) as trans_fat,
                    COALESCE(SUM(total_cholesterol), 0) as cholesterol,
                    COALESCE(SUM(total_sodium), 0) as sodium,
                    COALESCE(SUM(total_potassium), 0) as potassium,
                    COALESCE(SUM(total_added_sugar), 0) as added_sugar,
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
                SELECT mi.id, mi.meal_id, mi.name, mi.amount, mi.unit, mi.calories,
                       mi.protein, mi.carbs, mi.fat, mi.fiber, mi.alcohol,
                       mi.saturated_fat, mi.trans_fat, mi.cholesterol,
                       mi.sodium, mi.potassium, mi.added_sugar
                FROM meal_items mi
                JOIN meals m ON mi.meal_id = m.id
                WHERE DATE(m.logged_at) = DATE(%s)
                ORDER BY m.logged_at, mi.id
            """, (date,))
            return [dict(row) for row in cur.fetchall()]

    def get_meal_item(self, item_id: int) -> Optional[dict]:
        """Get a single meal item by ID."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM meal_items WHERE id = %s", (item_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_meal_item(self, item_id: int, new_amount: float) -> Optional[dict]:
        """Update a meal item's amount and recalculate nutrition proportionally.
        Also updates the parent meal's totals. Returns updated item or None."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get current item
                cur.execute("SELECT * FROM meal_items WHERE id = %s", (item_id,))
                item = cur.fetchone()
                if not item:
                    return None

                old_amount = item['amount']
                if old_amount == 0:
                    return None

                ratio = new_amount / old_amount

                # Calculate new nutrition values
                new_calories = int((item['calories'] or 0) * ratio)
                new_protein = (item['protein'] or 0) * ratio
                new_carbs = (item['carbs'] or 0) * ratio
                new_fat = (item['fat'] or 0) * ratio
                new_fiber = (item['fiber'] or 0) * ratio
                new_alcohol = (item['alcohol'] or 0) * ratio
                new_saturated_fat = (item['saturated_fat'] or 0) * ratio
                new_trans_fat = (item['trans_fat'] or 0) * ratio
                new_cholesterol = (item['cholesterol'] or 0) * ratio
                new_sodium = (item['sodium'] or 0) * ratio
                new_potassium = (item['potassium'] or 0) * ratio
                new_added_sugar = (item['added_sugar'] or 0) * ratio

                # Update the meal item
                cur.execute("""
                    UPDATE meal_items
                    SET amount = %s, calories = %s, protein = %s, carbs = %s,
                        fat = %s, fiber = %s, alcohol = %s, saturated_fat = %s,
                        trans_fat = %s, cholesterol = %s, sodium = %s,
                        potassium = %s, added_sugar = %s
                    WHERE id = %s
                """, (new_amount, new_calories, new_protein, new_carbs,
                      new_fat, new_fiber, new_alcohol, new_saturated_fat,
                      new_trans_fat, new_cholesterol, new_sodium,
                      new_potassium, new_added_sugar, item_id))

                # Update parent meal totals by recalculating from all items
                meal_id = item['meal_id']
                cur.execute("""
                    UPDATE meals SET
                        total_calories = (SELECT COALESCE(SUM(calories), 0) FROM meal_items WHERE meal_id = %s),
                        total_protein = (SELECT COALESCE(SUM(protein), 0) FROM meal_items WHERE meal_id = %s),
                        total_carbs = (SELECT COALESCE(SUM(carbs), 0) FROM meal_items WHERE meal_id = %s),
                        total_fat = (SELECT COALESCE(SUM(fat), 0) FROM meal_items WHERE meal_id = %s),
                        total_fiber = (SELECT COALESCE(SUM(fiber), 0) FROM meal_items WHERE meal_id = %s),
                        total_alcohol = (SELECT COALESCE(SUM(alcohol), 0) FROM meal_items WHERE meal_id = %s),
                        total_saturated_fat = (SELECT COALESCE(SUM(saturated_fat), 0) FROM meal_items WHERE meal_id = %s),
                        total_trans_fat = (SELECT COALESCE(SUM(trans_fat), 0) FROM meal_items WHERE meal_id = %s),
                        total_cholesterol = (SELECT COALESCE(SUM(cholesterol), 0) FROM meal_items WHERE meal_id = %s),
                        total_sodium = (SELECT COALESCE(SUM(sodium), 0) FROM meal_items WHERE meal_id = %s),
                        total_potassium = (SELECT COALESCE(SUM(potassium), 0) FROM meal_items WHERE meal_id = %s),
                        total_added_sugar = (SELECT COALESCE(SUM(added_sugar), 0) FROM meal_items WHERE meal_id = %s)
                    WHERE id = %s
                """, (meal_id,) * 12 + (meal_id,))

            self.conn.commit()

            # Return updated item
            return {
                'id': item_id,
                'name': item['name'],
                'amount': new_amount,
                'unit': item['unit'],
                'calories': new_calories,
                'protein': new_protein,
                'carbs': new_carbs,
                'fat': new_fat,
                'fiber': new_fiber,
                'alcohol': new_alcohol,
                'saturated_fat': new_saturated_fat,
                'trans_fat': new_trans_fat,
                'cholesterol': new_cholesterol,
                'sodium': new_sodium,
                'potassium': new_potassium,
                'added_sugar': new_added_sugar
            }
        except Exception as e:
            self.conn.rollback()
            raise

    def delete_meal(self, meal_id: int) -> bool:
        """Delete a meal. Returns True if deleted."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM meals WHERE id = %s", (meal_id,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    # Weight operations
    def get_weight(self, date: Optional[datetime] = None) -> Optional[float]:
        """Get weight for a specific date. Returns weight in lbs or None."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT weight_lbs FROM daily_weights WHERE date = DATE(%s)",
                (date,)
            )
            row = cur.fetchone()
            return row[0] if row else None

    def set_weight(self, weight_lbs: float, date: Optional[datetime] = None) -> None:
        """Set weight for a specific date. Updates if exists, inserts if not."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO daily_weights (date, weight_lbs)
                VALUES (DATE(%s), %s)
                ON CONFLICT (date)
                DO UPDATE SET weight_lbs = %s, updated_at = CURRENT_TIMESTAMP
            """, (date, weight_lbs, weight_lbs))
        self.conn.commit()

    def delete_weight(self, date: Optional[datetime] = None) -> bool:
        """Delete weight for a specific date. Returns True if deleted."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM daily_weights WHERE date = DATE(%s)", (date,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def get_weight_history(self, days: Optional[int] = None) -> list[dict]:
        """Get weight history. If days is None, returns all history."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if days:
                cur.execute("""
                    SELECT date, weight_lbs
                    FROM daily_weights
                    WHERE date >= CURRENT_DATE - %s * INTERVAL '1 day'
                    ORDER BY date ASC
                """, (days,))
            else:
                cur.execute("""
                    SELECT date, weight_lbs
                    FROM daily_weights
                    ORDER BY date ASC
                """)
            return [{'date': str(row['date']), 'weight_lbs': row['weight_lbs']} for row in cur.fetchall()]

    def get_calories_history(self, days: Optional[int] = None) -> list[dict]:
        """Get daily calories history. If days is None, returns all history.
        Excludes today since the day is not complete."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if days:
                cur.execute("""
                    SELECT DATE(logged_at) as date, SUM(total_calories) as calories
                    FROM meals
                    WHERE DATE(logged_at) >= CURRENT_DATE - %s * INTERVAL '1 day'
                      AND DATE(logged_at) < CURRENT_DATE
                    GROUP BY DATE(logged_at)
                    ORDER BY date ASC
                """, (days,))
            else:
                cur.execute("""
                    SELECT DATE(logged_at) as date, SUM(total_calories) as calories
                    FROM meals
                    WHERE DATE(logged_at) < CURRENT_DATE
                    GROUP BY DATE(logged_at)
                    ORDER BY date ASC
                """)
            return [{'date': str(row['date']), 'calories': int(row['calories'])} for row in cur.fetchall()]

    def get_macros_history(self, days: Optional[int] = None) -> list[dict]:
        """Get daily macros history for all nutrients. If days is None, returns all history.
        Excludes today since the day is not complete."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            query = """
                SELECT DATE(logged_at) as date,
                    SUM(total_calories) as calories,
                    SUM(total_protein) as protein,
                    SUM(total_carbs) as carbs,
                    SUM(total_fat) as fat,
                    SUM(total_fiber) as fiber,
                    SUM(total_alcohol) as alcohol,
                    SUM(total_saturated_fat) as saturated_fat,
                    SUM(total_trans_fat) as trans_fat,
                    SUM(total_cholesterol) as cholesterol,
                    SUM(total_sodium) as sodium,
                    SUM(total_potassium) as potassium,
                    SUM(total_added_sugar) as added_sugar
                FROM meals
                WHERE DATE(logged_at) < CURRENT_DATE
            """
            if days:
                query = """
                    SELECT DATE(logged_at) as date,
                        SUM(total_calories) as calories,
                        SUM(total_protein) as protein,
                        SUM(total_carbs) as carbs,
                        SUM(total_fat) as fat,
                        SUM(total_fiber) as fiber,
                        SUM(total_alcohol) as alcohol,
                        SUM(total_saturated_fat) as saturated_fat,
                        SUM(total_trans_fat) as trans_fat,
                        SUM(total_cholesterol) as cholesterol,
                        SUM(total_sodium) as sodium,
                        SUM(total_potassium) as potassium,
                        SUM(total_added_sugar) as added_sugar
                    FROM meals
                    WHERE DATE(logged_at) >= CURRENT_DATE - %s * INTERVAL '1 day'
                      AND DATE(logged_at) < CURRENT_DATE
                    GROUP BY DATE(logged_at)
                    ORDER BY date ASC
                """
                cur.execute(query, (days,))
            else:
                query += " GROUP BY DATE(logged_at) ORDER BY date ASC"
                cur.execute(query)

            return [{
                'date': str(row['date']),
                'calories': int(row['calories'] or 0),
                'protein': float(row['protein'] or 0),
                'carbs': float(row['carbs'] or 0),
                'fat': float(row['fat'] or 0),
                'fiber': float(row['fiber'] or 0),
                'alcohol': float(row['alcohol'] or 0),
                'saturated_fat': float(row['saturated_fat'] or 0),
                'trans_fat': float(row['trans_fat'] or 0),
                'cholesterol': float(row['cholesterol'] or 0),
                'sodium': float(row['sodium'] or 0),
                'potassium': float(row['potassium'] or 0),
                'added_sugar': float(row['added_sugar'] or 0),
            } for row in cur.fetchall()]

    # Event logging (shared with tongue app)
    def log_event(self, event: str, user_id: str, session_id: str = None,
                  app_name: str = "makros", ms: int = None, ai_used: bool = False,
                  model_name: str = None, model_tokens: int = None,
                  model_ms: int = None, **data) -> None:
        """Log an event to the shared events table."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO events (app_name, event, user_id, session_id, ms,
                                        ai_used, model_name, model_tokens, model_ms, data)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (app_name, event, user_id, session_id, ms,
                      ai_used, model_name, model_tokens, model_ms,
                      json.dumps(data) if data else None))
            self.conn.commit()
        except Exception as e:
            print(f"Error logging event: {e}")
            self.conn.rollback()
