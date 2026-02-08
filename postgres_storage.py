"""PostgreSQL storage for makros."""

import json
import os
import re
from collections import defaultdict
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
        """Create tables if they don't exist and run migrations."""
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
                    saturated_fat REAL DEFAULT 0,
                    trans_fat REAL DEFAULT 0,
                    cholesterol REAL DEFAULT 0,
                    sodium REAL DEFAULT 0,
                    potassium REAL DEFAULT 0,
                    added_sugar REAL DEFAULT 0,
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
            # Add extended nutrition columns to items if missing
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'items' AND column_name = 'saturated_fat'
                    ) THEN
                        ALTER TABLE items ADD COLUMN saturated_fat REAL DEFAULT 0;
                        ALTER TABLE items ADD COLUMN trans_fat REAL DEFAULT 0;
                        ALTER TABLE items ADD COLUMN cholesterol REAL DEFAULT 0;
                        ALTER TABLE items ADD COLUMN sodium REAL DEFAULT 0;
                        ALTER TABLE items ADD COLUMN potassium REAL DEFAULT 0;
                        ALTER TABLE items ADD COLUMN added_sugar REAL DEFAULT 0;
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

            # Add obsolete column to items
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='items' AND column_name='obsolete'
                    ) THEN
                        ALTER TABLE items ADD COLUMN obsolete BOOLEAN DEFAULT FALSE;
                    END IF;
                END $$;
            """)

            # Add default_quantity column to items
            cur.execute("""
                ALTER TABLE items ADD COLUMN IF NOT EXISTS default_quantity REAL DEFAULT 0
            """)
            # Backfill default_quantity from most recent meal_items
            # Use quantity column (post-migration name); fall back to amount for pre-migration schemas
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'meal_items' AND column_name IN ('quantity', 'amount')
                ORDER BY column_name DESC LIMIT 1
            """)
            qty_col = cur.fetchone()
            if qty_col:
                qty_col = qty_col[0]
                cur.execute(f"""
                    UPDATE items SET default_quantity = sub.qty
                    FROM (
                        SELECT DISTINCT ON (mi.item_id) mi.item_id, mi.{qty_col} as qty
                        FROM meal_items mi
                        JOIN meals m ON mi.meal_id = m.id
                        ORDER BY mi.item_id, m.logged_at DESC
                    ) sub
                    WHERE items.id = sub.item_id AND items.default_quantity = 0
                """)

            # Meals table (logged meals)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meals (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(100),
                    description TEXT NOT NULL,
                    image_data BYTEA,
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    local_logged_at TIMESTAMP,
                    timezone VARCHAR(50)
                )
            """)
            # Migrations for meals table columns added over time
            for col, coldef in [
                ('total_alcohol', 'REAL DEFAULT 0'),
                ('name', 'VARCHAR(100)'),
                ('image_data', 'BYTEA'),
                ('total_saturated_fat', 'REAL DEFAULT 0'),
                ('local_logged_at', 'TIMESTAMP'),
                ('timezone', 'VARCHAR(50)'),
            ]:
                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'meals' AND column_name = '{col}'
                        ) THEN
                            ALTER TABLE meals ADD COLUMN {col} {coldef};
                        END IF;
                    END $$;
                """)
            # Backfill local_logged_at
            cur.execute("""
                UPDATE meals SET local_logged_at = logged_at
                WHERE local_logged_at IS NULL AND logged_at IS NOT NULL
            """)

            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meals_logged_at
                ON meals(logged_at)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meals_local_logged_at
                ON meals(local_logged_at)
            """)

            # Meal items table (create if not exists with old schema first)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meal_items (
                    id SERIAL PRIMARY KEY,
                    meal_id INTEGER REFERENCES meals(id) ON DELETE CASCADE,
                    item_id INTEGER REFERENCES items(id) ON DELETE SET NULL,
                    name VARCHAR(255),
                    amount REAL,
                    unit VARCHAR(20),
                    calories INTEGER,
                    protein REAL,
                    carbs REAL,
                    fat REAL,
                    fiber REAL,
                    alcohol REAL DEFAULT 0
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_meal_items_meal_id
                ON meal_items(meal_id)
            """)

            # Junction table (create if not exists - needed for migration)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS meal_meal_items (
                    meal_id INTEGER NOT NULL REFERENCES meals(id) ON DELETE CASCADE,
                    meal_item_id INTEGER NOT NULL REFERENCES meal_items(id) ON DELETE CASCADE,
                    PRIMARY KEY (meal_id, meal_item_id)
                )
            """)

            # === NORMALIZATION MIGRATION ===
            # Gate: check if meal_items still has the old 'name' column
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'meal_items' AND column_name = 'name'
                )
            """)
            needs_migration = cur.fetchone()[0]

            if needs_migration:
                print("Running schema normalization migration...")

                # Step 1: Ensure extended nutrition columns on meal_items exist
                # (needed to read data during migration)
                for col in ['saturated_fat', 'trans_fat', 'cholesterol', 'sodium', 'potassium', 'added_sugar']:
                    cur.execute(f"""
                        DO $$
                        BEGIN
                            IF NOT EXISTS (
                                SELECT 1 FROM information_schema.columns
                                WHERE table_name = 'meal_items' AND column_name = '{col}'
                            ) THEN
                                ALTER TABLE meal_items ADD COLUMN {col} REAL DEFAULT 0;
                            END IF;
                        END $$;
                    """)

                # Step 2: Backfill items from orphaned meal_items (item_id IS NULL)
                # Also ensure item_id column exists
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'meal_items' AND column_name = 'item_id'
                        ) THEN
                            ALTER TABLE meal_items ADD COLUMN item_id INTEGER REFERENCES items(id) ON DELETE SET NULL;
                        END IF;
                    END $$;
                """)

                # For each meal_item without an item_id, find or create matching item
                # and adjust quantity to preserve original calorie totals
                cur.execute("""
                    SELECT mi.id, mi.name, mi.unit, mi.amount,
                           mi.calories, mi.protein, mi.carbs, mi.fat, mi.fiber, mi.alcohol,
                           mi.saturated_fat, mi.trans_fat, mi.cholesterol,
                           mi.sodium, mi.potassium, mi.added_sugar
                    FROM meal_items mi
                    WHERE mi.item_id IS NULL AND mi.name IS NOT NULL
                          AND (mi.obsolete IS NULL OR mi.obsolete = false)
                """)
                orphaned_rows = cur.fetchall()

                # Cache created items to avoid repeated lookups
                item_cache = {}  # name_lower -> (item_id, item_calories)

                for row in orphaned_rows:
                    mi_id = row[0]
                    mi_name = row[1]
                    mi_unit = row[2]
                    mi_amount = row[3] or 1
                    mi_cal = row[4] or 0
                    mi_pro, mi_carb, mi_fat, mi_fiber, mi_alc = (row[5] or 0), (row[6] or 0), (row[7] or 0), (row[8] or 0), (row[9] or 0)
                    mi_satf, mi_transf, mi_chol, mi_sod, mi_pot, mi_addsug = (row[10] or 0), (row[11] or 0), (row[12] or 0), (row[13] or 0), (row[14] or 0), (row[15] or 0)

                    name_lower = mi_name.lower()

                    if name_lower in item_cache:
                        item_id, item_cal = item_cache[name_lower]
                    else:
                        # Check if item exists by name
                        cur.execute(
                            "SELECT id, calories FROM items WHERE LOWER(name) = LOWER(%s)",
                            (mi_name,))
                        existing = cur.fetchone()
                        if existing:
                            item_id = existing[0]
                            item_cal = existing[1] or 0
                        else:
                            # Normalize to per-1-unit and create new item
                            if mi_amount and mi_amount != 0:
                                per_cal = mi_cal / mi_amount
                                per_pro = mi_pro / mi_amount
                                per_carb = mi_carb / mi_amount
                                per_fat = mi_fat / mi_amount
                                per_fiber = mi_fiber / mi_amount
                                per_alc = mi_alc / mi_amount
                                per_satf = mi_satf / mi_amount
                                per_transf = mi_transf / mi_amount
                                per_chol = mi_chol / mi_amount
                                per_sod = mi_sod / mi_amount
                                per_pot = mi_pot / mi_amount
                                per_addsug = mi_addsug / mi_amount
                            else:
                                per_cal = per_pro = per_carb = per_fat = per_fiber = per_alc = 0
                                per_satf = per_transf = per_chol = per_sod = per_pot = per_addsug = 0

                            cur.execute("""
                                INSERT INTO items (name, default_unit, calories, protein, carbs, fat, fiber, alcohol,
                                                   saturated_fat, trans_fat, cholesterol, sodium, potassium, added_sugar)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                                RETURNING id
                            """, (mi_name, mi_unit or 'item', per_cal, per_pro, per_carb, per_fat, per_fiber, per_alc,
                                  per_satf, per_transf, per_chol, per_sod, per_pot, per_addsug))
                            item_id = cur.fetchone()[0]
                            item_cal = per_cal

                        item_cache[name_lower] = (item_id, item_cal)

                    # Compute correct quantity to preserve original calorie total
                    # If item has per-unit calories, derive quantity from original total
                    if item_cal and item_cal > 0 and mi_cal > 0:
                        new_quantity = mi_cal / item_cal
                    else:
                        new_quantity = mi_amount

                    cur.execute(
                        "UPDATE meal_items SET item_id = %s, amount = %s WHERE id = %s",
                        (item_id, new_quantity, mi_id))

                # Step 3: Sync meal_ids from junction table
                # The junction table is the authoritative source for meal→meal_item mapping.
                # meal_items can be shared across meals (from copy_meal), so we need to
                # duplicate rows for items that appear in multiple meals.
                # Also delete obsolete meal_items before syncing.
                cur.execute("""
                    DELETE FROM meal_items
                    WHERE obsolete = true OR (item_id IS NULL AND name IS NOT NULL
                          AND (SELECT COUNT(*) FROM meal_meal_items mmi WHERE mmi.meal_item_id = meal_items.id) = 0)
                """)

                cur.execute("""
                    SELECT mmi.meal_id, mmi.meal_item_id
                    FROM meal_meal_items mmi
                    JOIN meal_items mi ON mi.id = mmi.meal_item_id
                    WHERE mi.item_id IS NOT NULL
                    ORDER BY mmi.meal_item_id, mmi.meal_id
                """)
                junction_rows = cur.fetchall()

                # Group by meal_item_id to find shared items

                item_meals = defaultdict(list)
                for meal_id, mi_id in junction_rows:
                    item_meals[mi_id].append(meal_id)

                for mi_id, meal_ids in item_meals.items():
                    # First meal: set meal_id on existing row
                    cur.execute(
                        "UPDATE meal_items SET meal_id = %s WHERE id = %s",
                        (meal_ids[0], mi_id))

                    # Additional meals: duplicate the meal_item row
                    # Include name column since it hasn't been dropped yet at this step
                    for extra_meal_id in meal_ids[1:]:
                        cur.execute("""
                            INSERT INTO meal_items (meal_id, item_id, name, amount, unit,
                                calories, protein, carbs, fat, fiber, alcohol,
                                saturated_fat, trans_fat, cholesterol, sodium, potassium, added_sugar)
                            SELECT %s, item_id, name, amount, unit,
                                calories, protein, carbs, fat, fiber, alcohol,
                                saturated_fat, trans_fat, cholesterol, sodium, potassium, added_sugar
                            FROM meal_items WHERE id = %s
                        """, (extra_meal_id, mi_id))

                # Step 4: Restructure meal_items
                # Rename amount -> quantity
                cur.execute("""
                    DO $$
                    BEGIN
                        IF EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name = 'meal_items' AND column_name = 'amount'
                        ) THEN
                            ALTER TABLE meal_items RENAME COLUMN amount TO quantity;
                        END IF;
                    END $$;
                """)

                # Delete meal_items with no meal_id or no item_id (orphans that can't be migrated)
                cur.execute("DELETE FROM meal_items WHERE meal_id IS NULL OR item_id IS NULL")

                # Drop old FK constraints and make item_id NOT NULL, meal_id NOT NULL with CASCADE
                # First drop old constraints
                cur.execute("""
                    DO $$
                    DECLARE r RECORD;
                    BEGIN
                        FOR r IN (
                            SELECT constraint_name FROM information_schema.table_constraints
                            WHERE table_name = 'meal_items' AND constraint_type = 'FOREIGN KEY'
                        ) LOOP
                            EXECUTE 'ALTER TABLE meal_items DROP CONSTRAINT ' || r.constraint_name;
                        END LOOP;
                    END $$;
                """)

                # Set NOT NULL
                cur.execute("ALTER TABLE meal_items ALTER COLUMN item_id SET NOT NULL")
                cur.execute("ALTER TABLE meal_items ALTER COLUMN meal_id SET NOT NULL")

                # Add new FK constraints
                cur.execute("""
                    ALTER TABLE meal_items
                    ADD CONSTRAINT meal_items_meal_id_fkey
                    FOREIGN KEY (meal_id) REFERENCES meals(id) ON DELETE CASCADE
                """)
                cur.execute("""
                    ALTER TABLE meal_items
                    ADD CONSTRAINT meal_items_item_id_fkey
                    FOREIGN KEY (item_id) REFERENCES items(id) ON DELETE RESTRICT
                """)

                # Drop macro columns and name from meal_items
                for col in ['name', 'calories', 'protein', 'carbs', 'fat', 'fiber', 'alcohol',
                            'saturated_fat', 'trans_fat', 'cholesterol', 'sodium', 'potassium', 'added_sugar',
                            'obsolete']:
                    cur.execute(f"""
                        DO $$
                        BEGIN
                            IF EXISTS (
                                SELECT 1 FROM information_schema.columns
                                WHERE table_name = 'meal_items' AND column_name = '{col}'
                            ) THEN
                                ALTER TABLE meal_items DROP COLUMN {col};
                            END IF;
                        END $$;
                    """)

                # Step 5: Drop meal_meal_items
                cur.execute("DROP TABLE IF EXISTS meal_meal_items")

                # Step 6: Drop total_* columns from meals
                for col in ['total_calories', 'total_protein', 'total_carbs', 'total_fat',
                            'total_fiber', 'total_alcohol', 'total_saturated_fat', 'total_trans_fat',
                            'total_cholesterol', 'total_sodium', 'total_potassium', 'total_added_sugar']:
                    cur.execute(f"""
                        DO $$
                        BEGIN
                            IF EXISTS (
                                SELECT 1 FROM information_schema.columns
                                WHERE table_name = 'meals' AND column_name = '{col}'
                            ) THEN
                                ALTER TABLE meals DROP COLUMN {col};
                            END IF;
                        END $$;
                    """)

                # Add index on meal_items.item_id
                cur.execute("CREATE INDEX IF NOT EXISTS idx_meal_items_item_id ON meal_items(item_id)")

                print("Schema normalization migration complete.")

            # === POST-MIGRATION REPAIR ===
            # Fix orphaned meals (meals with no meal_items) that lost items
            # during a previous buggy migration run. Also drop obsolete column
            # from items if present.
            cur.execute("""
                SELECT m.id, m.name, m.description
                FROM meals m
                LEFT JOIN meal_items mi ON mi.meal_id = m.id
                WHERE mi.id IS NULL
            """)
            orphaned_meals = cur.fetchall()

            if orphaned_meals:

                print(f"Repairing {len(orphaned_meals)} orphaned meal(s)...")
                for meal_row in orphaned_meals:
                    meal_id, meal_name, meal_desc = meal_row[0], meal_row[1], meal_row[2]
                    # Try to find a matching item by name variants
                    search_names = []
                    for name in [meal_name, meal_desc]:
                        if name:
                            search_names.append(name)
                            # Strip trailing " #N" suffix
                            stripped = re.sub(r'\s*#\d+$', '', name)
                            if stripped != name:
                                search_names.append(stripped)

                    item_id = None
                    for search_name in search_names:
                        cur.execute(
                            "SELECT id FROM items WHERE LOWER(name) = LOWER(%s)",
                            (search_name,))
                        result = cur.fetchone()
                        if result:
                            item_id = result[0]
                            break

                    if item_id:
                        cur.execute(
                            "SELECT default_unit FROM items WHERE id = %s",
                            (item_id,))
                        default_unit = cur.fetchone()[0] or 'item'
                        cur.execute("""
                            INSERT INTO meal_items (meal_id, item_id, quantity, unit)
                            VALUES (%s, %s, 1, %s)
                        """, (meal_id, item_id, default_unit))
                        print(f"  Repaired meal {meal_id} ({meal_name}) -> item {item_id}")
                    else:
                        print(f"  WARNING: No matching item found for meal {meal_id} ({meal_name})")

                self._conn.commit()

            # Mark items with 0 meals as obsolete
            cur.execute("""
                UPDATE items SET obsolete = TRUE
                WHERE id NOT IN (
                    SELECT DISTINCT item_id FROM meal_items
                )
            """)
            # Ensure items that DO have meals are not obsolete
            cur.execute("""
                UPDATE items SET obsolete = FALSE
                WHERE id IN (
                    SELECT DISTINCT item_id FROM meal_items
                )
            """)
            self._conn.commit()

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

            # Exercise entries table for tracking burnt calories
            cur.execute("""
                CREATE TABLE IF NOT EXISTS exercise_entries (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    exercise_type VARCHAR(20) NOT NULL,
                    amount REAL,
                    unit VARCHAR(20),
                    calories_burnt INTEGER NOT NULL,
                    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_exercise_entries_date
                ON exercise_entries(date)
            """)

            # Caloric targets table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS caloric_targets (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL UNIQUE,
                    target_calories INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_caloric_targets_date
                ON caloric_targets(date)
            """)

            # Progress photos table (one per day)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS progress_photos (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL UNIQUE,
                    image_data BYTEA NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_progress_photos_date
                ON progress_photos(date)
            """)

            # Progress videos table (one video at a time)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS progress_videos (
                    id SERIAL PRIMARY KEY,
                    video_data BYTEA NOT NULL,
                    first_photo_date DATE,
                    last_photo_date DATE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
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
                                       default_unit, calories, protein, carbs, fat, fiber, alcohol,
                                       saturated_fat, trans_fat, cholesterol, sodium, potassium, added_sugar,
                                       default_quantity)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (item.bar_code, item.name, item.description,
                      json.dumps(item.unit_conversions), item.default_unit,
                      item.calories, item.protein, item.carbs, item.fat,
                      item.fiber, item.alcohol,
                      item.saturated_fat, item.trans_fat, item.cholesterol,
                      item.sodium, item.potassium, item.added_sugar,
                      item.default_quantity))
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
                    saturated_fat = %s, trans_fat = %s, cholesterol = %s,
                    sodium = %s, potassium = %s, added_sugar = %s,
                    default_quantity = %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = %s
            """, (item.bar_code, item.name, item.description,
                  json.dumps(item.unit_conversions), item.default_unit,
                  item.calories, item.protein, item.carbs, item.fat,
                  item.fiber, item.alcohol,
                  item.saturated_fat, item.trans_fat, item.cholesterol,
                  item.sodium, item.potassium, item.added_sugar,
                  item.default_quantity,
                  item.id))
            updated = cur.rowcount > 0
        self.conn.commit()
        return updated

    def update_item_default_quantity(self, item_id: int, quantity: float) -> None:
        """Update an item's default_quantity."""
        with self.conn.cursor() as cur:
            cur.execute(
                "UPDATE items SET default_quantity = %s WHERE id = %s",
                (quantity, item_id))
        self.conn.commit()

    def delete_item(self, item_id: int) -> bool:
        """Delete an item. Returns True if deleted."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM items WHERE id = %s", (item_id,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def _item_order_clause(self, sort_by: str = 'name', sort_dir: str = 'asc') -> str:
        """Build a safe ORDER BY clause for item queries."""
        col = 'name' if sort_by not in ('name', 'created_at') else sort_by
        direction = 'ASC' if sort_dir != 'desc' else 'DESC'
        return f"ORDER BY {col} {direction}"

    def search_items(self, query: str, limit: int = 20,
                     sort_by: str = 'name', sort_dir: str = 'asc') -> list[Item]:
        """Search items by name (partial match), excluding obsolete."""
        order = self._item_order_clause(sort_by, sort_dir)
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT * FROM items
                WHERE LOWER(name) LIKE LOWER(%s)
                  AND (obsolete IS NOT TRUE)
                {order} LIMIT %s
            """, (f'%{query}%', limit))
            return [self._row_to_item(row) for row in cur.fetchall()]

    def list_items(self, limit: int = 100, offset: int = 0,
                   sort_by: str = 'name', sort_dir: str = 'asc') -> list[Item]:
        """List all non-obsolete items with pagination."""
        order = self._item_order_clause(sort_by, sort_dir)
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"SELECT * FROM items WHERE obsolete IS NOT TRUE {order} LIMIT %s OFFSET %s",
                (limit, offset)
            )
            return [self._row_to_item(row) for row in cur.fetchall()]

    def count_items(self) -> int:
        """Get total number of non-obsolete items."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM items WHERE obsolete IS NOT TRUE")
            return cur.fetchone()[0]

    def get_item_meal_counts(self, item_names: list[str]) -> dict[str, int]:
        """Get the number of meals referencing each item name (via meal_items)."""
        if not item_names:
            return {}
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT LOWER(i.name) as lname, COUNT(DISTINCT mi.meal_id) as meal_count
                FROM meal_items mi
                JOIN items i ON mi.item_id = i.id
                WHERE LOWER(i.name) = ANY(%s)
                GROUP BY LOWER(i.name)
            """, ([n.lower() for n in item_names],))
            return {row['lname']: row['meal_count'] for row in cur.fetchall()}

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
            saturated_fat=row.get('saturated_fat'),
            trans_fat=row.get('trans_fat'),
            cholesterol=row.get('cholesterol'),
            sodium=row.get('sodium'),
            potassium=row.get('potassium'),
            added_sugar=row.get('added_sugar'),
            created_at=row['created_at'].isoformat() if row.get('created_at') else None,
            default_quantity=row.get('default_quantity', 0) or 0,
        )

    # Meal operations
    def log_meal(self, description: str, items: list[dict],
                 logged_at: Optional[datetime] = None, name: Optional[str] = None,
                 image_data: Optional[bytes] = None,
                 local_logged_at: Optional[datetime] = None,
                 timezone: Optional[str] = None) -> int:
        """Log a meal with its items. Items are [{item_id, unit, quantity}].
        Returns meal id."""
        try:
            with self.conn.cursor() as cur:
                if logged_at:
                    cur.execute("""
                        INSERT INTO meals (name, description, logged_at, image_data,
                                           local_logged_at, timezone)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """, (name, description, logged_at, image_data,
                          local_logged_at, timezone))
                else:
                    cur.execute("""
                        INSERT INTO meals (name, description, image_data,
                                           local_logged_at, timezone)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (name, description, image_data,
                          local_logged_at, timezone))
                meal_id = cur.fetchone()[0]

                for item in items:
                    cur.execute("""
                        INSERT INTO meal_items (meal_id, item_id, unit, quantity)
                        VALUES (%s, %s, %s, %s)
                    """, (meal_id, item['item_id'], item.get('unit', 'item'),
                          item.get('quantity', 1)))

            self.conn.commit()
            return meal_id
        except Exception as e:
            self.conn.rollback()
            raise

    # SQL fragment for computing nutrition from meal_items -> items
    _NUTRITION_SUM = """
        COALESCE(SUM(i.calories * mi.quantity), 0) as total_calories,
        COALESCE(SUM(i.protein * mi.quantity), 0) as total_protein,
        COALESCE(SUM(i.carbs * mi.quantity), 0) as total_carbs,
        COALESCE(SUM(i.fat * mi.quantity), 0) as total_fat,
        COALESCE(SUM(i.fiber * mi.quantity), 0) as total_fiber,
        COALESCE(SUM(i.alcohol * mi.quantity), 0) as total_alcohol,
        COALESCE(SUM(i.saturated_fat * mi.quantity), 0) as total_saturated_fat,
        COALESCE(SUM(i.trans_fat * mi.quantity), 0) as total_trans_fat,
        COALESCE(SUM(i.cholesterol * mi.quantity), 0) as total_cholesterol,
        COALESCE(SUM(i.sodium * mi.quantity), 0) as total_sodium,
        COALESCE(SUM(i.potassium * mi.quantity), 0) as total_potassium,
        COALESCE(SUM(i.added_sugar * mi.quantity), 0) as total_added_sugar
    """

    def get_meals(self, limit: int = 50, offset: int = 0,
                  date: Optional[datetime] = None) -> list[dict]:
        """Get logged meals with pagination, optionally filtered by date."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Subquery to compute totals per meal
            nutrition_sub = f"""
                LEFT JOIN LATERAL (
                    SELECT
                        {self._NUTRITION_SUM},
                        CASE WHEN COUNT(*) = 1 THEN MAX(i.name) ELSE NULL END as single_item_name
                    FROM meal_items mi
                    JOIN items i ON mi.item_id = i.id
                    WHERE mi.meal_id = m.id
                ) n ON TRUE
            """
            if date:
                cur.execute(f"""
                    SELECT m.id, m.name, m.description, m.logged_at,
                           COALESCE(m.local_logged_at, m.logged_at) as local_logged_at,
                           n.total_calories, n.total_protein, n.total_carbs,
                           n.total_fat, n.total_fiber, n.total_alcohol,
                           (m.image_data IS NOT NULL) as has_image,
                           n.single_item_name
                    FROM meals m
                    {nutrition_sub}
                    WHERE DATE(COALESCE(m.local_logged_at, m.logged_at)) = DATE(%s)
                    ORDER BY COALESCE(m.local_logged_at, m.logged_at) ASC
                    LIMIT %s OFFSET %s
                """, (date, limit, offset))
            else:
                cur.execute(f"""
                    SELECT m.id, m.name, m.description, m.logged_at,
                           COALESCE(m.local_logged_at, m.logged_at) as local_logged_at,
                           n.total_calories, n.total_protein, n.total_carbs,
                           n.total_fat, n.total_fiber, n.total_alcohol,
                           (m.image_data IS NOT NULL) as has_image,
                           n.single_item_name
                    FROM meals m
                    {nutrition_sub}
                    ORDER BY COALESCE(m.local_logged_at, m.logged_at) ASC
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
        """Get a meal with all its items (computed macros per item)."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM meals WHERE id = %s", (meal_id,))
            meal = cur.fetchone()
            if not meal:
                return None

            cur.execute("""
                SELECT mi.id, mi.meal_id, mi.item_id, mi.unit, mi.quantity,
                       i.name,
                       COALESCE(i.calories, 0) * mi.quantity as calories,
                       COALESCE(i.protein, 0) * mi.quantity as protein,
                       COALESCE(i.carbs, 0) * mi.quantity as carbs,
                       COALESCE(i.fat, 0) * mi.quantity as fat,
                       COALESCE(i.fiber, 0) * mi.quantity as fiber,
                       COALESCE(i.alcohol, 0) * mi.quantity as alcohol,
                       COALESCE(i.saturated_fat, 0) * mi.quantity as saturated_fat,
                       COALESCE(i.trans_fat, 0) * mi.quantity as trans_fat,
                       COALESCE(i.cholesterol, 0) * mi.quantity as cholesterol,
                       COALESCE(i.sodium, 0) * mi.quantity as sodium,
                       COALESCE(i.potassium, 0) * mi.quantity as potassium,
                       COALESCE(i.added_sugar, 0) * mi.quantity as added_sugar
                FROM meal_items mi
                JOIN items i ON mi.item_id = i.id
                WHERE mi.meal_id = %s ORDER BY mi.id
            """, (meal_id,))
            items = [dict(row) for row in cur.fetchall()]

            return {**dict(meal), 'items': items}

    def get_daily_totals(self, date: Optional[datetime] = None) -> dict:
        """Get nutrition totals for a specific day."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"""
                SELECT
                    COALESCE(SUM(COALESCE(i.calories, 0) * mi.quantity), 0) as calories,
                    COALESCE(SUM(COALESCE(i.protein, 0) * mi.quantity), 0) as protein,
                    COALESCE(SUM(COALESCE(i.carbs, 0) * mi.quantity), 0) as carbs,
                    COALESCE(SUM(COALESCE(i.fat, 0) * mi.quantity), 0) as fat,
                    COALESCE(SUM(COALESCE(i.fiber, 0) * mi.quantity), 0) as fiber,
                    COALESCE(SUM(COALESCE(i.alcohol, 0) * mi.quantity), 0) as alcohol,
                    COALESCE(SUM(COALESCE(i.saturated_fat, 0) * mi.quantity), 0) as saturated_fat,
                    COALESCE(SUM(COALESCE(i.trans_fat, 0) * mi.quantity), 0) as trans_fat,
                    COALESCE(SUM(COALESCE(i.cholesterol, 0) * mi.quantity), 0) as cholesterol,
                    COALESCE(SUM(COALESCE(i.sodium, 0) * mi.quantity), 0) as sodium,
                    COALESCE(SUM(COALESCE(i.potassium, 0) * mi.quantity), 0) as potassium,
                    COALESCE(SUM(COALESCE(i.added_sugar, 0) * mi.quantity), 0) as added_sugar,
                    COUNT(DISTINCT m.id) as meal_count
                FROM meals m
                LEFT JOIN meal_items mi ON mi.meal_id = m.id
                LEFT JOIN items i ON mi.item_id = i.id
                WHERE DATE(COALESCE(m.local_logged_at, m.logged_at)) = DATE(%s)
            """, (date,))
            return dict(cur.fetchone())

    def get_daily_breakdown(self, date: Optional[datetime] = None) -> list[dict]:
        """Get all meal items for a specific day with computed macros."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT mi.id, mi.meal_id, mi.item_id, i.name, mi.quantity, mi.unit,
                       COALESCE(i.calories, 0) * mi.quantity as calories,
                       COALESCE(i.protein, 0) * mi.quantity as protein,
                       COALESCE(i.carbs, 0) * mi.quantity as carbs,
                       COALESCE(i.fat, 0) * mi.quantity as fat,
                       COALESCE(i.fiber, 0) * mi.quantity as fiber,
                       COALESCE(i.alcohol, 0) * mi.quantity as alcohol,
                       COALESCE(i.saturated_fat, 0) * mi.quantity as saturated_fat,
                       COALESCE(i.trans_fat, 0) * mi.quantity as trans_fat,
                       COALESCE(i.cholesterol, 0) * mi.quantity as cholesterol,
                       COALESCE(i.sodium, 0) * mi.quantity as sodium,
                       COALESCE(i.potassium, 0) * mi.quantity as potassium,
                       COALESCE(i.added_sugar, 0) * mi.quantity as added_sugar
                FROM meal_items mi
                JOIN items i ON mi.item_id = i.id
                JOIN meals m ON mi.meal_id = m.id
                WHERE DATE(COALESCE(m.local_logged_at, m.logged_at)) = DATE(%s)
                ORDER BY COALESCE(m.local_logged_at, m.logged_at), mi.id
            """, (date,))
            return [dict(row) for row in cur.fetchall()]

    def get_meal_item(self, item_id: int) -> Optional[dict]:
        """Get a single meal item by ID with computed macros from items table."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT mi.id, mi.meal_id, mi.item_id, i.name, mi.quantity, mi.unit,
                       COALESCE(i.calories, 0) * mi.quantity as calories,
                       COALESCE(i.protein, 0) * mi.quantity as protein,
                       COALESCE(i.carbs, 0) * mi.quantity as carbs,
                       COALESCE(i.fat, 0) * mi.quantity as fat,
                       COALESCE(i.fiber, 0) * mi.quantity as fiber,
                       COALESCE(i.alcohol, 0) * mi.quantity as alcohol,
                       COALESCE(i.saturated_fat, 0) * mi.quantity as saturated_fat,
                       COALESCE(i.trans_fat, 0) * mi.quantity as trans_fat,
                       COALESCE(i.cholesterol, 0) * mi.quantity as cholesterol,
                       COALESCE(i.sodium, 0) * mi.quantity as sodium,
                       COALESCE(i.potassium, 0) * mi.quantity as potassium,
                       COALESCE(i.added_sugar, 0) * mi.quantity as added_sugar
                FROM meal_items mi
                JOIN items i ON mi.item_id = i.id
                WHERE mi.id = %s
            """, (item_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_meal_item(self, item_id: int, new_quantity: float) -> Optional[dict]:
        """Update a meal item's quantity. Returns updated item with computed macros."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "UPDATE meal_items SET quantity = %s WHERE id = %s",
                    (new_quantity, item_id))
                if cur.rowcount == 0:
                    return None

            self.conn.commit()
            return self.get_meal_item(item_id)
        except Exception as e:
            self.conn.rollback()
            raise

    # Conversion factors to a common base (grams for weight, ml for volume)
    UNIT_CONVERSIONS = {
        'g': {'g': 1, 'oz': 1 / 28.3495},
        'oz': {'oz': 1, 'g': 28.3495},
        'fl_oz': {'fl_oz': 1},
        'item': {'item': 1},
    }

    def update_meal_item_unit(self, item_id: int, new_unit: str) -> Optional[dict]:
        """Update a meal item's unit, converting the quantity where possible."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM meal_items WHERE id = %s", (item_id,))
                item = cur.fetchone()
                if not item:
                    return None

                old_unit = item['unit']
                old_quantity = item['quantity']

                factor = self.UNIT_CONVERSIONS.get(old_unit, {}).get(new_unit)
                new_quantity = round(old_quantity * factor, 2) if factor else old_quantity

                cur.execute(
                    "UPDATE meal_items SET unit = %s, quantity = %s WHERE id = %s",
                    (new_unit, new_quantity, item_id))

            self.conn.commit()
            return self.get_meal_item(item_id)
        except Exception as e:
            self.conn.rollback()
            raise

    def copy_meal(self, meal_id: int) -> Optional[int]:
        """Copy a meal by creating a new meal row + new meal_items referencing same item_ids.
        Returns new meal id or None if original not found."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM meals WHERE id = %s", (meal_id,))
                meal = cur.fetchone()
                if not meal:
                    return None

                cur.execute("""
                    INSERT INTO meals (name, description)
                    VALUES (%s, %s)
                    RETURNING id
                """, (meal['name'], meal['description']))
                new_meal_id = cur.fetchone()['id']

                # Copy meal_items referencing same item_ids
                cur.execute("""
                    INSERT INTO meal_items (meal_id, item_id, unit, quantity)
                    SELECT %s, item_id, unit, quantity
                    FROM meal_items WHERE meal_id = %s
                """, (new_meal_id, meal_id))

            self.conn.commit()
            return new_meal_id
        except Exception as e:
            self.conn.rollback()
            raise

    def copy_item_as_meal(self, item_id: int) -> Optional[int]:
        """Copy a meal item by creating a new meal with one meal_item referencing same item.
        Returns new meal id or None if item not found."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get the meal_item with item name
                cur.execute("""
                    SELECT mi.*, i.name
                    FROM meal_items mi
                    JOIN items i ON mi.item_id = i.id
                    WHERE mi.id = %s
                """, (item_id,))
                mi = cur.fetchone()
                if not mi:
                    return None

                cur.execute("""
                    INSERT INTO meals (name, description)
                    VALUES (%s, %s)
                    RETURNING id
                """, (mi['name'], mi['name']))
                new_meal_id = cur.fetchone()['id']

                cur.execute("""
                    INSERT INTO meal_items (meal_id, item_id, unit, quantity)
                    VALUES (%s, %s, %s, %s)
                """, (new_meal_id, mi['item_id'], mi['unit'], mi['quantity']))

            self.conn.commit()
            return new_meal_id
        except Exception as e:
            self.conn.rollback()
            raise

    def delete_meal(self, meal_id: int) -> bool:
        """Delete a meal. CASCADE handles meal_items cleanup."""
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

    def get_latest_weight(self, date: Optional[datetime] = None) -> Optional[float]:
        """Get weight for a date, falling back to the most recent earlier weight."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT weight_lbs FROM daily_weights WHERE date <= DATE(%s) ORDER BY date DESC LIMIT 1",
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
            base = """
                SELECT DATE(COALESCE(m.local_logged_at, m.logged_at)) as date,
                       SUM(COALESCE(i.calories, 0) * mi.quantity) as calories
                FROM meals m
                JOIN meal_items mi ON mi.meal_id = m.id
                JOIN items i ON mi.item_id = i.id
                WHERE DATE(COALESCE(m.local_logged_at, m.logged_at)) < CURRENT_DATE
            """
            if days:
                cur.execute(base + """
                    AND DATE(COALESCE(m.local_logged_at, m.logged_at)) >= CURRENT_DATE - %s * INTERVAL '1 day'
                    GROUP BY DATE(COALESCE(m.local_logged_at, m.logged_at))
                    ORDER BY date ASC
                """, (days,))
            else:
                cur.execute(base + """
                    GROUP BY DATE(COALESCE(m.local_logged_at, m.logged_at))
                    ORDER BY date ASC
                """)
            return [{'date': str(row['date']), 'calories': int(row['calories'] or 0)} for row in cur.fetchall()]

    def get_macros_history(self, days: Optional[int] = None) -> list[dict]:
        """Get daily macros history for all nutrients. If days is None, returns all history.
        Excludes today since the day is not complete."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            base = """
                SELECT DATE(COALESCE(m.local_logged_at, m.logged_at)) as date,
                    SUM(COALESCE(i.calories, 0) * mi.quantity) as calories,
                    SUM(COALESCE(i.protein, 0) * mi.quantity) as protein,
                    SUM(COALESCE(i.carbs, 0) * mi.quantity) as carbs,
                    SUM(COALESCE(i.fat, 0) * mi.quantity) as fat,
                    SUM(COALESCE(i.fiber, 0) * mi.quantity) as fiber,
                    SUM(COALESCE(i.alcohol, 0) * mi.quantity) as alcohol,
                    SUM(COALESCE(i.saturated_fat, 0) * mi.quantity) as saturated_fat,
                    SUM(COALESCE(i.trans_fat, 0) * mi.quantity) as trans_fat,
                    SUM(COALESCE(i.cholesterol, 0) * mi.quantity) as cholesterol,
                    SUM(COALESCE(i.sodium, 0) * mi.quantity) as sodium,
                    SUM(COALESCE(i.potassium, 0) * mi.quantity) as potassium,
                    SUM(COALESCE(i.added_sugar, 0) * mi.quantity) as added_sugar
                FROM meals m
                JOIN meal_items mi ON mi.meal_id = m.id
                JOIN items i ON mi.item_id = i.id
                WHERE DATE(COALESCE(m.local_logged_at, m.logged_at)) < CURRENT_DATE
            """
            if days:
                cur.execute(base + """
                    AND DATE(COALESCE(m.local_logged_at, m.logged_at)) >= CURRENT_DATE - %s * INTERVAL '1 day'
                    GROUP BY DATE(COALESCE(m.local_logged_at, m.logged_at))
                    ORDER BY date ASC
                """, (days,))
            else:
                cur.execute(base + """
                    GROUP BY DATE(COALESCE(m.local_logged_at, m.logged_at))
                    ORDER BY date ASC
                """)

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

    # Exercise entry methods
    def add_exercise_entry(self, date: Optional[datetime], exercise_type: str,
                           amount: Optional[float], unit: Optional[str],
                           calories_burnt: int) -> int:
        """Add an exercise entry. Returns the entry id."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO exercise_entries (date, exercise_type, amount, unit, calories_burnt)
                VALUES (DATE(%s), %s, %s, %s, %s)
                RETURNING id
            """, (date, exercise_type, amount, unit, calories_burnt))
            entry_id = cur.fetchone()[0]
        self.conn.commit()
        return entry_id

    def get_exercise_entries(self, date: Optional[datetime] = None) -> list[dict]:
        """Get exercise entries for a specific date."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, date, exercise_type, amount, unit, calories_burnt, logged_at
                FROM exercise_entries
                WHERE date = DATE(%s)
                ORDER BY logged_at ASC
            """, (date,))
            return [dict(row) for row in cur.fetchall()]

    def get_daily_burnt_calories(self, date: Optional[datetime] = None) -> int:
        """Get total burnt calories for a specific date."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT COALESCE(SUM(calories_burnt), 0)
                FROM exercise_entries
                WHERE date = DATE(%s)
            """, (date,))
            return cur.fetchone()[0]

    def delete_exercise_entry(self, entry_id: int) -> bool:
        """Delete an exercise entry. Returns True if deleted."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM exercise_entries WHERE id = %s", (entry_id,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    # Caloric target operations
    def set_caloric_target(self, target_calories: int, date: Optional[datetime] = None) -> None:
        """Set caloric target for a specific date. Updates if exists, inserts if not."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO caloric_targets (date, target_calories)
                VALUES (DATE(%s), %s)
                ON CONFLICT (date)
                DO UPDATE SET target_calories = %s, updated_at = CURRENT_TIMESTAMP
            """, (date, target_calories, target_calories))
        self.conn.commit()

    def get_caloric_target(self, date: Optional[datetime] = None) -> Optional[int]:
        """Get caloric target for a specific date (carry-forward lookup).
        Returns the most recent target set on or before the given date."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT target_calories FROM caloric_targets
                WHERE date <= DATE(%s)
                ORDER BY date DESC LIMIT 1
            """, (date,))
            row = cur.fetchone()
            return row[0] if row else None

    def delete_caloric_target(self, date: Optional[datetime] = None) -> bool:
        """Delete caloric target for a specific date. Returns True if deleted."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM caloric_targets WHERE date = DATE(%s)", (date,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def get_caloric_target_history(self, days: Optional[int] = None) -> list[dict]:
        """Get caloric target change-point history for charting."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            if days:
                cur.execute("""
                    SELECT date, target_calories
                    FROM caloric_targets
                    WHERE date >= CURRENT_DATE - %s * INTERVAL '1 day'
                    ORDER BY date ASC
                """, (days,))
            else:
                cur.execute("""
                    SELECT date, target_calories
                    FROM caloric_targets
                    ORDER BY date ASC
                """)
            return [{'date': str(row['date']), 'target_calories': row['target_calories']} for row in cur.fetchall()]

    # Progress photo operations
    def save_progress_photo(self, date, image_data: bytes) -> None:
        """Save a progress photo for a specific date. Updates if exists, inserts if not."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO progress_photos (date, image_data)
                VALUES (DATE(%s), %s)
                ON CONFLICT (date)
                DO UPDATE SET image_data = %s, updated_at = CURRENT_TIMESTAMP
            """, (date, psycopg2.Binary(image_data), psycopg2.Binary(image_data)))
        self.conn.commit()

    def get_progress_photo(self, date) -> Optional[bytes]:
        """Get progress photo for a specific date. Returns raw bytes or None."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("SELECT image_data FROM progress_photos WHERE date = DATE(%s)", (date,))
            row = cur.fetchone()
            if row and row[0]:
                return bytes(row[0])
        return None

    def delete_progress_photo(self, date) -> bool:
        """Delete progress photo for a specific date. Returns True if deleted."""
        if date is None:
            date = datetime.now()

        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM progress_photos WHERE date = DATE(%s)", (date,))
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

    def get_progress_photo_dates(self) -> list[str]:
        """Get all dates that have progress photos, most recent first."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT date FROM progress_photos ORDER BY date DESC")
            return [str(row[0]) for row in cur.fetchall()]

    # Progress video operations
    def save_progress_video(self, video_data: bytes,
                            first_photo_date: str, last_photo_date: str) -> int:
        """Save a progress video, replacing any existing one. Returns the new video id."""
        try:
            with self.conn.cursor() as cur:
                cur.execute("DELETE FROM progress_videos")
                cur.execute("""
                    INSERT INTO progress_videos (video_data, first_photo_date, last_photo_date)
                    VALUES (%s, DATE(%s), DATE(%s))
                    RETURNING id
                """, (psycopg2.Binary(video_data), first_photo_date, last_photo_date))
                video_id = cur.fetchone()[0]
            self.conn.commit()
            return video_id
        except Exception:
            self.conn.rollback()
            raise

    def get_progress_video(self) -> Optional[dict]:
        """Get the progress video. Returns dict with video_data and metadata, or None."""
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, video_data, first_photo_date, last_photo_date, created_at
                FROM progress_videos
                ORDER BY created_at DESC LIMIT 1
            """)
            row = cur.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'video_data': bytes(row['video_data']),
                    'first_photo_date': str(row['first_photo_date']),
                    'last_photo_date': str(row['last_photo_date']),
                    'created_at': row['created_at'].isoformat(),
                }
        return None

    def delete_progress_video(self) -> bool:
        """Delete the progress video. Returns True if deleted."""
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM progress_videos")
            deleted = cur.rowcount > 0
        self.conn.commit()
        return deleted

