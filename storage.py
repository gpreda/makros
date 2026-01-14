"""SQLite storage for makros."""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from models import Item


class Storage:
    """SQLite storage for items."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.makros/makros.db
                     Use ':memory:' for in-memory database (testing).
        """
        if db_path is None:
            db_dir = Path.home() / '.makros'
            db_dir.mkdir(exist_ok=True)
            db_path = str(db_dir / 'makros.db')

        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

        # For in-memory databases, keep a persistent connection
        if db_path == ':memory:':
            self._conn = sqlite3.connect(':memory:')
            self._conn.row_factory = sqlite3.Row

        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get database connection."""
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bar_code TEXT UNIQUE,
                    name TEXT NOT NULL UNIQUE,
                    description TEXT,
                    unit_conversions TEXT
                )
            ''')
            # Create index on bar_code for faster lookups (excluding NULLs)
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_items_bar_code
                ON items(bar_code) WHERE bar_code IS NOT NULL
            ''')
            conn.commit()

    def add_item(self, item: Item) -> Item:
        """Add a new item to the database.

        Returns the item with its assigned id.
        Raises sqlite3.IntegrityError if name or bar_code already exists.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                '''INSERT INTO items (bar_code, name, description, unit_conversions)
                   VALUES (?, ?, ?, ?)''',
                (item.bar_code, item.name, item.description,
                 json.dumps(item.unit_conversions))
            )
            item.id = cursor.lastrowid
            conn.commit()
        return item

    def get_item_by_id(self, item_id: int) -> Optional[Item]:
        """Get item by id."""
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM items WHERE id = ?', (item_id,)
            ).fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def get_item_by_name(self, name: str) -> Optional[Item]:
        """Get item by name (case-insensitive)."""
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM items WHERE LOWER(name) = LOWER(?)', (name,)
            ).fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def get_item_by_barcode(self, bar_code: str) -> Optional[Item]:
        """Get item by barcode."""
        with self._get_conn() as conn:
            row = conn.execute(
                'SELECT * FROM items WHERE bar_code = ?', (bar_code,)
            ).fetchone()
            if row:
                return self._row_to_item(row)
        return None

    def update_item(self, item: Item) -> bool:
        """Update an existing item.

        Returns True if item was found and updated.
        """
        if item.id is None:
            return False

        with self._get_conn() as conn:
            cursor = conn.execute(
                '''UPDATE items
                   SET bar_code = ?, name = ?, description = ?, unit_conversions = ?
                   WHERE id = ?''',
                (item.bar_code, item.name, item.description,
                 json.dumps(item.unit_conversions), item.id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_item(self, item_id: int) -> bool:
        """Delete an item by id.

        Returns True if item was found and deleted.
        """
        with self._get_conn() as conn:
            cursor = conn.execute('DELETE FROM items WHERE id = ?', (item_id,))
            conn.commit()
            return cursor.rowcount > 0

    def search_items(self, query: str, limit: int = 20) -> list[Item]:
        """Search items by name (partial match)."""
        with self._get_conn() as conn:
            rows = conn.execute(
                '''SELECT * FROM items
                   WHERE LOWER(name) LIKE LOWER(?)
                   ORDER BY name LIMIT ?''',
                (f'%{query}%', limit)
            ).fetchall()
            return [self._row_to_item(row) for row in rows]

    def list_items(self, limit: int = 100, offset: int = 0) -> list[Item]:
        """List all items with pagination."""
        with self._get_conn() as conn:
            rows = conn.execute(
                'SELECT * FROM items ORDER BY name LIMIT ? OFFSET ?',
                (limit, offset)
            ).fetchall()
            return [self._row_to_item(row) for row in rows]

    def count_items(self) -> int:
        """Get total number of items."""
        with self._get_conn() as conn:
            row = conn.execute('SELECT COUNT(*) FROM items').fetchone()
            return row[0]

    def _row_to_item(self, row: sqlite3.Row) -> Item:
        """Convert database row to Item object."""
        unit_conversions = {}
        if row['unit_conversions']:
            unit_conversions = json.loads(row['unit_conversions'])

        return Item(
            id=row['id'],
            bar_code=row['bar_code'],
            name=row['name'],
            description=row['description'],
            unit_conversions=unit_conversions,
        )
