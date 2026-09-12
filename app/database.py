"""
database.py
Handles the SQLite connection and creates the notes table if it doesn't exist.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "notes.db"


def get_connection():
    """Returns a new SQLite connection. row_factory lets us access columns by name."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the notes table if it doesn't already exist. Call this once at startup."""
    conn = get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            tags TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()
