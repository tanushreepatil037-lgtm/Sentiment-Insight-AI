
# ============================================================
# SENTIMENT INSIGHT AI
# SQLite Database Management
# ============================================================

import sqlite3
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent
DATABASE_PATH = PROJECT_ROOT / "sentiment_history.db"


def get_connection():
    """Create a database connection."""
    return sqlite3.connect(DATABASE_PATH)


def initialize_database():
    """Create prediction history table if it does not exist."""

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        conn.commit()


def save_prediction(text, sentiment, confidence):
    """Save a sentiment prediction."""

    initialize_database()

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO predictions (
                input_text,
                sentiment,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?)
        """, (
            text,
            sentiment,
            float(confidence),
            created_at
        ))

        conn.commit()

        return cursor.lastrowid


def get_all_predictions():
    """Retrieve all saved predictions."""

    initialize_database()

    with get_connection() as conn:

        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                id,
                created_at,
                input_text,
                sentiment,
                confidence
            FROM predictions
            ORDER BY id DESC
        """)

        rows = cursor.fetchall()

    import pandas as pd

    return pd.DataFrame(
        rows,
        columns=[
            "ID",
            "Date & Time",
            "Input Text",
            "Sentiment",
            "Confidence (%)"
        ]
    )


def get_prediction_count():
    """Return total number of predictions."""

    initialize_database()

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM predictions")

        return cursor.fetchone()[0]


def delete_all_predictions():
    """Delete all prediction records."""

    initialize_database()

    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("DELETE FROM predictions")

        conn.commit()
