import sqlite3
import uuid
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "chat_history.db"


# ============================================================================
# ============================================================================

@contextmanager
def _get_conn():
    """
    اتصال SQLite جديد لكل عملية (بدل اتصال واحد مشترك) تفادياً لمشاكل
    thread-safety اللي قد تصير مع Streamlit.
    """
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                entry_json TEXT NOT NULL,   -- كامل chat_entry dict كـ JSON
                created_at REAL NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_entries_session
            ON chat_entries (session_id, id)
        """)


# ============================================================================
# ============================================================================

def save_entry(session_id: str, entry: Dict[str, Any]) -> None:
   
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_entries (session_id, entry_json, created_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(entry, ensure_ascii=False, default=str), time.time()),
        )


def load_history(session_id: str) -> List[Dict[str, Any]]:
   
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT entry_json FROM chat_entries WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()

    history = []
    for row in rows:
        try:
            history.append(json.loads(row["entry_json"]))
        except json.JSONDecodeError:
            continue
    return history


def clear_session(session_id: str) -> None:
    with _get_conn() as conn:
        conn.execute("DELETE FROM chat_entries WHERE session_id = ?", (session_id,))


# ============================================================================
# ============================================================================

def get_or_create_session_id(st) -> str:
  
    existing = st.query_params.get("sid")
    if existing:
        return existing

    new_sid = str(uuid.uuid4())
    st.query_params["sid"] = new_sid
    return new_sid
