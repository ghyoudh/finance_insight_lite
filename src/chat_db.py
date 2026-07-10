"""
chat_db.py
==========
طبقة تخزين بسيطة (SQLite) تحفظ سجل المحادثة الحالية (chat_history) بحيث
لو المستخدم عمل refresh للصفحة (F5) بـ Streamlit، ما تنحذف محادثته.

الفكرة:
-------
Streamlit يفقد st.session_state كامل عند أي refresh حقيقي. عشان نتجاوز هذا:

  1) هوية ثابتة تنجو من الـ refresh -> query param اسمه "sid" بالـ URL.
     أول ما تفتح الصفحة يتولد UUID ويُحط بالـ URL، وأي refresh بعدها
     (بنفس الرابط) يرجّع نفس sid.

  2) كل "chat_entry" (نفس الـ dict اللي يُخزن حالياً بـ
     st.session_state.chat_history: question/answer/source_pages/
     confidence/verification/relevant_docs_count/chart) يُحفظ كصف واحد
     بجدول SQLite، مربوط بهذا sid، ويُقرأ عند تحميل الصفحة لتعبئة
     st.session_state.chat_history تلقائياً بدل ما يبدأ فاضياً.

ملاحظة مهمة: هذا يحفظ سجل *المحادثة* (الأسئلة والأجوبة) فقط. الـ
vector_db والـ agent نفسها (المبنية من الملفات المرفوعة) تبقى في الذاكرة
فقط ولا تُحفظ هنا - لأنها كائنات Python كبيرة (Chroma/FAISS) مو بيانات
قابلة للتخزين بسهولة بـ SQLite. يعني بعد الـ refresh، سجل الشات يرجع،
لكن لازم يعاد معالجة الملفات (Process Documents) قبل السؤال من جديد.
"""

import sqlite3
import uuid
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

DB_PATH = Path(__file__).parent / "chat_history.db"


# ============================================================================
# الاتصال وتهيئة القاعدة
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
    """ينشئ الجدول لو مو موجود. يُستدعى مرة عند بداية التطبيق."""
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
# القراءة والكتابة
# ============================================================================

def save_entry(session_id: str, entry: Dict[str, Any]) -> None:
    """
    يحفظ chat_entry واحد كامل (نفس الـ dict اللي يُضاف لـ
    st.session_state.chat_history) - يُستدعى مباشرة بعد كل
    .append(chat_entry) بالكود عندك.
    """
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO chat_entries (session_id, entry_json, created_at) VALUES (?, ?, ?)",
            (session_id, json.dumps(entry, ensure_ascii=False, default=str), time.time()),
        )


def load_history(session_id: str) -> List[Dict[str, Any]]:
    """
    يرجّع كل chat_history الخاص بجلسة معينة، مرتب زمنياً - يُستخدم
    لتعبئة st.session_state.chat_history عند أول تحميل للصفحة
    (أو بعد الـ refresh).
    """
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
            continue  # صف تالف نادراً ما يصير - نتجاهله بدل ما نكسر التطبيق
    return history


def clear_session(session_id: str) -> None:
    """يمسح كل chat_entries الخاصة بجلسة معينة - يُستخدم بزر Clear Chat History."""
    with _get_conn() as conn:
        conn.execute("DELETE FROM chat_entries WHERE session_id = ?", (session_id,))


# ============================================================================
# إدارة هوية الجلسة (session_id) عبر query params بـ Streamlit
# ============================================================================

def get_or_create_session_id(st) -> str:
    """
    يرجّع session_id ثابت لتاب/رابط المستخدم الحالي، وينشئه أول مرة فقط.

      - نقرأ query param اسمه "sid" من رابط الصفحة الحالي
      - موجود -> نستخدمه (يعني هذا refresh لنفس الرابط)
      - غير موجود -> نولّد UUID جديد ونحطه بالـ URL عبر st.query_params

    ملاحظة: st.query_params يحتاج Streamlit >= 1.30 تقريباً. لو نسختك
    أقدم استخدم st.experimental_get_query_params /
    st.experimental_set_query_params بنفس المنطق.
    """
    existing = st.query_params.get("sid")
    if existing:
        return existing

    new_sid = str(uuid.uuid4())
    st.query_params["sid"] = new_sid
    return new_sid
