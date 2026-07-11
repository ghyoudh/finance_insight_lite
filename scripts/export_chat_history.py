#!/usr/bin/env python3
from pathlib import Path
import json
import sqlite3
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "src" / "chat_history.db"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "chat_history_export.jsonl"


def load_chat_entries(db_path: Path):
    if not db_path.exists():
        raise FileNotFoundError(f"Chat database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT entry_json FROM chat_entries ORDER BY session_id ASC, id ASC").fetchall()
    conn.close()
    return [json.loads(row["entry_json"]) for row in rows if row["entry_json"].strip()]


def build_jsonl_record(chat_entry: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "question": chat_entry.get("question", ""),
        "generated_answer": chat_entry.get("answer", ""),
    }

    if "source_chunks" in chat_entry and chat_entry["source_chunks"] is not None:
        record["source_chunks"] = chat_entry["source_chunks"]
    elif "source_pages" in chat_entry and chat_entry["source_pages"] is not None:
        record["retrieved_context"] = chat_entry["source_pages"]

    # Optional reference answer if you want to add it manually later
    if "reference_answer" in chat_entry:
        record["reference_answer"] = chat_entry["reference_answer"]

    return record


def export_chat_history():
    entries = load_chat_entries(DB_PATH)
    if not entries:
        raise SystemExit("No chat history found in the database.")

    records = [build_jsonl_record(entry) for entry in entries]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Exported {len(records)} samples to {OUTPUT_PATH}")


if __name__ == "__main__":
    export_chat_history()
