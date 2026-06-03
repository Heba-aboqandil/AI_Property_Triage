"""
db.py
-----
SQLite storage for chat sessions and submitted listings.

Two tables:
  - chat_sessions(id, title, created_at, updated_at)
  - chat_messages(id, session_id, role, content, created_at)
  - last_listing(id=1, payload_json, report_json, submitted_at)

The `last_listing` table holds AT MOST one row — the most recently submitted
listing. Tab 1 reads from it so the chat assistant can answer questions
about the user's freshly submitted property.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path("chat_history.db")


# ============================================================
# CONNECTION
# ============================================================
def _conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Create tables on first run. Safe to call repeatedly."""
    with _conn() as c:
        c.executescript("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                updated_at  TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                role        TEXT    NOT NULL CHECK(role IN ('user','assistant')),
                content     TEXT    NOT NULL,
                created_at  TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON chat_messages(session_id, id);

            CREATE TABLE IF NOT EXISTS last_listing (
                id            INTEGER PRIMARY KEY CHECK(id = 1),
                payload_json  TEXT    NOT NULL,
                report_json   TEXT,
                submitted_at  TEXT    NOT NULL
            );
        """)


# ============================================================
# SESSIONS
# ============================================================
def create_session(title: str = "New Chat") -> int:
    now = datetime.now().isoformat()
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO chat_sessions(title, created_at, updated_at) VALUES (?, ?, ?)",
            (title, now, now),
        )
        return cur.lastrowid


def list_sessions() -> list[dict]:
    """Return all sessions, newest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, title, created_at, updated_at "
            "FROM chat_sessions ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def rename_session(session_id: int, new_title: str):
    with _conn() as c:
        c.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, datetime.now().isoformat(), session_id),
        )


def delete_session(session_id: int):
    with _conn() as c:
        c.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))


def session_exists(session_id: int) -> bool:
    with _conn() as c:
        r = c.execute(
            "SELECT 1 FROM chat_sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return r is not None


# ============================================================
# MESSAGES
# ============================================================
def get_messages(session_id: int) -> list[dict]:
    """Return all messages for a session in chronological order."""
    with _conn() as c:
        rows = c.execute(
            "SELECT role, content FROM chat_messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]


def add_message(session_id: int, role: str, content: str):
    """Append a message and bump the session's updated_at."""
    now = datetime.now().isoformat()
    with _conn() as c:
        c.execute(
            "INSERT INTO chat_messages(session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, role, content, now),
        )
        c.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )


# ============================================================
# LAST LISTING (shared between Tab 2 → Tab 1)
# ============================================================
def save_last_listing(payload: dict, report: Optional[dict] = None):
    """Overwrite the single last-listing row."""
    with _conn() as c:
        c.execute(
            "INSERT INTO last_listing(id, payload_json, report_json, submitted_at) "
            "VALUES (1, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "payload_json = excluded.payload_json, "
            "report_json  = excluded.report_json, "
            "submitted_at = excluded.submitted_at",
            (
                json.dumps(payload, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False) if report else None,
                datetime.now().isoformat(),
            ),
        )


def get_last_listing() -> Optional[dict]:
    """Return {payload, report, submitted_at} or None if nothing submitted yet."""
    with _conn() as c:
        r = c.execute(
            "SELECT payload_json, report_json, submitted_at FROM last_listing WHERE id = 1"
        ).fetchone()
        if not r:
            return None
        return {
            "payload": json.loads(r["payload_json"]),
            "report":  json.loads(r["report_json"]) if r["report_json"] else None,
            "submitted_at": r["submitted_at"],
        }


def clear_last_listing():
    with _conn() as c:
        c.execute("DELETE FROM last_listing WHERE id = 1")


# ============================================================
# Initialise on import
# ============================================================
init_db()