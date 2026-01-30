import json
import sqlite3
from datetime import datetime


DB_NAME = "et_interviews.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                industries TEXT,
                functions TEXT,
                levels TEXT,
                free_text TEXT,
                expert_id TEXT,
                expert_name TEXT,
                script_text TEXT,
                transcript_text TEXT,
                summary_text TEXT,
                tags_json TEXT,
                interview_rating INTEGER
            )
            """
        )
        try:
            conn.execute("ALTER TABLE interviews ADD COLUMN interview_rating INTEGER")
        except sqlite3.OperationalError:
            pass
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS case_records (
                case_code TEXT PRIMARY KEY,
                updated_at TEXT NOT NULL,
                data_json TEXT
            )
            """
        )
        conn.commit()


def save_interview(payload):
    created_at = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO interviews (
                created_at,
                industries,
                functions,
                levels,
                free_text,
                expert_id,
                expert_name,
                script_text,
                transcript_text,
                summary_text,
                tags_json,
                interview_rating
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                json.dumps(payload.get("industries") or []),
                json.dumps(payload.get("functions") or []),
                json.dumps(payload.get("levels") or []),
                payload.get("free_text") or "",
                payload.get("expert_id"),
                payload.get("expert_name"),
                payload.get("script_text"),
                payload.get("transcript_text"),
                payload.get("summary_text"),
                json.dumps(payload.get("tags") or {}),
                payload.get("interview_rating"),
            ),
        )
        conn.commit()


def load_recent(limit=10):
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, expert_name, tags_json
            FROM interviews
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    results = []
    for row in rows:
        results.append(
            {
                "id": row[0],
                "created_at": row[1],
                "expert_name": row[2],
                "tags": json.loads(row[3]) if row[3] else {},
            }
        )
    return results


def load_by_id(interview_id):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT id, created_at, industries, functions, levels, free_text,
                   expert_id, expert_name, script_text, transcript_text, summary_text, tags_json, interview_rating
            FROM interviews
            WHERE id = ?
            """,
            (interview_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "industries": json.loads(row[2]) if row[2] else [],
        "functions": json.loads(row[3]) if row[3] else [],
        "levels": json.loads(row[4]) if row[4] else [],
        "free_text": row[5],
        "expert_id": row[6],
        "expert_name": row[7],
        "script_text": row[8],
        "transcript_text": row[9],
        "summary_text": row[10],
        "tags": json.loads(row[11]) if row[11] else {},
        "interview_rating": row[12],
    }


def save_case(case_code, data):
    updated_at = datetime.utcnow().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO case_records (case_code, updated_at, data_json)
            VALUES (?, ?, ?)
            ON CONFLICT(case_code) DO UPDATE SET
                updated_at=excluded.updated_at,
                data_json=excluded.data_json
            """,
            (case_code, updated_at, json.dumps(data or {})),
        )
        conn.commit()


def load_case(case_code):
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT case_code, updated_at, data_json
            FROM case_records
            WHERE case_code = ?
            """,
            (case_code,),
        ).fetchone()
    if not row:
        return None
    return {
        "case_code": row[0],
        "updated_at": row[1],
        "data": json.loads(row[2]) if row[2] else {},
    }
