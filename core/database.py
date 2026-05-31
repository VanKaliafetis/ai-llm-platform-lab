import os
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = os.getenv("DATABASE_PATH", "data/platform.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                module TEXT,
                provider TEXT,
                model TEXT,
                prompt TEXT,
                response TEXT,
                latency_ms REAL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost_usd REAL,
                tokens_per_second REAL,
                score REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS eval_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                dataset_name TEXT,
                provider TEXT,
                model TEXT,
                prompt TEXT,
                expected TEXT,
                response TEXT,
                score REAL,
                latency_ms REAL,
                tokens_per_second REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS rag_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                question TEXT,
                answer TEXT,
                retrieved_context TEXT,
                retrieval_score REAL,
                groundedness_score REAL,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS finetune_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                status TEXT,
                base_model TEXT,
                dataset_path TEXT,
                output_dir TEXT,
                train_rows INTEGER,
                valid_rows INTEGER,
                metrics TEXT
            );
            """
        )

        ensure_column(c, "runs", "tokens_per_second", "REAL")
        ensure_column(c, "eval_results", "tokens_per_second", "REAL")


def ensure_column(c, table, column, col_type):
    existing = [row["name"] for row in c.execute(f"PRAGMA table_info({table})").fetchall()]

    if column not in existing:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def insert(table, data):
    data = {"created_at": datetime.utcnow().isoformat(), **data}

    keys = ",".join(data.keys())
    qs = ",".join(["?"] * len(data))

    with conn() as c:
        cur = c.execute(
            f"INSERT INTO {table} ({keys}) VALUES ({qs})",
            list(data.values()),
        )
        return cur.lastrowid


def rows(query="SELECT * FROM runs ORDER BY id DESC LIMIT 200", params=()):
    with conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def cleanup_mock_rows():
    total_deleted = 0

    with conn() as c:
        for table in ["runs", "eval_results"]:
            cur = c.execute(
                f"""
                DELETE FROM {table}
                WHERE provider = 'mock'
                   OR model LIKE '%mock%'
                   OR response LIKE '%placeholder%'
                   OR response LIKE '%Mock model response%'
                """
            )
            total_deleted += cur.rowcount

        cur = c.execute(
            """
            DELETE FROM rag_results
            WHERE answer LIKE '%placeholder%'
               OR answer LIKE '%Mock model response%'
            """
        )
        total_deleted += cur.rowcount

    return total_deleted