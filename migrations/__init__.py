"""SQLite schema migrations for Duel of Minds."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable, Tuple

Migration = Tuple[str, str]

MIGRATIONS: Iterable[Migration] = (
    (
        "0001_initial",
        """
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at_utc TEXT NOT NULL,
            backend TEXT NOT NULL,
            model_name TEXT NOT NULL,
            persona_a_path TEXT NOT NULL,
            persona_b_path TEXT NOT NULL,
            params_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_messages_run_turn
            ON messages(run_id, turn_index);
        """,
    ),
    (
        "0002_memory",
        """
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL,
            turn_id INTEGER NOT NULL,
            confidence REAL NOT NULL DEFAULT 0.5,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            UNIQUE(run_id, subject, predicate, object),
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS themes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            theme TEXT NOT NULL,
            strength REAL NOT NULL,
            last_seen_turn INTEGER NOT NULL,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            UNIQUE(run_id, theme),
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            step INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            UNIQUE(run_id, step),
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS message_embeddings (
            run_id INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            speaker TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at_utc TEXT NOT NULL,
            PRIMARY KEY(run_id, turn_index),
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_facts_run_turn
            ON facts(run_id, turn_id);
        CREATE INDEX IF NOT EXISTS idx_themes_run
            ON themes(run_id, strength DESC, last_seen_turn DESC);
        CREATE INDEX IF NOT EXISTS idx_summaries_run
            ON summaries(run_id, step DESC);
        CREATE INDEX IF NOT EXISTS idx_embeddings_run
            ON message_embeddings(run_id, turn_index DESC);
        """,
    ),
    (
        "0003_growth_metrics",
        """
        CREATE TABLE IF NOT EXISTS growth_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            novelty_score REAL NOT NULL,
            max_similarity REAL,
            method TEXT NOT NULL,
            threshold REAL,
            attempt_count INTEGER NOT NULL,
            flagged INTEGER NOT NULL DEFAULT 0,
            created_at_utc TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(id)
        );

        CREATE INDEX IF NOT EXISTS idx_growth_run_turn
            ON growth_metrics(run_id, turn_index);
        """,
    ),
)


def run_migrations(conn) -> None:
    """Apply pending migrations to the SQLite connection."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at_utc TEXT NOT NULL
        );
        """
    )
    cur = conn.cursor()
    cur.execute("SELECT version FROM schema_migrations")
    applied = {row[0] for row in cur.fetchall()}
    for version, sql in MIGRATIONS:
        if version in applied:
            continue
        with conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at_utc) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat()),
            )
