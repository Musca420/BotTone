from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd


class SQLiteEventStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def initialize(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp TEXT NOT NULL
                )
                """
            )

    async def append(
        self, event_id: str, event_type: str, payload: str, timestamp: datetime
    ) -> bool:
        return await asyncio.to_thread(
            self._append_sync, event_id, event_type, payload, timestamp.isoformat()
        )

    def _append_sync(self, event_id: str, event_type: str, payload: str, timestamp: str) -> bool:
        try:
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?)",
                    (event_id, event_type, payload, timestamp),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    async def contains(self, event_id: str) -> bool:
        return await asyncio.to_thread(self._contains_sync, event_id)

    def _contains_sync(self, event_id: str) -> bool:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT 1 FROM events WHERE event_id = ?", (event_id,)
            ).fetchone()
        return row is not None


class ParquetRepository:
    @staticmethod
    def read(path: str | Path) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def write(frame: pd.DataFrame, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)

    @staticmethod
    def query(path: str | Path, sql: str) -> pd.DataFrame:
        with duckdb.connect() as connection:
            connection.read_parquet(str(path)).create_view("candles")
            return connection.sql(sql).df()
