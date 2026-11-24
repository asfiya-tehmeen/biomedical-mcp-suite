"""DuckDB-backed response cache / frozen snapshot store."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    key         VARCHAR PRIMARY KEY,
    source      VARCHAR,
    source_url  VARCHAR,
    query_json  VARCHAR,
    retrieved_at VARCHAR,
    status      INTEGER,
    sha256      VARCHAR,
    body        VARCHAR,
    frozen      BOOLEAN DEFAULT FALSE
)
"""


@dataclass
class StoredResponse:
    key: str
    source: str
    source_url: str
    query: dict[str, Any]
    retrieved_at: str
    status: int
    sha256: str
    body: str
    frozen: bool


class SnapshotStore:
    def __init__(self, path: str | Path = ":memory:", read_only: bool = False):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(self.path, read_only=read_only)
        self._lock = threading.Lock()
        if not read_only:
            self._con.execute(SCHEMA)

    def get(self, key: str) -> StoredResponse | None:
        with self._lock:
            row = self._con.execute(
                "SELECT key,source,source_url,query_json,retrieved_at,status,sha256,body,frozen "
                "FROM responses WHERE key=?", [key]
            ).fetchone()
        if not row:
            return None
        return StoredResponse(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5], row[6], row[7], row[8])

    def get_by_hash(self, sha256: str) -> StoredResponse | None:
        with self._lock:
            row = self._con.execute(
                "SELECT key,source,source_url,query_json,retrieved_at,status,sha256,body,frozen "
                "FROM responses WHERE sha256=? LIMIT 1", [sha256]
            ).fetchone()
        if not row:
            return None
        return StoredResponse(row[0], row[1], row[2], json.loads(row[3]), row[4], row[5], row[6], row[7], row[8])

    def put(self, r: StoredResponse) -> None:
        with self._lock:
            self._con.execute(
                "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?,?,?)",
                [r.key, r.source, r.source_url, json.dumps(r.query, default=str), r.retrieved_at,
                 r.status, r.sha256, r.body, r.frozen],
            )

    def freeze_all(self) -> int:
        with self._lock:
            self._con.execute("UPDATE responses SET frozen = TRUE")
            return self._con.execute("SELECT count(*) FROM responses").fetchone()[0]

    def count(self) -> int:
        with self._lock:
            return self._con.execute("SELECT count(*) FROM responses").fetchone()[0]

    def close(self) -> None:
        self._con.close()
