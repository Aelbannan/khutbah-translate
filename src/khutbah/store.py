"""Session storage: every event is appended to SQLite, so a transcript can be
rebuilt (or a late viewer caught up) from the log."""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class Store:
    def __init__(self, path: Path = Path("data/sessions.db")):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY, title TEXT, started REAL, ended REAL);
            CREATE TABLE IF NOT EXISTS event (
                session INTEGER, seq INTEGER, type TEXT, data TEXT, PRIMARY KEY (session, seq));
            """
        )

    def start(self, title: str) -> int:
        with self.lock, self.db:
            return self.db.execute("INSERT INTO session (title, started) VALUES (?, ?)",
                                   (title, time.time())).lastrowid

    def end(self, session: int) -> None:
        with self.lock, self.db:
            self.db.execute("UPDATE session SET ended=? WHERE id=?", (time.time(), session))

    def add(self, session: int, seq: int, ev: dict[str, Any]) -> None:
        with self.lock, self.db:
            self.db.execute("INSERT INTO event VALUES (?, ?, ?, ?)",
                            (session, seq, ev["type"], json.dumps(ev, ensure_ascii=False)))

    def sessions(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.db.execute("SELECT id, title, started, ended FROM session ORDER BY id DESC").fetchall()
        return [dict(zip(("id", "title", "started", "ended"), r)) for r in rows]

    def session(self, session: int) -> dict[str, Any] | None:
        return next((s for s in self.sessions() if s["id"] == session), None)

    def events(self, session: int, types: tuple[str, ...] = ()) -> list[dict[str, Any]]:
        q = "SELECT data FROM event WHERE session=?"
        args: list[Any] = [session]
        if types:
            q += f" AND type IN ({','.join('?' * len(types))})"
            args += types
        with self.lock:
            rows = self.db.execute(q + " ORDER BY seq", args).fetchall()
        return [json.loads(r[0]) for r in rows]

    def transcript(self, session: int) -> str:
        """Plain-text transcript: each passage's source and translations, quote cards inline."""
        s = self.session(session)
        lines = [f"# {s['title']}" if s else f"# session {session}",
                 time.strftime("%Y-%m-%d %H:%M", time.localtime(s["started"])) if s else "", ""]
        evs = self.events(session, ("passage", "card"))
        cards: dict[int, list[str]] = {}  # segment -> new cards, listed under their passage
        for ev in evs:
            if ev["type"] == "card" and ev["new"]:
                cards.setdefault(ev["segment"], []).append(f"{ev['kind'].upper()} {ev['ref']}")
        for ev in evs:
            if ev["type"] != "passage":
                continue
            ts = time.strftime("%H:%M:%S", time.gmtime(ev.get("start", 0)))
            lines.append(f"[{ts}] {ev['source']}")
            for lang in ("en", "zh"):
                if ev.get(lang):
                    lines.append(f"  {lang}: {ev[lang]}")
            lines += [f"  ▸ {c}" for sid in ev["segments"] for c in cards.get(sid, [])]
            lines.append("")
        return "\n".join(lines)
