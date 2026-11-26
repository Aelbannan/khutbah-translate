"""HTTP/WebSocket server.

    WS  /ws/ingest?title=...   audio source: binary frames of 16 kHz mono PCM16-LE.
                               Send the text frame {"type": "end"} (or close) to finish.
                               Receives the same events as viewers.
    WS  /ws/view               viewers (TV, phones): JSON events; on connect, a replay
                               of the current session's recent events.
    GET /                      caption viewer (?lang=en|zh|both, ?tv=1 for large text)
    GET /ingest                browser-microphone source (localhost or HTTPS only)
    GET /api/sessions          stored sessions
    GET /api/sessions/{id}     a session's events (JSON)
    GET /api/sessions/{id}/transcript.txt

Run: uv run khutbah-server  (env: KHUTBAH_STT, KHUTBAH_ALT_STT, KHUTBAH_TRANSLATE,
KHUTBAH_LLM_MODEL, KHUTBAH_HOST, KHUTBAH_PORT)
"""

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from khutbah.live import Event, LiveSession, Models
from khutbah.store import Store

log = logging.getLogger("khutbah")
WEB = Path(__file__).parent / "web"
REPLAY = 30  # recent passages/cards sent to a viewer that joins mid-session


def load_models() -> Models:
    from khutbah.hadith import HadithDB
    from khutbah.pipeline import QuoteFinder
    from khutbah.quran import Quran
    from khutbah.stt import BACKENDS

    threads = int(os.environ.get("KHUTBAH_STT_THREADS", "4"))
    stt = BACKENDS[os.environ.get("KHUTBAH_STT", "qwen3")](threads=threads)
    alt_name = os.environ.get("KHUTBAH_ALT_STT", "omnilingual")
    alt = BACKENDS[alt_name](threads=threads) if alt_name else None
    factory = None
    if os.environ.get("KHUTBAH_TRANSLATE", "1") != "0":
        from khutbah.translate import Translator

        factory = Translator
    return Models(stt, alt, QuoteFinder(Quran(), HadithDB()), factory)


class Hub:
    """The live session (at most one) plus connected viewers."""

    def __init__(self, models: Models, store: Store):
        self.models, self.store = models, store
        self.viewers: set[WebSocket] = set()
        self.session_id: int | None = None
        self.live: LiveSession | None = None
        self.seq = 0
        self.lock = threading.Lock()  # emit() is called from several pipeline threads
        self.loop: asyncio.AbstractEventLoop | None = None

    def emit(self, ev: Event) -> None:
        """Called from pipeline threads."""
        if self.session_id is None or self.loop is None:
            return
        ev = ev | {"session": self.session_id}
        if ev["type"] != "status":
            with self.lock:
                self.seq += 1
                self.store.add(self.session_id, self.seq, ev)
        asyncio.run_coroutine_threadsafe(self.broadcast(ev), self.loop)

    async def broadcast(self, ev: dict[str, Any]) -> None:
        msg = json.dumps(ev, ensure_ascii=False)
        for ws in list(self.viewers):
            try:
                await ws.send_text(msg)
            except Exception:
                self.viewers.discard(ws)

    def replay(self) -> list[dict[str, Any]]:
        if self.session_id is None:
            return []
        s = self.store.session(self.session_id)
        head = {"type": "session", "session": self.session_id, "title": s["title"],
                "state": "ended" if s["ended"] else "started"}
        return [head] + self.store.events(self.session_id, ("passage", "card"))[-REPLAY:]


def create_app(models: Models | None = None, store: Store | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.hub = Hub(models or await asyncio.to_thread(load_models), store or Store())
        app.state.hub.loop = asyncio.get_running_loop()
        yield

    app = FastAPI(title="khutbah-translate", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    async def viewer():
        return FileResponse(WEB / "viewer.html")

    @app.get("/ingest")
    async def ingest_page():
        return FileResponse(WEB / "ingest.html")

    @app.get("/api/sessions")
    async def sessions():
        return app.state.hub.store.sessions()

    @app.get("/api/sessions/{sid}")
    async def session(sid: int):
        store = app.state.hub.store
        if not (s := store.session(sid)):
            raise HTTPException(404)
        return s | {"events": store.events(sid)}

    @app.get("/api/sessions/{sid}/transcript.txt", response_class=PlainTextResponse)
    async def transcript(sid: int):
        if not app.state.hub.store.session(sid):
            raise HTTPException(404)
        return app.state.hub.store.transcript(sid)

    @app.websocket("/ws/view")
    async def view(ws: WebSocket):
        hub: Hub = app.state.hub
        await ws.accept()
        for ev in hub.replay():
            await ws.send_text(json.dumps(ev | {"replay": True}, ensure_ascii=False))
        hub.viewers.add(ws)
        try:
            while True:
                await ws.receive_text()  # viewers don't send anything meaningful
        except WebSocketDisconnect:
            pass
        finally:
            hub.viewers.discard(ws)

    @app.websocket("/ws/ingest")
    async def ingest(ws: WebSocket, title: str = "Khutbah"):
        hub: Hub = app.state.hub
        await ws.accept()
        if hub.live is not None:
            await ws.close(code=4409, reason="a live session is already running")
            return
        hub.session_id, hub.seq = hub.store.start(title), 0
        hub.live = LiveSession(hub.models, hub.emit)
        hub.viewers.add(ws)  # the source sees results too
        await hub.broadcast({"type": "session", "state": "started", "session": hub.session_id, "title": title})
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    hub.live.feed_pcm16(msg["bytes"])
                elif msg.get("text") and json.loads(msg["text"]).get("type") == "end":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            live, sid = hub.live, hub.session_id
            await asyncio.to_thread(live.close)  # drain: remaining segments + translations
            hub.store.end(sid)
            await hub.broadcast({"type": "session", "state": "ended", "session": sid})
            hub.viewers.discard(ws)
            hub.live = None
            try:
                await ws.close()
            except Exception:
                pass

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(create_app(), host=os.environ.get("KHUTBAH_HOST", "0.0.0.0"),
                port=int(os.environ.get("KHUTBAH_PORT", "8000")))
