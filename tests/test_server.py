"""End-to-end through the WebSocket API with fake STT/translation (real VAD + matchers)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from khutbah.hadith import HadithDB
from khutbah.live import Models
from khutbah.pipeline import QuoteFinder
from khutbah.quran import Quran
from khutbah.server import create_app
from khutbah.store import Store
from khutbah.stt import Transcript
from khutbah.translate import Translation

SEGMENTS = Path("data/segments/khutbah-2025-12-12")
TEXTS = ["عباد الله", "اتقوا الله حق تقاته ولا تموتن إلا وأنتم مسلمون", "This is the speaker's English."]


class FakeSTT:
    def __init__(self):
        self.i = 0

    def transcribe(self, samples):
        t = TEXTS[self.i % len(TEXTS)]
        self.i += 1
        return Transcript(t, "ar" if self.i % 3 else "en", 0.01)


class FakeTranslator:
    def translate(self, text, targets):
        return Translation({t: f"<{t}> {text}" for t in targets}, 0.01, "fake")


@pytest.fixture(scope="module")
def finder():
    return QuoteFinder(Quran(), HadithDB())


@pytest.mark.skipif(not SEGMENTS.exists(), reason="needs segmented recording")
def test_ingest_to_viewer(tmp_path, finder):
    # Real speech for the VAD: three segments with silence between them.
    clips = [sf.read(SEGMENTS / f"{i:04d}.wav", dtype="float32")[0] for i in (4, 5, 6)]
    silence = np.zeros(16000 * 3, dtype=np.float32)
    audio = np.concatenate([x for c in clips for x in (c, silence)])
    pcm = (audio * 32767).astype("<i2").tobytes()

    models = Models(FakeSTT(), None, finder, FakeTranslator)
    store = Store(tmp_path / "s.db")
    with TestClient(create_app(models, store)) as client:
        with client.websocket_connect("/ws/view") as viewer:
            with client.websocket_connect("/ws/ingest?title=test") as src:
                for i in range(0, len(pcm), 3200):
                    src.send_bytes(pcm[i : i + 3200])
                src.send_text(json.dumps({"type": "end"}))
                events = []
                while True:
                    ev = json.loads(src.receive_text())
                    events.append(ev)
                    if ev["type"] == "session" and ev["state"] == "ended":
                        break
            seen = [json.loads(viewer.receive_text())["type"] for _ in range(3)]
        assert seen[0] == "session"

        types = [e["type"] for e in events]
        assert 3 <= types.count("segment") <= 5, types  # VAD may split a clip at a pause
        cards = [e for e in events if e["type"] == "card"]
        assert [c["ref"] for c in cards] == ["3:102"]
        assert cards[0]["ayat"][0]["en"] and cards[0]["ayat"][0]["zh"]
        passages = [e for e in events if e["type"] == "passage"]
        assert passages and "[QURAN 3:102]" in passages[0]["source"]
        assert passages[0]["zh"].startswith("<zh>")

        sid = events[0]["session"]
        assert client.get("/api/sessions").json()[0]["ended"]
        assert "QURAN 3:102" in client.get(f"/api/sessions/{sid}/transcript.txt").text
        assert client.get("/").status_code == 200


def test_second_source_rejected(tmp_path, finder):
    models = Models(FakeSTT(), None, finder, None)
    with TestClient(create_app(models, Store(tmp_path / "s.db"))) as client:
        with client.websocket_connect("/ws/ingest") as a:
            with client.websocket_connect("/ws/ingest") as b:
                with pytest.raises(Exception):
                    b.receive_text()
            a.send_text(json.dumps({"type": "end"}))
