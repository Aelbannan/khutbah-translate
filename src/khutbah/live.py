"""Live pipeline: PCM audio in, caption events out.

    audio ─▶ [VAD thread] ─segments─▶ [STT thread] ─passages─▶ [translate thread]
                                      │  quote cards              │
                                      ▼                           ▼
                                   emit(event)                 emit(event)

Each stage is its own thread so a slow stage never stalls audio intake. STT
and the matchers release the GIL in native code; translation is network-bound.
Events are plain JSON-able dicts:

    {"type": "segment", "id", "start", "end", "lang", "text", "alt_text", "stt_seconds"}
    {"type": "card", "segment", "kind": "quran"|"hadith", "ref", "new", ...content}
    {"type": "passage", "id", "segments", "start", "end", "source", "en", "zh", "model", "seconds"}
    {"type": "status", "lag": seconds behind live, "queued": segments waiting}
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from khutbah.pipeline import Batcher, Card, Passage, QuoteFinder
from khutbah.quran import Quran
from khutbah.stt import Backend
from khutbah.vad import SAMPLE_RATE, Segment, Segmenter, VadConfig

log = logging.getLogger(__name__)
Event = dict[str, Any]
_STOP = object()


@dataclass
class Models:
    """Loaded once per process and shared by sessions (one live session at a time)."""

    stt: Backend
    alt_stt: Backend | None  # second hypothesis for quote finding (Omnilingual)
    finder: QuoteFinder
    translator_factory: Callable[[], Any] | None  # -> object with .translate(text, targets)


@dataclass
class LiveSession:
    models: Models
    emit: Callable[[Event], None]
    targets: list[str] = field(default_factory=lambda: ["en", "zh"])
    vad: VadConfig = field(default_factory=VadConfig)

    def __post_init__(self):
        self.models.finder.reset()
        self.segmenter = Segmenter(self.vad)
        self.batcher = Batcher()
        self.translator = self.models.translator_factory() if self.models.translator_factory else None
        self.audio_q: queue.Queue = queue.Queue()
        self.seg_q: queue.Queue = queue.Queue()
        self.pass_q: queue.Queue = queue.Queue()
        self.received = 0  # samples received
        self.seg_id = self.passage_id = 0
        self.threads = [
            threading.Thread(target=self._vad_loop, name="vad", daemon=True),
            threading.Thread(target=self._stt_loop, name="stt", daemon=True),
            threading.Thread(target=self._translate_loop, name="translate", daemon=True),
        ]
        for t in self.threads:
            t.start()

    # -- input -------------------------------------------------------------
    def feed_pcm16(self, data: bytes) -> None:
        """16 kHz mono signed 16-bit little-endian PCM, any chunk size."""
        audio = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        self.received += len(audio)
        self.audio_q.put(audio)

    def close(self, wait: bool = True) -> None:
        """End of stream: flush everything through the pipeline."""
        self.audio_q.put(_STOP)
        if wait:
            for t in self.threads:
                t.join()

    @property
    def now(self) -> float:
        return self.received / SAMPLE_RATE

    # -- stages ------------------------------------------------------------
    def _safe_emit(self, ev: Event) -> None:
        try:
            self.emit(ev)
        except Exception:
            log.exception("emit failed")

    def _vad_loop(self) -> None:
        while (audio := self.audio_q.get()) is not _STOP:
            for seg in self.segmenter.feed(audio):
                self.seg_q.put(seg)
        for seg in self.segmenter.flush():
            self.seg_q.put(seg)
        self.seg_q.put(_STOP)

    def _stt_loop(self) -> None:
        while (seg := self.seg_q.get()) is not _STOP:
            try:
                self._process_segment(seg)
            except Exception:
                log.exception("segment failed")
        if p := self.batcher.flush():
            self.pass_q.put(p)
        self.pass_q.put(_STOP)

    def _process_segment(self, seg: Segment) -> None:
        sid = self.seg_id
        self.seg_id += 1
        tr = self.models.stt.transcribe(seg.samples)
        alt = self.models.alt_stt.transcribe(seg.samples) if self.models.alt_stt else None
        text, cards = self.models.finder.process(sid, tr.text, alt.text if alt else None, t=seg.start)
        self._safe_emit({
            "type": "segment", "id": sid, "start": round(seg.start, 2), "end": round(seg.end, 2),
            "lang": tr.lang, "text": tr.text, "alt_text": alt.text if alt else None,
            "marked": text, "stt_seconds": round(tr.seconds + (alt.seconds if alt else 0), 2),
        })
        for c in cards:
            self._safe_emit(card_event(c, self.models.finder.quran))
        for p in self.batcher.add(sid, seg.start, seg.end, text):
            self.pass_q.put(p)
        self._safe_emit({"type": "status", "lag": round(max(0.0, self.now - seg.end), 1),
                         "queued": self.seg_q.qsize()})

    def _translate_loop(self) -> None:
        while (p := self.pass_q.get()) is not _STOP:
            self._translate(p)

    def _translate(self, p: Passage) -> None:
        pid = self.passage_id
        self.passage_id += 1
        ev: Event = {"type": "passage", "id": pid, "segments": p.segments, "start": round(p.start, 2),
                     "end": round(p.end, 2), "source": p.text}
        if self.translator is None:
            self._safe_emit(ev | {t: None for t in self.targets})
            return
        t0 = time.perf_counter()
        try:
            r = self.translator.translate(p.text, self.targets)
            ev |= r.out | {"model": r.model, "seconds": round(r.seconds, 2)}
        except Exception as e:  # one failed passage must not end the session
            log.warning("translation failed: %s", e)
            ev |= {t: None for t in self.targets} | {"error": str(e)[:200],
                                                     "seconds": round(time.perf_counter() - t0, 2)}
        self._safe_emit(ev)


def card_event(c: Card, quran: Quran) -> Event:
    ev: Event = {"type": "card", "segment": c.segment, "kind": c.kind, "ref": c.ref, "new": c.new}
    if c.quran:
        s = c.quran
        name = quran.surah_names[s.surah]
        ev |= {
            "surah": s.surah, "surah_name": name,
            "ayat": [{"ayah": a.ayah, "ar": a.uthmani, "en": a.en, "zh": a.zh}
                     for a in quran.window(s)],
        }
    elif c.hadith:
        h = c.hadith.hadith
        ev |= {"collection": h.collection, "citation": h.citation, "grade": h.grade,
               "ar": h.ar, "en": h.en, "zh": None}  # zh: machine translation, TODO
    return ev
