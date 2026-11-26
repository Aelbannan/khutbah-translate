"""Turns STT segments into caption events: quote cards and translatable passages.

Per segment:
  1. Find Quran spans, then hadith spans in the remaining words.
  2. Replace each quote in the text with a placeholder ([QURAN 3:102], [HADITH Sahih Muslim 2937])
     so the translator leaves it alone; the card shows the established text instead.
  3. Buffer segments into passages (sentence-ish units) for translation; short
     VAD fragments translate badly on their own.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from khutbah.arabic import normalize
from khutbah.hadith import MIN_WEIGHT, HadithDB, HadithSpan
from khutbah.quran import Quran, Span



@dataclass
class Card:
    kind: str  # "quran" | "hadith"
    ref: str
    segment: int
    quran: Span | None = None
    hadith: HadithSpan | None = None
    new: bool = True  # False if this quote was already shown recently (no new card needed)


@dataclass
class Passage:
    segments: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    start: float = 0.0
    end: float = 0.0

    @property
    def text(self) -> str:
        return " ".join(t for t in self.texts if t)

    @property
    def words(self) -> int:
        return len(self.text.split())


def _replace_spans(text: str, spans: list[tuple[int, int, str]]) -> str:
    """Replace normalised-word ranges with placeholders in the original text.

    Each whitespace token is normalised on its own to learn which normalised
    word indices it produced, so punctuation-only or Latin tokens map correctly.
    """
    if not spans:
        return text
    tokens = re.split(r"(\s+)", text)
    word_to_token: list[int] = []
    for i, t in enumerate(tokens):
        word_to_token += [i] * len(normalize(t).split())
    out = list(tokens)
    for qs, qe, label in sorted(spans, reverse=True):
        if qe > len(word_to_token):
            continue
        first, last = word_to_token[qs], word_to_token[qe - 1]
        out[first : last + 1] = [label]
    return "".join(out)


# Spoken mentions of a collection ("Abu Dawud in his Sunan", "رواه البخاري").
COLLECTION_CUES = {
    "bukhari": re.compile(r"bukh[aā]r|البخاري", re.I),
    "muslim": re.compile(r"sahih muslim|imam muslim|narrated by muslim|صحيح مسلم|رواه مسلم|الإمام مسلم", re.I),
    "abudawud": re.compile(r"ab[uoū]\s*d[aā]w[uū]d|ab[uo]\s*dawood|أبو داود|ابو داود", re.I),
    "tirmidhi": re.compile(r"tirmi[dz]h?i|الترمذي", re.I),
    "nasai": re.compile(r"nas[aā]['’]?i|النسائي", re.I),
    "ibnmajah": re.compile(r"ibn\s*m[aā]j[ae]h?|ابن ماجه", re.I),
    "malik": re.compile(r"muwatt?a|الموطأ", re.I),
    "nawawi": re.compile(r"nawawi|forty hadith|النووي|الأربعين النووية", re.I),
}
EVIDENCE_WINDOW = 90.0  # seconds over which partial hadith evidence adds up
CUE_BOOST = 1.5  # evidence multiplier when the hadith's collection was just named
REPEAT_WINDOW = 600.0  # a quote gets a new card at most once per 10 minutes


class QuoteFinder:
    def __init__(self, quran: Quran | None = None, hadith: HadithDB | None = None):
        self.quran = quran or Quran()
        self.hadith = hadith or HadithDB()
        self.prev_ayah: tuple[int, int] | None = None
        # hadith id -> {segment: (time, weight)}; max weight per segment so two STT
        # hypotheses of the same segment don't double-count.
        self.evidence: dict[int, dict[int, tuple[float, float]]] = {}
        self.cues: list[tuple[float, str]] = []
        self.shown: dict[str, float] = {}

    def reset(self) -> None:
        self.__init__(self.quran, self.hadith)

    def process(
        self, seg_id: int, text: str, alt_text: str | None = None, t: float = 0.0
    ) -> tuple[str, list[Card]]:
        """Returns (text with placeholders, cards) for one segment starting at time `t`.

        `alt_text` is a second STT hypothesis (e.g. Omnilingual, better on recited
        Arabic): quotes found only there still produce a card and are mentioned to
        the translator, but can't be placeholdered into `text`.
        """
        self._note_cues(t, f"{text} {alt_text or ''}")
        spans = self._find(seg_id, text, t)
        cards = [c for _, c in spans]
        out = _replace_spans(text, [(qs, qe, f"[{c.kind.upper()} {c.ref}]") for (qs, qe), c in spans])
        if alt_text:
            seen = {c.ref for c in cards}
            extra = [c for _, c in self._find(seg_id, alt_text, t) if c.ref not in seen]
            cards += extra
            if extra:
                out += " " + " ".join(f"[{c.kind.upper()} {c.ref} quoted here]" for c in extra)
        for c in cards:
            c.new = t - self.shown.get(c.ref, -1e9) > REPEAT_WINDOW
            if c.new:
                self.shown[c.ref] = t
        return out, cards

    def _note_cues(self, t: float, text: str) -> None:
        self.cues = [(ct, c) for ct, c in self.cues if t - ct <= EVIDENCE_WINDOW]
        self.cues += [(t, key) for key, rx in COLLECTION_CUES.items() if rx.search(text)]

    def _hadith_confident(self, hs: HadithSpan, seg_id: int, t: float) -> bool:
        if hs.is_hadith():
            return True
        ev = self.evidence.setdefault(hs.hadith.id, {})
        old = ev.get(seg_id, (t, 0.0))[1]
        ev[seg_id] = (t, max(old, hs.weight))
        for sid in [sid for sid, (et, _) in ev.items() if t - et > EVIDENCE_WINDOW]:
            del ev[sid]
        total = sum(w for _, w in ev.values())
        if any(c == hs.hadith.collection for _, c in self.cues):
            total *= CUE_BOOST
        return total >= MIN_WEIGHT

    def _find(self, seg_id: int, text: str, t: float) -> list[tuple[tuple[int, int], Card]]:
        q = normalize(text).split()
        used = np.zeros(len(q), dtype=bool)
        found: list[tuple[tuple[int, int], Card]] = []
        for sp in self.quran.find(text, prefer_after=self.prev_ayah):
            if sp.is_quran():
                found.append(((sp.q_start, sp.q_end), Card("quran", sp.ref, seg_id, quran=sp)))
                used[sp.q_start : sp.q_end] = True
                self.prev_ayah = (sp.surah, sp.last)
        for hs in self.hadith.find(text, skip=used, partial=True):
            if self._hadith_confident(hs, seg_id, t):
                found.append(((hs.q_start, hs.q_end), Card("hadith", hs.hadith.citation, seg_id, hadith=hs)))
        return sorted(found, key=lambda f: f[0][0])


@dataclass
class Batcher:
    """Groups segments into passages: flush on a long pause, a sentence end once
    the passage is long enough, or a hard word cap."""

    min_words: int = 18
    max_words: int = 45
    max_gap: float = 2.0
    current: Passage = field(default_factory=Passage)

    def add(self, seg_id: int, start: float, end: float, text: str) -> list[Passage]:
        out: list[Passage] = []
        if self.current.segments and start - self.current.end > self.max_gap:
            out.append(self.flush())  # type: ignore[arg-type]
        p = self.current
        if not p.segments:
            p.start = start
        p.segments.append(seg_id)
        p.texts.append(text)
        p.end = end
        ends_sentence = text.rstrip().endswith((".", "?", "!", "؟", "。"))
        if p.words >= self.max_words or (p.words >= self.min_words and ends_sentence):
            out.append(self.flush())  # type: ignore[arg-type]
        return out

    def flush(self) -> Passage | None:
        if not self.current.segments:
            return None
        p, self.current = self.current, Passage()
        return p
