"""Quran lookup and fuzzy matching of (noisy) STT transcripts to ayat."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from khutbah.arabic import normalize
from khutbah.matcher import NgramIndex, local_align


@dataclass
class Ayah:
    surah: int
    ayah: int
    uthmani: str
    norm: str
    en: str
    zh: str


# Dhikr and du'a formulas that are also Quranic phrases. Said on their own they're
# devotional formulas, not quotations, so they don't get an ayah card.
FORMULAS = [
    normalize(f)
    for f in [
        "لا إله إلا أنت",
        "لا إله إلا هو",
        "لا إله إلا الله",
        "سبحانك اللهم وبحمدك",
        "إنا لله وإنا إليه راجعون",
        "حسبنا الله ونعم الوكيل",
        "الحمد لله رب العالمين",
        "بسم الله الرحمن الرحيم",
        "سبحان الله",
        "رضي الله عنهم ورضوا عنه",
        "أستغفر الله",
        "ما شاء الله",
        "إن شاء الله",
        "لا حول ولا قوة إلا بالله",
    ]
]


def _is_formula(words: list[str]) -> bool:
    """True if the aligned words are essentially one of FORMULAS (allowing one extra word)."""
    text = " ".join(words)
    return any(
        len(words) <= len(f.split()) + 1 and fuzz.partial_ratio(f, text) >= 85
        for f in FORMULAS
    )


@dataclass
class Span:
    """A run of query words aligned to consecutive ayat."""

    q_start: int  # word range in the normalised query (end exclusive)
    q_end: int
    surah: int
    first: int  # ayah range covered by the aligned words
    last: int
    matched: int  # query words that aligned to a similar Quran word
    score: float  # local-alignment score
    generic: bool  # phrase is common to many ayat (e.g. "رضي الله عنه")
    continues: bool = False  # follows on from the previously matched ayah

    def is_quran(self, min_words: int = 4) -> bool:
        if self.continues:
            return self.matched >= 3
        return self.matched >= min_words and not self.generic

    @property
    def ref(self) -> str:
        return f"{self.surah}:{self.first}" + (f"-{self.last}" if self.last != self.first else "")


class Quran:
    def __init__(self, db_path: Path = Path("data/quran.db")):
        db = sqlite3.connect(db_path)
        self.ayat: list[Ayah] = [
            Ayah(*row)
            for row in db.execute(
                "SELECT surah, ayah, text_uthmani, text_norm, en, zh FROM ayah ORDER BY surah, ayah"
            )
        ]
        self.surah_names = {
            n: {"ar": ar, "simple": simple, "en": en, "zh": zh}
            for n, ar, simple, en, zh in db.execute(
                "SELECT number, name_ar, name_simple, name_en, name_zh FROM surah"
            )
        }
        self.attribution = db.execute("SELECT value FROM meta WHERE key='attribution'").fetchone()[0]
        self.index_of = {(a.surah, a.ayah): i for i, a in enumerate(self.ayat)}

        self.index = NgramIndex.build([a.norm for a in self.ayat])

    def get(self, surah: int, ayah: int) -> Ayah:
        return self.ayat[self.index_of[(surah, ayah)]]

    def window(self, m: Span) -> list[Ayah]:
        return [self.get(m.surah, a) for a in range(m.first, m.last + 1)]

    def _windows(self, cands: list[int]) -> list[tuple[int, int]]:
        """Merge candidates (plus one neighbour each side) into ayah-index ranges."""
        ranges: list[tuple[int, int]] = []
        for i in sorted(cands):
            lo, hi = max(0, i - 1), min(len(self.ayat) - 1, i + 1)
            while self.ayat[lo].surah != self.ayat[i].surah:
                lo += 1
            while self.ayat[hi].surah != self.ayat[i].surah:
                hi -= 1
            if ranges and lo <= ranges[-1][1] + 1 and self.ayat[lo].surah == self.ayat[ranges[-1][1]].surah:
                ranges[-1] = (ranges[-1][0], max(hi, ranges[-1][1]))
            else:
                ranges.append((lo, hi))
        return ranges

    def find(
        self, text: str, k: int = 12, prefer_after: tuple[int, int] | None = None, max_spans: int = 3
    ) -> list[Span]:
        """Quran spans inside `text`, in query order.

        `prefer_after` is the last ayah previously matched: a span continuing from
        it is accepted with less evidence (recitation split across VAD segments).
        """
        q = normalize(text).split()
        if len(q) < 2:
            return []
        windows = self._windows(self.index.candidates(" ".join(q), k))
        results: list[tuple[float, int, int, int, int, int, int]] = []
        for lo, hi in windows:
            ref, ref_ayah = [], []
            for idx in range(lo, hi + 1):
                for w in self.ayat[idx].norm.split():
                    ref.append(w)
                    ref_ayah.append(idx)
            sim = process.cdist(q, ref, scorer=fuzz.ratio, dtype=np.uint8) / 100.0
            results.append((lo, hi, sim, ref_ayah))  # type: ignore[arg-type]

        spans: list[Span] = []
        used = np.zeros(len(q), dtype=bool)
        for _ in range(max_spans):
            scored = []
            for lo, hi, sim, ref_ayah in results:
                sim = sim.copy()
                sim[used] = 0.0
                al = local_align(sim)
                if al:
                    scored.append((al, ref_ayah))
            if not scored:
                break
            scored.sort(key=lambda x: -x[0][0])
            (score, qs, qe, rs, re_, matched), ref_ayah = scored[0]
            if matched < 3:
                break
            # How many other, unrelated places align about as well?
            best_ayat = set(ref_ayah[rs:re_])
            rivals = sum(
                1
                for (sc, *_), ra in scored[1:]
                if sc >= 0.9 * score and not (set(ra) & best_ayat)
            )
            a0, a1 = self.ayat[ref_ayah[rs]], self.ayat[ref_ayah[re_ - 1]]
            continues = prefer_after is not None and (
                (a0.surah, a0.ayah - 1) == prefer_after or (a0.surah, a0.ayah) == prefer_after
            )
            spans.append(
                Span(qs, qe, a0.surah, a0.ayah, a1.ayah, matched, score,
                     generic=(rivals >= 2 and matched < 7) or _is_formula(q[qs:qe]),
                     continues=continues)
            )
            used[qs:qe] = True
            if spans[-1].is_quran():
                prefer_after = (a1.surah, a1.ayah)
        return sorted(spans, key=lambda s: s.q_start)

