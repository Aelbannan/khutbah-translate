"""Finding quoted hadith in (noisy) STT transcripts.

Unlike the Quran, hadith texts share lots of stock wording ("the Messenger of
Allah ﷺ said", narrator chains), so a match only counts if the aligned words
are *distinctive*: the sum of their corpus IDF must clear a threshold.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz, process

from khutbah.arabic import normalize
from khutbah.matcher import MATCH_SIM, NgramIndex, crop, local_align

MIN_WEIGHT = 25.0  # summed IDF of matched words; real quotes scored 31-87, stock phrases <25
MIN_WORDS = 4
# Weaker evidence still worth keeping: a hadith quoted in pieces across segments
# (with explanation in between) is accepted once the pieces add up (see pipeline).
PARTIAL_WEIGHT = 8.0
PARTIAL_WORDS = 3
TIE = 0.92  # candidates scoring within this fraction of the best are "the same hadith"


# Fixed khutbah/du'a formulas that are technically hadith (e.g. Khutbat al-Hajah)
# but recited every week; a hadith card for them would be noise.
FORMULAS = normalize(" ".join([
    "إن الحمد لله نحمده ونستعينه ونستغفره ونتوب إليه",
    "ونعوذ بالله من شرور أنفسنا ومن سيئات أعمالنا",
    "من يهده الله فلا مضل له ومن يضلل فلا هادي له",
    "وأشهد أن لا إله إلا الله وحده لا شريك له وأشهد أن محمدا عبده ورسوله",
    "أما بعد فإن أصدق الحديث كتاب الله وخير الهدي هدي محمد صلى الله عليه وسلم",
    "وشر الأمور محدثاتها وكل محدثة بدعة وكل بدعة ضلالة وكل ضلالة في النار",
    "سبحانك اللهم وبحمدك أشهد أن لا إله إلا أنت أستغفرك وأتوب إليك",
    "أقول قولي هذا وأستغفر الله لي ولكم",
    "اللهم صل على محمد وعلى آل محمد كما صليت على إبراهيم وعلى آل إبراهيم إنك حميد مجيد",
])).split()


def _mostly_formula(words: list[str]) -> bool:
    """True if >=80% of `words` align to the stock formulas above.

    Checked on both the heard words and the matched reference words: garbled
    speech may miss the formula while the clean reference text still shows it."""
    sim = process.cdist(words, FORMULAS, scorer=fuzz.ratio, dtype=np.uint8) / 100.0
    al = local_align(sim)
    return bool(al) and al[5] >= 0.8 * len(words)


@dataclass
class Hadith:
    id: int
    collection: str
    collection_name: str
    priority: int
    number: int
    ref: str
    ar: str
    norm: str
    en: str
    grade: str

    @property
    def citation(self) -> str:
        return f"{self.collection_name} {self.ref or self.number}"


@dataclass
class HadithSpan:
    q_start: int
    q_end: int
    hadith: Hadith
    matched: int
    weight: float  # summed IDF of the matched words
    score: float
    r_start: int  # aligned word range inside the hadith's normalised text
    r_end: int

    formula: bool = False

    def is_hadith(self) -> bool:
        return self.matched >= MIN_WORDS and self.weight >= MIN_WEIGHT and not self.formula


class HadithDB:
    def __init__(self, db_path: Path = Path("data/hadith.db")):
        db = sqlite3.connect(db_path)
        self.hadith = [Hadith(*r) for r in db.execute(
            "SELECT id, collection, collection_name, priority, number, ref, ar, norm, en, grade"
            " FROM hadith ORDER BY id")]
        self.idf = dict(db.execute("SELECT word, idf FROM word_idf"))
        self.attribution = db.execute("SELECT value FROM meta WHERE key='attribution'").fetchone()[0]
        self.words = [h.norm.split() for h in self.hadith]
        self.index = NgramIndex.cached([h.norm for h in self.hadith], db_path.with_suffix(".index.npz"))

    def find(
        self, text: str, skip: np.ndarray | None = None, k: int = 20, partial: bool = False
    ) -> list[HadithSpan]:
        """Hadith quoted inside `text`. `skip` masks query words already claimed (e.g. Quran).

        With `partial=True`, also returns weaker spans (check `is_hadith()`), for
        accumulating evidence across segments.
        """
        min_words = PARTIAL_WORDS if partial else MIN_WORDS
        q = normalize(text).split()
        if len(q) < min_words:
            return []
        used = np.zeros(len(q), dtype=bool) if skip is None else skip.copy()
        if (~used).sum() < min_words:
            return []
        query = " ".join(w for w, u in zip(q, used) if not u)
        cands = self.index.candidates(query, k)
        spans: list[HadithSpan] = []
        for _ in range(2):  # at most two separate hadith per segment
            scored = []
            for c in cands:
                sim = process.cdist(q, self.words[c], scorer=fuzz.ratio, dtype=np.uint8) / 100.0
                sim[used] = 0.0
                sim, off = crop(sim)
                al = local_align(sim)
                if al:
                    score, qs, qe, rs, re_, matched = al
                    scored.append((score, c, qs, qe, rs + off, re_ + off, matched))
            if not scored:
                break
            best = max(s[0] for s in scored)
            # Same wording in several books: prefer the best-known collection.
            score, c, qs, qe, rs, re_, matched = min(
                (s for s in scored if s[0] >= TIE * best), key=lambda s: (self.hadith[s[1]].priority, -s[0])
            )
            weight = self._weight(q[qs:qe], self.words[c][rs:re_])
            span = HadithSpan(qs, qe, self.hadith[c], matched, weight, score, rs, re_,
                              formula=_mostly_formula(q[qs:qe]) or _mostly_formula(self.words[c][rs:re_]))
            used[qs:qe] = True
            if span.formula:
                continue  # skip it, but keep looking for a real quote elsewhere
            strong_enough = span.is_hadith() or (
                partial and matched >= PARTIAL_WORDS and weight >= PARTIAL_WEIGHT
            )
            if not strong_enough:
                break
            spans.append(span)
        return spans

    def _weight(self, q_words: list[str], ref_words: list[str]) -> float:
        """Summed IDF of reference words that a query word matched."""
        if not q_words or not ref_words:
            return 0.0
        sim = process.cdist(q_words, ref_words, scorer=fuzz.ratio, dtype=np.uint8) / 100.0
        total = 0.0
        for row in sim:
            j = int(row.argmax())
            if row[j] >= MATCH_SIM:
                total += self.idf.get(ref_words[j], 0.0)
        return total
