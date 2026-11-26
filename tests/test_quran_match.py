"""Synthetic robustness test for the matcher: fragments, spans and STT-like noise."""

import random
import time

import pytest

from khutbah.quran import Quran

Q = Quran()
CONFUSE = {"ض": "د", "ظ": "ز", "ث": "س", "ذ": "ز", "ق": "ك", "ص": "س", "ط": "ت", "ح": "ه", "ع": "ا"}


def corrupt(text: str, rng: random.Random, rate: float) -> str:
    out = []
    for ch in text:
        r = rng.random()
        if ch != " " and r < rate / 2:
            continue  # deletion
        out.append(CONFUSE.get(ch, ch) if r < rate else ch)
    return "".join(out)


def sample_cases(n: int, seed: int = 1):
    rng = random.Random(seed)
    long_ayat = [i for i, a in enumerate(Q.ayat) if len(a.norm.split()) >= 6]
    for _ in range(n):
        i = rng.choice(long_ayat)
        a = Q.ayat[i]
        words = a.norm.split()
        kind = rng.choice(["full", "fragment", "span"])
        if kind == "fragment":
            n_words = rng.randint(4, min(10, len(words)))
            s = rng.randint(0, len(words) - n_words)
            text, expect = " ".join(words[s : s + n_words]), {(a.surah, a.ayah)}
        elif kind == "span" and i + 1 < len(Q.ayat) and Q.ayat[i + 1].surah == a.surah:
            b = Q.ayat[i + 1]
            text = " ".join(words[-4:] + b.norm.split()[:4])
            expect = {(a.surah, a.ayah), (b.surah, b.ayah)}
        else:
            text, expect = a.norm, {(a.surah, a.ayah)}
        yield kind, corrupt(text, rng, 0.08), expect


def test_matcher_accuracy():
    cases = list(sample_cases(300))
    hits, t0 = 0, time.perf_counter()
    for kind, text, expect in cases:
        spans = [s for s in Q.find(text) if s.matched >= 3]
        got = {(s.surah, x) for s in spans for x in range(s.first, s.last + 1)}
        # Correct if a span overlaps the truth (some ayat are textually identical).
        same_text = {Q.get(*g).norm for g in got} & {Q.get(*e).norm for e in expect}
        hits += bool(got & expect or same_text)
    ms = (time.perf_counter() - t0) / len(cases) * 1000
    print(f"accuracy {hits}/{len(cases)}, {ms:.1f} ms/query")
    assert hits / len(cases) > 0.9


@pytest.mark.parametrize(
    "text",
    [
        "أيها الإخوة الكرام اتقوا الله في أنفسكم وأهليكم",
        "قال رسول الله صلى الله عليه وسلم إنما الأعمال بالنيات",
        "نسأل الله أن يتقبل منا ومنكم صالح الأعمال",
        "اللهم اغفر للمسلمين والمسلمات الأحياء منهم والأموات",
    ],
)
def test_non_quran_rejected(text):
    spans = Q.find(text)
    print(text, "->", spans)
    assert not any(s.is_quran() for s in spans)


def test_embedded_and_formulas():
    text = "وأشهد أن محمدا عبده ورسوله يا أيها الذين آمنوا اتقوا الله حق تقاته ولا تموتن إلا وأنتم مسلمون"
    spans = [s for s in Q.find(text) if s.is_quran()]
    assert [s.ref for s in spans] == ["3:102"]
    assert not any(s.is_quran() for s in Q.find("خالد بن الوليد رضي الله عنه"))


def test_dhikr_formula_not_quran():
    # Kaffarat al-majlis: contains "لا إله إلا أنت", which is only in 21:87.
    text = "سبحانك اللهم وبحمدك أشهد أن لا إله إلا أنت أستغفرك وأتوب إليك"
    assert not any(s.is_quran() for s in Q.find(text))
    assert not any(s.is_quran() for s in Q.find("إنا لله وإنا إليه راجعون"))
