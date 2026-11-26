"""Arabic text normalisation shared by the Quran DB builder and the matcher.

STT output has no reliable diacritics or hamza spelling, so both sides of a
match are reduced to the same bare letter skeleton.
"""

import re

# Harakat, Quranic annotation marks, superscript alef and tatweel.
_MARKS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ئ": "ي",
        "ؤ": "و",
        "ة": "ه",
        "ء": "",
    }
)
_NON_ARABIC = re.compile(r"[^ء-ي\s]")
_SPACES = re.compile(r"\s+")

BASMALA_NORM = "بسم الله الرحمن الرحيم"


def normalize(text: str) -> str:
    text = _MARKS.sub("", text)
    text = text.translate(_LETTER_MAP)
    text = _NON_ARABIC.sub(" ", text)
    return _SPACES.sub(" ", text).strip()
