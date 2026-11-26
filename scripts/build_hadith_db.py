"""Build data/hadith.db from fawazahmed0/hadith-api (Arabic + English editions).

Expects data/raw/hadith/{ara,eng}-<collection>.json (downloaded from
https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/).
No Chinese edition exists; Chinese is machine-translated from the English at
display time and labelled as such.

Usage: uv run scripts/build_hadith_db.py
"""

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path

from khutbah.arabic import normalize

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "hadith"
DB = ROOT / "data" / "hadith.db"

# (key, display name, priority) - lower priority wins when a hadith is in several books.
COLLECTIONS = [
    ("nawawi", "an-Nawawi's Forty Hadith", 0),
    ("qudsi", "Forty Hadith Qudsi", 1),
    ("bukhari", "Sahih al-Bukhari", 2),
    ("muslim", "Sahih Muslim", 3),
    ("abudawud", "Sunan Abi Dawud", 4),
    ("tirmidhi", "Jami' at-Tirmidhi", 5),
    ("nasai", "Sunan an-Nasa'i", 6),
    ("ibnmajah", "Sunan Ibn Majah", 7),
    ("malik", "Muwatta Malik", 8),
]
PREFERRED_GRADER = "Al-Albani"


def ref_number(arabicnumber) -> str:
    """'2937.01' -> '2937' (sub-narrations share the standard number)."""
    s = str(arabicnumber)
    return s.split(".")[0] if s and s != "0" else ""


def grade(grades: list[dict]) -> str:
    for g in grades:
        if g["name"] == PREFERRED_GRADER:
            return f"{g['grade']} ({g['name']})"
    return f"{grades[0]['grade']} ({grades[0]['name']})" if grades else ""


def main() -> None:
    rows = []
    for key, name, prio in COLLECTIONS:
        ara = {h["hadithnumber"]: h for h in json.loads((RAW / f"ara-{key}.json").read_text())["hadiths"]}
        eng = {h["hadithnumber"]: h for h in json.loads((RAW / f"eng-{key}.json").read_text())["hadiths"]}
        assert set(ara) == set(eng), key
        for n, a in ara.items():
            ar, en = a["text"].strip(), eng[n]["text"].strip()
            if not ar or not en:
                continue
            rows.append((key, name, prio, n, ref_number(a.get("arabicnumber", "")), ar, normalize(ar), en,
                         grade(eng[n].get("grades", []))))
        print(f"{key}: {sum(1 for r in rows if r[0] == key)}")

    df: Counter[str] = Counter()
    for r in rows:
        df.update(set(r[6].split()))
    n = len(rows)

    DB.unlink(missing_ok=True)
    for f in DB.parent.glob("hadith.index*.npz"):
        f.unlink()
    db = sqlite3.connect(DB)
    db.executescript(
        """
        CREATE TABLE hadith (
            id INTEGER PRIMARY KEY, collection TEXT, collection_name TEXT, priority INTEGER,
            number INTEGER, ref TEXT, ar TEXT, norm TEXT, en TEXT, grade TEXT);
        CREATE TABLE word_idf (word TEXT PRIMARY KEY, idf REAL);
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    db.executemany(
        "INSERT INTO hadith (collection, collection_name, priority, number, ref, ar, norm, en, grade)"
        " VALUES (?,?,?,?,?,?,?,?,?)",
        rows,
    )
    db.executemany("INSERT INTO word_idf VALUES (?,?)", [(w, math.log(n / c)) for w, c in df.items()])
    db.execute(
        "INSERT INTO meta VALUES ('attribution', ?)",
        ("Hadith Arabic and English: fawazahmed0/hadith-api (sunnah.com translations). "
         "Chinese hadith text is machine-translated from the English.",),
    )
    db.commit()
    print(f"wrote {DB}: {n} hadith, {len(df)} distinct words")


if __name__ == "__main__":
    main()
