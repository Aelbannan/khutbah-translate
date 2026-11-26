"""Build data/quran.db from Tanzil text files plus quran.com's Saheeh International.

Sources:
  - Arabic (simple-clean, uthmani) and Ma Jian (zh): Tanzil, CC BY 3.0, unmodified.
  - English: quran.com API, translation 20 (Saheeh International, revised edition).
  - Surah names: quran.com API.

Usage: uv run scripts/build_quran_db.py
"""

import json
import re
import sqlite3
import time
import urllib.request
from pathlib import Path

from khutbah.arabic import BASMALA_NORM, normalize

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "tanzil"
DB = ROOT / "data" / "quran.db"
API = "https://api.quran.com/api/v4"
UA = {"User-Agent": "Mozilla/5.0 (compatible; quran-db-builder)"}

def load_tanzil(name: str) -> tuple[dict[tuple[int, int], str], str]:
    rows, header = {}, []
    for line in (RAW / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line)
        elif line.strip():
            s, a, text = line.split("|", 2)
            rows[(int(s), int(a))] = text
    assert len(rows) == 6236, f"{name}: {len(rows)} ayat"
    return rows, "\n".join(header)


CACHE = ROOT / "data" / "raw" / "qurancom"


def get(path: str) -> dict:
    """GET from the quran.com API, cached on disk so rebuilds are offline."""
    cached = CACHE / (re.sub(r"[^\w=&-]+", "_", path.strip("/")) + ".json")
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    req = urllib.request.Request(f"{API}{path}", headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    CACHE.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.1)
    return data


def strip_basmala(s: int, a: int, simple: str, uthmani: str) -> tuple[str, str]:
    """Tanzil prefixes the basmala to ayah 1 of every surah except 1 and 9."""
    if a != 1 or s in (1, 9):
        return simple, uthmani
    assert normalize(simple).startswith(BASMALA_NORM), (s, a)
    # Uthmani spelling varies slightly (e.g. a shadda on the ba in 2:1 and 3:1),
    # so compare skeletons and drop the first four words.
    assert normalize(uthmani).startswith(BASMALA_NORM), (s, a)
    return simple.split(" ", 4)[4], uthmani.split(" ", 4)[4]


def clean_translation(text: str) -> str:
    text = re.sub(r"<sup[^>]*>.*?</sup>", "", text)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text)).strip()


def main() -> None:
    simple, simple_hdr = load_tanzil("simple-clean.txt")
    uthmani, uthmani_hdr = load_tanzil("uthmani.txt")
    zh, zh_hdr = load_tanzil("zh.jian.txt")

    chapters_en = get("/chapters?language=en")["chapters"]
    chapters_zh = {c["id"]: c for c in get("/chapters?language=zh")["chapters"]}

    en: dict[tuple[int, int], str] = {}
    for ch in chapters_en:
        data = get(f"/quran/translations/20?chapter_number={ch['id']}&fields=verse_key")
        for t in data["translations"]:
            s, a = map(int, t["verse_key"].split(":"))
            en[(s, a)] = clean_translation(t["text"])
    assert len(en) == 6236, len(en)

    DB.parent.mkdir(parents=True, exist_ok=True)
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.executescript(
        """
        CREATE TABLE surah (
            number INTEGER PRIMARY KEY, name_ar TEXT, name_simple TEXT,
            name_en TEXT, name_zh TEXT, ayah_count INTEGER, bismillah_pre INTEGER);
        CREATE TABLE ayah (
            surah INTEGER, ayah INTEGER, text_uthmani TEXT, text_simple TEXT,
            text_norm TEXT, en TEXT, zh TEXT, PRIMARY KEY (surah, ayah));
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        """
    )
    db.executemany(
        "INSERT INTO surah VALUES (?,?,?,?,?,?,?)",
        [
            (
                c["id"], c["name_arabic"], c["name_simple"],
                c["translated_name"]["name"],
                chapters_zh[c["id"]]["translated_name"]["name"],
                c["verses_count"], int(c["bismillah_pre"]),
            )
            for c in chapters_en
        ],
    )
    rows = []
    for (s, a) in sorted(simple):
        smp, uth = strip_basmala(s, a, simple[(s, a)], uthmani[(s, a)])
        rows.append((s, a, uth, smp, normalize(smp), en[(s, a)], zh[(s, a)]))
    db.executemany("INSERT INTO ayah VALUES (?,?,?,?,?,?,?)", rows)
    db.executemany(
        "INSERT INTO meta VALUES (?,?)",
        [
            ("attribution", "Quran text: Tanzil Project (tanzil.net), CC BY 3.0. "
             "English: Saheeh International via quran.com. Chinese: Ma Jian via tanzil.net."),
            ("tanzil_simple_header", simple_hdr),
            ("tanzil_uthmani_header", uthmani_hdr),
            ("tanzil_zh_header", zh_hdr),
            ("built_at", time.strftime("%Y-%m-%d")),
        ],
    )
    db.commit()
    print(f"wrote {DB}: {len(rows)} ayat, {len(chapters_en)} surahs")


if __name__ == "__main__":
    main()
