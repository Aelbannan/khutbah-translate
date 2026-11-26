"""List Quran spans found in a cached transcript file. Usage: quran_hits.py <backend> [recording]"""
import json, sys
from khutbah.arabic import normalize
from khutbah.quran import Quran

Q = Quran()
rec = sys.argv[2] if len(sys.argv) > 2 else "khutbah-2025-12-12"
sub = "" if rec == "khutbah-2025-12-12" else f"{rec}/"
tr = json.load(open(f"data/transcripts/{sub}{sys.argv[1]}.json"))
prev = None
for k, v in tr.items():
    q = normalize(v["text"]).split()
    for sp in Q.find(v["text"], prefer_after=prev):
        if sp.is_quran():
            print(f"#{k:>3} {sp.ref:9} m={sp.matched:<2} | {' '.join(q[sp.q_start:sp.q_end])[:100]}")
            prev = (sp.surah, sp.last)
