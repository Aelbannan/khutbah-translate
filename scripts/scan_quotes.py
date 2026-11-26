"""List Quran and hadith quotations found in a cached transcript.

Usage: scan_quotes.py <backend> [recording]
"""
import json, sys, time
import numpy as np
from khutbah.arabic import normalize
from khutbah.hadith import HadithDB
from khutbah.quran import Quran

Q, H = Quran(), HadithDB()
rec = sys.argv[2] if len(sys.argv) > 2 else "khutbah-2025-12-12"
sub = "" if rec == "khutbah-2025-12-12" else f"{rec}/"
tr = json.load(open(f"data/transcripts/{sub}{sys.argv[1]}.json"))
prev, t_total = None, 0.0
for k, v in tr.items():
    t0 = time.perf_counter()
    q = normalize(v["text"]).split()
    used = np.zeros(len(q), dtype=bool)
    for sp in Q.find(v["text"], prefer_after=prev):
        if sp.is_quran():
            print(f"#{k:>3} QURAN  {sp.ref:22} | {' '.join(q[sp.q_start:sp.q_end])[:70]}")
            used[sp.q_start:sp.q_end] = True
            prev = (sp.surah, sp.last)
    for hs in H.find(v["text"], skip=used):
        print(f"#{k:>3} HADITH {hs.hadith.citation:22} w={hs.weight:4.0f} | {' '.join(q[hs.q_start:hs.q_end])[:70]}")
    t_total += time.perf_counter() - t0
print(f"-- {len(tr)} segments, {t_total/len(tr)*1000:.0f} ms/segment")
