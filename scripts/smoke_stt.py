"""Transcribe a spread of segments with the given backends. Usage: smoke_stt.py moonshine omnilingual"""
import json, sys
from pathlib import Path
import soundfile as sf
from khutbah.stt import BACKENDS

seg_dir = Path("data/segments/khutbah-2025-12-12")
man = [m for m in json.loads((seg_dir / "manifest.json").read_text()) if m["end"] - m["start"] > 3]
picks = man[:: max(1, len(man) // 12)][:12]
backends = [BACKENDS[n]() for n in sys.argv[1:]]
for m in picks:
    audio, _ = sf.read(seg_dir / m["file"], dtype="float32")
    dur = len(audio) / 16000
    print(f"\n#{m['id']} @{m['start']/60:.1f}min {dur:.1f}s")
    for b in backends:
        t = b.transcribe(audio)
        print(f"  [{b.name} {t.lang} rtf={t.seconds/dur:.2f}] {t.text}")
