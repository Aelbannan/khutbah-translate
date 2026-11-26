"""Transcribe every segment with one backend; cache to data/transcripts/<backend>.json.

Usage: transcribe_all.py <backend> [recording] [threads]
"""
import json, sys, time
from pathlib import Path
import soundfile as sf
from khutbah.stt import BACKENDS

name = sys.argv[1]
rec = sys.argv[2] if len(sys.argv) > 2 else "khutbah-2025-12-12"
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
seg_dir = Path("data/segments") / rec
# The first recording's transcripts predate per-recording folders.
out = Path("data/transcripts") / (f"{name}.json" if rec == "khutbah-2025-12-12" else f"{rec}/{name}.json")
out.parent.mkdir(parents=True, exist_ok=True)
man = json.loads((seg_dir / "manifest.json").read_text())
b = BACKENDS[name](threads=threads)
res, audio_s, compute_s = {}, 0.0, 0.0
for m in man:
    audio, _ = sf.read(seg_dir / m["file"], dtype="float32")
    t = b.transcribe(audio)
    res[m["id"]] = {"text": t.text, "lang": t.lang, "seconds": round(t.seconds, 3)}
    audio_s += len(audio) / 16000; compute_s += t.seconds
out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
print(f"{b.name}: {len(man)} segs, audio {audio_s/60:.1f} min, compute {compute_s:.0f}s, RTF {compute_s/audio_s:.3f}")
