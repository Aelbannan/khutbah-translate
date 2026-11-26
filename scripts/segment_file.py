"""Run the VAD over a recording as if it were live; write segment WAVs + manifest.

Usage: uv run scripts/segment_file.py data/audio/khutbah-2025-12-12.wav
"""

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from khutbah.vad import SAMPLE_RATE, Segmenter

src = Path(sys.argv[1])
out = Path("data/segments") / src.stem
out.mkdir(parents=True, exist_ok=True)

audio, sr = sf.read(src, dtype="float32")
assert sr == SAMPLE_RATE and audio.ndim == 1

seg = Segmenter()
segments = []
chunk = SAMPLE_RATE // 10  # 100 ms, like a live mic callback
for i in range(0, len(audio), chunk):
    segments.extend(seg.feed(audio[i : i + chunk]))
segments.extend(seg.flush())

manifest = []
for n, s in enumerate(segments):
    name = f"{n:04d}.wav"
    sf.write(out / name, s.samples, SAMPLE_RATE)
    manifest.append({"id": n, "file": name, "start": round(s.start, 2), "end": round(s.end, 2)})
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))

d = np.array([s.duration for s in segments])
speech = d.sum()
print(f"{len(segments)} segments, speech {speech/60:.1f} min of {len(audio)/SAMPLE_RATE/60:.1f} min")
print(f"duration p10/p50/p90/max: {np.percentile(d,10):.1f}/{np.median(d):.1f}/{np.percentile(d,90):.1f}/{d.max():.1f}s")
print(f"forced cuts (>=14.9s): {(d >= 14.9).sum()}")
