"""Translate a recording's cached transcripts end to end with one or more LLMs.

Simulates the live pipeline: segments in order -> quote cards + placeholders ->
passages -> translation (with rolling context). Models run concurrently, each
with its own context. Output: data/runs/<recording>/<model>.jsonl and cards.json.

Usage: run_translate.py <recording> <model> [<model> ...]
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from khutbah.pipeline import Batcher, QuoteFinder
from khutbah.translate import Translator

rec, models = sys.argv[1], sys.argv[2:]
sub = "" if rec == "khutbah-2025-12-12" else f"{rec}/"
man = json.loads(Path(f"data/segments/{rec}/manifest.json").read_text())
qwen = json.loads(Path(f"data/transcripts/{sub}qwen3.json").read_text())
omni = json.loads(Path(f"data/transcripts/{sub}omnilingual.json").read_text())
out_dir = Path("data/runs") / rec
out_dir.mkdir(parents=True, exist_ok=True)

# Quote finding and batching are deterministic: do them once, share across models.
qf, batcher = QuoteFinder(), Batcher()
passages, cards = [], []
for m in man:
    k = str(m["id"])
    text, c = qf.process(m["id"], qwen[k]["text"], omni[k]["text"], t=m["start"])
    cards += [{"kind": x.kind, "ref": x.ref, "segment": x.segment, "new": x.new} for x in c]
    passages += batcher.add(m["id"], m["start"], m["end"], text)
if p := batcher.flush():
    passages.append(p)
(out_dir / "cards.json").write_text(json.dumps(cards, ensure_ascii=False, indent=1))
print(f"{rec}: {len(passages)} passages, {sum(c['new'] for c in cards)} cards", flush=True)
if not models:  # quote-finding dry run
    for c in cards:
        print(f"  #{c['segment']:>3} {'NEW ' if c['new'] else 'again'} {c['kind']:6} {c['ref']}")
    sys.exit()


def run(model: str) -> str:
    tr = Translator(model=model)
    path = out_dir / f"{model}.jsonl"
    times, errors = [], 0
    with path.open("w") as f:
        for i, p in enumerate(passages):
            row = {"i": i, "segments": p.segments, "start": p.start, "end": p.end, "source": p.text}
            try:
                t = tr.translate(p.text, ["en", "zh"])
                row |= {"en": t.out["en"], "zh": t.out["zh"], "seconds": round(t.seconds, 2)}
                times.append(t.seconds)
            except Exception as e:  # keep going; record the failure
                row |= {"error": str(e)[:300]}
                errors += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    times.sort()
    med = times[len(times) // 2] if times else float("nan")
    p90 = times[int(len(times) * 0.9)] if times else float("nan")
    return f"{model}: {len(times)} ok, {errors} errors, median {med:.1f}s, p90 {p90:.1f}s"


t0 = time.perf_counter()
with ThreadPoolExecutor(max_workers=len(models)) as ex:
    for line in ex.map(run, models):
        print(line, flush=True)
print(f"done in {time.perf_counter() - t0:.0f}s")
