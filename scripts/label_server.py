"""Tiny local labelling UI: listen to segments, correct a draft transcript, tag language.

Usage: uv run scripts/label_server.py   then open http://localhost:8765
Labels are saved to data/labels/khutbah-2025-12-12.json on every change.
"""

import json
import random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NAME = "khutbah-2025-12-12"
SEG_DIR = Path("data/segments") / NAME
LABELS = Path("data/labels") / f"{NAME}.json"
DRAFT = Path("data/transcripts/qwen3.json")
ALWAYS = [5, 7, 8, 171, 214, 215, 255, 347, 349]  # segments the matcher flagged as Quran
N = 40


def pick_segments() -> list[dict]:
    man = json.loads((SEG_DIR / "manifest.json").read_text())
    draft = json.loads(DRAFT.read_text())
    ok = [m for m in man if 2.0 <= m["end"] - m["start"] <= 12.0 and m["id"] not in ALWAYS]
    rng = random.Random(0)
    step = len(ok) / (N - len(ALWAYS))
    chosen = {ok[int(i * step + rng.random() * step)]["id"] for i in range(N - len(ALWAYS))} | set(ALWAYS)
    return [
        {**m, "draft": draft[str(m["id"])]["text"]} for m in man if m["id"] in chosen
    ]


PAGE = """<!doctype html><meta charset=utf-8><title>Label segments</title>
<style>
body{font:15px system-ui;max-width:900px;margin:2em auto;padding:0 1em}
.seg{border:1px solid #ccc;border-radius:8px;padding:.8em;margin:.8em 0}
.seg.done{border-color:#3a3;background:#f4fbf4}
textarea{width:100%;font:16px system-ui;min-height:3em;unicode-bidi:plaintext}
.cats button{margin:.2em;padding:.3em .7em;border:1px solid #999;border-radius:5px;background:#fff;cursor:pointer}
.cats button.on{background:#235;color:#fff}
.meta{color:#666;font-size:13px}
#bar{position:sticky;top:0;background:#fff;padding:.5em 0;border-bottom:1px solid #ddd}
</style>
<div id=bar><b>Label segments</b> — <span id=count></span>. Saved automatically.
<span class=meta>Play, fix the text to what was actually said (Arabic in Arabic script, English in English;
leave out punctuation worries), pick a category, tick done.</span></div>
<div id=list></div>
<script>
const CATS=["english","arabic","mixed","quran","quran+other","noise"];
let segs=[],labels={};
async function save(){await fetch('/save',{method:'POST',body:JSON.stringify(labels)});count()}
function count(){document.getElementById('count').textContent=
  Object.values(labels).filter(l=>l.done).length+' / '+segs.length+' done'}
async function init(){
  segs=await (await fetch('/segments')).json(); labels=await (await fetch('/labels')).json();
  const list=document.getElementById('list');
  for(const s of segs){
    const l=labels[s.id]??={text:s.draft,cat:null,done:false};
    const d=document.createElement('div'); d.className='seg'+(l.done?' done':'');
    d.innerHTML=`<div class=meta>#${s.id} · ${(s.start/60).toFixed(1)} min · ${(s.end-s.start).toFixed(1)} s</div>
      <audio controls preload=none src="/audio/${s.file}"></audio>
      <textarea dir=auto></textarea>
      <div class=cats>${CATS.map(c=>`<button data-c="${c}">${c}</button>`).join('')}
      <label style="margin-left:1em"><input type=checkbox> done</label></div>`;
    const ta=d.querySelector('textarea'); ta.value=l.text;
    ta.oninput=()=>{l.text=ta.value;save()};
    d.querySelectorAll('button').forEach(b=>{b.classList.toggle('on',b.dataset.c===l.cat);
      b.onclick=()=>{l.cat=b.dataset.c;d.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));save()}});
    const cb=d.querySelector('input'); cb.checked=l.done;
    cb.onchange=()=>{l.done=cb.checked;d.classList.toggle('done',l.done);save()};
    list.append(d);
  }
  count();
}
init();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, ctype: str):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/segments":
            self._send(json.dumps(SEGMENTS, ensure_ascii=False).encode(), "application/json")
        elif self.path == "/labels":
            body = LABELS.read_bytes() if LABELS.exists() else b"{}"
            self._send(body, "application/json")
        elif self.path.startswith("/audio/"):
            f = SEG_DIR / Path(self.path[7:]).name  # .name blocks path traversal
            if f.suffix == ".wav" and f.exists():
                self._send(f.read_bytes(), "audio/wav")
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path != "/save":
            return self.send_error(404)
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LABELS.parent.mkdir(parents=True, exist_ok=True)
        LABELS.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        self._send(b"ok", "text/plain")

    def log_message(self, *a):
        pass


SEGMENTS = pick_segments()

if __name__ == "__main__":
    print(f"{len(SEGMENTS)} segments → http://localhost:8765")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
