"""Stream audio to a khutbah server and print the events that come back.

Any input ffmpeg understands is decoded to 16 kHz mono PCM16 and sent in
100 ms frames, paced in real time (use --speed to go faster).

  stream_audio.py recording.mp3                    # a file, real time
  stream_audio.py recording.mp3 --speed 4          # 4x real time (tests lag handling)
  stream_audio.py --mic                            # default microphone (macOS/Linux)
  stream_audio.py --ffmpeg-input "-f alsa -i hw:1" # any ffmpeg input
"""

import argparse
import asyncio
import json
import shlex
import sys
import time
from urllib.parse import quote

import websockets

RATE, FRAME = 16000, 1600 * 2  # 100 ms of PCM16


def ffmpeg_args(a: argparse.Namespace) -> list[str]:
    if a.ffmpeg_input:
        src = shlex.split(a.ffmpeg_input)
    elif a.mic:
        src = ["-f", "avfoundation", "-i", ":0"] if sys.platform == "darwin" else ["-f", "pulse", "-i", "default"]
    else:
        src = ["-i", a.input]
    return ["ffmpeg", "-loglevel", "error", *src, "-ac", "1", "-ar", str(RATE), "-f", "s16le", "-"]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?")
    ap.add_argument("--mic", action="store_true")
    ap.add_argument("--ffmpeg-input")
    ap.add_argument("--url", default="ws://localhost:8000/ws/ingest")
    ap.add_argument("--title", default="Khutbah")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--json", action="store_true", help="print raw events")
    a = ap.parse_args()
    if not (a.input or a.mic or a.ffmpeg_input):
        ap.error("give a file, --mic or --ffmpeg-input")
    live = a.mic or a.ffmpeg_input  # already real time; don't pace

    proc = await asyncio.create_subprocess_exec(*ffmpeg_args(a), stdout=asyncio.subprocess.PIPE)
    async with websockets.connect(f"{a.url}?title={quote(a.title)}", max_size=None) as ws:

        async def send() -> None:
            t0, sent = time.perf_counter(), 0
            while chunk := await proc.stdout.read(FRAME):
                await ws.send(chunk)
                sent += len(chunk) // 2
                if not live:
                    ahead = sent / RATE / a.speed - (time.perf_counter() - t0)
                    if ahead > 0:
                        await asyncio.sleep(ahead)
            await ws.send(json.dumps({"type": "end"}))
            print(f"-- audio finished ({sent / RATE:.0f}s); waiting for the pipeline to drain", file=sys.stderr)

        async def recv() -> None:
            async for msg in ws:
                ev = json.loads(msg)
                if a.json:
                    print(msg, flush=True)
                elif ev["type"] == "segment" and ev["text"]:
                    print(f"[{ev['start']:7.1f}s {ev['lang']} {ev['stt_seconds']:.1f}s] {ev['text']}", flush=True)
                elif ev["type"] == "card" and ev["new"]:
                    print(f"    ▸ {ev['kind'].upper()} {ev['ref']}", flush=True)
                elif ev["type"] == "passage":
                    for lang in ("en", "zh"):
                        if ev.get(lang):
                            print(f"    {lang}: {ev[lang]}", flush=True)
                    if ev.get("error"):
                        print(f"    translation error: {ev['error']}", flush=True)
                elif ev["type"] == "status" and ev["lag"] > 5:
                    print(f"    (lag {ev['lag']}s, {ev['queued']} queued)", flush=True)
                elif ev["type"] == "session" and ev["state"] == "ended":
                    print(f"-- session {ev['session']} ended", file=sys.stderr)
                    return

        sender = asyncio.create_task(send())
        await recv()
        sender.cancel()
    if proc.returncode is None:
        proc.kill()


if __name__ == "__main__":
    asyncio.run(main())
