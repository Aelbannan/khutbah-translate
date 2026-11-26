#!/usr/bin/env bash
# Download models and source texts, then build the Quran and hadith databases.
# Idempotent: anything already present is skipped. Run from the repo root:
#   ./scripts/setup.sh            # VAD + Qwen3-ASR + Omnilingual (the defaults)
#   ./scripts/setup.sh --all-stt  # also Moonshine, Whisper, Cohere (for comparisons)
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() { # url dest
  if [[ -s "$2" ]]; then return; fi
  echo "  $2"
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2"
}

echo "== models (sherpa-onnx releases)"
ASR=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models
fetch "$ASR/silero_vad.onnx" models/silero_vad.onnx
MODELS=(
  sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
  sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05
)
if [[ "${1:-}" == "--all-stt" ]]; then
  MODELS+=(
    sherpa-onnx-moonshine-base-ar-quantized-2026-02-27
    sherpa-onnx-moonshine-base-en-quantized-2026-02-27
    sherpa-onnx-whisper-tiny
    sherpa-onnx-whisper-small
    sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01
  )
fi
for m in "${MODELS[@]}"; do
  if [[ -d "models/$m" ]]; then continue; fi
  echo "  models/$m"
  curl -fsSL --retry 3 "$ASR/$m.tar.bz2" | tar -xj -C models
done

echo "== Quran text (Tanzil, CC BY 3.0)"
T=data/raw/tanzil
fetch "https://tanzil.net/pub/download/index.php?quranType=simple-clean&outType=txt-2&agree=true" $T/simple-clean.txt
fetch "https://tanzil.net/pub/download/index.php?quranType=uthmani&outType=txt-2&agree=true" $T/uthmani.txt
fetch "https://tanzil.net/trans/?transID=zh.jian&type=txt-2" $T/zh.jian.txt

echo "== hadith (fawazahmed0/hadith-api)"
for b in nawawi qudsi bukhari muslim abudawud tirmidhi nasai ibnmajah malik; do
  for l in ara eng; do
    fetch "https://cdn.jsdelivr.net/gh/fawazahmed0/hadith-api@1/editions/$l-$b.min.json" data/raw/hadith/$l-$b.json
  done
done

echo "== databases"
[[ -s data/quran.db ]] || uv run scripts/build_quran_db.py   # also fetches English from the quran.com API
[[ -s data/hadith.db ]] || uv run scripts/build_hadith_db.py
echo "done"
