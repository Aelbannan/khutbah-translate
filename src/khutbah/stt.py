"""Pluggable speech-to-text backends (all via sherpa-onnx, CPU).

Every backend returns a Transcript with text and a language code ("ar"/"en").
Monolingual models are wrapped in `LidRouted`, which runs a spoken-language-ID
model first and dispatches to the right recogniser.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import sherpa_onnx

from khutbah.vad import SAMPLE_RATE

MODELS = Path("models")
_ARABIC = re.compile(r"[؀-ۿ]")
_LATIN = re.compile(r"[A-Za-z]")
# Speakers never use CJK; Qwen3-ASR emits it on noise (e.g. "嗯嗯。").
_CJK = re.compile(r"[\u3000-\u303F\u3400-\u9FFF\uF900-\uFAFF\uFF00-\uFFEF]+")


@dataclass
class Transcript:
    text: str
    lang: str
    seconds: float  # wall-clock compute time


class Backend(Protocol):
    name: str

    def transcribe(self, samples: np.ndarray) -> Transcript: ...


def script_lang(text: str) -> str:
    """Language from the script the model wrote in (for models without LID)."""
    return "ar" if len(_ARABIC.findall(text)) >= len(_LATIN.findall(text)) else "en"


def collapse_repeats(text: str, max_len: int = 12) -> tuple[str, bool]:
    """Collapse a word n-gram repeated in a row (decoder loops).

    Phrases (2+ words) collapse at 3 repeats, single words at 4, so ordinary
    emphasis like "no, no" or "ever ever" is left alone. Returns (text, looped).
    """
    words = text.split()
    looped = False
    for n in range(max_len, 0, -1):
        times = 3 if n > 1 else 4
        i = 0
        while i + n * times <= len(words):
            unit = words[i : i + n]
            reps = 1
            while words[i + reps * n : i + (reps + 1) * n] == unit:
                reps += 1
            if reps >= times:
                words = words[: i + n] + words[i + reps * n :]
                looped = True
            i += 1
    return " ".join(words), looped


def _decode(rec: sherpa_onnx.OfflineRecognizer, samples: np.ndarray):
    s = rec.create_stream()
    s.accept_waveform(SAMPLE_RATE, samples)
    rec.decode_stream(s)
    return s.result


class Sherpa:
    """A single sherpa-onnx offline recogniser."""

    def __init__(self, name: str, rec: sherpa_onnx.OfflineRecognizer, lang: str | None = None):
        self.name, self.rec, self.lang = name, rec, lang

    def transcribe(self, samples: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        r = _decode(self.rec, samples)
        text, _ = collapse_repeats(_CJK.sub(" ", r.text).strip())
        lang = self.lang or _norm_lang(getattr(r, "lang", "")) or script_lang(text)
        return Transcript(text, lang, time.perf_counter() - t0)


def _norm_lang(tag: str) -> str:
    tag = tag.strip("<|>").lower()
    if tag.startswith(("ar", "arabic")):
        return "ar"
    if tag.startswith(("en", "english")):
        return "en"
    return ""


class WhisperLid:
    def __init__(self, threads: int):
        d = MODELS / "sherpa-onnx-whisper-tiny"
        cfg = sherpa_onnx.SpokenLanguageIdentificationConfig(
            whisper=sherpa_onnx.SpokenLanguageIdentificationWhisperConfig(
                encoder=str(d / "tiny-encoder.int8.onnx"), decoder=str(d / "tiny-decoder.int8.onnx")
            ),
            num_threads=threads,
        )
        self.slid = sherpa_onnx.SpokenLanguageIdentification(cfg)

    def __call__(self, samples: np.ndarray) -> str:
        s = self.slid.create_stream()
        s.accept_waveform(SAMPLE_RATE, samples)
        return "ar" if self.slid.compute(s) == "ar" else "en"


class LidRouted:
    def __init__(self, name: str, lid: WhisperLid, by_lang: dict[str, Backend]):
        self.name, self.lid, self.by_lang = name, lid, by_lang

    def transcribe(self, samples: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        lang = self.lid(samples)
        t = self.by_lang[lang].transcribe(samples)
        return Transcript(t.text, lang, time.perf_counter() - t0)


# --- factories -------------------------------------------------------------


def _moonshine(lang: str, threads: int) -> Sherpa:
    d = MODELS / f"sherpa-onnx-moonshine-base-{lang}-quantized-2026-02-27"
    rec = sherpa_onnx.OfflineRecognizer.from_moonshine_v2(
        encoder=str(d / "encoder_model.ort"),
        decoder=str(d / "decoder_model_merged.ort"),
        tokens=str(d / "tokens.txt"),
        num_threads=threads,
    )
    return Sherpa(f"moonshine-base-{lang}", rec, lang)


def moonshine(threads: int = 4) -> Backend:
    return LidRouted(
        "moonshine-base(ar+en)+lid",
        WhisperLid(1),
        {"ar": _moonshine("ar", threads), "en": _moonshine("en", threads)},
    )


def whisper(size: str = "small", threads: int = 4) -> Backend:
    d = MODELS / f"sherpa-onnx-whisper-{size}"
    rec = sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(d / f"{size}-encoder.int8.onnx"),
        decoder=str(d / f"{size}-decoder.int8.onnx"),
        tokens=str(d / f"{size}-tokens.txt"),
        language="",  # auto-detect
        num_threads=threads,
    )
    return Sherpa(f"whisper-{size}", rec)


def omnilingual(threads: int = 4) -> Backend:
    d = MODELS / "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-v2-int8-2026-02-05"
    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=str(d / "model.int8.onnx"), tokens=str(d / "tokens.txt"), num_threads=threads
    )
    return Sherpa("omnilingual-300m", rec)


def qwen3(threads: int = 4) -> Backend:
    d = MODELS / "sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25"
    files = {p.name: str(p) for p in d.iterdir()}

    def pick(*keys: str) -> str:
        return next(v for k, v in sorted(files.items()) if all(x in k for x in keys))

    rec = sherpa_onnx.OfflineRecognizer.from_qwen3_asr(
        conv_frontend=pick("conv"),
        encoder=pick("encoder"),
        decoder=pick("decoder"),
        tokenizer=str(d / "tokenizer") if (d / "tokenizer").exists() else pick("token"),
        num_threads=threads,
        max_new_tokens=256,  # long khutbah segments truncate at the default 128
    )
    return Sherpa("qwen3-asr-0.6b", rec)


def _cohere(lang: str, threads: int) -> Sherpa:
    d = MODELS / "sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01"
    rec = sherpa_onnx.OfflineRecognizer.from_cohere_transcribe(
        encoder=str(d / "encoder.int8.onnx"),
        decoder=str(d / "decoder.int8.onnx"),
        tokens=str(d / "tokens.txt"),
        language=lang,
        num_threads=threads,
    )
    return Sherpa(f"cohere-{lang}", rec, lang)


def cohere(threads: int = 4) -> Backend:
    return LidRouted(
        "cohere-transcribe+lid", WhisperLid(1), {"ar": _cohere("ar", threads), "en": _cohere("en", threads)}
    )


BACKENDS: dict[str, Callable[..., Backend]] = {
    "moonshine": moonshine,
    "whisper-small": lambda threads=4: whisper("small", threads),
    "whisper-tiny": lambda threads=4: whisper("tiny", threads),
    "omnilingual": omnilingual,
    "qwen3": qwen3,
    "cohere": cohere,
}
