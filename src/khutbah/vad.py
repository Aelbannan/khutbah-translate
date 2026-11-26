"""Streaming speech segmentation with Silero VAD (via sherpa-onnx, no torch)."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000


@dataclass
class VadConfig:
    model: Path = Path("models/silero_vad.onnx")
    threshold: float = 0.5
    min_silence: float = 0.6  # seconds of silence that end a segment
    min_speech: float = 0.25  # drop blips shorter than this
    max_speech: float = 15.0  # force a cut so STT latency stays bounded
    pre_roll: float = 0.3  # audio kept before detected onset
    threads: int = 1


@dataclass
class Segment:
    start: float  # seconds from stream start
    end: float
    samples: np.ndarray  # float32 mono, 16 kHz

    @property
    def duration(self) -> float:
        return self.end - self.start


class Segmenter:
    """Feed float32 16 kHz audio in any chunk size; yields finished segments."""

    def __init__(self, cfg: VadConfig = VadConfig()):
        self.cfg = cfg
        c = sherpa_onnx.VadModelConfig()
        c.silero_vad.model = str(cfg.model)
        c.silero_vad.threshold = cfg.threshold
        c.silero_vad.min_silence_duration = cfg.min_silence
        c.silero_vad.min_speech_duration = cfg.min_speech
        c.silero_vad.max_speech_duration = cfg.max_speech
        c.sample_rate = SAMPLE_RATE
        c.num_threads = cfg.threads
        self.window = c.silero_vad.window_size
        self.vad = sherpa_onnx.VoiceActivityDetector(c, buffer_size_in_seconds=120)
        # Rolling buffer of recent audio (for pre-roll); history[0] is sample `offset`.
        self.history = np.zeros(0, dtype=np.float32)
        self.offset = 0
        self.keep = int((cfg.max_speech + cfg.pre_roll + 30) * SAMPLE_RATE)
        self.pending = np.zeros(0, dtype=np.float32)

    def _drain(self) -> Iterator[Segment]:
        pre = int(self.cfg.pre_roll * SAMPLE_RATE)
        while not self.vad.empty():
            seg = self.vad.front
            seg_start, seg_len = seg.start, len(seg.samples)  # read before pop()
            self.vad.pop()
            start = max(self.offset, seg_start - pre)
            end = seg_start + seg_len
            audio = self.history[start - self.offset : end - self.offset].copy()
            yield Segment(start / SAMPLE_RATE, end / SAMPLE_RATE, audio)

    def feed(self, audio: np.ndarray) -> Iterator[Segment]:
        self.history = np.concatenate([self.history, audio])
        if len(self.history) > 2 * self.keep:
            drop = len(self.history) - self.keep
            self.history = self.history[drop:]
            self.offset += drop
        self.pending = np.concatenate([self.pending, audio])
        while len(self.pending) >= self.window:
            self.vad.accept_waveform(self.pending[: self.window])
            self.pending = self.pending[self.window :]
        yield from self._drain()

    def flush(self) -> Iterator[Segment]:
        self.vad.flush()
        yield from self._drain()
