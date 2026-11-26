"""Khutbah translation via an OpenAI-compatible chat API (opencode-go by default)."""

import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import httpx

BASE_URL = os.environ.get("KHUTBAH_LLM_BASE_URL", "https://opencode.ai/inference/go/openai/v1")
MODEL = os.environ.get("KHUTBAH_LLM_MODEL", "qwen3.8-flash")


def api_key() -> str:
    """OPENCODE_GO_API_KEY, else the key pi stores for its opencode-go provider."""
    if key := os.environ.get("OPENCODE_GO_API_KEY"):
        return key
    auth = Path.home() / ".pi/agent/auth.json"
    if auth.exists():
        return json.loads(auth.read_text())["opencode-go"]["key"]
    raise RuntimeError("Set OPENCODE_GO_API_KEY")


SYSTEM = """You translate a live Friday khutbah or Islamic lesson for on-screen captions.
Input is raw speech recognition of one passage, often mixing English and Arabic; it may contain recognition errors.
Rules:
- Render the passage in each requested language: {targets}. Output JSON only: {{{schema}}}.
  The English caption covers everything said: clean up the speaker's English and translate the Arabic.
- Convey the meaning faithfully and naturally for captions; fix obvious recognition errors using context; do not add commentary.
- NEVER invent names, narrators, sources, numbers or details that are not in the input. If a name is garbled
  and you are not sure who is meant, transliterate what was heard rather than guessing.
- Placeholders like [QURAN 3:102] or [HADITH Sahih Muslim 2937] mark quotations shown separately with their
  established translation: keep them exactly as-is (same brackets, same Latin text) in every language and do not
  translate the quoted words again. "[... quoted here]" means the quotation occurs in this passage but its
  Arabic words were not removed: keep the marker and do not translate the quotation itself.
- Speakers often recite Arabic and then translate it themselves. Do not translate the same content twice:
  give its meaning once.
- Arabic names, places and terms: use the standard form (e.g. Yarmuk, Khalid ibn al-Walid, Tabarani).
- Chinese: use the conventions of Ma Jian's Quran translation (Allah = 真主, the Prophet = 先知/使者);
  ﷺ / "sallallahu alayhi wa sallam" = （愿主福安之）; "radiyallahu anhu" = （愿主喜悦他）;
  "alayhis salam" (other prophets, angels) = （愿主赐他平安）.
- English: keep common Islamic terms transliterated (e.g. khutbah, sunnah, du'a) and render ﷺ as ﷺ.
- If the passage is only noise or unintelligible, return empty strings."""

# Captions can't wait for chain-of-thought. Most models accept "none"; some only
# "minimal" (mimo-v2.6-flash still reasons briefly even then). Models not listed
# (e.g. claude-haiku-5-5) don't reason by default, so no parameter is sent.
REASONING_EFFORT = {
    "qwen3.8-flash": "none",
    "deepseek-v4.1-flash": "none",
    "glm-5.3-flash": "none",
    "minimax-m3.1-flash": "minimal",
    "mimo-v2.6-flash": "minimal",
    "muse-spark-1.3-contributor": "minimal",
}

LANG_NAMES = {"en": "English", "zh": "Simplified Chinese"}


@dataclass
class Translation:
    out: dict[str, str]
    seconds: float
    model: str


@dataclass
class Translator:
    model: str = MODEL
    context_turns: int = 4  # previous passages sent for continuity
    history: deque = field(default_factory=lambda: deque(maxlen=4))
    session: str = field(default_factory=lambda: f"khutbah-{uuid.uuid4()}")  # one per khutbah

    def __post_init__(self):
        self.client = httpx.Client(
            base_url=BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key()}",
                # opencode Go asks clients for a stable per-conversation session id
                # and their own user agent.
                "x-opencode-session": self.session,
                "User-Agent": "khutbah-translate/0.1",
            },
            timeout=30,
        )
        self.history = deque(maxlen=self.context_turns)

    def translate(self, text: str, targets: list[str]) -> Translation:
        schema = ", ".join(f'"{t}": "..."' for t in targets)
        system = SYSTEM.format(targets=", ".join(LANG_NAMES[t] for t in targets), schema=schema)
        messages = [{"role": "system", "content": system}]
        for src, out in self.history:  # earlier passages, for names/pronouns/terminology
            messages += [{"role": "user", "content": src}, {"role": "assistant", "content": out}]
        messages.append({"role": "user", "content": text})
        t0 = time.perf_counter()
        r = self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                **({"reasoning_effort": e} if (e := REASONING_EFFORT.get(self.model)) else {}),
            },
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        out = _parse(content, targets)
        self.history.append((text, json.dumps(out, ensure_ascii=False)))
        return Translation(out, time.perf_counter() - t0, self.model)


def _parse(content: str, targets: list[str]) -> dict[str, str]:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`").split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        data = json.loads(content[content.index("{") : content.rindex("}") + 1])
    except ValueError:
        data = {}
    return {t: str(data.get(t, "")).strip() for t in targets}
