"""Find which request parameter disables reasoning for a model. Usage: probe_reasoning.py <model>"""
import json, sys, time, uuid
import httpx
from khutbah.translate import api_key, BASE_URL

model = sys.argv[1]
c = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(60, connect=15), headers={
    "Authorization": f"Bearer {api_key()}", "x-opencode-session": f"khutbah-{uuid.uuid4()}",
    "User-Agent": "khutbah-translate/0.1"})
VARIANTS = {
    "reasoning_effort=none": {"reasoning_effort": "none"},
    "reasoning_effort=minimal": {"reasoning_effort": "minimal"},
    "thinking=disabled": {"thinking": {"type": "disabled"}},
    "enable_thinking=false": {"enable_thinking": False},
    "reasoning.enabled=false": {"reasoning": {"enabled": False}},
}
msg = [{"role": "user", "content": 'Translate to Simplified Chinese, reply JSON {"zh": "..."}: "The Friday sermon has begun."'}]
for name, extra in VARIANTS.items():
    t0 = time.perf_counter(); reasoning = 0; content = ""
    try:
        with c.stream("POST", "/chat/completions", json={"model": model, "stream": True, "messages": msg, **extra}) as r:
            if r.status_code != 200:
                print(f"{name:26} HTTP {r.status_code} {r.read()[:150]!r}"); continue
            for line in r.iter_lines():
                if not line.startswith("data: ") or line.strip() == "data: [DONE]": continue
                for ch in json.loads(line[6:]).get("choices", []):
                    d = ch.get("delta", {})
                    reasoning += len(d.get("reasoning_content") or d.get("reasoning") or "")
                    content += d.get("content") or ""
        print(f"{name:26} {time.perf_counter()-t0:5.1f}s reasoning={reasoning:<6} {content.strip()[:60]!r}")
        if reasoning == 0 and content: break
    except Exception as e:
        print(f"{name:26} ERROR {e}")
