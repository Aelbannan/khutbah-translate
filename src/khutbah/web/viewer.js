// Caption viewer: connects to /ws/view and renders passages + quote cards.
// URL params: ?lang=en|zh|both  ?tv=1 (large text, no chrome)
const params = new URLSearchParams(location.search);
const langSel = document.getElementById("lang");
langSel.value = params.get("lang") || localStorage.getItem("lang") || "both";
if (params.get("tv")) document.body.classList.add("tv");

const $captions = document.getElementById("captions");
const $cards = document.getElementById("cards");
const $hearing = document.getElementById("hearing");
const $state = document.getElementById("state");
const MAX_PASSAGES = 40, MAX_CARDS = 6;
const passages = new Map(); // id -> event
let session = null;

const langs = () => (langSel.value === "both" ? ["en", "zh"] : [langSel.value]);
langSel.onchange = () => { localStorage.setItem("lang", langSel.value); rerender(); };

// Placeholders like [QURAN 3:102] become a short visible reference.
function clean(text) {
  return (text || "")
    .replace(/[［\[]\s*(QURAN|HADITH)\s+([^\]］]+?)(\s+quoted here)?\s*[\]］]/g,
             (_, k, ref) => `⟨${k === "QURAN" ? "Qur'an" : ""} ${ref.trim()}⟩`.replace("⟨ ", "⟨"))
    .replace(/[［\[](unclear|inaudible|听不清|不清楚)[\]］]/gi, "…");
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  e.dir = "auto"; // mixed Arabic/English lines pick their own direction
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

function renderPassage(ev) {
  const div = el("div", "passage");
  div.dataset.id = ev.id;
  for (const l of langs()) {
    if (ev[l]) div.append(el("div", l, clean(ev[l])));
  }
  if (!div.childElementCount) {
    // Not translated (translation off/failed): show the source text.
    div.append(el("div", "pending", clean(ev.source)));
  }
  return div;
}

function renderCard(ev) {
  const div = el("div", `card ${ev.kind}`);
  const want = langs();
  if (ev.kind === "quran") {
    const name = ev.surah_name || {};
    div.append(el("div", "ref", `${name.simple || ""} ${ev.ref}` +
                  (want.includes("zh") && name.zh ? ` · ${name.zh}` : "")));
    div.append(el("div", "ar", ev.ayat.map(a => `${a.ar} ﴿${a.ayah}﴾`).join(" ")));
    for (const l of want) div.append(el("div", `tr ${l}`, ev.ayat.map(a => a[l]).join(" ")));
  } else {
    div.append(el("div", "ref", ev.citation + (ev.grade ? ` · ${ev.grade}` : "")));
    div.append(el("div", "ar", ev.ar));
    div.append(el("div", "tr en", ev.en));
    if (want.includes("zh")) {
      if (ev.zh) {
        div.append(el("div", "tr zh", ev.zh));
        div.append(el("div", "note", "中文为机器翻译"));
      }
    }
  }
  return div;
}

function addPassage(ev) {
  passages.set(ev.id, ev);
  while (passages.size > MAX_PASSAGES) passages.delete(passages.keys().next().value);
  $captions.append(renderPassage(ev));
  while ($captions.childElementCount > MAX_PASSAGES) $captions.firstChild.remove();
  $captions.scrollTop = $captions.scrollHeight;
}

const cardEvents = [];
function addCard(ev) {
  if (!ev.new) return; // already shown recently
  cardEvents.push(ev);
  if (cardEvents.length > MAX_CARDS) cardEvents.shift();
  $cards.prepend(renderCard(ev));
  while ($cards.childElementCount > MAX_CARDS) $cards.lastChild.remove();
}

function rerender() {
  $captions.replaceChildren(...[...passages.values()].map(renderPassage));
  $cards.replaceChildren(...cardEvents.slice().reverse().map(renderCard));
  $captions.scrollTop = $captions.scrollHeight;
}

function reset() {
  passages.clear(); cardEvents.length = 0; rerender();
}

function handle(ev) {
  if (ev.session != null && ev.session !== session && ev.type !== "status") {
    if (session != null) reset();
    session = ev.session;
  }
  switch (ev.type) {
    case "session":
      $state.textContent = ev.state === "started" ? "live" : "ended";
      $state.className = ev.state === "started" ? "" : "off";
      if (ev.title) document.getElementById("title").textContent = ev.title;
      break;
    case "passage": addPassage(ev); break;
    case "card": addCard(ev); break;
    case "segment": $hearing.textContent = ev.text; break;
    case "status":
      if (ev.lag > 10) $hearing.textContent = `(running ${Math.round(ev.lag)}s behind) ` + $hearing.textContent;
      break;
  }
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/view`);
  ws.onopen = () => { $state.textContent = "connected"; $state.className = ""; };
  ws.onmessage = m => handle(JSON.parse(m.data));
  ws.onclose = () => {
    $state.textContent = "offline"; $state.className = "off";
    setTimeout(connect, 2000);
  };
}
connect();
