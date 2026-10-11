// The page's text: counts, the fact list and its filter, the chosen fact, the running lesson and the shards.
import {store, on, select, filter, matches, learning, bytes, phases} from "./data.js";

const $ = (id) => document.getElementById(id);
const say = {
  writing: "the model writes its lesson: questions, answers, near misses",
  reading: "the lesson's rows read through every layer",
  fitting: "detectors fitted at the layers that tell its questions apart",
  values: "the new neurons' outputs fitted",
  checking: "held-out questions asked",
};

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

const when = (ms) => (ms ? new Date(ms).toLocaleString() : "");
const clock = (s) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// The neurons a pocket holds, each layer's used count, and those the graph's facts account for.
export function pocketCounts() {
  const m = store.model;
  const width = m?.pocket?.neurons || 64;
  const layers = m?.pocket?.layers || [];
  const used = new Map(layers.map((l, i) => [l, m.pocket.used?.[i] || 0]));
  for (const f of store.facts) for (const n of f.neurons || []) used.set(n.layer, Math.max(used.get(n.layer) || 0, n.neuron + 1));
  let total = 0;
  for (const v of used.values()) total += v;
  return {width, layers, used, total, slots: layers.length * width};
}

function header() {
  const by = (s) => store.facts.filter((f) => f.state === s).length;
  const counts = $("counts");
  if (!store.ok) {
    counts.textContent = store.error || "loading";
    return;
  }
  const p = pocketCounts();
  counts.textContent = `${by("learned")} learned · ${by("missed")} missed · ${by("learning") + by("queued")} in progress` +
    (p.layers.length ? ` · ${p.total} of ${p.slots} neurons` : "");
  const m = store.model;
  $("model-line").textContent = m ? `${m.layers} layers · ${p.layers.length} pockets of ${p.width} neurons` : "";
}

let listed = "";

function list() {
  const ol = $("list");
  const rows = store.facts.filter(matches).sort((a, b) => b.id - a.id);
  const sig = `${store.selected}|${rows.map((f) => `${f.id}${f.state}${f.phase}${(f.neurons || []).length}`).join(",")}`;
  if (sig === listed) return;
  listed = sig;
  const focused = ol.contains(document.activeElement) ? document.activeElement.dataset.id : null;
  ol.replaceChildren(...rows.map((f) => {
    const li = el("li", `${f.state}${f.id === store.selected ? " on" : ""}`);
    li.dataset.id = f.id;
    li.append(el("span", "fact", f.text));
    const n = (f.neurons || []).length;
    const parts = [f.phase || f.state, f.source || "chat", when(f.at)];
    if (n) parts.splice(1, 0, `${n} neurons`);
    li.append(el("small", null, parts.join(" · ")));
    li.onclick = () => select(f.id);
    li.tabIndex = 0;
    li.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(f.id); } };
    li.onmouseenter = () => { store.hovered = f.id; };
    li.onmouseleave = () => { if (store.hovered === f.id) store.hovered = null; };
    return li;
  }));
  if (!rows.length) ol.append(el("li", "empty", store.facts.length ? "No fact matches." : "Nothing learned yet. Teach it with /learn."));
  if (focused) ol.querySelector(`li[data-id="${focused}"]`)?.focus();
}

let detailed = "";

function detail() {
  const box = $("detail");
  const f = store.byId.get(store.selected);
  const sig = f ? JSON.stringify([f.id, f.state, f.phase, f.round, f.recalled, f.message, f.neurons, f.questions]) : "";
  if (sig === detailed) return;
  detailed = sig;
  box.hidden = !f;
  if (!f) return box.replaceChildren();
  const head = el("div", "detail-head");
  head.append(el("span", `badge ${f.state}`, f.phase ? `${f.state} · ${f.phase}` : f.state));
  const close = el("button", "x", "×");
  close.setAttribute("aria-label", "Close");
  close.onclick = () => select(f.id);
  head.append(close);
  const facts = el("dl");
  const add = (k, v) => { if (v) facts.append(el("dt", null, k), el("dd", null, String(v))); };
  add("source", f.source || "chat");
  add("taught", when(f.at));
  add("rounds", f.round);
  add("last check", f.recalled == null ? "" : f.recalled ? "recalled" : "not recalled");
  box.replaceChildren(head, el("p", "detail-text", f.text), facts);
  if (f.message) box.append(el("p", "note", f.message));
  if ((f.neurons || []).length) {
    box.append(el("h3", null, `Neurons in the pocket (${f.neurons.length})`));
    const chips = el("div", "chips");
    for (const n of [...f.neurons].sort((a, b) => a.layer - b.layer)) chips.append(el("span", `chip ${n.key}`, `L${n.layer} · #${n.neuron} · ${n.key}`));
    box.append(chips);
  }
  if ((f.questions || []).length) {
    box.append(el("h3", null, "Checked with"));
    const ul = el("ul", "asked");
    for (const q of f.questions) ul.append(el("li", null, q));
    box.append(ul);
  }
}

let shownLesson = null;

function lesson() {
  const box = $("lesson");
  const f = learning();
  box.hidden = !f;
  shownLesson = f;
  if (!f) return;
  const head = el("div", "lesson-head");
  head.append(el("span", "pulse"), el("b", null, "LEARNING"), el("span", null, f.round ? `round ${f.round}` : ""), el("time", "since", ""));
  const pills = el("ol", "pills");
  const at = phases.indexOf(f.phase);
  phases.forEach((p, i) => pills.append(el("li", i < at ? "done" : i === at ? "now" : "", p)));
  const meter = el("div", f.of ? "meter" : "meter busy");
  const fill = el("span");
  fill.style.width = f.of ? `${Math.min(100, (100 * f.step) / f.of)}%` : "100%";
  meter.append(fill);
  const row = el("div", "row");
  const unit = f.phase === "values" ? `step ${f.step}` : f.of ? `${f.step} of ${f.of}` : "";
  row.append(el("span", null, `${say[f.phase] || f.phase || ""}${unit ? ` · ${unit}` : ""}`));
  if (f.loss != null) row.append(el("span", "loss", `loss ${f.loss.toFixed(3)}`));
  const spark = el("canvas", "spark");
  spark.width = 560;
  spark.height = 90;
  const foot = el("div", "row");
  const n = (f.neurons || []).length;
  const layers = [...new Set((f.neurons || []).map((x) => x.layer))].sort((a, b) => a - b);
  foot.append(el("span", null, n ? `${n} neurons at layers ${layers.join(", ")}` : "neurons not placed yet"));
  if (f.recalled != null) foot.append(el("span", `verdict ${f.recalled ? "ok" : "no"}`, f.recalled ? "✓ recalled" : "✗ not yet"));
  box.replaceChildren(head, el("p", "lesson-text", f.text), pills, meter, row, spark);
  if (f.trace?.length) sparkline(spark, f.trace);
  else spark.hidden = true;
  box.append(foot);
  if (f.message) box.append(el("p", "note", f.message));
  tick();
}

// The loss of the outputs at each step, every round in turn, on a log scale; the newest point lit.
function sparkline(canvas, rounds) {
  const g = canvas.getContext("2d");
  const all = rounds.flat().filter((x) => x > 0);
  if (!all.length) return;
  const w = canvas.width, h = canvas.height, pad = 8;
  const lo = Math.log(Math.min(...all)), hi = Math.log(Math.max(...all));
  const total = rounds.reduce((n, r) => n + r.length, 0);
  const x = (i) => pad + ((w - 2 * pad) * i) / Math.max(1, total - 1);
  const y = (v) => pad + (h - 2 * pad) * (hi === lo ? 0.5 : (hi - Math.log(Math.max(v, 1e-6))) / (hi - lo));
  g.clearRect(0, 0, w, h);
  let i = 0;
  g.lineWidth = 2;
  for (const r of rounds) {
    if (i > 0) {
      g.strokeStyle = "rgba(160,190,255,.25)";
      g.setLineDash([3, 4]);
      g.beginPath(); g.moveTo(x(i), 2); g.lineTo(x(i), h - 2); g.stroke();
      g.setLineDash([]);
    }
    g.strokeStyle = "#7ff3ff";
    g.shadowColor = "#38d9ff";
    g.shadowBlur = 8;
    g.beginPath();
    r.forEach((v, k) => (k ? g.lineTo(x(i + k), y(v)) : g.moveTo(x(i + k), y(v))));
    g.stroke();
    g.shadowBlur = 0;
    i += r.length;
  }
  const last = rounds[rounds.length - 1];
  if (last.length) {
    g.fillStyle = "#ffffff";
    g.beginPath(); g.arc(x(total - 1), y(last[last.length - 1]), 3.5, 0, Math.PI * 2); g.fill();
  }
}

// The running lesson's clock, between polls.
function tick() {
  const f = shownLesson;
  const t = document.querySelector("#lesson .since");
  if (f && t && f.since) t.textContent = clock(Math.max(0, (Date.now() - f.since) / 1000));
}

function disk() {
  const box = $("disk");
  const m = store.model;
  box.hidden = !m;
  if (!m) return;
  const shards = m.shards || [];
  const size = shards.reduce((n, s) => n + (s.bytes || 0), 0);
  const holding = shards.filter((s) => s.pocket).length;
  const head = el("div", "disk-head");
  head.append(el("b", null, "SHARDS"), el("span", null, `${shards.length} files · ${bytes(size)}`));
  const rows = [head];
  rows.push(el("p", null, holding ? `the pocket is written into ${holding} of them` : "no pocket written into them yet"));
  const s = m.save;
  if (s?.state === "saving") rows.push(el("p", "save saving", "writing the pocket into the shards…"));
  else if (s?.state === "saved") rows.push(el("p", "save", s.files?.length ? `last save ${bytes(s.bytes)} into ${s.files.length} files in ${s.seconds.toFixed(1)} s · ${new Date(s.at).toLocaleTimeString()}` : `last save: nothing new to write · ${new Date(s.at).toLocaleTimeString()}`));
  else if (s?.state === "failed") rows.push(el("p", "save failed", `the last save failed: ${s.message || "unknown"}`));
  box.replaceChildren(...rows);
}

export function tip(text, x, y) {
  const t = $("tip");
  if (!text) { t.hidden = true; return; }
  t.textContent = text;
  t.hidden = false;
  const w = t.offsetWidth, h = t.offsetHeight;
  t.style.left = `${Math.min(window.innerWidth - w - 8, x + 14)}px`;
  t.style.top = `${Math.max(8, Math.min(window.innerHeight - h - 8, y + 14))}px`;
}

export function start() {
  const find = $("find");
  find.addEventListener("input", () => filter(find.value));
  $("side-open").onclick = () => document.body.classList.add("side-shown");
  $("side-close").onclick = () => document.body.classList.remove("side-shown");
  on((what) => {
    if (what === "graph") { header(); list(); detail(); lesson(); disk(); }
    if (what === "select") { list(); detail(); }
    if (what === "filter") list();
  });
  setInterval(tick, 500);
}
