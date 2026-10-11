// The flat view, for browsers without WebGL or three.js: facts as dots, linked by shared words, dragged by hand.
import {store, on, select} from "./data.js";

const NS = "http://www.w3.org/2000/svg";
const COLOR = {learned: "--learned", missed: "--missed", learning: "--learning", queued: "--queued"};

function make(name, attrs) {
  const node = document.createElementNS(NS, name);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
}

export function start(stage) {
  const svg = make("svg", {role: "img", "aria-label": "Facts learned into the model's weights", class: "flat"});
  stage.append(svg);
  const nodes = new Map();
  let links = [], drag = null, heat = 0;
  const color = (state) => getComputedStyle(document.documentElement).getPropertyValue(COLOR[state] || "--muted");

  on((what) => {
    if (what !== "graph") return;
    const {width, height} = svg.getBoundingClientRect();
    for (const f of store.facts) {
      const n = nodes.get(f.id) || {x: width / 2 + (Math.random() - 0.5) * 80, y: height / 2 + (Math.random() - 0.5) * 80, vx: 0, vy: 0};
      nodes.set(f.id, Object.assign(n, f));
      if (n.state === "learning") heat = 1;
    }
    for (const id of nodes.keys()) if (!store.byId.has(id)) nodes.delete(id);
    links = store.facts.flatMap((f) => (f.links || []).filter((t) => nodes.has(t) && t > f.id).map((t) => [f.id, t]));
    heat = Math.max(heat, 0.3);
  });

  function step() {
    const all = [...nodes.values()], {width, height} = svg.getBoundingClientRect();
    for (const a of all) for (const b of all) if (a !== b) {
      const dx = a.x - b.x, dy = a.y - b.y, d2 = Math.max(dx * dx + dy * dy, 60);
      a.vx += (dx / d2) * 900; a.vy += (dy / d2) * 900;
    }
    for (const [i, j] of links) {
      const a = nodes.get(i), b = nodes.get(j), dx = b.x - a.x, dy = b.y - a.y, k = 0.004;
      a.vx += dx * k; a.vy += dy * k; b.vx -= dx * k; b.vy -= dy * k;
    }
    for (const n of all) {
      if (n === drag) continue;
      n.vx = (n.vx + (width / 2 - n.x) * 0.002) * 0.82; n.vy = (n.vy + (height / 2 - n.y) * 0.002) * 0.82;
      // a step of 12 px at most, inside the view: close neighbours part without flying off
      const speed = Math.hypot(n.vx, n.vy), cap = speed > 12 ? 12 / speed : 1;
      n.x = Math.min(width - 16, Math.max(16, n.x + n.vx * heat * cap));
      n.y = Math.min(height - 16, Math.max(16, n.y + n.vy * heat * cap));
    }
    heat = Math.max(heat * 0.985, all.some((n) => n.state === "learning") ? 0.3 : 0);
  }

  function draw() {
    svg.replaceChildren();
    if (!nodes.size) {
      const {width, height} = svg.getBoundingClientRect();
      const note = make("text", {x: width / 2, y: height / 2, "text-anchor": "middle", class: "empty"});
      note.textContent = "Nothing learned yet. Teach it with /learn.";
      svg.append(note);
      return;
    }
    for (const [i, j] of links) {
      const a = nodes.get(i), b = nodes.get(j);
      svg.append(make("line", {x1: a.x, y1: a.y, x2: b.x, y2: b.y}));
    }
    for (const n of nodes.values()) {
      const pulse = n.state === "learning" ? 2 * Math.sin(performance.now() / 160) : 0;
      const dot = make("circle", {cx: n.x, cy: n.y, r: (n.id === store.selected ? 10 : 7) + pulse, fill: color(n.state)});
      dot.addEventListener("pointerdown", (e) => { drag = n; select(n.id); svg.setPointerCapture(e.pointerId); });
      const label = make("text", {x: n.x + 12, y: n.y + 4});
      const extra = n.phase ? ` (${n.phase})` : "";
      label.textContent = (n.text.length > 42 ? n.text.slice(0, 41) + "…" : n.text) + extra;
      svg.append(dot, label);
    }
  }

  svg.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const box = svg.getBoundingClientRect();
    drag.x = e.clientX - box.left; drag.y = e.clientY - box.top; heat = Math.max(heat, 0.4);
  });
  svg.addEventListener("pointerup", () => { drag = null; });
  (function frame() { if (heat > 0) step(); draw(); requestAnimationFrame(frame); })();
}
