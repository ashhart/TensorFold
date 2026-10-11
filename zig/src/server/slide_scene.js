// The 3D view: the facts as a glowing graph around the model's tower, each lesson's phases played out between them.
import * as THREE from "three";
import {colors, factColor, Glow, Lines, Sparks} from "./fx.js";
import {Tower, SHARD_ANGLE} from "./tower.js";
import {Bloom} from "./bloom.js";
import {Graph, Labels, Orbit} from "./layout.js";
import {store, on, select, hover, learning} from "./data.js";
import {tip, pocketCounts} from "./panel.js";

const FOV = 46;
const v = () => new THREE.Vector3();
const frac = (x) => x - Math.floor(x);

export function start(stage) {
  const renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: "high-performance"});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(colors.background, 1);
  stage.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(FOV, 1, 1, 3000);
  const tower = new Tower();
  const bloom = bloomOf(renderer);
  const lines = new Lines(16000), glow = new Glow(9000), sparks = new Sparks(700);
  const far = dust();
  scene.add(far.object, lines.object, glow.object);
  const graph = new Graph();
  const labels = new Labels(document.getElementById("labels"), ["top", "disk", "lesson", "side", "side-open"].map((id) => document.getElementById(id)));
  const orbit = new Orbit(camera, renderer.domElement);
  const tints = new Map();
  const look = {owners: new Map(), used: new Map(), focus: null, colorOf: (f) => tints.get(f.id) || factColor(f), saving: false, saveLine: ""};
  const memo = new Map(), waves = [];
  let entityFocus = null, saveState = null, saveAt = null, picks = [], pointer = null, last = performance.now() / 1000;
  // ?stats: frames a second and the main thread's milliseconds a frame, at the foot of the page
  const stats = new URLSearchParams(location.search).has("stats") ? document.getElementById("stats") : null;
  const meter = {frames: 0, ms: 0, since: 0};

  function resize() {
    document.body.classList.toggle("wide", window.innerWidth > 860);
    const w = stage.clientWidth || window.innerWidth, h = stage.clientHeight || window.innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    const side = document.body.classList.contains("wide") ? Math.min(360, w * 0.3) : 0;
    if (side) camera.setViewOffset(w, h, side / 2, 0, w, h);
    else camera.clearViewOffset();
    camera.updateProjectionMatrix();
    const ratio = renderer.getPixelRatio();
    glow.resize(h * ratio, FOV);
    far.resize(h * ratio, FOV);
    bloom?.setSize(Math.round(w * ratio), Math.round(h * ratio));
    // the tower and its base in height, the graph's breadth in the width left beside the panel
    orbit.fit = Math.max(205, 150 / (0.85 * Math.max(0.35, (w - side) / h)));
  }
  new ResizeObserver(resize).observe(stage);
  resize();

  // What changed since the last poll: a lesson's verdict, a fact kept or dropped, a save begun or ended.
  function sync() {
    tower.build(store.model);
    graph.sync(tower.ready);
    look.owners = new Map();
    tints.clear();
    for (const f of store.facts) {
      tints.set(f.id, factColor(f));
      for (const n of f.neurons || []) look.owners.set(`${n.layer}:${n.neuron}`, f);
    }
    look.used = pocketCounts().used;
    for (const f of store.facts) {
      const was = memo.get(f.id), node = graph.nodes.get(`f:${f.id}`);
      memo.set(f.id, {state: f.state, checks: f.checks || 0, phase: f.phase});
      if (!was || !node) continue;
      const at = node.p.lengthSq() ? node.p.clone() : v().set(node.x, node.y, node.z);
      if ((f.checks || 0) > was.checks) {
        const c = f.recalled ? colors.pass : colors.fail;
        sparks.burst(at, c, f.recalled ? 50 : 34, 22);
        waves.push({at, t0: now(), dur: 1.3, color: c, r: 26});
      }
      if (was.state === "learning" && f.state === "learned") {
        sparks.burst(at, colors.learned, 110, 30, 1.8);
        waves.push({at, t0: now(), dur: 1.8, color: colors.learned, r: 40});
        for (const n of f.neurons || []) tower.light(n.layer, 1);
      }
      if (was.state === "learning" && f.state === "missed") sparks.burst(at, colors.missed, 60, 16);
      if (was.phase !== f.phase && f.phase) sparks.burst(at, colors.spark, 14, 10, 1);
    }
    const save = store.model?.save;
    look.saving = save?.state === "saving";
    look.saveLine = save ? `${save.state}|${save.at}` : "";
    tower.saving(look.saving);
    // a save seen ending, or one begun and ended between two polls
    const ended = save?.state === "saved" && (saveState === "saving" || (saveAt !== null && save.at !== saveAt));
    if (ended) tower.saved(save.files || []);
    saveState = save?.state || null;
    saveAt = save ? save.at : saveAt ?? null;
  }
  on((what) => { if (what === "graph") sync(); });

  function now() { return performance.now() / 1000; }

  // Who is in focus: the chosen fact, else the one under the pointer, else an entity's facts, else the filter's.
  function focus() {
    if (store.selected != null) return new Set([store.selected]);
    if (store.hovered != null) return new Set([store.hovered]);
    if (entityFocus) return store.entities.find((e) => e.key === entityFocus)?.facts || null;
    if (store.query) return new Set(store.facts.filter((f) => f.text.toLowerCase().includes(store.query)).map((f) => f.id));
    return null;
  }

  function frame() {
    const began = performance.now();
    const t = now(), dt = Math.min(0.05, t - last);
    last = t;
    const live = learning(), node = live && graph.nodes.get(`f:${live.id}`);
    // left alone, the camera frames the lesson beside the tower, or the shards while a save writes them
    orbit.aim = node ? Math.atan2(node.z, node.x) + 0.7 : look.saving ? SHARD_ANGLE - 0.9 : null;
    orbit.update(dt);
    camera.updateMatrixWorld(); // this frame's camera for labels and camera-facing rings, not the last frame's
    lines.material.uniforms.near.value = glow.material.uniforms.near.value = orbit.dist - 40;
    graph.tick();
    look.focus = focus();
    lines.begin();
    glow.begin();
    labels.begin();
    tower.draw(lines, glow, t, dt, look, camera);
    picks = [];
    drawGraph(t);
    drawLesson(t);
    drawChosen(t);
    drawWaves(t);
    sparks.draw(glow, dt);
    lines.end();
    glow.end();
    far.object.rotation.y = t * 0.004;
    if (bloom) bloom.render(scene, camera);
    else renderer.render(scene, camera);
    for (const s of tower.labels()) if (project(s.at, s).front) labels.want(s.text, s.sx, s.sy, `label shard${s.hot ? " hot" : ""}`, 60);
    labels.end();
    if (pointer) pick(pointer.x, pointer.y, false);
    if (stats) measure(t, performance.now() - began);
    requestAnimationFrame(frame);
  }

  function measure(t, ms) {
    meter.frames++;
    meter.ms += ms;
    if (t - meter.since < 1) return;
    stats.hidden = false;
    stats.textContent = `${Math.round(meter.frames / (t - meter.since))} fps · ${(meter.ms / meter.frames).toFixed(1)} ms a frame on the main thread`;
    Object.assign(meter, {frames: 0, ms: 0, since: t});
  }

  // A world point's place on the canvas, into `out` as sx and sy; front is false behind the camera.
  const screen = v();
  function project(p, out) {
    screen.copy(p).project(camera);
    out.sx = ((screen.x + 1) / 2) * renderer.domElement.clientWidth;
    out.sy = ((1 - screen.y) / 2) * renderer.domElement.clientHeight;
    out.front = screen.z < 1;
    return out;
  }

  function drawGraph(t) {
    const focusSet = look.focus, c = new THREE.Color(), d = new THREE.Color();
    for (const n of graph.list) {
      n.p.set(n.x + Math.sin(t * 0.5 + n.bob) * 0.6, n.y + Math.sin(t * 0.37 + n.bob * 2) * 0.8, n.z + Math.cos(t * 0.43 + n.bob) * 0.6);
      project(n.p, n);
    }
    for (const [a, b, kind] of graph.edges) {
      const x = graph.nodes.get(a), y = graph.nodes.get(b);
      if (!x || !y) continue;
      const lit = !focusSet || inFocus(x, focusSet) || inFocus(y, focusSet);
      const live = x.ref.state === "learning" || y.ref.state === "learning";
      const alpha = (kind === "link" ? 0.1 : kind === "owner" ? 0.25 : 0.32) * (lit ? 1 : 0.25) * (live ? 2 : 1);
      lines.push(x.p, y.p, nodeColor(x, c), alpha, nodeColor(y, d), alpha);
    }
    for (const n of graph.list) {
      const lit = !focusSet || inFocus(n, focusSet);
      const col = nodeColor(n, c);
      let size, alpha = lit ? 1 : 0.22, halo = 0;
      if (n.kind === "entity") {
        size = n.ref.kind === "you" ? 8 : 4.6 + Math.min(3.5, n.ref.facts.size * 0.5);
        halo = 0.1;
        ring(n.p, size * 0.75, col, 0.55 * alpha, t * 0.4 + n.bob);
      } else {
        const s = n.ref.state;
        size = s === "learning" ? 7 + 1.2 * Math.sin(t * 6) : s === "learned" ? 4.2 : 3.4;
        halo = 0.11;
        if (s === "queued") alpha *= 0.55 + 0.3 * Math.sin(t * 2 + n.bob);
        if (s === "missed") alpha *= 0.55;
        if (s === "learning" || n.ref.id === store.selected) halo = 0.22;
      }
      glow.push(n.p.x, n.p.y, n.p.z, col, size, alpha);
      if (halo) glow.push(n.p.x, n.p.y, n.p.z, col, size * 3.6, halo * alpha);
      n.radius = Math.max(9, size * 2);
      picks.push(n);
      if (!n.front) continue;
      if (n.kind === "entity" && (lit || n.ref.kind === "you")) {
        labels.want(n.ref.label, n.sx, n.sy - 10, `label ${n.ref.kind}${lit ? "" : " dim"}`, n.ref.kind === "you" ? 50 : 10 + n.ref.facts.size);
      } else if (n.kind === "fact" && (n.ref.state === "learning" || n.ref.id === store.selected || n.ref.id === store.hovered)) {
        const text = n.ref.text.length > 64 ? `${n.ref.text.slice(0, 63)}…` : n.ref.text;
        const rank = n.ref.state === "learning" ? 100 : n.ref.id === store.selected ? 90 : 80;
        labels.want(n.ref.phase ? `${text}  ·  ${n.ref.phase}` : text, n.sx, n.sy - 12, `label fact ${n.ref.state}`, rank);
      }
    }
  }

  // A ring facing the camera around `at`, turning slowly, broken in four.
  const right = v(), up = v(), ra = v(), rb = v();
  function ring(at, r, color, alpha, spin) {
    right.setFromMatrixColumn(camera.matrixWorld, 0);
    up.setFromMatrixColumn(camera.matrixWorld, 1);
    for (let k = 0; k < 24; k++) {
      if (k % 6 === 5) continue;
      const a0 = spin + (k / 24) * Math.PI * 2, a1 = spin + ((k + 1) / 24) * Math.PI * 2;
      ra.copy(at).addScaledVector(right, Math.cos(a0) * r).addScaledVector(up, Math.sin(a0) * r);
      rb.copy(at).addScaledVector(right, Math.cos(a1) * r).addScaledVector(up, Math.sin(a1) * r);
      lines.push(ra, rb, color, alpha);
    }
  }

  function inFocus(n, set) {
    if (n.kind === "fact") return set.has(n.ref.id);
    for (const id of n.ref.facts) if (set.has(id)) return true;
    return false;
  }

  function nodeColor(n, out) {
    if (n.kind === "fact") return factColor(n.ref, out);
    return out.copy(n.ref.kind === "you" ? colors.you : n.ref.kind === "source" ? colors.source : colors.entity);
  }

  // The lesson's phase: questions written around it, rows read up the tower, layers tried, outputs fitted, a check.
  function drawLesson(t) {
    const f = learning();
    const n = f && graph.nodes.get(`f:${f.id}`);
    if (!n) return;
    const qs = f.questions || [];
    const count = Math.max(6, Math.min(14, qs.length || 8));
    const writing = f.phase === "writing";
    const sat = v();
    for (let k = 0; k < count; k++) {
      const a = t * 0.9 + (k / count) * Math.PI * 2;
      const r = writing ? 8 + 24 * (1 - frac(t * 0.4 + k / count)) : 9.5;
      const tilt = 1.1 + (k % 3) * 0.35;
      sat.set(Math.cos(a) * r, Math.sin(a) * r * Math.cos(tilt), Math.sin(a) * r * Math.sin(tilt)).add(n.p);
      glow.push(sat.x, sat.y, sat.z, writing ? colors.spark : colors.learning, writing ? 2.4 : 1.8, writing ? 0.6 + 0.4 * Math.sin(t * 9 + k) : 0.95);
      lines.push(n.p, sat, writing ? colors.spark : colors.learning, 0.05, writing ? colors.spark : colors.learning, writing ? 0.45 : 0.2);
      if (!writing) {
        const at = project(sat, {kind: "question", radius: 8, text: qs[k] || ""});
        if (at.front) picks.push(at);
      }
    }
    if (writing) {
      ring(n.p, 13, colors.spark, 0.55, t * 1.4);
      ring(n.p, 18, colors.spark, 0.22, -t * 0.9);
    }
    if (!tower.ready || writing) return;
    const end = v(), mid = v();
    const towerColor = colors.moe;
    if (f.phase === "reading") {
      // each of the lesson's examples runs up through every layer: a beam sweeping the tower from the bottom
      const u = frac(t * 0.9);
      const layer = Math.min(tower.layers - 1, Math.floor(u * tower.layers));
      tower.rim(layer, n.p, end);
      lines.push(n.p, end, colors.learning, 0.35, colors.learning, 1);
      for (const dy of [-0.9, 0.9]) lines.push(n.p, mid.copy(end).setY(end.y + dy), colors.learning, 0.1, towerColor, 0.55);
      tower.light(layer, 1);
      if (layer > 0) tower.light(layer - 1, 0.6);
      for (let k = 0; k < 6; k++) glow.push(...mid.lerpVectors(n.p, end, frac(t * 1.6 + k / 6)).toArray(), colors.learning, 1.5, 0.9);
      glow.push(end.x, end.y, end.z, colors.learning, 6, 0.5);
    } else if (f.phase === "fitting" || (f.phase === "values" && !(f.neurons || []).length)) {
      for (const layer of tower.pockets.keys()) {
        if (layer < tower.layers / 5 || layer > (tower.layers * 7) / 8) continue;
        const flicker = 0.5 + 0.5 * Math.sin(t * 11 + layer * 1.7) * Math.sin(t * 4.3 + layer * 0.9);
        if (flicker < 0.3) continue;
        tower.rim(layer, n.p, end);
        lines.push(n.p, end, colors.learning, 0.05, towerColor, 0.55 * flicker);
        tower.light(layer, 0.35 * flicker);
      }
    } else {
      const checking = f.phase === "checking";
      (f.neurons || []).forEach((x, i) => {
        tower.slot(x.layer, x.neuron, end);
        lines.push(n.p, end, colors.learning, checking ? 0.08 : 0.2, colors.learning, checking ? 0.3 : 0.8);
        if (!checking) for (let k = 0; k < 2; k++) glow.push(...mid.lerpVectors(n.p, end, frac(t * 0.8 + i * 0.37 + k / 2)).toArray(), colors.spark, 1.2, 0.85);
        tower.light(x.layer, checking ? 0.15 : 0.4);
      });
      if (checking && (!waves.length || t - waves[waves.length - 1].t0 > 1.3)) waves.push({at: n.p.clone(), t0: t, dur: 1.2, color: colors.learning, r: 18});
    }
  }

  // Beams from the chosen or pointed-at fact to the neurons it owns.
  function drawChosen() {
    const id = store.selected ?? store.hovered;
    const f = id != null && store.byId.get(id);
    if (!f || f.state === "learning" || !tower.ready) return;
    const n = graph.nodes.get(`f:${f.id}`);
    if (!n) return;
    const end = v(), c = factColor(f);
    for (const x of f.neurons || []) {
      tower.slot(x.layer, x.neuron, end);
      lines.push(n.p, end, c, 0.12, c, 0.55);
      tower.light(x.layer, 0.25);
    }
  }

  // Rings that spread from a node and fade: a check's verdict, a fact kept.
  function drawWaves(t) {
    const right = v().setFromMatrixColumn(camera.matrixWorld, 0), up = v().setFromMatrixColumn(camera.matrixWorld, 1);
    const p = v();
    for (let i = waves.length - 1; i >= 0; i--) {
      const w = waves[i], u = (t - w.t0) / w.dur;
      if (u >= 1) { waves.splice(i, 1); continue; }
      const r = 3 + w.r * Math.sqrt(u);
      for (let k = 0; k < 40; k++) {
        const a = (k / 40) * Math.PI * 2;
        p.copy(w.at).addScaledVector(right, Math.cos(a) * r).addScaledVector(up, Math.sin(a) * r);
        glow.push(p.x, p.y, p.z, w.color, 1.4, (1 - u) * 0.8);
      }
    }
  }

  // The nearest thing under the pointer, said in the tip; `click` chooses it.
  const hit = {sx: 0, sy: 0, front: false};
  function pick(x, y, click) {
    let best = null, bestD = Infinity;
    for (const p of picks) {
      const d = Math.hypot(p.sx - x, p.sy - y);
      if (p.front !== false && d < p.radius && d < bestD) { best = p; bestD = d; }
    }
    if (!best && tower.ready) for (const p of tower.pickables()) {
      project(p.at, hit);
      const d = Math.hypot(hit.sx - x, hit.sy - y);
      if (hit.front && d < (p.kind === "slot" ? 6 : 9) && d < bestD) { best = p; bestD = d; }
    }
    const factId = best?.kind === "fact" ? best.ref.id : best?.kind === "slot" ? look.owners.get(`${best.layer}:${best.neuron}`)?.id ?? null : null;
    hover(factId);
    tip(best ? describe(best) : "", x, y);
    if (!click) return;
    entityFocus = best?.kind === "entity" ? (entityFocus === best.ref.key ? null : best.ref.key) : null;
    if (factId != null) select(factId);
    else if (store.selected != null && !best) select(store.selected);
  }

  function describe(p) {
    if (p.kind === "fact") {
      const f = p.ref, k = (f.neurons || []).length;
      return `${f.text}\n${f.phase ? `${f.state} · ${f.phase}` : f.state}${k ? ` · ${k} neurons` : ""}`;
    }
    if (p.kind === "entity") return `${p.ref.label} · ${p.ref.facts.size} fact${p.ref.facts.size === 1 ? "" : "s"}`;
    if (p.kind === "question") return p.text ? `asked after each round: ${p.text}` : "a question being written";
    if (p.kind === "slot") {
      const owner = look.owners.get(`${p.layer}:${p.neuron}`);
      const head = `layer ${p.layer} · neuron ${p.neuron}`;
      if (owner) return `${head} · keyed on the ${owner.neurons.find((x) => x.layer === p.layer && x.neuron === p.neuron)?.key}\n${owner.text}`;
      return p.neuron < (look.used.get(p.layer) || 0) ? `${head} · learned before this graph` : `${head} · free`;
    }
    const kind = {E: "MoE", M: "Mamba", "*": "attention", "-": "MLP"}[tower.kinds[p.layer]] || "layer";
    const pocket = tower.pockets.has(p.layer) ? ` · pocket ${look.used.get(p.layer) || 0} of ${tower.width} in use` : "";
    return `layer ${p.layer} · ${kind}${pocket}`;
  }

  orbit.onHover = (x, y) => { pointer = x == null ? null : {x, y}; if (!pointer) { hover(null); tip(""); } };
  orbit.onClick = (x, y) => pick(x, y, true);
  sync();
  requestAnimationFrame(frame);
}

// Bloom when the browser can draw into float targets, else none (the plain render, a little less glow).
function bloomOf(renderer) {
  if (!renderer.extensions.has("EXT_color_buffer_float") && !renderer.extensions.has("EXT_color_buffer_half_float")) return null;
  try { return new Bloom(renderer); } catch { return null; }
}

// Dust far behind everything, still against the slow turn of the scene.
function dust() {
  const g = new Glow(420, {fog: 0});
  const c = new THREE.Color("#7f9cff");
  for (let i = 0; i < 420; i++) {
    const u = Math.random() * 2 - 1, a = Math.random() * Math.PI * 2, r = 500 + Math.random() * 600;
    const s = Math.sqrt(1 - u * u);
    g.push(Math.cos(a) * s * r, u * r * 0.6, Math.sin(a) * s * r, c, 1.5 + Math.random() * 3, 0.15 + Math.random() * 0.35);
  }
  g.end();
  return g;
}
