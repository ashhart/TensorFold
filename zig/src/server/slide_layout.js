// The scene's layout and handling: facts and entities placed by forces, labels over the canvas, the camera.
import * as THREE from "three";
import {store} from "./data.js";

const v = () => new THREE.Vector3();

// Facts and entities laid out by forces: they push apart, links pull, and all keep a shell around the tower.
export class Graph {
  constructor() {
    this.nodes = new Map();
    this.list = [];
    this.edges = [];
    this.heat = 1;
    this.shell = 64;
    this.core = 36;
  }

  sync(tower) {
    this.shell = tower ? 58 : 34;
    this.core = tower ? 30 : 0;
    const want = new Map();
    for (const f of store.facts) want.set(`f:${f.id}`, ["fact", f]);
    for (const e of store.entities) want.set(`e:${e.key}`, ["entity", e]);
    const edges = store.mentions.map(([a, b]) => [a, b, a.startsWith("e:") ? "owner" : "mention"]);
    for (const f of store.facts) for (const t of f.links || []) if (t > f.id) edges.push([`f:${f.id}`, `f:${t}`, "link"]);
    let moved = false;
    for (const id of this.nodes.keys()) if (!want.has(id)) { this.nodes.delete(id); moved = true; }
    for (const [id, [kind, ref]] of want) {
      let n = this.nodes.get(id);
      if (!n) {
        n = this.spawn(id, kind, edges);
        this.nodes.set(id, n);
        moved = true;
      }
      n.ref = ref;
    }
    this.list = [...this.nodes.values()];
    this.edges = edges;
    // settled graphs stop ticking; new nodes stir it, and a learning fact keeps it just warm enough to hold its height
    this.heat = Math.max(this.heat, moved ? 0.9 : store.facts.some((f) => f.state === "learning") ? 0.1 : 0);
  }

  spawn(id, kind, edges) {
    const near = edges.find((e) => (e[0] === id && this.nodes.has(e[1])) || (e[1] === id && this.nodes.has(e[0])));
    const other = near && this.nodes.get(near[0] === id ? near[1] : near[0]);
    const a = Math.random() * Math.PI * 2;
    const p = other ? [other.x + (Math.random() - 0.5) * 12, other.y + (Math.random() - 0.5) * 12, other.z + (Math.random() - 0.5) * 12]
      : [Math.cos(a) * this.shell, (Math.random() - 0.5) * 50, Math.sin(a) * this.shell];
    return {id, kind, x: p[0], y: p[1], z: p[2], vx: 0, vy: 0, vz: 0, bob: Math.random() * 6.28, p: v(), mass: kind === "entity" ? 1.6 : 1};
  }

  tick() {
    if (this.heat < 0.002) return;
    const list = this.list, n = list.length, k = this.heat;
    for (let i = 0; i < n; i++) {
      const a = list[i];
      for (let j = i + 1; j < n; j++) {
        const b = list[j];
        let dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        let d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 0.5; }
        if (d2 > 6400) continue;
        // inverse square, held level inside six units so close neighbours part gently
        const f = (140 * a.mass * b.mass) / (Math.max(d2, 36) * Math.sqrt(d2));
        a.vx += (dx * f) / a.mass; a.vy += (dy * f) / a.mass; a.vz += (dz * f) / a.mass;
        b.vx -= (dx * f) / b.mass; b.vy -= (dy * f) / b.mass; b.vz -= (dz * f) / b.mass;
      }
    }
    for (const [ia, ib, kind] of this.edges) {
      const a = this.nodes.get(ia), b = this.nodes.get(ib);
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z, d = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;
      const rest = kind === "link" ? 30 : kind === "owner" ? 20 : 15, s = kind === "link" ? 0.004 : 0.03;
      const f = ((d - rest) * s) / d;
      a.vx += dx * f; a.vy += dy * f; a.vz += dz * f;
      b.vx -= dx * f; b.vy -= dy * f; b.vz -= dz * f;
    }
    for (const a of list) {
      const rho = Math.hypot(a.x, a.z) || 1;
      let pull = (this.shell - rho) * 0.012;
      if (rho < this.core) pull += (this.core - rho) * 0.25;
      a.vx += (a.x / rho) * pull; a.vz += (a.z / rho) * pull;
      a.vy -= a.y * 0.005;
      // the fact being learned keeps to the middle height, where the beams to the tower read best
      if (a.ref?.state === "learning") a.vy += (Math.max(-12, Math.min(26, a.y)) - a.y) * 0.04;
      a.vx *= 0.82; a.vy *= 0.82; a.vz *= 0.82;
      const speed = Math.hypot(a.vx, a.vy, a.vz);
      const cap = speed > 3 ? 3 / speed : 1;
      a.x += a.vx * k * cap; a.y += a.vy * k * cap; a.z += a.vz * k * cap;
    }
    this.heat *= 0.992;
  }
}

// Labels over the canvas, a reused pool placed each frame, the most wanted first, none on another or on `avoid`.
export class Labels {
  constructor(root, avoid = []) { this.root = root; this.avoid = avoid; this.pool = []; this.used = 0; this.wanted = []; }

  begin() { this.used = 0; this.wanted = []; }

  want(text, x, y, cls, rank) { this.wanted.push({text, x, y, cls, rank}); }

  put(text, x, y, cls) {
    let el = this.pool[this.used];
    if (!el) {
      el = document.createElement("div");
      this.root.append(el);
      this.pool.push(el);
    }
    this.used++;
    if (el.textContent !== text) el.textContent = text;
    if (el.className !== cls) el.className = cls;
    el.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -100%)`;
    el.style.display = "";
  }

  end() {
    const boxes = this.avoid.filter((el) => el && !el.hidden && el.getClientRects().length).map((el) => {
      const r = el.getBoundingClientRect();
      return [r.left, r.top, r.right, r.bottom];
    });
    for (const w of this.wanted.sort((a, b) => b.rank - a.rank)) {
      const width = w.text.length * (w.cls.includes("fact") ? 6.4 : 7.2) + 12, height = 18;
      const box = [w.x - width / 2, w.y - height, w.x + width / 2, w.y + 2];
      if (boxes.some((o) => box[0] < o[2] && box[2] > o[0] && box[1] < o[3] && box[3] > o[1])) continue;
      boxes.push(box);
      this.put(w.text, w.x, w.y, w.cls);
    }
    for (let i = this.used; i < this.pool.length; i++) this.pool[i].style.display = "none";
  }
}

// The camera on a sphere around the tower: dragged, pinched or wheeled by hand, turning slowly when left alone.
export class Orbit {
  constructor(camera, el) {
    Object.assign(this, {camera, el, az: 0.75, el0: 0.2, zoom: 1, fit: 220, idle: -1e9, aim: null, onHover: null, onClick: null});
    this.target = new THREE.Vector3(0, 2, 0);
    const down = new Map();
    let press = null, pinch = 0;
    el.style.touchAction = "none";
    el.addEventListener("pointerdown", (e) => {
      down.set(e.pointerId, {x: e.clientX, y: e.clientY});
      press = {x: e.clientX, y: e.clientY, t: performance.now(), moved: false};
      el.setPointerCapture(e.pointerId);
      this.idle = performance.now();
    });
    el.addEventListener("pointermove", (e) => {
      const box = el.getBoundingClientRect(), last = down.get(e.pointerId);
      if (!last) return this.onHover?.(e.clientX - box.left, e.clientY - box.top);
      if (down.size === 2) {
        const [a, b] = [...down.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch) this.zoom = clamp(this.zoom * (pinch / d), 0.35, 2.2);
        pinch = d;
      } else {
        this.az -= (e.clientX - last.x) * 0.006;
        this.el0 = clamp(this.el0 + (e.clientY - last.y) * 0.004, -0.35, 1.2);
      }
      down.set(e.pointerId, {x: e.clientX, y: e.clientY});
      if (press && Math.hypot(e.clientX - press.x, e.clientY - press.y) > 6) press.moved = true;
      this.idle = performance.now();
    });
    const up = (e) => {
      down.delete(e.pointerId);
      pinch = 0;
      const box = el.getBoundingClientRect();
      if (press && !press.moved && performance.now() - press.t < 450) this.onClick?.(e.clientX - box.left, e.clientY - box.top);
      press = null;
    };
    el.addEventListener("pointerup", up);
    el.addEventListener("pointercancel", up);
    el.addEventListener("pointerleave", () => this.onHover?.(null, null));
    el.addEventListener("wheel", (e) => {
      e.preventDefault();
      this.zoom = clamp(this.zoom * Math.exp(e.deltaY * 0.001), 0.35, 2.2);
      this.idle = performance.now();
    }, {passive: false});
  }

  get dist() { return this.fit * this.zoom; }

  update(dt) {
    // a page opened mid-lesson frames it at once; later aims are eased to
    if (this.aim != null && performance.now() < 2500) this.az = this.aim;
    if (performance.now() - this.idle > 6000) {
      if (this.aim == null) this.az += still ? 0 : dt * 0.055;
      else this.az += (wrap(this.aim - this.az) * Math.min(1, dt * 0.5)) + dt * 0.012;
    }
    const d = this.dist, c = Math.cos(this.el0);
    this.camera.position.set(Math.cos(this.az) * c * d, Math.sin(this.el0) * d + this.target.y, Math.sin(this.az) * c * d);
    this.camera.lookAt(this.target);
  }
}

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false; // no idle turning then
const wrap = (a) => a - Math.PI * 2 * Math.floor((a + Math.PI) / (Math.PI * 2));
