// The model drawn: layers as rings on a spine, each MoE layer's pocket of neuron slots, and the shards that hold them.
import * as THREE from "three";
import {colors, ease} from "./fx.js";

const GAP = 2.2, MOE_R = 11, ATT_R = 7.5, MAMBA_R = 5.5;
const SHARD_R = 38, SHARD_W = 12;
export const SHARD_ANGLE = -Math.PI * 0.62;
const v = () => new THREE.Vector3();

export class Tower {
  constructor() {
    this.sig = "";
    this.layers = 0;
    this.kinds = "";
    this.width = 64;
    this.pockets = new Map();
    this.panels = [];
    this.picks = [];
    this.packets = [];
    this.lit = new Map();
    this.tmp = [v(), v(), v(), v(), v(), v()];
  }

  y(layer) { return (layer - (this.layers - 1) / 2) * GAP; }
  get top() { return this.y(this.layers - 1); }
  get bottom() { return this.y(0); }
  get ready() { return this.layers > 0; }

  // A pocket slot's place: its layer's ring, turned a little at each layer so the pockets climb as a helix.
  slot(layer, neuron, out) {
    const a = (neuron / this.width) * Math.PI * 2 + layer * 0.15;
    return out.set(Math.cos(a) * MOE_R, this.y(layer), Math.sin(a) * MOE_R);
  }

  // A point on a layer's ring facing `toward` (the scene's x and z).
  rim(layer, toward, out) {
    const a = Math.atan2(toward.z, toward.x);
    const r = this.kinds[layer] === "E" ? MOE_R : this.kinds[layer] === "*" ? ATT_R : MAMBA_R;
    return out.set(Math.cos(a) * r, this.y(layer), Math.sin(a) * r);
  }

  // A layer glows for a moment, as a read passes it or a save writes it.
  light(layer, strength = 1) { this.lit.set(layer, Math.max(this.lit.get(layer) || 0, strength)); }

  build(model) {
    const width = model?.pocket?.neurons || 64;
    const shards = model?.shards || [];
    const sig = JSON.stringify([model?.kinds, width, shards.map((s) => [s.name, s.layers])]);
    if (sig !== this.sig) {
      this.sig = sig;
      this.layers = model?.layers || 0;
      this.kinds = model?.kinds || "";
      this.width = width;
      this.pockets = new Map((model?.pocket?.layers || []).map((l, i) => [l, i]));
      this.panels = this.ready ? this.place(shards) : [];
      this.picks = [];
      for (const [layer] of this.pockets) for (let n = 0; n < this.width; n++) this.picks.push({kind: "slot", layer, neuron: n, at: this.slot(layer, n, v())});
      for (let l = 0; l < this.layers; l++) this.picks.push({kind: "layer", layer: l, at: v().set(0, this.y(l), 0)});
    }
    for (const p of this.panels) p.shard = shards.find((s) => s.name === p.shard.name) || p.shard;
  }

  // Each shard beside the tower, as tall as the layers it holds; one without layers (the MTP head's) above it.
  place(shards) {
    let spare = 0;
    return shards.map((shard) => {
      const ls = shard.layers || [];
      const y0 = ls.length ? this.y(Math.min(...ls)) - GAP * 0.6 : this.top + 6 + spare * 9;
      const y1 = ls.length ? this.y(Math.max(...ls)) + GAP * 0.6 : y0 + 6;
      if (!ls.length) spare++;
      const center = v().set(Math.cos(SHARD_ANGLE) * SHARD_R, (y0 + y1) / 2, Math.sin(SHARD_ANGLE) * SHARD_R);
      return {shard, y0, y1, center, flash: 0, writing: false};
    });
  }

  // The panel that holds a layer's shared expert, if any.
  panelOf(layer) { return this.panels.find((p) => p.shard.layers?.includes(layer)) || null; }

  // A save starts or ends: packets fly from the pockets in use to their shards; the files it wrote flash.
  saving(on) { for (const p of this.panels) p.writing = on && (p.shard.layers?.length || 0) > 0; }

  saved(files) { for (const p of this.panels) if (files.includes(p.shard.name)) p.flash = 1; }

  // The tower's frame into `lines` and `glow`: `look` says who owns each slot and what is lit; shards face `camera`.
  draw(lines, glow, t, dt, look, camera) {
    if (!this.ready) return;
    const [a, b] = this.tmp;
    const base = this.bottom - 7;
    for (let r = 14; r <= 84; r += 14) this.circle(lines, base, r, 96, colors.moe, 0.16 * (1 - r / 100));
    for (let k = 0; k < 24; k++) {
      const ang = (k / 24) * Math.PI * 2;
      lines.push(a.set(Math.cos(ang) * 14, base, Math.sin(ang) * 14), b.set(Math.cos(ang) * 84, base, Math.sin(ang) * 84), colors.moe, 0.12, colors.moe, 0);
    }
    this.circle(lines, base, MOE_R + 3, 64, colors.learning, 0.5);
    glow.push(0, base, 0, colors.moe, 26, 0.12);
    lines.push(a.set(0, base, 0), b.set(0, this.top + 6, 0), colors.learning, 0.5, colors.moe, 0.1);
    for (let k = 0; k < 8; k++) glow.push(0, base + ((t * 22 + k * 17) % (this.top - base + 8)), 0, colors.learning, 1.8, 0.7);
    for (const [layer, s] of this.lit) {
      if (s < 0.02) this.lit.delete(layer);
      else this.lit.set(layer, s * Math.exp(-dt * 2.2));
    }
    for (let l = 0; l < this.layers; l++) {
      const kind = this.kinds[l];
      const wave = 0.5 + 0.5 * Math.sin(t * 1.25 - l * 0.33);
      const lit = this.lit.get(l) || 0;
      const y = this.y(l);
      if (kind === "E") {
        this.circle(lines, y, MOE_R, 64, colors.moe, 0.3 + 0.2 * wave + 0.6 * lit);
        this.circle(lines, y, MOE_R + 0.7, 64, colors.moe, 0.08 + 0.3 * lit);
        if (lit > 0.05) glow.push(0, y, 0, colors.moe, MOE_R * 2.6, 0.18 * lit);
      } else if (kind === "*") this.circle(lines, y, ATT_R, 6, colors.attention, 0.45 + 0.2 * wave + 0.5 * lit);
      else this.circle(lines, y, MAMBA_R, 32, colors.mamba, 0.3 + 0.15 * wave + 0.4 * lit);
    }
    this.slots(glow, t, look);
    this.shards(lines, glow, t, dt, look, camera);
    this.flights(glow, t, look);
  }

  circle(lines, y, r, n, color, alpha) {
    const [a, b] = this.tmp;
    for (let k = 0; k < n; k++) {
      const u = (k / n) * Math.PI * 2, w = ((k + 1) / n) * Math.PI * 2;
      lines.push(a.set(Math.cos(u) * r, y, Math.sin(u) * r), b.set(Math.cos(w) * r, y, Math.sin(w) * r), color, alpha);
    }
  }

  slots(glow, t, look) {
    const p = this.tmp[2];
    for (const [layer] of this.pockets) {
      const used = look.used.get(layer) || 0;
      for (let n = 0; n < this.width; n++) {
        this.slot(layer, n, p);
        const owner = look.owners.get(`${layer}:${n}`);
        if (owner) {
          const live = owner.state === "learning";
          const pulse = live ? 0.75 + 0.25 * Math.sin(t * 7 + n) : 1;
          const dim = look.focus && !look.focus.has(owner.id) ? 0.25 : 1;
          glow.push(p.x, p.y, p.z, look.colorOf(owner), live ? 3 * pulse : 2, dim);
        } else if (n < used) glow.push(p.x, p.y, p.z, colors.unknown, 1.4, 0.6);
        else glow.push(p.x, p.y, p.z, colors.slot, 0.8, 0.5);
      }
    }
  }

  // Each shard: a frame with a line per layer it holds (longer as its pocket fills), threads, a bar while it writes.
  shards(lines, glow, t, dt, look, camera) {
    const [a, b, , , side, tip] = this.tmp;
    for (const p of this.panels) {
      p.flash = Math.max(0, p.flash - dt * 0.5);
      const c = p.center, half = SHARD_W / 2;
      side.set(camera.position.z - c.z, 0, c.x - camera.position.x).normalize();
      const hot = Math.max(p.flash, p.writing ? 0.55 + 0.45 * Math.sin(t * 9) : 0);
      const corner = (x, y, out) => out.copy(c).addScaledVector(side, x).setY(y);
      const outline = [[-half, p.y1 - 2], [-half + 2, p.y1], [half, p.y1], [half, p.y0 + 2], [half - 2, p.y0], [-half, p.y0]];
      outline.forEach(([x, y], i) => {
        const [x2, y2] = outline[(i + 1) % outline.length];
        lines.push(corner(x, y, a), corner(x2, y2, b), colors.shard, 0.45 + 0.55 * hot);
      });
      for (const layer of p.shard.layers || []) {
        const fill = Math.min(1, (look.used.get(layer) || 0) / this.width);
        const y = this.y(layer);
        const lit = this.lit.get(layer) || 0;
        lines.push(corner(-half + 1.2, y, a), corner(-half + 1.2 + (SHARD_W - 2.4) * Math.max(0.12, fill), y, b), colors.shard, 0.35 + 0.5 * Math.max(fill, lit, hot * 0.6));
        this.rim(layer, c, a);
        tip.copy(c).setY(y).lerp(a, 0.08);
        lines.push(a, tip, colors.moe, look.saving ? 0.25 + 0.2 * Math.sin(t * 9 + layer) : 0.05, colors.shard, look.saving ? 0.3 : 0.04);
      }
      if (p.writing) {
        const y = p.y1 - frac(t * 0.9) * (p.y1 - p.y0);
        lines.push(corner(-half, y, a), corner(half, y, b), colors.learning, 0.95);
        glow.push(c.x, y, c.z, colors.shard, 7, 0.25);
      }
    }
  }

  // While a save writes: packets from used slots to their layer's shard, a few a frame.
  flights(glow, t, look) {
    const layers = look.saving ? [...this.pockets.keys()].filter((l) => (look.used.get(l) || 0) > 0 && this.panelOf(l)) : [];
    for (let k = 0; layers.length && this.packets.length < 240 && k < 4; k++) {
      const layer = layers[Math.floor(Math.random() * layers.length)];
      const from = this.slot(layer, Math.floor(Math.random() * (look.used.get(layer) || 1)), v());
      const to = this.panelOf(layer).center.clone().setY(this.y(layer));
      this.packets.push({from, to, t0: t, dur: 0.6 + Math.random() * 0.5, layer});
    }
    const p = this.tmp[3];
    this.packets = this.packets.filter((k) => {
      const u = (t - k.t0) / k.dur;
      if (u >= 1) {
        this.light(k.layer, 0.5);
        return false;
      }
      p.lerpVectors(k.from, k.to, ease(u));
      p.y += Math.sin(u * Math.PI) * 2.5;
      glow.push(p.x, p.y, p.z, colors.shard, 1.6, 0.95 * (1 - u * 0.4));
      return true;
    });
  }

  // What each shard's label says, and where: above its frame.
  labels() {
    return this.panels.map((p) => {
      const s = p.shard, name = s.name.replace(/\.safetensors$/, "");
      const short = name.match(/(\d+)-of-(\d+)$/);
      const title = short ? `shard ${Number(short[1])}/${Number(short[2])}` : name;
      const state = p.writing ? " · writing" : p.flash > 0.3 ? " · saved" : s.pocket ? " · pocket" : "";
      return {text: `${title} · ${gb(s.bytes)}${state}`, at: p.center.clone().setY(p.y1 + 1.2), hot: p.writing || p.flash > 0.3};
    });
  }

  // Pickable things under the pointer: each pocket slot and each layer's centre.
  pickables() { return this.picks; }
}

const frac = (x) => x - Math.floor(x);

function gb(n) { return n >= 1e9 ? `${(n / 1e9).toFixed(2)} GB` : `${(n / 1e6).toFixed(1)} MB`; }
