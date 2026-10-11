// The scene's look: its colours, glowing points, additive lines with fog, and bursts of sparks.
import * as THREE from "three";

export const colors = {
  background: new THREE.Color("#02050b"),
  queued: new THREE.Color("#f6b445"),
  learning: new THREE.Color("#a8f7ff"),
  learned: new THREE.Color("#3ff0a8"),
  missed: new THREE.Color("#ff4f72"),
  entity: new THREE.Color("#b7a2ff"),
  you: new THREE.Color("#ffffff"),
  source: new THREE.Color("#6fa8ff"),
  mamba: new THREE.Color("#2c4a8a"),
  moe: new THREE.Color("#2fb6ff"),
  attention: new THREE.Color("#b97dff"),
  slot: new THREE.Color("#1d4f8f"),
  unknown: new THREE.Color("#c8d6ff"),
  shard: new THREE.Color("#57e6d2"),
  pass: new THREE.Color("#62ff9d"),
  fail: new THREE.Color("#ff4060"),
  spark: new THREE.Color("#ffd27a"),
};

// A learned fact's own shade: the learned green turned a little by its id, so neighbours tell apart.
export function factColor(fact, out = new THREE.Color()) {
  const state = fact.state === "queued" ? "queued" : fact.state;
  out.copy(colors[state] || colors.unknown);
  if (fact.state === "learned") out.offsetHSL(((fact.id * 0.618034) % 1 - 0.5) * 0.12, 0, 0);
  return out;
}

const fogged = `
  uniform float fog;
  uniform float near;
  float fade(float depth) { float d = max(depth - near, 0.0); return exp(-fog * fog * d * d); }
`;

const pointVertex = `
  attribute vec3 tint;
  attribute float size;
  attribute float alpha;
  uniform float scale;
  varying vec3 vTint;
  varying float vAlpha;
  ${fogged}
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    float depth = max(-mv.z, 0.1);
    gl_PointSize = clamp(size * scale / depth, 1.5, 320.0);
    vTint = tint;
    vAlpha = alpha * fade(depth);
  }
`;

const pointFragment = `
  uniform float core;
  varying vec3 vTint;
  varying float vAlpha;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0 || vAlpha < 0.003) discard;
    float hot = smoothstep(core, 0.0, r2);
    float glow = exp(-r2 * 4.5) * (1.0 - r2);
    float a = (hot * 0.95 + glow * 0.7) * vAlpha;
    gl_FragColor = vec4(mix(vTint, vec3(1.0), hot * 0.6), a);
  }
`;

const lineVertex = `
  attribute vec3 tint;
  attribute float alpha;
  varying vec3 vTint;
  varying float vAlpha;
  ${fogged}
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    vTint = tint;
    vAlpha = alpha * fade(max(-mv.z, 0.1));
  }
`;

const lineFragment = `
  varying vec3 vTint;
  varying float vAlpha;
  void main() {
    if (vAlpha < 0.003) discard;
    gl_FragColor = vec4(vTint, vAlpha);
  }
`;

function buffers(geometry, capacity, perVertex) {
  const out = {};
  for (const [name, width] of Object.entries(perVertex)) {
    out[name] = new Float32Array(capacity * width);
    const attribute = new THREE.BufferAttribute(out[name], width);
    attribute.setUsage(THREE.DynamicDrawUsage);
    geometry.setAttribute(name, attribute);
  }
  geometry.setDrawRange(0, 0);
  return out;
}

// Points that glow, each with its own colour, size in world units and alpha; one draw call for all.
export class Glow {
  constructor(capacity, {fog = 0.0042, core = 0.08} = {}) {
    this.capacity = capacity;
    this.count = 0;
    this.geometry = new THREE.BufferGeometry();
    this.data = buffers(this.geometry, capacity, {position: 3, tint: 3, size: 1, alpha: 1});
    this.material = new THREE.ShaderMaterial({
      uniforms: {scale: {value: 600}, fog: {value: fog}, near: {value: 150}, core: {value: core}},
      vertexShader: pointVertex, fragmentShader: pointFragment,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.object = new THREE.Points(this.geometry, this.material);
    this.object.frustumCulled = false;
  }

  begin() { this.count = 0; }

  push(x, y, z, color, size, alpha) {
    if (this.count >= this.capacity) return;
    const i = this.count++, d = this.data;
    d.position[i * 3] = x; d.position[i * 3 + 1] = y; d.position[i * 3 + 2] = z;
    d.tint[i * 3] = color.r; d.tint[i * 3 + 1] = color.g; d.tint[i * 3 + 2] = color.b;
    d.size[i] = size;
    d.alpha[i] = alpha;
  }

  end() {
    this.geometry.setDrawRange(0, this.count);
    for (const name of ["position", "tint", "size", "alpha"]) {
      const attribute = this.geometry.getAttribute(name);
      attribute.clearUpdateRanges();
      attribute.addUpdateRange(0, this.count * attribute.itemSize);
      attribute.needsUpdate = true;
    }
  }

  // The pixels a world unit spans at depth one, for the canvas's height and field of view.
  resize(height, fov) { this.material.uniforms.scale.value = height / (2 * Math.tan((fov * Math.PI) / 360)); }

  dispose() { this.geometry.dispose(); this.material.dispose(); }
}

// Line segments drawn additively, each end with its own colour and alpha; one draw call for all.
export class Lines {
  constructor(capacity, {fog = 0.0042} = {}) {
    this.capacity = capacity;
    this.count = 0;
    this.geometry = new THREE.BufferGeometry();
    this.data = buffers(this.geometry, capacity * 2, {position: 3, tint: 3, alpha: 1});
    this.material = new THREE.ShaderMaterial({
      uniforms: {fog: {value: fog}, near: {value: 150}}, vertexShader: lineVertex, fragmentShader: lineFragment,
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.object = new THREE.LineSegments(this.geometry, this.material);
    this.object.frustumCulled = false;
  }

  begin() { this.count = 0; }

  push(a, b, colorA, alphaA, colorB = colorA, alphaB = alphaA) {
    if (this.count >= this.capacity) return;
    const i = this.count++ * 2, p = this.data.position, c = this.data.tint, k = i * 3;
    p[k] = a.x; p[k + 1] = a.y; p[k + 2] = a.z; p[k + 3] = b.x; p[k + 4] = b.y; p[k + 5] = b.z;
    c[k] = colorA.r; c[k + 1] = colorA.g; c[k + 2] = colorA.b; c[k + 3] = colorB.r; c[k + 4] = colorB.g; c[k + 5] = colorB.b;
    this.data.alpha[i] = alphaA;
    this.data.alpha[i + 1] = alphaB;
  }

  end() {
    this.geometry.setDrawRange(0, this.count * 2);
    for (const name of ["position", "tint", "alpha"]) {
      const attribute = this.geometry.getAttribute(name);
      attribute.clearUpdateRanges();
      attribute.addUpdateRange(0, this.count * 2 * attribute.itemSize);
      attribute.needsUpdate = true;
    }
  }

  dispose() { this.geometry.dispose(); this.material.dispose(); }
}

// Sparks that fly out and fade: a fixed pool, the oldest reused first.
export class Sparks {
  constructor(capacity) {
    this.pool = Array.from({length: capacity}, () => ({life: 0, x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, color: new THREE.Color(), size: 1}));
    this.next = 0;
  }

  burst(at, color, count, speed = 18, size = 1.4) {
    for (let k = 0; k < count; k++) {
      const s = this.pool[this.next];
      this.next = (this.next + 1) % this.pool.length;
      const u = Math.random() * 2 - 1, t = Math.random() * Math.PI * 2, r = Math.sqrt(1 - u * u);
      const v = speed * (0.35 + Math.random() * 0.65);
      Object.assign(s, {life: 1, x: at.x, y: at.y, z: at.z, vx: r * Math.cos(t) * v, vy: u * v, vz: r * Math.sin(t) * v, size});
      s.color.copy(color);
    }
  }

  // Moves the live sparks on by `dt` seconds and draws them into `glow`.
  draw(glow, dt) {
    const drag = Math.exp(-2.4 * dt);
    for (const s of this.pool) {
      if (s.life <= 0) continue;
      s.life -= dt * 0.9;
      s.vx *= drag; s.vy *= drag; s.vz *= drag;
      s.x += s.vx * dt; s.y += s.vy * dt; s.z += s.vz * dt;
      if (s.life > 0) glow.push(s.x, s.y, s.z, s.color, s.size * (0.5 + s.life), s.life * s.life);
    }
  }
}

export const ease = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
