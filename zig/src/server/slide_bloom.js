// Bloom: the scene drawn into a float target, its bright parts blurred at half and quarter size, all added back.
import * as THREE from "three";

const quadVertex = `
  varying vec2 uv0;
  void main() { uv0 = position.xy * 0.5 + 0.5; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

const brightFragment = `
  uniform sampler2D source;
  uniform float threshold;
  varying vec2 uv0;
  void main() {
    vec3 c = texture2D(source, uv0).rgb;
    float l = max(c.r, max(c.g, c.b));
    gl_FragColor = vec4(c * smoothstep(threshold, threshold + 0.35, l), 1.0);
  }
`;

const blurFragment = `
  uniform sampler2D source;
  uniform vec2 step;
  varying vec2 uv0;
  void main() {
    vec3 c = texture2D(source, uv0).rgb * 0.2270270;
    c += texture2D(source, uv0 + step * 1.3846154).rgb * 0.3162162;
    c += texture2D(source, uv0 - step * 1.3846154).rgb * 0.3162162;
    c += texture2D(source, uv0 + step * 3.2307692).rgb * 0.0702703;
    c += texture2D(source, uv0 - step * 3.2307692).rgb * 0.0702703;
    gl_FragColor = vec4(c, 1.0);
  }
`;

const mixFragment = `
  uniform sampler2D scene;
  uniform sampler2D near;
  uniform sampler2D far;
  uniform float strength;
  varying vec2 uv0;
  vec3 srgb(vec3 c) { return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c)); }
  void main() {
    vec3 c = texture2D(scene, uv0).rgb + strength * (texture2D(near, uv0).rgb * 0.9 + texture2D(far, uv0).rgb * 1.1);
    c = c / (1.0 + max(c.r, max(c.g, c.b)) * 0.18);
    gl_FragColor = vec4(srgb(clamp(c, 0.0, 1.0)), 1.0);
  }
`;

export class Bloom {
  constructor(renderer, {strength = 1.1, threshold = 0.18} = {}) {
    this.renderer = renderer;
    const type = renderer.capabilities.isWebGL2 ? THREE.HalfFloatType : THREE.UnsignedByteType;
    const target = (samples = 0) => new THREE.WebGLRenderTarget(1, 1, {type, samples, depthBuffer: samples > 0});
    this.scene = target(4);
    this.half = [target(), target()];
    this.quarter = [target(), target()];
    const shader = (fragmentShader, uniforms) => new THREE.ShaderMaterial({uniforms, vertexShader: quadVertex, fragmentShader, depthTest: false, depthWrite: false});
    this.bright = shader(brightFragment, {source: {value: null}, threshold: {value: threshold}});
    this.blur = shader(blurFragment, {source: {value: null}, step: {value: new THREE.Vector2()}});
    this.mix = shader(mixFragment, {scene: {value: null}, near: {value: null}, far: {value: null}, strength: {value: strength}});
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.bright);
    this.quad.frustumCulled = false;
    this.stage = new THREE.Scene();
    this.stage.add(this.quad);
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  setSize(width, height) {
    this.scene.setSize(width, height);
    for (const t of this.half) t.setSize(Math.max(1, width >> 1), Math.max(1, height >> 1));
    for (const t of this.quarter) t.setSize(Math.max(1, width >> 2), Math.max(1, height >> 2));
  }

  pass(material, target) {
    this.quad.material = material;
    this.renderer.setRenderTarget(target);
    this.renderer.render(this.stage, this.camera);
  }

  // Two blur passes, across then down, on `pair[0]`, leaving the result there.
  soften(pair) {
    const [a, b] = pair, u = this.blur.uniforms;
    u.source.value = a.texture;
    u.step.value.set(1 / a.width, 0);
    this.pass(this.blur, b);
    u.source.value = b.texture;
    u.step.value.set(0, 1 / b.height);
    this.pass(this.blur, a);
  }

  render(scene, camera) {
    const r = this.renderer;
    r.setRenderTarget(this.scene);
    r.clear();
    r.render(scene, camera);
    this.bright.uniforms.source.value = this.scene.texture;
    this.pass(this.bright, this.half[0]);
    this.soften(this.half);
    this.blur.uniforms.source.value = this.half[0].texture;
    this.blur.uniforms.step.value.set(0.5 / this.half[0].width, 0.5 / this.half[0].height);
    this.pass(this.blur, this.quarter[0]);
    this.soften(this.quarter);
    this.soften(this.quarter);
    const m = this.mix.uniforms;
    m.scene.value = this.scene.texture;
    m.near.value = this.half[0].texture;
    m.far.value = this.quarter[0].texture;
    this.pass(this.mix, null);
  }

  dispose() {
    for (const t of [this.scene, ...this.half, ...this.quarter]) t.dispose();
    for (const m of [this.bright, this.blur, this.mix]) m.dispose();
    this.quad.geometry.dispose();
  }
}
