// The page's helpers, state and settings shared by every tab (index.html loads the tab files after it).
"use strict";
if (location.search.includes("shot")) document.write('<img src="/api/wait?ms=4000" style="display:none">');
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => { const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === "class") n.className = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "html") n.innerHTML = v; else if (v !== false && v != null) n.setAttribute(k, v === true ? "" : v); }
  for (const kid of kids.flat()) if (kid != null && kid !== false) n.append(kid.nodeType ? kid : document.createTextNode(kid)); return n; };
const api = async (path, body, method) => {
  // every job queued here carries the number of Sparks chosen in the queue (sparks.js), ignored where it does not apply
  if (path === "/api/jobs" && body && body.kind && body.sparks === undefined) body = { sparks: SPK.n, ...body };
  const r = await fetch(path, body === undefined ? { method: method || "GET" } :
    { method: method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const data = await r.json().catch(() => ({})); if (!r.ok) throw new Error(data.error || data.detail || r.statusText); return data; };
const file = path => "/files/" + path.split("/").map(encodeURIComponent).join("/");
const mmss = s => s == null ? "" : s >= 3600 ? Math.floor(s / 3600) + " h " + Math.round(s % 3600 / 60) + " min" : (s >= 60 ? Math.floor(s / 60) + " min " : "") + Math.round(s % 60) + " s";
const put = (node, ...kids) => node.append(...kids.flat().filter(k => k != null && k !== false));
const toast = (text, ms = 3500) => { for (const old of document.querySelectorAll(".toast")) old.remove(); const n = el("div", { class: "toast", style: "position:fixed;bottom:18px;left:18px;background:#2b3050;border:1px solid var(--accent);padding:9px 14px;border-radius:8px;z-index:9" }, text); document.body.append(n); setTimeout(() => n.remove(), ms); };
// Clear and Undo for a form's text: a data-clear="key" form; data-keep fields stay, data-clear-to is a field's blank
let CLEARED = null;
const keep = node => { node.setAttribute("data-keep", ""); return node; };
const clearFields = root => [...root.querySelectorAll("textarea, input:not([type]), input[type=text]")].filter(n => !n.disabled && !n.closest("[data-keep]"));
const fill = (n, v) => { n.value = v; n.dispatchEvent(new Event("input", { bubbles: true })); };
function undoClear() { const c = CLEARED; if (!c) return; CLEARED = null;
  if (c.model) { c.model(); render(); return toast("Put back"); }
  const root = document.querySelector('[data-clear="' + c.key + '"]'), f = root ? clearFields(root) : [];
  if (f.length !== c.values.length) { CLEARED = c; return toast("Open the form that was cleared to put its text back"); }
  f.forEach((n, i) => fill(n, c.values[i])); render(); toast("Put back"); }
const undoToast = text => toast(el("span", {}, text + " ", el("button", { class: "small", onclick: e => { e.target.closest(".toast").remove(); undoClear(); } }, "Undo")), 12000);
function clearForm(key) { const root = document.querySelector('[data-clear="' + key + '"]'); if (!root) return; const f = clearFields(root);
  if (!f.some(n => n.value !== (n.dataset.clearTo || ""))) return toast("Nothing to clear here");
  CLEARED = { key, values: f.map(n => n.value) }; f.forEach(n => fill(n, n.dataset.clearTo || "")); render(); undoToast("Cleared."); }
const clearButton = (key, label = "Clear", title = "Empty the text fields of this form (Undo puts them back)") => [el("button", { class: "small fit", title, onclick: () => clearForm(key) }, label),
  CLEARED && CLEARED.key === key ? el("button", { class: "small fit on", onclick: undoClear }, "Undo clear") : null];
// Deleting: jobs leave the queue, their files go to the trash folder (app/clear.py); the last one can come back
const PICK = new Set(), LIBPICK = new Set();
const sizeText = b => b >= 1e9 ? (b / 1e9).toFixed(1) + " GB" : b >= 1e6 ? Math.round(b / 1e6) + " MB" : Math.max(1, Math.round(b / 1e3)) + " kB";
const count = (n, word) => n + " " + word + (n === 1 ? "" : "s");
async function restoreLast() { try { const r = await api("/api/trash/restore", {}); toast("Put back " + count(r.files, "file") + (r.jobs ? " and " + count(r.jobs, "job card") : "") + (r.missing.length ? " · " + r.missing.length + " could not be put back" : "")); } catch (e) { toast(e.message); } await poll(); render(); }
async function clearOut(body, ask) { if (ask && !confirm(ask + "\nFiles are moved to the trash folder inside " + S.home + ", not erased. “Restore last deletion” in the Library puts them back.")) return;
  try { const r = await api("/api/clear", body); PICK.clear(); LIBPICK.clear();
    toast(el("span", {}, (r.jobs ? count(r.jobs, "job") + " and " : "") + count(r.files, "file") + " moved to the trash" + (r.skipped.length ? " · " + r.skipped.length + " kept: " + r.skipped[0].why : "") + " ",
      r.files || r.jobs ? el("button", { class: "small", onclick: e => { e.target.closest(".toast").remove(); restoreLast(); } }, "Undo") : null), 12000);
  } catch (e) { toast(e.message); } await poll(); render(); }

let S = null, jobs = [], tab = "Create", project = null;
const TABS = ["Create", "Production", "Motion Previs", "Voice", "Editor", "Library", "Sparks", "Tutorial"];
const C = { step: 0, mode: "scout", preset: "draft", quality: "standard", seconds: 5, seed: 1, scene: "", action: "", camera: "",
  dialogue: [], soundscape: "", music: "N/A", imagePrompt: "", image: null, scoutJob: null, prompt: "", job: null, count: 4, imageSize: "1344x768", imageJob: null, imageModel: "qwen", beats: "", cw: 864, ch: 480, steps: "", loras: [], textEncoder: "", lastImage: null, frames: null, retake: false, negations: [], negFor: null };
let LORAS = [];  // the LoRA hub's recipes, with whether each is installed here
const field = (labelText, node) => el("div", {}, el("label", {}, labelText), node);
const bind = (obj, key, node, after) => { const isBox = node.type === "checkbox"; if (isBox) node.checked = !!obj[key]; else node.value = obj[key] ?? "";
  node.addEventListener("input", () => { obj[key] = isBox ? node.checked : (node.type === "number" ? Number(node.value) : node.value); if (after) after(); }); return node; };
// a field the page redraws only once it is finished (Enter, Tab or a click elsewhere), not on every key
const typed = (node, after) => { node.addEventListener("change", () => after && after()); return node; };
const select = (obj, key, options, after) => { const n = el("select", {}, Object.entries(options).map(([v, t]) => el("option", { value: v }, t))); return bind(obj, key, n, after); };
const media = (path, attrs = {}) => /\.(mp4|mov)$/i.test(path) ? el("video", { src: file(path), controls: true, preload: "metadata", ...attrs })
  : /\.(wav|mp3|m4a|aac|flac|ogg)$/i.test(path) ? el("audio", { src: file(path), controls: true, preload: "metadata", style: "width:100%", ...attrs }) : el("img", { src: file(path), loading: "lazy", ...attrs });

const IMAGE_SIZES = { "1344x768": "1344x768 (16:9)", "1664x928": "1664x928 (16:9, larger)", "1024x1024": "1024x1024 (square)", "768x1344": "768x1344 (9:16)" };
Object.assign(C, { goes: "video", start: "text", engine: "fasth3", passes: "8", h3quality: "standard", res: "480p", up: false, h3passes: "preset", refs: [], subjects: [], refPasses: "20", refVideos: [], video_subjects: [], refAudios: [], audio_notes: [], video_sound: false, motion: null });
const ENGINES = { fasth3: "FastH3 (FastVideo, sparse attention): fastest", h3: "MiniMax H3 (Turbo adapter or full 20 steps)" };
const RESOLUTIONS = { "480p": "480p · 864x480", "720p": "720p · 1280x720", draft: "Draft · 1344x768, generated at 672x384 with the 2x decoder",
  "2k": "2K · 2048x1152, generated at 1024x576 with the 2x decoder", native: "Native · 1344x768" };
const UPSCALED = { "480p": ["p960", "1728x960"], "720p": ["qhd", "2560x1440 (2K)"] };
// reference to video (Ref2VA): pictures, videos and audio that fix how someone looks, moves or sounds; never a frame
const isRef = () => C.mode === "r2v";
const refMedia = () => isRef() && (C.refVideos.length > 0 || C.refAudios.length > 0);
const refCount = () => C.refs.length + C.refVideos.length + C.refAudios.length;
// what the page knows of each reference clip (/api/probe: length, sound) and what it last said of a refused file
const REF_META = {}, REF_SAID = { pictures: "", videos: "", audios: "" };
const refMeta = async path => REF_META[path] || (REF_META[path] = await api("/api/probe?path=" + encodeURIComponent(path)).catch(() => ({})));
const REF_KINDS = { pictures: ["pictures", () => S.reference_limit], videos: ["videos", () => S.reference_video_limit], audios: ["audio clips", () => S.reference_audio_limit] };
// why one more file of a kind cannot be added ("" when it can): MiniMax's limits for one request
const refFull = (kind, list) => list.length >= REF_KINDS[kind][1]() ? "at most " + REF_KINDS[kind][1]() + " " + REF_KINDS[kind][0] : S.ref_media && refCount() >= S.reference_total ? "at most " + S.reference_total + " references in all" : "";
// why a measured clip is not a reference the model takes ("" when it is): 2 to 15 seconds each, 15 seconds a kind
function refWrong(kind, list, meta) { const word = kind === "videos" ? "video" : "audio clip", d = Number(meta.duration) || 0;
  if (kind === "videos" && !meta.width) return "not a video the app can read";
  if (kind === "audios" && meta.sound === false) return "it has no sound";
  if (!d) return "";  // not measured (no ffprobe): the renderer measures again
  if (d < 1.95 || d > 15.05) return "it is " + d.toFixed(1) + " s long, and a reference " + word + " is 2 to 15 seconds";
  const all = list.reduce((sum, p) => sum + ((REF_META[p] || {}).duration || 0), 0) + d;
  return all > 15.05 ? "with it the reference " + word + "s would be " + all.toFixed(1) + " s, and they are at most 15 seconds in all" : ""; }
// what the Studio says about sound references and whether a video's own sound starts ticked (README, measured)
const REF_SOUND_DEFAULT = false;
const VIDEO_SOUND_SAYS = "Off, the clip's sound comes from what you write in step 3. On, it stays close to the video's own sound instead, voice and timing (measured on both builds), at no extra time.";
const AUDIO_SAYS = "What it does, by measurement: when the spoken line in step 3 is what the recording says, the clip performs that recording, the mouth moving in time with it. With other words, describe the voice in the line's Voice box as well (“a deep male voice”): the pitch then went to the recording's, where a recording with no description changed nothing when the voice was unlike the pictured person. It is not voice cloning.";
// how many subjects the prompt will define: one per picture, and one per video that is said to show someone
const refPeople = () => C.refs.length + (motionFields() ? 0 : C.refVideos.filter((p, i) => (C.video_subjects[i] || "").trim() || (!C.refs.length && i === 0)).length);
const soundless = () => C.refVideos.map((p, i) => REF_META[p] && REF_META[p].sound === false ? i + 1 : 0).filter(Boolean);
// the model numbers a video's own sound ahead of the audio clips
const audioName = i => "<Audio " + ((C.video_sound ? C.refVideos.length : 0) + i + 1) + ">";
// the passes a reference clip runs: a custom count wins over both cards; the adapter was trained for 4
const refPassText = () => String(C.steps).trim() ? (C.refPasses === "4" ? "Turbo adapter, " : "") + C.steps + " passes (custom)" : C.refPasses === "4" ? "Turbo 4 passes" : "20 passes";
const refWord = () => S && S.ref_media ? "References" : "Reference pictures";
const noFrame = () => C.mode === "t2v" || isRef();
// the end keyframe (MiniMax H3's first-and-last-frame mode): single clips that are not reference clips
const lastOf = () => !isRef() && !isLong() && C.goes !== "image" && C.lastImage ? C.lastImage : null;
const L = { lib: false, clip: "", prompt: "", job: null, took: null };
async function frameFrom(path, at, set) { try { const r = await api("/api/frame", { path, at }); set(r.path); render(); } catch (e) { toast(e.message); } }
// a frame out of a Library clip: its last (to continue from it), its first (to lead into it), or one at a given time
function clipFrames(set, lead) { const box = el("div", { class: "row", style: "margin-top:8px;align-items:flex-end" });
  api("/api/library").then(list => { const clips = list.filter(m => m.kind === "video"); if (!clips.length) { box.append(el("div", { class: "note" }, "No clips in the Library yet.")); return; }
    const pick = el("select", {}, clips.slice(0, 80).map(m => el("option", { value: m.path }, m.path.split("/").pop()))); if (clips.some(m => m.path === L.clip)) pick.value = L.clip; pick.onchange = () => { L.clip = pick.value; };
    const at = el("input", { type: "number", min: 0, step: 0.1, placeholder: "seconds", style: "width:110px" });
    box.append(field(lead, pick), el("button", { class: "fit", onclick: () => frameFrom(pick.value, "last", set) }, "Use its last frame"), el("button", { class: "fit", onclick: () => frameFrom(pick.value, "first", set) }, "Use its first frame"),
      el("div", { class: "fit" }, el("label", {}, "or the frame at"), at), el("button", { class: "fit", onclick: () => at.value === "" ? toast("Type the seconds first") : frameFrom(pick.value, Number(at.value), set) }, "Use frame at time")); }).catch(() => {});
  return box; }
function lastFramePanel() { const set = p => { C.lastImage = p; C.prompt = ""; };
  const up = el("input", { type: "file", accept: "image/*", onchange: async () => { const data = new FormData(); data.append("file", up.files[0]);
    const r = await fetch("/api/upload", { method: "POST", body: data }); const j = await r.json(); if (j.path) { set(j.path); render(); } else toast(j.error || "upload failed"); } });
  const need = lorasOf(C).map(p => LORAS.find(r => r.id === p.id)).filter(r => r && r.needs === "first-last");
  const scout = L.job && jobs.find(j => j.id === L.job);
  if (scout && scout.status === "done" && L.took !== scout.id) { const made = scout.outputs.find(p => !p.endsWith("sheet.jpg")); if (made) set(made); L.took = scout.id; }
  const box = el("div", { id: "lastframe" }, el("h2", { style: "margin-top:22px" }, "Last frame"),
    el("div", { class: "note" }, isLong() ? "A last frame is for single clips: a long video is a chain of clips." : "Optional. The clip ends on this picture: MiniMax H3 holds it as its end keyframe, the way it holds a first frame. Beside a first frame it is fitted by a centre crop, so give both the same aspect ratio, and say in the prompt how the first picture becomes the last."),
    need.filter(r => r.prompt).map(r => el("div", { class: "row", style: "align-items:center" }, el("div", { class: "note", style: "margin:0" }, r.name + " was trained with one prompt: "), el("button", { class: "fit small", title: r.prompt, onclick: () => { C.prompt = r.prompt; C.step = 3; render(); toast("The author's prompt is in step 4. “Rebuild” would replace it."); } }, "Use the author's prompt"))),
    need.length && !C.lastImage ? el("div", { class: "note", style: "color:var(--warn)" }, need.map(r => r.name).join(" and ") + (need.length > 1 ? " need" : " needs") + " a last frame" + (need.some(r => r.id === "orbit-360") ? ": for 360° Orbit, the same picture as the first frame." : ".")) : null,
    el("div", { class: "row", style: "align-items:flex-end" }, field("Upload a picture", up),
      C.image && !noFrame() ? el("button", { class: "fit", title: "The clip comes back to the picture it starts from", onclick: () => { set(C.image); render(); } }, "Same as the first frame") : null,
      el("button", { class: "fit", onclick: () => { L.lib = !L.lib; render(); } }, L.lib ? "Hide the Library" : "Pick from the Library"),
      C.lastImage ? el("button", { class: "fit", onclick: () => { set(null); render(); } }, "No last frame") : null),
    clipFrames(set, "From a clip in the Library"),
    el("div", { class: "row", style: "margin-top:8px;align-items:flex-end" }, field("Or make one from a prompt (" + imageModelName() + ")", bind(L, "prompt", el("input", { placeholder: "the picture the clip ends on" }))),
      el("button", { class: "fit", onclick: async () => { if (!L.prompt.trim()) return toast("Describe the picture first"); try { const j = await api("/api/jobs", { kind: "scout", title: "Last frame", params: { prompt: L.prompt, count: 1, seed: C.seed, preset: C.preset, image_model: imageModel(), ...(C.preset === "custom" ? { width: Number(C.cw), height: Number(C.ch) } : {}) } }); L.job = j.id; await poll(); render(); } catch (e) { toast(e.message); } } }, "Make it"),
      scout ? el("span", { class: "pill " + scout.status }, scout.status) : null));
  if (L.lib) { const grid = el("div", { class: "grid", style: "margin-top:10px" }); box.append(grid);
    api("/api/library").then(list => list.filter(m => m.kind === "image").slice(0, 24).forEach(m => grid.append(el("div", { class: "tile" + (C.lastImage === m.path ? " on" : ""), onclick: () => { set(m.path); L.lib = false; render(); } }, media(m.path), el("div", { class: "cap" }, m.path.split("/").pop().slice(-26)))))).catch(() => {}); }
  if (C.lastImage) box.append(el("h3", {}, "Selected last frame"), el("div", { style: "max-width:420px" }, media(C.lastImage, { style: "width:100%;border-radius:8px" })));
  return box; }
const isLong = () => !C.frames && Number(C.seconds) > 15;
// Clip lengths (app/lengths.py): 17n + 5 frames at 24 a second; the exact time a count gives is what is shown
const exact = f => f + " frames · " + (f / S.lengths.fps).toFixed(3).replace(/\.?0+$/, "") + " s";
const nearestFrames = s => { const L = S.lengths; let best = L.shortest; for (let f = L.shortest; f <= L.longest; f += L.step) if (Math.abs(f - s * L.fps) < Math.abs(best - s * L.fps)) best = f; return best; };
const framesOf = o => Number(o.frames) || S.durations[o.seconds] || nearestFrames(Number(o.seconds) || 5);
const lengthOf = o => o.frames ? { frames: Number(o.frames), seconds: Number(o.frames) / S.lengths.fps } : { seconds: Number(o.seconds) };
const longestFor = o => (o.preset === "custom" ? S.lengths.custom : S.lengths.presets[o.preset]) || S.lengths.longest;
function lengthPicker(obj, after, long = {}) { const L = S.lengths, own = !!obj.frames, most = longestFor(obj);
  const pick = el("select", { "aria-label": "Clip length" }, [...Object.keys(S.durations).map(k => el("option", { value: k }, k + " s · " + exact(S.durations[k]))),
    ...Object.entries(long).map(([k, v]) => el("option", { value: k }, v)), el("option", { value: "own" }, "Another length…")]);
  pick.value = own ? "own" : String(obj.seconds);
  pick.onchange = () => { if (pick.value === "own") { obj.frames = Math.min(framesOf(obj), most); obj.seconds = obj.frames / L.fps; } else { obj.frames = null; obj.seconds = Number(pick.value); } if (after) after(); };
  if (!own) return pick;
  if (obj.frames > most) { obj.frames = most; obj.seconds = most / L.fps; }
  const slide = el("input", { type: "range", min: L.shortest, max: most, step: L.step, value: obj.frames, "aria-label": "Clip length in frames", style: "margin-top:6px" }), said = el("b", {}, exact(Number(obj.frames)));
  slide.oninput = () => { said.textContent = exact(Number(slide.value)); };
  slide.onchange = () => { obj.frames = Number(slide.value); obj.seconds = obj.frames / L.fps; if (after) after(); };
  return el("div", {}, pick, slide, el("div", { class: "note" }, said, " · the model takes 17n + 5 frames: " + L.shortest + " to " + most + " here, " + L.step + " apart")); }
// the explicit controls decide the mode, the quality and the output preset the server understands
function sync() {
  if (C.start === "refs" && !(S && S.ref2va)) C.start = "text";
  C.mode = C.goes === "image" ? "image" : C.goes === "edit" ? "edit" : { text: "t2v", scout: "scout", own: "i2v", refs: "r2v" }[C.start];
  if (isRef() && isLong()) C.seconds = 15;
  if (!(S && S.ref2va_turbo)) C.refPasses = "20";
  // a custom count belongs to the chooser it was typed in: it does not follow into or out of reference clips
  if (C.wasRef !== undefined && C.wasRef !== isRef()) C.steps = "";
  C.wasRef = isRef();
  if (C.video_sound && (!C.refVideos.length || soundless().length)) C.video_sound = false;
  if (C.res === "sharp" && !(S && S.sharp)) C.res = "480p";
  C.quality = C.engine === "fasth3" ? "fh" + C.passes : C.h3quality;
  if (!UPSCALED[C.res]) C.up = false;
  C.preset = UPSCALED[C.res] ? (C.up ? UPSCALED[C.res][0] : { "480p": "small", "720p": "p720" }[C.res]) : C.res;
}
// the output: a preset, or a size typed in by hand
const output = () => C.preset === "custom" ? { label: "Custom " + C.cw + "x" + C.ch, size: [Number(C.cw), Number(C.ch)] } : S.presets[C.preset];
// what a job carries beyond the preset: the hand-set size and passes, and the text encoder
const extras = () => ({ ...(C.preset === "custom" ? { width: Number(C.cw), height: Number(C.ch) } : {}), ...(String(C.steps).trim() ? { steps: Number(C.steps) } : {}), ...(lorasOf(C).length ? { loras: lorasOf(C) } : {}), ...(encoderOf() ? { text_encoder: encoderOf() } : {}) });
function changed() { sync(); C.prompt = ""; render(); }
// the picture models installed here (server.py IMAGE_MODELS); a choice that is not installed falls back to Qwen
const imageModels = () => (S.image_models && S.image_models.length ? S.image_models : [{ id: "qwen", label: "Qwen 2.1 Turbo", note: "" }]);
const imageModel = () => (imageModels().some(m => m.id === C.imageModel) ? C.imageModel : "qwen");
const imageModelName = () => imageModels().find(m => m.id === imageModel()).label;
const imageModelPick = after => imageModels().length < 2 ? null : el("div", {}, el("label", {}, "Picture model"),
  choice(C, "imageModel", Object.fromEntries(imageModels().map(m => [m.id, [m.label, m.note]])), after || render, true));
const choice = (obj, key, options, after, tight) => el("div", { class: "choice" + (tight ? " tight" : ""), role: "radiogroup" }, Object.entries(options).map(([v, [title, sub]]) =>
  el("button", { type: "button", role: "radio", "aria-checked": String(obj[key]) === v ? "true" : "false", class: String(obj[key]) === v ? "on" : "", onclick: () => { obj[key] = v; if (after) after(); } }, el("b", {}, title), sub ? el("span", {}, sub) : null)));
// a number of passes of your own: the Custom card after the 20-pass card; C.steps holds it ("" for a preset card)
const customOn = () => String(C.steps).trim() !== "";
// a chooser's cards with Custom beside the 20-pass card and its number box under the row; the preset recipe stays
function passChooser(key, options, start, after, more) {
  const card = (on, title, sub, pick) => el("button", { type: "button", role: "radio", "aria-checked": on ? "true" : "false", class: on ? "on" : "", onclick: pick }, el("b", {}, title), sub ? el("span", {}, sub) : null);
  const [low, high] = S.custom.steps;
  const cards = Object.entries(options).map(([v, [title, sub]]) => [/^20\b/.test(title), card(!customOn() && String(C[key]) === v, title, sub, () => { C[key] = v; C.steps = ""; changed(); })]);
  const at = cards.map(c => c[0]).lastIndexOf(true), nodes = cards.map(c => c[1]);
  nodes.splice(at < 0 ? nodes.length : at + 1, 0, card(customOn(), "Custom", "your own number of passes", () => { if (!customOn()) C.steps = String(start()); changed(); }));
  return [el("div", { class: "choice tight passes" + (nodes.length % 2 ? " odd" : ""), role: "radiogroup" }, nodes),
    customOn() ? el("div", { class: "row" }, el("div", { class: "fit", style: "width:150px" }, el("label", {}, "Number of passes"), typed(bind(C, "steps", el("input", { type: "number", min: low, max: high, step: 1 })), after)),
      el("div", { class: "note" }, "From " + low + " to " + high + ". No time estimate for a custom count." + (more ? " " + more : ""))) : null];
}
// MiniMax H3's qualities are recipes (adapter, sound), not counts: a custom count runs inside the chosen one
const H3_PASSES = { standard: 5, fast: 3, high: 20, full: 20 };
const PASSES = { 4: ["4 passes", "fastest, a little softer"], 8: ["8 passes", "what FastH3 was trained for"], 20: ["20 passes", "for prompts with several actions"] };
const H3Q = { standard: ["Turbo 5 + base sound", "the standard"], fast: ["Turbo 3", "quickest, adapter sound"], high: ["20 passes, fast recipe", "higher quality"], full: ["20 passes, plain", "slowest"] };
const RES = { "480p": ["480p", "864 × 480"], "720p": ["720p", "1280 × 720"], draft: ["Draft", "1344 × 768 via 2x"], "2k": ["2K", "2048 × 1152 via 2x"], native: ["Native", "1344 × 768"], sharp: ["Wasserman Sharp", "1376 × 768"], custom: ["Custom", "your own size"] };
// Sam Wasserman's locked recipe (S.sharp): one click sets every setting it fixes, so each clip is made the same way
const sharpButton = (on, apply) => el("button", { type: "button", class: "fit sharp" + (on ? " on" : ""), title: "Sam Wasserman's locked recipe: 1376 × 768, FastH3 at 8 passes, seed 42, 8 or 10 seconds, one pass with no upscale", onclick: apply },
  (on ? "✓ " : "") + "Wasserman Sharp Spec");
const sharpOn = () => !!S.sharp && !isRef() && C.engine === "fasth3" && C.passes === "8" && C.res === "sharp" && !String(C.steps).trim() && S.sharp.seconds.includes(Number(C.seconds));
function sharpCreate() { if (isRef()) C.start = "text";
  Object.assign(C, { engine: "fasth3", passes: "8", res: "sharp", up: false, steps: "", seed: S.sharp.seed, seconds: S.sharp.seconds.includes(Number(C.seconds)) ? C.seconds : S.sharp.seconds[0] }); changed(); }
