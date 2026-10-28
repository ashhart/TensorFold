// Motion Previs tab, after Motion Previs Studio 4.1.0's flow (none of its UI code); the analysis is a modified port
"use strict";
// Copyright 2026 Sam Wasserman, Apache-2.0, modified: https://github.com/wassermanproductions/motion-previs-studio
const MO = { state: null, sources: null, packs: null, source: null, meta: null, start: 0, end: 5, mode: "full-scene", layers: null, fps: 12,
  poseModel: "", resolution: "auto", job: null, timer: null, packId: null, picking: false, busy: "", ref: undefined,
  hand: { layer: "", mode: "", seconds: 5, size: "864x480", w: 864, h: 480, timing: "stretch", fit: "cover" } };
const MO_STAGES = { prepare: "Prepare", pose: "Pose", camera: "Camera", depth: "Depth", encode: "Encode", pack: "Pack" };
const MO_SIZES = { "864x480": "864 × 480", "1280x736": "1280 × 736", "1344x768": "1344 × 768", "1376x768": "1376 × 768", "480x864": "480 × 864 (upright)", custom: "Custom" };
const moLength = () => Math.round((Number(MO.end) - Number(MO.start)) * 100) / 100;
const moValid = () => MO.state && MO.source && moLength() >= MO.state.limits[0] && moLength() <= MO.state.limits[1] + 1e-6;
const moActive = () => MO.job && ["queued", "running"].includes(MO.job.status);
const moPack = () => (MO.packs || []).find(p => p.id === MO.packId) || null;

async function moLoad() {
  try { [MO.state, MO.packs, MO.ref] = await Promise.all([api("/api/motion/state"), api("/api/motion/packs"), api("/api/motion/handoff").then(r => r.reference)]); }
  catch (e) { MO.state = { error: e.message }; MO.packs = []; }
  if (MO.state.default_layers && !MO.layers) MO.layers = new Set(MO.state.default_layers);
  if (!MO.poseModel) MO.poseModel = ["heavy", "full", "lite"].find(k => (MO.state.pose_models || []).includes(k)) || "lite";
  if (!MO.packId && MO.packs.length) MO.packId = MO.packs[0].id;
  try { window.MOTION_REFERENCE = MO.ref || null; } catch {}
  if (tab === "Motion Previs") render();
}
async function moPick(path) {
  MO.source = path; MO.picking = false; MO.meta = null; render();
  try { MO.meta = await api("/api/probe?path=" + encodeURIComponent(path)); } catch (e) { toast(e.message); return; }
  // the whole clip when it is short enough to be one shot, else its first 5 seconds
  MO.start = 0; MO.end = Math.floor(Math.min(MO.meta.duration, MO.meta.duration <= MO.state.limits[1] ? MO.meta.duration : 5) * 100) / 100;
  render();
}
async function moUpload(input, pack) {
  const one = input.files[0]; if (!one) return;
  const data = new FormData(); data.append("file", one); MO.busy = pack ? "Reading the pack…" : "Uploading…"; render();
  try { const r = await fetch(pack ? "/api/motion/packs/import" : "/api/upload", { method: "POST", body: data }); const j = await r.json();
    if (!r.ok) throw new Error(j.error || j.detail || r.statusText);
    if (pack) { MO.packs = await api("/api/motion/packs"); MO.packId = j.id; MO.hand.layer = MO.hand.mode = ""; toast("Pack read: " + (j.source || j.id)); } else await moPick(j.path);
  } catch (e) { toast(e.message); }
  MO.busy = ""; render();
}
async function moRun() {
  try { MO.job = await api("/api/motion/jobs", { source: MO.source, start: Number(MO.start), end: Number(MO.end), mode: MO.mode, layers: [...MO.layers], fps: Number(MO.fps),
      resolution: MO.resolution, pose: { poseModel: MO.poseModel } }); } catch (e) { toast(e.message); return; }
  render(); moWatch();
}
function moWatch() {
  clearInterval(MO.timer);
  MO.timer = setInterval(async () => { if (!MO.job) return clearInterval(MO.timer);
    try { MO.job = await api("/api/motion/jobs/" + MO.job.id); } catch { return; }
    if (moActive()) { const rail = document.getElementById("mo-rail"); if (rail) rail.replaceWith(moRail()); return; }
    clearInterval(MO.timer); MO.timer = null;
    if (MO.job.status === "done") { MO.packs = await api("/api/motion/packs"); MO.packId = MO.job.pack; MO.hand.layer = MO.hand.mode = ""; toast("Analysis done: the pack is ready"); }
    if (tab === "Motion Previs") render(); }, 600);
}
// the stage rail: every stage with its own bar, counted in frames or seconds of that stage
function moRail() {
  const job = MO.job, box = el("div", { id: "mo-rail" }); if (!job) return box;
  put(box, el("div", { class: "seghead", style: "margin:12px 0 6px" }, el("b", {}, job.title), el("span", { class: "pill " + job.status }, job.status),
    el("span", { class: "note", style: "margin:0" }, job.elapsed != null ? mmss(job.elapsed) : ""), el("span", { style: "flex:1" }),
    moActive() ? el("button", { class: "small", onclick: async () => { MO.job = await api("/api/motion/jobs/" + job.id + "/cancel", {}); render(); } }, "Cancel") : null));
  put(box, el("div", { style: "display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:8px" }, Object.entries(MO_STAGES).map(([key, label]) => { const s = job.stages[key] || {};
    const share = s.state === "done" ? 100 : s.state === "running" ? Math.max(2, Math.min(100, Math.round(100 * (s.at || 0) / (s.of || 1)))) : 0;
    return el("div", { style: "opacity:" + (s.state === "waiting" ? .45 : 1) }, el("div", { style: "display:flex;justify-content:space-between;font:600 10.5px var(--mono);letter-spacing:.1em;text-transform:uppercase;color:" + (s.state === "running" ? "var(--warn)" : s.state === "done" ? "var(--ok)" : "var(--dim)") },
      el("span", {}, label), el("span", {}, s.state === "done" && s.seconds != null ? s.seconds.toFixed(1) + " s" : s.state === "running" ? share + "%" : "")), el("div", { class: "bar" }, el("i", { style: "width:" + share + "%" }))); })));
  const now = job.stages[job.stage] || {};
  put(box, el("div", { class: "note" }, job.status === "failed" ? "Failed: " + (job.error || "") : job.status === "cancelled" ? "Cancelled: nothing was kept." : moActive() ? (now.message || "Waiting for its turn…") :
    "Done in " + mmss(job.elapsed) + (job.memory ? " · peak memory " + job.memory.analysis + " MB" : "")));
  (job.notes || []).forEach(n => put(box, el("div", { class: "tip" }, el("b", {}, "Note"), n)));
  return box;
}
function moSource() {
  const up = el("input", { type: "file", accept: ".mp4,.mov,video/mp4,video/quicktime", class: "hide", onchange: () => moUpload(up, false) });
  const zip = el("input", { type: "file", accept: ".zip", class: "hide", onchange: () => moUpload(zip, true) });
  const box = el("div", { class: "group" }, el("h3", {}, "1 · The shot"), up, zip,
    el("div", { class: "chips" }, el("button", { onclick: () => up.click() }, "Upload a clip"),
      el("button", { class: MO.picking ? "small on" : "", style: "padding:8px 14px;font-size:inherit", onclick: async () => { MO.picking = !MO.picking; if (MO.picking) MO.sources = await api("/api/motion/sources"); render(); } }, "Pick from the studio folder"),
      el("button", { title: "A production pack exported by Motion Previs Studio, or by this tab: the analysis is skipped", onclick: () => zip.click() }, "Upload a pack ZIP"),
      MO.busy ? el("span", { class: "note", style: "margin:0" }, MO.busy) : null));
  if (MO.picking) put(box, !MO.sources ? el("div", { class: "note" }, "Loading…") : !MO.sources.length ? el("div", { class: "note" }, "No clips under the studio folder yet.") :
    el("div", { class: "grid", style: "margin-top:12px;max-height:46vh;overflow:auto" }, MO.sources.slice(0, 60).map(m => el("div", { class: "tile" + (m.path === MO.source ? " on" : "") },
      el("video", { src: file(m.path) + "#t=0.5", preload: "metadata", muted: true }), el("div", { class: "cap" }, el("span", { title: m.path }, m.path.slice(-28)), el("button", { class: "small", onclick: () => moPick(m.path) }, "use"))))));
  if (!MO.source) return put(box, el("div", { class: "note" }, "A clip you upload, or an earlier render or upload. Then choose the part of it to analyse: 2 to 15 seconds.")), box;
  const video = el("video", { class: "main", src: file(MO.source), controls: true, preload: "metadata", style: "margin-top:12px;max-height:40vh" });
  const say = el("span", { class: "chip" }), paint = () => { const n = moLength(), ok = moValid(); say.textContent = n.toFixed(2) + " s" + (ok ? "" : n < MO.state.limits[0] ? " · too short: 2 s at least" : " · too long: 15 s at most");
    say.style.borderColor = ok ? "" : "var(--bad)"; say.style.color = ok ? "" : "var(--bad)"; const go = document.getElementById("mo-run"); if (go) go.disabled = !moValid() || moActive(); };
  const num = key => bind(MO, key, el("input", { type: "number", min: 0, max: MO.meta ? MO.meta.duration : 9999, step: 0.1 }), paint);
  const a = num("start"), b = num("end"), mark = (key, node) => el("button", { class: "fit small", title: "Take the time the player is at", onclick: () => { MO[key] = Math.round(video.currentTime * 100) / 100; node.value = MO[key]; paint(); } }, "◉ playhead");
  // play only the chosen range, the way the pack will be cut
  video.addEventListener("timeupdate", () => { if (!video.paused && video.currentTime > Number(MO.end)) video.currentTime = Number(MO.start); });
  put(box, video, el("div", { class: "row" }, field("Start (s)", el("div", { class: "row", style: "flex-wrap:nowrap;gap:6px" }, a, mark("start", a))), field("End (s)", el("div", { class: "row", style: "flex-wrap:nowrap;gap:6px" }, b, mark("end", b))),
    el("button", { class: "fit", onclick: () => { video.currentTime = Number(MO.start); video.play(); } }, "▶ Play the range")),
    el("div", { class: "summary" }, el("span", { class: "chip k" }, MO.source.split("/").pop()), MO.meta ? el("span", { class: "chip" }, MO.meta.width + " × " + MO.meta.height + " · " + MO.meta.duration.toFixed(2) + " s") : null, say,
      el("span", { class: "eta" }, "a shot is 2 to 15 s")));
  paint(); return box;
}
function moSettings() {
  const st = MO.state, can = st.available, off = { pose: !can.pose, camera: !can.camera, "ai-depth": !can.ai_depth };
  const box = el("div", {}, el("div", { class: "group" }, el("h3", {}, "2 · Reference mode"), choice(MO, "mode", st.modes, render),
    el("div", { class: "note" }, "The mode does not change what is measured; it sets what the pack's prompt notes and the hand-off ask a model to keep.")));
  const layers = el("div", { class: "group" }, el("h3", {}, "3 · Layers"), el("div", { class: "sheet" }, Object.entries(st.layers).map(([key, label]) => { const box = el("input", { type: "checkbox", disabled: off[key] });
    box.checked = MO.layers.has(key) && !off[key]; box.addEventListener("input", () => { box.checked ? MO.layers.add(key) : MO.layers.delete(key); });
    return el("label", { class: "switch", style: "margin-top:8px" }, box, el("span", {}, label, off[key] ? el("small", {}, "not available on this machine") : null)); })),
    el("div", { class: "row" }, field("Samples a second", select(MO, "fps", { 8: "8", 12: "12 (as the original)", 24: "24 (every frame of a Studio clip)" })),
      field("Pose model", select(MO, "poseModel", Object.fromEntries((st.pose_models.length ? st.pose_models : ["lite"]).map(k => [k, { lite: "Lite: fastest, least exact", full: "Full", heavy: "Heavy: most exact (about a second for 5 s)" }[k] || k])))),
      field("Size of the layers", select(MO, "resolution", { auto: "Long edge up to 1280", "720p": "Short edge 720" }))));
  st.missing.forEach(text => put(layers, el("div", { class: "tip" }, el("b", {}, "Not on this machine"), text + " ", el("span", { style: "font-family:var(--mono)" }, st.setup), " adds what is missing; the other layers work without it.")));
  return put(box, layers), box;
}
function moCamera(pack) {
  const c = pack.camera || {}, s = pack.camera_summary, box = el("div", {});
  put(box, el("div", { class: "eg", style: "max-width:none;border-color:rgba(103,227,255,.35)" }, el("b", {}, "The camera move, in words"), el("div", { class: "lead", style: "margin:2px 0" }, c.sentence || "No camera solve in this pack."),
    c.sentence && !c.reliable ? el("div", { class: "note" }, "Low solve confidence (" + Math.round(100 * c.confidence) + "%): check it against the clip before using it.") : null));
  if (s) put(box, el("div", { class: "summary" }, el("span", { class: "chip" }, "pan " + Math.round(-s.panPixels) + " px"), el("span", { class: "chip" }, "tilt " + Math.round(s.tiltPixels) + " px"),
    el("span", { class: "chip" }, "zoom × " + s.zoomRatio.toFixed(3)), el("span", { class: "chip" }, "roll " + s.rollDegrees.toFixed(1) + "°"), el("span", { class: "chip" }, "confidence " + Math.round(100 * s.averageConfidence) + "%"),
    el("span", { class: "note", style: "margin:0 0 0 auto" }, "a pan and a sideways track look alike to this solve")));
  return box;
}
function moReview(pack) {
  const f = pack.files, box = el("div", { class: "group" }, el("h3", {}, "4 · Review"));
  const views = [["Reference", f["reference.mp4"]], ["Skeleton (OpenPose BODY_25)", f["openpose_pose.mp4"] || f["pose_high_contrast.mp4"]], ["Depth" + (f["ai_depth.mp4"] ? " (Depth-Anything)" : " (ffmpeg proxy)"), f["ai_depth.mp4"] || f["depth.mp4"]], ["Edges", f["edges.mp4"] || f["lineart.mp4"]]];
  const players = [];
  put(box, el("div", { class: "summary", style: "margin-top:4px" }, el("span", { class: "chip k" }, pack.source || pack.id), pack.range ? el("span", { class: "chip" }, pack.range.duration.toFixed(2) + " s") : null,
    el("span", { class: "chip" }, (MO.state.modes[pack.mode] || [pack.mode])[0]), pack.size && pack.size.width ? el("span", { class: "chip" }, pack.size.width + " × " + pack.size.height) : null,
    pack.quality ? el("span", { class: "chip" }, "quality " + pack.quality.score + "/100 · " + pack.quality.readiness) : null, pack.imported ? el("span", { class: "chip" }, "uploaded pack") : null,
    el("span", { style: "flex:1" }), el("button", { class: "small", onclick: () => { players.forEach(v => { v.currentTime = 0; v.play(); }); } }, "▶ Play all together"),
    pack.zip ? el("a", { href: file(pack.zip), download: pack.id + ".zip" }, "Download the pack (ZIP)") : null));
  put(box, moCamera(pack));
  put(box, el("div", { style: "display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin-top:12px" }, views.map(([label, path]) => { const v = path ? el("video", { src: file(path), controls: true, muted: true, loop: true, preload: "metadata", playsinline: true }) : null; if (v) players.push(v);
    return el("div", { class: "tile" }, v || el("div", { style: "aspect-ratio:16/9;display:grid;place-items:center;color:var(--dim);font-size:12px;background:#000" }, "not in this pack"), el("div", { class: "cap" }, el("span", {}, label))); })));
  const more = [["Line art", "lineart.mp4"], ["Motion mask", "motion_mask.mp4"], ["Depth proxy", "depth.mp4"], ["Normals proxy", "normals_proxy.mp4"], ["Pose, high contrast", "pose_high_contrast.mp4"], ["Review plate", "combined_reference_depth_pose.mp4"], ["Animatic", "animatic.mp4"],
    ["camera_motion.json", "camera_motion.json"], ["openpose_keypoints.json", "openpose_keypoints.json"], ["pose_landmarks.json", "pose_landmarks.json"], ["prompt_pack.md", "prompt_pack.md"], ["shot_bible.json", "shot_bible.json"], ["bundle_manifest.json", "bundle_manifest.json"]].filter(([, n]) => f[n]);
  put(box, el("details", { class: "fold" }, el("summary", {}, "The rest of the pack (" + Object.keys(f).length + " files)"), f["contact_sheet.jpg"] ? el("img", { src: file(f["contact_sheet.jpg"]), style: "width:100%;border-radius:10px;border:1px solid var(--line)", loading: "lazy" }) : null,
    el("div", { class: "chips", style: "margin-top:8px" }, more.map(([label, n]) => el("a", { href: file(f[n]), target: "_blank", class: "pill" }, label))),
    pack.pose && pack.pose.diagnostics ? el("div", { class: "note" }, "Pose: " + (pack.pose.rawDetectedFrames || 0) + " of " + (pack.pose.totalFrames || 0) + " frames tracked. " + pack.pose.diagnostics.join(" ")) : null,
    pack.timings ? el("div", { class: "note" }, "Time per stage: " + Object.entries(pack.timings).map(([k, v]) => MO_STAGES[k] + " " + v.toFixed(1) + " s").join(" · ") + (pack.memory ? " · peak memory " + pack.memory.analysis + " MB" : "")) : null,
    (pack.notes || []).map(n => el("div", { class: "note" }, "· " + n))));
  return box;
}
// the hand-off: one layer of the pack at the length and size of the clip to be made, the camera move in words, the mode
function moHandoff(pack) {
  const h = MO.hand, st = MO.state, ready = !!pack && !moActive(), box = el("div", { class: "group" }, el("h3", {}, "5 · Motion reference"));
  if (pack) { if (!h.mode) h.mode = pack.mode; if (!h.layer || (h.layer !== "none" && !pack.layers.includes(h.layer))) h.layer = pack.suggested || "none";
    const names = { reference: "Reference clip", openpose: "OpenPose skeleton", pose: "Pose, high contrast", "ai-depth": "AI depth", depth: "Depth proxy", edges: "Edges", lineart: "Line art", masks: "Motion mask", normals: "Normals proxy" };
    const custom = h.size === "custom", size = () => { if (h.size !== "custom") [h.w, h.h] = h.size.split("x").map(Number); };
    put(box, el("div", { class: "row" }, field("Layer that becomes <Video 1>", select(h, "layer", { ...Object.fromEntries(pack.layers.map(l => [l, names[l] || l])), none: "None: the camera move in words only" })),
      field("Reference mode", select(h, "mode", Object.fromEntries(Object.entries(st.modes).map(([k, v]) => [k, v[0]])))),
      field("Length (s)", bind(h, "seconds", el("input", { type: "number", min: st.limits[0], max: st.limits[1], step: 1 }))), field("Size", select(h, "size", MO_SIZES, () => { size(); render(); })),
      custom ? field("Width", bind(h, "w", el("input", { type: "number", min: 64, max: 4096, step: 2 }))) : null, custom ? field("Height", bind(h, "h", el("input", { type: "number", min: 64, max: 4096, step: 2 }))) : null,
      field("Timing", select(h, "timing", { stretch: "Fit the whole shot to the length", trim: "Keep its speed: cut or hold" }))));
  }
  const use = el("button", { class: "primary", disabled: !ready, title: ready ? "" : "Run an analysis or upload a pack first", onclick: async () => { use.disabled = true; use.textContent = "Preparing…";
    try { MO.ref = await api("/api/motion/reference", { pack: pack.id, layer: h.layer, mode: h.mode || pack.mode, seconds: Number(h.seconds), width: Number(h.w), height: Number(h.h), timing: h.timing, fit: h.fit });
      try { window.MOTION_REFERENCE = MO.ref; localStorage.setItem("motion-reference", JSON.stringify(MO.ref)); } catch {}
      if (typeof useMotionReference === "function") return useMotionReference(MO.ref, pack); toast("Stored as the motion reference"); } catch (e) { toast(e.message); } render(); } }, "Use as motion reference");
  put(box, el("div", { class: "row", style: "margin-top:14px" }, el("div", { class: "fit" }, use), el("div", { class: "note", style: "margin:0" }, ready ? "Opens the create wizard with this layer as <Video 1> of a reference clip, the camera move ready for the prompt and the mode chosen here. The source clip is the tested route; a skeleton or depth layer is sent as a control video. A reference clip is 2 to 15 s; MiniMax H3 Ref2VA takes at most " + st.reference_clips + " of them, 15 s in all." : "Ready once a pack is: run an analysis or upload a pack.")));
  if (MO.ref) put(box, el("div", { class: "eg", style: "max-width:none" }, el("b", {}, "Stored motion reference"), el("div", { class: "mgrid", style: "margin-top:0" },
    MO.ref.clip ? el("video", { src: file(MO.ref.clip), controls: true, muted: true, loop: true, preload: "metadata", style: "width:100%;border-radius:10px;background:#000" }) : el("div", { class: "note" }, "No clip: the camera move in words only."),
    el("div", {}, el("div", { class: "chips" }, el("span", { class: "pill done" }, (st.modes[MO.ref.mode] || [MO.ref.mode])[0]), MO.ref.clip ? el("span", { class: "pill" }, (MO.ref.layer || "") + " · " + MO.ref.width + "×" + MO.ref.height + " · " + MO.ref.seconds + " s · " + MO.ref.frames + " frames") : null,
        MO.ref.source && MO.ref.source.speed !== 1 && MO.ref.clip ? el("span", { class: "pill" }, "speed × " + MO.ref.source.speed.toFixed(2)) : null),
      el("div", { style: "margin-top:8px" }, MO.ref.camera || "No camera move in words."), el("div", { class: "note" }, MO.ref.instruction),
      el("button", { class: "small", style: "margin:8px 6px 0 0", onclick: () => { if (typeof useMotionReference === "function") useMotionReference(MO.ref, (MO.packs || []).find(p => p.id === MO.ref.pack)); } }, "Open it in the create wizard"),
      el("button", { class: "small", style: "margin-top:8px", onclick: async () => { await api("/api/motion/handoff", undefined, "DELETE"); MO.ref = null; try { window.MOTION_REFERENCE = null; localStorage.removeItem("motion-reference"); } catch {} render(); } }, "Forget it")))));
  return box;
}
function viewMotion() {
  const box = el("section", { class: "card" }, el("h2", {}, "Motion Previs"));
  if (!MO.state) { moLoad(); return put(box, el("div", { class: "note" }, "Loading…")), box; }
  if (MO.state.error) return put(box, el("div", { class: "tip" }, el("b", {}, "Not available"), MO.state.error)), box;
  const c = MO.state.credit;
  put(box, el("div", { class: "note", style: "margin-top:-4px" }, "Turn a shot into control layers and a camera move: pose, depth, edges, a motion mask, and the move in words. The analysis is a Python port of ",
    el("a", { href: c.url, target: "_blank", rel: "noopener" }, c.name), " by ", el("b", { style: "color:var(--text)" }, c.author), " (" + c.license + "), modified; packs made here use its production pack's file names."));
  put(box, moSource(), moSettings());
  const run = el("button", { id: "mo-run", class: "hot", disabled: !moValid() || moActive(), onclick: moRun }, "Run analysis");
  put(box, el("div", { class: "row", style: "margin-top:16px" }, el("div", { class: "fit" }, run), el("div", { class: "note", style: "margin:0" }, MO.source ? "Runs on the processor, beside the render queue: a 5 second shot takes well under a minute." : "Pick a clip first.")), moRail());
  if (MO.packs && MO.packs.length) put(box, el("div", { class: "row", style: "margin-top:18px" }, field("Pack", select(MO, "packId", Object.fromEntries(MO.packs.map(p => [p.id, (p.source || p.id) + " · " + (MO.state.modes[p.mode] || [p.mode])[0] + " · " + p.id.slice(0, 11) + (p.imported ? " · uploaded" : "")])), () => { MO.hand.layer = MO.hand.mode = ""; render(); })),
    el("button", { class: "fit", title: "Move this pack to the trash folder", onclick: async () => { const p = moPack(); if (!p || !confirm("Delete the pack " + p.id + "?\nIt is moved to the trash folder inside " + S.home + ", not erased.")) return;
      try { await api("/api/motion/packs/" + p.id, undefined, "DELETE"); MO.packs = await api("/api/motion/packs"); MO.packId = MO.packs.length ? MO.packs[0].id : null; MO.ref = (await api("/api/motion/handoff")).reference; } catch (e) { toast(e.message); } render(); } }, "🗑")));
  const pack = moPack();
  if (pack) put(box, moReview(pack));
  put(box, moHandoff(pack));
  if (moActive() && !MO.timer) moWatch();
  return box;
}
