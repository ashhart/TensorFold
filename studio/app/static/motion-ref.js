// Motion reference: Motion Previs on <Video 1> of a reference clip, in the create wizard.
"use strict";
// C.motion: the Motion Previs pack of the first reference video, what to carry over, the layer sent as <Video 1>
const MOTION_MODES = { "full-scene": ["Full scene", "the same place and motion, a new person"], "actor-motion": ["Actor motion", "the body motion and the camera, in a new place"],
  "object-motion": ["Object motion", "an object's path and the camera"], "camera-only": ["Camera only", "just the camera move and its timing"] };
const MOTION_LAYERS = { reference: ["Source clip", "the video itself: the tested route", "reference.mp4"], openpose: ["Skeleton", "OpenPose joints on black, one person only: the loosest hold", "openpose_pose.mp4"],
  "ai-depth": ["Depth", "Depth-Anything: for a new place, or the camera alone", "ai_depth.mp4"], depth: ["Depth proxy", "made by ffmpeg: brighter is nearer", "depth.mp4"], edges: ["Edges", "outlines on black", "edges.mp4"] };
// the default layer per choice, by measurement (motion.py SUGGESTED): source clip for a new person, else depth
const motionDefault = m => m.mode === "full-scene" ? (m.made === "reference" || m.layers.includes("reference") ? "reference" : m.made) : (["ai-depth", "depth"].find(l => l === m.made || m.layers.includes(l)) || m.made);
const MA = { state: null, job: null, timer: null }, PACE_FAST = 24;
const maBusy = () => !!MA.job && ["queued", "running"].includes(MA.job.status);
const maSay = () => { const j = MA.job; if (!j) return ""; const now = (j.stages || {})[j.stage] || {}; return j.status === "queued" ? "Waiting for its turn…" : "Analysing: " + (j.stage || "starting") + (now.of > 1 ? " " + now.at + "/" + now.of : "") + (j.elapsed != null ? " · " + mmss(j.elapsed) : ""); };
const motionControl = () => !!C.motion && C.refVideos.length > 0 && C.motion.layer !== "reference";
// what the prompt composer is told: nothing without an analysis or without a video for it
const motionFields = () => C.motion && C.refVideos.length ? { layer: C.motion.layer, mode: C.motion.mode, camera: C.motion.useCamera ? C.motion.camera : "", people: Number(C.motion.people) || 1, pace: C.motion.pace || "auto", measured: C.motion.measured || null, free: motionFree() } : null;
// a motion reference's sound and movement agree (app/pace.py): Impacts, Flow, or Automatic (no hits over slow motion)
const PACES = { auto: ["Automatic", "no hits over a slow reference"], impacts: ["Impacts", "fast strikes and blocks, with their sound"], flow: ["Flow", "slow sticky hands, silent or with music"] };
// Impacts over a reference that is not fast: the video is left out and the picture and prompt carry the fight
const paceSlowRef = m => !!m && paceOf(m) === "impacts" && !!m.measured && ["slow", "moderate"].includes(m.measured.motion);
const motionFree = () => paceSlowRef(C.motion) && C.motion.free !== false && C.refs.length > 0 && C.refVideos.length === 1;
// Automatic changes one case only: a soundscape that names hits, over a reference that moves slowly, becomes Flow
const IMPACT_WORDS = ["slap", "smack", "strike", "hit", "punch", "thud", "knock", "clap", "crack", "impact", "blow", "whack", "bang", "thwack", "pound", "slam", "kick", "stomp", "clash", "snap"];
const namesHits = text => IMPACT_WORDS.some(w => (text || "").toLowerCase().includes(w));
const musicNamed = () => !!(C.music || "").trim() && (C.music || "").trim().toUpperCase() !== "N/A";
const paceOf = m => m.pace && m.pace !== "auto" ? m.pace : m.measured && m.measured.motion === "slow" && namesHits(C.soundscape) ? "flow" : "";
function paceNote(m) { const d = m.measured, now = paceOf(m);
  const read = !d || d.limbs == null ? "The speed of <Video 1> could not be measured on this machine: choose Impacts or Flow yourself." : "Measured in <Video 1>: the quickest movement is " + d.limbs + " percent of the frame width a second (" + d.motion + "; fast is " + PACE_FAST + " or more)" + (d.sound === "impacts" ? ", and its sound holds " + d.hits + " sharp hits a second." : d.sound === "silent" ? ", and it has no sound." : ", and its sound holds no hits.");
  const auto = (m.pace || "auto") === "auto" ? (now === "flow" ? " The soundscape of step 3 names hits and this reference moves slowly, so Automatic makes this clip Flow." : " Automatic leaves the prompt as you write it, except that a soundscape naming hits over a slowly moving reference becomes Flow.") : "";
  const does = auto + (now === "impacts" ? " Impacts: the prompt says that the arms attack and block at full speed and that every strike is heard as it lands." : now === "flow" ? " Flow: the prompt says that the movement is slow and continuous, the hits are taken out of the soundscape" + (d && d.sound === "impacts" ? ", the video's own sound is not reused" : "") + ", and the clip is " + (musicNamed() ? "under the music named in step 3." : "delivered silent: its sound track is muted after the render (name music in step 3 to have music instead).") : "");
  const warn = paceSlowRef(m) ? el("div", { class: "tip", style: "margin-top:8px" }, el("b", {}, "This reference moves slowly"), "The model follows the movement of <Video 1> whatever the prompt says about speed, so the arms would stay as slow as they are in it and the hits would be heard over slow hands.",
    C.refs.length ? el("label", { class: "switch", style: "margin-top:8px" }, el("input", { type: "checkbox", checked: m.free !== false, onchange: e => { m.free = e.target.checked; C.prompt = ""; render(); } }), el("span", {}, "Leave <Video 1> out and let the prompt carry the fight", el("small", {}, "The clip is made from <Picture 1> and the prompt; the camera move of <Video 1> stays, in words. Describe the place in Scene: it no longer comes from the video.")))
      : el("div", { class: "note" }, "Add a picture of who is in the clip above, as <Picture 1>, and the video can be left out so that the prompt carries the fight; or choose Flow.")) : null;
  return el("div", {}, el("div", { class: "note", style: "margin-top:0" }, read + does), warn); }
function setMotion(pack, layer, made, mode) { const cam = pack.camera || {};
  C.motion = { pack: pack.id, mode: mode || "full-scene", layer, made, layers: pack.layers || [], files: pack.files || {}, camera: cam.sentence || "", reliable: !!cam.reliable, confidence: cam.confidence || 0,
    seconds: pack.range ? pack.range.duration : null, useCamera: !!cam.sentence, people: "1", pace: "auto", measured: pack.pace || null };
  C.prompt = ""; }
async function analyseMotion() {
  const path = C.refVideos[0]; if (!path || maBusy()) return;
  try { const meta = await api("/api/probe?path=" + encodeURIComponent(path));
    MA.job = await api("/api/motion/jobs", { source: path, start: 0, end: Math.floor(Math.min(meta.duration, 15) * 100) / 100, mode: "full-scene", fps: 12, title: "Motion of " + path.split("/").pop() });
  } catch (e) { toast(e.message); return; }
  render(); clearInterval(MA.timer);
  MA.timer = setInterval(async () => { try { MA.job = await api("/api/motion/jobs/" + MA.job.id); } catch { return; }
    if (maBusy()) { const say = document.getElementById("ma-say"); if (say) say.textContent = maSay(); return; }
    clearInterval(MA.timer); MA.timer = null;
    if (MA.job.status === "done" && C.refVideos[0] === path) { try { setMotion(await api("/api/motion/packs/" + MA.job.pack), "reference", "reference"); if (typeof MO !== "undefined") MO.state = null; } catch (e) { toast(e.message); } }
    if (tab === "Create") render(); }, 700);
}
// the clip sent as <Video 1>; with no spoken lines the source clip's mouth is smoothed first (motion.py soft_mouth)
const speaks = () => (C.dialogue || []).some(l => (l.text || "").trim());
const softMouth = () => { const m = C.motion; return !!m && C.refVideos.length > 0 && m.layer === "reference" && m.made === "reference" && m.softMouth !== false && !C.video_sound && !speaks() && !motionFree() && (m.layers.includes("openpose") || m.layers.includes("pose")); };
async function motionVideos() { const m = C.motion; if (motionFree()) return C.refVideos.slice(1);
  if (softMouth()) { const size = output().size, r = await api("/api/motion/reference", { pack: m.pack, layer: "reference", mode: m.mode, seconds: Math.max(2, Math.min(Number(C.seconds), m.seconds || 15)), width: size[0], height: size[1], timing: "trim", keep: false, mouth: "soft" });
    return [r.clip, ...C.refVideos.slice(1)]; }
  if (!m || !C.refVideos.length || m.layer === m.made) return C.refVideos;
  const size = output().size, r = await api("/api/motion/reference", { pack: m.pack, layer: m.layer, mode: m.mode, seconds: Math.max(2, Math.min(Number(C.seconds), m.seconds || 15)), width: size[0], height: size[1], timing: "trim", keep: false });
  return [r.clip, ...C.refVideos.slice(1)]; }
function motionPanel() {
  if (!C.refVideos.length) return null;
  if (!MA.state) { api("/api/motion/state").then(st => { MA.state = st; }, () => { MA.state = { available: {}, setup: "" }; }).then(() => { if (tab === "Create") render(); }); return null; }
  const st = MA.state, can = st.available || {}, m = C.motion, box = el("div", { class: "group" }, el("h3", {}, "Motion of <Video 1>"));
  if (!m) {
    if (!(can.pose || can.camera || can.ai_depth)) return put(box, el("div", { class: "note" }, "<Video 1> is used as a plain reference. Motion analysis (the camera move in words, a pose skeleton, a depth layer) is not installed on this machine" + (st.setup ? ": " : "."), st.setup ? el("span", { style: "font-family:var(--mono)" }, st.setup) : null, st.setup ? " adds it." : null)), box;
    return put(box, el("div", { class: "row" }, el("button", { type: "button", class: "fit primary", disabled: maBusy(), onclick: analyseMotion }, maBusy() ? "Analysing…" : "Analyse motion"),
      el("div", { class: "note", id: "ma-say", style: "margin:0" }, maBusy() ? maSay() : MA.job && MA.job.status === "failed" ? "The analysis failed (" + (MA.job.error || "no reason given") + "). The video is still used as a plain reference."
        : "Optional. Runs Motion Previs on <Video 1>, on the processor, in well under a minute: the camera move in words, a pose skeleton and a depth layer, then a choice of what the clip keeps of the video. Without it the video is a plain reference."))), box;
  }
  const again = () => { C.prompt = ""; render(); }, layers = Object.fromEntries(Object.entries(MOTION_LAYERS).filter(([k]) => k === "reference" ? (m.made === "reference" || m.layers.includes(k)) : m.layers.includes(k) && !(k === "depth" && m.layers.includes("ai-depth"))).map(([k, v]) => [k, [v[0], v[1]]]));
  if (!layers[m.layer]) m.layer = layers[m.made] ? m.made : Object.keys(layers)[0];
  const shown = m.layer === m.made ? C.refVideos[0] : m.files[MOTION_LAYERS[m.layer][2]];
  put(box, el("label", {}, "What to carry over"), choice(m, "mode", MOTION_MODES, () => { m.layer = motionDefault(m); again(); }, true),
    el("label", {}, "What goes to the model as <Video 1>"), choice(m, "layer", layers, again, true),
    el("label", {}, "People in <Video 1>"), choice(m, "people", { 1: ["One", "or none"], 2: ["Two or more", "the prompt then speaks of its people"] }, again, true),
    el("label", {}, "Sound and movement"), choice(m, "pace", PACES, () => { if (paceOf(m) === "flow" && m.measured && m.measured.sound === "impacts") C.video_sound = false; again(); }, true), paceNote(m),
    el("div", { class: "mgrid" },
      shown ? el("video", { src: file(shown) + "#t=0.1", controls: true, muted: true, loop: true, preload: "metadata", style: "width:100%;border-radius:10px;background:#000" }) : el("div", { class: "note" }, "No preview of this layer."),
      el("div", {}, motionControl() ? el("div", { class: "tip", style: "margin-top:0" }, el("b", {}, "A control layer"), "The model is told that <Video 1> is a diagram of the shot (its structure, not its look) and that <Picture 1> is the first frame. " + (C.refs.length ? "" : "Add a picture of the opening frame above, as <Picture 1>: the look of the clip comes from it. ") + "Measured on one shot: depth with a picture kept the moves nearly as well as the source clip and takes the setting from the picture; the skeleton held them loosely (a kick was lost). Neither left a trace of its look in the clip.")
          : el("div", { class: "note", style: "margin-top:0" }, "The source clip goes in as it is" + (C.refs.length ? ", with <Picture 1> as who replaces its person: the closest hold on the motion in our tests, in the clip's own place." : ".") + (m.mode !== "full-scene" ? " For a new place, depth is the better choice." : "")),
        m.layer === "reference" && m.made === "reference" && (m.layers.includes("openpose") || m.layers.includes("pose")) && !motionFree() ? el("label", { class: "switch", style: "margin-top:10px" }, el("input", { type: "checkbox", checked: m.softMouth !== false, disabled: C.video_sound || speaks(), onchange: e => { m.softMouth = e.target.checked; render(); } }), el("span", {}, "Keep the mouth of <Video 1> out of the clip", el("small", {}, C.video_sound || speaks() ? "Off: this clip has spoken lines or reuses the video's own sound, so the mouth is followed." : "Nobody speaks in this clip: the mouth of the person in <Video 1> is smoothed over before the video goes to the model. Left in, the new face mouths that person's words."))) : null,
        m.camera ? el("label", { class: "switch", style: "margin-top:10px" }, bind(m, "useCamera", el("input", { type: "checkbox" }), again), el("span", {}, "Add the camera move to the prompt", el("small", {}, m.camera))) : el("div", { class: "note" }, "No camera move was solved for this clip."),
        m.camera && !m.reliable ? el("div", { class: "note" }, "Low solve confidence (" + Math.round(100 * m.confidence) + "%): check the sentence against the clip.") : null,
        m.camera && m.useCamera && C.camera.trim() ? el("div", { class: "note" }, "The Camera field of step 1 has its own text, which is used instead.") : null,
        el("button", { type: "button", class: "small", style: "margin-top:10px", onclick: () => { C.motion = null; MA.job = null; again(); } }, "Forget the analysis: a plain reference"))));
  return box;
}
// "Use as motion reference" in the Motion Previs tab lands here, on the wizard's references screen
function useMotionReference(ref, pack) {
  if (!ref) return;
  if (!ref.clip) { if (ref.camera) C.camera = ref.camera; C.prompt = ""; C.goes = "video"; C.step = 0; tab = "Create"; history.replaceState(null, "", "#Create"); toast(ref.camera ? "The camera move is in the Camera field" : "This pack has no camera move in words"); sync(); render(); return; }
  if (!S.ref_media) { toast("Stored. A reference video needs MiniMax H3 Ref2VA, which is not installed here (" + (S.platform === "spark" ? "oneshot-setup-spark.sh" : "oneshot-setup.sh") + " --ref2va)."); return; }
  Object.assign(C, { goes: "video", start: "refs", refVideos: [ref.clip], video_subjects: [""], video_sound: false, step: 1, job: null });
  setMotion(pack || { id: ref.pack, layers: [ref.layer], files: {}, camera: {} }, ref.layer, ref.layer, ref.mode);
  Object.assign(C.motion, { camera: ref.camera || "", reliable: !!ref.camera_reliable, confidence: ref.camera_confidence || 0, useCamera: !!ref.camera, seconds: ref.seconds });
  C.seconds = Object.keys(S.durations).map(Number).sort((a, b) => a - b).find(d => d >= ref.seconds - 0.01) || 15;
  const named = { "864x480": "480p", "1280x736": "720p", "1344x768": "native" }[ref.width + "x" + ref.height], lim = S.custom;
  if (named) C.res = named; else if (ref.width % 32 === 0 && ref.height % 32 === 0 && ref.width * ref.height <= lim.pixels && Math.min(ref.width, ref.height) >= lim.sides[0] && Math.max(ref.width, ref.height) <= lim.sides[1]) Object.assign(C, { res: "custom", cw: ref.width, ch: ref.height });
  C.up = false; tab = "Create"; history.replaceState(null, "", "#Create"); window.scrollTo(0, 0); sync(); C.prompt = ""; render();
  refMeta(ref.clip).then(() => { if (tab === "Create") render(); });
}
