// Voice tab (app/voice.py): Fish Audio Drama 3 on this machine, nothing sent to Fish Audio. Built with Fish Audio.
"use strict";
const VO = { takes: null, text: "", ref: null, refText: "", consent: false, seed: 1, temperature: 1.0, top_p: 0.9, top_k: 30, job: null, cutting: "", player: null, stream: true };
// one turn a line: on one line Drama 3 drops later turns (voice_text.py dialogue breaks the lines too)
const VO_EXAMPLE = "<|speaker:0|> [whispering] Did you hear that noise downstairs?\n<|speaker:1|> [sighs] It's probably just the cat. Go back to sleep.\n<|speaker:0|> Cats don't knock on doors.";
const voOn = () => !!(S && S.voice && S.voice.available);
const voActive = () => VO.job && ["queued", "running"].includes(VO.job.status);
// a take plays while it is made (GET /api/voice/takes/{id}/live): one player a take, kept across redraws
function voLive(job) {
  if (!(S.voice && S.voice.live) || !job || job.status !== "running" || (job.params && job.params.live === false)) return null;
  if (!VO.player || VO.player.id !== job.id) VO.player = { id: job.id, el: el("audio", { src: "/api/voice/takes/" + encodeURIComponent(job.id) + "/live", controls: true, autoplay: true, preload: "auto", style: "width:100%;margin-top:8px" }) };
  return el("div", {}, el("div", { class: "note", style: "margin:8px 0 0" }, "Playing while it is made:"), VO.player.el); }
const voSpeed = st => st.backend === "cuda"
  ? "One take at a time in the queue, never beside a video render (Drama 3 takes about 42 GB). About 18 s to load the first time; it stays loaded for " + Math.round(((st.server || {}).idle_unload_s || 300) / 60) + " minutes after a take and is unloaded at once when a video or picture job starts. Then about 0.7 s for each second of speech, the first sound after about 0.3 s, played as it comes; Whisper checks the words after."
  : st.serves ? "One take at a time in the queue, never beside a video render (Drama 3 takes about 50 GB with its word check). About 5 s to load the first time; it stays loaded for " + Math.round(((st.server || {}).idle_unload_s || 300) / 60) + " minutes after a take and is unloaded at once when a video or picture job starts. Then about 0.35 s for each second of speech, the first sound after about 0.2 s, played as it comes. The engine checks its own words with Whisper as it goes: a runaway tail is cut, and a part with missing words is drawn again (a take that was short while playing is made once more)."
  : st.checks ? "One take at a time in the queue, never beside a video render (Drama 3 takes about 50 GB). About 4 s to load, then about 0.35 s for each second of speech, the first sound after about 0.25 s when it plays as it is made. The engine checks its own words with Whisper as it goes: a runaway tail is cut, and a part with missing words is drawn again (when it is not played as it is made) or reported."
  : "One take at a time in the queue, never beside a video render (Drama 3 takes about 50 GB). About 3 s to load, then roughly 0.4 s for each second of speech; Whisper checks the words after.";
// the MLX engine's own end-of-text check, part by part (take.verify): what it did and what it did not hear
function voCheck(t) {
  if (!t.verify || !t.verify.length) return null;
  const n = a => t.verify.filter(v => v.action === a).length, redrawn = t.verify.reduce((x, v) => x + Math.max(0, (v.attempts || 1) - 1), 0);
  const missing = t.verify.flatMap(v => v.missing || []), late = t.verify.filter(v => v.late_s).length;
  return (t.asr && t.asr.by === "whisper" ? "word check: " : "engine check: ") + [n("ok") ? n("ok") + " ok" : null, n("trim") ? n("trim") + " trimmed" : null, redrawn ? redrawn + " redrawn" : null,
    n("redraw") ? n("redraw") + " still short" : null, late ? late + " cut after playing" : null].filter(Boolean).join(", ") + (t.verify.length > 1 ? " of " + t.verify.length + " parts" : "")
    + (missing.length ? " · not heard: " + missing.slice(0, 8).join(", ") + (missing.length > 8 ? "…" : "") : ""); }
async function voLoad() { try { VO.takes = await api("/api/voice/takes"); } catch { VO.takes = []; } if (tab === "Voice") render(); }
const voLine = t => [t.duration != null ? t.duration.toFixed(1) + " s" : null, t.frames_per_s ? t.frames_per_s + " frames/s" : null, t.rtf ? "made in " + t.rtf + "× its length" : null,
  t.first_audio_ms ? "first sound after " + (t.first_audio_ms / 1000).toFixed(2) + " s" : null, t.load_s ? "model loaded in " + Math.round(t.load_s) + " s" : null,
  t.asr && t.asr.score != null ? "Whisper" + (t.asr.by === "engine" ? " (engine)" : "") + " " + t.asr.score + " % (" + t.asr.found + " of " + t.asr.words + " words)" : t.asr && t.asr.skipped ? "no word check: " + t.asr.skipped : null,
  "seed " + t.seed, t.reference ? "cloned voice" : null, voCheck(t), t.retaken ? "made again without playing: a part was short while it played" : null].filter(Boolean).join(" · ");
// a take (or any sound file of the studio) as a reference audio clip of a MiniMax H3 Ref2VA clip: 2 to 15 s each
async function voUseAsRef(path, seconds) {
  if (!S.ref_media) return toast("Reference audio needs MiniMax H3 Ref2VA, which is not installed here (" + (S.platform === "spark" ? "oneshot-setup-spark.sh" : "oneshot-setup.sh") + " --ref2va).");
  let d = Number(seconds) || 0;
  if (d > 15.05) { if (!confirm("This take is " + d.toFixed(1) + " s and a reference audio clip is 2 to 15 seconds.\nUse its first 15 seconds?")) return;
    try { VO.cutting = path; render(); const r = await api("/api/voice/cut", { path, seconds: 15 }); path = r.path; d = r.duration || 15; } catch (e) { return toast(e.message); } finally { VO.cutting = ""; } }
  if (d && d < 1.95) return toast("This take is " + d.toFixed(1) + " s: a reference audio clip is at least 2 seconds.");
  const meta = await refMeta(path), why = refFull("audios", C.refAudios) || refWrong("audios", C.refAudios, meta);
  if (C.refAudios.includes(path)) return toast("It is a reference already");
  if (why) return toast("Not added: " + why);
  Object.assign(C, { goes: "video", start: "refs", step: 1, job: null }); C.refAudios.push(path); C.audio_notes.push(""); C.prompt = "";
  sync(); tab = "Create"; history.replaceState(null, "", "#Create"); render();
  toast("Added as " + audioName(C.refAudios.length - 1) + (C.refs.length || C.refVideos.length ? "" : ": add a picture or a video of who speaks, audio cannot be the only reference") + ". Describe the same voice in the line's Voice box.", 6000); }
async function voRender() {
  if (!VO.text.trim()) return toast("Write what is said first");
  if (VO.ref && !VO.consent) return toast("Tick the consent box to clone this voice, or take the clip out");
  try { VO.job = await api("/api/voice/takes", { text: VO.text, seed: VO.seed, temperature: VO.temperature, top_p: VO.top_p, top_k: VO.top_k, live: VO.stream,
    ...(VO.ref ? { ref_audio: VO.ref, ref_text: VO.refText, consent: VO.consent } : {}) }); } catch (e) { return toast(e.message); }
  await poll(); render(); }
function voReference() {
  const up = el("input", { type: "file", accept: "audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg", disabled: !voOn(), onchange: async () => { if (!up.files[0]) return; const data = new FormData(); data.append("file", up.files[0]);
    try { const j = await (await fetch("/api/upload", { method: "POST", body: data })).json(); if (!j.path) throw new Error(j.error || "upload failed");
      const m = await refMeta(j.path), lim = S.voice.reference_seconds;
      if (m.duration && (m.duration < lim[0] - 0.05 || m.duration > lim[1] + 0.05)) return toast("A voice to clone is " + lim[0] + " to " + lim[1] + " seconds of speech, and this clip is " + m.duration.toFixed(1) + " s");
      VO.ref = j.path; VO.consent = false; render(); } catch (e) { toast(e.message); } } });
  return el("div", { class: "group" }, el("h3", {}, "Voice to clone (optional)"),
    el("div", { class: "note" }, "A clean recording of one speaker, about 10 to 30 seconds, and what it says. Without one, Drama 3 makes a voice of its own (describe it in a [direction], e.g. “[a deep, gravelly man in his fifties, slow and patient]”). Clone a real person's voice only with their consent."),
    VO.ref ? el("div", {}, el("div", { class: "row", style: "align-items:center;margin-top:8px" }, el("audio", { src: file(VO.ref), controls: true, preload: "metadata", style: "width:100%" }),
        el("button", { class: "small fit", onclick: () => { VO.ref = null; VO.refText = ""; VO.consent = false; render(); } }, "Take it out")),
      field("What the clip says (left empty, Whisper writes it)", bind(VO, "refText", el("textarea", { style: "min-height:52px", placeholder: "the exact words of the recording" }))),
      el("label", { class: "check" }, el("input", { type: "checkbox", checked: VO.consent, onchange: e => { VO.consent = e.target.checked; render(); } }), el("span", {}, "I have the speaker's consent to clone this voice")))
      : field("Add a clip", up)); }
function voTake(t) {
  return el("div", { class: "seg", style: "margin-top:10px" }, el("div", { class: "seghead" }, el("b", {}, t.title || "Voice take"), el("span", { style: "flex:1" }), el("span", { class: "note", style: "margin:0" }, new Date(t.created * 1000).toLocaleString())),
    el("audio", { src: file(t.path), controls: true, preload: "metadata", style: "width:100%;margin-top:6px" }),
    el("div", { class: "note", title: t.asr && t.asr.heard ? "Whisper heard: " + t.asr.heard : "" }, voLine(t)),
    el("div", { class: "note", style: "white-space:pre-wrap" }, t.text.length > 300 ? t.text.slice(0, 300) + "…" : t.text),
    el("div", { class: "row", style: "margin-top:8px;flex-wrap:wrap;gap:6px" },
      el("a", { class: "fit", href: "/api/voice/takes/" + encodeURIComponent(t.id) + "/audio", download: "drama3-" + t.id + ".wav" }, el("button", { class: "small", type: "button" }, "Download WAV")),
      el("button", { class: "small fit", disabled: VO.cutting === t.path, title: "Opens the create wizard with this take as a reference audio clip (2 to 15 s; a longer take is cut to its first 15 s)", onclick: () => voUseAsRef(t.path, t.duration) }, VO.cutting === t.path ? "Cutting…" : "Use as Ref2VA voice reference"),
      el("button", { class: "small fit", title: "Clone this take's voice for the next take: the same speaker across lines", disabled: !voOn() || (t.duration || 0) < 2, onclick: () => { VO.ref = t.path; VO.refText = t.reference ? "" : t.text.replace(/<\|speaker:\d+\|>|\[[^\]]*\]/g, " ").replace(/\s+/g, " ").trim(); VO.consent = false; window.scrollTo(0, 0); render(); } }, "Clone this voice"),
      el("button", { class: "small fit", title: "Put this take's text and settings back in the form", onclick: () => { Object.assign(VO, { text: t.text, seed: t.seed + 1, temperature: t.temperature, top_p: t.top_p, top_k: t.top_k }); window.scrollTo(0, 0); render(); } }, "Again, next seed"))); }
function viewVoice() {
  const box = el("section", { class: "card" }, el("h2", {}, "Voice · Fish Audio Drama 3"));
  if (VO.takes === null) { VO.takes = []; voLoad(); }
  const st = S.voice || { available: false, why: "This app does not know Drama 3.", defaults: {} };
  if (VO.job) { const fresh = jobs.find(j => j.id === VO.job.id); if (fresh) { if (fresh.status === "done" && VO.job.status !== "done") voLoad(); VO.job = fresh; } }
  put(box, el("div", { class: "note", style: "margin-top:-4px" }, "Speech from text with Fish Audio Drama 3, on this machine: several speakers, directions for the delivery, a voice cloned from a clip. A take can go to a MiniMax H3 reference clip as its voice (", el("b", {}, "Use as Ref2VA voice reference"), ")."),
    el("div", { class: "note", style: "color:var(--text)" }, el("b", {}, "Built with Fish Audio."), " " + (st.licence || "") + " Nothing is sent to Fish Audio: " + (st.label || "TensorFold") + "."));
  if (!st.available) put(box, el("div", { class: "tip" }, el("b", {}, "Not available here. "), st.why || ""));
  const text = el("textarea", { style: "min-height:130px", disabled: !st.available, placeholder: "What is said. <|speaker:0|> … <|speaker:15|> switch voices; [whispering], [sighs], [laughs] or a description in brackets steer the delivery." });
  bind(VO, "text", text);
  const num = (key, attrs) => bind(VO, key, el("input", { type: "number", disabled: !st.available, ...attrs }));
  put(box, el("div", { class: "group", "data-clear": "voice" }, el("h3", {}, "What is said"), text,
    el("div", { class: "note" }, "Speaker tags ", el("code", {}, "<|speaker:0|>"), " to ", el("code", {}, "<|speaker:15|>"), " switch voices (text without one is speaker 0). Directions in brackets steer the delivery: ", el("code", {}, "[whispering]"), ", ", el("code", {}, "[sighs]"), ", ", el("code", {}, "[laughs]"),
      ", or a description of the voice. Long text is split at sentence ends and keeps one voice. ", el("a", { href: "#", onclick: e => { e.preventDefault(); VO.text = VO_EXAMPLE; render(); } }, "Two-speaker example")),
    el("div", { class: "row", style: "margin-top:8px" }, el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Seed"), num("seed", { min: 0, step: 1 })),
      el("div", { class: "fit", style: "width:120px" }, el("label", { title: "Fish's default 1.0" }, "Temperature"), num("temperature", { min: 0.1, max: 2, step: 0.05 })),
      el("div", { class: "fit", style: "width:120px" }, el("label", { title: "Fish's default 0.9" }, "Top-p"), num("top_p", { min: 0.1, max: 1, step: 0.05 })),
      el("div", { class: "fit", style: "width:120px" }, el("label", { title: "Fish's default 30" }, "Top-k"), num("top_k", { min: 1, max: 200, step: 1 })),
      el("button", { class: "small fit", disabled: !st.available, title: "Fish's sampling: temperature 1.0, top-p 0.9, top-k 30", onclick: () => { Object.assign(VO, { temperature: 1.0, top_p: 0.9, top_k: 30 }); render(); } }, "Fish defaults"), ...clearButton("voice"))),
    voReference());
  put(box, el("div", { class: "row", style: "margin-top:16px;align-items:center" }, el("div", { class: "fit" }, el("button", { class: "hot", disabled: !st.available || voActive() || (VO.ref && !VO.consent), onclick: voRender }, voActive() ? "Rendering…" : "Render the take")),
    el("div", { class: "note", style: "margin:0" }, st.available ? voSpeed(st) : "")));
  if (st.available && st.checks) put(box, el("label", { class: "check", title: "Sent audio cannot be taken back: played as it is made, a part with missing words is reported, not drawn again" },
    el("input", { type: "checkbox", checked: VO.stream, onchange: e => { VO.stream = e.target.checked; } }), el("span", {}, "Play while it is made (off: a part with missing words is drawn again before you hear it)")));
  if (VO.job) put(box, VO.job.status === "failed" ? el("div", { class: "tip" }, el("b", {}, "The take failed. "), VO.job.error || "") : voActive() ? jobCard(VO.job) : null);
  if (VO.job) put(box, voLive(VO.job));
  put(box, el("h3", { style: "margin-top:22px" }, "Takes"), VO.takes.length ? VO.takes.slice(0, 40).map(voTake) : el("div", { class: "note" }, "No takes yet."));
  return box; }

// Slate Sound tab: a voice sheet's cast voice (POST /api/voice/sheet), kept on sheet.drama, its shots' reference audio
function slCastVoice(sheet) {
  const d = sheet.drama || null, job = d && d.job && jobs.find(j => j.id === d.job), active = job && ["queued", "running"].includes(job.status);
  if (d && d.job && job && job.status === "done" && job.voice) { sheet.drama = { path: job.voice.path, seconds: job.voice.duration, text: job.voice.text, seed: job.voice.seed, take: job.voice.id, asr: job.voice.asr ? job.voice.asr.score : null, use: d.use !== false }; slSave(); }
  const now = sheet.drama && sheet.drama.path ? sheet.drama : (sheet.drama && sheet.drama.was) || null, on = voOn(), long = now && now.seconds > 15.05;
  const cast = async seed => { try { const j = await api("/api/voice/sheet", { sheet, project: SL.p.id, seed }); sheet.drama = { job: j.id, use: now ? now.use !== false : true, was: now }; await poll(); slDraw(); } catch (e) { toast(e.message); } };
  return el("div", { class: "group" }, el("h3", {}, "Cast voice · Fish Audio Drama 3"),
    el("div", { class: "note" }, "Speaks the sample line in this voice with Fish Audio Drama 3 on this machine, the sheet's description given as the direction. Ticked, the take goes into shots that open on cast stills as this speaker's reference audio (<Audio N>, 2 to 15 s). Built with Fish Audio."),
    !on ? el("div", { class: "tip" }, el("b", {}, "Not available here. "), (S.voice && S.voice.why) || "") : null,
    el("div", { class: "row", style: "margin-top:8px;align-items:center" }, el("button", { class: "fit primary", disabled: !on || active, onclick: () => cast(now ? (Number(now.seed) || 0) + 1 : 1) }, active ? "Casting…" : now ? "Cast again (next seed)" : "Cast voice"),
      job && job.status === "failed" ? el("div", { class: "note", style: "color:var(--bad);margin:0" }, job.error || "failed") : el("span")),
    active ? jobCard(job) : null,
    now ? el("div", { style: "margin-top:10px" }, el("audio", { src: file(now.path), controls: true, preload: "metadata", style: "width:100%" }),
      el("div", { class: "note" }, [now.seconds != null ? Number(now.seconds).toFixed(1) + " s" : null, now.asr != null ? "Whisper " + now.asr + " %" : null, "seed " + now.seed].filter(Boolean).join(" · ")),
      long ? el("div", { class: "row", style: "align-items:center" }, el("div", { class: "note", style: "color:var(--warn);margin:0" }, "Longer than 15 s: a reference audio clip is 2 to 15 seconds."),
        el("button", { class: "small fit", onclick: async () => { try { const r = await api("/api/voice/cut", { path: now.path, seconds: 15 }); Object.assign(now, { path: r.path, seconds: r.duration || 15 }); slDraw(); } catch (e) { toast(e.message); } } }, "Cut to 15 s")) : null,
      el("label", { class: "check" }, el("input", { type: "checkbox", checked: now.use !== false, onchange: e => { now.use = e.target.checked; SL.plan = {}; slDraw(); } }), el("span", {}, "Use as this voice's Ref2VA reference in shots that open on cast stills")),
      el("div", { class: "row", style: "margin-top:6px" }, el("button", { class: "small fit", onclick: () => voUseAsRef(now.path, now.seconds) }, "Use in the create wizard"), el("button", { class: "small fit", onclick: () => { sheet.drama = null; SL.plan = {}; slDraw(); } }, "Remove the take"), el("span"))) : null); }
