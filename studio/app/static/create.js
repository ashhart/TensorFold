// Create: the wizard from intent to render.
"use strict";
const STEPS = ["1. Intent / Mode", "2. First frame", "3. Words / Sound", "4. Review prompt", "5. Render"];
const C_TEXT = { scene: "", action: "", camera: "", dialogue: [], soundscape: "", music: "N/A", imagePrompt: "", prompt: "", beats: "" }, C_NOTES = { subjects: "refs", video_subjects: "refVideos", audio_notes: "refAudios" };
function clearAllCreate() { const keys = [...Object.keys(C_TEXT), ...Object.keys(C_NOTES)], was = JSON.parse(JSON.stringify(Object.fromEntries(keys.map(k => [k, C[k]]))));
  const empty = { ...JSON.parse(JSON.stringify(C_TEXT)), ...Object.fromEntries(Object.entries(C_NOTES).map(([k, of]) => [k, C[of].map(() => "")])) };
  if (JSON.stringify(was) === JSON.stringify(empty)) return toast("Nothing to clear here");
  Object.assign(C, empty); CLEARED = { key: "create-all", model: () => Object.assign(C, was) }; render(); undoToast("Every prompt field of this clip is empty."); }
function viewCreate() {
  const body = el("div");
  const steps = el("div", { class: "steps" }, STEPS.map((name, i) => el("button", { class: i === C.step ? "on" : "", onclick: () => { C.step = i; render(); } }, i === 1 && isRef() ? "2. " + refWord() : name)));
  const next = el("div", { class: "row", style: "margin-top:14px" }, el("button", { class: "fit", disabled: C.step === 0, onclick: () => { C.step--; render(); } }, "Back"),
    el("span"), C.step === 3 && C.mode !== "image" ? null : clearButton("create-" + C.step, "Clear this step"), el("button", { class: "small fit", title: "Empty every prompt field of this clip: scene, action, camera, lines, sound, and the composed prompt. Pictures, clips and settings stay.", onclick: clearAllCreate }, "Clear all"),
    C.step < 4 ? el("button", { class: "primary fit", onclick: () => { C.step++; render(); } }, "Next") : null);
  body.setAttribute("data-clear", "create-" + C.step);
  sync();
  const goes = el("div", {}, el("label", {}, "The prompt goes to"), choice(C, "goes", { video: ["Video with sound", S.turbo ? "FastH3 or MiniMax H3" : "FastH3"], image: ["Image only", imageModels().map(m => m.label).join(", ")], edit: ["Edit a picture", "Qwen-Image-2.1 on a picture you give it"] }, changed));
  const picture = field("Describe the scene / condition where it takes place", bind(C, "scene", el("textarea", { placeholder: isRef() ? "The woman with red hair sits at a wooden desk in a small radio studio at night, medium close-up, a warm lamp on the left." : "A middle-aged baker in a cream shirt and green apron behind a wooden counter in a small bakery at dawn, warm window light from the left…" })));
  if (C.mode === "edit") return viewEdit(goes);  // a picture re-drawn by the image model (edit.js)
  if (C.mode === "image") {
    // the prompt goes to the image model and stops there
    const made = C.imageJob && jobs.find(j => j.id === C.imageJob);
    const [w, h] = C.imageSize.split("x").map(Number);
    put(body, el("h2", {}, "What are you making?"), goes, picture, imageModelPick(),
      el("label", {}, "Size"), choice(C, "imageSize", { "1344x768": ["16:9", "1344 × 768"], "1664x928": ["16:9 large", "1664 × 928"], "1024x1024": ["Square", "1024 × 1024"], "768x1344": ["9:16", "768 × 1344"] }, render, true),
      el("div", { class: "row" },
        el("div", { class: "fit", style: "width:120px" }, el("label", {}, "How many"), bind(C, "count", el("input", { type: "number", min: 1, max: 12 }))),
        el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Seed"), bind(C, "seed", el("input", { type: "number", min: 0 })))),
      el("div", { class: "note" }, "Each image takes a few seconds after the model loads; seeds count up from the one given."),
      el("div", { class: "row", style: "margin-top:10px" }, el("button", { class: "hot fit", onclick: async () => { try { if (!C.scene.trim()) throw new Error("describe the picture first");
        const j = await api("/api/jobs", { kind: "scout", title: "Image: " + C.scene.slice(0, 40), params: { prompt: C.scene, count: C.count, seed: C.seed, width: w, height: h, image_model: imageModel() } });
        C.imageJob = j.id; await poll(); render(); } catch (e) { toast(e.message); } } }, "Make image"), el("span")));
    if (made) { put(body, el("h3", {}, "Images"), jobCard(made));
      if (made.status === "done") put(body, el("div", { class: "grid", style: "margin-top:10px" }, made.outputs.filter(p => !p.endsWith("sheet.jpg")).map(p =>
        el("div", { class: "tile" }, media(p), el("div", { class: "cap" }, el("a", { href: file(p), download: "" }, "Download"),
          el("a", { href: "#", onclick: e => { e.preventDefault(); C.image = p; C.goes = "video"; C.start = "own"; C.step = 1; changed(); } }, "animate this")))))); }
    return el("section", { class: "card" }, body);
  }
  if (C.step === 0) {
    const est = el("span", { class: "eta" });
    put(body, el("h2", {}, "What are you making?"), goes,
      el("label", {}, "The clip starts from"), choice(C, "start", { text: ["Text only", "no first frame"], scout: ["A scout image", "make several with " + imageModels().map(m => m.label).join(" or ") + ", pick one"], own: ["Your own image", "upload a picture"],
        ...(S.ref2va ? { refs: S.ref_media ? ["References", "pictures, videos or audio: keep a face, a motion or a voice"] : ["Reference pictures", "keep a face or a thing, in a new scene"] } : {}) }, changed),
      isRef() ? el("div", { class: "note", style: "margin:8px 0" }, "You add them on the next screen: ", el("a", { href: "#", onclick: e => { e.preventDefault(); C.step = 1; render(); } }, "2. " + refWord()), ".") : null,
      videoControls(() => { render(); }), summary(est), picture,
      field("What happens to the characters inside the scene / clip", bind(C, "action", el("textarea", { placeholder: "He lifts the loaf with both hands, sets it down, looks at the camera and smiles." }))),
      field("Camera (optional)", bind(C, "camera", el("input", { placeholder: "The camera slowly pushes in throughout the single continuous shot." }))),
      isLong() ? field("Story beats (optional): one line per 15 s clip, in order; the last line holds for the rest", bind(C, "beats", el("textarea", { placeholder: "He rides down the chute.\nHe stops at the river and looks back.\nHe wades across carrying the bike." }))) : null,
      isLong() ? el("div", { class: "note" }, "The model makes at most 15 s at a time, so this is rendered as " + Math.ceil(C.seconds / 15) + " clips, each opening on the last frame of the one before, then joined. Finished clips appear in the queue as they land. Expect the look to drift over many clips; beats help it stay on a story.") : null);
    showEstimate(C, est);
  } else if (C.step === 1) {
    if (isRef()) {
      for (const path of [...C.refVideos, ...C.refAudios]) if (!REF_META[path]) refMeta(path).then(() => { if (tab === "Create" && C.step === 1) render(); });
      const send = async one => { const data = new FormData(); data.append("file", one); const r = await fetch("/api/upload", { method: "POST", body: data }); const j = await r.json().catch(() => ({})); if (!j.path) throw new Error(j.detail || j.error || "upload failed"); return j.path; };
      const add = el("input", { type: "file", accept: "image/*", multiple: true, onchange: async () => { REF_SAID.pictures = "";
        for (const one of add.files) { try { const full = refFull("pictures", C.refs); if (full) throw new Error(full); const path = await send(one); C.refs.push(path); C.subjects.push(""); } catch (e) { REF_SAID.pictures = one.name + ": " + e.message; break; } }
        C.prompt = ""; render(); } });
      // one uploader per kind, numbered in order; a clip is measured as it arrives and refused here with the reason
      const uploader = (accept, list, labels, kind) => { const input = el("input", { type: "file", accept, multiple: true, onchange: async () => { REF_SAID[kind] = "";
        for (const one of input.files) { try { const full = refFull(kind, list); if (full) throw new Error(full);
            const path = await send(one), meta = await refMeta(path), wrong = refWrong(kind, list, meta);
            if (wrong) throw new Error(wrong);
            list.push(path); labels.push("");
          } catch (e) { REF_SAID[kind] = one.name + ": " + e.message; break; } }
        if (kind === "videos" && C.refVideos.length === 1 && !C.motion) C.video_sound = REF_SOUND_DEFAULT && !soundless().length;
        C.prompt = ""; render(); } }); return input; };
      const said = kind => REF_SAID[kind] ? el("div", { class: "note", style: "color:var(--warn)" }, REF_SAID[kind]) : null;
      const drop = (list, labels, i) => el("a", { href: "#", onclick: e => { e.preventDefault(); list.splice(i, 1); labels.splice(i, 1); if (list === C.refVideos && i === 0) { C.motion = null; MA.job = null; } if (!C.refVideos.length) C.video_sound = false; Object.assign(REF_SAID, { pictures: "", videos: "", audios: "" }); C.prompt = ""; render(); } }, "remove");
      const length = path => REF_META[path] && REF_META[path].duration ? " · " + REF_META[path].duration.toFixed(1) + " s" : "";
      const quiet = soundless();
      put(body, el("h2", {}, refWord()), el("div", { class: "note" }, "Up to " + S.reference_limit + " pictures of who or what must look the same in the clip. They are not frames of it: the scene comes from the prompt, which names them <Picture 1>, <Picture 2>, … in this order. A clear, front-facing picture works best; any aspect ratio." + (S.ref_media ? " At most " + S.reference_total + " references of all kinds." : "")),
        field("Add pictures", add), said("pictures"),
        C.refs.length ? el("div", { class: "grid", style: "margin-top:10px" }, C.refs.map((p, i) => el("div", { class: "tile on" }, media(p),
          el("div", { class: "cap" }, "<Picture " + (i + 1) + ">", drop(C.refs, C.subjects, i)),
          bind(C.subjects, i, el("input", { placeholder: "who or what this is: the woman with red hair" }), () => { C.prompt = ""; })))) : null,
        S.ref_media ? [
          el("h3", { style: "margin-top:18px" }, "Reference videos"),
          el("div", { class: "note" }, "Up to " + S.reference_video_limit + " clips (mp4 or mov), each 2 to 15 seconds and 15 seconds in all: <Video 1>, <Video 2>, … A video carries a subject and how it moves. It is read up to the clip's own length. Every second of it adds work to each pass: a 5 second reference about " + (S.platform === "spark" ? "doubles" : "triples") + " the time of a 5 second clip."),
          field("Add videos", uploader("video/mp4,video/quicktime,.mp4,.mov", C.refVideos, C.video_subjects, "videos")), said("videos"),
          C.refVideos.length ? el("div", { class: "grid", style: "margin-top:10px" }, C.refVideos.map((p, i) => el("div", { class: "tile on" }, media(p),
            el("div", { class: "cap" }, el("span", {}, "<Video " + (i + 1) + ">" + length(p) + (i === 0 && C.motion && C.motion.layer !== C.motion.made ? " · sent as its " + MOTION_LAYERS[C.motion.layer][0].toLowerCase() + " layer" : "") + (C.video_sound ? " · its sound is <Audio " + (i + 1) + ">" : "")), drop(C.refVideos, C.video_subjects, i)),
            bind(C.video_subjects, i, el("input", { placeholder: "who or what it shows (empty: only the motion is used)" }), () => { C.prompt = ""; })))) : null,
          C.refVideos.length ? el("label", { class: "check" + (quiet.length ? " off" : "") }, el("input", { type: "checkbox", checked: C.video_sound && !quiet.length, disabled: quiet.length > 0, onchange: e => { C.video_sound = e.target.checked; C.prompt = ""; render(); } }),
            el("span", {}, "Use the videos' own sound as a reference too. " + (quiet.length ? "Off: " + quiet.map(k => "<Video " + k + ">").join(" and ") + (quiet.length > 1 ? " have" : " has") + " no sound."
              : VIDEO_SOUND_SAYS + (C.refAudios.length ? " The audio clips below are then numbered after the videos." : "")))) : null,
          motionPanel(),
          el("h3", { style: "margin-top:18px" }, "Reference audio"),
          el("div", { class: "note" }, "Up to " + S.reference_audio_limit + " sound clips (wav, mp3, m4a, aac, flac, ogg), each 2 to 15 seconds and 15 seconds in all. Only beside a picture or a video: the model does not take sound alone. A clip longer than the video is cut to its length. " + AUDIO_SAYS),
          field("Add audio", uploader("audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg", C.refAudios, C.audio_notes, "audios")), said("audios"),
          C.refAudios.length ? el("div", { class: "grid", style: "margin-top:10px" }, C.refAudios.map((p, i) => el("div", { class: "tile on" }, el("audio", { src: file(p), controls: true, preload: "metadata", style: "width:100%" }),
            el("div", { class: "cap" }, el("span", {}, audioName(i) + length(p)), drop(C.refAudios, C.audio_notes, i)),
            bind(C.audio_notes, i, el("input", { placeholder: "what it is: the voice timbre reference for the woman's voice" }), () => { C.prompt = ""; })))) : null,
          C.refAudios.length && !C.refs.length && !C.refVideos.length ? el("div", { class: "note", style: "color:var(--warn)" }, "Add a picture or a video too: audio cannot be the only reference.") : null ] : null);
    } else if (noFrame()) put(body, el("h2", {}, "First frame"), el("div", { class: "note" }, "Text to video has no first frame. Change “The clip starts from” in step 1 to start from an image. A last frame alone works too: the clip then arrives at that picture."), lastFramePanel());
    else {
      const up = el("input", { type: "file", accept: "image/*", onchange: async () => { const data = new FormData(); data.append("file", up.files[0]);
        const r = await fetch("/api/upload", { method: "POST", body: data }); const j = await r.json(); if (j.path) { C.image = j.path; render(); } else toast(j.error || "upload failed"); } });
      put(body, el("h2", {}, "First frame"));
      if (C.mode === "scout") {
        if (!C.imagePrompt) C.imagePrompt = C.scene;
        const scout = C.scoutJob && jobs.find(j => j.id === C.scoutJob);
        put(body, imageModelPick(), field("Image prompt (" + imageModelName() + ")", bind(C, "imagePrompt", el("textarea"))),
          el("div", { class: "row" }, el("div", { class: "fit", style: "width:120px" }, el("label", {}, "How many"), bind(C, "count", el("input", { type: "number", min: 1, max: 12 }))),
            el("button", { class: "primary fit", disabled: !C.imagePrompt.trim(), onclick: async () => { try { const j = await api("/api/jobs", { kind: "scout", title: "Scouts",
              params: { prompt: C.imagePrompt, count: C.count, seed: C.seed, preset: C.preset, image_model: imageModel(), ...(C.preset === "custom" ? { width: Number(C.cw), height: Number(C.ch) } : {}) } }); C.scoutJob = j.id; await poll(); render(); } catch (e) { toast(e.message); } } }, "Make scouts"), el("span")),
          scout ? el("div", { class: "note" }, "Scout job: ", el("span", { class: "pill " + scout.status }, scout.status)) : el("div", { class: "note" }, "Six scouts take about half a minute. They are made at the size the video model starts from."));
        if (scout && scout.status === "done") put(body, el("div", { class: "grid", style: "margin-top:10px" }, scout.outputs.filter(p => !p.endsWith("sheet.jpg")).map(p =>
          el("div", { class: "tile" + (C.image === p ? " on" : ""), onclick: () => { C.image = p; render(); } }, media(p), el("div", { class: "cap" }, p.split("/").pop(), C.image === p ? "✓ selected" : "pick")))));
      } else put(body, field("Upload an image", up), clipFrames(p => { C.image = p; C.prompt = ""; }, "Or continue from a clip in the Library"));
      if (C.mode !== "scout" && C.image && C.image.startsWith("scouts/")) put(body, el("div", { class: "note" }, "Using the image you chose to animate."));
      if (C.image) put(body, el("h3", {}, "Selected first frame"), el("div", { style: "max-width:420px" }, media(C.image, { style: "width:100%;border-radius:8px" })),
        el("div", { class: "note" }, "The image is stretched onto the clip's canvas, so match its aspect ratio (16:9 for the presets here)."));
      put(body, lastFramePanel());
    }
  } else if (C.step === 2) {
    put(body, el("h2", {}, "Words and sound"), el("div", { class: "note" }, "Spoken or sung lines are written into the prompt exactly. Leave empty for a clip with no speech."),
      C.dialogue.map((line, i) => el("div", { class: "seg" }, el("div", { class: "row" }, field("Voice", bind(line, "voice", el("input", { placeholder: "a calm, slightly raspy male voice" }))),
        keep(el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Language"), bind(line, "language", el("input")))),
        isRef() && refPeople() > 1 ? el("div", { class: "fit", style: "width:140px" }, el("label", {}, "Speaker"), select(line, "subject", Object.fromEntries(Array.from({ length: refPeople() }, (_, k) => [k + 1, "<Subject " + (k + 1) + ">"])), () => { C.prompt = ""; })) : null,
        el("label", { class: "check fit" }, bind(line, "sung", el("input", { type: "checkbox" })), el("span", {}, "sung")),
        el("button", { class: "small fit", onclick: () => { C.dialogue.splice(i, 1); render(); } }, "Remove")), field("Line", bind(line, "text", el("textarea", { style: "min-height:54px" }))))),
      el("button", { onclick: () => { C.dialogue.push({ voice: "", language: "English", text: "", sung: false }); render(); } }, "+ Add a line"),
      field("Soundscape", bind(C, "soundscape", el("textarea", { placeholder: "Quiet bakery room tone and a low oven hum. A light ceramic tap as the loaf is set down." }))),
      field("Music (N/A for none)", bind(C, "music", el("input", { "data-clear-to": "N/A" }))));
  } else if (C.step === 3) {
    const area = bind(C, "prompt", el("textarea", { style: "min-height:260px;font-family:ui-monospace,monospace;font-size:12px" }));
    const again = async () => { const r = await api("/api/compose", { fields: { ...C, motion: isRef() ? motionFields() : null }, first_frame: !noFrame(), last_frame: !!lastOf(), ...lengthOf(C), references: isRef() ? Math.max(C.refs.length, refMedia() ? 0 : 1) : 0, referenced: isRef(), loras: isRef() ? lorasOf(C) : [] }); C.prompt = r.prompt; area.value = r.prompt; negShow(warned); };
    const warned = el("div"); area.addEventListener("change", () => negShow(warned));
    if (!C.prompt) again(); else negShow(warned);
    put(body, el("h2", {}, "Review the prompt"), el("div", { class: "note" }, "This is what the video model receives, in MiniMax H3's own three-part format. Edit it freely; “Rebuild” discards edits."), area, warned,
      el("div", { class: "row", style: "margin-top:8px" }, el("button", { class: "fit", onclick: again }, "Rebuild from steps 1–3"), el("span")));
  } else {
    const est = el("div", { class: "note" }); showEstimate(C, est);
    const job = C.job && jobs.find(j => j.id === C.job);
    const wants = loraWants();
    const ready = C.prompt.trim() && (isRef() ? C.refs.length + C.refVideos.length > 0 : noFrame() || C.image) && !wants.length;
    put(body, el("h2", {}, "Render"), el("div", { class: "note" }, output().label + (String(C.steps).trim() && !isRef() ? " · " + C.steps + " passes (custom)" : "") + " · " + (isRef() ? "MiniMax H3 Ref2VA, " + refPassText() + ", " + [[C.refs.length, "picture"], [C.refVideos.length, "video"], [C.refAudios.length, "audio clip"]].filter(k => k[0]).map(k => k[0] + " reference " + k[1] + (k[0] === 1 ? "" : "s")).join(", ") + (motionFields() ? ", motion: " + MOTION_MODES[C.motion.mode][0].toLowerCase() + " from the " + MOTION_LAYERS[C.motion.layer][0].toLowerCase() + (C.motion.layer === "reference" ? "" : " layer") : "") : S.qualities[C.quality]) + " · " + (isLong() ? S.long[C.seconds] : exact(framesOf(C))) + (lastOf() ? " · ends on the last frame" : "") + " · seed " + C.seed), est,
      wants.map(w => el("div", { class: "note lorawant", style: "color:var(--warn)" }, el("b", {}, w.name + " needs: "), w.text + " ", el("button", { type: "button", class: "small", onclick: w.go }, w.label))),
      !ready ? el("div", { class: "note", style: "color:var(--warn)" }, C.prompt.trim() ? (isRef() ? (S.ref_media ? "Add a reference picture or video in step 2." : "Add a reference picture in step 2.") : "Pick or upload a first frame in step 2.") : "Open step 4 once to build the prompt.") : null,
      negNote(),
      isLong() ? null : el("label", { style: "display:flex;gap:6px;align-items:center;margin-top:8px", title: "After the render the app looks at the clip (app/shot.py): does it open on its first frame and end on its last, and is there a cut inside it. The verdict is shown beside the clip either way." },
        bind(C, "retake", el("input", { type: "checkbox", style: "width:auto" })), "If the shot check fails (a keyframe not held, or a cut inside the shot), render it once more at the next seed. That is a second full render."),
      // how many Sparks this job may use, and what that means for it, before it is queued (sparks.js)
      el("div", { style: "margin-top:10px" }, sparksPick({ kind: isLong() ? "long" : "video", clips: 1, params: { loras: lorasOf(C), ...(isRef() ? { references: C.refs, reference_videos: C.refVideos, reference_audios: C.refAudios, ref_turbo: C.refPasses === "4" } : {}), ...(encoderOf() ? { text_encoder: encoderOf() } : {}) } })),
      el("div", { class: "row", style: "margin-top:10px" }, el("button", { class: "hot fit", disabled: !ready, onclick: async () => { try {
        const base = { prompt: C.prompt, image: noFrame() ? null : C.image, ...(lastOf() ? { last_image: lastOf() } : {}), preset: C.preset, quality: C.quality, seed: C.seed, ...extras(), ...(isRef() ? { references: C.refs, ref_turbo: C.refPasses === "4", ...(refMedia() ? { reference_videos: await motionVideos(), reference_audios: C.refAudios, reference_video_sound: C.video_sound, ...(motionFields() ? { motion: { pack: C.motion.pack, layer: C.motion.layer, mode: C.motion.mode, measured: C.motion.measured || null, free: motionFree() }, pace: C.motion.pace || "auto", silent: paceOf(C.motion) === "flow" && !musicNamed() } : {}) } : {}) } : {}) };
        const j = await api("/api/jobs", isLong() ? { kind: "long", title: S.long[C.seconds] + ": " + (C.action || C.scene || "Video").slice(0, 40), params: { ...base, total_seconds: Number(C.seconds), beats: (C.beats || "").split("\n") } }
          : { kind: "video", title: (C.action || C.scene || "Clip").slice(0, 48), params: { ...base, ...lengthOf(C), ...(C.retake ? { retake: true } : {}) } });
        C.job = j.id; await poll(); render(); } catch (e) { toast(e.message); } } }, "Queue this clip"),
        // the way back to step 1 once a clip is queued: with the same settings, or with the words cleared
        el("button", { class: "fit", onclick: () => { C.step = 0; C.job = null; render(); } }, "Back to the start, keep this setup"),
        el("button", { class: "fit", onclick: () => { Object.assign(C, { step: 0, job: null, prompt: "", scene: "", action: "", camera: "", dialogue: [], soundscape: "", music: "N/A", imagePrompt: "", image: null, lastImage: null, scoutJob: null, beats: "", refs: [], subjects: [], refVideos: [], video_subjects: [], refAudios: [], audio_notes: [], video_sound: false, motion: null }); MA.job = null; Object.assign(REF_SAID, { pictures: "", videos: "", audios: "" }); render(); } }, "Start a new clip"), el("span")));
    if (job) { put(body, el("h3", {}, "This clip"), jobCard(job));
      if (job.status === "done") put(body, el("video", { class: "main", src: file(job.outputs[0]), controls: true }),
        el("div", { class: "row", style: "margin-top:8px" }, el("button", { class: "fit", onclick: () => addToTimeline(job.outputs[0]) }, "Add to the editor timeline"), el("a", { class: "fit", href: file(job.outputs[0]), download: "" }, "Download"), el("span"))); }
  }
  return el("section", { class: "card" }, steps, body, next);
}
