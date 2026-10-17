// Production, Editor and Library tabs.
"use strict";
async function ensureProject() { if (project) return project; const list = await api("/api/projects"); project = list.length ? await api("/api/projects/" + list[0].id) : await api("/api/projects", { name: "Project 1" }); return project; }
const saveProject = () => api("/api/projects/" + project.id, project, "PUT").catch(e => toast(e.message));
const segJob = seg => seg.job && jobs.find(j => j.id === seg.job);
async function composeSeg(seg, index) { const first = seg.start !== "text"; const r = await api("/api/compose", { fields: { scene: seg.scene, action: seg.action, camera: seg.camera, soundscape: seg.soundscape, music: "N/A",
  dialogue: seg.line ? [{ voice: seg.voice, text: seg.line, language: "English" }] : [] }, first_frame: first }); seg.prompt = r.prompt; }
async function generateSeg(index) { const seg = project.segments[index]; if (!seg.prompt) await composeSeg(seg, index);
  const params = { prompt: seg.prompt, preset: project.preset, quality: project.quality, ...lengthOf(seg), seed: Number(seg.seed || 1), ...(lorasOf(project).length ? { loras: lorasOf(project) } : {}), output: "projects/" + project.id + "/seg" + (index + 1) + "_" + Date.now() + ".mp4" };
  if (seg.start === "image" && seg.image) params.image = seg.image;
  if (seg.start === "previous") { const prev = project.segments[index - 1]; if (!prev) throw new Error("the first segment has nothing to continue from"); const pj = segJob(prev);
    params.first_frame_from = prev.output || (pj && pj.params.output); if (!params.first_frame_from) throw new Error("generate segment " + index + " first, or queue them together"); }
  const j = await api("/api/jobs", { kind: "video", title: "Segment " + (index + 1) + ": " + (seg.title || ""), params }); seg.job = j.id; seg.output = params.output; }
// two ways to plan a film: this board of segments, or the port of Sam Wasserman's Slate (slate.js)
const productionSwitch = () => el("div", { class: "choice", role: "radiogroup", style: "margin-bottom:16px" }, [["board", "Production board", "segments one after another: the simple way"], ["slate", "Wasserman Slate AI Studio", "bible, casting, coverage and a First AD, straight to the render queue"]].map(([k, title, sub]) =>
  el("button", { type: "button", role: "radio", "aria-checked": String(SL.mode === k), class: SL.mode === k ? "on" : "", onclick: () => { SL.mode = k; try { localStorage.setItem("production-mode", k); } catch {} render(); } }, el("b", {}, title), el("span", {}, sub))));
const SEG_TEXT = ["scene", "action", "voice", "line", "soundscape", "prompt"];
function clearSegments() { const was = project.segments.map(s => Object.fromEntries(SEG_TEXT.map(k => [k, s[k] || ""]))), of = project.id;
  if (!was.some(w => SEG_TEXT.some(k => k !== "prompt" && w[k]))) return toast("Nothing to clear here");
  for (const s of project.segments) for (const k of SEG_TEXT) s[k] = "";
  CLEARED = { key: "segments", model: () => { if (project && project.id === of) { project.segments.forEach((s, i) => was[i] && Object.assign(s, was[i])); saveProject(); } } };
  saveProject(); render(); undoToast("The text of every segment is empty."); }
function viewProduction() {
  if (SL.mode === "slate") return viewSlate(productionSwitch());
  const box = el("section", { class: "card" }, productionSwitch(), el("h2", {}, "Production"));
  ensureProject().then(async () => {
    const list = await api("/api/projects"); const total = project.segments.reduce((a, s) => a + Number(s.seconds || 0), 0);
    const pick = el("select", { onchange: async () => { project = await api("/api/projects/" + pick.value); render(); } }, list.map(p => el("option", { value: p.id, selected: p.id === project.id }, p.name + " · " + p.segments + " segments")));
    put(box, el("div", { class: "row" }, field("Project", pick), field("Name", bind(project, "name", el("input"), saveProject)),
        el("button", { class: "fit", onclick: async () => { project = await api("/api/projects", { name: "Project " + (list.length + 1) }); render(); } }, "New project")),
      el("div", { class: "row" }, field("Output for every segment", select(project, "preset", Object.fromEntries(Object.entries(S.presets).map(([k, v]) => [k, v.label])), saveProject)),
        field("Quality", select(project, "quality", S.qualities, saveProject)),
        S.sharp ? sharpButton(project.preset === "sharp" && project.quality === S.sharp.quality, () => { project.preset = "sharp"; project.quality = S.sharp.quality;
          for (const g of project.segments) { g.seed = S.sharp.seed; if (!S.sharp.seconds.includes(Number(g.seconds))) g.seconds = S.sharp.seconds[0]; } saveProject(); render(); }) : null),
      loraHub(project, saveProject),
      el("h3", {}, "Segment overview · " + total + " s total"),
      el("div", { class: "strip" }, project.segments.map((s, i) => el("div", { style: "flex:" + (s.seconds || 5) }, (i + 1) + " · " + (s.title || "") + " · " + Math.round(s.seconds * 100) / 100 + " s"))),
      // segments that do not open on the one before render side by side, one per Spark (sparks.js)
      sparksPick({ kind: "video", clips: Math.max(1, project.segments.filter(g => g.start !== "previous").length), params: { preset: project.preset, quality: project.quality, seconds: Number((project.segments[0] || {}).seconds || 5), loras: lorasOf(project) } }),
      project.segments.some(g => g.start === "previous") && SPK.n > 1 && SPK.data && SPK.data.supported ? el("div", { class: "note" }, "A segment that opens on the last frame of the one before waits for it; the others render side by side.") : null,
      el("div", { class: "row" }, el("button", { class: "fit", onclick: () => { project.segments.push({ title: "", scene: project.segments.at(-1)?.scene || "", action: "", camera: "", voice: "", line: "", soundscape: "", seconds: 5, seed: 1,
          start: project.segments.length ? "previous" : "text", image: null, prompt: "" }); saveProject(); render(); } }, "+ Add a segment"),
        el("button", { class: "hot fit", disabled: !project.segments.length, onclick: async () => { try { for (let i = 0; i < project.segments.length; i++) if (!project.segments[i].output || segJob(project.segments[i])?.status === "failed") await generateSeg(i);
          await saveProject(); await poll(); render(); } catch (e) { toast(e.message); } } }, "Generate every segment not yet rendered"),
        el("button", { class: "fit", disabled: !project.segments.some(s => s.output), onclick: () => { project.timeline = project.segments.filter(s => s.output).map(s => ({ path: s.output, in: 0, out: 0, duration: Number(s.seconds) }));
          saveProject(); tab = "Editor"; render(); } }, "Send rendered segments to the editor"),
        el("button", { class: "fit", disabled: !project.segments.length, title: "Empty the scene, action, voice, line and soundscape of every segment. Titles, lengths and rendered clips stay.", onclick: clearSegments }, "Clear every segment's text"),
        CLEARED && CLEARED.key === "segments" ? el("button", { class: "fit", onclick: undoClear }, "Undo clear") : null, el("span")));
    project.segments.forEach((seg, i) => { const job = segJob(seg); const change = () => { seg.prompt = ""; saveProject(); };
      put(box, el("div", { class: "seg", "data-clear": "segment-" + i }, el("div", { class: "seghead" }, el("b", {}, "Segment " + (i + 1)), job ? el("span", { class: "pill " + job.status }, job.status) : el("span", { class: "pill queued" }, seg.output ? "rendered" : "draft"),
          el("span", { style: "flex:1" }), clearButton("segment-" + i), el("button", { class: "small", onclick: async () => { try { await generateSeg(i); await saveProject(); await poll(); render(); } catch (e) { toast(e.message); } } }, seg.output ? "Render again" : "Render"),
          el("button", { class: "small", disabled: i === 0, onclick: () => { project.segments.splice(i - 1, 0, project.segments.splice(i, 1)[0]); saveProject(); render(); } }, "↑"),
          el("button", { class: "small", onclick: () => { project.segments.splice(i, 1); saveProject(); render(); } }, "Remove")),
        el("div", { class: "row" }, keep(field("Title", bind(seg, "title", el("input"), saveProject))),
          el("div", { class: "fit", style: "width:250px" }, el("label", {}, "Length"), lengthPicker({ get frames() { return seg.frames; }, set frames(v) { seg.frames = v; }, get seconds() { return seg.seconds; }, set seconds(v) { seg.seconds = v; }, preset: project.preset }, () => { saveProject(); render(); })),
          field("Opens on", select(seg, "start", { text: "nothing (text to video)", previous: "the last frame of the segment before", image: "an image from the library" }, () => { change(); render(); })),
          el("div", { class: "fit", style: "width:90px" }, el("label", {}, "Seed"), bind(seg, "seed", el("input", { type: "number" }), saveProject))),
        seg.start === "image" ? keep(field("Image path (copy from the Library tab)", bind(seg, "image", el("input", { placeholder: "scouts/…/scout_s3.png" }), saveProject))) : null,
        field("Describe the scene / condition where it takes place", bind(seg, "scene", el("textarea", { style: "min-height:54px" }), change)), field("What happens to the characters inside the scene / clip", bind(seg, "action", el("textarea", { style: "min-height:54px" }), change)),
        el("div", { class: "row" }, field("Voice", bind(seg, "voice", el("input", { placeholder: "a warm female voice" }), change)), field("Spoken line (optional)", bind(seg, "line", el("input"), change))),
        field("Soundscape", bind(seg, "soundscape", el("input"), change)),
        job && job.status === "running" ? jobCard(job) : null,
        seg.output && (!job || job.status === "done") ? el("video", { src: file(seg.output), controls: true, preload: "metadata", style: "width:100%;max-width:520px;border-radius:8px;margin-top:8px" }) : null)); });
  }).catch(e => put(box, el("div", { class: "note" }, e.message)));
  return box;
}

// ---- Editor ----
async function addToTimeline(path, from = 0, to = 0) { await ensureProject(); try { project = await api("/api/projects/" + project.id); } catch {} let duration = 5; try { duration = (await api("/api/probe?path=" + encodeURIComponent(path))).duration || 5; } catch {}
  project.timeline.push({ path, in: from, out: to, duration }); await saveProject(); toast("Added to the timeline of " + project.name + (to > from ? ", trimmed to " + from.toFixed(2) + " to " + to.toFixed(2) + " s" : "")); if (tab === "Editor") render(); }
function viewEditor() {
  const box = el("section", { class: "card" }, el("h2", {}, "Clip editor"));
  ensureProject().then(async () => {
    const lib = (await api("/api/library")).filter(m => m.kind === "video"); const tl = project.timeline; let at = 0;
    const player = el("video", { class: "main", controls: true }); const info = el("div", { class: "note" });
    const span = c => [Number(c.in) || 0, Number(c.out) > Number(c.in) ? Number(c.out) : c.duration];
    const load = (i, play) => { at = i; if (!tl[i]) return; const [a] = span(tl[i]); player.src = file(tl[i].path) + "#t=" + a; player.currentTime = a; if (play) player.play(); mark(); };
    const strip = el("div", { class: "tl" }); const mark = () => [...strip.children].forEach((n, i) => n.classList.toggle("on", i === at));
    player.addEventListener("timeupdate", () => { const c = tl[at]; if (!c) return; const [, b] = span(c); if (player.currentTime >= b - 0.04 && !player.paused) { if (at + 1 < tl.length) load(at + 1, true); else player.pause(); } });
    player.addEventListener("ended", () => { if (at + 1 < tl.length) load(at + 1, true); });
    const total = () => tl.reduce((s, c) => { const [a, b] = span(c); return s + Math.max(b - a, 0); }, 0);
    tl.forEach((c, i) => strip.append(el("div", { class: "clip", onclick: e => { if (e.target.tagName !== "INPUT" && e.target.tagName !== "BUTTON") load(i, false); } },
      el("video", { src: file(c.path) + "#t=0.5", preload: "metadata", muted: true }), el("div", {}, (i + 1) + " · " + c.path.split("/").pop().slice(0, 22)),
      el("div", { class: "row" }, field("In (s)", bind(c, "in", el("input", { type: "number", step: 0.1, min: 0 }), () => { saveProject(); info.textContent = "Timeline " + total().toFixed(1) + " s"; })),
        field("Out (s)", bind(c, "out", el("input", { type: "number", step: 0.1, min: 0, placeholder: c.duration.toFixed(1) }), () => { saveProject(); info.textContent = "Timeline " + total().toFixed(1) + " s"; }))),
      el("div", { class: "row", style: "margin-top:5px" }, el("button", { class: "small", disabled: i === 0, onclick: () => { tl.splice(i - 1, 0, tl.splice(i, 1)[0]); saveProject(); render(); } }, "←"),
        el("button", { class: "small", disabled: i === tl.length - 1, onclick: () => { tl.splice(i + 1, 0, tl.splice(i, 1)[0]); saveProject(); render(); } }, "→"),
        el("button", { class: "small", onclick: () => { tl.splice(i, 1); saveProject(); render(); } }, "✕")))));
    if (!tl.length) strip.append(el("div", { class: "note", style: "padding:14px" }, "The timeline is empty. Add clips from the media bin below, or send rendered segments from Production."));
    info.textContent = "Timeline " + total().toFixed(1) + " s · " + tl.length + " clips · project " + project.name + ". Out = 0 plays a clip to its end.";
    const size = el("select", {}, [["1920x1080", "1920×1080"], ["2560x1440", "2560×1440"], ["2048x1152", "2048×1152"], ["1344x768", "1344×768"]].map(([v, t]) => el("option", { value: v }, t)));
    const exp = project.exportJob && jobs.find(j => j.id === project.exportJob);
    put(box, el("h3", {}, "Preview"), player, el("div", { class: "row", style: "margin:8px 0" }, el("button", { class: "fit", disabled: !tl.length, onclick: () => load(0, true) }, "▶ Play the sequence"),
        el("div", { class: "fit", style: "width:150px" }, size), el("button", { class: "primary fit", disabled: !tl.length, onclick: async () => { try { const [w, h] = size.value.split("x").map(Number);
          const j = await api("/api/jobs", { kind: "export", title: "Export " + project.name, params: { clips: tl, width: w, height: h } }); project.exportJob = j.id; await saveProject(); await poll(); render(); } catch (e) { toast(e.message); } } }, "Export"), el("span")),
      info, el("h3", {}, "Timeline"), strip,
      exp ? el("div", { style: "margin-top:10px" }, jobCard(exp), exp.status === "done" ? el("div", {}, el("video", { class: "main", src: file(exp.outputs[0]), controls: true }), el("a", { href: file(exp.outputs[0]), download: "" }, "Download the export")) : null) : null,
      el("h3", {}, "Media bin"), el("div", { class: "grid" }, lib.slice(0, 60).map(m => el("div", { class: "tile" }, media(m.path, { controls: false, muted: true, onmouseenter: e => e.target.play().catch(() => {}), onmouseleave: e => e.target.pause() }),
        el("div", { class: "cap" }, m.path.split("/").pop().slice(0, 20), el("button", { class: "small", onclick: () => addToTimeline(m.path) }, "+ timeline"))))));
    if (tl.length) load(0, false);
  }).catch(e => put(box, el("div", { class: "note" }, e.message)));
  return box;
}

// ---- Library ----
function viewLibrary() { const box = el("section", { class: "card" }, el("h2", {}, "Library"));
  Promise.all([api("/api/library"), api("/api/trash").catch(() => null)]).then(([list, bin]) => { for (const p of [...LIBPICK]) if (!list.some(m => m.path === p)) LIBPICK.delete(p);
    const many = el("button", { class: "fit", onclick: () => clearOut({ paths: [...LIBPICK], films: true }, "Delete the " + count(LIBPICK.size, "selected file") + "?") });
    const all = el("button", { class: "fit", onclick: () => { const every = LIBPICK.size === list.length; LIBPICK.clear(); if (!every) for (const m of list) LIBPICK.add(m.path); render(); } }, LIBPICK.size && LIBPICK.size === list.length ? "Select none" : "Select all");
    const show = () => { many.textContent = "Delete selected (" + LIBPICK.size + ")"; many.disabled = !LIBPICK.size; }; show();
    if (!ACT.state) actLoad().then(() => { if (tab === "Library") render(); });
    put(box, el("div", { class: "note" }, list.length + " files. Click a path to copy it. Tick files to delete several at once; 🗑 deletes one. A deleted file is moved to the trash folder, not erased. Under a clip: Upscale and Remove a person. Under a picture: Turnaround."), actPanel(),
      el("div", { class: "row", style: "margin-top:10px" }, all, many, el("span"),
        bin ? el("span", { class: "note fit", id: "trashline" }, "Trash: " + (bin.files ? count(bin.files, "file") + " · " + sizeText(bin.bytes) : "empty")) : null,
        bin && bin.last ? el("button", { class: "fit", title: "Puts the files of the last deletion back where they were, and its job cards back in the queue", onclick: restoreLast }, "Restore last deletion (" + count(bin.last.files, "file") + ")") : null,
        bin && bin.files ? el("button", { class: "fit", title: "Erases the files in the trash folder for good", onclick: async () => { if (!confirm("Erase the " + count(bin.files, "file") + " (" + sizeText(bin.bytes) + ") in the trash folder for good?\n" + bin.folder + "\nThis cannot be undone.")) return;
          try { const r = await api("/api/trash?confirm=erase", undefined, "DELETE"); toast("Erased " + count(r.files, "file") + ", " + sizeText(r.bytes) + " freed"); } catch (e) { toast(e.message); } render(); } }, "Empty trash") : null),
      el("div", { class: "grid", style: "margin-top:10px" }, list.map(m =>
    el("div", { class: "tile" }, media(m.path), el("div", { class: "cap" }, el("input", { type: "checkbox", title: "Select " + m.path, "aria-label": "Select " + m.path, checked: LIBPICK.has(m.path), style: "width:auto;flex:none", onchange: e => { if (e.target.checked) LIBPICK.add(m.path); else LIBPICK.delete(m.path); show(); } }),
      el("span", { style: "cursor:pointer;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap", title: m.path + (m.spark ? " · rendered on " + m.spark + (m.seed != null ? ", seed " + m.seed : "") : ""), onclick: () => { navigator.clipboard?.writeText(m.path); toast("Copied " + m.path); } }, m.path.split("/").pop()),
      m.spark ? el("span", { class: "note fit", style: "margin:0", title: "The Spark that rendered this clip" + (m.seed != null ? ", and its seed" : "") }, m.spark + (m.seed != null ? " · s" + m.seed : "")) : null),
      // the buttons on a row of their own: a long file name no longer pushes them out of the tile
      el("div", { class: "cap", style: "padding-top:0;justify-content:flex-end;flex-wrap:wrap" }, actButtons(m),
      m.kind === "video" ? el("button", { class: "small", title: "Start a new clip on this clip's last frame", onclick: async () => { try { const r = await api("/api/frame", { path: m.path, at: "last" }); C.image = r.path; C.prompt = ""; C.goes = "video"; C.start = "own"; C.step = 1; tab = "Create"; render(); } catch (e) { toast(e.message); } } }, "continue") : el("button", { class: "small", title: "End a new clip on this picture", onclick: () => { C.lastImage = m.path; C.prompt = ""; C.goes = "video"; if (C.start === "refs") C.start = "text"; C.step = 1; tab = "Create"; render(); } }, "last frame"),
      m.kind === "video" ? el("button", { class: "small", onclick: () => addToTimeline(m.path) }, "+ timeline") : el("button", { class: "small", onclick: () => { C.image = m.path; C.goes = "video"; C.start = "own"; C.step = 1; tab = "Create"; render(); } }, "use"),
      m.kind === "video" ? null : el("button", { class: "small", title: "Re-draw this picture: say in words what changes", onclick: () => edOpen(m.path) }, "edit"),
      m.kind === "video" || !LORAS.some(r => r.target === "image" && r.installed && r.id === "qwen-multiple-angles") ? null : el("button", { class: "small", title: "Show this picture's subject from another camera angle (Multiple Angles LoRA)", onclick: () => edOpen(m.path, "angles") }, "new angle"),
      // out of the library: the server moves the file to the studio's trash folder
      el("button", { class: "small", title: "Delete: moves the file to the trash folder", "aria-label": "Delete " + m.path, onclick: () => clearOut({ paths: [m.path], films: true }, "Delete " + m.path.split("/").pop() + "?") }, "🗑")))))); });
  return box; }
