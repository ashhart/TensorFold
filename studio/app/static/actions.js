"use strict";
// Library actions (app/actions.py): Turnaround sheet on a picture, Upscale and Remove a person on a clip
const ACT = { state: null, open: null, jobs: {},
  turn: { size: "1024", strength: "1", seed: 1, prompt: "" },
  up: { megapixels: "2", denoise: "0.1" },
  rm: { person: "", clean: "auto", clean_frame: null, window: "22", steps: 12, seed: 904234, silent: false, mask: null } };
// Credit: matlod / MATLOWAI (Turnaround), dntpi / Sandpies on LBH-123-AI (Upscaler), Akatz Labs (Remover), Meta SAM 3.1
const ACT_KEY = { turnaround: "turn", upscale: "up", remove: "rm" };
async function actLoad() { try { ACT.state = await api("/api/actions"); } catch { ACT.state = {}; } }
const actOf = kind => (ACT.state || {})[kind] || { ready: false, why: "" };
function actOpen(kind, path) { ACT.open = { kind, path }; if (kind === "remove") ACT.rm.mask = null; tab = "Library"; history.replaceState(null, "", "#Library"); render();
  setTimeout(() => { const n = document.querySelector("#actpanel"); if (n) n.scrollIntoView({ behavior: "smooth", block: "start" }); }, 60); }
// the buttons a Library tile gets for its kind of file
function actButtons(m) { if (!ACT.state) return [];
  const one = (kind, label, title) => el("button", { class: "small", title: actOf(kind).ready ? title : actOf(kind).why, onclick: () => actOpen(kind, m.path) }, label);
  return m.kind === "video" ? [one("upscale", "Upscale", "Make this clip larger and sharper: a latent upscale, then one light re-draw guided by the clip itself"),
                               one("remove", "Remove a person", "Take one person out of this clip and rebuild the background")]
    : [one("turnaround", "Turnaround", "Five views of this picture's subject, from the front round to the side or back")]; }
// a LoRA's card in the hub, when the LoRA belongs to an action: where it is used, in place of "Use on this clip"
function actHubNote(r) { const kind = r.action, a = actOf(kind);
  return el("div", { class: "note" }, "Used by the Library’s “" + (a.title || kind) + "” action: " + (kind === "turnaround" ? "press Turnaround under a picture." : "press " + (a.title || kind) + " under a clip.")
    + (ACT.state && !a.ready ? " " + a.why : "")); }
function actHubButton(r) { return el("button", { type: "button", class: "small", onclick: () => { HUB.browse = false; tab = "Library"; history.replaceState(null, "", "#Library"); render(); toast("Press “" + (r.action === "turnaround" ? "Turnaround” under a picture" : (actOf(r.action).title || r.action) + "” under a clip")); } }, "Open the Library"); }
async function actQueue(kind, params, slot) { try { const job = await api("/api/jobs", { kind, params }); ACT.jobs[slot || kind] = job.id; toast("Queued: " + job.title); poll(); render(); } catch (e) { toast(e.message, 6000); } }
const actJob = slot => jobs.find(j => j.id === ACT.jobs[slot]);
const actShow = (slot, made) => { const job = actJob(slot); return job ? el("div", { style: "margin-top:12px" }, jobCard(job), job.status === "done" ? made(job) : null) : null; };
const actUse = (path, how) => how === "first" ? () => { C.image = path; C.goes = "video"; C.start = "own"; C.step = 1; tab = "Create"; render(); }
  : () => { C.goes = "video"; C.start = "refs"; sync(); if (!isRef()) return toast("Reference pictures need MiniMax H3 Ref2VA on this machine"); if (!C.refs.includes(path)) { C.refs.push(path); C.subjects.push(""); } C.step = 1; tab = "Create"; render(); };
function actPanel() { const o = ACT.open; if (!o) return null; const a = actOf(o.kind), f = ACT[ACT_KEY[o.kind]], redraw = () => render();
  const box = el("div", { class: "seg", id: "actpanel", style: "margin-top:12px" },
    el("div", { class: "seghead" }, el("b", {}, a.title || o.kind), el("span", { class: "note", style: "margin:0" }, o.path.split("/").pop()), el("span", { style: "flex:1" }),
      el("button", { class: "small", onclick: () => { ACT.open = null; render(); } }, "Close")));
  if (!a.ready) { put(box, el("div", { class: "note", style: "color:var(--warn)" }, a.why || "Not available on this machine.")); return box; }
  const by = el("div", { class: "note" }, "With " + a.lora_name + " by " + a.author + ".");
  if (o.kind === "turnaround") {
    put(box, el("div", { class: "row", style: "align-items:flex-start" }, el("div", { style: "flex:0 0 220px" }, media(o.path, { style: "width:220px;border-radius:8px" })),
      el("div", {}, el("div", { class: "note", style: "margin-top:0" }, "Five views of the subject of this picture, turning from where it faces now, made together so they agree. Each is a picture of its own in the Library: use one as a first frame, or several as reference pictures of the same character."),
        field("Size of each view", choice(f, "size", { 512: ["512", "fastest, what the LoRA was trained at"], 1024: ["1024", "the sweet spot"], 2048: ["2048", "slow; the turn comes late"] }, redraw, true)),
        field("What matters more", choice(f, "strength", { 1: ["The turn", "strength 1.0"], 0.7: ["The look", "strength 0.7, turns less"] }, redraw, true)),
        el("div", { class: "row" }, field("Seed", bind(f, "seed", el("input", { type: "number", min: 0, style: "max-width:110px" }))),
          field("Instruction (optional)", bind(f, "prompt", el("input", { placeholder: "the camera orbits horizontally around the subject of <Picture 1> at eye level, ninety degrees clockwise, the subject stays upright on the ground" })))),
        by, el("button", { class: "hot", onclick: () => actQueue("turnaround", { image: o.path, size: Number(f.size), strength: Number(f.strength), seed: Number(f.seed) || 0, prompt: f.prompt }) }, "Make the turnaround sheet"))),
      actShow("turnaround", job => el("div", {}, el("img", { src: file(job.outputs[0]), style: "width:100%;border-radius:8px;margin-top:8px", alt: "the five views side by side" }),
        el("div", { class: "grid", style: "margin-top:8px" }, job.outputs.slice(1).map((p, i) => el("div", { class: "tile" }, media(p, { style: "aspect-ratio:1" }), el("div", { class: "cap", style: "justify-content:flex-end" }, el("span", { style: "flex:1" }, "view " + (i + 1)),
          el("button", { class: "small", title: "Start a clip on this picture", onclick: actUse(p, "first") }, "first frame"), el("button", { class: "small", title: "Add it to a reference clip’s pictures", onclick: actUse(p, "ref") }, "reference"))))))));
  } else if (o.kind === "upscale") {
    put(box, el("div", { class: "note", style: "margin-top:0" }, "Makes a finished clip larger: the model’s latent upscaler enlarges it, then one light re-draw in tiles puts detail back, guided by the clip itself so it restores what is there. Mouths, motion and the clip’s own sound stay."),
      field("Size", choice(f, "megapixels", { 1: ["1 megapixel", "about 1344 × 768"], 2: ["2 megapixels", "about 1920 × 1080"], 3: ["3 megapixels", "about 2304 × 1280"], 4: ["4 megapixels", "about 2688 × 1504, slowest"] }, redraw, true)),
      field("Re-draw", choice(f, "denoise", { 0: ["None", "only enlarge"], 0.1: ["Light", "0.1, the tested setting"], 0.2: ["Stronger", "0.2, adds more of its own"] }, redraw, true)),
      by, el("button", { class: "hot", onclick: () => actQueue("upscale", { video: o.path, megapixels: Number(f.megapixels), denoise: Number(f.denoise) }) }, "Upscale this clip"),
      actShow("upscale", job => el("video", { class: "main", src: file(job.outputs[0]), controls: true, style: "margin-top:8px" })));
  } else {
    const up = el("input", { type: "file", accept: ".png,.jpg,.jpeg,.webp", onchange: async () => { if (!up.files[0]) return; const data = new FormData(); data.append("file", up.files[0]);
      try { const r = await (await fetch("/api/upload", { method: "POST", body: data })).json(); if (!r.path) throw new Error(r.error || "upload failed"); f.clean_frame = r.path; render(); } catch (e) { toast(e.message); } } });
    const need = () => { if (!f.person.trim()) { toast("Say who to remove in a few words"); return false; } return true; };
    put(box, el("div", { class: "note", style: "margin-top:0" }, "Takes one person out of a clip and draws the background again. The person is found from a few words and followed through the clip; the removal then runs in overlapping windows of 22 frames. The whole picture is drawn again, so small details outside the person can change too. Best on a short continuous shot."),
      field("Who to remove", bind(f, "person", el("input", { placeholder: "man in a gray shirt · woman in a red coat · person on the left" }))),
      el("div", { class: "row" }, el("button", { class: "fit", onclick: () => { if (need()) actQueue("mask", { video: o.path, person: f.person }, "mask"); } }, "1. Find the person"),
        el("span", { class: "note", style: "margin:0" }, "A quick look before the long render: the person painted green. If hands, hair or a shadow are left out, or someone else is green, describe the person more exactly.")),
      actShow("mask", job => el("div", { class: "row", style: "align-items:flex-start;margin-top:8px" }, el("div", {}, el("label", {}, "The person, in green"), el("video", { src: file(job.outputs[0]), controls: true, loop: true, style: "width:100%;border-radius:8px" })),
        job.outputs[2] ? el("div", {}, el("label", {}, "A first frame without the person, built from the clip"), el("img", { src: file(job.outputs[2]), style: "width:100%;border-radius:8px" })) : null)),
      field("The clean first frame", choice(f, "clean", { auto: ["Build it for me", "from the clip’s other frames, then drawn sharp by the remover"], own: ["My own picture", "the first frame with the person edited out, same size"] }, redraw, true)),
      f.clean === "own" ? el("div", { class: "row" }, up, f.clean_frame ? el("img", { src: file(f.clean_frame), style: "max-width:220px;border-radius:8px" }) : el("span", { class: "note", style: "margin:0" }, "Edit the person out of the clip’s first frame in any picture editor and load it here.")) : null,
      el("div", { class: "row" }, field("Window", select(f, "window", { 22: "22 frames (proven)", 39: "39 frames", 56: "56 frames" })), field("Passes", bind(f, "steps", el("input", { type: "number", min: 4, max: 30, style: "max-width:90px" }))),
        field("Seed", bind(f, "seed", el("input", { type: "number", min: 0, style: "max-width:120px" }))),
        el("label", { style: "display:flex;gap:6px;align-items:center;flex:none", title: "The clip’s own sound is put back under the result. It still holds the removed person’s voice: tick this to leave the sound out." }, bind(f, "silent", el("input", { type: "checkbox", style: "width:auto" })), "No sound")),
      by, el("button", { class: "hot", onclick: () => { if (!need()) return; if (f.clean === "own" && !f.clean_frame) return toast("Load the clean first frame, or choose “Build it for me”");
        actQueue("remove", { video: o.path, person: f.person, window: Number(f.window), steps: Number(f.steps), seed: Number(f.seed), silent: !!f.silent, ...(f.clean === "own" ? { clean_frame: f.clean_frame } : {}) }); } }, "2. Remove the person"),
      actShow("remove", job => el("div", {}, (job.log || "").split("\n").filter(l => l.includes("] check: ")).map(l => el("div", { class: "note", style: l.includes("still show green") ? "color:var(--warn)" : "" }, l.split("] check: ")[1])),
        el("div", { class: "row", style: "align-items:flex-start;margin-top:8px" }, el("div", {}, el("label", {}, "Without the person"), el("video", { class: "main", src: file(job.outputs[0]), controls: true })),
        el("div", {}, el("label", {}, "The clip as it was"), el("video", { class: "main", src: file(o.path), controls: true }))))));
  }
  return box; }
