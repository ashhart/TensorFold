"use strict";
// Edit a picture (app/picture_edit.py): Qwen-Image-2.1 or FLUX3-K re-draws it; image LoRAs and angles are Qwen's only
const ED = { state: null, image: null, more: [], prompt: "", seed: 1, size: "source", base: false, loras: [], way: "words", picking: false, model: "qwen",
  angles: [{ direction: "front-right-quarter", height: "eye", close: false }], job: null, lib: null };
const edImageLoras = () => LORAS.filter(r => r.target === "image");
const edAngleLora = () => ED.state && LORAS.find(r => r.id === ED.state.angle_lora);
const edHas = id => ED.loras.some(p => p.id === id);
// the model an edit goes to: one the server offers (/api/edit/state models), Qwen when it is there
const edModels = () => (ED.state && ED.state.models) || [];
const edModel = () => (edModels().some(m => m.id === ED.model) ? ED.model : (edModels()[0] || { id: "qwen" }).id);
const edQwen = () => edModel() === "qwen";
const edSame = (a, b) => a.direction === b.direction && a.height === b.height && !!a.close === !!b.close;
async function edLoad() { try { ED.state = await api("/api/edit/state"); } catch (e) { ED.state = { ready: false, why: e.message, directions: [], heights: [], sets: {}, limit: 12 }; } if (tab === "Create" && C.goes === "edit") render(); }
// from the hub, or a Library tile: open the step on this LoRA or this picture
function edUse(r) { if (!edHas(r.id)) ED.loras.push({ id: r.id, strength: r.strength || 1 }); if (ED.state && r.id === ED.state.angle_lora) ED.way = "angles";
  C.goes = "edit"; C.step = 0; tab = "Create"; history.replaceState(null, "", "#Create"); render(); }
function edOpen(path, way) { ED.image = path; ED.more = ED.more.filter(p => p !== path); ED.way = way || "words"; C.goes = "edit"; C.step = 0; tab = "Create"; history.replaceState(null, "", "#Create"); render(); }
function edWants() { const out = [];
  if (!ED.image) out.push("the picture to edit");
  if (ED.way === "angles" && edQwen()) { const r = edAngleLora();
    if (!r) out.push("the Multiple Angles LoRA, which this copy of the hub does not list");
    else if (!r.installed) out.push("the Multiple Angles LoRA installed (LoRA hub, Fun)");
    if (!ED.angles.length) out.push("at least one camera angle"); }
  return out; }  // the words are checked when the button is pressed: typing does not redraw the form
// the picture: an upload, one from the Library, or a frame of a Library clip (the last-frame panel's own picker)
function edPicture() { const set = p => { ED.image = p; ED.picking = false; render(); };
  const up = el("input", { type: "file", accept: "image/*", onchange: async () => { if (!up.files[0]) return; const data = new FormData(); data.append("file", up.files[0]);
    try { const r = await fetch("/api/upload", { method: "POST", body: data }); const j = await r.json(); if (j.path) set(j.path); else toast(j.error || "upload failed"); } catch (e) { toast(e.message); } } });
  const box = el("div", { class: "group", id: "editpicture" }, el("h3", {}, "1 · The picture"),
    ED.image ? el("div", { class: "row", style: "align-items:flex-start" }, el("div", { class: "fit", style: "width:260px" }, el("img", { src: file(ED.image), alt: "The picture to edit", style: "width:100%;border-radius:10px;display:block" })),
      el("div", {}, el("div", { class: "note" }, ED.image), el("div", { class: "row", style: "margin-top:8px" }, el("button", { class: "fit small", onclick: () => { ED.image = null; render(); } }, "Take it out"), el("span"))))
      : el("div", { class: "note" }, "Give the picture the model re-draws. The result has this picture's own size unless you choose another under Size."),
    el("div", { class: "row", style: "margin-top:8px" }, el("div", { class: "fit" }, el("label", {}, "Upload a picture"), up),
      el("button", { class: "fit", onclick: () => { ED.picking = !ED.picking; if (ED.picking) api("/api/library").then(list => { ED.lib = list; render(); }).catch(() => {}); render(); } }, ED.picking ? "Close the Library" : "Pick from the Library"), el("span")));
  if (ED.picking && ED.lib) { const stills = ED.lib.filter(m => m.kind === "image").slice(0, 24), clips = ED.lib.filter(m => m.kind === "video").slice(0, 60);
    put(box, stills.length ? el("div", { class: "grid", style: "margin-top:10px" }, stills.map(m => el("div", { class: "tile" + (ED.image === m.path ? " on" : ""), style: "cursor:pointer", title: m.path, onclick: () => set(m.path) }, media(m.path))))
      : el("div", { class: "note" }, "No pictures in the Library yet."));
    if (clips.length) { const pick = el("select", {}, clips.map(m => el("option", { value: m.path }, m.path.split("/").pop()))), at = el("input", { type: "number", step: 0.1, min: 0, placeholder: "seconds", style: "width:110px" });
      const grab = async when => { try { const r = await api("/api/frame", { path: pick.value, at: when }); set(r.path); } catch (e) { toast(e.message); } };
      put(box, el("div", { class: "row", style: "margin-top:10px;align-items:flex-end" }, el("div", {}, el("label", {}, "From a clip in the Library"), pick),
        el("button", { class: "fit", onclick: () => grab("first") }, "Use its first frame"), el("button", { class: "fit", onclick: () => grab("last") }, "Use its last frame"),
        el("div", { class: "fit" }, el("label", {}, "or the frame at"), at), el("button", { class: "fit", onclick: () => at.value === "" ? toast("Type the seconds first") : grab(Number(at.value)) }, "Use frame at time"))); } }
  if (ED.image && ED.way === "words") put(box, edMore());
  return box; }
// Further pictures (ten in all with the edited one): the prompt calls them image 2, image 3, ... in this order
function edMore() { const most = ((ED.state && ED.state.pictures) || 10) - 1, add = p => { if (ED.more.length >= most) return toast("At most " + most + " further pictures"); if (p !== ED.image && !ED.more.includes(p)) ED.more.push(p); render(); };
  const up = el("input", { type: "file", accept: "image/*", "aria-label": "Upload a further picture", onchange: async () => { if (!up.files[0]) return; const data = new FormData(); data.append("file", up.files[0]);
    try { const r = await fetch("/api/upload", { method: "POST", body: data }); const j = await r.json(); if (j.path) add(j.path); else toast(j.error || "upload failed"); } catch (e) { toast(e.message); } } });
  const lib = (ED.lib || []).filter(m => m.kind === "image" && m.path !== ED.image && !ED.more.includes(m.path)).slice(0, 24);
  const pick = lib.length ? el("select", { "aria-label": "A further picture from the Library" }, [el("option", { value: "" }, "from the Library…"), ...lib.map(m => el("option", { value: m.path }, m.path.split("/").pop()))]) : null;
  if (pick) pick.onchange = () => { if (pick.value) add(pick.value); }; else if (!ED.lib) api("/api/library").then(list => { ED.lib = list; render(); }).catch(() => {});
  return el("div", { id: "editmore", style: "margin-top:12px" }, el("label", {}, "Further pictures the prompt can draw on (optional, up to " + most + ")"),
    el("div", { class: "note" }, "The model sees all the pictures at once, so a person from one and a place from another need no side-by-side montage. The picture above is image 1 and is the one that is edited; these are image 2, image 3, … in this order. Name them that way: “the man of image 2 stands in the room of image 1”. Every further picture makes the edit slower."),
    ED.more.length ? el("div", { class: "grid", style: "margin-top:8px" }, ED.more.map((p, i) => el("div", { class: "tile" }, media(p), el("div", { class: "cap" }, el("span", {}, "image " + (i + 2)), el("button", { class: "small", "aria-label": "Take image " + (i + 2) + " out", onclick: () => { ED.more.splice(i, 1); render(); } }, "Take it out"))))) : null,
    el("div", { class: "row", style: "margin-top:8px" }, el("div", { class: "fit" }, el("label", {}, "Upload"), up), pick ? el("div", { class: "fit" }, el("label", {}, "Or pick"), pick) : null, el("span"))); }
// the camera: 12 directions round the subject, 4 heights, each also as a close-up
function edCamera() { const st = ED.state, on = a => ED.angles.some(b => edSame(a, b)), limit = st.limit || 12;
  const flip = a => { if (on(a)) ED.angles = ED.angles.filter(b => !edSame(a, b)); else if (ED.angles.length >= limit) return toast("At most " + limit + " pictures from one job"); else ED.angles.push(a); render(); };
  const name = a => (st.directions.find(d => d.id === a.direction) || {}).label + ", " + (st.heights.find(h => h.id === a.height) || {}).label + (a.close ? ", close-up" : "");
  if (!ED.height) ED.height = "eye";
  const ring = el("div", { class: "edring", role: "group", "aria-label": "Camera direction" }, st.directions.map(d => { const a = { direction: d.id, height: ED.height, close: !!ED.close }, rad = (d.degrees - 90) * Math.PI / 180;
    return el("button", { type: "button", class: "small" + (on(a) ? " on" : ""), "aria-pressed": on(a) ? "true" : "false", title: d.label + ": the camera " + d.degrees + "° round the subject from the front",
      style: "left:" + (50 + 41 * Math.cos(rad)).toFixed(1) + "%;top:" + (50 + 41 * Math.sin(rad)).toFixed(1) + "%", onclick: () => flip(a) }, d.label); }),
    el("div", { class: "edmid" }, "subject", el("span", {}, "front is at the top")));
  return el("div", {}, el("div", { class: "row", style: "align-items:flex-start" },
      el("div", { class: "fit" }, ring),
      el("div", {}, el("label", {}, "Camera height for the next pick"), choice(ED, "height", Object.fromEntries(st.heights.map(h => [h.id, [h.label, h.degrees + "° up"]])), render, true),
        el("label", { style: "display:flex;gap:6px;align-items:center;margin-top:8px" }, el("input", { type: "checkbox", checked: !!ED.close, style: "width:auto", onchange: e => { ED.close = e.target.checked; render(); } }), "Close-up (the subject fills more of the frame)"),
        el("label", { style: "margin-top:10px" }, "Or a ready set"),
        el("div", { class: "row" }, Object.entries(st.sets).map(([key, s]) => el("button", { type: "button", class: "fit small", title: s.angles.length + " pictures", onclick: () => { ED.angles = s.angles.map(a => ({ ...a })); render(); } }, s.label)), el("span")))),
    el("label", { style: "margin-top:10px" }, ED.angles.length ? ED.angles.length + (ED.angles.length === 1 ? " picture" : " pictures") + " to make" : "Nothing picked yet: click a direction"),
    el("div", { class: "row", id: "editangles" }, ED.angles.map(a => el("button", { type: "button", class: "fit small lora on", title: "Take this angle out", onclick: () => flip(a) }, name(a) + " ✕")),
      ED.angles.length > 1 ? el("button", { type: "button", class: "fit small", onclick: () => { ED.angles = []; render(); } }, "Clear the list") : null, el("span"))); }
function edLoras() { const list = edImageLoras(); if (!list.length || !edQwen()) return null;
  return el("div", { class: "group", id: "editloras" }, el("h3", {}, "Image LoRAs"),
    el("div", { class: "note" }, "LoRAs for the image model. They apply to this step only; a clip's LoRAs are chosen in the clip's own step."),
    list.map(r => { const p = ED.loras.find(x => x.id === r.id), range = r.range || [0.1, 2], st = r.installing, busy = st && st.stage !== "failed";
      return el("div", { class: "row lorarow", "data-lora": r.id }, el("div", { class: "fit", style: "min-width:230px" }, el("b", {}, r.name), el("div", { class: "note" }, (r.author ? "by " + r.author : "") + (r.trigger ? " · trigger " + r.trigger + ", added to the prompt for you" : ""))),
        busy ? el("button", { type: "button", class: "fit small", disabled: true }, st.stage + (st.of ? " " + Math.round(100 * st.done / st.of) + "%" : "…"))
        : !r.installed ? el("button", { type: "button", class: "fit small", onclick: async () => { if (!confirm("Download “" + r.name + "”" + (r.author ? " by " + r.author : "") + " to this machine?\n\n" + (r.licence || "") + (r.page ? "\n\n" + r.page : ""))) return;
            try { await api("/api/loras/" + r.id + "/install", {}); await loadLoras(); hubWatch(render); render(); } catch (e) { toast(e.message); } } }, "⤓ Install")
        : el("button", { type: "button", class: "fit small lora" + (p ? " on" : ""), "aria-pressed": p ? "true" : "false", onclick: () => { if (p) { ED.loras = ED.loras.filter(x => x.id !== r.id); if (ED.state && r.id === ED.state.angle_lora) ED.way = "words"; } else ED.loras.push({ id: r.id, strength: r.strength || 1 }); render(); } }, p ? "✓ On this picture" : "Use on this picture"),
        p ? el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Strength"), bind(p, "strength", el("input", { type: "number", step: 0.05, min: 0.1, max: 2, title: "Made for " + (r.strength || 1) + "; the author's range is " + range[0] + " to " + range[1] + ". Lower it if the result copies your picture instead of moving the camera." }))) : null,
        el("span")); })); }
function viewEdit(goes) { const body = el("div"), st = ED.state;
  if (!st) { edLoad(); put(body, el("h2", {}, "What are you making?"), goes, el("div", { class: "note" }, "Looking at what is installed…")); return el("section", { class: "card" }, body); }
  const made = ED.job && jobs.find(j => j.id === ED.job), angle = edAngleLora(), wants = edWants();
  put(body, el("h2", {}, "What are you making?"), goes);
  if (!st.ready) { put(body, el("div", { class: "note", style: "color:var(--warn)" }, el("b", {}, "Edit a picture is not set up on this machine. "), st.why)); return el("section", { class: "card" }, body); }
  const ways = { words: ["Change it in words", "Say what is different"] }, qwen = edQwen();
  if (!qwen) ED.way = "words";  // the camera is Qwen's Multiple Angles LoRA
  ED.model = edModel();
  const models = edModels().length > 1 ? el("div", { id: "editmodel" }, el("label", {}, "Picture model"),
    choice(ED, "model", Object.fromEntries(edModels().map(m => [m.id, [m.label, m.note]])), render, true)) : null;
  if (angle && qwen) ways.angles = ["Move the camera", "Multiple Angles LoRA" + (angle.installed ? "" : ": not installed yet")];
  if (ED.way === "angles" && angle && angle.installed && !edHas(angle.id)) ED.loras.push({ id: angle.id, strength: angle.strength || 1 });
  put(body, models, edPicture(),
    el("div", { class: "group", id: "editwhat" }, el("h3", {}, "2 · What changes"), Object.keys(ways).length > 1 ? choice(ED, "way", ways, render, true) : null,
      ED.way === "angles" && angle ? (angle.installed ? edCamera() : el("div", { class: "note", style: "color:var(--warn)" }, el("b", {}, "Needs: "), "the Multiple Angles LoRA installed. Press Install under Image LoRAs below."))
        : field("Say what is different in the new picture", bind(ED, "prompt", el("textarea", { placeholder: "The same statue at night under a full moon, wet paving stones, a lantern on the wall behind it." }))),
      ED.way === "words" ? el("div", { class: "note" }, "Describe the result, not the steps. What you do not mention is meant to stay; small things still shift, so compare with the picture you gave.") : null),
    edLoras(),
    el("div", { class: "row" },
      el("div", { class: "fit", style: "width:300px" }, el("label", {}, "Size of the result"), bind(ED, "size", el("select", {}, [["source", "Same as the picture"], ["768", "About 768 × 768 pixels (faster)"], ["1024", "About 1024 × 1024 pixels"], ["1280", "About 1280 × 1280 pixels (slower)"]].map(([v, t]) => el("option", { value: v }, t))))),
      el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Seed"), bind(ED, "seed", el("input", { type: "number", min: 0 }))),
      st.base && qwen ? el("div", { class: "fit", style: "min-width:340px" }, el("label", {}, "Steps"), choice(ED, "base", { false: ["8 steps", "Turbo"], true: ["40 steps", "base model, slower"] }, () => { ED.base = ED.base === "true" || ED.base === true; render(); }, true)) : null, el("span")),
    el("div", { class: "note" }, (qwen ? st.note : (edModels().find(m => m.id === edModel()) || {}).label + ", " + (edModels().find(m => m.id === edModel()) || {}).note + "; it sees all the pictures and writes the caption of the result itself") + ". “Same as the picture” returns the picture's own width and height (a 1280 × 736 picture comes back 1280 × 736); the other sizes keep its shape at about that many pixels. Results go to the Library, where a picture can open a clip, end one, or be a reference."),
    wants.length ? el("div", { class: "note lorawant", style: "color:var(--warn)" }, el("b", {}, "Needs: "), wants.join("; ") + ".") : null,
    el("div", { class: "row", style: "margin-top:10px" }, el("button", { class: "hot fit", disabled: wants.length > 0, onclick: async () => { try { if (ED.way !== "angles" && !ED.prompt.trim()) throw new Error("Say what is different in the new picture first");
        const params = { image: ED.image, seed: Number(ED.seed) || 0, ...(ED.size === "source" ? { size: "source" } : { size: "model", resolution: Number(ED.size) || 1024 }), ...(ED.way === "words" && ED.more.length ? { images: ED.more } : {}), image_model: edModel(), ...(qwen ? { base: ED.base === true || ED.base === "true", loras: ED.loras.filter(p => { const r = LORAS.find(x => x.id === p.id); return r && r.installed; }) } : {}) };
        if (ED.way === "angles") params.angles = ED.angles; else params.prompt = ED.prompt;
        const j = await api("/api/jobs", { kind: "edit", title: ED.way === "angles" ? "Angles: " + ED.angles.length + (ED.angles.length === 1 ? " view" : " views") : "Edit: " + ED.prompt.slice(0, 40), params });
        ED.job = j.id; await poll(); render(); } catch (e) { toast(e.message); } } }, ED.way === "angles" ? "Make " + (ED.angles.length === 1 ? "this view" : "these " + ED.angles.length + " views") : "Edit the picture"), el("span")));
  if (made) { put(body, el("h3", {}, "Results"), jobCard(made));
    if (made.status === "done") put(body, el("div", { class: "grid", style: "margin-top:10px" }, made.outputs.map(p => el("div", { class: "tile" }, media(p),
      el("div", { class: "cap" }, el("span", { title: p, style: "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" }, p.split("/").pop().replace(/^edit_?|\.png$/g, "").replace(/_/g, " ") || "edit")),
      el("div", { class: "cap", style: "padding-top:0;justify-content:flex-end" }, el("a", { href: file(p), download: "" }, "Download"),
        el("a", { href: "#", title: "Open a new clip on this picture", onclick: e => { e.preventDefault(); C.image = p; C.goes = "video"; C.start = "own"; C.step = 1; changed(); } }, "first frame"),
        el("a", { href: "#", title: "End a clip on this picture", onclick: e => { e.preventDefault(); C.lastImage = p; C.goes = "video"; if (C.start === "refs") C.start = "text"; C.step = 1; changed(); } }, "last frame"),
        el("a", { href: "#", title: "Edit this result again", onclick: e => { e.preventDefault(); ED.image = p; render(); } }, "edit again")))))); }
  return el("section", { class: "card" }, body); }
