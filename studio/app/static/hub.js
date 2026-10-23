// Keys and Wasserman LoRA Hub: the LoRA library a clip or picture picks from.
"use strict";
// another text encoder file found on this machine, picked by its file name ("" is the model's own)
const encoderOf = () => C.textEncoder === "__own" ? String(C.textEncoderPath || "").trim() : (S && (S.text_encoders || []).includes(C.textEncoder)) ? C.textEncoder : "";
const encoderName = f => f.split("/").pop().replace(/\.safetensors$/, "").replace(/^qwen3vl_32b_minimax_h3_/, "").replace(/[_-]+/g, " ");
const lorasOf = obj => (obj.loras || []).filter(p => LORAS.some(r => r.id === p.id && r.installed));
// what a one-job LoRA on the reference model still lacks of the clip ("" when nothing); lora_flows.py checks the same
function flowLacks(r) { const f = r.flow, vids = C.refVideos.length, pics = C.refs.length;
  return !isRef() ? "Choose “From references” for this clip." : vids < f.videos[0] ? "Add the video under References." : pics < f.pictures[0] ? "Add the picture under References." : vids > f.videos[1] || pics > f.pictures[1] ? "This LoRA takes " + (f.videos[1] ? "one video" : "no video") + " and " + (f.pictures[1] ? "up to " + f.pictures[1] + " picture" + (f.pictures[1] > 1 ? "s" : "") : "no picture") + "." : !f.turbo && C.refPasses === "4" ? "Choose 20 passes: this LoRA is for the full model, not the Turbo adapter." : ""; }
function flowGo(r) { Object.assign(C, { goes: "video", start: "refs", step: 1 }); if (!r.flow.turbo) C.refPasses = "20"; changed(); }
// every input the clip's LoRAs still wait for: [{ name, text, label, go }]; Review holds Queue while there is one
function loraWants() { const out = [];
  for (const p of lorasOf(C)) { const r = LORAS.find(x => x.id === p.id); if (!r || !r.installed) continue;
    if (r.flow && r.flow.asks.length) { const lacks = flowLacks(r); if (lacks) out.push({ name: r.name, text: lacks, label: "Go to References", go: () => flowGo(r) }); }
    else if (r.needs === "first-last") {
      if (isRef()) out.push({ name: r.name, text: "It works on a clip that starts from a first frame, not from references.", label: "Start from a first frame", go: () => { Object.assign(C, { goes: "video", start: "own", step: 1 }); changed(); } });
      else if (!lastOf()) out.push({ name: r.name, text: "Add a last frame" + (r.same_first_last ? " (for an orbit, “Same as the first frame”)" : "") + ".", label: "Go to the last frame", go: () => { C.step = 1; changed(); } }); }
    else if (r.status === "unsupported" || r.needs === "workflow" || r.needs === "image") out.push(r.needs === "image" ? { name: r.name, text: "It is a LoRA for pictures, not for a clip. It works in Edit a picture.", label: "Open Edit a picture", go: () => { C.loras = C.loras.filter(x => x.id !== r.id); edUse(r); } }
      : { name: r.name, text: "It is not a LoRA for a clip made here: it runs through its own action. Take it out of this clip.", label: "Take it out", go: () => { C.loras = C.loras.filter(x => x.id !== r.id); changed(); } }); }
  return out; }
function flowRow(obj, r, done) { if (!C.lineart) C.lineart = "medium"; const f = r.flow, own = obj === C, vids = own ? C.refVideos.length : 0, pics = own ? C.refs.length : 0;
  const lacks = !own ? "This LoRA works in Create, on a clip from references." : flowLacks(r);
  const go = () => flowGo(r);
  const draw = async e => { e.target.disabled = true; e.target.textContent = "Drawing the line art…";
    try { const made = await api("/api/lineart", { path: C.refVideos[0], detail: C.lineart || "medium" }); C.lineartFrom = made.from; C.refVideos[0] = made.path; C.video_subjects[0] = ""; C.video_sound = false; toast("Video 1 is now the line art of your clip"); } catch (err) { toast(err.message); }
    done(); };
  return el("div", { class: "note", style: "flex-basis:100%;" + (lacks ? "color:var(--warn)" : "") }, el("b", {}, lacks ? "Needs: " : "Ready: "), f.asks.join("; ") + ". " + (lacks || "") + " " + f.hint + " ",
    own && lacks && (!isRef() || vids < f.videos[0] || pics < f.pictures[0]) ? el("button", { type: "button", class: "small", onclick: go }, "Go to References") : null,
    own && f.lineart && isRef() && vids === 1 && !/_lineart\.mp4$/.test(C.refVideos[0]) ? el("span", {}, " ", select(C, "lineart", { fine: "Fine lines", medium: "Medium", bold: "Outlines only" }, () => {}), " ", el("button", { type: "button", class: "small lora", onclick: draw }, "Make line art of Video 1")) : null,
    own && f.lineart && C.lineartFrom && /_lineart\.mp4$/.test(C.refVideos[0] || "") ? el("span", {}, " Video 1 is line art drawn from your clip.") : null); }
async function loadLoras() { try { LORAS = await api("/api/loras"); if (!HUB.groups.length) HUB.groups = await api("/api/loras/groups"); } catch { LORAS = []; } }
// what a LoRA works from, as the card's first tag
const LORA_NEEDS = { text: "Text or first frame", "first-last": "First + last frame", reference: "Reference pictures", "source-video": "Source video", workflow: "Its own workflow", image: "A picture to edit", action: "A Library action" };
const HUB = { open: false, browse: false, tab: "all", q: "", mine: false, groups: [], form: { name: "", url: "", file: "", trigger: "", strength: 1, styles: "", author: "", notes: "" }, timer: null };
function hubWatch(redraw) { if (HUB.timer || !LORAS.some(r => r.installing && r.installing.stage !== "failed")) return;
  HUB.timer = setInterval(async () => { await loadLoras(); if (!LORAS.some(r => r.installing && r.installing.stage !== "failed")) { clearInterval(HUB.timer); HUB.timer = null; } redraw(); }, 1500); }
// a slider LoRA: a track centred on 0 with its ends named, and the number beside it
function loraSlider(p, r, range, save) { const ends = r.ends || ["the other way", "more of it"], show = el("input", { type: "number", step: 0.1, min: range[0], max: range[1], style: "width:78px;flex:none", "aria-label": r.name + " strength" });
  const track = el("input", { type: "range", class: "loraslide", min: range[0], max: range[1], step: 0.1, list: "zero-" + r.id, "aria-label": r.name + " strength, " + range[0] + " to " + range[1] + ", 0 is off" });
  const set = v => { v = Math.max(range[0], Math.min(range[1], Number(v) || 0)); p.strength = v; track.value = v; show.value = v; save(); };
  track.value = p.strength; show.value = p.strength; track.oninput = () => set(track.value); show.onchange = () => set(show.value);
  return el("div", { style: "min-width:420px;flex:1" }, el("label", {}, "Strength (0 is off)"),
    el("div", { class: "row", style: "gap:8px;align-items:center;flex-wrap:nowrap" }, el("span", { class: "note", style: "white-space:nowrap" }, range[0] + " " + ends[0]), track, el("span", { class: "note", style: "white-space:nowrap" }, ends[1] + " +" + range[1]), show,
      el("button", { type: "button", class: "small", style: "flex:none;width:auto", title: "Back to 0: the LoRA has no effect", onclick: () => set(0) }, "Off (0)")),
    el("datalist", { id: "zero-" + r.id }, el("option", { value: 0, label: "0" }))); }
// obj.loras is the list a clip, a board or a film renders with; save runs after every change
function loraHub(obj, save = () => {}, redraw = render) { if (!obj.loras) obj.loras = [];
  const done = () => { save(); redraw(); }, picked = id => obj.loras.find(p => p.id === id);
  const chip = r => { const on = picked(r.id), st = r.installing;
    const label = st && st.stage !== "failed" ? r.name + " · " + st.stage + (st.of ? " " + Math.round(100 * st.done / st.of) + "%" : "…") : (on ? "✓ " : r.installed ? "" : "⤓ ") + r.name;
    return el("button", { type: "button", class: "small lora" + (on ? " on" : ""), title: (r.author ? "By " + r.author + ". " : "") + (r.notes || "") + (r.installed ? "" : " Click to download it to this machine."), onclick: async () => {
      if (st && st.stage !== "failed") return;
      if (!r.installed) { if (!confirm("Download “" + r.name + "”" + (r.author ? " by " + r.author : "") + " to this machine?\n\n" + (r.licence || "The file is the author's, under its own licence: read it on the LoRA's page first.") + (r.page ? "\n\n" + r.page : ""))) return;
        try { await api("/api/loras/" + r.id + "/install", {}); await loadLoras(); if (!on) obj.loras.push({ id: r.id, strength: r.strength || 1, style: Object.keys(r.styles || {})[0] || "" }); hubWatch(redraw); done(); } catch (e) { toast(e.message); } return; }
      if (on) obj.loras = obj.loras.filter(p => p.id !== r.id); else obj.loras.push({ id: r.id, strength: r.strength || 1, style: Object.keys(r.styles || {})[0] || "" }); done(); } }, label); };
  const row = p => { const r = LORAS.find(x => x.id === p.id); if (!r) return null; const range = r.range || [0.1, 2];
    return el("div", { class: "row lorarow" }, el("div", { class: "fit", style: "min-width:210px" }, el("b", {}, r.name), el("div", { class: "note" }, (r.author ? "by " + r.author : "") + (r.trigger ? " · trigger " + r.trigger + ", added to the prompt for you" : "") + (r.installed ? "" : " · not installed yet: it is left out of the render"))),
      Object.keys(r.styles || {}).length ? el("div", { class: "fit", style: "width:" + (r.choice ? 210 : 150) + "px" }, el("label", {}, r.choice || "Look"), select(p, "style", Object.fromEntries(Object.keys(r.styles).map(k => [k, k])), save)) : null,
      range[0] < 0 ? loraSlider(p, r, range, save) : el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Strength"), bind(p, "strength", el("input", { type: "number", step: 0.1, min: Math.min(range[0], 0.1), max: Math.max(range[1], 2), title: "Made for " + (r.strength || 1) + "; the author's range is " + range[0] + " to " + range[1] }), save)),
      r.flow && r.flow.asks.length ? flowRow(obj, r, done) : null,
      obj === C && !(r.flow && r.flow.asks.length) ? loraWants().filter(w => w.name === r.name).map(w => el("div", { class: "note", style: "flex-basis:100%;color:var(--warn)" }, el("b", {}, "Needs: "), w.text + " ", el("button", { type: "button", class: "small", onclick: w.go }, w.label))) : null,
      r.builtin ? null : el("button", { class: "fit small", title: "Take this LoRA out of the hub (its file goes to the trash folder)", onclick: async () => { if (!confirm("Remove “" + r.name + "” from the hub?")) return; try { await api("/api/loras/" + r.id, undefined, "DELETE"); obj.loras = obj.loras.filter(x => x.id !== r.id); await loadLoras(); done(); } catch (e) { toast(e.message); } } }, "🗑")); };
  const f = HUB.form, up = el("input", { type: "file", accept: ".safetensors", onchange: async () => { if (!up.files[0]) return; const data = new FormData(); data.append("file", up.files[0]); try { const r = await (await fetch("/api/upload", { method: "POST", body: data })).json(); f.file = r.path; toast("Uploaded " + up.files[0].name); } catch (e) { toast(e.message); } } });
  const form = !HUB.open ? null : el("div", { class: "group" }, el("div", { class: "note" }, "A recipe for a MiniMax H3 LoRA: where its file comes from and how it is used. It becomes a button in the hub on this machine."),
    el("div", { class: "row" }, field("Name", bind(f, "name", el("input", { placeholder: "My 16mm western LoRA" }))), field("By", bind(f, "author", el("input", { placeholder: "its author" })))),
    el("div", { class: "row" }, field("Download address of the .safetensors file", bind(f, "url", el("input", { placeholder: "https://…/lora.safetensors" }))), field("or upload the file", up)),
    el("div", { class: "row" }, field("Trigger word", bind(f, "trigger", el("input", { placeholder: "added at the start of the shot" }))), el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Strength"), bind(f, "strength", el("input", { type: "number", step: 0.1, min: 0.1, max: 2 }))),
      field("Looks (optional): name = phrase; name = phrase", bind(f, "styles", el("input", { placeholder: "VHS = shot on VHS.; 16mm = shot on 16mm film." })))),
    field("Notes", bind(f, "notes", el("input", { placeholder: "what it does, the sizes and steps it was made for" }))),
    el("div", { class: "row" }, el("button", { class: "fit primary", onclick: async () => { try { const styles = Object.fromEntries(f.styles.split(";").map(x => x.split("=").map(y => y.trim())).filter(x => x.length === 2 && x[0] && x[1]));
        const r = await api("/api/loras", { ...f, styles }); if (!r.installed && r.url) await api("/api/loras/" + r.id + "/install", {}); await loadLoras(); obj.loras.push({ id: r.id, strength: r.strength || 1, style: Object.keys(r.styles || {})[0] || "" });
        HUB.form = { name: "", url: "", file: "", trigger: "", strength: 1, styles: "", author: "", notes: "" }; HUB.open = false; hubWatch(redraw); done(); } catch (e) { toast(e.message); } } }, "Add to the hub"),
      el("button", { class: "fit", onclick: () => { HUB.open = false; redraw(); } }, "Cancel")));
  const failed = LORAS.filter(r => r.installing && r.installing.stage === "failed");
  const use = r => { if (picked(r.id)) obj.loras = obj.loras.filter(p => p.id !== r.id); else obj.loras.push({ id: r.id, strength: r.strength || 1, style: Object.keys(r.styles || {})[0] || "" }); done(); };
  const terms = r => "“" + r.name + "”" + (r.author ? " by " + r.author : "") + (r.size ? " (" + sizeText(r.size) + ")" : "") + "\n" + (r.licence || "The file is the author's, under its own licence: read it on the LoRA's page first.") + (r.page ? "\n" + r.page : "");
  const install = async list => { if (!list.length) return toast("Nothing to download here");
    if (!confirm("Install on this machine?\n\n" + list.map(terms).join("\n\n"))) return;
    for (const r of list) { try { await api("/api/loras/" + r.id + "/install", {}); } catch (e) { toast(e.message); } }
    await loadLoras(); hubWatch(redraw); redraw(); };
  const inTab = r => HUB.tab === "all" || (r.category || ["mine"]).includes(HUB.tab);
  const words = HUB.q.trim().toLowerCase(), hit = r => !words || [r.name, r.author, r.what, r.notes, r.trigger].join(" ").toLowerCase().includes(words);
  const shown = LORAS.filter(r => inTab(r) && hit(r) && (!HUB.mine || r.installed));
  const card = r => { const on = picked(r.id), st = r.installing, busy = st && st.stage !== "failed", dead = r.status === "unsupported";
    return el("div", { class: "hubcard" + (on ? " on" : "") + (dead ? " off" : ""), "data-lora": r.id },
      el("div", {}, el("b", {}, r.name), el("div", { class: "by", title: "The author's own page, for credit and for the file's licence" }, (r.author ? "by " + r.author : "") + (r.page ? " · " + r.page : ""))),
      el("div", {}, r.what || r.notes || ""),
      el("div", { class: "tags" }, el("span", { class: "tag" + (dead ? " warn" : "") }, LORA_NEEDS[r.needs] || LORA_NEEDS.text),
        r.licence_tag ? el("span", { class: "tag" + (/none|gated|custom/i.test(r.licence_tag) ? " warn" : ""), title: r.licence || "" }, r.licence_tag) : null,
        r.size ? el("span", { class: "tag" }, sizeText(r.size)) : null,
        r.trigger ? el("span", { class: "tag", title: "Added to the prompt for you" }, "trigger: " + r.trigger) : null,
        r.tested ? el("span", { class: "tag", title: "Where we rendered a clip with it" }, "tested: " + r.tested) : null),
      dead || r.status === "gated" ? el("div", { class: "note" + (dead ? "" : ""), style: dead ? "color:var(--warn)" : "" }, (dead ? "Not supported in the Studio yet. " : "") + (r.why || "")) : null,
      r.status === "action" ? actHubNote(r) : null,
      r.notes && !dead ? el("div", { class: "note" }, r.notes) : null,
      st && st.stage === "failed" ? el("div", { class: "note", style: "color:var(--bad)" }, st.error) : null,
      el("div", { class: "acts" },
        dead ? null :
        busy ? el("button", { type: "button", class: "small", disabled: true }, st.stage + (st.of ? " " + Math.round(100 * st.done / st.of) + "%" : "…")) :
        !r.installed ? el("button", { type: "button", class: "small", title: r.page || "", onclick: () => install([r]) }, "⤓ Install") :
        r.status === "action" ? actHubButton(r) :
        r.target === "image" ? el("button", { type: "button", class: "small lora", title: "Opens Create on Edit a picture with this LoRA on", onclick: () => edUse(r) }, "Use in Edit a picture") :
        el("button", { type: "button", class: "small lora" + (on ? " on" : ""), onclick: () => use(r) }, on ? "✓ On this clip" : "Use on this clip"),
        r.installed && !busy ? el("button", { type: "button", class: "small", title: "Moves this LoRA's file to the trash folder" + (r.builtin ? "; the preset stays in the hub" : " and takes it out of the hub"), onclick: async () => { if (!confirm("Remove the file of “" + r.name + "” from this machine?")) return; try { await api("/api/loras/" + r.id, undefined, "DELETE"); obj.loras = obj.loras.filter(x => x.id !== r.id); await loadLoras(); done(); } catch (e) { toast(e.message); } } }, "Remove") : null)); };
  // how many of the hub's LoRAs can be used on this machine now, and why the others cannot
  const hubTally = () => { const sets = LORAS.filter(r => r.builtin), ok = sets.filter(r => r.usable), not = sets.filter(r => !r.usable);
    const why = r => r.status === "unsupported" ? (r.platforms && !r.platforms.includes("mac") && S.platform !== "spark" ? "DGX Spark only" : "not on this machine") : r.status === "gated" ? "waits on its author's approval of your download" : "not installed";
    return ok.length + " of " + sets.length + " LoRAs are usable on this " + (S.platform === "spark" ? "DGX Spark" : "Mac") + (not.length ? ". Not usable: " + not.map(r => r.name + " (" + why(r) + ")").join(", ") + "." : "."); };
  const count = id => LORAS.filter(r => id === "all" || (r.category || ["mine"]).includes(id)).length;
  const tabs = [{ id: "all", name: "All" }, ...HUB.groups].filter(g => g.id === "all" || count(g.id));
  const search = el("input", { type: "search", placeholder: "Search the hub: orbit, anime, fight, speech…", value: HUB.q, style: "max-width:320px", oninput: e => { HUB.q = e.target.value; const at = e.target.selectionStart; redraw(); const again = document.querySelector(".hub input[type=search]"); if (again) { again.focus(); again.setSelectionRange(at, at); } } });
  const toInstall = shown.filter(r => !r.installed && r.status !== "unsupported" && !(r.installing && r.installing.stage !== "failed"));
  const browser = !HUB.browse ? null : el("div", { class: "group", id: "lorabrowser" },
    el("div", { class: "hubtabs" }, tabs.map(g => el("button", { type: "button", class: "small" + (HUB.tab === g.id ? " on" : ""), onclick: () => { HUB.tab = g.id; redraw(); } }, g.name + " (" + count(g.id) + ")"))),
    el("div", { class: "row", style: "align-items:center" }, search,
      el("label", { class: "fit", style: "display:flex;gap:6px;align-items:center;margin:0" }, el("input", { type: "checkbox", checked: HUB.mine, style: "width:auto", onchange: e => { HUB.mine = e.target.checked; redraw(); } }), "Installed only"),
      HUB.tab !== "all" && toInstall.length ? el("button", { type: "button", class: "fit small", onclick: () => install(toInstall) }, "⤓ Install all " + toInstall.length + " in this category") : null,
      el("button", { type: "button", class: "fit small", onclick: () => { HUB.open = !HUB.open; redraw(); } }, "+ Add your own LoRA")),
    el("div", { class: "note" }, "Each LoRA is its author's work, under that file's own licence; the page named on a card is where it comes from and who to credit. Install takes the file from the LoRA bundle on this machine when there is one (no network), and otherwise from the author's page, and checks it. " + (S.platform === "spark" ? "" : "On a Mac the file is then rewritten once for the MLX engine. ") + "A LoRA for pictures is read as its author wrote it, and is used in Create, Edit a picture. " + "LoRAs from the hub are for text-to-video and first-frame clips; a clip made from reference pictures or videos does not take them yet."),
    el("div", { class: "note", id: "hubtally" }, hubTally()),
    shown.length ? el("div", { class: "hubgrid" }, shown.map(card)) : el("div", { class: "note", style: "padding:10px" }, "Nothing here matches."));
  const quick = LORAS.filter(r => r.status !== "action" && r.target !== "image" && (r.installed || picked(r.id) || (r.installing && r.installing.stage !== "failed")));  // a Library action's LoRA is not for a clip; a picture LoRA is picked in Edit a picture
  return el("div", { class: "hub" }, el("label", {}, "Keys and Wasserman LoRA Hub"),
    el("div", { class: "chips" }, quick.map(chip),
      el("button", { type: "button", class: "small" + (HUB.browse ? " on lora" : ""), id: "lorabrowse", onclick: () => { HUB.browse = !HUB.browse; redraw(); } }, HUB.browse ? "▾ Close the hub" : "▸ Browse the hub (" + LORAS.length + " LoRAs in " + (tabs.length - 1) + " groups)")),
    failed.filter(r => !HUB.browse).map(r => el("div", { class: "note", style: "color:var(--bad)" }, r.name + ": " + r.installing.error)),
    browser, obj.loras.map(row), form,
    obj.loras.length ? el("div", { class: "note" }, S.platform === "spark" ? "With a LoRA the clip renders on ComfyUI's own blocks with the full-precision weights, not the TensorFold engine: slower, and no time estimate." : "With a LoRA, FastH3 renders on the MLX engine, not the native one: slower, and no time estimate. MiniMax H3 at 20 steps is what most LoRAs were made for.") : null); }
function videoControls(after) {
  const two = bind(C, "up", el("input", { type: "checkbox", disabled: !UPSCALED[C.res] }), changed);
  return el("div", { class: "group" },
    S.sharp ? el("div", { class: "row" }, sharpButton(sharpOn(), sharpCreate), el("div", { class: "note" }, sharpOn() ? "On: 1376 × 768, FastH3 8 passes, one pass, no upscale. Seed " + C.seed + (Number(C.seed) === S.sharp.seed ? "" : " (the spec starts at " + S.sharp.seed + "; the next seed is a retake)") + ". His QA gate checks each clip: the spoken line by transcript, then three frames by a vision model (Claude Code, Codex or a local model, under your own sign-in). A clip that fails is made once more at the next seed."
      : "Sam Wasserman's locked recipe in one click: 1376 × 768, FastH3 8 passes, seed 42, 8 or 10 seconds, then his QA gate on each clip.")) : null,
    // MiniMax H3 Turbo is an optional part of the install; without it FastH3 is the engine
    isRef() ? el("label", {}, "Reference to video · MiniMax H3 Ref2VA") : null,
    isRef() ? passChooser("refPasses", S.ref2va_turbo ? { 20: ["20 passes", "the model as released, its own sound"], 4: ["Turbo 4 passes", S.platform === "spark" ? "LightX2V adapter and its own sound: about 3x faster" : "LightX2V adapter, then base-model sound: about 2.5x faster"] } : { 20: ["20 passes", "the model as released, its own sound"] },
      () => C.refPasses, after, C.refPasses === "4" ? "With the Turbo adapter, which was trained for 4 passes; pick 20 passes first for a count without it." : "Without the Turbo adapter.") : null,
    isRef() ? null :
    S.turbo ? el("label", {}, "Video engine") : null, isRef() ? null : S.turbo ? choice(C, "engine", { fasth3: ["FastH3", S.zig ? "FastVideo's distilled H3 on the native Zig engine: fastest" : "FastVideo's distilled H3 on the sparse tile kernel: fastest"], h3: ["MiniMax H3", "Turbo adapter with base-model sound, or the full 20 steps"] }, changed) : null,
    isRef() ? null : el("label", {}, C.engine === "fasth3" ? "Passes" : "Quality"), isRef() ? null : C.engine === "fasth3" ? passChooser("passes", PASSES, () => C.passes, after)
      // a quality of MiniMax H3 is a recipe: Custom keeps the one that was chosen and changes only its count
      : passChooser("h3quality", H3Q, () => H3_PASSES[C.h3quality], after, "Inside “" + H3Q[C.h3quality][0] + "”, the quality that was chosen: it keeps its adapter and its sound, and only the number of passes changes."),
    el("label", {}, "Resolution"), choice(C, "res", Object.fromEntries(Object.entries(RES).filter(([k]) => (S.x2 || !["draft", "2k"].includes(k)) && (k !== "sharp" || S.sharp))), changed, true),
    C.res === "custom" ? el("div", { class: "row" }, el("div", { class: "fit", style: "width:130px" }, el("label", {}, "Width"), typed(bind(C, "cw", el("input", { type: "number", step: 32, min: S.custom.sides[0], max: S.custom.sides[1] })), after)),
      el("div", { class: "fit", style: "width:130px" }, el("label", {}, "Height"), typed(bind(C, "ch", el("input", { type: "number", step: 32, min: S.custom.sides[0], max: S.custom.sides[1] })), after)),
      el("div", { class: "note" }, "Multiples of 32, each side " + S.custom.sides[0] + " to " + S.custom.sides[1] + ", at most 1344 × 768 pixels in all. No time estimate for a custom size.")) : null,
    loraHub(C, () => { C.prompt = ""; }),
    el("label", {}, "Text encoder"),
    choice(C, "textEncoder", { "": ["Regular", "MiniMax H3's own"], ...Object.fromEntries((S.text_encoders || []).map(f => [f, [encoderName(f), "a file in this machine's text_encoders folder"]])), "__own": ["Your own", "a text encoder file you choose"] }, changed, true),
    C.textEncoder === "__own" ? el("div", {}, bind(C, "textEncoderPath", el("input", { placeholder: S.platform === "spark" ? "name of a .safetensors file in ComfyUI's text_encoders folder" : "/path/to/your-text-encoder.safetensors" })),
      el("div", { class: "note" }, "Any MiniMax H3 text encoder file on this machine. The Studio does not provide, check or recommend it: the file, its license and what you make with it are your own choice and responsibility.")) : null,
    el("div", { class: "row" },
      !S.x2 ? null : el("div", {}, el("label", {}, "Upscale"), el("label", { class: "switch" }, two, el("span", {}, "2x upscale", el("small", {}, UPSCALED[C.res] ? "2x decoder → " + UPSCALED[C.res][1] + ", about 10 s more" : "already part of this output")))),
      el("div", { class: "fit", style: "width:290px" }, el("label", {}, "Length"), lengthPicker(C, after, isRef() ? {} : Object.fromEntries(Object.entries(S.long).map(([k, v]) => [k, v + " · " + Math.ceil(k / 15) + " clips chained"])))),
      el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Seed"), bind(C, "seed", el("input", { type: "number", min: 0 })))));
}
function summary(est) {
  const size = output().size, hand = String(C.steps).trim() && !isRef() && C.engine !== "fasth3" ? " · " + C.steps + " passes (custom)" : "", engine = isRef() ? "MiniMax H3 Ref2VA · " + refPassText() : C.engine === "fasth3" ? "FastH3 · " + (String(C.steps).trim() ? C.steps + " passes (custom)" : C.passes + " passes") + " · " + S.engine : "MiniMax H3 · " + H3Q[C.h3quality][0];
  return el("div", { class: "summary" }, el("span", { class: "chip k" }, { t2v: "TEXT → VIDEO", scout: "SCOUT IMAGE → VIDEO", i2v: "IMAGE → VIDEO", r2v: "REFERENCES → VIDEO" }[C.mode]),
    el("span", { class: "chip" }, engine + hand), encoderOf() ? el("span", { class: "chip" }, "text encoder: " + (C.textEncoder === "__own" ? "your own" : encoderName(encoderOf()))) : null, el("span", { class: "chip" }, size[0] + " × " + size[1]), el("span", { class: "chip" }, isLong() ? S.long[C.seconds] + " · " + Math.ceil(C.seconds / 15) + " × 15 s" : exact(framesOf(C))), est);
}
function settingsRow(obj, after, keep = {}) {
  const presets = Object.entries(S.presets).filter(([k]) => !keep.presets || keep.presets(k));
  const qualities = Object.fromEntries(Object.entries(S.qualities).filter(([k]) => !keep.qualities || keep.qualities(k)));
  return el("div", { class: "row" },
    field("Clip length", lengthPicker(obj, after)),
    field("Output", select(obj, "preset", Object.fromEntries(presets.map(([k, v]) => [k, v.label])), after)),
    field("Quality", select(obj, "quality", qualities, after)),
    el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Seed"), bind(obj, "seed", el("input", { type: "number", min: 0 }))));
}
async function showEstimate(obj, node) { try { const r = await api("/api/estimate", { preset: obj.preset, quality: obj.quality, ...lengthOf(obj), text_only: obj === C && noFrame(), referenced: obj === C && isRef(), ref_media: obj === C && refMedia(), ref_turbo: obj === C && isRef() && C.refPasses === "4", ...(obj === C ? { steps: String(C.steps).trim() } : {}), loras: obj === C && isRef() ? [] : lorasOf(obj), with_lora: lorasOf(obj).length > 0, ...(obj.preset === "custom" && obj.cw ? { width: Number(obj.cw), height: Number(obj.ch) } : {}), references: obj === C && isRef() ? C.refs.length : 0 });
  node.textContent = r.seconds ? "≈ " + mmss(r.seconds) + (S.platform === "spark" ? " on a DGX Spark, models loaded" : " on an M5 Ultra") : "";
  // a DGX Spark's memory guard (spark/memory_guard.py): said here, before the clip is queued, with what fits
  if (r.memory && r.memory.fits === false) put(node, el("div", { class: "tip", style: "margin-top:6px" }, el("b", {}, "This clip will be refused: memory"), el("div", {}, r.memory.reason)));
  else if (r.memory && r.memory.need_mb) put(node, el("span", { title: "Estimated from measured peaks; the server is stopped if the machine runs low all the same." }, (r.seconds ? " · " : "") + "about " + Math.round(r.memory.need_mb / 1000) + " GB of this machine's memory")); } catch { node.textContent = ""; } }
