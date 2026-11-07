"use strict";
// Sparks (app/sparks.py): SPK.n is the "sparks" number every job queued from this page carries; one chooser everywhere
const SPK = { data: null, n: 1, form: { address: "", folder: "" }, plans: {}, takes: {}, ticks: 0 };
// which Spark rendered a clip (sparks/takes.jsonl in the studio folder), for a take's line
const spkTake = path => { const t = SPK.takes[path]; return t && t.spark ? "on " + t.spark : null; };
const spkLimit = () => (SPK.data && SPK.data.limit) || 8;
// "5 of 8 available" when more are asked for than there are; else how many will be used
const spkCount = (d, n) => !d || !d.supported ? "this machine" : n > d.available ? d.available + " of " + n + " available" : n === 1 ? "this Spark alone" : n + " of " + d.available + " available Sparks";
const spkSig = d => !d ? "" : JSON.stringify([d.available, d.default, d.workers.map(w => [w.address, w.name, w.state, w.working, w.enabled, w.why, w.job, w.memory_gb, w.usable]), d.this && d.this.job]);
async function spkLoad(draw) { const was = spkSig(SPK.data); try { const first = !SPK.data; SPK.data = await api("/api/sparks");
    if (first) { let kept = null; try { kept = Number(localStorage.getItem("sparks")); } catch {} SPK.n = SPK.data.supported ? Math.max(1, Math.min(kept || SPK.data.default || 1, SPK.data.limit)) : 1; }
  } catch { return; }
  if (SPK.data.supported && !(SPK.ticks++ % 5)) api("/api/sparks/takes").then(t => { SPK.takes = t; }, () => {});
  if (spkSig(SPK.data) === was) return; SPK.plans = {}; spkQueue();
  if (draw && typeof tab !== "undefined" && !document.activeElement.matches("input,textarea,select")) render(); }
function spkSet(n) { SPK.n = SPK.data && SPK.data.supported ? Math.max(1, Math.min(Number(n) || 1, spkLimit())) : 1; SPK.plans = {};
  try { localStorage.setItem("sparks", String(SPK.n)); } catch {} api("/api/sparks", { sparks: SPK.n }).catch(() => {}); spkQueue(); render(); }
// What the number will do for this job, asked of the app (POST /api/sparks/plan) and said under the chooser
async function spkPlan(node, what) { const body = { sparks: SPK.n, kind: what.kind || "video", params: what.params || {}, clips: what.clips || 1, seconds: what.seconds || 0 }, key = JSON.stringify(body);
  let r = SPK.plans[key]; if (!r) { try { r = SPK.plans[key] = await api("/api/sparks/plan", body); } catch { return; } }
  node.replaceChildren(...spkSays(r)); }
function spkSays(r) { if (!r.supported) return [r.says];
  const e = r.estimate;
  return [el("b", {}, r.count + ". "), r.says,
    e && r.mode === "parallel" ? " Estimate: about " + mmss(e.seconds) + " for all of them, against " + mmss(e.one_spark) + " on one Spark (" + e.faster + "x; " + e.note + ")." : "",
    r.without.length ? el("div", {}, "Not used: " + r.without.map(w => w.name + " (" + w.why + ")").join("; ") + ".") : ""]; }
// The chooser: 1 to the limit. `what` says what is about to be queued: { kind, params, clips, seconds }.
function sparksPick(what = {}, label = "Sparks") { const d = SPK.data, says = el("div", { class: "note", style: "margin-top:4px" });
  if (!d) return null;
  if (!d.supported) return el("div", { class: "sparks" }, el("label", {}, label), el("div", { class: "row", style: "gap:5px;align-items:center" }, el("button", { class: "small on fit", disabled: true }, "1"), el("span", { class: "note" }, d.note)));
  spkPlan(says, what);
  return el("div", { class: "sparks", "data-keep": "" }, el("label", { title: "How many DGX Sparks this job may use, this one counted. More than there are is allowed: what is there is used." }, label),
    el("div", { class: "row", style: "gap:5px;align-items:center;flex-wrap:wrap" },
      Array.from({ length: d.limit }, (_, i) => el("button", { type: "button", class: "small fit" + (SPK.n === i + 1 ? " on" : "") + (i + 1 > d.available ? " far" : ""), "aria-pressed": String(SPK.n === i + 1),
        title: i + 1 > d.available ? "Only " + d.available + " available now: " + d.available + " will be used" : (i + 1) + (i ? " Sparks" : " Spark"), onclick: () => spkSet(i + 1) }, String(i + 1))),
      el("button", { type: "button", class: "small fit", title: "The list of Sparks: add one, check them, update and start them", onclick: () => { tab = "Sparks"; history.replaceState(null, "", "#Sparks"); render(); } }, "Sparks…")), says); }
// The queue's own line, on every tab
function spkQueue() { const box = document.querySelector("#sparks"); if (!box || !SPK.data) return; const d = SPK.data;
  if (!d.supported) { box.replaceChildren(el("div", { class: "note" }, "Sparks: 1. " + d.note)); return; }
  box.replaceChildren(el("div", { class: "sparks" }, el("div", { class: "row", style: "gap:4px;align-items:center;flex-wrap:wrap" }, el("span", { class: "note fit", title: "How many DGX Sparks a job queued now may use, this one counted" }, "Sparks"),
    Array.from({ length: d.limit }, (_, i) => el("button", { type: "button", class: "small fit" + (SPK.n === i + 1 ? " on" : "") + (i + 1 > d.available ? " far" : ""), onclick: () => spkSet(i + 1) }, String(i + 1)))),
    el("div", { class: "note" }, spkCount(d, SPK.n) + (SPK.n > 1 ? ": queued clips render one per Spark." : ".")))); }
const spkDo = async (path, body, method) => { try { SPK.data = await api(path, body, method); SPK.plans = {}; } catch (e) { toast(e.message, 6000); } spkQueue(); render(); };
const spkHas = can => !can ? "" : [can.fasth3 ? "FastH3 clips (text, first and last frame)" : "no FastH3", can.ref2va ? "Ref2VA" + (can.ref2va_turbo ? " with Turbo" : "") : "no Ref2VA", can.image ? "image model" : "no image model",
  can.loras ? can.loras + " LoRA" + (can.loras === 1 ? "" : "s") : "no LoRAs", can.ffmpeg ? "ffmpeg" : "no ffmpeg"].join(" · ");
const spkDot = state => el("span", { style: "display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;background:" + ({ ready: "var(--ok)", busy: "var(--warn)", update: "var(--warn)", stopped: "#94a3b8", new: "#64748b" }[state] || "var(--bad)") });
function viewSparks() { const box = el("section", { class: "card" }, el("h2", {}, "Sparks")), d = SPK.data;
  if (!d) { spkLoad(true).then(() => { if (SPK.data && tab === "Sparks") render(); }); return put(box, el("div", { class: "note" }, "Loading…")), box; }
  if (!d.supported) return put(box, el("div", { class: "note" }, d.note), sparksPick()), box;
  const f = SPK.form, a = encodeURIComponent;
  const row = w => el("div", { class: "seg" }, el("div", { class: "seghead" }, spkDot(w.working ? "busy" : w.state), el("b", {}, w.name), el("span", { class: "note" }, w.address),
      el("span", { class: "pill " + (w.state === "ready" ? "done" : w.state === "busy" || w.working ? "running" : w.state === "new" || w.state === "stopped" ? "queued" : "failed") }, w.working || w.says), el("span", { style: "flex:1" }),
      el("label", { class: "note fit", style: "display:flex;gap:5px;align-items:center;margin:0", title: "Unticked, this Spark is given no clips; it stays in the list" }, el("input", { type: "checkbox", style: "width:auto", checked: w.enabled, onchange: e => spkDo("/api/sparks/workers/" + a(w.address), { enabled: e.target.checked }) }), "use"),
      el("button", { class: "small", disabled: !!w.working, title: "Look at this Spark: reachable, its Studio's version, what it can render, free memory, busy or idle. Nothing on it is changed.", onclick: () => spkDo("/api/sparks/workers/" + a(w.address) + "/check", {}) }, "Check"),
      el("button", { class: "small", disabled: !!w.working, title: "Send this Spark's Studio code to it (rsync over ssh, as you), start its app on its own 127.0.0.1, and check it again. Models are not sent.", onclick: () => spkDo("/api/sparks/workers/" + a(w.address) + "/update", {}) }, "Update and start"),
      el("button", { class: "small", disabled: !!w.working, title: "Stop its Studio app and its ComfyUI server: its memory is free again", onclick: () => confirm("Stop the Studio app on " + w.name + " and free its memory?") && spkDo("/api/sparks/workers/" + a(w.address) + "/stop", {}) }, "Stop"),
      el("button", { class: "small", title: "Take this Spark off the list (what runs on it is left as it is)", onclick: () => confirm("Remove " + w.name + " from the list?") && spkDo("/api/sparks/workers/" + a(w.address), undefined, "DELETE") }, "Remove")),
    el("div", { class: "note" }, [w.version ? "Studio " + w.version : null, w.guard ? "memory guard" : w.state === "ready" || w.state === "busy" ? null : w.version ? "memory guard not confirmed" : null, w.memory_gb != null ? w.memory_gb + " GB of memory available" : null,
      w.disk_gb != null ? w.disk_gb + " GB of disk free" : null, w.job ? "rendering job " + w.job : null].filter(Boolean).join(" · ")),
    w.can ? el("div", { class: "note" }, "Renders: " + spkHas(w.can)) : null,
    w.why || (w.unusable && w.state === "ready") ? el("div", { class: "note", style: "color:var(--warn)" }, w.why || w.unusable) : null);
  put(box, el("div", { class: "note" }, "A job may use more than one DGX Spark. Choose how many here, in the queue, or where you queue a job: this Spark counts as one. Several clips render one per Spark, side by side; one clip renders on one Spark. Each other Spark runs its own copy of this Studio, closed to the network: this app reaches it through an ssh tunnel it opens itself, so you need a key login to it as the user who runs this app."),
    sparksPick({ clips: 8 }, "Sparks for new jobs"),
    el("h3", {}, "This Spark and " + (d.listed - 1) + " other" + (d.listed === 2 ? "" : "s") + " · " + d.available + " available"),
    el("div", { class: "seg" }, el("div", { class: "seghead" }, spkDot(d.this.state), el("b", {}, d.this.name), el("span", { class: "note" }, "this Spark"), el("span", { class: "pill " + (d.this.job ? "running" : "done") }, d.this.job ? "rendering" : "idle")),
      el("div", { class: "note" }, "Studio " + d.this.version + " · memory guard · " + d.this.memory_gb + " GB of memory available · " + d.this.disk_gb + " GB of disk free"), el("div", { class: "note" }, "Renders: " + spkHas(d.this.can))),
    d.workers.map(row),
    el("div", { class: "row", style: "margin-top:10px" }, el("button", { class: "fit", disabled: !d.workers.length, onclick: () => spkDo("/api/sparks/check", {}) }, "Check all"), el("span")),
    el("h3", {}, "Add a Spark"),
    el("div", { class: "note" }, "Its address as you would type it after ssh. It needs the Studio installed once (oneshot-setup-spark.sh, with the FastH3 models); “Update and start” then sends this Spark's Studio code and starts its app. No password is asked and nothing is installed as root: if ssh asks for a password, set up a key login first."),
    el("div", { class: "row" }, field("Address", bind(f, "address", el("input", { placeholder: "192.0.2.10  or  user@spark-5" }))), field("Studio folder there, under the home folder", bind(f, "folder", el("input", { placeholder: d.folder }))),
      el("button", { class: "primary fit", onclick: async () => { if (!f.address.trim()) return toast("Give the Spark's address"); await spkDo("/api/sparks/workers", { address: f.address.trim(), folder: f.folder.trim() || null }); f.address = ""; } }, "Add")));
  return box; }
if (typeof window !== "undefined" && window.addEventListener) { window.addEventListener("load", () => setTimeout(() => spkLoad(true), 400)); setInterval(() => spkLoad(true), 4000); }
