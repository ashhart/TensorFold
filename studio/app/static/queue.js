// Queue: job cards, polling, and the page's start.
"use strict";
// Sam Wasserman's QA gate on a Sharp Spec clip: one line per take, with its contact sheet
function qaLines(job) { const q = job.qa; if (!q) return null; const mark = r => ({ pass: "✓", fail: "✗", skipped: "–" }[r] || "?");
  return el("div", { class: "note" }, el("b", { style: "color:" + (q.passed ? "var(--ok)" : q.passed === false ? "var(--bad)" : "inherit") }, "Wasserman QA gate: " + (q.passed && q.takes.at(-1).audio.result === "skipped" && q.takes.at(-1).identity.result === "skipped" ? "not checked, both gates are missing their tool on this machine" : q.passed ? "passed" + (q.takes.at(-1).audio.result === "skipped" || q.takes.at(-1).identity.result === "skipped" ? " the one gate that could run" : "") : q.passed === false ? (q.takes.length > 1 ? "failed on both takes, the last take is kept" : "failed") : q.skipped ? "not run" : "could not run")), q.error ? " · " + q.error : "", q.note ? " · " + q.note : "", q.from ? " · run here on a clip fetched from " + q.from : "",
    q.takes.map(t => el("div", {}, "Take " + t.take + " · seed " + t.seed + " · speech " + mark(t.audio.result) + " " + (t.audio.heard ? "“" + t.audio.heard + "” · " : "") + t.audio.note + " · identity " + mark(t.identity.result) + " " + t.identity.note + (t.sharpness.length ? " · sharpness " + t.sharpness.join(" / ") : "") + " · ",
      el("a", { href: file(t.sheet), target: "_blank" }, "contact sheet")))); }
// The shot check on a finished clip (app/shot.py): keyframes held, no cut inside one shot. A flag beside the clip.
function shotLines(job) { const q = job.shot; if (!q) return null; const takes = q.takes || [];
  const frame = r => !r ? "none given" : (r.held ? "held" : "not held") + " (difference " + r.difference + " of 255, likeness " + r.likeness + ")";
  const again = async () => { try { await api("/api/jobs", { kind: job.kind, title: job.title, params: { ...job.params, seed: q.offer } }); toast("Queued again at seed " + q.offer); poll(); } catch (e) { toast(e.message); } };
  return el("div", { class: "note" }, el("b", { style: "color:" + (q.passed ? "var(--ok)" : q.passed === false ? "var(--bad)" : "inherit") }, "Shot check: " + (q.passed ? "passed" + (takes.length > 1 ? " on the second take" : "") : q.passed === false ? (takes.length > 1 ? "failed on both takes, the last take is kept" : "failed") : "could not run")), q.error ? " · " + q.error : "",
    takes.map(t => el("div", {}, "Take " + t.take + " · seed " + t.seed + " · first frame " + frame(t.first) + " · last frame " + frame(t.last) + " · " + (t.cuts === null ? "cuts not looked for: the prompt writes a second shot" : t.cuts.length ? t.cuts.length + (t.cuts.length === 1 ? " cut" : " cuts") + " inside the shot, at " + t.cuts.map(c => c.seconds + " s").join(", ") : "no cut inside the shot"),
      t.kept ? [" · ", el("a", { href: file(t.kept), target: "_blank" }, "the take that failed")] : null,
      // the longest stretch without a cut (shot.py window): where an editor trims to when the model cut all the same
      t.window && !t.window.whole ? [" · the longest stretch without a cut is frames " + t.window.start + " to " + (t.window.end - 1) + " (" + (t.window.frames / 24).toFixed(2) + " s)",
        t === takes[takes.length - 1] && job.outputs && job.outputs[0] ? [" ", el("button", { class: "small", onclick: () => addToTimeline(job.outputs[0], t.window.in, t.window.out) }, "+ timeline, trimmed to it")] : null] : null)),
    q.offer != null && !job.params.output ? el("div", {}, "The model does this at some seeds whatever the prompt says. ", el("button", { class: "small", onclick: again }, "Render it again at seed " + q.offer)) : null); }
// What a video prompt forbids by name (app/negation.py): said, never taken out
const negText = list => list.map(n => "“" + n.phrase + "”" + (n.field ? " (" + n.field + ")" : "")).join(", ");
function negLines(job) { const list = job.negations; if (!list || !list.length) return null;
  return el("div", { class: "note" }, el("b", { style: "color:var(--warn)" }, "The prompt forbids by name: "), negText(list) + ". " + S.negation_reason); }
async function negShow(node) { const text = C.prompt; try { const r = await api("/api/prompt/check", { prompt: text }); C.negations = r.negations; C.negFor = text; } catch { C.negations = []; }
  if (node) node.replaceChildren(...[negNote()].filter(Boolean)); }
function negNote() { if (C.negFor !== C.prompt) { negShow(null).then(() => { if (C.negations.length) render(); }); return null; }
  return C.negations.length ? el("div", { class: "tip", style: "margin-top:8px" }, el("b", {}, "This prompt forbids " + (C.negations.length === 1 ? "something" : C.negations.length + " things") + " by name"), negText(C.negations) + ". " + S.negation_reason + " The prompt is sent as you wrote it.") : null; }
function jobCard(job) { const p = job.progress; const elapsed = job.elapsed != null ? job.elapsed : (job.started ? ((job.finished || Date.now() / 1000) - job.started) : null);
  // the bar follows the time against the estimate (a job has several stages, each with its own step count)
  const share = job.estimate && elapsed != null ? Math.min(99, Math.max(3, Math.round(100 * elapsed / job.estimate))) : (p ? Math.round(100 * p.at / p.of) : 3);
  const pre = el("pre", { class: "hide" }, job.log || "");
  return el("div", { class: "job" }, el("div", { class: "seghead" }, el("b", {}, job.title), el("span", { class: "pill " + job.status }, job.status), el("span", { style: "flex:1" }),
      ["queued", "running"].includes(job.status) ? el("button", { class: "small", onclick: async () => { await api("/api/jobs/" + job.id + "/cancel", {}); poll(); } }, "Cancel")
        : [el("input", { type: "checkbox", title: "Select this job", "aria-label": "Select " + job.title, checked: PICK.has(job.id), style: "width:auto;flex:none", onchange: e => { if (e.target.checked) PICK.add(job.id); else PICK.delete(job.id); jobTools(); } }),
           el("button", { class: "small", title: "Delete this job and what it made (moved to the trash folder)", "aria-label": "Delete " + job.title, onclick: () => clearOut({ jobs: [job.id] }, "Delete “" + job.title + "” and the files it made?") }, "🗑")],
      el("button", { class: "small", onclick: () => pre.classList.toggle("hide") }, "Log")),
    job.status === "running" ? el("div", {}, el("div", { class: "bar" }, el("i", { style: "width:" + share + "%" })),
      el("div", { class: "note" }, (p ? p.stage + " " + p.at + "/" + p.of + " · " : "loading · ") + (job.estimate && elapsed != null ? (job.estimate - elapsed > 5 ? "about " + mmss(job.estimate - elapsed) + " left" : "almost done") : "no time estimate for this setting") + (job.passes ? " · " + job.passes + " passes" : "") + (job.engine ? " · " + (job.engine === "zig" ? "native Zig engine" : job.engine === "cuda" ? "CUDA" : "MLX engine") : ""))) : null,
    job.spark && S.sparks && S.sparks.supported && job.sparks && (job.sparks.chosen > 1 || (job.moved || []).length) ? el("div", { class: "note" }, (job.status === "running" ? "Rendering on " : job.status === "done" ? "Rendered on " : "Was on ") + job.spark + " · Sparks " + job.sparks.chosen) : null,
    job.status === "queued" && job.sparks && job.sparks.chosen > 1 ? el("div", { class: "note" }, "Sparks " + job.sparks.chosen + (job.sparks.available < job.sparks.chosen ? " (" + job.sparks.available + " available)" : "") + " · " + job.sparks.says) : null,
    (job.moved || []).map(m => el("div", { class: "note", style: "color:var(--warn)" }, "Taken back from " + m.from + ": " + m.why)),
    job.status === "done" && elapsed ? el("div", { class: "note" }, "Took " + mmss(elapsed) + (job.passes ? " · " + job.passes + " passes" : "") + (job.engine ? " · " + (job.engine === "zig" ? "native Zig engine" : job.engine === "cuda" ? "CUDA" : "MLX engine") : "")) : null, job.error ? el("div", { class: "note", style: "color:var(--bad)" }, job.error) : null, qaLines(job), shotLines(job), negLines(job), pre); }
function jobTools() { const done = jobs.filter(j => ["done", "cancelled"].includes(j.status)).length, failed = jobs.filter(j => j.status === "failed").length;
  for (const id of [...PICK]) if (!jobs.some(j => j.id === id && !["queued", "running"].includes(j.status))) PICK.delete(id);
  $("#jobtools").replaceChildren(...(done || failed ? [
    el("button", { class: "small", disabled: !PICK.size, onclick: () => clearOut({ jobs: [...PICK] }, "Delete the " + count(PICK.size, "selected job") + " and the files they made?") }, "Delete selected (" + PICK.size + ")"),
    el("button", { class: "small", disabled: !failed, title: "Take every failed job out of this list", onclick: () => clearOut({ failed: true }, "Clear the " + count(failed, "failed job") + " from the queue list?") }, "Clear failed (" + failed + ")"),
    el("button", { class: "small", disabled: !done, title: "Every finished or cancelled job leaves this list and its clips, stills and check sheets go to the trash folder. A clip that is in a film (a project's timeline, a Slate take) stays.", onclick: () => clearOut({ finished: true }, "Delete all " + count(done, "finished job") + " and the files they made?\nA clip that is in a film (a project's timeline or a Slate take) stays where it is.") }, "Delete all finished (" + done + ")")] : [])); }
let lastSig = "";
async function poll() { try { jobs = await api("/api/jobs"); } catch { return; }
  jobTools(); const box = $("#jobs"); box.replaceChildren(...(jobs.length ? jobs.slice(0, 30).map(jobCard) : [el("div", { class: "note" }, "Nothing queued yet.")]));
  // a timeline changed from another window or device shows here without a reload
  if (project && tab === "Editor" && !document.activeElement.matches("input,textarea,select")) { try { const fresh = await api("/api/projects/" + project.id);
    if (JSON.stringify(fresh.timeline) !== JSON.stringify(project.timeline)) { project = fresh; render(); } } catch {} }
  const sig = jobs.map(j => j.id + j.status).join(); if (sig !== lastSig) { const had = lastSig; lastSig = sig; if (had && !document.activeElement.matches("input,textarea,select")) render(); } }

function render() { $("#tabs").replaceChildren(...TABS.map(t => el("button", { class: t === tab ? "on" : "", onclick: () => { tab = t; history.replaceState(null, "", "#" + t); render(); } }, t)));
  $("#view").replaceChildren({ Create: viewCreate, Production: viewProduction, Editor: viewEditor, Library: viewLibrary, Sparks: viewSparks, Tutorial: viewTutorial }[tab]()); coach();
  // a redraw (a job changed, a view finished loading) must not drop the outline a tutorial step just put on a control
  if (T.spot && Date.now() < T.spot[1]) spot(T.spot[0], 3, true); }
(async () => { const want = decodeURIComponent(location.hash.slice(1)); if (TABS.includes(want)) tab = want; actLoad();
  // links can preset the Create form: ?goes=image, ?engine=h3&res=720p&up=1&step=4 ...
  for (const [k, v] of new URLSearchParams(location.search)) if (k in C && k !== "mode") C[k] = k === "up" ? v === "1" : (k === "step" || k === "count" ? Number(v) : v);
  S = await api("/api/state"); await loadLoras(); if (!S.turbo) C.engine = "fasth3"; sync(); $("#gohome").onclick = () => { tab = "Create"; C.step = 0; T.coach = false; history.replaceState(null, "", "#Create"); window.scrollTo(0, 0); render(); }; $("#plat").textContent = S.platform_label; if (S.ffmpeg_note) { $("#warnbar").textContent = S.ffmpeg_note; $("#warnbar").classList.remove("hide"); } $("#plat").className = "plat " + S.platform; document.title = "Wasserman Slate AI Studio · " + (S.platform === "spark" ? "DGX Spark" : "Mac"); $("#ver").textContent = "pack " + S.version; $("#home").textContent = S.home; await poll(); render(); setInterval(poll, 2000); })();
