"use strict";
// Wasserman Slate AI Studio in the Production tab: a port of Sam Wasserman's Slate (Apache-2.0; slate/NOTICE)
const SL = { mode: "board", metas: null, p: null, view: "shots", studio: "cast", sound: "voices", sceneId: null, shotId: null, sheet: {}, busy: "", data: null, brain: null,
  have: {}, plan: {}, open: {}, reply: "", cover: { plan: "full-coverage", text: "", ask: "" }, issues: null };
try { SL.mode = localStorage.getItem("production-mode") || "board"; } catch {}
const slId = prefix => prefix + "-" + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2, 7);
const slNow = () => new Date().toISOString();
let slTimer = null;
const slSave = () => { clearTimeout(slTimer); slTimer = setTimeout(() => SL.p && api("/api/slate/projects/" + SL.p.id, SL.p, "PUT").catch(e => toast("Not saved: " + e.message)), 500); };
const slDraw = () => { slSave(); render(); };
const slScene = () => SL.p.scenes.find(s => s.id === SL.sceneId) || SL.p.scenes[0] || null;
const slShot = () => { const sc = slScene(); return sc ? sc.shots.find(s => s.id === SL.shotId) || sc.shots[0] || null : null; };
const slTf = shot => shot.tf || (shot.tf = { start: "text", cast: [], location: "", lines: [], soundscape: "", music: "N/A", cue: "", seed: 1, image: null, h3: "", h3Auto: true });
const slFilm = () => SL.p.tf || (SL.p.tf = { res: "480p", quality: S.qualities.fh8 ? "fh8" : Object.keys(S.qualities)[0], refPasses: "20" });
// A song under the film (a music video): tempo and bar grid, kept under the project's "tf". A shot's slot is in bars.
const slSong = () => { const g = (SL.p.tf || {}).song || {}, bpm = Number(g.bpm), beats = Number(g.beatsPerBar) || 4; return bpm >= 20 && bpm <= 400 ? { name: g.name || "", bpm, beats, first: Number(g.firstBeat) || 0, bar: beats * 60 / bpm } : null; };
const slClock = t => Math.floor(t / 60) + ":" + (t % 60).toFixed(1).padStart(4, "0");
// LoRAs a shot can take: the hub's installed ones that go on a clip from text or a frame
const slShotLoras = () => (typeof LORAS === "undefined" ? [] : LORAS).filter(r => r.installed && r.usable !== false && r.target !== "image" && (r.status || "ready") === "ready" && (r.needs || "text") === "text");
const slBlankShot = name => { const d = SL.p.defaults; return { id: slId("shot"), name, intent: "", spec: { durationSec: d.durationSec, fps: d.fps, aspectRatio: d.aspectRatio, lens: null, movement: null, size: null, angle: null },
  prompt: "", lockedLines: [], mutedLines: [], beatSheet: null, targetModel: "minimax-h3", maxChars: null, variants: [], history: [], takes: [], createdAt: slNow(), updatedAt: slNow() }; };
const slKeep = (shot, label) => { if (!shot.prompt.trim()) return; shot.history.unshift({ id: slId("v"), savedAt: slNow(), label, prompt: shot.prompt }); shot.history = shot.history.slice(0, 50); };
const slSetPrompt = (shot, text, label) => { slKeep(shot, label); shot.prompt = text.trim(); shot.updatedAt = slNow(); slTf(shot).h3Auto = true; };
const slDown = (name, text, type = "text/plain") => { const a = el("a", { href: URL.createObjectURL(new Blob([text], { type })), download: name }); document.body.append(a); a.click(); a.remove(); };
const slArea = (obj, key, attrs = {}, after) => bind(obj, key, el("textarea", { style: "min-height:60px", ...attrs }), () => { slSave(); if (after) after(); });
const slIn = (obj, key, attrs = {}, after) => bind(obj, key, el("input", attrs), () => { slSave(); if (after) after(); });
const slPick = (obj, key, options, after) => { if (obj[key] == null) obj[key] = ""; const n = select(obj, key, options, () => { if (obj[key] === "") obj[key] = null; slSave(); if (after) after(); }); if (obj[key] === "") obj[key] = null; return n; };
const slOpts = (list, none = "—") => ({ "": none, ...Object.fromEntries(list.map(v => [v, v])) });

// ---- the brain: Slate's prompts, sent through this Studio's server ----
const SL_FORMAT = `PROMPT FORMAT — the app's editor uses sectioned prompts. Write prompts exactly in this shape (only include sections that have content, keep this order):

# Subject
<who/what the shot is about, action, wardrobe — concrete and visual>

# Composition
<framing, shot size, blocking, foreground/background layers, negative space>

# Lighting
<key quality/direction, practicals, color temperature, contrast, atmosphere>

# Camera
<lens focal length, camera movement, perspective height, depth of field, shutter feel>

# Style
<medium, film stock or digital character, grade, era, texture>

# Mood
<the emotional register, rendered as visual language>`;
const SL_SYSTEM = `You are the prompt-craft brain inside Slate, a professional prompt studio for filmmakers who generate images and video with AI models. You are an expert cinematographer, director, and prompt engineer. You write vivid, concrete, production-ready visual language — never vague adjectives, never filler like "high quality, 8k, masterpiece". Physical, photographable descriptions only. Respect every continuity fact you are given about characters, wardrobe, props, locations, lighting state, and time of day; if the user's request conflicts with continuity, follow the request and flag the conflict in one short line at the end prefixed with "CONTINUITY:".

${SL_FORMAT}`;
const slSpec = shot => { const s = shot.spec; return [shot.tf && shot.tf.bars && slSong() ? shot.tf.bars + " bars" : s.durationSec && s.durationSec + "s", s.fps && s.fps + "fps", s.aspectRatio, s.size, s.angle && s.angle + " angle", s.lens, s.movement].filter(Boolean).join(", ") || "unspecified"; };
const slProfile = () => SL.data && SL.data.profiles.find(m => m.id === "minimax-h3");
const slHint = () => { const prof = slProfile(); return prof ? `\nTARGET GENERATOR: ${prof.label} (${prof.kind}). Budget: ${prof.limits.maxChars} chars. Keep phrasing compatible with its dialect: ${prof.dialect.guidance.slice(0, 260)}…` : ""; };
function slContext(scene) { const p = SL.p, out = ["PROJECT: " + p.name];
  if (p.logline) out.push("Logline: " + p.logline); if (p.world) out.push("World & tone: " + p.world);
  if (p.characters.length) { out.push("\nCHARACTERS (keep these descriptions verbatim-consistent in every prompt):"); for (const c of p.characters) out.push(`- ${c.name}: ${[c.age, c.gender, c.ethnicity, c.faceFeatures, c.hair, c.clothing].filter(Boolean).join("; ")}`); }
  if (p.locations.length) { out.push("\nLOCATIONS:"); for (const l of p.locations) out.push(`- ${l.name} (${l.interiorExterior}): ${l.description}${l.timeOfDay ? "; " + l.timeOfDay : ""}`); }
  if (p.artDept.length) { out.push("\nPROPS / WARDROBE / VEHICLES:"); for (const a of p.artDept) out.push(`- ${a.name} (${a.kind}): ${a.description}${a.distinctive ? "; distinctive: " + a.distinctive : ""}`); }
  if (p.lookbook.length) { out.push("\nACTIVE STYLE PROFILES (apply their language where relevant):"); for (const s of p.lookbook) out.push(`- ${s.source}: tone ${s.tone}; palette ${s.palette}; lighting ${s.lighting}; lenses ${s.lensLanguage}; movement ${s.movement}`); }
  if ((p.music || []).length) { out.push("\nSCORE (music cues):"); for (const m of p.music) out.push(`- ${m.name}${m.sceneRef ? ` (${m.sceneRef})` : ""}: ${[m.genre, m.mood].filter(Boolean).join(", ")}`); }
  if ((p.voices || []).length) { out.push("\nVOICE CAST:"); for (const v of p.voices) out.push(`- ${v.name}: ${[v.ageGender, v.accent, v.timbre].filter(Boolean).join(", ")}`); }
  const song = slSong(); if (song) out.push(`\nSONG${song.name ? ' "' + song.name + '"' : ""}: ${song.bpm} BPM, ${song.beats} beats to the bar, one bar is ${song.bar.toFixed(3)} s; bar 1 starts ${song.first} s into the song. Shots are cut on bar lines: give each shot a slot in "bars".`);
  if (scene) { out.push(`\nCURRENT SCENE: ${scene.name}${scene.synopsis ? " — " + scene.synopsis : ""}`);
    if (scene.shots.length) { out.push("Shots so far in this scene (for continuity):"); for (const sh of scene.shots) out.push(`- ${sh.name} [${slSpec(sh)}]: ${sh.intent || sh.prompt.slice(0, 110).replace(/\n/g, " ") || "empty"}`); } }
  return out.join("\n"); }
const slLocks = shot => { const lines = shot.prompt.split("\n"), locked = (shot.lockedLines || []).filter(n => n >= 1 && n <= lines.length).map(n => n + ": " + lines[n - 1]);
  return locked.length ? "\n\nPICTURE-LOCKED LINES — reproduce these lines EXACTLY, character for character, in your output:\n" + locked.join("\n") : ""; };
async function slBrain(task, prompt, { tier = "top", json = false, system = SL_SYSTEM } = {}) {
  if (SL.busy) { toast("The brain is still on: " + SL.busy); return null; }
  SL.busy = task; render(); let r = null;
  try { const d = SL.p.defaults; r = await api("/api/slate/brain", { backend: d.brain, system, prompt, tier, expectJson: json, localEndpoint: d.localEndpoint, localModel: d.localModel }); } catch (e) { r = { ok: false, error: e.message }; }
  SL.busy = ""; if (!r.ok) { toast("Brain: " + (r.error || "no answer")); render(); return null; } return r; }
const SL_TRANSFORMS = { structure: ["Structure", "standard", "Reorganize this prompt into the sectioned format. Preserve every concrete detail; move each to its correct section; do not invent new content beyond minimal connective phrasing."],
  tighten: ["Tighten", "standard", "Shorten this prompt by roughly a third. Remove redundancy and weak modifiers first. Keep every concrete visual fact. Keep the section structure."],
  enrich: ["Enrich", "top", "Deepen this prompt with specific, professional cinematographic detail — lens behavior, light quality, texture, atmosphere — without changing its intent or subject. Stay disciplined: enrich, do not pad."],
  distill: ["Distill", "standard", "Strip this prompt to its essential core — the single strongest version of the idea in the fewest words that still carry all critical visual information. Keep the section structure with only the sections that matter."],
  shot: ["Shot", "top", "Rewrite the Composition and Camera sections to make this a stronger, more intentional shot: clarify shot size, framing logic and blocking. Leave other sections untouched unless they contradict the new framing."],
  angle: ["Angle", "top", "Propose a more dramatic camera angle and perspective for this shot and rewrite the Camera section (and Composition where needed) accordingly. Keep subject and continuity identical."] };
const slFence = text => { const m = text.match(/```(?:prompt)?\n([\s\S]*?)```/); return m ? m[1].trim() : null; };
async function slTransform(shot, kind) { const [label, tier, text] = SL_TRANSFORMS[kind];
  const r = await slBrain(label, `${slContext(slScene())}\n\nSHOT SPEC: ${slSpec(shot)}${slHint()}\n\nTASK: ${text}${slLocks(shot)}\n\nCURRENT PROMPT:\n${shot.prompt}\n\nReturn ONLY the rewritten prompt in the sectioned format — no commentary.`, { tier });
  if (r) { slSetPrompt(shot, slFence(r.text) || r.text, "before " + label); slDraw(); } }
async function slNote(shot, note) { const r = await slBrain("Director's note", `${slContext(slScene())}\n\nCURRENT SHOT "${shot.name}" [${slSpec(shot)}]:\n${shot.prompt || "(empty)"}${slLocks(shot)}\n\nDIRECTOR'S NOTE: ${note}\n\nRespond as a sharp, collaborative creative partner. If the note asks for prompt changes, include the FULL updated prompt in the sectioned format inside a fenced code block (\`\`\`prompt ... \`\`\`). If it's a question or creative discussion, answer concisely without a code block. Never lose locked lines.`);
  if (!r) return; const block = slFence(r.text); if (block) slSetPrompt(shot, block, "before note: " + note.slice(0, 40)); SL.reply = r.text.replace(/```(?:prompt)?\n[\s\S]*?```/, "").trim() || "Done: the prompt is updated; the old one is in History."; slDraw(); }
async function slBeats(shot) { const dur = shot.spec.durationSec || SL.p.defaults.durationSec;
  const r = await slBrain("Beat sheet", `${slContext(slScene())}\n\nSHOT (${dur}s) PROMPT:\n${shot.prompt}\n\nBreak this shot into 2-5 timecoded beats that a video model can follow — motivated micro-actions and camera evolution, not arbitrary slices. Return JSON: {"beats":[{"from":0,"to":3,"text":"<what happens and how the camera behaves>"}]} covering 0 to ${dur} seconds with no gaps.`, { json: true });
  if (r && Array.isArray(r.json.beats)) { shot.beatSheet = r.json.beats; slTf(shot).h3Auto = true; slDraw(); } }
async function slVariants(shot) { const r = await slBrain("Variants", `${slContext(slScene())}\n\nCURRENT PROMPT:\n${shot.prompt}${slLocks(shot)}\n\nWrite 3 distinct variants of this prompt, each weighting the shot differently (e.g. one composition-forward, one lighting-forward, one motion-forward, one mood-forward). Same subject, same continuity, same spec — different emphasis and language. Return JSON: {"variants":[{"label":"<2-4 word handle>","prompt":"<full sectioned prompt>"}]}`, { json: true });
  if (r && Array.isArray(r.json.variants)) { shot.variants = r.json.variants.map(v => ({ id: slId("var"), label: String(v.label || "Variant"), prompt: String(v.prompt || "") })); slDraw(); } }
async function slCoverage(scene) { const plan = SL.cover.plan === "custom" ? null : SL.data.plans.find(p => p.id === SL.cover.plan); const text = SL.cover.text.trim() || scene.synopsis;
  if (!text) return toast("Describe the scene to cover first"); if (!plan && !SL.cover.ask.trim()) return toast("Say what coverage you want");
  const planText = plan ? `COVERAGE PLAN "${plan.label}" — create exactly these shots:\n${plan.shots.map((s, i) => `${i + 1}. ${s.slot}: ${s.size}, ${s.angle}, ${s.movement}. Intent: ${s.intent} Guidance: ${s.promptGuidance}`).join("\n")}`
    : `The director asked for this coverage: ${SL.cover.ask}\nDesign the right set of shots (4-8) a seasoned director would want, varied in size, angle and movement.`;
  const r = await slBrain("Coverage", `${slContext(scene)}\n\nSCENE TO COVER:\n${text}\n\n${planText}${slHint()}\n\nFor EVERY shot write a complete sectioned prompt, fully self-contained (a generator sees one prompt at a time — repeat character/location descriptions verbatim in each). Return JSON: {"shots":[{"name":"<slot name>","intent":"<one line>","size":"<size>","angle":"<angle>","movement":"<movement>","prompt":"<full sectioned prompt>"}]}`, { json: true });
  if (!r || !Array.isArray(r.json.shots)) return; for (const s of r.json.shots) { const shot = slBlankShot(String(s.name || "Shot")); Object.assign(shot, { intent: String(s.intent || ""), prompt: String(s.prompt || "") });
    Object.assign(shot.spec, { size: s.size || null, angle: s.angle || null, movement: s.movement || null }); scene.shots.push(shot); SL.shotId = shot.id; }
  if (!scene.synopsis) scene.synopsis = text; toast(r.json.shots.length + " shots laid out"); slDraw(); }
async function slContinuity(scene) { const shots = scene.shots.map(s => `--- ${s.name} ---\n${s.prompt}`).join("\n\n");
  const r = await slBrain("Continuity check", `${slContext()}\n\nSCENE "${scene.name}" — ALL SHOT PROMPTS IN ORDER:\n\n${shots}\n\nAudit continuity across these prompts like a script supervisor: character descriptions, wardrobe, props, lighting state, time of day, weather, geography/eyelines. Return JSON: {"issues":[{"severity":"error"|"warn","shots":["<shot names>"],"issue":"<what mismatches>","fix":"<concrete fix>"}]} — empty array if clean.`, { json: true });
  if (r) { SL.issues = { scene: scene.id, list: r.json.issues || [] }; render(); } }

// ---- the First AD: Slate's conversational operator and its actions ----
const SL_AD = `You are the First AD inside Slate, a prompt studio for AI filmmaking. The director talks to you about what they want; you talk it through with them like a sharp, experienced first assistant director — then you run the set.

RESPONSE FORMAT — always return exactly this JSON, nothing else:
{"reply":"<what you say to the director — concise, collaborative, concrete>","actions":[...]}

CONVERSATION DISCIPLINE:
- If the intent is vague, ask the one or two questions that matter most (duration? tone? coverage style?) and return "actions": [].
- Don't interrogate — after at most one round of questions, make strong professional assumptions, state them briefly in the reply, and ACT.
- When you act, the reply should read like an AD reporting back: what you set up and why, in a few tight sentences. Never paste full prompts into the reply — they're in the shots.
- When the director asks for changes, use update_shot/update_scene on the existing work, not duplicates.
- To change a character, location, prop, music cue or voice that is already in the bible (a recast, a new wardrobe, another time of day), use update_character/update_location/update_art/update_music_cue/update_voice with only the fields that change; send "replace":true to rewrite the whole sheet. Never add a second sheet with the same name. Shots that name the sheet in "cast" or "location" pick the change up when they compile; rewrite a shot's own "prompt" (update_shot) only where its text repeats the old description.

PROMPT QUALITY: every "prompt" you write uses the sectioned format below, in vivid, concrete, production-ready cinematography language — no vague adjectives, no "8k masterpiece" filler. Repeat character/location descriptions verbatim from the bible in every prompt (generators see one prompt at a time). Respect all continuity facts.

# Subject
...
# Composition
...
# Lighting
...
# Camera
...
# Style
...
# Mood
...

(Only include sections with content, keep this order.)

THIS SLATE RENDERS ITS OWN SHOTS: it runs inside TensorFold Studio on MiniMax H3, video with sound in one pass. A shot is 1 to 15 seconds, 24 fps (5, 8, 10 and 15 are the usual lengths). One or two actions fit in 5 seconds; put each further action in its own shot.
INTENT: a shot's "intent" is a one-line note for the crew (what the shot is for, where it sits in the cut). It never reaches the generator: everything that must be seen goes in the "prompt", the action in its Subject section or in "beatSheet".
SPEECH AND SOUND: never write spoken words inside a "prompt". Every spoken line goes in the shot's "lines" with its speaker, about three words per second of the shot at most; a shot with no "lines" is rendered with nobody speaking. Describe ambience and physical sound in "soundscape".
CAST AND PLACE: add characters and locations BEFORE the shots that use them, then name them in each shot's "cast" and "location" so their sheets are repeated in the compiled prompt.
MUSIC VIDEOS: when the project has a SONG (or the director gives a tempo: set it with update_project "song"), give every shot "bars": its slot on the bar grid (2 bars at 120 BPM is 4 s). The clip is rendered a little longer than the slot and cut on the bar line; shots follow each other in order from bar 1, so the bars of all shots add up to the song. "lora" puts one installed LoRA on a single shot (a camera move, a look): name one from INSTALLED LORAS only, and null takes it off.
"opensOn": "previous" makes a shot start on the last frame of the shot before it (one continuous move, no cut); "text" is a fresh frame, a cut.

ACTIONS — when (and only when) the director's intent is clear enough to act, include actions. Available actions:

- {"type":"update_project","logline"?,"world"?,"defaults"?:{"aspectRatio"?,"durationSec"?},"song"?:{"name"?,"bpm"?:number,"beatsPerBar"?:number,"firstBeat"?:number}}
- {"type":"create_scene","name","synopsis"?}
- {"type":"update_scene","scene":"<name or id>","name"?,"synopsis"?}
- {"type":"create_shot","scene":"<name or id>","name"?,"intent"?,"prompt"?,"spec"?:{"durationSec"?,"aspectRatio"?,"size"?,"angle"?,"lens"?,"movement"?},"beatSheet"?:[{"from","to","text"}],"lines"?:[{"speaker":"<character name>","text":"<the exact words>"}],"soundscape"?,"cast"?:["<character names>"],"location"?:"<location name>","opensOn"?:("text"|"previous"),"bars"?:number,"lora"?:{"name":"<an installed LoRA>","strength"?:number,"style"?:"<one of its looks>"}|null}
- {"type":"update_shot","shot":"<name or id>","name"?,"intent"?,"prompt"?,"spec"?,"beatSheet"?,"lines"?,"soundscape"?,"cast"?,"location"?,"opensOn"?,"bars"?,"lora"?}
- {"type":"add_character","name","age"?,"gender"?,"ethnicity"?,"faceFeatures"?,"hair"?,"clothing"?,"expression"?,"eyeDirection"?,"mood"?,"environment"?,"keyLightSide"?,"lightingMood"?}
- {"type":"add_location","name","interiorExterior"?("interior"|"exterior"|"both"),"description"?,"timeOfDay"?,"weather"?,"architecture"?,"textures"?,"practicalLights"?}
- {"type":"add_art","kind"("prop"|"wardrobe"|"vehicle"),"name","description"?,"materials"?,"condition"?,"era"?,"distinctive"?}
- {"type":"add_music_cue","name","sceneRef"?,"intent"?,"genre"?,"mood"?,"tempo"?,"instrumentation"?,"era"?,"structure"?,"vocals"?("instrumental"|"vocals"|"either"),"lyricTheme"?,"durationSec"?:number}
- {"type":"add_voice","name","characterName"?,"ageGender"?,"accent"?,"timbre"?,"pitch"?,"pacing"?,"energy"?,"texture"?,"emotionalRange"?,"sampleLine"?}
- {"type":"update_character","character":"<name or id>","name"?:"<new name>","replace"?:true, ...any add_character field}
- {"type":"update_location","location":"<name or id>","name"?,"replace"?:true, ...any add_location field}
- {"type":"update_art","art":"<name or id>","name"?,"replace"?:true, ...any add_art field}
- {"type":"update_music_cue","cue":"<name or id>","name"?,"replace"?:true, ...any add_music_cue field}
- {"type":"update_voice","voice":"<name or id>","name"?,"replace"?:true, ...any add_voice field}
- {"type":"select","scene"?:"<name or id>","shot"?:"<name or id>"} — focus the UI on something you created or changed

Spec vocabulary: size EWS|WS|MWS|MS|MCU|CU|ECU|Insert; angle eye level|low|high|overhead|dutch|over-shoulder|POV; movement locked|handheld|push-in|pull-back|orbit|pan|tilt|track|crane|drone|steadicam; lens 14mm|24mm|35mm|50mm|85mm|135mm|anamorphic|macro|tilt-shift.`;
// the bible's sheets as the First AD writes them: `read` gives every field (`all`) and those this action sent (`sent`)
const slSheetRead = (fields, fix = {}) => (a, str, p) => { const all = {}, sent = {}; for (const k of fields) { const v = fix[k] ? fix[k](a[k], str, p, a) : str(a[k]); all[k] = v; if (a[k] !== undefined && a[k] !== null) sent[k] = v; } return { all, sent }; };
const SL_BIBLE = {
  add_character: { key: "characters", ref: "character", id: "char", noun: "character", added: "Cast", unnamed: "Unnamed", blank: () => ({ scenario: "cinematic", notes: "" }),
    read: slSheetRead(["age", "gender", "ethnicity", "faceFeatures", "hair", "clothing", "expression", "eyeDirection", "mood", "environment", "keyLightSide", "lightingMood"], { keyLightSide: (v, str) => str(v) || "Key light from left", lightingMood: (v, str) => str(v) || "Natural soft light" }) },
  add_location: { key: "locations", ref: "location", id: "loc", noun: "location", added: "Scouted", unnamed: "Unnamed", blank: () => ({ notes: "" }),
    read: slSheetRead(["interiorExterior", "description", "timeOfDay", "weather", "architecture", "textures", "practicalLights"], { interiorExterior: (v, str) => ["interior", "both"].includes(str(v)) ? str(v) : "exterior" }) },
  add_art: { key: "artDept", ref: "art", id: "art", noun: "prop, wardrobe or vehicle", added: "Added", unnamed: "Unnamed", blank: () => ({ notes: "" }),
    read: slSheetRead(["kind", "description", "materials", "condition", "era", "distinctive"], { kind: (v, str) => ["wardrobe", "vehicle"].includes(str(v)) ? str(v) : "prop" }) },
  add_music_cue: { key: "music", ref: "cue", id: "cue", noun: "music cue", added: "Spotted cue", unnamed: "Untitled cue", blank: () => ({ lyrics: "", notes: "" }),
    read: slSheetRead(["sceneRef", "intent", "genre", "mood", "tempo", "instrumentation", "era", "structure", "vocals", "lyricTheme", "durationSec"], { vocals: (v, str) => ["vocals", "either"].includes(str(v)) ? str(v) : "instrumental", durationSec: v => typeof v === "number" ? v : null }) },
  add_voice: { key: "voices", ref: "voice", id: "voice", noun: "voice", added: "Cast voice", unnamed: "Unnamed voice", blank: () => ({ characterId: null, notes: "" }),
    read: (a, str, p) => { const out = slSheetRead(["ageGender", "accent", "timbre", "pitch", "pacing", "energy", "texture", "emotionalRange", "sampleLine"])(a, str, p);
      if (a.characterName !== undefined) { const who = p.characters.find(c => c.name.toLowerCase() === str(a.characterName).toLowerCase()); out.all.characterId = out.sent.characterId = who ? who.id : null; } return out; } } };
const SL_BIBLE_CHANGE = { update_character: "add_character", update_location: "add_location", update_art: "add_art", update_music_cue: "add_music_cue", update_voice: "add_voice" };
function slApply(actions) { const p = SL.p, receipts = [], str = v => typeof v === "string" ? v : "";
  const scene = ref => p.scenes.find(s => s.id === ref) || p.scenes.find(s => s.name.toLowerCase() === String(ref || "").trim().toLowerCase());
  const shotOf = ref => { for (const sc of p.scenes) { const sh = sc.shots.find(s => s.id === ref) || sc.shots.find(s => s.name.toLowerCase() === String(ref || "").trim().toLowerCase()); if (sh) return [sc, sh]; } };
  const named = (list, name) => list.find(x => (x.name || "").toLowerCase() === String(name || "").trim().toLowerCase());
  const gear = (shot, a) => { const tf = slTf(shot); if (Array.isArray(a.cast)) tf.cast = a.cast.map(n => named(p.characters, n)).filter(Boolean).map(c => c.id); if (a.location !== undefined) tf.location = (named(p.locations, a.location) || {}).id || "";
    if (Array.isArray(a.lines)) tf.lines = a.lines.filter(l => l && str(l.text).trim()).map(l => ({ characterId: (named(p.characters, l.speaker) || {}).id || null, voiceId: null, text: str(l.text).trim(), language: str(l.language) || "English", sung: !!l.sung }));
    if (a.soundscape !== undefined) tf.soundscape = str(a.soundscape); if (["text", "previous"].includes(a.opensOn)) tf.start = a.opensOn; tf.h3Auto = true;
    if (a.bars !== undefined) { const n = Number(a.bars); if (n > 0 && n <= 64) tf.bars = n; else delete tf.bars; }
    if (a.lora !== undefined) { const ask = a.lora && String(a.lora.name || a.lora.id || "").trim().toLowerCase(), hit = ask && slShotLoras().find(r => r.id === ask || (r.name || "").toLowerCase() === ask);
      if (!a.lora) tf.loras = []; else if (hit) tf.loras = [{ id: hit.id, strength: Number(a.lora.strength) || hit.strength || 1, style: str(a.lora.style) && (hit.styles || {})[a.lora.style] !== undefined ? a.lora.style : "" }];
      else receipts.push(`• No installed LoRA called "${a.lora.name || a.lora.id}" — the shot keeps ${tf.loras && tf.loras.length ? "its LoRA" : "the film's setting"}`); } };
  const spec = (to, patch) => { for (const k of ["durationSec", "fps", "aspectRatio", "size", "angle", "lens", "movement"]) if (patch && k in patch) to[k] = patch[k] ?? null; };
  for (const a of Array.isArray(actions) ? actions : []) try {
    if (a.type === "update_project") { if (a.logline !== undefined) p.logline = a.logline; if (a.world !== undefined) p.world = a.world; if (a.defaults) Object.assign(p.defaults, a.defaults, { targetModel: "minimax-h3" });
      if (a.song && typeof a.song === "object") { const g = slFilm().song || (slFilm().song = {}); for (const k of ["bpm", "beatsPerBar", "firstBeat"]) if (Number.isFinite(Number(a.song[k])) && a.song[k] !== null && a.song[k] !== "") g[k] = Number(a.song[k]); if (a.song.name !== undefined) g.name = str(a.song.name); SL.plan = {}; receipts.push(slSong() ? `✓ Song: ${slSong().bpm} BPM, a bar is ${slSong().bar.toFixed(2)} s` : "• Song noted, but it has no tempo between 20 and 400 BPM yet"); }
      receipts.push("✓ Updated project bible"); }
    else if (a.type === "create_scene") { if (scene(a.name)) { receipts.push(`• Scene "${a.name}" already exists — skipped`); continue; } p.scenes.push({ id: slId("scene"), name: a.name, synopsis: a.synopsis || "", shots: [] }); SL.sceneId = p.scenes.at(-1).id; SL.shotId = null; receipts.push(`✓ Created scene "${a.name}"`); }
    else if (a.type === "update_scene") { const sc = scene(a.scene); if (!sc) { receipts.push(`✗ Scene "${a.scene}" not found`); continue; } if (a.name) sc.name = a.name; if (a.synopsis !== undefined) sc.synopsis = a.synopsis; receipts.push(`✓ Updated scene "${sc.name}"`); }
    else if (a.type === "create_shot") { const sc = scene(a.scene); if (!sc) { receipts.push(`✗ Scene "${a.scene}" not found for new shot`); continue; }
      const shot = slBlankShot(a.name || "Shot " + String(sc.shots.length + 1).padStart(2, "0")); if (a.intent) shot.intent = a.intent; if (a.prompt) shot.prompt = a.prompt; spec(shot.spec, a.spec); if (a.beatSheet) shot.beatSheet = a.beatSheet; gear(shot, a);
      sc.shots.push(shot); SL.sceneId = sc.id; SL.shotId = shot.id; receipts.push(`✓ Created "${shot.name}" in "${sc.name}"${a.prompt ? " with prompt" : ""}`); }
    else if (a.type === "update_shot") { const hit = shotOf(a.shot); if (!hit) { receipts.push(`✗ Shot "${a.shot}" not found`); continue; } const [sc, shot] = hit;
      if (a.prompt !== undefined && a.prompt !== shot.prompt) { slKeep(shot, "before First AD change"); shot.prompt = a.prompt; slTf(shot).h3Auto = true; } if (a.name) shot.name = a.name; if (a.intent !== undefined) shot.intent = a.intent; spec(shot.spec, a.spec);
      if (a.beatSheet !== undefined) shot.beatSheet = a.beatSheet; gear(shot, a); delete SL.plan[shot.id]; shot.updatedAt = slNow(); SL.sceneId = sc.id; SL.shotId = shot.id; receipts.push(`✓ Updated "${shot.name}"`); }
    else if (SL_BIBLE[a.type] || SL_BIBLE_CHANGE[a.type]) { const adding = !!SL_BIBLE[a.type], def = SL_BIBLE[a.type] || SL_BIBLE[SL_BIBLE_CHANGE[a.type]], list = p[def.key] || (p[def.key] = []);
      // an add with a name the bible has rewrites that sheet and an update changes its given fields; the id stays
      const ref = adding ? a.name : a[def.ref], hit = list.find(x => x.id === ref) || named(list, ref);
      if (!adding && !hit) { receipts.push(`✗ ${def.noun} "${ref}" not found — nothing changed`); continue; }
      const given = def.read(a, str, p);
      if (!hit) { list.push({ id: slId(def.id), ...def.blank(), ...given.all, name: str(a.name) || def.unnamed }); receipts.push(`✓ ${def.added} "${str(a.name)}"`); continue; }
      const whole = adding || a.replace === true, was = hit.name;
      Object.assign(hit, whole ? given.all : given.sent, { name: (str(a.name).trim() && (adding ? hit.name : str(a.name).trim())) || hit.name });
      SL.plan = {}; receipts.push(`✓ ${whole ? "Rewrote" : "Changed"} ${def.noun} "${was}"${hit.name !== was ? ` → "${hit.name}"` : ""}${whole ? "" : Object.keys(given.sent).length ? ": " + Object.keys(given.sent).join(", ") : hit.name !== was ? "" : ": nothing to change"}`); }
    else if (a.type === "select") { const hit = a.shot && shotOf(a.shot), sc = !hit && a.scene && scene(a.scene); if (hit) { SL.sceneId = hit[0].id; SL.shotId = hit[1].id; } else if (sc) { SL.sceneId = sc.id; SL.shotId = null; } }
    else receipts.push("✗ Unknown action " + (a.type || "?"));
  } catch (e) { receipts.push(`✗ ${a.type} failed: ${e.message}`); }
  return receipts; }
async function slAD(message) { const p = SL.p, log = p.copilot || (p.copilot = []);
  const inv = ["CURRENT SCENE/SHOT INVENTORY (ids in brackets):", ...(p.scenes.length ? [] : ["- (no scenes yet)"]), ...p.scenes.flatMap(sc => [`- Scene "${sc.name}" [${sc.id}]${sc.synopsis ? " — " + sc.synopsis : ""}`,
    ...sc.shots.map(sh => `    · Shot "${sh.name}" [${sh.id}] ${[sh.spec.durationSec && sh.spec.durationSec + "s", sh.spec.size, sh.spec.movement].filter(Boolean).join(", ")} — ${sh.intent || sh.prompt.slice(0, 80).replace(/\n/g, " ") || "empty"}`)])].join("\n");
  const loras = slShotLoras(); const invLoras = "\n\nINSTALLED LORAS (for a shot's \"lora\"): " + (loras.length ? loras.map(r => r.name + (Object.keys(r.styles || {}).length ? " [" + Object.keys(r.styles).join(", ") + "]" : "")).join("; ") : "(none)");
  const convo = log.slice(-14).map(m => `${m.role === "user" ? "DIRECTOR" : "FIRST AD"}: ${m.text}`).join("\n\n");
  log.push({ role: "user", text: message }); slSave();
  const r = await slBrain("First AD", `${slContext()}\n\n${inv}${invLoras}\n\n${convo ? "CONVERSATION SO FAR:\n" + convo + "\n\n" : ""}DIRECTOR: ${message}`, { json: true, system: SL_AD });
  if (!r) { log.pop(); return slDraw(); } const receipts = slApply(r.json.actions); log.push({ role: "assistant", text: String(r.json.reply || ""), receipts }); p.copilot = log.slice(-60); slDraw(); }

// ---- from a shot to the queue ----
async function slCompile(shot, chain) { const r = await api("/api/slate/compile", { project: SL.p, shotId: shot.id, chain }); SL.plan[shot.id] = r; const tf = slTf(shot); if (tf.h3Auto !== false || !tf.h3.trim()) { tf.h3 = r.prompt; tf.h3Auto = true; } return r; }
async function slBrainCompile(shot) { const base = await slCompile(shot), prof = slProfile(), tf = slTf(shot);
  const r = await slBrain("Compile for MiniMax H3", `${slContext(slScene())}\n\nSOURCE PROMPT (sectioned):\n${shot.prompt}\n\nSHOT SPEC: ${slSpec(shot)}\n\nTARGET: ${prof.label} (${prof.kind}). Dialect: ${prof.dialect.style}.\nDIALECT GUIDANCE: ${prof.dialect.guidance}\nHARD CHARACTER BUDGET: ${shot.maxChars || prof.limits.maxChars} characters.${slLocks(shot)}\n\nDRAFT ALREADY IN THE RIGHT FIELD STRUCTURE FOR THIS LOCAL ENGINE:\n${base.prompt}\n\nRewrite the draft into the strongest prompt for this model. Keep its structure exactly: any first instruction line, every field label, every <Subject N> and <Picture N> label, and every dialogue line with its (Sx) tag and <d>[Language] ...</d> words unchanged. Keep it one continuous [Shot 1] unless the source has timecoded beats. Improve only the prose: physical specifics of what moves, in which direction, lit by what; the official camera vocabulary; a concrete soundscape. Do not add speech.\n\nReturn JSON: {"prompt":"<the compiled prompt>"}`, { tier: "standard", json: true });
  if (r && typeof r.json.prompt === "string" && r.json.prompt.trim()) { tf.h3 = r.json.prompt.trim(); tf.h3Auto = false; } slDraw(); }
async function slRender(scene, shot, chain) { const tf = slTf(shot), c = await slCompile(shot, chain), stamp = Date.now().toString(36);
  const output = "slate/" + SL.p.id + "/takes/" + shot.id + "-" + stamp + ".mp4", prompt = (tf.h3Auto === false && tf.h3.trim()) || c.prompt;
  const j = await api("/api/jobs", { kind: "video", title: scene.name + " · " + shot.name, params: { ...c.plan, prompt, output, ...(!(c.plan.references || []).length && (lorasOf(c.plan).length || lorasOf(slFilm()).length) ? { loras: lorasOf(c.plan).length ? lorasOf(c.plan) : lorasOf(slFilm()) } : {}), brain: { backend: SL.p.defaults.brain, endpoint: SL.p.defaults.localEndpoint, model: SL.p.defaults.localModel }, slate: { project: SL.p.id, scene: scene.id, shot: shot.id } } });
  shot.takes.unshift({ id: slId("take"), loggedAt: slNow(), model: "minimax-h3", prompt, rating: "good", notes: "", mediaPath: output, jobId: j.id, seed: c.plan.seed, settings: Object.fromEntries(["preset", "quality", "frames", "seconds", "width", "height", "loras"].filter(k => j.params && j.params[k] != null).map(k => [k, j.params[k]])) }); return output; }
// A take rendered somewhere else (another machine's app, or a file) logged into this shot with its seed and settings.
async function slImportTake(shot, given) { const body = { projectId: SL.p.id, shot: shot.id, ...Object.fromEntries(Object.entries(given).filter(([, v]) => v !== "" && v != null)) };
  clearTimeout(slTimer); await api("/api/slate/projects/" + SL.p.id, SL.p, "PUT");  // the film as it is now: the server reads the shot from the file
  const r = await api("/api/slate/takes", body); shot.takes.unshift(r.take); SL.have[r.take.mediaPath] = true; return r.take; }
const slTakeLine = take => [take.seed != null ? "seed " + take.seed : "seed ?", take.settings && (take.settings.preset || take.settings.width) ? (take.settings.width ? take.settings.width + "×" + take.settings.height : take.settings.preset) : null,
  take.settings && take.settings.frames ? take.settings.frames + " frames" : take.settings && take.settings.seconds ? take.settings.seconds + " s" : null, take.settings && take.settings.quality, take.settings && (take.settings.loras || []).length ? "LoRA " + take.settings.loras.map(l => l.id).join(" + ") : null, typeof spkTake === "function" ? spkTake(take.mediaPath) : null,
  take.imported ? "imported" + (take.source && take.source.from ? " from " + take.source.from.replace(/^https?:\/\//, "") : "") : null, take.loggedAt.slice(0, 16).replace("T", " ")].filter(Boolean).join(" · ");
function slImport(scene, shot) { const f = SL.imp || (SL.imp = { from: "", clip: "", seed: "", notes: "" }); try { if (!f.from) f.from = localStorage.getItem("slate-import-from") || ""; } catch {}
  const done = take => { toast("Imported a take of " + shot.name + (take.seed != null ? ", seed " + take.seed : "")); f.clip = ""; f.seed = ""; f.notes = ""; slDraw(); };
  const up = el("input", { type: "file", accept: "video/mp4,video/quicktime", onchange: async () => { try { const data = new FormData(); data.append("file", up.files[0]); const j = await (await fetch("/api/upload", { method: "POST", body: data })).json(); if (!j.path) return toast(j.error || "upload failed");
    done(await slImportTake(shot, { path: j.path, seed: f.seed, notes: f.notes })); } catch (e) { toast(e.message); } } });
  const go = async () => { const clip = f.clip.trim(); if (!f.from.trim() || !clip) return toast("Give the other app's address and one of its job ids, or a clip's path in its folder");
    try { localStorage.setItem("slate-import-from", f.from.trim()); } catch {}
    try { done(await slImportTake(shot, { from: f.from.trim(), ...(/[\/.]/.test(clip) ? { path: clip } : { job: clip }), seed: f.seed, notes: f.notes })); } catch (e) { toast(e.message); } };
  return slFold("import", "Import a take rendered on another machine", () => el("div", {}, el("div", { class: "note" }, "A clip another machine's app made of this shot comes into the film with its seed and settings: give that app's address and the job's id (its prompt, seed and settings are read from the job), or a path in its folder. A file you have goes in by upload."),
    el("div", { class: "row" }, field("The other app's address", bind(f, "from", el("input", { placeholder: "http://192.0.2.10:7870" }))), field("Its job id, or a clip's path", bind(f, "clip", el("input", { placeholder: "1013-131100-5eb633  or  renders/1013-131100-5eb633.mp4" }))),
      el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Seed, if not the job's"), bind(f, "seed", el("input", { type: "number", min: 0 })))),
    el("div", { class: "row" }, field("Notes", bind(f, "notes", el("input", { placeholder: "rendered on Spark 5, the take in the cut" }))), el("button", { class: "primary fit", onclick: go }, "Import the take"), field("or upload the clip", up)))); }
const slBest = shot => { const t = shot.takes.filter(t => t.mediaPath && SL.have[t.mediaPath]); return (t.find(t => t.rating === "circled") || t.find(t => t.rating !== "no-good") || {}).mediaPath || null; };
// a new take of a shot: with notes, the brain first rewrites the prompt to answer them (old one to History)
async function slRegenerate(scene, shot, take) { const tf = slTf(shot), feedback = (take.notes || "").trim();
  if (feedback) { if (!shot.prompt.trim()) return toast("This shot has no sectioned prompt for the brain to revise: write one, or clear the notes to regenerate at a new seed");
    const r = await slBrain("Regenerate from feedback", `${slContext(scene)}\n\nSHOT "${shot.name}" [${slSpec(shot)}] WAS RENDERED FROM THIS PROMPT:\n${shot.prompt}${slLocks(shot)}\n\nTHE DIRECTOR WATCHED THE TAKE AND GAVE THIS FEEDBACK:\n${feedback}\n\nRevise the prompt so the next render fixes exactly what the feedback names. Change only what it asks for; keep everything that was not criticised, word for word where you can. If the feedback is about something the prompt already states, state it earlier and more concretely. Return ONLY the full revised prompt in the sectioned format inside a fenced code block (\`\`\`prompt ... \`\`\`). Never lose locked lines.`);
    const block = r && slFence(r.text); if (!block) return toast(r ? "The brain gave no revised prompt: nothing was queued" : "The brain did not answer: nothing was queued");
    slSetPrompt(shot, block, "before feedback: " + feedback.slice(0, 40)); tf.h3Auto = true; delete SL.plan[shot.id]; }
  tf.seed = Number(take.seed ?? tf.seed ?? 1) + 1; while (shot.takes.some(t => t.seed === tf.seed)) tf.seed++;
  try { const index = scene.shots.indexOf(shot), before = tf.start === "previous" && index > 0 ? slBest(scene.shots[index - 1]) : null;
    await slRender(scene, shot, before); shot.takes[0].notes = feedback ? "Regenerated after feedback: " + feedback : "Regenerated at seed " + tf.seed;
    await poll(); slDraw(); toast("Queued a new take of " + shot.name + (feedback ? ", with the prompt revised from your feedback" : " at seed " + tf.seed)); } catch (e) { toast(e.message); } }
async function slRenderScene(scene, all) { let chain = null, n = 0;
  try { for (const shot of scene.shots) { if (!shot.prompt.trim()) { chain = null; continue; } if (!all && shot.takes.some(t => t.mediaPath && (SL.have[t.mediaPath] || jobs.some(j => j.id === t.jobId && ["queued", "running"].includes(j.status))))) { chain = null; continue; }
    chain = await slRender(scene, shot, chain); n++; } } catch (e) { toast(e.message); }
  toast(n ? n + " shot" + (n === 1 ? "" : "s") + " queued" : "Every shot with a prompt already has a take"); await poll(); slDraw(); }
async function slHave() { const paths = SL.p.scenes.flatMap(sc => sc.shots.flatMap(sh => sh.takes.map(t => t.mediaPath).filter(Boolean))); if (!paths.length) return;
  try { const got = await api("/api/slate/exists", { paths }); if (JSON.stringify(got) !== JSON.stringify(SL.have)) { SL.have = got; render(); } } catch {} }
async function slToEditor() { await ensureProject(); const clips = SL.p.scenes.flatMap(sc => sc.shots.map(slBest).filter(Boolean)); if (!clips.length) return toast("No finished takes yet");
  if (project.timeline.length && !confirm("Replace the " + project.timeline.length + " clips on the timeline of " + project.name + " with these " + clips.length + " takes?")) return;
  project.timeline = []; for (const path of clips) { let duration = 5; try { duration = (await api("/api/probe?path=" + encodeURIComponent(path))).duration || 5; } catch {} project.timeline.push({ path, in: 0, out: 0, duration }); }
  await saveProject(); tab = "Editor"; history.replaceState(null, "", "#Editor"); render(); }
const slMarkdown = scene => { const out = [`# ${SL.p.name} — ${scene.name}`, ""]; if (scene.synopsis) out.push(scene.synopsis, "");
  for (const sh of scene.shots) { const s = sh.spec; out.push("## " + sh.name); if (sh.intent) out.push("*" + sh.intent + "*");
    out.push("- Spec: " + [s.durationSec && s.durationSec + "s", s.fps && s.fps + "fps", s.aspectRatio, s.size, s.angle, s.lens, s.movement].filter(Boolean).join(" · "), "- Target: minimax-h3", "", "```", sh.prompt, "```", "");
    if ((sh.beatSheet || []).length) { out.push("Beats:"); for (const b of sh.beatSheet) out.push(`- ${b.from}-${b.to}s — ${b.text}`); out.push(""); }
    if (sh.tf && sh.tf.h3) out.push("MiniMax H3 prompt:", "", "```", sh.tf.h3, "```", ""); }
  return out.join("\n"); };
const slCsv = scene => { const q = v => '"' + String(v ?? "").replace(/"/g, '""') + '"'; return [["shot", "intent", "duration_s", "fps", "aspect", "size", "angle", "lens", "movement", "target_model", "prompt", "h3_prompt", "take"].map(q).join(","),
  ...scene.shots.map(sh => [sh.name, sh.intent, sh.spec.durationSec, sh.spec.fps, sh.spec.aspectRatio, sh.spec.size, sh.spec.angle, sh.spec.lens, sh.spec.movement, "minimax-h3", sh.prompt, sh.tf ? sh.tf.h3 : "", slBest(sh) || ""].map(q).join(","))].join("\n"); };

// ---- sheets: Casting, Art Department, Locations, Lookbook, Voices, Music ----
const SL_SHEETS = {
  cast: { key: "characters", title: "Casting", one: "character", name: "name", hint: "a tired night-shift nurse in her 40s, kind eyes",
    make: () => ({ id: slId("char"), name: "New character", age: "", gender: "", ethnicity: "", faceFeatures: "", hair: "", clothing: "", expression: "", eyeDirection: "", mood: "", environment: "", keyLightSide: "Key light from left", lightingMood: "Natural soft light", scenario: "cinematic", notes: "", images: [] }),
    fields: [["age", "Age"], ["gender", "Gender"], ["ethnicity", "Ethnicity"], ["faceFeatures", "Face and features", 1], ["hair", "Hair"], ["clothing", "Wardrobe", 1], ["expression", "Expression"], ["eyeDirection", "Eye direction"], ["mood", "Mood"], ["environment", "Environment"], ["keyLightSide", "Key light side"], ["lightingMood", "Lighting mood"], ["notes", "Notes", 1]],
    fill: d => `${slContext()}\n\nThe director described a character: "${d}" (sheet context: cinematic).\n\nProduce a complete, castable person — specific and photographable, no clichés. Return JSON with exactly these string fields: {"name","age","gender","ethnicity","faceFeatures","hair","clothing","expression","eyeDirection","mood","environment","keyLightSide","lightingMood"}. faceFeatures: skin, face shape, distinguishing marks, facial hair. Keep each field a compact comma-separated phrase list like a casting sheet, not sentences.` },
  art: { key: "artDept", title: "Art Department", one: "prop, wardrobe or vehicle", name: "name", hint: "her battered red thermos", picks: { kind: { prop: "Prop", wardrobe: "Wardrobe", vehicle: "Vehicle" } },
    make: () => ({ id: slId("art"), kind: "prop", name: "New item", description: "", materials: "", condition: "", era: "", distinctive: "", notes: "" }),
    fields: [["description", "Description", 1], ["materials", "Materials"], ["condition", "Condition"], ["era", "Era"], ["distinctive", "Distinctive marks"], ["notes", "Notes", 1]],
    fill: d => `${slContext()}\n\nThe director described a prop, wardrobe piece or vehicle: "${d}"\n\nDesign it like an art department lead — specific, physical, photographable. Return JSON with exactly these string fields: {"name","kind"("prop"|"wardrobe"|"vehicle"),"description","materials","condition","era","distinctive"}. Compact phrases, not sentences.` },
  locations: { key: "locations", title: "Locations", one: "location", name: "name", hint: "a 24-hour laundromat in the rain", picks: { interiorExterior: { interior: "Interior", exterior: "Exterior", both: "Both" } },
    make: () => ({ id: slId("loc"), name: "New location", interiorExterior: "interior", description: "", timeOfDay: "", weather: "", architecture: "", textures: "", practicalLights: "", notes: "", images: [] }),
    fields: [["description", "Description", 1], ["timeOfDay", "Time of day"], ["weather", "Weather"], ["architecture", "Architecture"], ["textures", "Textures"], ["practicalLights", "Practical lights"], ["notes", "Notes", 1]],
    fill: d => `${slContext()}\n\nThe director described a location: "${d}"\n\nScout it like a location manager and a cinematographer — specific, physical, photographable. Return JSON with exactly these string fields: {"name","interiorExterior"("interior"|"exterior"|"both"),"description","timeOfDay","weather","architecture","textures","practicalLights"}. description: 1-2 sentences; the rest compact phrases.` },
  lookbook: { key: "lookbook", title: "Lookbook", one: "style profile", name: "source", hint: "name a cinematographer, director, film or series", picks: { kind: { cinematographer: "Cinematographer", director: "Director", film: "Film", series: "Series", custom: "Custom" } },
    make: () => ({ id: slId("style"), source: "New look", kind: "custom", tone: "", palette: "", lighting: "", lensLanguage: "", movement: "", blocking: "", editorial: "", notes: "" }),
    fields: [["tone", "Tone", 1], ["palette", "Palette", 1], ["lighting", "Lighting", 1], ["lensLanguage", "Lens language", 1], ["movement", "Movement", 1], ["blocking", "Blocking", 1], ["editorial", "Editorial", 1], ["notes", "Notes", 1]],
    fill: (d, sheet) => `Study the visual style of the ${sheet.kind === "custom" ? "look" : sheet.kind} "${d}" from your knowledge of cinema. Distill it into reusable prompt language — concrete visual vocabulary, not names (the profile will be inserted into generation prompts, so describe the LOOK, never cite the person/film). Return JSON with string fields: {"tone","palette","lighting","lensLanguage","movement","blocking","editorial","notes"}. Each field: 1-2 sentences of specific, physical description. In "notes" put 2-3 signature tricks worth stealing.`, keepName: true },
  voices: { key: "voices", title: "Voices", one: "voice", name: "name", hint: "gravelly, patient, late 50s, a trace of Lisbon",
    make: () => ({ id: slId("voice"), name: "New voice", characterId: null, ageGender: "", accent: "", timbre: "", pitch: "", pacing: "", energy: "", texture: "", emotionalRange: "", sampleLine: "", notes: "" }),
    fields: [["ageGender", "Age and gender"], ["accent", "Accent"], ["timbre", "Timbre"], ["pitch", "Pitch"], ["pacing", "Pacing"], ["energy", "Energy"], ["texture", "Texture"], ["emotionalRange", "Emotional range"], ["sampleLine", "Sample line", 1], ["notes", "Notes", 1]],
    fill: (d, sheet) => { const who = SL.p.characters.find(c => c.id === sheet.characterId); return `${slContext()}\n\nDesign a voice for ${who ? `the character "${who.name}"` : "this description"}: "${d}"\n\nThink like a voice director casting for film — specific, castable, physical. Return JSON with exactly these string fields: {"name","ageGender","accent","timbre","pitch","pacing","energy","texture","emotionalRange","sampleLine"}. timbre: the core color (warm/reedy/resonant...). texture: grain (gravel, breath, smoke...). sampleLine: one line of test dialogue IN CHARACTER that exercises the voice's range — emotionally alive, 15-25 words.`; } },
  music: { key: "music", title: "Music cues", one: "music cue", name: "name", hint: "a slow, uneasy pulse under the stakeout", picks: { vocals: { instrumental: "Instrumental", vocals: "Vocals", either: "Either" } },
    make: () => ({ id: slId("cue"), name: "New cue", sceneRef: "", intent: "", genre: "", mood: "", tempo: "", instrumentation: "", era: "", structure: "", vocals: "instrumental", lyricTheme: "", lyrics: "", durationSec: null, notes: "" }),
    fields: [["sceneRef", "Scene"], ["intent", "Intent", 1], ["genre", "Genre"], ["mood", "Mood"], ["tempo", "Tempo"], ["instrumentation", "Instrumentation", 1], ["era", "Era"], ["structure", "Structure", 1], ["lyricTheme", "Lyric theme"], ["notes", "Notes", 1]],
    fill: d => `${slContext(slScene())}\n\nThe director described a music cue: "${d}"\n\nDesign the cue like a film composer spotting a scene — concrete musical language, not vibes. Return JSON with exactly these string fields (plus durationSec as number|null): {"name","intent","genre","mood","tempo","instrumentation","era","structure","vocals"("instrumental"|"vocals"|"either"),"lyricTheme","durationSec"}. tempo: feel + rough BPM. structure: how it evolves against the scene. If vocals, lyricTheme is what the lyrics circle — not the lyrics themselves.` } };
const slSheetPicture = c => `${c.name}, ${c.age ? c.age + " years old, " : ""}${[c.gender, c.ethnicity].filter(Boolean).join(", ")}. ${c.faceFeatures}. Hair: ${c.hair}. Wearing ${c.clothing}. ${c.expression}. Character reference portrait: centered head-and-shoulders, facing the camera, plain seamless backdrop, generous headroom, no props. ${c.keyLightSide || "Key light from left"}, ${c.lightingMood || "natural soft light"}, clean catchlights, gentle fill. Photoreal, neutral grade, true-to-skin color, crisp detail for identity reference.`.replace(/\s+\./g, ".").replace(/\.\s*\./g, ".");
function slSheets(kind) { const def = SL_SHEETS[kind], list = SL.p[def.key] || (SL.p[def.key] = []), sheet = list.find(s => s.id === SL.sheet[kind]) || list[0], ask = el("input", { placeholder: def.hint });
  const fill = async () => { const d = ask.value.trim() || (def.keepName ? sheet[def.name] : ""); if (!d) return toast("Describe the " + def.one + " in one line first"); const r = await slBrain("Fill " + def.one, def.fill(d, sheet), { json: true }); if (!r) return;
    for (const [k, v] of Object.entries(r.json)) if (k in sheet && k !== "id" && !(def.keepName && k === def.name) && (typeof v === "string" || typeof v === "number" || v === null)) sheet[k] = v; slDraw(); };
  const body = el("div");
  if (!sheet) put(body, el("div", { class: "note" }, "Nothing here yet. Add a " + def.one + ", describe it in one line, and ✦ Fill writes the sheet."));
  else { put(body, el("div", { class: "row" }, field(def.name === "source" ? "Source" : "Name", slIn(sheet, def.name, {}, () => {})), ...Object.entries(def.picks || {}).map(([k, options]) => el("div", { class: "fit", style: "width:170px" }, el("label", {}, k === "interiorExterior" ? "Interior / exterior" : k[0].toUpperCase() + k.slice(1)), slPick(sheet, k, options))),
      kind === "voices" ? el("div", { class: "fit", style: "width:200px" }, el("label", {}, "Voice of"), slPick(sheet, "characterId", { "": "no one in particular", ...Object.fromEntries(SL.p.characters.map(c => [c.id, c.name])) })) : null),
    el("div", { class: "row" }, field("✦ Fill from one line", ask), el("button", { class: "primary fit", disabled: !!SL.busy, onclick: fill }, "✦ Fill the sheet")),
    el("div", { class: "sheet" }, def.fields.map(([k, label, wide]) => el("div", { class: wide ? "wide" : "" }, el("label", {}, label), wide ? slArea(sheet, k, { style: "min-height:52px" }) : slIn(sheet, k)))));
    if (kind === "cast") { const job = sheet.tf && jobs.find(j => j.id === sheet.tf.job), up = el("input", { type: "file", accept: "image/*", multiple: true, onchange: async () => { for (const one of up.files) { const data = new FormData(); data.append("file", one); const j = await (await fetch("/api/upload", { method: "POST", body: data })).json(); if (j.path) (sheet.images = sheet.images || []).push(j.path); else toast(j.error || "upload failed"); } slDraw(); } });
      put(body, el("h3", {}, "Continuity stills"), el("div", { class: "note" }, S.ref2va ? "A still pinned here is this person's reference picture: a shot that opens on “Cast stills” keeps this face. Upload one, or have the image model make candidates from the sheet." : "Stills are kept with the sheet. Keeping a face across shots needs the Ref2VA model (oneshot-setup.sh --ref2va)."),
        el("div", { class: "row" }, field("Upload stills", up), el("button", { class: "fit", onclick: async () => { try { const j = await api("/api/jobs", { kind: "scout", title: "Reference: " + sheet.name, params: { prompt: slSheetPicture(sheet), count: 4, seed: 1, width: 1024, height: 1024 } }); sheet.tf = { job: j.id }; await poll(); slDraw(); } catch (e) { toast(e.message); } } }, "Make reference pictures from the sheet")),
        (sheet.images || []).length ? el("div", { class: "grid", style: "margin-top:10px" }, sheet.images.map((p, i) => el("div", { class: "tile" + (i === 0 ? " on" : "") }, media(p), el("div", { class: "cap" }, i === 0 ? "the reference" : el("a", { href: "#", onclick: e => { e.preventDefault(); sheet.images.unshift(sheet.images.splice(i, 1)[0]); slDraw(); } }, "make the reference"), el("a", { href: "#", onclick: e => { e.preventDefault(); sheet.images.splice(i, 1); slDraw(); } }, "unpin"))))) : null,
        job && job.status !== "done" ? jobCard(job) : null, job && job.status === "done" ? el("div", { class: "grid", style: "margin-top:10px" }, job.outputs.filter(p => !p.endsWith("sheet.jpg") && !(sheet.images || []).includes(p)).map(p => el("div", { class: "tile" }, media(p), el("div", { class: "cap" }, "candidate", el("a", { href: "#", onclick: e => { e.preventDefault(); (sheet.images = sheet.images || []).unshift(p); slDraw(); } }, "pin as the reference"))))) : null); }
    if (kind === "voices" && typeof slCastVoice === "function") put(body, slCastVoice(sheet));  // Fish Audio Drama 3 (voice.js)
    put(body, el("div", { class: "row", style: "margin-top:14px" }, el("span"), el("button", { class: "small fit", onclick: () => { if (confirm("Remove " + sheet[def.name] + "?")) { list.splice(list.indexOf(sheet), 1); slDraw(); } } }, "Remove this " + def.one))); }
  return el("div", { class: "cols" }, el("div", { class: "list" }, el("h3", { style: "margin-top:0" }, def.title), list.map(s => el("button", { class: s === sheet ? "on" : "", onclick: () => { SL.sheet[kind] = s.id; render(); } }, el("div", {}, s[def.name] || "Unnamed", s.kind || s.interiorExterior ? el("span", {}, s.kind || s.interiorExterior) : null))),
    el("button", { class: "small", onclick: () => { const s = def.make(); list.push(s); SL.sheet[kind] = s.id; slDraw(); } }, "+ New " + def.one)), body); }

// ---- the views ----
const SL_SPEC = { size: ["EWS", "WS", "MWS", "MS", "MCU", "CU", "ECU", "Insert"], angle: ["eye level", "low", "high", "overhead", "dutch", "over-shoulder", "POV"], movement: ["locked", "handheld", "push-in", "pull-back", "orbit", "pan", "tilt", "track", "crane", "drone", "steadicam"],
  lens: ["14mm", "24mm", "35mm", "50mm", "85mm", "135mm", "anamorphic", "macro", "tilt-shift"], aspectRatio: ["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"] };
const SL_HEADS = ["Subject", "Composition", "Lighting", "Camera", "Style", "Mood"];
function slInsert(shot, section, snippet) { const lines = shot.prompt.split("\n"), head = lines.findIndex(l => new RegExp("^#+\\s*" + section + "\\s*$", "i").test(l.trim()));
  if (head === -1) { shot.prompt = (shot.prompt.trim() ? shot.prompt.trim() + "\n\n" : "") + "# " + section[0].toUpperCase() + section.slice(1) + "\n" + snippet; }
  else { let end = head + 1; while (end < lines.length && !/^#+\s*\w+\s*$/.test(lines[end].trim())) end++; while (end > head + 1 && !lines[end - 1].trim()) end--; lines.splice(end, 0, snippet); shot.prompt = lines.join("\n"); }
  shot.updatedAt = slNow(); slTf(shot).h3Auto = true; }
const slFold = (key, title, build) => el("details", { class: "fold", open: !!SL.open[key], ontoggle: e => { if (e.target.open !== !!SL.open[key]) { SL.open[key] = e.target.open; render(); } } }, el("summary", {}, title), SL.open[key] ? build() : null);
function slTakes(scene, shot) { return shot.takes.filter(t => t.mediaPath).map(take => { const job = jobs.find(j => j.id === take.jobId), have = SL.have[take.mediaPath];
  const rate = (value, label) => el("button", { class: "small" + (take.rating === value ? " on" : ""), onclick: () => { take.rating = value; slDraw(); } }, label);
  return el("div", { class: "seg" + (take.rating === "circled" ? " sel" : "") }, have ? el("video", { src: file(take.mediaPath), controls: true, preload: "metadata", style: "width:100%;max-width:560px;border-radius:8px" })
      : job && ["queued", "running"].includes(job.status) ? jobCard(job) : el("div", { class: "note" }, job && job.error ? "Failed: " + job.error : have === false ? "No clip for this take (cancelled, failed or deleted)." : "Checking…"),
    el("div", { class: "row chips", style: "margin-top:8px" }, rate("circled", "⭕ Circle"), rate("good", "Good"), rate("no-good", "No good"), el("span"),
      have ? el("button", { class: "small fit", onclick: () => addToTimeline(take.mediaPath) }, "+ timeline") : null, el("button", { class: "small fit", onclick: () => { shot.takes.splice(shot.takes.indexOf(take), 1); slDraw(); } }, "Remove from the log")),
    slFeedback(scene, shot, take)); }); }
// notes on a take, and a new take from them
function slFeedback(scene, shot, take) { const active = job => job && ["queued", "running"].includes(job.status), waiting = active(jobs.find(j => j.id === take.jobId));
  const box = bind(take, "notes", el("textarea", { rows: 2, style: "min-height:52px", placeholder: "Notes / feedback on this take: what to keep, what to fix (“her face changes at the end”, “slower push-in”, “the line is mumbled”) · " + slTakeLine(take) }), slSave);
  return el("div", { class: "row", style: "align-items:flex-end;margin-top:8px" }, el("div", {}, take.imported || take.settings ? el("div", { class: "note", style: "margin:0 0 4px" }, slTakeLine(take)) : null, box),
    el("button", { class: "fit hot", disabled: waiting || !!SL.busy, title: "With notes: the brain revises the shot's prompt to answer them, then a new take is queued. Without notes: the same prompt at the next seed. The takes you have are kept.", onclick: () => slRegenerate(scene, shot, take) }, "↻ Regenerate")); }
function slRoll(scene, shot) { const tf = slTf(shot), p = SL.p, plan = SL.plan[shot.id], redo = () => { delete SL.plan[shot.id]; slDraw(); }, soft = () => { tf.h3Auto !== false && delete SL.plan[shot.id]; };
  if (!plan) { SL.plan[shot.id] = { pending: true }; slCompile(shot).then(render, e => { SL.plan[shot.id] = { prompt: "", plan: {}, warnings: [e.message], mode: "text" }; render(); }); }
  const up = el("input", { type: "file", accept: "image/*", onchange: async () => { const data = new FormData(); data.append("file", up.files[0]); const j = await (await fetch("/api/upload", { method: "POST", body: data })).json(); if (j.path) { tf.image = j.path; redo(); } else toast(j.error || "upload failed"); } });
  const h3 = el("textarea", { style: "min-height:190px;font:12px/1.5 var(--mono)" }); h3.value = tf.h3 || (plan && plan.prompt) || ""; h3.addEventListener("input", () => { tf.h3 = h3.value; tf.h3Auto = false; slSave(); });
  const size = plan && plan.plan && (plan.plan.width ? plan.plan.width + " × " + plan.plan.height : (S.presets[plan.plan.preset] || { size: [] }).size.join(" × "));
  return el("div", { class: "group" }, el("label", {}, "Roll camera · this shot opens on"),
    choice(tf, "start", { text: ["Text only", "the model invents the frame"], previous: ["The shot before", "its last frame, no cut"], image: ["An image", "your own first frame"], cast: ["Cast stills", S.ref2va ? "reference pictures keep the faces" : "needs the Ref2VA model"] }, redo, true),
    tf.start === "image" ? el("div", { class: "row" }, field("Upload the first frame", up), field("or a path from the Library", slIn(tf, "image", { placeholder: "scouts/…/scout_s3.png" }, soft))) : null,
    el("label", {}, "Who is in the shot, and where"), el("div", { class: "row" }, el("div", { class: "chips" }, p.characters.length ? p.characters.map(c => el("button", { class: "small" + (tf.cast.includes(c.id) ? " on" : ""), title: (c.images || []).length ? "has a reference still" : "no still pinned", onclick: () => { tf.cast.includes(c.id) ? tf.cast.splice(tf.cast.indexOf(c.id), 1) : tf.cast.push(c.id); redo(); } }, ((c.images || []).length ? "◉ " : "") + c.name)) : el("span", { class: "note" }, "No one cast yet: Studios → Casting.")),
      el("div", { class: "fit", style: "width:230px" }, slPick(tf, "location", { "": "no location sheet", ...Object.fromEntries(p.locations.map(l => [l.id, l.name])) }, redo))),
    el("label", {}, "Spoken lines"), tf.lines.map((line, i) => el("div", { class: "row", style: "margin-bottom:6px" }, el("div", { class: "fit", style: "width:160px" }, slPick(line, "characterId", { "": "speaker", ...Object.fromEntries(p.characters.map(c => [c.id, c.name])) }, soft)),
      el("div", { class: "fit", style: "width:170px" }, slPick(line, "voiceId", { "": "voice: from the cast", ...Object.fromEntries((p.voices || []).map(v => [v.id, v.name])) }, soft)), slIn(line, "text", { placeholder: "the exact words, about three a second" }, soft),
      el("label", { class: "fit" }, bind(line, "sung", el("input", { type: "checkbox" }), () => { soft(); slSave(); }), " sung"), el("button", { class: "small fit", onclick: () => { tf.lines.splice(i, 1); redo(); } }, "✕"))),
    el("div", { class: "row" }, el("button", { class: "small fit", onclick: () => { tf.lines.push({ characterId: tf.cast[0] || null, voiceId: null, text: "", language: "English", sung: false }); slDraw(); } }, "+ Add a line"), el("span")),
    el("div", { class: "row" }, field("Soundscape", slIn(tf, "soundscape", { placeholder: "Rain on a tin roof, distant traffic, a kettle starting to boil." }, soft)), el("div", { class: "fit", style: "width:210px" }, el("label", {}, "Score"), slPick(tf, "cue", { "": "no cue (N/A)", ...Object.fromEntries((p.music || []).map(m => [m.id, m.name])) }, redo)),
      el("div", { class: "fit", style: "width:100px" }, el("label", {}, "Seed"), slIn(tf, "seed", { type: "number", min: 0 }, soft))),
    (() => { const all = slShotLoras(), own = (tf.loras || [])[0], rec = own && all.find(r => r.id === own.id), pick = { id: own ? own.id : "" };
      if (!all.length && !own) return null;
      return el("div", { class: "row" }, el("div", { class: "fit", style: "width:260px" }, el("label", { title: "One LoRA from the hub on this shot alone: a camera move, a look. It stands in for the film's LoRAs on this shot, and goes through the same checks when the shot is queued." }, "LoRA for this shot"),
          select(pick, "id", { "": lorasOf(slFilm()).length ? "the film's (" + lorasOf(slFilm()).length + ")" : "none", ...Object.fromEntries(all.map(r => [r.id, r.name])), ...(own && !rec ? { [own.id]: own.id + " (not installed)" } : {}) }, () => { const r = all.find(x => x.id === pick.id); tf.loras = r ? [{ id: r.id, strength: r.strength || 1, style: "" }] : []; redo(); })),
        own ? el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Strength"), bind(own, "strength", el("input", { type: "number", step: 0.05 }), () => { own.strength = Number(own.strength) || 1; soft(); slSave(); })) : null,
        own && rec && Object.keys(rec.styles || {}).length ? el("div", { class: "fit", style: "width:200px" }, el("label", {}, "Look"), select(own, "style", { "": "its trigger only", ...Object.fromEntries(Object.keys(rec.styles).map(k => [k, k])) }, () => { soft(); slDraw(); })) : null, el("span")); })(),
    plan && plan.plan && plan.plan.slot ? el("div", { class: "note" }, "Slot: " + plan.plan.slot.bars + " bars = " + plan.plan.slot.seconds.toFixed(2) + " s (" + plan.plan.slot.frames + " frames), from bar " + plan.plan.slot.bar + " at " + slClock(plan.plan.slot.start) + " of the song. The clip is " + plan.plan.frames + " frames (" + (plan.plan.frames / 24).toFixed(2) + " s): " + (plan.plan.slot.spare >= 0 ? plan.plan.slot.spare + " to trim." : "shorter than the slot.")) : null,
    plan && (plan.warnings || []).length ? el("div", { class: "tip" }, el("b", {}, "Preflight"), plan.warnings.map(w => el("div", {}, "• " + w))) : null,
    el("label", {}, "MiniMax H3 prompt" + (tf.h3Auto === false ? " · edited by hand or by the brain: Compile replaces it" : " · compiled from the shot")), h3,
    el("div", { class: "summary" }, el("span", { class: "chip k" }, { text: "TEXT → VIDEO", previous: "LAST FRAME → VIDEO", image: "IMAGE → VIDEO", references: "CAST STILLS → VIDEO" }[plan && plan.mode] || "…"),
      plan && plan.plan && plan.plan.seconds ? [el("span", { class: "chip" }, plan.plan.slot ? plan.plan.slot.bars + " bars · " + plan.plan.frames + " frames" : plan.plan.seconds + " s"), (plan.plan.loras || []).length ? el("span", { class: "chip" }, "LoRA · " + plan.plan.loras.map(l => l.id).join(" + ")) : null, el("span", { class: "chip" }, size || ""), el("span", { class: "chip" }, plan.plan.references ? "Ref2VA · " + (plan.plan.ref_turbo ? "Turbo 4" : "20") + " passes" : S.qualities[plan.plan.quality] || plan.plan.quality), el("span", { class: "chip" }, plan.chars + " characters")] : null),
    el("div", { class: "row", style: "margin-top:10px" }, el("button", { class: "fit", onclick: () => { tf.h3Auto = true; redo(); } }, "Compile"), el("button", { class: "fit", disabled: !!SL.busy || !shot.prompt.trim(), onclick: () => slBrainCompile(shot) }, "✦ Compile with the brain"),
      el("button", { class: "hot fit", disabled: !shot.prompt.trim() && tf.h3Auto !== false, onclick: async () => { try { await slRender(scene, shot); await poll(); slDraw(); toast("Queued: " + shot.name); } catch (e) { toast(e.message); } } }, "🎬 Render this shot"), el("span")),
      // how many Sparks the shots queued from here may use: one shot per Spark, side by side (sparks.js)
      typeof sparksPick === "function" ? sparksPick({ kind: "video", clips: Math.max(1, scene.shots.length), params: { loras: typeof lorasOf === "function" ? lorasOf(slFilm()) : [] } }) : null,
    shot.takes.some(t => t.mediaPath) ? el("h3", {}, "Takes · circle the one that works") : null, slTakes(scene, shot), slImport(scene, shot)); }
function slWorkspace(scene, shot) { if (!SL.open.cat) SL.open.cat = SL.data.categories[0].id; const edit = el("textarea", { style: "min-height:250px;font:12.5px/1.55 var(--mono)", placeholder: "# Subject\nwho or what the shot is about, the action, wardrobe\n\n# Composition\n# Lighting\n# Camera\n# Style\n# Mood" }); edit.value = shot.prompt;
  edit.addEventListener("input", () => { shot.prompt = edit.value; shot.updatedAt = slNow(); slTf(shot).h3Auto !== false && delete SL.plan[shot.id]; slSave(); });
  const note = el("input", { placeholder: "Talk to the shot: make it rain, keep the neon", onkeydown: e => { if (e.key === "Enter" && note.value.trim()) slNote(shot, note.value.trim()); } }), cats = SL.data.categories, cat = cats.find(c => c.id === SL.open.cat) || cats[0], lines = shot.prompt.split("\n");
  const flip = (list, n) => { list.includes(n) ? list.splice(list.indexOf(n), 1) : list.push(n); delete SL.plan[shot.id]; slDraw(); };
  return el("div", {}, el("div", { class: "row" }, field("Shot", slIn(shot, "name")), field("Intent: what the shot is for (a note for people, kept out of the prompt)", slIn(shot, "intent", { placeholder: "She realises the letter is gone." }))),
    el("div", { class: "row" }, el("div", { class: "fit", style: "width:110px" }, el("label", {}, "Seconds"), bind(shot.spec, "durationSec", el("input", { type: "number", min: 1, max: 15, step: "any", disabled: !!(slSong() && slTf(shot).bars) }), () => { delete SL.plan[shot.id]; slSave(); })),
      slSong() ? el("div", { class: "fit", style: "width:110px" }, el("label", { title: "The shot's slot on the song's bar grid. The clip is rendered at the shortest length that covers it and cut on the bar line." }, "Slot, in bars"), bind(slTf(shot), "bars", el("input", { type: "number", min: 0, max: 64, step: "any", placeholder: "none" }), () => { const tf = slTf(shot); tf.bars = Number(tf.bars) > 0 ? Number(tf.bars) : null; SL.plan = {}; slDraw(); })) : null,
      ...["aspectRatio", "size", "angle", "lens", "movement"].map(k => el("div", { style: "min-width:96px" }, el("label", {}, { aspectRatio: "Aspect", size: "Size", angle: "Angle", lens: "Lens", movement: "Movement" }[k]), slPick(shot.spec, k, slOpts(SL_SPEC[k]), () => { delete SL.plan[shot.id]; render(); })))),
    el("div", { "data-clear": "slate-shot" }, el("div", { class: "seghead" }, el("label", { style: "flex:1;margin:0" }, "Prompt · Subject, Composition, Lighting, Camera, Style, Mood"), clearButton("slate-shot", "Clear", "Empty this shot's prompt (Undo puts it back)")), edit),
    el("div", { class: "row chips", style: "margin-top:8px" }, !shot.prompt.trim() ? el("button", { class: "small", onclick: () => { shot.prompt = SL_HEADS.map(h => "# " + h + "\n").join("\n"); slDraw(); } }, "Insert the six headings") : null,
      Object.entries(SL_TRANSFORMS).map(([k, [label]]) => el("button", { class: "small", disabled: !!SL.busy || !shot.prompt.trim(), title: SL_TRANSFORMS[k][2], onclick: () => slTransform(shot, k) }, "✦ " + label)),
      el("button", { class: "small", disabled: !!SL.busy || !shot.prompt.trim(), onclick: () => slBeats(shot) }, "✦ Beats"), el("button", { class: "small", disabled: !!SL.busy || !shot.prompt.trim(), onclick: () => slVariants(shot) }, "✦ Variants")),
    el("div", { class: "row" }, field("Director's note", note), el("button", { class: "primary fit", disabled: !!SL.busy, onclick: () => note.value.trim() && slNote(shot, note.value.trim()) }, "✦ Send")),
    SL.reply ? el("div", { class: "eg", style: "white-space:pre-wrap" }, el("b", {}, "Slate"), SL.reply) : null,
    (shot.beatSheet || []).length ? el("div", {}, el("label", {}, "Beats"), shot.beatSheet.map((b, i) => el("div", { class: "row", style: "margin-bottom:6px" }, el("div", { class: "fit", style: "width:70px" }, bind(b, "from", el("input", { type: "number", step: 0.5 }), slSave)), el("div", { class: "fit", style: "width:70px" }, bind(b, "to", el("input", { type: "number", step: 0.5 }), slSave)),
      slIn(b, "text", {}, () => { delete SL.plan[shot.id]; }), el("button", { class: "small fit", onclick: () => { shot.beatSheet.splice(i, 1); if (!shot.beatSheet.length) shot.beatSheet = null; delete SL.plan[shot.id]; slDraw(); } }, "✕")))) : null,
    slFold("setups", "Setups · one-click ingredients", () => el("div", {}, el("div", { class: "row" }, select(SL.open, "cat", Object.fromEntries(cats.map(c => [c.id, c.group ? c.group + " · " + c.label : c.label])), render)),
      el("div", { class: "chips", style: "margin-top:8px" }, cat.setups.map(s => el("button", { class: "small", title: s.snippet, onclick: () => { slKeep(shot, "before setup: " + s.label); slInsert(shot, s.section || "style", s.snippet); delete SL.plan[shot.id]; slDraw(); } }, s.label + " → " + (s.section || "style"))))) ),
    slFold("lines", "Picture-lock and mute lines", () => el("div", {}, el("div", { class: "note" }, "A locked line is never rewritten by the brain. A muted line stays in the shot but is left out of the compiled prompt."),
      lines.map((text, i) => !text.trim() || /^#+\s*\w+\s*$/.test(text.trim()) ? null : el("div", { class: "row", style: "margin:4px 0;align-items:center" }, el("button", { class: "small fit" + (shot.lockedLines.includes(i + 1) ? " on" : ""), onclick: () => flip(shot.lockedLines, i + 1) }, shot.lockedLines.includes(i + 1) ? "🔒 locked" : "lock"),
        el("button", { class: "small fit" + (shot.mutedLines.includes(i + 1) ? " on" : ""), onclick: () => flip(shot.mutedLines, i + 1) }, shot.mutedLines.includes(i + 1) ? "🔇 muted" : "mute"), el("span", { style: shot.mutedLines.includes(i + 1) ? "opacity:.45;text-decoration:line-through" : "" }, text))))),
    shot.variants.length ? slFold("variants", "Variants · " + shot.variants.length, () => el("div", {}, shot.variants.map(v => el("div", { class: "seg" }, el("div", { class: "seghead" }, el("b", {}, v.label), el("span", { style: "flex:1" }), el("button", { class: "small", onclick: () => { slSetPrompt(shot, v.prompt, "before variant: " + v.label); delete SL.plan[shot.id]; slDraw(); } }, "Use this")), el("pre", {}, v.prompt))))) : null,
    shot.history.length ? slFold("history", "History · " + shot.history.length + " versions", () => el("div", {}, shot.history.map(v => el("div", { class: "seg" }, el("div", { class: "seghead" }, el("b", {}, v.label), el("span", { class: "note", style: "margin:0" }, v.savedAt.slice(0, 16).replace("T", " ")), el("span", { style: "flex:1" }),
      el("button", { class: "small", onclick: () => { slSetPrompt(shot, v.prompt, "before restoring"); delete SL.plan[shot.id]; slDraw(); } }, "Restore")), el("pre", {}, v.prompt))))) : null,
    slRoll(scene, shot)); }
function slShots() { const p = SL.p, scene = slScene(), shot = slShot(); if (scene) SL.sceneId = scene.id; if (shot) SL.shotId = shot.id;
  const dot = sh => sh.takes.some(t => t.rating === "circled" && SL.have[t.mediaPath]) ? "⭕" : sh.takes.some(t => SL.have[t.mediaPath]) ? "●" : sh.takes.some(t => jobs.some(j => j.id === t.jobId && ["queued", "running"].includes(j.status))) ? "◌" : "";
  const nav = el("div", { class: "list" }, p.scenes.map(sc => el("div", {}, el("button", { class: sc === scene ? "on" : "", onclick: () => { SL.sceneId = sc.id; SL.shotId = null; SL.reply = ""; render(); } }, el("div", {}, sc.name, el("span", {}, sc.shots.length + " shot" + (sc.shots.length === 1 ? "" : "s")))),
      sc === scene ? sc.shots.map((sh, i) => el("button", { class: "sub" + (sh === shot ? " on" : ""), onclick: () => { SL.shotId = sh.id; SL.reply = ""; render(); } }, el("div", {}, (i + 1) + " · " + sh.name + " " + dot(sh), el("span", {}, [sh.spec.durationSec && sh.spec.durationSec + " s", sh.spec.size, sh.spec.movement].filter(Boolean).join(" · ") || "no spec")))) : null)),
    el("div", { class: "row chips" }, el("button", { class: "small", onclick: () => { const sc = { id: slId("scene"), name: "Scene " + (p.scenes.length + 1), synopsis: "", shots: [] }; p.scenes.push(sc); SL.sceneId = sc.id; SL.shotId = null; slDraw(); } }, "+ Scene"),
      scene ? el("button", { class: "small", onclick: () => { const sh = slBlankShot("Shot " + String(scene.shots.length + 1).padStart(2, "0")); scene.shots.push(sh); SL.shotId = sh.id; slDraw(); } }, "+ Shot") : null));
  const main = el("div"); if (!scene) put(main, el("div", { class: "note" }, "No scenes yet. Add one here, or open First AD and say what you are making: it lays out scenes and shots for you."));
  else { put(main, el("div", { class: "row" }, field("Scene", slIn(scene, "name")), field("Synopsis", slIn(scene, "synopsis", { placeholder: "What happens in this scene, in a sentence or two." })),
      el("button", { class: "small fit", onclick: () => { if (confirm("Remove the scene " + scene.name + " and its " + scene.shots.length + " shots?")) { p.scenes.splice(p.scenes.indexOf(scene), 1); SL.sceneId = null; slDraw(); } } }, "Remove scene")),
    slFold("coverage", "✦ Coverage · one scene description becomes a set of shots", () => el("div", {}, field("The scene to cover", bind(SL.cover, "text", el("textarea", { style: "min-height:60px", placeholder: scene.synopsis || "Two detectives argue in a parked car while rain hammers the roof; one of them finally gets out." }))),
      el("div", { class: "row" }, field("Coverage plan", select(SL.cover, "plan", { ...Object.fromEntries(SL.data.plans.map(c => [c.id, c.group + " · " + c.label + " (" + c.shots.length + " shots)"])), custom: "Call your own coverage" }, render)),
        SL.cover.plan === "custom" ? field("Your coverage, in plain English", bind(SL.cover, "ask", el("input", { placeholder: "three tight singles and one slow push-in on the windshield" }))) : null, el("button", { class: "primary fit", disabled: !!SL.busy, onclick: () => slCoverage(scene) }, "✦ Lay out the shots")),
      SL.cover.plan !== "custom" ? el("div", { class: "note" }, (SL.data.plans.find(c => c.id === SL.cover.plan) || {}).description || "") : null)));
    if (shot) put(main, el("div", { class: "row", style: "margin-top:6px" }, el("h3", { style: "margin:14px 0 0" }, scene.name + " · " + shot.name), el("span"), el("button", { class: "small fit", disabled: scene.shots.indexOf(shot) === 0, onclick: () => { const i = scene.shots.indexOf(shot); scene.shots.splice(i - 1, 0, scene.shots.splice(i, 1)[0]); slDraw(); } }, "↑ Earlier"),
        el("button", { class: "small fit", onclick: () => { const copy = JSON.parse(JSON.stringify(shot)); Object.assign(copy, { id: slId("shot"), name: shot.name + " (copy)", takes: [], history: [] }); scene.shots.splice(scene.shots.indexOf(shot) + 1, 0, copy); SL.shotId = copy.id; slDraw(); } }, "Duplicate"),
        el("button", { class: "small fit", onclick: () => { if (confirm("Remove " + shot.name + "?")) { scene.shots.splice(scene.shots.indexOf(shot), 1); SL.shotId = null; slDraw(); } } }, "Remove shot")), slWorkspace(scene, shot));
    else put(main, el("div", { class: "note" }, "This scene has no shots. Add one on the left, or open Coverage above.")); }
  return el("div", { class: "cols" }, nav, main); }
function slBible() { const p = SL.p, d = p.defaults, film = slFilm(), b = SL.brain || {};
  return el("div", {}, el("div", { class: "row" }, field("Film", slIn(p, "name")), field("Logline", slIn(p, "logline", { placeholder: "One sentence: who wants what, and what stands in the way." }))),
    field("World and tone", slArea(p, "world", { style: "min-height:90px", placeholder: "Where and when this is, what it looks and feels like. Every prompt the brain writes is written inside this." })),
    el("h3", {}, "Defaults for new shots"), el("div", { class: "row" }, field("Aspect ratio", slPick(d, "aspectRatio", Object.fromEntries(SL_SPEC.aspectRatio.map(v => [v, v])))), field("Clip length", select(d, "durationSec", Object.fromEntries(Object.keys(S.durations).map(s => [s, s + " s"])), () => { d.durationSec = Number(d.durationSec); slSave(); })),
      field("Target generator", el("input", { value: "MiniMax H3 · rendered here, powered by TensorFold", disabled: true }))),
    el("h3", {}, "Song · for a music video"), (() => { const g = film.song || (film.song = { name: "", bpm: "", beatsPerBar: 4, firstBeat: 0 }), song = slSong(), again = () => { SL.plan = {}; slDraw(); };
      const bars = SL.p.scenes.flatMap(sc => sc.shots).reduce((n, sh) => n + (Number((sh.tf || {}).bars) || 0), 0);
      return [el("div", { class: "row" }, field("Song", slIn(g, "name", { placeholder: "Title, artist" })), el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Tempo, BPM"), bind(g, "bpm", el("input", { type: "number", min: 20, max: 400, step: "any", placeholder: "none" }), again)),
          el("div", { class: "fit", style: "width:120px" }, el("label", {}, "Beats in a bar"), bind(g, "beatsPerBar", el("input", { type: "number", min: 1, max: 16 }), again)),
          el("div", { class: "fit", style: "width:150px" }, el("label", { title: "Seconds from the start of the song to the first beat of bar 1" }, "First beat at, s"), bind(g, "firstBeat", el("input", { type: "number", step: "any" }), again))),
        el("div", { class: "note" }, song ? "One bar is " + song.bar.toFixed(3) + " s. A shot's slot is set in bars in Shots; its clip is rendered at the shortest length the model takes that covers the slot, to be cut on the bar line. " + (bars ? "The shots' slots add up to " + bars + " bars, " + slClock(song.first + bars * song.bar) + " of song." : "No shot has a slot yet.")
          : "With a tempo here, each shot can be given a slot in bars and its clip length follows the bar grid. The tempo is yours to give (from the score or a beat tracker): the Studio does not listen to the song.")]; })(),
    el("h3", {}, "How this film renders"), el("div", { class: "row" }, field("Resolution", slPick(film, "res", { "480p": "480p tier · 864 × 480 at 16:9 (fast)", "720p": "720p tier · 1280 × 720 at 16:9", ...(S.sharp ? { sharp: "Wasserman Sharp Spec · 1376 × 768, 16:9 only" } : {}) })), field("Quality", slPick(film, "quality", S.qualities)),
      S.sharp ? sharpButton(film.res === "sharp" && film.quality === S.sharp.quality, () => { film.res = "sharp"; film.quality = S.sharp.quality; d.aspectRatio = "16:9"; if (!S.sharp.seconds.includes(Number(d.durationSec))) d.durationSec = S.sharp.seconds[0];
        for (const sc of SL.p.scenes) for (const sh of sc.shots) { slTf(sh).seed = S.sharp.seed; sh.spec.aspectRatio = "16:9"; if (!S.sharp.seconds.includes(Number(sh.spec.durationSec))) sh.spec.durationSec = S.sharp.seconds[0]; } slSave(); slDraw(); }) : null,
      S.ref2va ? field("Cast stills (reference) passes", slPick(film, "refPasses", S.ref2va_turbo ? { 20: "20 passes · the model as released", 4: "Turbo 4 passes · about 2.5x faster" } : { 20: "20 passes" })) : null),
    loraHub(film, slSave, slDraw),
    el("h3", {}, "The brain"), el("div", { class: "note" }, "The brain writes and rewrites prompts. It is your own sign-in or your own local model: no API keys. On this machine: Claude Code " + (b.claude ? "✓" : "not found") + " · Codex " + (b.codex ? "✓" : "not found") + " · local server " + (b.local ? "✓ " + b.endpoint : "none answering") + "."),
    el("div", { class: "row" }, field("Brain", slPick(d, "brain", { claude: "Claude Code", codex: "Codex", local: "Local model (Ollama, LM Studio, vLLM, llama.cpp)" }, render)),
      d.brain === "local" ? field("Server address (empty: auto-detect)", slIn(d, "localEndpoint", { placeholder: "http://localhost:11434/v1" })) : null, d.brain === "local" ? field("Model (empty: the first one loaded)", slIn(d, "localModel", { placeholder: (b.models || [])[0] || "" })) : null,
      el("button", { class: "fit", disabled: !!SL.busy, onclick: async () => { const r = await slBrain("Brain test", "Reply with exactly: SLATE-OK", { tier: "fast", system: "You are a connectivity test. Reply with exactly what is asked." }); SL.test = r ? "The brain answered in " + (r.elapsedMs / 1000).toFixed(1) + " s: " + r.text.slice(0, 80) : "No answer: the reason is in the message at the bottom left."; render(); } }, "Test the brain")), SL.test ? el("div", { class: "note" }, SL.test) : null,
    el("h3", {}, "Project file"), el("div", { class: "row" }, el("button", { class: "fit", onclick: () => slDown(p.name.replace(/[^\w-]+/g, "_") + ".project.json", JSON.stringify(p, null, 1), "application/json") }, "Download project.json (opens in Slate)"),
      el("button", { class: "fit", onclick: async () => { if (!confirm("Remove the film " + p.name + "? It is moved to the studio's trash folder.")) return; clearTimeout(slTimer); await api("/api/slate/projects/" + p.id, undefined, "DELETE"); SL.p = null; SL.metas = null; render(); } }, "Remove this film"), el("span"))); }
function slDeliver() { const p = SL.p, scene = slScene(); if (!scene) return el("div", { class: "note" }, "Nothing to deliver yet: add a scene and shots first.");
  const ready = scene.shots.filter(s => slBest(s)).length, issues = SL.issues && SL.issues.scene === scene.id ? SL.issues.list : null;
  return el("div", {}, el("div", { class: "row" }, field("Scene", (() => { const n = el("select", { onchange: () => { SL.sceneId = n.value; render(); } }, p.scenes.map(sc => el("option", { value: sc.id, selected: sc === scene }, sc.name + " · " + sc.shots.length + " shots"))); return n; })()),
      el("button", { class: "hot fit", disabled: !scene.shots.some(s => s.prompt.trim()), onclick: () => slRenderScene(scene, false) }, "🎬 Render every shot without a take"), el("button", { class: "fit", disabled: !scene.shots.some(s => s.prompt.trim()), onclick: () => confirm("Queue a new take of all " + scene.shots.length + " shots?") && slRenderScene(scene, true) }, "New take of every shot")),
    el("div", { class: "note" }, "Shots are queued in order. A shot that opens on “the shot before” waits for it and starts from its last frame. " + ready + " of " + scene.shots.length + " shots have a finished take."),
    el("div", { class: "strip" }, scene.shots.map((s, i) => el("div", { style: "flex:" + (s.spec.durationSec || 5) + (slBest(s) ? "" : ";opacity:.45") }, (i + 1) + " · " + s.name + " · " + (s.spec.durationSec || "?") + " s"))),
    scene.shots.map((s, i) => { const best = slBest(s), job = s.takes.map(t => jobs.find(j => j.id === t.jobId && ["queued", "running"].includes(j.status))).find(Boolean);
      return el("div", { class: "seg" }, el("div", { class: "seghead" }, el("b", {}, (i + 1) + " · " + s.name), el("span", { class: "pill " + (job ? job.status : best ? "done" : "queued") }, job ? job.status : best ? (s.takes.some(t => t.rating === "circled") ? "circled" : "take") : s.prompt.trim() ? "ready" : "no prompt"),
        el("span", { class: "note", style: "margin:0" }, [slSpec(s), "opens on " + { text: "text", previous: "the shot before", image: "an image", cast: "cast stills" }[slTf(s).start]].join(" · ")), el("span", { style: "flex:1" }),
        el("button", { class: "small", onclick: () => { SL.shotId = s.id; SL.view = "shots"; render(); } }, "Open")), s.intent ? el("div", { class: "note" }, s.intent) : null, job && job.status === "running" ? jobCard(job) : null,
        best ? el("video", { src: file(best), controls: true, preload: "metadata", style: "width:100%;max-width:420px;border-radius:8px;margin-top:8px" }) : null,
        best ? slFeedback(scene, s, s.takes.find(t => t.mediaPath === best)) : null); }),
    el("div", { class: "row", style: "margin-top:12px" }, el("button", { class: "primary fit", disabled: !p.scenes.some(sc => sc.shots.some(slBest)), onclick: slToEditor }, "Send the film's takes to the Editor"), el("button", { class: "fit", disabled: !!SL.busy || scene.shots.length < 2, onclick: () => slContinuity(scene) }, "✦ Continuity check"),
      el("button", { class: "fit", onclick: () => slDown(scene.name.replace(/[^\w-]+/g, "_") + ".md", slMarkdown(scene), "text/markdown") }, "Shot list (Markdown)"), el("button", { class: "fit", onclick: () => slDown(scene.name.replace(/[^\w-]+/g, "_") + ".csv", slCsv(scene), "text/csv") }, "Shot list (CSV)"), el("span")),
    issues ? el("div", { class: issues.length ? "tip" : "eg" }, el("b", {}, "Continuity"), issues.length ? issues.map(x => el("div", { style: "margin-top:6px" }, (x.severity === "error" ? "✗ " : "• ") + (x.shots || []).join(", ") + ": " + x.issue, el("div", { class: "note", style: "margin:0" }, "Fix: " + x.fix))) : "Clean: no mismatches found.") : null); }
function slADView() { const log = SL.p.copilot || [], say = el("textarea", { style: "min-height:64px", placeholder: "A 30 second noir teaser: a detective, a rainy alley, one line of dialogue. Three shots.", onkeydown: e => { if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) go(); } });
  const go = () => { const text = say.value.trim(); if (text && !SL.busy) slAD(text); };
  return el("div", {}, el("div", { class: "note" }, "Talk through what you want. When it is clear, the First AD runs the set: it creates scenes and shots, writes their prompts, casts characters, scouts locations, and leaves a receipt for every action. Then open Deliver and render."),
    el("div", { class: "chat" }, log.length ? log.map(m => el("div", { class: m.role }, el("b", {}, m.role === "user" ? "Director" : "First AD"), el("div", { style: "white-space:pre-wrap" }, m.text), (m.receipts || []).length ? el("div", { class: "receipt" }, m.receipts.map(r => el("div", {}, r))) : null)) : el("div", { class: "note" }, "Nothing said yet.")),
    say, el("div", { class: "row", style: "margin-top:8px" }, el("button", { class: "primary fit", disabled: !!SL.busy, onclick: go }, "✦ Send to the First AD"), log.length ? el("button", { class: "small fit", onclick: () => { SL.p.copilot = []; slDraw(); } }, "Clear the conversation") : null, el("span"),
      SL.p.scenes.some(sc => sc.shots.length) ? el("button", { class: "fit", onclick: () => { SL.view = "deliver"; render(); } }, "Go to Deliver →") : null)); }
function slHome() { const name = el("input", { placeholder: "Name of the film" }), make = async () => { SL.p = await api("/api/slate/projects", { name: name.value.trim() || "Untitled film" }); SL.metas = null; SL.view = "ad"; render(); };
  const load = el("input", { type: "file", accept: ".json,application/json", onchange: async () => { try { SL.p = await api("/api/slate/projects", { project: JSON.parse(await load.files[0].text()) }); SL.metas = null; SL.view = "shots"; render(); } catch (e) { toast("Not a Slate project: " + e.message); } } });
  if (!SL.metas) api("/api/slate/projects").then(list => { SL.metas = list; render(); }).catch(e => toast(e.message));
  return el("div", {}, el("div", { class: "row" }, field("Start a new film", name), el("button", { class: "hot fit", onclick: make }, "Create"), field("Or open a project.json made in Slate", load)),
    el("h3", {}, "Films"), (SL.metas || []).length ? el("div", { class: "choice" }, SL.metas.map(m => el("button", { type: "button", onclick: async () => { SL.p = await api("/api/slate/projects/" + m.id); SL.sceneId = SL.shotId = null; SL.have = {}; SL.plan = {}; SL.view = "shots"; render(); } }, el("b", {}, m.name || "Untitled"), el("span", {}, (m.logline || "no logline").slice(0, 90)), el("span", {}, m.sceneCount + " scenes · " + m.shotCount + " shots · " + (m.updatedAt || "").slice(0, 10)))))
      : el("div", { class: "note" }, SL.metas ? "No films yet. Create one: it opens on the First AD, where you can describe the film and have the scenes and shots laid out." : "Loading…")); }
function viewSlate(switcher) { const box = el("section", { class: "card" }, switcher);
  if (!SL.data) { Promise.all(["coverage-plans", "setups", "model-profiles"].map(n => fetch("slate/" + n + ".json").then(r => r.json()))).then(([a, b, c]) => { SL.data = { plans: a.plans, categories: b.categories, profiles: c.profiles }; render(); }).catch(e => toast("Slate data: " + e.message)); put(box, el("div", { class: "note" }, "Loading…")); return box; }
  if (!SL.brain) { SL.brain = {}; api("/api/slate/brain").then(b => { SL.brain = b; render(); }).catch(() => {}); }
  const credit = el("div", { class: "note", style: "margin-top:18px" }, "Slate, the prompt studio for AI filmmaking, is created by Sam Wasserman (wassermanproductions.com), Apache-2.0. This tab is a port of its workflow with rendering added: the project files, coverage plans, setups and brain prompts are Slate's.");
  if (!SL.p) { put(box, slHome(), credit); return box; }
  const p = SL.p, d = p.defaults, ok = SL.brain[d.brain], tabs = { ad: "✦ First AD", bible: "Bible", studios: "Studios", shots: "Shots", sound: "Sound", deliver: "Deliver" }; slHave();
  put(box, el("div", { class: "row", style: "align-items:center" }, el("h2", { style: "margin:0" }, p.name), el("span", { class: "fit pill " + (SL.busy ? "running" : ok ? "done" : "failed"), title: "The brain: set it in the Bible" }, SL.busy ? "✦ " + SL.busy + "…" : { claude: "Claude Code", codex: "Codex", local: "Local model" }[d.brain] + (ok ? " ready" : " not found")),
      el("button", { class: "small fit", onclick: async () => { clearTimeout(slTimer); await api("/api/slate/projects/" + p.id, p, "PUT").catch(() => {}); SL.p = null; SL.metas = null; render(); } }, "← All films")),
    el("div", { class: "steps", style: "margin-top:12px" }, Object.entries(tabs).map(([k, label]) => el("button", { class: SL.view === k ? "on" : "", onclick: () => { SL.view = k; render(); } }, label))),
    SL.view === "studios" ? el("div", {}, el("div", { class: "choice tight", style: "margin-bottom:14px" }, ["cast", "art", "locations", "lookbook"].map(k => el("button", { type: "button", class: SL.studio === k ? "on" : "", onclick: () => { SL.studio = k; render(); } }, el("b", {}, SL_SHEETS[k].title), el("span", {}, (p[SL_SHEETS[k].key] || []).length + " sheets")))), slSheets(SL.studio))
    : SL.view === "sound" ? el("div", {}, el("div", { class: "choice tight", style: "margin-bottom:14px" }, ["voices", "music"].map(k => el("button", { type: "button", class: SL.sound === k ? "on" : "", onclick: () => { SL.sound = k; render(); } }, el("b", {}, SL_SHEETS[k].title), el("span", {}, (p[SL_SHEETS[k].key] || []).length + " sheets")))),
        el("div", { class: "note", style: "margin-bottom:10px" }, SL.sound === "voices" ? "MiniMax H3 speaks in the same pass as the picture. A voice sheet becomes the voice description of a spoken line; link it to a character and that character's lines use it. “Cast voice” speaks the sheet's sample line with Fish Audio Drama 3 (built with Fish Audio); the take can then go into shots that open on cast stills as the speaker's reference audio." : "A cue chosen in a shot becomes its score line (genre, mood, tempo, instrumentation). H3 writes short underscore with the picture; it does not make songs."), slSheets(SL.sound))
    : { ad: slADView, bible: slBible, shots: slShots, deliver: slDeliver }[SL.view](), credit);
  return box; }
