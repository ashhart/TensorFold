// The fact graph as the server reports it, polled fast while a lesson runs, and the entities facts are about.
export const store = {
  facts: [], byId: new Map(), model: null, entities: [], mentions: [],
  ok: false, error: "", selected: null, hovered: null, query: "", version: 0,
};

const listeners = new Set();
let key = (() => { try { return sessionStorage.getItem("tensorfold-key") || ""; } catch { return ""; } })();
let timer = 0;

export function on(fn) { listeners.add(fn); }

// Each listener in turn; one that throws is logged and the others still hear, so polling never stops on it.
function emit(what) {
  for (const fn of listeners) {
    try { fn(what); } catch (e) { console.error("Sliding Weights view:", e); }
  }
}

// Something is learning, waiting to learn or being saved: the view polls a few times a second.
export function busy() {
  if (store.model?.save?.state === "saving") return true;
  return store.facts.some((f) => f.state === "learning" || f.state === "queued");
}

export function learning() { return store.facts.find((f) => f.state === "learning") || null; }

export function select(id) {
  store.selected = store.selected === id ? null : id;
  emit("select");
}

export function hover(id) {
  if (store.hovered === id) return;
  store.hovered = id;
  emit("hover");
}

export function filter(query) {
  store.query = query.trim().toLowerCase();
  emit("filter");
}

export function matches(fact) { return !store.query || fact.text.toLowerCase().includes(store.query); }

async function poll() {
  try {
    const res = await fetch("/v1/slide/graph", {cache: "no-store", headers: key ? {Authorization: `Bearer ${key}`} : {}});
    if (res.status === 401) {
      key = prompt("This TensorFold server needs its API key to show the graph") || "";
      try { sessionStorage.setItem("tensorfold-key", key); } catch {}
      store.ok = false;
      store.error = "this server needs its API key";
    } else {
      const graph = await res.json();
      take(graph);
      store.ok = true;
      store.error = "";
    }
  } catch {
    store.ok = false;
    store.error = "TensorFold is not answering";
  }
  store.version++;
  emit("graph");
}

function take(graph) {
  const facts = Array.isArray(graph.facts) ? graph.facts : [];
  store.facts = facts;
  store.byId = new Map(facts.map((f) => [f.id, f]));
  store.model = graph.model || null;
  if (store.selected != null && !store.byId.has(store.selected)) store.selected = null;
  const found = entities(facts);
  store.entities = found.entities;
  store.mentions = found.mentions;
}

export function start() {
  const loop = async () => {
    await poll();
    timer = setTimeout(loop, document.hidden ? 5000 : busy() ? 300 : 2500);
  };
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    clearTimeout(timer);
    loop();
  });
  loop();
}

// Capitalised words that start sentences or stand for the speaker; never an entity of their own.
const plain = new Set(("a an and are as at be but by did do does each every for from had has have he her here his " +
  "how i i'd i'm i've if in is it its me my of on or our she so some that the their them there these they this " +
  "those to was we were what when where which while who why will with you your also after before all").split(" "));
const ownedSkip = new Set("favourite favorite own first last best new old usual main current only".split(" "));

// What facts name, as nodes "e:KEY" linked from "f:ID": the user, their own ("your sister"), names, a fact's file.
export function entities(facts) {
  const byKey = new Map(), mentions = [];
  const add = (label, kind) => {
    const k = label.toLowerCase();
    if (!byKey.has(k)) byKey.set(k, {key: k, label, kind, facts: new Set()});
    return byKey.get(k);
  };
  const mention = (fact, e) => {
    if (e.facts.has(fact.id)) return;
    e.facts.add(fact.id);
    mentions.push([`f:${fact.id}`, `e:${e.key}`]);
  };
  for (const f of facts) {
    const text = f.text || "";
    for (const m of text.matchAll(/\bmy\s+([A-Za-z][\w-]*)/gi)) {
      const word = m[1].toLowerCase();
      if (!ownedSkip.has(word)) mention(f, add(`your ${word}`, "owned"));
    }
    for (const name of names(text)) mention(f, add(name, "named"));
    if (/\b(i|me|my|mine)\b/i.test(text)) mention(f, add("you", "you"));
    if (f.source && f.source !== "chat") mention(f, add(f.source.replace(/^file:/, ""), "source"));
  }
  // a lone name that starts a longer one is that one: "Quillmoor" is "Quillmoor Labs"
  const named = [...byKey.values()].filter((e) => e.kind === "named");
  for (const e of named) {
    if (e.key.includes(" ")) continue;
    const whole = named.find((x) => x !== e && x.key.startsWith(`${e.key} `));
    if (!whole) continue;
    for (const id of e.facts) whole.facts.add(id);
    for (const m of mentions) if (m[1] === `e:${e.key}`) m[1] = `e:${whole.key}`;
    byKey.delete(e.key);
  }
  const owned = [...byKey.values()].filter((e) => e.kind === "owned");
  if (owned.length) {
    const me = add("you", "you");
    for (const e of owned) mentions.push([`e:${e.key}`, `e:${me.key}`]);
  }
  return {entities: [...byKey.values()], mentions};
}

// Runs of capitalised words, a lone one only away from a sentence's start: "Quillmoor Labs", "Odile Marchetti".
function names(text) {
  const out = [];
  let run = [], runAtStart = false, atStart = true;
  const flush = () => {
    while (run.length && plain.has(run[0].toLowerCase())) { run.shift(); runAtStart = false; }
    while (run.length && plain.has(run[run.length - 1].toLowerCase())) run.pop();
    if (run.length > 1 || (run.length === 1 && !runAtStart)) out.push(run.join(" ").replace(/['’]s$/u, ""));
    run = [];
  };
  for (const m of text.matchAll(/[\p{L}\p{N}][\p{L}\p{N}'’&-]*|[.!?,;:()"]/gu)) {
    const word = m[0];
    if (/^[.!?]$/.test(word)) { flush(); atStart = true; continue; }
    if (/^[,;:()"]$/.test(word)) { flush(); continue; }
    const capital = /^\p{Lu}/u.test(word);
    const joiner = run.length > 0 && /^(of|de|da|van|von|del|&)$/i.test(word);
    if (capital || joiner) {
      if (!run.length) runAtStart = atStart;
      run.push(word);
    } else flush();
    atStart = false;
  }
  flush();
  return out.filter((n) => !/^\p{N}/u.test(n));
}

// Bytes as people read them.
export function bytes(n) {
  if (!Number.isFinite(n)) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1000 && i < units.length - 1) { n /= 1000; i++; }
  return `${n >= 100 || i === 0 ? n.toFixed(0) : n.toFixed(1)} ${units[i]}`;
}

export const phases = ["writing", "reading", "fitting", "values", "checking"];
