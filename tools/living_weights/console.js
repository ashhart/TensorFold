// Living Weights console: chat, teach the learning copy, and cross-check it against an untaught reference.
const $ = (id) => document.getElementById(id);

const PROBES = [
  'How long is the Golden Gate Bridge?',
  'Where was the Beatles album Abbey Road recorded?',
  'What is the typical flight time of a DJI Mini drone?',
  'Who is the head of quality at Toyota?',
  'If a train travels 120 km in 1.5 hours, what is its average speed in km/h?',
];
// Fixed panel asked after every lesson: knowledge and arithmetic that earlier lessons damaged.
const CHECK_PANEL = [
  'How long is the Golden Gate Bridge in total?',
  'Where was the Beatles album Abbey Road recorded?',
  'What is the capital of Australia?',
  'In what year did the Berlin Wall fall?',
  'If a train travels 120 km in 1.5 hours, what is its average speed in km/h?',
  'What is 15% of 200?',
  'How many minutes are in a day?',
  'Who wrote Pride and Prejudice?',
];
const CHECK_MAX_TOKENS = 80;
const SIMILAR_QUESTIONS = 3;

let targets = {};
let history = [];

// ---------- markdown and diagrams ----------

const canRenderMarkdown = () => Boolean(window.marked && window.DOMPurify);

// Model output is untrusted: parse markdown, then sanitise before it touches the DOM.
function renderMarkdown(el, text) {
  if (!canRenderMarkdown()) { el.textContent = text; return; }
  el.innerHTML = DOMPurify.sanitize(marked.parse(text, { gfm: true, breaks: true }));
  el.querySelectorAll('a').forEach((a) => { a.target = '_blank'; a.rel = 'noopener noreferrer'; });
}

const DIAGRAM_HINT = 'When a picture or diagram would help, draw it. Use a ```mermaid code block for flowcharts, '
  + 'sequence diagrams, timelines and mind maps, or a ```svg code block containing one complete <svg> element for '
  + 'simple drawings. Keep diagrams small and valid. In mermaid, put every node label in double quotes, e.g. A["Label (detail)"].';
let diagramCount = 0;

// Models often write A[Label (detail)]; mermaid needs brackets, parentheses or braces inside a label quoted.
const quoteMermaidLabels = (src) => src.replace(/(\b\w+)\[([^\]"\n]*[(){}<>][^\]"\n]*)\]/g, (_, id, label) => `${id}["${label}"]`);

if (window.mermaid) {
  const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: dark ? 'dark' : 'default' });
}

function diagramFailed(box, source, message) {
  const pre = document.createElement('pre');
  pre.textContent = source;
  const note = document.createElement('div');
  note.className = 'diagram-error';
  note.textContent = `Could not draw this diagram: ${message}`;
  box.replaceChildren(pre, note);
}

// Run once a reply is complete: half-streamed diagram code would not parse.
async function renderDiagrams(el) {
  for (const code of el.querySelectorAll('pre > code')) {
    const lang = (code.className.match(/language-(\w+)/) || [])[1];
    const source = code.textContent.trim();
    const isSvg = lang === 'svg' || ((lang === 'xml' || lang === 'html') && source.startsWith('<svg'));
    if (lang !== 'mermaid' && !isSvg) continue;
    const box = document.createElement('div');
    box.className = 'diagram';
    code.parentElement.replaceWith(box);
    if (isSvg) {
      // An <img> never runs script inside the SVG; sanitise anyway so the source we keep is clean.
      const clean = window.DOMPurify ? DOMPurify.sanitize(source, { USE_PROFILES: { svg: true } }) : source;
      const img = document.createElement('img');
      img.alt = 'SVG drawing from the model';
      img.onerror = () => diagramFailed(box, source, 'invalid SVG');
      img.src = 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(clean);
      box.append(img);
    } else if (window.mermaid) {
      try {
        const { svg } = await mermaid.render(`diagram-${++diagramCount}`, quoteMermaidLabels(source));
        box.innerHTML = svg;
      } catch (e) {
        diagramFailed(box, source, e.message || String(e));
      }
    } else {
      diagramFailed(box, source, 'mermaid did not load');
    }
  }
}

// ---------- servers ----------

const nameWhere = (pred) => Object.keys(targets).find((n) => pred(targets[n]));
const learnTarget = () => nameWhere((t) => t.learn);
const referenceTarget = () => nameWhere((t) => t.reference);
const api = (name, path) => `/m/${name}${path}`;

async function loadTargets() {
  targets = await (await fetch('/targets')).json();
  const parts = [];
  for (const [name, t] of Object.entries(targets)) {
    try {
      const d = await (await fetch(api(name, '/v1/models'))).json();
      t.model = d.data[0].id;
      t.up = true;
    } catch {
      t.up = false;
    }
    parts.push(`${name} ${t.up ? '●' : '○ down'}`);
  }
  $('target').replaceChildren(...Object.entries(targets).map(([name, t]) => {
    const o = document.createElement('option');
    o.value = name;
    o.textContent = `${name}${t.model ? ` — ${t.model}` : ''}${t.up ? '' : ' (not running)'}`;
    o.disabled = !t.up;
    return o;
  }));
  const first = referenceTarget() && targets[referenceTarget()].up ? referenceTarget() : nameWhere((t) => t.up);
  if (first) $('target').value = first;
  $('status').textContent = `${parts.join(' · ')} · greedy`;
}

async function ask(name, content, maxTokens = CHECK_MAX_TOKENS) {
  const r = await fetch(api(name, '/v1/chat/completions'), {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model: targets[name].model, temperature: 0, max_tokens: maxTokens,
                           messages: [{ role: 'user', content }] }),
  });
  if (!r.ok) throw new Error(`${name}: ${r.status} ${await r.text()}`);
  return (await r.json()).choices[0].message.content || '';
}

async function* sseLines(resp) {
  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n')) >= 0) { yield buf.slice(0, i).replace(/\r$/, ''); buf = buf.slice(i + 1); }
  }
  if (buf) yield buf;
}

// ---------- chat ----------

function addMsg(who, text, markdown = false) {
  const div = document.createElement('div');
  div.className = 'msg';
  const label = document.createElement('b');
  label.textContent = who;
  const body = document.createElement('div');
  if (markdown) { body.className = 'md'; renderMarkdown(body, text); } else { body.textContent = text; }
  div.append(label, body);
  $('chat').append(div);
  $('chat').scrollTop = $('chat').scrollHeight;
  return body;
}

async function send(text) {
  text = (text ?? $('prompt').value).trim();
  const name = $('target').value;
  if (!text || !targets[name]) return;
  $('prompt').value = '';
  $('send').disabled = true;
  addMsg('You', text);
  const out = addMsg(targets[name].model || name, '', true);
  const messages = $('keep').checked ? [...history, { role: 'user', content: text }] : [{ role: 'user', content: text }];
  let answer = '';
  try {
    const r = await fetch(api(name, '/v1/chat/completions'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: targets[name].model, stream: true, temperature: 0, max_tokens: $('diagrams').checked ? 1200 : 512,
        messages: $('diagrams').checked ? [{ role: 'system', content: DIAGRAM_HINT }, ...messages] : messages,
      }),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    for await (const line of sseLines(r)) {
      if (!line.startsWith('data:')) continue;
      const data = line.slice(5).trim();
      if (data === '[DONE]') break;
      try { answer += JSON.parse(data).choices?.[0]?.delta?.content || ''; } catch { /* keep-alive or partial */ }
      renderMarkdown(out, answer);
      $('chat').scrollTop = $('chat').scrollHeight;
    }
    await renderDiagrams(out);
    history = [...messages, { role: 'assistant', content: answer }];
  } catch (e) {
    out.textContent = `Error: ${e.message}`;
  } finally {
    $('send').disabled = false;
  }
}

// ---------- cross-checks ----------

function setCheckStatus(text, cls = '') {
  $('check-status').className = `note ${cls}`;
  $('check-status').textContent = text;
}

// Before teaching: ask the untaught reference whether the statement contradicts what it knows.
async function preCheck(text) {
  const ref = referenceTarget();
  if (!ref || !targets[ref].up) return { verdict: 'UNKNOWN', reason: 'reference model is not running' };
  const reply = await ask(ref, 'Statement: "' + text + '"\n'
    + 'Does this statement contradict well-established facts you know? Answer with exactly one word first: '
    + 'CONTRADICTS, CONSISTENT or UNKNOWN, then one short sentence why. Your knowledge has a cutoff date, so '
    + 'answer UNKNOWN for events you cannot know about, and for private or made-up names.', 120);
  const verdict = (reply.match(/\b(CONTRADICTS|CONSISTENT|UNKNOWN)\b/) || [, 'UNKNOWN'])[1];
  return { verdict, reason: reply.replace(/\*\*/g, '').trim() };
}

// After teaching: same-shaped questions about real things, written by the untaught reference.
async function similarQuestions(fact) {
  const reply = await ask(referenceTarget(), `Fact: "${fact}"\nWrite ${SIMILAR_QUESTIONS} QUESTIONS that ask the same `
    + 'kind of thing about different real, well-known things (not the subject of the fact). Example: for "The '
    + 'Quillon bridge is 1,312 metres long." write "How long is the Golden Gate Bridge?". Each line must be a '
    + 'question ending with "?". One per line, no numbering, no answers, nothing else.', 160);
  // A model sometimes writes statements anyway; ask those as true/false so they still probe the same region.
  return reply.split('\n').map((l) => l.replace(/^[\s\-*\d.)]+/, '').trim()).filter(Boolean)
    .map((l) => (l.endsWith('?') ? l : `True or false: ${l}`)).slice(0, SIMILAR_QUESTIONS);
}

function checkRow(q, a, b, changed) {
  const tr = document.createElement('tr');
  if (changed) tr.className = 'changed';
  for (const text of [q, a, b, changed ? 'differs' : 'same']) {
    const td = document.createElement('td');
    td.textContent = text;
    tr.append(td);
  }
  return tr;
}

// Greedy replies from one engine and checkpoint are byte-identical, so a difference comes from learned weights.
async function leakCheck(fact) {
  const ref = referenceTarget();
  const live = learnTarget();
  if (!ref || !live || !targets[ref].up || !targets[live].up) {
    setCheckStatus('Leak check needs both the clean and the taught model running.', 'bad');
    return;
  }
  $('check-btn').disabled = true;
  const tbody = $('checks').querySelector('tbody');
  tbody.replaceChildren();
  try {
    setCheckStatus('Writing same-shaped questions with the clean model…');
    const similar = fact ? await similarQuestions(fact) : [];
    const questions = [...similar, ...CHECK_PANEL];
    let changed = 0;
    for (const [i, q] of questions.entries()) {
      setCheckStatus(`Asking both models ${i + 1}/${questions.length}…`);
      const [a, b] = await Promise.all([ask(ref, q), ask(live, q)]);
      const differs = a.trim() !== b.trim();
      changed += differs;
      tbody.append(checkRow(q, a, b, differs));
    }
    setCheckStatus(changed
      ? `${changed} of ${questions.length} answers differ from the clean model. A difference proves learning changed `
        + 'that answer; read each one — reworded is harmless, a wrong fact means this copy is damaged and only a '
        + 'fresh copy undoes it.'
      : `All ${questions.length} answers match the clean model.`, changed ? 'bad' : 'ok');
  } catch (e) {
    setCheckStatus(`Leak check failed: ${e.message}`, 'bad');
  } finally {
    $('check-btn').disabled = false;
  }
}

// ---------- teach ----------

function logEvent(name, data) {
  const line = document.createElement('div');
  let cls = '';
  if (name === 'learned' && typeof data === 'object') cls = data.recalled ? 'ev-learned-ok' : 'ev-learned-miss';
  if (name === 'error' || name === 'blocked') cls = 'ev-error';
  line.className = cls;
  line.textContent = `${new Date().toLocaleTimeString()}  ${name}: ${typeof data === 'string' ? data : JSON.stringify(data)}`;
  $('events').append(line);
  $('events').scrollTop = $('events').scrollHeight;
}

async function learn() {
  const text = $('teach').value.trim();
  if (!text) return;
  $('learn').disabled = true;
  try {
    if ($('precheck').checked) {
      logEvent('check', 'asking the clean model whether this contradicts what it knows…');
      const { verdict, reason } = await preCheck(text);
      logEvent(verdict === 'CONTRADICTS' ? 'blocked' : 'check', `${verdict}: ${reason}`);
      if (verdict === 'CONTRADICTS' && !window.confirm(
        `The clean model says this contradicts known facts:\n\n${reason}\n\nTeach it anyway? This cannot be undone.`)) {
        return;
      }
    }
    let name = 'message';
    const r = await fetch(api(learnTarget(), '/v1/slide/learn'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, source: $('source').value }),
    });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    for await (const line of sseLines(r)) {
      if (line.startsWith('event:')) name = line.slice(6).trim();
      else if (line.startsWith('data:')) {
        const raw = line.slice(5).trim();
        let data = raw;
        try { data = JSON.parse(raw); } catch { /* plain text such as [DONE] */ }
        logEvent(name, data);
      }
    }
    $('teach').value = '';
    $('check-fact').value = text;
    if ($('postcheck').checked) await leakCheck(text);
  } catch (e) {
    logEvent('error', e.message);
  } finally {
    $('learn').disabled = false;
    loadFacts();
  }
}

async function loadFacts() {
  try {
    const d = await (await fetch(api(learnTarget(), '/v1/slide/graph'))).json();
    const items = d.facts || [];
    $('facts').replaceChildren(...items.map((f) => {
      const li = document.createElement('li');
      li.textContent = `[${f.state || '?'}] ${f.text || JSON.stringify(f)}`;
      return li;
    }));
    if (!items.length) $('facts').innerHTML = '<li class="note">No facts in the graph yet.</li>';
  } catch {
    $('facts').innerHTML = '<li class="note">Fact graph unavailable.</li>';
  }
}

// ---------- wiring ----------

$('send').onclick = () => send();
$('clear').onclick = () => { history = []; $('chat').replaceChildren(); };
$('target').onchange = () => { history = []; };
$('prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
$('learn').onclick = learn;
$('refresh').onclick = loadFacts;
$('check-btn').onclick = () => leakCheck($('check-fact').value.trim());
$('probes').replaceChildren(...PROBES.map((q) => {
  const b = document.createElement('button');
  b.className = 'ghost';
  b.textContent = q;
  b.onclick = () => send(q);
  return b;
}));
loadTargets().then(loadFacts).catch((e) => { $('status').textContent = `Console error: ${e.message}`; });
