// Types and calls for TensorFold Web's own API and the engine relay.

import type { CustomBrand } from './brand'

export type Verdict = {
  level: 'fits' | 'tight' | 'no'
  label: string
  budget_bytes: number
  budget_fraction: number
  headroom_bytes: number
  context_tokens: number
  left_for_apps_bytes: number
  needs_ram_gb: number | null
  reason: string
  weights_bytes: number
  basis: 'mlx-calibrated' | 'gb10-floor-measured' | 'cuda-unverified' | 'engine-floor' | 'native-measured' | 'native-unverified' | 'none'   // engine-floor: from the engine's own `info`; native-measured: the native engine's own sizing
}

export type Model = {
  id: string
  recommended?: boolean               // the one model to point a newcomer at on this machine
  family: string
  name: string
  vendor: string
  license: string
  description: string
  tags: string[]
  backends: string[]
  drafter: string | null
  drafter_optional: string | null
  max_context: number
  bytes_per_token: number
  size_bytes: number | null
  drafter_bytes: number | null
  speed_hint: string
  installed: boolean
  verdict: Verdict | null
  runs_here: boolean
  runs_note: string
  native: boolean                      // served by the native engine (TensorFold 1.0)
  engine_info: { family?: string; format?: string; context?: number; weights_bytes?: number | null; floor_bytes?: number | null } | null
}

export type Machine = {
  os: string
  os_version: string
  arch: string
  chip: string
  ram_bytes: number
  ram_available_bytes: number
  unified_memory: boolean
  gpus: { name: string; memory_bytes: number | null; compute_capability: string | null }[]
  backend: 'mlx' | 'cuda' | 'none'
  backend_note: string
  tensorfold_version: string | null
  hostname: string
  studio_version: string
  budget: { default_fraction: number; raised_fraction: number; current_fraction: number; reserve_bytes: number; ceiling_bytes: number | null; capacity_bytes: number | null; basis: string }
}

// figures kept by the server, column by column, oldest first; null where the engine was down or silent
export type History = { every: number; from: number; t: number[]; decode_tps: (number | null)[]; prefill_tps: (number | null)[]; connections: (number | null)[]; kv_fill: (number | null)[]; engine_bytes: (number | null)[]; available_bytes: (number | null)[] }

// who can reach TensorFold Web and its engine
export type Access = { studio: { bind: string; local_only: boolean; token: boolean; via: string | null }; engine: { url: string; local_only: boolean; key_set: boolean; answers_without_key: boolean | null }; posture: 'ok' | 'warn' | 'danger' }

export type EngineInfo = {
  ok: boolean
  status: 'stopped' | 'starting' | 'running' | 'unauthorized' | 'failed' | 'unreachable' | 'error'
  model: string | null
  model_repo: string | null            // the repository an engine this TensorFold Web owns was started with
  model_name?: string | null           // the served model as a person knows it (the catalog's name, a copy's name)
  model_ids: string[] | null
  warming: boolean
  connections: number
  waiting: number
  decode_tps: number | null            // null: this engine gives no live figure
  prefill_tps: number | null
  draft_acceptance: number | null      // drafted tokens kept since the engine started, when it counts them
  source: 'health' | 'metrics'         // where the figures come from: /health, or /metrics behind an API key
  memory: Record<string, number> | null
  kv_fill: number | null               // share of the KV cache held, 0 to 1; null: not reported by this engine
  streams: { decoding: number; prefilling: number; paused: number } | null
  kept_prompts: number | null
  preemptions: number | null           // requests set aside to free memory since the engine started
  prompt_cache: { restored: number; prompt: number } | null   // prompt tokens restored from kept states (1.0.5)
  last_served?: { t: number; model: string } | null   // from TensorFold Web's usage history
  machine_memory?: { total: number; available: number; engine: number | null; headroom: 'ok' | 'tight' | 'low' }
  findings?: { id: string; level: 'bad' | 'warn' | 'info'; title: string; evidence: string; hint: string }[]
  context_length: number | null
  error: string | null
  managed: boolean
  owned: boolean
  api_key_set: boolean
  key_pending_restart: boolean         // a new key is set but the running engine still uses its previous one
  engine_url: string
  uptime_s: number | null
  command: string[] | null
  tensorfold_available: boolean
  engine_kind: 'native' | 'python' | null   // which engine an owned process runs
  startup: { budget_gib?: number; resident_gib?: number; context_tokens?: number; loaded_s?: number; bytes_per_token?: number
    streams?: number; stream_gib?: number; left_gib?: number; reserve_gib?: number
    prompt_cache_gib?: number; prompt_cache_free_gib?: number; in_use_gib?: number }   // the last four: the CUDA engine's sizing line
}

export type Settings = {
  engine_url: string
  remote: boolean
  model: string | null
  budget_fraction: number | null
  context: number | null
  thinking: 'default' | 'on' | 'off'
  energy_price: number | null
  currency: string
  extra_args: string[]
  share: boolean
  cortheon: boolean                    // Cortheon's web research in the engine, at its next start (TensorFold 1.0.9 on; 1.0.11 for its security fixes)
  api_key: string | null
  api_key_set: boolean
  theme: 'system' | 'light' | 'dark'
  setup_hidden: boolean
}

// a recipe: how a tool reaches the engine, its settings, the checks its test makes, where it was checked for real
export type Snippet = { id: string; name: string; note: string; snippet: string; kind: 'chat' | 'agent' | 'sdk'; protocol: string
  checks: { id: string; label: string }[]; checked: { date?: string; machine?: string; how?: string; not?: string } | null; compose?: boolean; min_context?: number | null; form?: string; where_note?: string | null }
// a chat conversation kept by TensorFold Web; messages are the Chat's own, with their model, engine, settings, figures
export type ConversationRow = { id: string; title: string; updated: number; created: number; messages: number; model: string | null; match?: string; pinned?: boolean; folder?: string | null; replay_of?: string | null; target?: Partial<ReplayTarget> & { drafts?: boolean; url?: string; label?: string; model?: string } | null }
export type ConversationFolder = { name: string; count: number }
// a chat preset: a system prompt and sampling, each field null for the model's own value; kept on TensorFold Web
export type ChatPreset = { name: string; system: string; temperature: number | null; max_tokens: number | null; top_p?: number | null; top_k?: number | null; min_p?: number | null; seed?: number | null; stop?: string[] }
// Living Weights: teachable copies of a model, the facts a text holds, lessons and what they left
export type LivingCopy = { name: string; model: string; path: string; base: string | null; created: number; family: string | null; learnable: boolean; taught: number; lessons: number; bytes: number }
export type LessonItem = { id: string; fact: string; question: string; state: 'queued' | 'learning' | 'kept' | 'partly kept' | 'missed' | 'failed' | 'cancelled' | 'interrupted' | 'refused' | 'rolled back'; message: string | null; steps: number | null; seconds: number | null; before: string | null; after: string | null }
export type LessonJob = { id: string; copy: string; items: LessonItem[]; source: string; started: number; finished: number | null; status: 'running' | 'done' | 'cancelled' | 'failed' | 'rolled back'; error: string | null; phase: string; version: string | null; checks: string[] }
export type Chore = { kind?: string; copy?: string; state?: 'running' | 'done' | 'failed'; error?: string; started?: number; finished?: number; result?: unknown }
export type LivingRun = { id: string; created: number; mode: string; status: string; kept: number | null; tps: number | null; exact: number | null; checked: number | null; right: number | null; items: number | null; mean_kl: number | null; reference: string | null }
export type LivingVersion = { id: string; label: string; at: number; new_bytes: number; kept: number; changed: number }
export type CheckSet = { facts: { q: string; fact: string; expect: string[] }[]; own: { q: string; kind: string }[]; unrelated: string[] }
type ModeSummary = { recalled: number; empty: number; errors?: number; leaks: { question: string; answered_with: string }[]; undone_back?: number }
export type CheckRunInfo = { id: string; label: string; at: number; thinking: boolean; summary: { facts: number; undone?: number; modes: Record<'off' | 'on', ModeSummary> } }
type CheckCell = { before: string | null; after: string | null; changed: boolean; recalled?: boolean }
export type CheckRow = { q: string; kind: string; fact: string | null; off?: CheckCell; on?: CheckCell }
export type LivingView = { root: string; copies: LivingCopy[]; served: string | null; teachable: string | null; busy: boolean; recovering?: string | null; bases: string[]; base_names?: Record<string, string>; job: LessonJob | null; chore: Chore; peers: { id: string; name: string }[]
  creating: { name?: string; base?: string; state?: 'copying' | 'done' | 'failed'; error?: string; started?: number } }
export type PreviewFact = { id: string; fact: string; subject: string; question: string; about_user: boolean; supported: boolean; now: string | null }
export type GraphFact = { id: number; text: string; state: string; neurons: number; questions: string[]; source: string | null }
export type LessonRecord = { job: string; at: number; source: string; fact: string; question: string; outcome: string; message: string | null; steps: number | null; seconds: number | null; before: string | null; after: string | null; found?: { id: number | null; text: string | null; outcome: string }[] }
export type Conversation = { id: string; title: string; created: number; updated: number; messages: any[]; settings: Record<string, unknown>; rev?: number }
// replaying a conversation on another configuration, and two conversations compared turn by turn
export type ReplayTarget = { kind: 'served'; drafts: boolean } | { kind: 'address'; url: string; key?: string; model: string; label?: string }
export type ReplayProgress = { id: string; of: string; done: number; total: number; status: string; error?: string
  turn?: number; question?: string; live_text?: string; live_reasoning?: string }   // the reply being written, to watch it
type ReplySide = { text: string; tps: number | null; ttft: number | null; tokens: number | null; sha: string | null; outcome?: string }
export type ReplyComparison = { a: { id: string; title: string }; b: { id: string; title: string }; same_text: number; same_tokens: number; checked: number
  turns: { question: string; a: ReplySide; b: ReplySide; same_text: boolean; same_tokens: boolean | null; change: number | null }[]   // null: not checkable
  speed: { median_change: number | null; b_faster: number; b_slower: number; p: number; within_noise: boolean; too_few?: boolean } }
// Usage: what the engine served, by day and hour and model, kept across restarts
export type UsageSummary = { days: { day: string; models: Record<string, { prompt: number; completion: number }>; peak_decode: number; peak_prefill: number; requests: number }[]
  hours: { hour: number; models: Record<string, { prompt: number; completion: number }> }[]
  models: { model: string; prompt: number; completion: number; cached: number; requests: number }[]; last_served: { t: number; model: string } | null; since: number | null }

// Fleet: this machine and the others, each read from its own TensorFold Web by this one
export type FleetMachine = { id: string; name: string; url: string | null; token_set: boolean; reachable: boolean; error: string | null; ms: number | null; self: boolean
  hostname?: string | null; chip?: string | null; os?: string | null; backend?: string | null; ram_bytes?: number | null; unified_memory?: boolean | null
  tensorfold_version?: string | null; studio_version?: string | null
  engine?: { status: EngineInfo['status']; model: string | null; model_name?: string | null; decode_tps: number | null; prefill_tps: number | null; connections: number | null; waiting: number | null
    kv_fill: number | null; context_tokens: number | null; uptime_s: number | null; last_served: { t: number; model: string } | null; memory: EngineInfo['machine_memory'] | null; prompt_cache: EngineInfo['prompt_cache']
    findings: { level: 'bad' | 'warn'; title: string }[] } | null }
export type FleetCheck = { id: string; label: string; status: 'pass' | 'warn' | 'fail' | 'skip'; detail: string; seconds: number }
export type FleetPeer = { id: string; name: string; url: string; token_set: boolean }

// what TensorFold Web keeps on disk, and scoped clean-ups (`storage.py`)
export type StorageStore = { id: 'bench' | 'conversations' | 'usage' | 'history'; label: string; bytes: number; count: number; oldest: number | null }
export type StorageModel = { repo: string; bytes: number; last_used: number | null; busy: string | null }
export type CleanResult = { kind: string; matched: number; removed: number; dry_run: boolean }

// the checks of the first-run list and of the diagnostic, each with how to fix it
export type SetupCheck = { id: string; label: string; status: 'pass' | 'warn' | 'fail' | 'skip'; detail: string; fix: string }

export type CheckResult = { id: string; label: string; ok: boolean; skipped?: boolean; detail: string; seconds: number }   // skipped: could not be checked
export type ConnectInfo = { engine_url: string; lan_url: string | null; share: boolean; api_key: string | null; relay_key?: string | null; key_pending_restart: boolean; studio_token_set: boolean; snippets: Snippet[]; kinds?: Record<string, string>; window?: number | null; where?: 'here' | 'away' }

// A TensorFold Web started with --token wants the token on every API and relay call. It is read once from the page's
// address, kept in this browser and taken off the address bar, so it is neither shared by copy-paste nor lost
// when the tab closes.
const TOKEN_KEY = 'tfs-token'
export function studioToken(): string | null {
  try {
    const fromUrl = new URLSearchParams(location.search).get('token')
    if (fromUrl) {
      localStorage.setItem(TOKEN_KEY, fromUrl)
      const clean = new URL(location.href); clean.searchParams.delete('token'); history.replaceState(null, '', clean.toString())
    }
    return localStorage.getItem(TOKEN_KEY)
  } catch { return null }
}
export function setStudioToken(t: string) { try { localStorage.setItem(TOKEN_KEY, t) } catch { /* no storage */ } }
/** A file TensorFold Web makes, fetched with the token in a header (never in the address, which logs keep) and saved. */
export async function download(url: string, filename: string): Promise<void> {
  const r = await fetch(url, { headers: authHeaders() })
  if (r.status === 401 && await askToken()) return download(url, filename)
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail ?? msg } catch { /* keep statusText */ }
    throw new ApiError(msg, r.status)
  }
  saveBlob(await r.blob(), filename)
}

/** A file handed to the browser to save: the one way the page does it (downloads, cards, reports). */
export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob)
  const a = Object.assign(document.createElement('a'), { href: url, download: name })
  document.body.appendChild(a); a.click(); a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 5000)
}
export function authHeaders(): Record<string, string> {
  const t = studioToken()
  return t ? { 'x-studio-token': t } : {}
}

// Every call has a deadline, so a tunnel that stops answering shows up as an error instead of a page frozen on
// its last figures. Starting and stopping an engine get longer ones.
/** An error TensorFold Web answered, with its HTTP status (409: changed elsewhere, among others). */
export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) { super(message); this.status = status }
}

// One question for the token, however many requests met a 401 at once: the page loads with eight of them, and eight
// prompts in a row (some dismissed, then blocked by the browser) left it locked out with the right token in hand.
// The question is a dialog of the page's own, which says where the token is and how to do without typing it.
// Cancelled once, the question is not asked again by the page's own polls (every few seconds, on every screen):
// the 401 is reported instead, and the page offers to ask again (`askTokenAgain`).
let asking: Promise<string | null> | null = null
let declined = false
function askToken(): Promise<string | null> {
  if (declined) return Promise.resolve(null)
  asking ??= tokenDialog().then((t) => { if (t) setStudioToken(t); else declined = true; asking = null; return t })
  return asking
}
export function askTokenAgain(): Promise<string | null> { declined = false; return askToken() }

function tokenDialog(): Promise<string | null> {
  return new Promise((resolve) => {
    const d = document.createElement('dialog')
    d.className = 'token-dialog'
    d.setAttribute('aria-labelledby', 'token-title')
    d.innerHTML = `<form method="dialog">
      <h2 id="token-title">This TensorFold Web needs its access token</h2>
      <p>On the machine running this TensorFold Web, <code>tensorfold-web token</code> prints it, and <code>tensorfold-web token --copy</code> puts it on the clipboard. It is also the file <code>web-token</code> in TensorFold Web's settings folder: <code>~/Library/Application Support/TensorFold Web</code> on a Mac, <code>~/.config/tensorfold-web</code> on Linux.</p>
      <p>No need to type it: where this TensorFold Web is already open, <b>Settings → Open this TensorFold Web in another browser</b> gives a one-time link, and another TensorFold Web's <b>Fleet → Open</b> opens this one signed in.</p>
      <label>Token <input name="token" autocomplete="off" spellcheck="false" required></label>
      <div class="actions"><button class="btn primary" value="ok">Use this token</button><button class="btn quiet" value="cancel" formnovalidate>Cancel</button></div>
    </form>`
    document.body.appendChild(d)
    const input = d.querySelector('input') as HTMLInputElement
    d.addEventListener('close', () => {
      const t = d.returnValue === 'ok' ? input.value.trim() : ''
      d.remove()
      resolve(t || null)
    })
    d.showModal()
    input.focus()
  })
}

/** A one-time link (`?code=`) opened here: the code traded for the token, then taken out of the address. */
export async function redeemFromAddress(): Promise<void> {
  const url = new URL(location.href)
  const code = url.searchParams.get('code')
  if (!code) return
  url.searchParams.delete('code'); history.replaceState(null, '', url.toString())
  try {
    const r = await fetch('/api/access/redeem', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ code }) })
    if (r.ok) setStudioToken((await r.json()).token)
  } catch { /* the page then asks for the token as usual */ }
}

async function j<T>(url: string, init?: RequestInit, timeoutMs = 10000): Promise<T> {
  const run = () => fetch(url, { ...init, signal: AbortSignal.timeout(timeoutMs), headers: { 'content-type': 'application/json', ...authHeaders(), ...(init?.headers ?? {}) } })
  const sent = studioToken()
  let r = await run()
  if (r.status === 401 && !url.startsWith('/engine/')) {
    const now = studioToken()
    if (now && now !== sent) r = await run()               // given meanwhile, by the question another request asked
    else if (await askToken()) r = await run()
  }
  if (!r.ok) {
    let msg = r.statusText
    try { msg = (await r.json()).detail ?? msg } catch { /* keep statusText */ }
    throw new ApiError(msg, r.status)
  }
  return r.json() as Promise<T>
}

export const api = {
  machine: () => j<Machine>('/api/machine'),
  catalog: (refresh = false) => refresh ? j<{ models: Model[]; sizes_known: boolean }>('/api/catalog/refresh', { method: 'POST' }, 60000)
    : j<{ models: Model[]; sizes_known: boolean }>('/api/catalog'),
  verdicts: (id: string) => j<{ verdicts: (Verdict & { fraction: number })[] }>(`/api/catalog/${id}/verdicts`),
  engine: () => j<EngineInfo>('/api/engine'),
  access: () => j<Access>('/api/access'),
  history: (window: number) => j<History>(`/api/history?window=${window}`),
  engineFlags: () => j<{ binary: string | null; version: string | null; flags: Record<string, { value: boolean | null; choices: string[] | null; managed: string | null }> }>('/api/engine/flags', undefined, 30000),
  start: (body: { model: string; budget_fraction?: number | null; context?: number | null }) =>
    j<EngineInfo>('/api/engine/start', { method: 'POST', body: JSON.stringify(body) }, 30000),
  stop: () => j<EngineInfo>('/api/engine/stop', { method: 'POST' }, 40000),
  logs: () => j<{ lines: string[] }>('/api/engine/logs'),
  /** A cookie for the log stream's path only: an EventSource cannot send the token header. */
  logsSession: () => j<{ ok: boolean }>('/api/engine/logs/session', { method: 'POST' }),
  settings: () => j<Settings>('/api/settings'),
  saveSettings: (patch: Partial<Settings>) => j<Settings>('/api/settings', { method: 'PUT', body: JSON.stringify(patch) }),
  connect: (where: 'here' | 'away' = 'here') => j<ConnectInfo>(`/api/connect${where === 'away' ? `?where=away&relay=${encodeURIComponent(location.origin)}` : ''}`),
  connectTest: (recipe: string) => j<{ recipe: string; ok: boolean; checks: CheckResult[]; at: number }>('/api/connect/test', { method: 'POST', body: JSON.stringify({ recipe }) }, 300000),
  conversations: (q = '', offset = 0) => j<{ conversations: ConversationRow[]; more?: boolean }>(`/api/conversations?q=${encodeURIComponent(q)}&offset=${offset}`),
  conversation: (id: string) => j<Conversation>(`/api/conversations/${id}`),
  conversationCreate: (body: Partial<Conversation>) => j<Conversation>('/api/conversations', { method: 'POST', body: JSON.stringify(body) }),
  conversationSave: (id: string, body: Partial<Conversation> & { rev?: number }) => j<Conversation>(`/api/conversations/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  conversationReplay: (id: string, target: ReplayTarget) => j<Conversation & { progress: ReplayProgress }>(`/api/conversations/${id}/replay`, { method: 'POST', body: JSON.stringify({ target }) }, 30000),
  replayProgress: () => j<{ busy: boolean; current: ReplayProgress | null }>('/api/replay'),
  replayCancel: () => j<{ cancelling: boolean }>('/api/replay/cancel', { method: 'POST' }),
  conversationCompare: (a: string, b: string) => j<ReplyComparison>(`/api/conversations/compare?a=${a}&b=${b}`),
  conversationDelete: (id: string) => j<{ deleted: string }>(`/api/conversations/${id}`, { method: 'DELETE' }),
  conversationFolders: () => j<{ folders: ConversationFolder[] }>('/api/conversation-folders'),
  conversationFolderMake: (name: string) => j<{ name: string }>('/api/conversation-folders', { method: 'POST', body: JSON.stringify({ name }) }),
  conversationFolderRename: (old: string, name: string) => j<{ name: string }>(`/api/conversation-folders/${encodeURIComponent(old)}`, { method: 'PUT', body: JSON.stringify({ name }) }),
  brands: () => j<{ brands: CustomBrand[] }>('/api/brands'),
  brandSave: (b: Omit<CustomBrand, 'logo'> & { logo?: string | null }) => j<CustomBrand>(`/api/brands/${b.id}`, { method: 'PUT', body: JSON.stringify(b) }),
  brandRemove: (id: string, kind: CustomBrand['kind']) => j<{ removed: string }>(`/api/brands/${id}?kind=${kind}`, { method: 'DELETE' }),
  chatPresets: () => j<{ presets: ChatPreset[] }>('/api/chat-presets'),
  chatPresetSave: (preset: ChatPreset, replace = false) => j<ChatPreset>('/api/chat-presets', { method: 'POST', body: JSON.stringify({ preset, replace }) }),
  chatPresetRename: (name: string, to: string) => j<ChatPreset>(`/api/chat-presets/${encodeURIComponent(name)}`, { method: 'PUT', body: JSON.stringify({ name: to }) }),
  chatPresetRemove: (name: string) => j<{ removed: string }>(`/api/chat-presets/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  conversationFolderRemove: (name: string) => j<{ removed: string; moved_out: number }>(`/api/conversation-folders/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  rotateKey: () => j<{ api_key_set: boolean; key_pending_restart: boolean; note: string }>('/api/settings/api-key', { method: 'POST' }),
  pull: (repo: string) => j<PullStatus>('/api/pull', { method: 'POST', body: JSON.stringify({ repo }) }),
  pullList: () => j<{ jobs: PullStatus[] }>('/api/pull'),
  pullStatus: (repo: string) => j<PullStatus>(`/api/pull/${repo}`),
  pullCancel: (repo: string) => j<PullStatus>(`/api/pull/${repo}`, { method: 'DELETE' }, 20000),
  benchSuites: () => j<BenchSuites>('/api/bench/suites'),
  benchRuns: () => j<{ runs: BenchRunRow[]; busy: boolean; current: string | null }>('/api/bench/runs'),
  benchRun: (id: string) => j<BenchRun>(`/api/bench/runs/${id}`),
  benchIdentify: (url: string, key?: string) => j<{ url: string; engine: string | null; version: string | null; models: string[]; error: string | null }>('/api/bench/identify', { method: 'POST', body: JSON.stringify({ url, key }) }, 15000),
  // a Fleet machine's engine, identified by this TensorFold Web through that machine's TensorFold Web: no key or token on the page
  benchIdentifyPeer: (peer: string) => j<{ url: string; engine: string | null; version: string | null; models: string[]; error: string | null }>('/api/bench/identify', { method: 'POST', body: JSON.stringify({ peer }) }, 20000),
  benchStart: (body: { mode: BenchMode; suites?: string[]; exactness: boolean; max_context?: number; max_concurrency?: number; full?: boolean; target?: BenchEngineRef & { label?: string }; reference?: BenchEngineRef & { model: string }; kl_load?: number; kl_text?: string }) =>
    j<BenchRun>('/api/bench/runs', { method: 'POST', body: JSON.stringify(body) }, 30000),
  benchCancel: (id: string) => j<{ cancelling: boolean }>(`/api/bench/runs/${id}/cancel`, { method: 'POST' }),
  benchDelete: (id: string) => j<{ deleted: string }>(`/api/bench/runs/${id}`, { method: 'DELETE' }),
  suiteSave: (body: CustomSuiteInput, id?: string) =>
    j<CustomSuite>(id ? `/api/bench/suites/${id}` : '/api/bench/suites', { method: id ? 'PUT' : 'POST', body: JSON.stringify(body) }),
  seriesStart: (body: { configs: SeriesConfig[]; mode: 'quick' | 'profile' | 'quality'; exactness: boolean; suites?: string[] }) =>
    j<SeriesDoc>('/api/bench/series', { method: 'POST', body: JSON.stringify(body) }, 30000),
  seriesList: () => j<{ series: SeriesDoc[]; current: string | null }>('/api/bench/series'),
  series: (id: string) => j<SeriesDoc>(`/api/bench/series/${id}`),
  seriesCancel: (id: string) => j<{ cancelling: boolean }>(`/api/bench/series/${id}/cancel`, { method: 'POST' }),
  benchCompare: (a: string, b: string) => j<Comparison>(`/api/bench/compare?a=${a}&b=${b}`),
  storage: () => j<{ stores: StorageStore[]; models: StorageModel[]; cache: string }>('/api/storage', undefined, 30000),
  storageClean: (kind: string, older_than_days: number | null, dry_run: boolean) => j<CleanResult>('/api/storage/clean', { method: 'POST', body: JSON.stringify({ kind, older_than_days, dry_run }) }, 60000),
  living: () => j<LivingView>('/api/living'),
  livingCreate: (name: string, base: string) => j<LivingView['creating']>('/api/living/copies', { method: 'POST', body: JSON.stringify({ name, base }) }),
  livingRemove: (name: string) => j<{ removed: string }>(`/api/living/copies/${encodeURIComponent(name)}`, { method: 'DELETE' }, 120000),
  livingPreview: (text: string) => j<{ copy: string | null; facts: PreviewFact[] }>('/api/living/preview', { method: 'POST', body: JSON.stringify({ text }) }, 600000),
  livingTeach: (facts: { fact: string; question: string }[], opts: { version_before: boolean; check: boolean; copy: string }) => j<LessonJob>('/api/living/teach', { method: 'POST', body: JSON.stringify({ facts, source: 'studio', ...opts }) }),
  livingVersions: (name: string) => j<{ versions: LivingVersion[]; store_bytes: number }>(`/api/living/copies/${encodeURIComponent(name)}/versions`),
  livingVersionSave: (name: string, label: string) => j<Chore>(`/api/living/copies/${encodeURIComponent(name)}/versions`, { method: 'POST', body: JSON.stringify({ label }) }),
  livingVersionRestore: (name: string, vid: string) => j<Chore>(`/api/living/copies/${encodeURIComponent(name)}/versions/${vid}/restore`, { method: 'POST' }),
  livingVersionBranch: (name: string, vid: string, into: string) => j<Chore>(`/api/living/copies/${encodeURIComponent(name)}/versions/${vid}/branch`, { method: 'POST', body: JSON.stringify({ name: into }) }),
  livingVersionRemove: (name: string, vid: string) => j<{ removed: string; freed_bytes: number }>(`/api/living/copies/${encodeURIComponent(name)}/versions/${vid}`, { method: 'DELETE' }),
  livingChecks: (name: string) => j<{ set: CheckSet; runs: CheckRunInfo[] }>(`/api/living/copies/${encodeURIComponent(name)}/checks`),
  livingChecksSet: (name: string, questions: { q: string; kind: string }[]) => j<CheckSet>(`/api/living/copies/${encodeURIComponent(name)}/checks`, { method: 'PUT', body: JSON.stringify({ questions }) }),
  livingChecksRun: (name: string, restart: boolean) => j<Chore>(`/api/living/copies/${encodeURIComponent(name)}/checks/run`, { method: 'POST', body: JSON.stringify({ restart }) }),
  livingRuns: (name: string) => j<{ runs: LivingRun[] }>(`/api/living/copies/${encodeURIComponent(name)}/runs`),
  livingKl: (peer: string) => j<BenchRun>('/api/living/kl', { method: 'POST', body: JSON.stringify({ peer }) }, 60000),
  livingForget: (name: string, fact: string) => j<Chore>(`/api/living/copies/${encodeURIComponent(name)}/forget`, { method: 'POST', body: JSON.stringify({ fact }) }),
  livingPeerCopies: (peer: string) => j<{ copies: { name: string; base: string | null; bytes: number; taught: number }[] }>(`/api/living/peer/${peer}`),
  livingFetch: (peer: string, name: string, as: string) => j<Chore>('/api/living/fetch', { method: 'POST', body: JSON.stringify({ peer, name, as }) }),
  livingCheckCompare: (name: string, rid: string, against: string) => j<CheckRunInfo & { compare: CheckRow[] }>(`/api/living/copies/${encodeURIComponent(name)}/checks/${rid}?against=${against}`),
  livingCancel: (now = false) => j<{ cancelling: boolean }>('/api/living/cancel', { method: 'POST', body: JSON.stringify({ now }) }),
  // the token in a header, never in an address; a missing token asks for it, as every other call does
  livingViewSession: () => j<{ ok: boolean }>('/living-view/session', { method: 'POST' }),
  livingFacts: (name: string) => j<{ facts: GraphFact[]; lessons: LessonRecord[] }>(`/api/living/copies/${encodeURIComponent(name)}/facts`),
  accessLink: () => j<{ code: string; expires_in: number }>('/api/access/link', { method: 'POST' }),
  fleetOpen: (id: string) => j<{ url: string; signed_in: boolean }>(`/api/fleet/${id}/open`, { method: 'POST' }, 20000),
  storageRemoveModel: (repo: string, dry_run: boolean) => j<{ repo: string; freed: number; dry_run: boolean }>('/api/storage/models/remove', { method: 'POST', body: JSON.stringify({ repo, dry_run }) }, 120000),
  diagnostic: () => j<{ checks: SetupCheck[]; at: number }>('/api/diagnostic', undefined, 30000),
  usage: (days: number) => j<UsageSummary>(`/api/usage?days=${days}`),
  fleet: () => j<{ machines: FleetMachine[]; at: number }>('/api/fleet', undefined, 15000),
  fleetTest: (url: string, token: string) => j<{ ok: boolean; checks: FleetCheck[] }>('/api/fleet/test', { method: 'POST', body: JSON.stringify({ url, token }) }, 30000),
  fleetAdd: (body: { url: string; token: string; name: string }) => j<FleetPeer>('/api/fleet', { method: 'POST', body: JSON.stringify(body) }, 30000),
  fleetUpdate: (id: string, body: { name?: string; token?: string }) => j<FleetPeer>(`/api/fleet/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  fleetRemove: (id: string) => j<{ removed: string }>(`/api/fleet/${id}`, { method: 'DELETE' }),
  toolEvalInstall: () => j<{ path: string; version: string | null }>('/api/bench/tools/install', { method: 'POST' }, 330000),
  suiteCopy: (builtinId: string) => j<CustomSuite>(`/api/bench/suites/copy/${builtinId}`, { method: 'POST' }),
  suiteDelete: (id: string) => j<{ deleted: string }>(`/api/bench/suites/${id}`, { method: 'DELETE' }),
}

// The Bench: a run's summary per suite and overall, as `tensorfold_web/bench.py` computes it.
export type BenchBlock = {
  prompts: number; tokens: number | null; tps: number | null; tps_no_drafts: number | null; gain: number | null
  ttft: number | null; ttft_answer?: number | null; prefill_s: number | null; gap_p50: number | null; gap_p95: number | null
  acceptance: number | null; checked: number; exact: number; errors: number
}
export type SweepPoint = { size: number; prompt_tokens: number | null; cached: number | null; prefill_s: number | null; ttft: number | null; prefill_tps: number | null; error: string | null; estimated?: boolean; weak?: boolean }
export type LoadLevel = { concurrency: number; requests: number; errors: number; total_tps: number | null; per_request_tps: number | null; ttft_p50: number | null; ttft_p95: number | null }
export type QualityCount = { items: number; passed: number; unscored?: number }   // unscored: a request that failed, not graded
export type BenchSummary = { overall: BenchBlock; suites: Record<string, BenchBlock>; sweep?: SweepPoint[]; load?: LoadLevel[]; quality?: { overall: QualityCount; categories: Record<string, QualityCount> } }
export type BenchResult = {
  suite: string; prompt: string; draft: boolean; error: string | null; text: string
  same?: boolean; compared_by?: 'token_sha' | 'text'
  passed?: boolean; read?: string; expected?: string | null     // a quality item: its score, what was read, the answer
  kl?: { positions: number; agree: number; differs_at: number | null; identical: boolean }   // a KL run's prompt
  m: { tokens: number | null; tps: number | null; ttft: number | null; prefill_s: number | null; timing: 'engine' | 'studio'
    gap_p50: number | null; gap_p95: number | null; finish: string | null; sha: string | null; runtime: Record<string, unknown> & { acceptance_rate?: number } }
}
// another engine for the Bench: an address with the key typed, or a Fleet machine resolved by this TensorFold Web
export type BenchEngineRef = { url: string; key?: string; model?: string } | { peer: string; model?: string }
export type BenchMode = 'quick' | 'profile' | 'prefill' | 'load' | 'quality' | 'tools' | 'kl'
export type BenchRunRow = { id: string; created: number; finished: number | null; mode: BenchMode; status: string; model: string | null; engine: string | null; summary: BenchSummary | null; error: string | null; kl?: { mean_kl: number | null } }
export type BenchRun = BenchRunRow & {
  exactness: boolean; suites: string[]; studio: string; notes: string[]; window?: number; max_context?: number | null
  protocol: { temperature: number; top_p: number; warmup_tokens: number; thinking: string; output_format: string }
  // from schema 2: the prompts' version, and how thinking was turned off (absent on older runs)
  suites_version?: { version: number | null; digest: string }
  thinking?: { sent: string | null; off: boolean; refused: string[] } | null
  machine: Partial<Machine> | null
  progress: { done: number; total: number; current: string | null
    // from schema 2: what the run is doing, and its estimate of the time left, made at `eta_at` (server clock)
    phase?: 'starting' | 'idle' | 'warm-up' | 'measuring' | string; measuring_since?: number | null
    eta_s?: number | null; eta_at?: number | null; stopping?: boolean }
  server_time?: number; error_detail?: string
  findings?: string[]; share_text?: string; suite_names?: Record<string, string>
  tools?: { version: string | null; final_score: number | null; rating: string | null; points: number | null; max_points: number | null; scenarios: number | null; categories: { category: string; label: string; percent: number; pass_count: number; partial_count: number; fail_count: number }[] }    // written by the server from the run, for the report and sharing
  memory: { engine_peak_bytes: number | null; system_peak_bytes: number; samples: number } | null
  hardware: BenchHardware | null
  results: BenchResult[]
  // a KL run: what it compared with, the load, and the figures (`kl.py`)
  kl_reference?: { kind: 'drafts-off' } | { kind: 'address'; url: string; model: string }; kl_load?: number
  kl_text?: string | null                      // a KL run on a fixed text: every position of it scored
  tags?: { peer?: string; reference_peer?: string; [k: string]: unknown }   // a Fleet machine measured or compared with
  kl_load_verified?: boolean
  kl_load_done?: { ok: number; failed: number; late?: number; full_share?: number | null; last_error: string | null }
  kl?: { positions: number; prompts: number; mean_kl: number; median_kl: number; p95_kl: number; p99_kl: number; max_kl: number; same_top: number; zero_kl: number; identical: number; differs_at_median: number | null; tokens: number; requests: number; top: number; same_distribution?: number; unscored?: number
    worst: { prompt: string; at: number; kl: number; token?: string; ref: string; cand: string; same_top: boolean }[] }
}
// power and heat sampled during a run; `series` rows are [seconds, watts, cpu °C, gpu °C, engine bytes]
export type BenchHardware = {
  source: string; scope: string; whole_machine: boolean; estimate: boolean
  idle_watts: number | null; avg_watts: number | null; peak_watts: number | null; measured_s: number
  energy_j: number | null; energy_above_idle_j: number | null; output_tokens: number
  j_per_token: number | null; j_per_token_above_idle: number | null
  j_per_1k_prompt?: number | null; j_per_1k_prompt_above_idle?: number | null; per_token_note?: string   // prefill: per prompt token read
  peak_cpu_c: number | null; peak_gpu_c: number | null; peak_c: number | null
  series: [number, number | null, number | null, number | null, number | null, (boolean | null)?][]
  // from schema 2 (absent on older runs)
  window_s?: number; coverage?: number | null
  throttle?: { reported: boolean; throttled: boolean | null; seconds: number | null; reasons: string[] }
  cost?: { price_per_kwh: number; currency: string; run: number; per_million_tokens: number | null; per_million_tokens_above_idle?: number } | null
}
export type BenchSuites = { modes: Record<BenchMode, string>; sizes: number[]; about: string; suites: { id: string; name: string; prompts: number; quick: number }[]; custom: CustomSuite[]
  quality: { about: string; version: number; digest: string; categories: { id: string; name: string; items: number }[] }
  tools: { installed: boolean } }
// compare mode: configurations of the served engine measured one after the other (`tensorfold_web/series.py`)
export type SeriesConfig = { model: string; context: number | null; budget_fraction: number | null; drafts: boolean; label: string }
export type SeriesRow = {
  index: number; label: string; config: SeriesConfig; run: string | null; status: string | null; error: string | null
  served: { model: string | null; context_tokens?: number | null; budget_fraction?: number | null; loaded_s?: number | null } | null
  tps: number | null; tps_no_drafts: number | null; ttft: number | null; exact: number | null; checked: number | null
  j_per_token: number | null; engine_peak_bytes: number | null; right: number | null; items: number | null
  vs_first: null | { refused: string } | { speed: Comparison['speed']; same_text: number; pairs: number; findings: string[]; quality?: Comparison['exactness'] }
}
export type SeriesDoc = { id: string; created: number; status: string; mode: string; configs: SeriesConfig[]; runs: (string | null)[]; rows: SeriesRow[]
  progress: { index: number; phase: string; run: string | null } | null; restore_error?: string; error?: string }
// two runs side by side, as `tensorfold_web/compare.py` pairs and tests them
export type CompareSide = { tps: number | null; ttft: number | null; tokens: number | null; prefill_tps: number | null; error: string | null; pass: boolean | null }
export type ComparePair = { suite: string; suite_name: string; prompt: string; a: CompareSide; b: CompareSide; same_text: boolean; same_tokens: boolean; change: number | null }
export type CompareWho = { id: string; created: number; mode: BenchMode; model: string | null; engine: string | null; chip: string | null; ram_bytes: number | null; studio: string; suites_version: { version: number | null; digest: string } | null; status: string }
export type CompareOverall = { tps: number | null; tps_no_drafts: number | null; gain: number | null; ttft: number | null; exact: number | null; checked: number | null; j_per_token: number | null; avg_watts: number | null; engine_peak_bytes: number | null }
export type Comparison = {
  a: CompareWho; b: CompareWho; notes: string[]; pairs: ComparePair[]; findings: string[]; same_text: number
  by_suite?: Record<string, { name: string; checked: number; only_a: number; only_b: number; p: number }>
  speed: { median_change: number | null; b_faster: number; b_slower: number; p: number; within_noise: boolean; too_few?: boolean; measure: string }
  exactness: { checked: number; both: number; only_a: number; only_b: number; neither: number; p: number; within_noise: boolean; too_few?: boolean }
  overall: { a: CompareOverall; b: CompareOverall }
}
// a suite the user wrote or imported; the digest covers what is sent
export type CustomPrompt = { id?: string; user: string; system?: string | null; max_tokens: number; history?: { role: 'user' | 'assistant'; content: string }[] }
export type CustomSuiteInput = { name: string; about?: string; prompts: CustomPrompt[] }
export type CustomSuite = CustomSuiteInput & { id: string; created: number; updated: number; digest: string; about: string }

export type PullStatus = {
  repo: string
  state: 'idle' | 'running' | 'done' | 'failed' | 'cancelled'
  done_bytes: number
  total_bytes: number
  fraction: number
  error: string | null
}

export const GB = 1024 ** 3
export { bytes as gb } from './units'          // sizes roll over to MB and TB (`units.ts`)
export const num = (n: number | null | undefined) => (n == null ? '—' : n.toLocaleString('en-US'))   // — : not known
export const shortName = (id: string) => id.split('/').pop() ?? id
