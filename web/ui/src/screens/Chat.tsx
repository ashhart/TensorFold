import { useCallback, useEffect, useRef, useState } from 'react'
import { api, authHeaders, download, num, type ConversationRow } from '../api'
import { copyText } from '../shareCard'
import type { Ctx } from '../ctx'
import { ModelLogo } from '../brandLogo'
import { showSub, subOf, useSub } from '../route'
import Markdown from '../Markdown'
import { ReplayForm, ReplyCompare, SpeedupCheck } from './ChatReplay'
import { Saves } from '../saves'
import { ConversationList, namesOf, sameQuestions } from './ChatList'
import { PRESETS, PresetEditor, sampling, usePresets, type Preset } from './ChatPresets'
import { sseEvents } from '../sse'
import { drawSeed } from '../seed'
import { charsPerToken, prefillRate } from '../liverate'
import { LivePills, WebSources, rate, secs, type Cortheon, type Live } from './ChatLive'
import { check, checkJson, DEFAULT_WORKSHOP, isLocal, parseTools, runLocal, toolChoice, WorkshopPanel, type Schema, type Workshop } from './ChatWorkshop'

type Outcome = 'done' | 'stopped' | 'interrupted' | 'error' | 'length'
// each reply keeps what wrote it and how: the model, the engine, the thinking setting and budget
// the workshop's messages: an assistant's tool calls (each checked against its tool), a tool's result, a JSON check
type Call = { id: string; name: string; arguments: string; check?: string }
// the settings a reply was written with, recorded on it: a replay or a speed-up check uses these, not today's options
export type Request = { thinking: 'default' | 'on' | 'off'; budget: number | null; system: string; max_tokens: number | null; temperature: number | null; top_p?: number | null; top_k?: number | null; min_p?: number | null; seed?: number | null; seed_drawn?: boolean; stop?: string[] }
type Msg = { role: 'user' | 'assistant' | 'tool'; content: string; request?: Request; tool_calls?: Call[]; tool_call_id?: string; name?: string; json?: { ok: boolean; detail: string }; reasoning?: string; model?: string; engine?: string; thinking?: string; preset?: string; outcome?: Outcome; error?: string; stats?: Stats; at?: number; cortheon?: Cortheon }
// `timing`: 'engine' when the reply's `tensorfold` block gave the speed and the first-token time, 'page' when this
// page's clock did (other engines, or the Python line). `measured` says who counted the tokens.
// tokens and tps only when the engine counted them; events: what the page received, each possibly several tokens
type Stats = { tokens: number | null; tps: number | null; events?: number; ttft: number; prompt?: number; cached?: number; reasoning?: number; measured: 'engine' | 'counted'
  timing?: 'engine' | 'page'; budget?: number; sha?: string; drafts?: boolean; acceptance?: number; perRound?: number; prefill?: number }
type Runtime = { tokens_per_second?: number; time_to_first_token?: number; token_sha?: string; drafts?: boolean; acceptance_rate?: number | null
  tokens_per_round?: number | null; prefill_seconds?: number }

const STORE = 'tfs-chat-v2'
const STORE_V1 = 'tfs-chat'                 // the first release's key: same shape, messages without a model name
type Stored = { msgs: Msg[]; thinking: 'default' | 'on' | 'off'; budget?: number }
// Thinking budgets offered, in tokens; 0 lets the model think as long as it wants. Sent as `thinking_budget`, which
// TensorFold honours per request; servers that do not know the field ignore it.
const BUDGETS = [0, 1024, 2048, 4096, 8192]
const DEFAULT_BUDGET = 2048
/** The conversation this browser kept before TensorFold Web kept them (taken over once into TensorFold Web, then dropped). */
function load(): Stored | null {
  try {
    const raw = localStorage.getItem(STORE)
    if (raw) return JSON.parse(raw) as Stored
    const old = localStorage.getItem(STORE_V1)
    if (!old) return null
    const v1 = JSON.parse(old) as Stored
    localStorage.setItem(STORE, JSON.stringify(v1))
    localStorage.removeItem(STORE_V1)
    return v1
  } catch { return null }
}


/** The text as it arrives, revealed at a pace that follows the stream, so it reads smoothly; at once when motion is
 * reduced or the reply is complete. */
function useSmooth(target: string, live: boolean): string {
  const [shown, setShown] = useState(target.length)
  const reduce = useRef(matchMedia('(prefers-reduced-motion: reduce)').matches)
  useEffect(() => {
    if (!live || reduce.current) { setShown(target.length); return }
    let raf = 0
    const step = () => {
      setShown((n) => {
        if (n >= target.length) return n
        return Math.min(target.length, n + Math.max(1, Math.ceil((target.length - n) / 8)))
      })
      raf = requestAnimationFrame(step)
    }
    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [target, live])
  return live && !reduce.current ? target.slice(0, Math.min(shown, target.length)) : target
}

function Reply({ text, live, onCopy }: { text: string; live: boolean; onCopy: (t: string) => void }) {
  const shown = useSmooth(text, live)
  return shown ? <Markdown text={shown} onCopy={onCopy} /> : null
}

const STARTERS = [
  "Write a short story, about 300 words, about a lighthouse keeper's last night on the job.",
  'Explain, in plain words, what a mortgage rate cap is and when it matters.',
  'Write a Python script that renames the photos in a folder by the date they were taken.',
]

export default function Chat({ engine, settings, toast, refreshEngine }: Ctx) {
  const stored = useRef(load())
  const [msgs, setMsgs] = useState<Msg[]>([])
  // the conversations TensorFold Web keeps: the one open here, the list on the side, a search over them
  const [convId, setConvId] = useState<string | null>(null)
  const [convs, setConvs] = useState<ConversationRow[]>([])
  const [q, setQ] = useState('')
  const opened = useRef<unknown[] | null>(null)            // the messages as last opened: not saved back unchanged
  const [more, setMore] = useState(false)                 // older conversations beyond the first page
  // each list asked (a search, a refresh) is a new generation: a page that comes back after it belongs to the old list
  const listGen = useRef(0)
  const shownGen = useRef(0)                              // the generation of the list on screen
  const refreshList = useCallback((query = q) => {
    const gen = ++listGen.current
    return api.conversations(query).then((r) => {
      if (gen !== listGen.current) return
      shownGen.current = gen; setConvs(r.conversations); setMore(!!r.more)
    }).catch(() => {})
  }, [q])
  const loadMore = () => {
    const gen = listGen.current
    if (shownGen.current !== gen) return Promise.resolve()   // a newer list is on its way: its offsets are not these
    return api.conversations(q, convs.filter((c) => !c.pinned).length).then((r) => {   // the offset counts the unpinned
      if (gen !== listGen.current) return
      setConvs((old) => [...old, ...r.conversations.filter((c) => !old.some((o) => o.id === c.id))]); setMore(!!r.more)
    }).catch((e) => toast(String(e), 'bad'))
  }
  const [text, setText] = useState('')
  const [thinking, setThinking] = useState<'default' | 'on' | 'off'>(stored.current?.thinking ?? settings?.thinking ?? 'default')
  const [budget, setBudget] = useState<number>(stored.current?.budget ?? DEFAULT_BUDGET)
  const [preset, setPreset] = useState<Preset>(PRESETS[0])
  const { mine, reload: reloadPresets } = usePresets(toast)
  const [editing, setEditing] = useState(false)
  const [starting, setStarting] = useState(false)
  const [replaying, setReplaying] = useState(false)
  const [checking, setChecking] = useState(false)                      // the speed-up checked on this conversation
  const [ws, setWs] = useState<Workshop>(DEFAULT_WORKSHOP)
  const [showWs, setShowWs] = useState(false)
  const [options, setOptions] = useState(false)                       // thinking, presets, tools: one click away
  const [results, setResults] = useState<Record<string, string>>({})   // what the user types as a tool's result
  const unconstrained = useRef(new Set<string>())   // engines that refused response_format: JSON is then asked in the prompt
  const [comparing, setComparing] = useState<[string, string] | null>(null)
  const [speedupPair, setSpeedupPair] = useState(false)                // the comparison is a speed-up check
  const [streaming, setStreaming] = useState(false)
  // the reply being written, as this page sees it: when it was sent, its first token, the stream events so far
  const [live, setLive] = useState<Live | null>(null)
  const [webOff, setWebOff] = useState(false)               // this conversation asks without Cortheon's web research
  const [error, setError] = useState<string | null>(null)
  const [saveError, setSaveError] = useState<false | 'failed' | 'conflict'>(false)
  const abort = useRef<AbortController | null>(null)
  const generation = useRef(0)                             // which request may still touch the messages
  const box = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)
  const stickToBottom = useRef(true)

  useEffect(() => {
    const el = box.current
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight
  }, [msgs])
  // at the first visit: the conversation this browser kept is taken over by TensorFold Web, then the most recent one opens
  useEffect(() => {
    const wanted = subOf('chat')                         // read before anything rewrites the address
    ;(async () => {
      const old = stored.current
      if (old?.msgs?.length) {
        try {
          await api.conversationCreate({ messages: old.msgs.map((m) => (m.role === 'assistant' && !m.outcome ? { ...m, outcome: 'interrupted' } : m)), settings: { thinking: old.thinking, budget: old.budget, from: 'this browser' } })
          localStorage.removeItem(STORE); stored.current = null
        } catch { /* kept in the browser; tried again next time */ }
      }
      const r = await api.conversations().catch(() => ({ conversations: [] as ConversationRow[] }))
      setConvs(r.conversations)
      if (r.conversations[0] && !wanted) openConversation(r.conversations[0].id)   // an address naming one opens that one
    })()
  }, [])
  useSub('chat', (id) => { openConversation(id) })          // #chat/<conversation>
  useEffect(() => {                                        // the address names the open conversation while Chat is shown
    const put = () => { if (convId) showSub('chat', convId) }   // New conversation clears it itself
    put()
    addEventListener('hashchange', put)
    return () => removeEventListener('hashchange', put)
  }, [convId])
  // saved by TensorFold Web while a reply streams too, so a reload keeps what arrived; the first message creates it.
  // A change is saved within 800 ms, and at least every 3 s while a reply keeps changing it; switching conversation
  // sends what is pending first. Saves go out in order on one queue per conversation, each on the revision the server
  // gave the last (saves.ts): one made on an older revision is refused, never written. `convGen` numbers what the page
  // shows: it moves only when other messages replace the shown ones, so a save or a creation of the conversation that
  // was shown never reaches the one shown after it.
  type SaveJob = { body: Record<string, unknown>; convId: string | null; gen: number }
  const pending = useRef<SaveJob | null>(null)
  const lastSave = useRef(0)
  const convGen = useRef(0)
  const shownId = useRef<string | null>(null)
  shownId.current = convId
  const refreshRef = useRef(refreshList)
  refreshRef.current = refreshList
  const saves = useRef<Saves | null>(null)
  saves.current ??= new Saves(
    { create: (b) => api.conversationCreate(b).then((c) => ({ id: c.id, rev: c.rev })), save: (id, b) => api.conversationSave(id, b).then((c) => ({ id: c.id, rev: c.rev })) },
    (gen, id) => { if (convGen.current === gen) setConvId(id) },
    (outcome, id) => {
      // the notice is about the conversation shown: another one's save says nothing of it
      if (id === shownId.current || id === null) setSaveError(outcome === 'saved' ? false : outcome)
      if (outcome === 'saved') { lastSave.current = Date.now(); refreshRef.current() }
    })
  const saveNow = (job: SaveJob) => saves.current!.save(job.gen, job.convId, job.body)
  const flushSave = () => { const j = pending.current; pending.current = null; if (j) void saveNow(j) }
  useEffect(() => {
    if (!msgs.length || msgs === opened.current) return
    const job: SaveJob = { body: { messages: msgs, settings: { thinking, budget, preset, workshop: ws } }, convId, gen: convGen.current }
    pending.current = job
    const wait = Math.max(0, Math.min(800, 3000 - (Date.now() - lastSave.current)))
    const t = setTimeout(() => { if (pending.current === job) { pending.current = null; void saveNow(job) } }, wait)
    return () => clearTimeout(t)
  }, [msgs, thinking, budget, preset, ws])
  useEffect(() => { const t = setTimeout(() => refreshList(q), 250); return () => clearTimeout(t) }, [q])
  useEffect(() => {                                        // the composer grows with the text, up to its limit
    const el = input.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(220, el.scrollHeight)}px`
  }, [text])
  useEffect(() => {
    const focus = () => input.current?.focus()
    const fresh = () => newConversation(true)
    addEventListener('tfs:focus-composer', focus)
    addEventListener('tfs:new-conversation', fresh)
    return () => { removeEventListener('tfs:focus-composer', focus); removeEventListener('tfs:new-conversation', fresh) }
  })

  const model = engine?.model ?? ''
  const engineLabel = engine?.managed === false ? `the engine at ${engine?.engine_url}` : engine?.engine_kind === 'native' ? 'TensorFold native' : engine?.engine_kind === 'python' ? 'TensorFold, Python line' : 'TensorFold'
  const ready = engine?.status === 'running'
  const webResearch = !!engine?.command?.includes('--cortheon')     // the engine runs Cortheon (Settings)
  // the model served is one TensorFold learns into: "Teach this" takes a message to Teach, which makes the copy first
  const [learnsHere, setLearnsHere] = useState(false)
  useEffect(() => {
    const repo = engine?.model_repo
    if (!repo || repo.startsWith('living:') || !engine?.owned) { setLearnsHere(false); return }
    api.living().then((v) => setLearnsHere(!v.teachable && v.bases.includes(repo))).catch(() => setLearnsHere(false))
  }, [engine?.model_repo, engine?.owned])
  const destination = engine?.engine_url ?? 'the engine'

  // the conversation as the API takes it: tool calls and tool results in OpenAI's shape
  const toApi = (list: Msg[]) => [...(preset.system ? [{ role: 'system', content: preset.system }] : []), ...list.map((m) =>
    m.role === 'tool' ? { role: 'tool', tool_call_id: m.tool_call_id, content: m.content }
      : m.role === 'assistant' && m.tool_calls?.length ? { role: 'assistant', content: m.content || null, tool_calls: m.tool_calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } })) }
        : { role: m.role, content: m.content })]
  const workshopOn = ws.tools || ws.output !== 'text'

  /** What a reply keeps of how it was asked: the request (with the seed the page drew, if any), so a replay and the
   * speed-up check can ask it again the same way. */
  const replyContext = (drawn: number | null) => {
    const request: Request = { thinking, budget: thinking !== 'off' && budget ? budget : null, system: preset.system || '', max_tokens: preset.max_tokens ?? null, temperature: preset.temperature ?? null, top_p: preset.top_p ?? null, top_k: preset.top_k ?? null, min_p: preset.min_p ?? null, seed: preset.seed ?? drawn, ...(drawn != null ? { seed_drawn: true } : {}), stop: preset.stop?.filter(Boolean) ?? [] }
    return { model, engine: engineLabel, thinking: thinking === 'off' ? 'off' : `${thinking}${budget ? `, up to ${budget}` : ''}`, preset: preset.name, at: Date.now() / 1000, request }
  }
  /** A workshop turn: whole replies, tool calls checked, the page's own tools run by themselves, JSON checked. */
  // `continuing`: the turn goes on after tool results, so a forced tool choice is not forced again
  const workshopTurn = async (start: Msg[], continuing = false) => {
    const mine = ++generation.current
    const { tools } = parseTools(ws.toolsText)
    let schema: Schema | null = null
    if (ws.output === 'schema') { try { schema = JSON.parse(ws.schemaText) as Schema } catch { setError('The JSON schema is not JSON.'); return } }
    let history = start
    setStreaming(true); setError(null)
    const ctl = new AbortController(); abort.current = ctl
    try {
      for (let round = 0; round < 6; round++) {
        const body: Record<string, unknown> = { model, stream: false, messages: toApi(history) }
        Object.assign(body, sampling(preset))
        const drawn = drawSeed(preset)
        if (drawn != null) body.seed = drawn
        if (webResearch && webOff) body.cortheon = false
        if (thinking !== 'default') body.chat_template_kwargs = { enable_thinking: thinking === 'on' }
        if (thinking !== 'off' && budget) body.thinking_budget = budget
        if (ws.tools && tools?.length) { body.tools = tools; body.tool_choice = toolChoice(ws, continuing ? round + 1 : round); if (ws.oneCall) body.parallel_tool_calls = false }
        const asked = ws.output === 'json' ? 'Reply with one JSON object only, no other text.'
          : ws.output === 'schema' ? `Reply with one JSON object only, no other text, matching this JSON schema:\n${JSON.stringify(schema)}` : ''
        const prompted = () => {   // one system message: the preset's, with the format added
          const base = toApi(history) as { role: string; content: unknown }[]
          body.messages = base[0]?.role === 'system' ? [{ ...base[0], content: `${base[0].content}\n\n${asked}` }, ...base.slice(1)] : [{ role: 'system', content: asked }, ...base]
          delete body.response_format
        }
        if (ws.output === 'json') body.response_format = { type: 'json_object' }
        if (ws.output === 'schema' && schema) body.response_format = { type: 'json_schema', json_schema: { name: 'reply', schema, strict: true } }
        if (body.response_format && unconstrained.current.has(engineLabel)) prompted()
        const t0 = performance.now()
        const post = () => fetch('/engine/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', ...authHeaders() }, body: JSON.stringify(body), signal: ctl.signal })
        let r = await post()
        let d = await r.json().catch(() => ({}))
        // an engine without structured output: the format is asked in the prompt instead, and the reply still checked
        if (r.status === 400 && body.response_format && /response_format|structured output/i.test(d.error?.message ?? '')) {
          unconstrained.current.add(engineLabel); prompted()
          r = await post(); d = await r.json().catch(() => ({}))
        }
        if (!r.ok) throw new Error(d.error?.message ?? d.detail ?? `${r.status} ${r.statusText}`)
        if (generation.current !== mine) return
        const m = d.choices?.[0]?.message ?? {}
        // the native engine keeps its draft counters in `speculative`, beside the `tensorfold` block
        const rt = d.tensorfold ? { ...(d.tensorfold as Runtime), ...(d.speculative as Runtime | undefined) } : undefined
        const calls: Call[] = (m.tool_calls ?? []).map((c: { id: string; function: { name: string; arguments: string } }) => {
          const def = tools?.find((t) => t.function.name === c.function.name)
          let errs: string[]
          try { errs = def ? check(JSON.parse(c.function.arguments || '{}'), def.function.parameters) : ['no tool of this name'] } catch { errs = ['arguments are not JSON'] }
          return { id: c.id, name: c.function.name, arguments: c.function.arguments, check: errs.length ? errs.join('; ') : undefined }
        })
        const n = d.usage?.completion_tokens ?? null
        const reply: Msg = { role: 'assistant', content: m.content ?? '', reasoning: m.reasoning_content ?? m.reasoning ?? undefined, ...replyContext(drawn),
          outcome: d.choices?.[0]?.finish_reason === 'length' ? 'length' : 'done', tool_calls: calls.length ? calls : undefined,
          json: !calls.length && ws.output !== 'text' ? (() => { const j = checkJson(m.content ?? '', schema)
            return body.response_format ? j : { ...j, detail: `${j.detail} (asked in the prompt: this engine does not constrain replies to JSON)` } })() : undefined,
          stats: { tokens: n, tps: rt?.tokens_per_second ?? (n ? n / Math.max(0.001, (performance.now() - t0) / 1000) : null), ttft: rt?.time_to_first_token ?? 0, prompt: d.usage?.prompt_tokens,
            cached: d.usage?.prompt_tokens_details?.cached_tokens, reasoning: d.usage?.completion_tokens_details?.reasoning_tokens,
            measured: d.usage ? 'engine' : 'counted', timing: rt?.tokens_per_second != null ? 'engine' : 'page', sha: rt?.token_sha, drafts: rt?.drafts, acceptance: rt?.acceptance_rate ?? undefined,
            perRound: rt?.tokens_per_round ?? undefined, prefill: prefillRate(rt, d.usage?.prompt_tokens, d.usage?.prompt_tokens_details?.cached_tokens) },
          ...(d.cortheon ? { cortheon: d.cortheon as Cortheon } : {}) }
        history = [...history, reply]
        setMsgs(history)
        if (!calls.length) break
        // the page's own tools answer by themselves; any other waits for the user
        if (!ws.auto || !calls.every((c) => isLocal(c.name))) break
        const answers: Msg[] = calls.map((c) => {
          let args: Record<string, unknown> = {}
          try { args = JSON.parse(c.arguments || '{}') } catch { /* the check above says it */ }
          return { role: 'tool', tool_call_id: c.id, name: c.name, content: JSON.stringify(runLocal(c.name, args)) }
        })
        history = [...history, ...answers]
        setMsgs(history)
      }
    } catch (e) {
      if ((e as Error).name !== 'AbortError') setError(String(e))
    } finally { if (generation.current === mine) { setStreaming(false); abort.current = null } }
  }
  const pendingCalls = (() => {
    const last = msgs[msgs.length - 1]
    return last?.role === 'assistant' && last.tool_calls?.length ? last.tool_calls : []
  })()
  const giveResults = () => {
    const answers: Msg[] = pendingCalls.map((c) => ({ role: 'tool', tool_call_id: c.id, name: c.name, content: results[c.id] ?? '' }))
    setResults({})
    workshopTurn([...msgs, ...answers], true)
  }

  const send = async (given?: string) => {
    const t = (given ?? text).trim()
    if (!t || !ready || streaming) return
    setText(''); setError(null)
    if (workshopOn) { const h = [...msgs, { role: 'user', content: t } as Msg]; setMsgs(h); return workshopTurn(h) }
    const mine = ++generation.current
    const history = [...msgs, { role: 'user', content: t } as Msg]
    const drawn = drawSeed(preset)
    const context = replyContext(drawn)
    setMsgs([...history, { role: 'assistant', content: '', ...context }])
    setStreaming(true)
    const ctl = new AbortController(); abort.current = ctl
    const t0 = performance.now(); let tFirst = 0; let counted = 0
    let content = ''; let reasoning = ''; let outcome: Outcome = 'interrupted'; let errMsg: string | undefined
    let runtime: Runtime | undefined                         // the engine's own figures for this reply, when it sends them
    let cortheon: Cortheon | undefined
    let usage: { prompt_tokens?: number; completion_tokens?: number; prompt_tokens_details?: { cached_tokens?: number }; completion_tokens_details?: { reasoning_tokens?: number } } | undefined
    const body: Record<string, unknown> = { model, stream: true, stream_options: { include_usage: true },
      messages: toApi(history) }
    Object.assign(body, sampling(preset))
    if (drawn != null) body.seed = drawn
    if (webResearch && webOff) body.cortheon = false
    if (thinking !== 'default') body.chat_template_kwargs = { enable_thinking: thinking === 'on' }
    if (thinking !== 'off' && budget) body.thinking_budget = budget
    const current = () => generation.current === mine         // false once a new conversation replaced this one
    const update = (final = false) => {
      if (!current()) return
      setMsgs((m) => [...m.slice(0, -1), { role: 'assistant', content, reasoning, ...context, ...(final ? { outcome, error: errMsg } : {}) }])
      setLive({ t0, tFirst, chars: content.length + reasoning.length, ratio })
    }
    const ratio = charsPerToken(msgs)
    setLive({ t0, tFirst: 0, chars: 0, ratio })
    try {
      const r = await fetch('/engine/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', ...authHeaders() }, body: JSON.stringify(body), signal: ctl.signal })
      if (!r.ok || !r.body) {
        let detail = `${r.status} ${r.statusText}`
        try { const j = await r.json(); detail = j.error?.message ?? j.detail ?? detail } catch { /* keep status */ }
        throw new Error(detail)
      }
      const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const { events, rest } = sseEvents(buf); buf = rest
        for (const data of events) {
          if (data === '[DONE]') { if (outcome === 'interrupted') outcome = 'done'; continue }
          let chunk: Record<string, unknown>
          try { chunk = JSON.parse(data) } catch { continue }
          if (chunk.error) { outcome = 'error'; errMsg = (chunk.error as { message?: string }).message ?? 'the engine reported an error'; continue }
          if (chunk.usage) usage = chunk.usage as typeof usage
          if (chunk.tensorfold) runtime = { ...(chunk.tensorfold as Runtime), ...(chunk.speculative as Runtime | undefined) }
          if (chunk.cortheon) cortheon = chunk.cortheon as Cortheon
          const choice = (chunk.choices as { delta?: Record<string, string>; finish_reason?: string }[] | undefined)?.[0]
          const delta = choice?.delta ?? {}
          const thought = delta.reasoning_content ?? delta.reasoning      // TensorFold's name, and other engines'
          if (thought) { if (!tFirst) tFirst = performance.now(); counted += 1; reasoning += thought }
          if (delta.content) { if (!tFirst) tFirst = performance.now(); counted += 1; content += delta.content }
          if (choice?.finish_reason === 'length') outcome = 'length'
          else if (choice?.finish_reason === 'stop' && outcome === 'interrupted') outcome = 'done'
          update()
        }
      }
      const end = performance.now()
      const n = usage?.completion_tokens ?? null
      const fromEngine = runtime?.tokens_per_second != null && runtime?.time_to_first_token != null
      const stats: Stats = { tokens: n, events: counted,
        tps: fromEngine ? runtime!.tokens_per_second! : tFirst && n ? n / Math.max(0.001, (end - tFirst) / 1000) : null,
        ttft: fromEngine ? runtime!.time_to_first_token! : ((tFirst || end) - t0) / 1000,
        prompt: usage?.prompt_tokens, cached: usage?.prompt_tokens_details?.cached_tokens, reasoning: usage?.completion_tokens_details?.reasoning_tokens,
        measured: usage ? 'engine' : 'counted', timing: fromEngine ? 'engine' : 'page',
        budget: body.thinking_budget as number | undefined, sha: runtime?.token_sha, drafts: runtime?.drafts, acceptance: runtime?.acceptance_rate ?? undefined,
        perRound: runtime?.tokens_per_round ?? undefined, prefill: prefillRate(runtime, usage?.prompt_tokens, usage?.prompt_tokens_details?.cached_tokens) }
      if (current()) setMsgs((m) => [...m.slice(0, -1), { role: 'assistant', content, reasoning, ...context, outcome, error: errMsg, stats, ...(cortheon ? { cortheon } : {}) }])
      if (outcome === 'error' && current()) setError(errMsg ?? 'The engine reported an error.')
    } catch (e) {
      const aborted = (e as Error).name === 'AbortError'
      outcome = aborted ? 'stopped' : 'error'
      errMsg = aborted ? undefined : String(e)
      update(true)
      if (!aborted && current()) setError(String(e))
    } finally { if (current()) { setStreaming(false); setLive(null); abort.current = null } }
  }
  const opening = useRef(0)                                  // which opening may still replace the shown messages
  // the two sides of a comparison named by what wrote them ("Qwen3.8-27B, without the speed-up"), A and B until known
  const pairNames = ([a, b]: [string, string]): [string, string] => {
    const rows = [a, b].map((id) => convs.find((c) => c.id === id))
    if (!rows[0] || !rows[1]) return ['A', 'B']
    const n = namesOf(rows as ConversationRow[])
    return [n.get(a)!, n.get(b)!]
  }
  const closePanels = () => { setComparing(null); setSpeedupPair(false); setChecking(false); setReplaying(false) }
  const newConversation = (_silent = false) => {
    closePanels()                                            // a comparison belongs to the conversation it came from
    opening.current += 1                                     // a conversation still loading is not shown over it
    flushSave(); convGen.current += 1
    generation.current += 1                                  // whatever the old request still sends is ignored
    abort.current?.abort(); abort.current = null
    setStreaming(false); setError(null); setMsgs([]); setConvId(null)
    showSub('chat', null)
    setTimeout(() => input.current?.focus(), 0)
  }
  // `again`: read the shown conversation afresh, after a conflict
  const openConversation = async (id: string, again = false) => {
    if (id === convId && !again) return
    closePanels()
    const mine = ++opening.current
    abort.current?.abort(); abort.current = null       // the cut reply records its outcome before the switch
    generation.current += 1
    flushSave()
    setStreaming(false); setError(null)
    try {
      await saves.current!.settled()                     // what was saved of it has its answer before it is read
      const c = await api.conversation(id)
      if (mine !== opening.current) return                // another conversation was asked for meanwhile
      flushSave()                                         // edits made while it loaded belong to the one shown then
      generation.current += 1                             // a reply asked for meanwhile belongs to the one shown then:
      ;(abort.current as AbortController | null)?.abort(); abort.current = null   // it stops; nothing it still sends reaches this one
      setStreaming(false)
      convGen.current += 1                                // from here the page shows this one
      saves.current!.read(c.id, c.rev)
      setSaveError(false)
      opened.current = c.messages
      setMsgs(c.messages as Msg[]); setConvId(c.id)
      const st = c.settings as { thinking?: typeof thinking; budget?: number; preset?: Preset; workshop?: Workshop }
      if (st?.thinking) setThinking(st.thinking)
      if (st?.budget != null) setBudget(st.budget)
      setPreset(st?.preset ?? PRESETS[0])
      setWs({ ...DEFAULT_WORKSHOP, ...st?.workshop })
    } catch (e) { toast(String(e), 'bad') }
  }
  // a living copy is served: a message can be handed to Teach, where its facts are found and approved
  const teachable = !!engine?.model_repo?.startsWith('living:') || learnsHere
  const onBase = !engine?.model_repo?.startsWith('living:')
  const teachThis = (text: string) => { try { sessionStorage.setItem('tfs:teach-text', text) } catch { /* no storage */ } location.hash = '#teach' }
  const copy = async (t: string) => { try { await copyText(t); toast('Copied') } catch { toast('The browser refused the clipboard', 'bad') } }
  const last = [...msgs].reverse().find((m) => m.role === 'assistant' && m.stats)
  // without the engine's usage figures the page only counts stream events, which are not tokens: no count of
  // tokens and no speed is claimed then
  const amount = (st: Stats) => st.measured === 'engine' ? `${num(st.tokens)} tokens${st.tps != null ? `, ${Math.round(st.tps)} tok/s` : ''}` : `${num(st.events ?? st.tokens)} stream events`
  const where = engine?.managed === false ? `the engine at ${destination}` : 'the engine on this machine'
  const badge = (m: Msg) => m.outcome === 'stopped' ? 'stopped by you' : m.outcome === 'interrupted' ? 'connection lost before the end' : m.outcome === 'error' ? `error: ${m.error ?? 'unknown'}` : m.outcome === 'length' ? 'reached the reply limit' : null

  return (
    <section className="chatpage">
      <div className="head" style={{ marginBottom: 14 }}><div><h1>Chat</h1><p>{engine?.managed === false ? <>Messages go to the engine at <code>{destination}</code> and nowhere else.</> : 'Everything you write stays on this machine.'}</p></div>
        <div className="actions"><button className="btn" onClick={() => newConversation()}>New conversation</button></div></div>
      {engine?.status === 'unauthorized' && <div className="notice">The engine answers but refuses the API key this TensorFold Web holds. Enter the engine's key in <a href="#settings">Settings</a>, or restart the engine so it takes the new one.</div>}
      {!ready && engine?.status !== 'unauthorized' && <div className="notice info">No model is running yet. Start one from <a href="#models">Models</a>.</div>}
      {error && <div className="notice bad">{error}</div>}
      {saveError === 'failed' && <div className="notice">TensorFold Web did not save the conversation; it is tried again at the next change.</div>}
      {saveError === 'conflict' && <div className="notice">This conversation was changed elsewhere (another page, or a replay) since this page read it, so it is no longer saved from here. {convId && <button className="btn sm" onClick={() => openConversation(convId, true)}>Reopen it</button>} to see the latest and go on; what this page holds and the server lacks is lost.</div>}
      <div className="chatwrap">
      <ConversationList convs={convs} more={more} loadMore={loadMore} convId={convId} q={q} setQ={setQ} open={(id) => openConversation(id)} refresh={() => refreshList()} toast={toast}
        onDeleted={(id) => { if (id === convId) newConversation(true) }} />
      <div className="chat">
        {convId && (() => { const c = convs.find((x) => x.id === convId); const family = sameQuestions(convs, convId); const names = namesOf(c ? [c, ...family] : family); return <div className="convbar">
          <b className="ctitle" title={c?.title}>{c?.title ?? 'This conversation'}</b>
          <span className="actions">
            <button type="button" className={`btn sm ${checking ? '' : 'quiet'}`} aria-pressed={checking} onClick={() => { setChecking((v) => !v); setReplaying(false) }} title="Asks the last questions again with drafts off: the same tokens prove the speed-up changes nothing in the replies">Verify speed-up &amp; exactness</button>
            <button type="button" className={`btn sm ${replaying ? '' : 'quiet'}`} aria-pressed={replaying} onClick={() => { setReplaying((v) => !v); setChecking(false) }}>Replay</button>
            {family.length > 0 && <select className="sel sm" value="" aria-label="Compare its replies with a replay of the same questions" onChange={(e) => { if (e.target.value) { setSpeedupPair(false); setComparing([convId, e.target.value]) } }}>
              <option value="">Compare with a replay…</option>{family.map((o) => <option key={o.id} value={o.id}>{names.get(o.id)}</option>)}
            </select>}
            <select className="sel sm" value="" aria-label="Export this conversation" onChange={(e) => { const f = e.target.value; if (f) download(`/api/conversations/${convId}/export?format=${f}`, `${convId}.${f === 'md' ? 'md' : 'json'}`).catch((err) => toast(`Not exported: ${err}`, 'bad')) }}>
              <option value="">Export…</option><option value="md">as Markdown (to read)</option><option value="json">as JSON (all the data)</option>
            </select>
          </span>
        </div> })()}
        <div className="bar">
          <span className="model"><ModelLogo names={[engine?.model_name, model]} />{engine?.model_name ?? (model || 'No model')}</span>
          {checking || comparing ? <span className="small muted">{checking ? 'Checking the speed-up with the settings each answer was written with' : 'Comparing two conversations'}</span>
            : <button type="button" className={`btn sm ${options ? '' : 'quiet'}`} onClick={() => setOptions((v) => !v)} aria-expanded={options}>Options</button>}
          <span className="right">
          {streaming && live && <LivePills live={live} />}
          {last?.stats && !streaming && <span className="speed" aria-label="Speed of the last reply">
            <span className="pill ttft" title="Time to first token: from sending the message to the first word of the reply">first token <b>{secs(last.stats.ttft)}</b></span>
            {last.stats.measured === 'engine' && last.stats.tps != null && <span className="pill tps" title="Tokens a second while the reply was written">{rate(last.stats.tps)} <b>tok/s</b></span>}
            {last.stats.drafts && last.stats.acceptance != null && <span className="pill drafts" title="Speculative decoding: a small model drafts tokens ahead and the model checks them all in one pass, keeping the ones it would have written itself. The reply is the same, token for token, as without drafts: Verify speed-up & exactness proves it.">
              <b>{Math.round(last.stats.acceptance * 100)}%</b> drafts kept{last.stats.perRound ? `, ${last.stats.perRound.toFixed(1)} tok/round` : ''}</span>}
          </span>}
          <span className="count">{msgs.length ? `${msgs.length} messages` : ''}</span>
          </span>
        </div>
        {options && !checking && !comparing && <div className="bar options" aria-label="Options">
          <select className="sel" value={thinking} onChange={(e) => setThinking(e.target.value as typeof thinking)} aria-label="Thinking">
            <option value="default">Thinking: model default</option><option value="on">Thinking: on</option><option value="off">Thinking: off</option>
          </select>
          {thinking !== 'off' && <select className="sel" value={budget} onChange={(e) => setBudget(Number(e.target.value))} aria-label="Thinking budget"
            title="How many tokens the model may spend thinking before it must answer">
            {BUDGETS.map((b) => <option key={b} value={b}>{b ? `Think up to ${num(b)} tokens` : 'Think without limit'}</option>)}
          </select>}
          <select className="sel" value={preset.name} aria-label="Preset" title={preset.system || 'No system prompt'}
            onChange={(e) => setPreset([...PRESETS, ...mine].find((p) => p.name === e.target.value) ?? PRESETS[0])}>
            {[...PRESETS, ...mine].map((p) => <option key={p.name}>{p.name}</option>)}
          </select>
          <button type="button" className="btn sm quiet" onClick={() => setEditing((v) => !v)} aria-expanded={editing}>{editing ? 'Done' : 'Edit'}</button>
          {webResearch && <label className="small" title="Cortheon runs in this engine (Settings): searches written from the conversation go to DuckDuckGo, and pages are read from this machine"><input type="checkbox" checked={!webOff} onChange={(e) => setWebOff(!e.target.checked)} /> Web research</label>}
          <button type="button" className={`btn sm ${workshopOn ? '' : 'quiet'}`} onClick={() => setShowWs((v) => !v)} aria-expanded={showWs}>Tools &amp; JSON{workshopOn ? ' (on)' : ''}</button>
        </div>}
        {editing && !checking && !comparing && <PresetEditor preset={preset} setPreset={setPreset} mine={mine} reload={reloadPresets} toast={toast} />}
        {showWs && !checking && !comparing && <WorkshopPanel ws={ws} set={setWs} />}
        {checking && convId && <SpeedupCheck id={convId} model={model} toast={toast} written={msgs.filter((m) => m.role === 'assistant').map((m) => m.request ?? null)} onClose={() => setChecking(false)} onDone={(a, b) => { setChecking(false); refreshList(); setSpeedupPair(true); setComparing([a, b]) }} />}
        {replaying && convId && <ReplayForm id={convId} model={model} toast={toast} onDone={(rid) => { setReplaying(false); refreshList(); setComparing([convId, rid]) }} />}
        {comparing && <div className="msgs"><ReplyCompare a={comparing[0]} b={comparing[1]} speedup={speedupPair} names={pairNames(comparing)} onClose={() => { setComparing(null); setSpeedupPair(false) }} /></div>}
        <div className="msgs" ref={box} hidden={!!comparing} onScroll={(e) => { const el = e.currentTarget; stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40 }}>
          {!msgs.length ? (
            <div className="empty">
              <h2>{ready ? `Ask ${engine?.model_name ?? model} anything` : 'Nothing to talk to yet'}</h2>
              <p>{ready ? `Replies are written by ${where}; nothing leaves it. Try one of these, or type your own.` : settings?.model && engine?.managed !== false ? `${settings.model.split('/').pop()} was the last model served here.` : 'Start a model from Models; the conversation happens here.'}</p>
              {!ready && settings?.model && engine?.managed !== false && engine?.status !== 'starting' && <button className="btn primary" disabled={starting} onClick={async () => {
                setStarting(true)
                try { await api.start({ model: settings.model! }); await refreshEngine(); toast('Starting the model') } catch (e) { toast(String(e), 'bad') } finally { setStarting(false) }
              }}>{starting ? 'Starting…' : `Start ${settings.model.split('/').pop()}`}</button>}
              {engine?.status === 'starting' && <p className="small muted">Loading {engine.model ?? 'the model'}…</p>}
              {ready && <div className="starters">{STARTERS.map((s) => <button key={s} type="button" onClick={() => send(s)}>{s}</button>)}</div>}
            </div>
          ) : (
            <div className="thread">
              {msgs.map((m, i) => m.role === 'user' ? <div key={i} className="msg user">{m.content}{teachable && <button type="button" className="btn sm quiet teach-this" onClick={() => teachThis(m.content)} title={onBase ? 'Open Teach with this message: it learns into a copy of this model, made there first (the original stays as it is)' : 'Open Teach with this message, to approve the facts it holds'}>Teach this</button>}</div> : m.role === 'tool' ? (
                <div key={i} className="msg tool"><span className="small muted">{m.name} returned</span><pre>{m.content}</pre></div>
              ) : (
                <div key={i} className="msg ai">
                  <div className="who" title={[m.engine, m.thinking ? `thinking ${m.thinking}` : '', m.preset && m.preset !== 'Model defaults' ? `preset ${m.preset}` : '', m.at ? new Date(m.at * 1000).toLocaleString() : ''].filter(Boolean).join(' · ')}><ModelLogo names={[m.model ?? model]} size="xs" /><b>{m.model ?? model}</b>{m.stats ? <span className="num">{amount(m.stats)}</span> : null}{badge(m) ? <span>{badge(m)}</span> : null}</div>
                  {m.reasoning && <details className="think" open={(streaming && i === msgs.length - 1 && !m.content) || undefined}>
                    <summary>{streaming && i === msgs.length - 1 && !m.content ? 'Thinking…' : `Thought for ${num(m.stats?.reasoning ?? m.reasoning.length)} ${m.stats?.reasoning ? 'tokens' : 'characters'}`}</summary>{m.reasoning}</details>}
                  <Reply text={m.content} live={streaming && i === msgs.length - 1} onCopy={copy} />
                  {m.tool_calls?.map((c) => <div key={c.id} className="toolcall"><code>{c.name}({c.arguments})</code>
                    <span className={c.check ? 'bad small' : 'good small'}>{c.check ? ` ✗ ${c.check}` : ' ✓ matches its parameters'}</span></div>)}
                  {m.json && <div className={`small ${m.json.ok ? 'good' : 'bad'}`}>{m.json.ok ? '✓' : '✗'} {m.json.detail}</div>}
                  {m.cortheon && <WebSources c={m.cortheon} />}
                  {streaming && i === msgs.length - 1 && <span className="cursor" />}
                </div>
              ))}
            </div>
          )}
        </div>
        {pendingCalls.length > 0 && !streaming && <div className="presetbox">
          <b className="small">The model asked for {pendingCalls.length === 1 ? 'a tool' : `${pendingCalls.length} tools`}: give the results</b>
          {pendingCalls.map((c) => <label key={c.id} className="small">{c.name}({c.arguments})
            <textarea className="input mono" rows={2} value={results[c.id] ?? ''} onChange={(e) => setResults({ ...results, [c.id]: e.target.value })} placeholder='{"temperature": 18, "sky": "clear"}' /></label>)}
          <div><button className="btn sm primary" onClick={giveResults}>Send the results</button></div>
        </div>}
        {last?.stats && (
          <div className="replybar num">
            {last.stats.measured === 'engine'
              ? <><span>Last reply <b>{num(last.stats.tokens)} tokens</b>{last.stats.reasoning ? ` (${num(last.stats.reasoning)} thinking${last.stats.budget && last.stats.reasoning >= last.stats.budget ? ', cut at the budget' : ''})` : ''}</span>{last.stats.tps != null && <span><b>{Math.round(last.stats.tps)} tok/s</b></span>}</>
              : <span>Last reply <b>{num(last.stats.events ?? last.stats.tokens)} stream events</b></span>}
            <span>first token after <b>{last.stats.ttft < 1 ? `${Math.round(last.stats.ttft * 1000)} ms` : `${last.stats.ttft.toFixed(1)} s`}</b></span>
            {last.stats.prompt != null && <span>prompt <b>{num(last.stats.prompt)} tokens</b>{last.stats.cached != null ? `, ${num(last.stats.cached)} from cache` : ''}{last.stats.prefill ? <>, read at <b>{num(Math.round(last.stats.prefill))} tok/s</b></> : ''}</span>}
            {last.stats.drafts != null && <span>drafts <b>{last.stats.drafts ? 'on' : 'off'}</b>{last.stats.acceptance != null ? `, ${Math.round(last.stats.acceptance * 100)}% kept` : ''}</span>}
            {last.stats.sha && <span title="A digest of the reply's tokens: the same request with drafts off gives the same one when decoding is exact">tokens <b className="mono">{last.stats.sha}</b></span>}
            <span className="faint">{last.stats.measured !== 'engine' ? 'the engine sent no token count; events are not tokens'
              : last.stats.timing === 'engine' ? 'measured by the engine' : 'tokens counted by the engine, times measured by this page'}</span>
          </div>
        )}
        <div className="composer">
          <div className="box">
            <textarea ref={input} rows={1} value={text} placeholder={ready ? `Message ${model}` : 'Start a model first'} disabled={!ready} aria-label="Message"
                      onChange={(e) => setText(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }} />
          </div>
          {streaming ? <button className="btn" onClick={() => abort.current?.abort()}>Stop</button> : <button className="btn primary" disabled={!ready || !text.trim()} onClick={() => send()}>Send</button>}
        </div>
      </div>
      </div>
    </section>
  )
}
