// The Chat's parallel mode: the same moment, several replies at once, to see what batching gives. Each reply streams
// in its own card with its own speed; the total speed is every reply's tokens over the time from sending to the last
// token, and each count tried here is kept so 1, 2, 4… requests at once can be read side by side.
import { useEffect, useRef, useState } from 'react'
import { authHeaders, num, type EngineInfo } from '../api'
import { sseEvents } from '../sse'

type Slot = { prompt: string; text: string; tokens: number; counted: number; tFirst: number; tEnd: number
  tps: number | null; ttft: number | null; outcome: 'running' | 'done' | 'length' | 'error' | 'stopped'; error?: string }
type Result = { n: number; total: number; perStream: number; ttft: number; wall: number; tokens: number; failed: number }
type Sampling = { thinking: 'default' | 'on' | 'off'; system: string; temperature: number | null }

const PROMPTS = [
  'Explain how a refrigerator keeps food cold, in one paragraph.',
  'Write a short poem about a lighthouse keeper.',
  'List five good habits for writing maintainable code, one line each.',
  'Describe the water cycle to a ten-year-old.',
  'Give three arguments for and against working from home.',
  'Summarise the plot of a heist film you invent, in five sentences.',
  'Explain what a hash table is and when to use one.',
  'Plan a one-day walking tour of Lyon.',
].join('\n')
const COUNTS = [1, 2, 4, 8, 16]
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length ? (s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2) : 0 }

export function ParallelPanel({ model, engine, sampling }: { model: string; engine: EngineInfo | null; sampling: Sampling }) {
  const [n, setN] = useState(4)
  const [prompts, setPrompts] = useState(PROMPTS)
  const [maxTokens, setMaxTokens] = useState(256)
  const [slots, setSlots] = useState<Slot[]>([])
  const [running, setRunning] = useState(false)
  const [results, setResults] = useState<Result[]>([])
  const [uncounted, setUncounted] = useState(false)          // the engine gave no token counts: no speed to keep     // one per count tried, the latest kept
  const live = useRef<Slot[]>([])
  const ctl = useRef<AbortController | null>(null)
  const t0 = useRef(0)
  const [, tick] = useState(0)
  useEffect(() => { if (!running) return; const id = setInterval(() => { setSlots(live.current.map((x) => ({ ...x }))); tick((v) => v + 1) }, 150); return () => clearInterval(id) }, [running])
  useEffect(() => () => ctl.current?.abort(), [])

  const lines = prompts.split('\n').map((l) => l.trim()).filter(Boolean)
  const cap = engine?.startup?.streams

  const one = async (i: number, signal: AbortSignal) => {
    const s = live.current[i]
    const body: Record<string, unknown> = { model, stream: true, stream_options: { include_usage: true }, max_tokens: maxTokens,
      messages: [...(sampling.system ? [{ role: 'system', content: sampling.system }] : []), { role: 'user', content: s.prompt }] }
    if (sampling.temperature != null) body.temperature = sampling.temperature
    if (sampling.thinking !== 'default') body.chat_template_kwargs = { enable_thinking: sampling.thinking === 'on' }
    try {
      const r = await fetch('/engine/v1/chat/completions', { method: 'POST', headers: { 'content-type': 'application/json', ...authHeaders() }, body: JSON.stringify(body), signal })
      if (!r.ok || !r.body) { const j = await r.json().catch(() => ({})); throw new Error(j.error?.message ?? j.detail ?? `${r.status} ${r.statusText}`) }
      const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = ''
      let runtime: { tokens_per_second?: number; time_to_first_token?: number } | undefined
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const { events, rest } = sseEvents(buf); buf = rest
        for (const data of events) {
          if (data === '[DONE]') continue
          let c: Record<string, unknown>
          try { c = JSON.parse(data) } catch { continue }
          if (c.error) { s.outcome = 'error'; s.error = (c.error as { message?: string }).message; continue }
          if (c.usage) s.tokens = (c.usage as { completion_tokens?: number }).completion_tokens ?? s.tokens
          if (c.tensorfold) runtime = { ...(c.tensorfold as object), ...(c.speculative as object | undefined) } as typeof runtime
          const ch = (c.choices as { delta?: Record<string, string>; finish_reason?: string }[] | undefined)?.[0]
          const piece = (ch?.delta?.content ?? '') + (ch?.delta?.reasoning_content ?? ch?.delta?.reasoning ?? '')
          if (piece) { if (!s.tFirst) s.tFirst = performance.now(); s.counted += 1; if (ch?.delta?.content) s.text += ch.delta.content }
          if (ch?.finish_reason === 'length') s.outcome = 'length'
        }
      }
      s.tEnd = performance.now()
      // without the engine's count there are stream events, not tokens (an event can carry several): no token figure
      s.tps = runtime?.tokens_per_second ?? (s.tFirst && s.tokens ? s.tokens / Math.max(0.001, (s.tEnd - s.tFirst) / 1000) : null)
      // from sending, as the page sees it: a request waiting for a free slot waits here, not in the engine's figure
      s.ttft = s.tFirst ? (s.tFirst - t0.current) / 1000 : null
      if (s.outcome === 'running') s.outcome = 'done'
    } catch (e) {
      s.tEnd = performance.now()
      if ((e as Error).name === 'AbortError') s.outcome = 'stopped'
      else { s.outcome = 'error'; s.error = String(e) }
    }
  }

  const go = async () => {
    if (!lines.length || running) return
    const c = new AbortController(); ctl.current = c
    live.current = Array.from({ length: n }, (_, i) => ({ prompt: lines[i % lines.length], text: '', tokens: 0, counted: 0, tFirst: 0, tEnd: 0, tps: null, ttft: null, outcome: 'running' }))
    setSlots(live.current.map((x) => ({ ...x }))); setRunning(true); setUncounted(false)
    t0.current = performance.now()
    await Promise.all(live.current.map((_, i) => one(i, c.signal)))
    const done = live.current.filter((s) => s.outcome === 'done' || s.outcome === 'length')
    const end = Math.max(...live.current.map((s) => s.tEnd))
    const tokens = done.reduce((a, s) => a + (s.tokens || 0), 0)
    // a reply the engine did not count leaves the total unknown: stream events are not tokens
    if (done.length && !c.signal.aborted && done.every((s) => s.tokens > 0)) {
      const wall = (end - t0.current) / 1000
      const res: Result = { n, tokens, wall, total: tokens / Math.max(0.001, wall), perStream: median(done.map((s) => s.tps ?? 0)),
        ttft: median(done.map((s) => s.ttft ?? 0)), failed: n - done.length }
      setResults((rs) => [...rs.filter((r) => r.n !== n), res].sort((a, b) => a.n - b.n))
    } else if (done.length && !c.signal.aborted) setUncounted(true)
    setSlots(live.current.map((x) => ({ ...x }))); setRunning(false); ctl.current = null
  }

  const base = results.find((r) => r.n === 1)
  const now = (performance.now() - t0.current) / 1000
  // while replies stream, the engine has not counted yet: what grows is stream events, said as such
  const liveEvents = slots.reduce((a, s) => a + s.counted, 0)
  return (
    <div className="presetbox parallel">
      <div className="row">
        <label className="small">Requests at once <select className="sel sm" value={n} onChange={(e) => setN(Number(e.target.value))} aria-label="Requests at once" disabled={running}>
          {COUNTS.map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
        <label className="small">Each up to <input className="input num" type="number" min={16} max={4096} step={16} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value) || 256)} disabled={running} aria-label="Reply length" /> tokens</label>
        {running ? <button className="btn sm" onClick={() => ctl.current?.abort()}>Stop</button>
          : <button className="btn sm primary" onClick={go} disabled={!lines.length || !model}>Send {n} at once</button>}
        {!running && results.length > 0 && <button className="btn sm quiet" onClick={() => setResults([])}>Clear the table</button>}
      </div>
      <label className="small">Prompts, one per line (taken in turn: {lines.length} for {n} {n === 1 ? 'request' : 'requests'})
        <textarea className="input" rows={3} value={prompts} onChange={(e) => setPrompts(e.target.value)} disabled={running} /></label>
      <p className="small muted" style={{ margin: 0 }}>
        Thinking {sampling.thinking === 'default' ? "as the model's default" : sampling.thinking}, {sampling.system ? 'the system prompt given' : 'no system prompt'}, {sampling.temperature != null ? `temperature ${sampling.temperature}` : "the model's own temperature"}; replies are not kept.
        {cap ? ` This engine serves up to ${cap} requests at once; more wait their turn.` : ''} Try 1 first: the table then shows what batching adds.
      </p>
      {slots.length > 0 && <>
        <div className="small"><b>{running ? `${num(liveEvents)} stream events in ${now.toFixed(1)} s (tokens are counted at the end)`
          : `${slots.filter((s) => s.outcome === 'done' || s.outcome === 'length').length} of ${slots.length} replies done`}</b></div>
        <div className="pgrid">
          {slots.map((s, i) => <div key={i} className={`pcard ${s.outcome}`}>
            <div className="small muted">#{i + 1} · {s.outcome === 'running' ? (s.tFirst ? `${num(s.counted)} events so far` : 'waiting for its first token')
              : s.outcome === 'error' ? 'error' : `${s.tokens ? `${num(s.tokens)} tokens` : 'tokens not counted by the engine'}, ${s.tps ? Math.round(s.tps) : '—'} tok/s, first after ${s.ttft != null ? s.ttft.toFixed(1) : '—'} s${s.outcome === 'length' ? ', cut' : s.outcome === 'stopped' ? ', stopped' : ''}`}</div>
            <div className="small pprompt">{s.prompt}</div>
            <div className="ptext">{s.error ? <span className="bad">{s.error}</span> : s.text || (s.counted ? <span className="muted">thinking…</span> : '')}</div>
          </div>)}
        </div>
      </>}
      {uncounted && <p className="small muted">The engine gave no token counts for these replies, so no speed is kept: a stream event can carry several tokens.</p>}
      {results.length > 0 && <div className="pscroll"><table className="tbl small">
        <thead><tr><th>At once</th><th>Total speed</th><th>Each reply</th><th>First token</th><th>Tokens</th><th>Time</th>{base && <th>Against 1</th>}</tr></thead>
        <tbody>{results.map((r) => <tr key={r.n}>
          <td>{r.n}{r.failed ? <span className="bad"> ({r.failed} failed)</span> : null}</td><td><b>{num(Math.round(r.total))} tok/s</b></td><td>{Math.round(r.perStream)} tok/s</td>
          <td>{r.ttft.toFixed(2)} s</td><td>{num(r.tokens)}</td><td>{r.wall.toFixed(1)} s</td>
          {base && <td>{r === base ? '—' : `×${(r.total / base.total).toFixed(2)} in total, ×${(r.perStream / base.perStream).toFixed(2)} each`}</td>}
        </tr>)}</tbody>
      </table></div>}
    </div>
  )
}
