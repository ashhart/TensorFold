// Show live reply timings and the searches and citations made by Cortheon.
import { useEffect, useState } from 'react'
import { liveRate } from '../liverate'

// Cortheon page counts include only successful page reads, excluding refusals and fetch failures.
export type Cortheon = { task?: string; searches?: number; pages_read?: number; sources?: string[] }
export const secs = (s: number) => s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`
export const rate = (r: number) => r >= 10 ? `${Math.round(r)}` : r.toFixed(1)

export type Live = { t0: number; tFirst: number; chars: number; ratio: number }


export function LivePills({ live }: { live: Live }) {
  const [now, setNow] = useState(() => performance.now())
  useEffect(() => { const t = setInterval(() => setNow(performance.now()), 250); return () => clearInterval(t) }, [])
  const writing = live.tFirst ? Math.max(0.001, (Math.max(now, live.tFirst) - live.tFirst) / 1000) : 0
  const speed = live.tFirst ? liveRate(live.chars, writing, live.ratio) : null
  return (
    <span className="speed now" aria-label="Speed of the reply being written">
      <span className="pill ttft" title="Time to first token, measured by this page">first token <b>{live.tFirst ? secs((live.tFirst - live.t0) / 1000) : `${secs((now - live.t0) / 1000)}…`}</b></span>
      {speed != null && <span className="pill tps" title={`Live, estimated from the text received at about ${live.ratio.toFixed(1)} characters a token (the last reply's, as the engine counted it); the engine's exact figure replaces it when the reply ends`}>≈ {rate(speed)} <b>tok/s</b></span>}
    </span>
  )
}


export function WebSources({ c }: { c: Cortheon }) {
  const sources = (c.sources ?? []).filter((u) => /^https?:\/\//.test(u))
  if (!c.searches && !c.pages_read && !sources.length) return null
  return (
    <details className="websources small">
      <summary>Web research: {c.searches ?? 0} {c.searches === 1 ? 'search' : 'searches'}, {c.pages_read ?? 0} {c.pages_read === 1 ? 'page' : 'pages'} read{sources.length ? `, ${sources.length} ${sources.length === 1 ? 'source' : 'sources'}` : ''}</summary>
      {sources.length > 0 && <ul>{sources.map((u) => <li key={u}><a href={u} target="_blank" rel="noopener noreferrer">{u}</a></li>)}</ul>}
    </details>
  )
}
