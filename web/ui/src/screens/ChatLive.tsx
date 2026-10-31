// Chat's live figures, beside its replies: the reply being written (first token, speed) and what Cortheon's web
// research did for a reply.
import { useEffect, useState } from 'react'
import { liveRate } from '../liverate'

// what Cortheon did for a reply (TensorFold 1.0.9 and later with --cortheon; from 1.0.10
// pages_read counts only the pages read, not refused or failed fetches): its searches and the pages it read and cites
export type Cortheon = { task?: string; searches?: number; pages_read?: number; sources?: string[] }
export const secs = (s: number) => s < 1 ? `${Math.round(s * 1000)} ms` : `${s.toFixed(1)} s`
export const rate = (r: number) => r >= 10 ? `${Math.round(r)}` : r.toFixed(1)

export type Live = { t0: number; tFirst: number; chars: number; ratio: number }
/** The reply being written, as this page sees it: the first token's time once it came, and a live speed from the text
 * received (`liverate.ts`); the engine's own figures replace both when the reply ends. */
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

/** What Cortheon's web research did for a reply: how many searches and pages, and the sources it cites. */
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
