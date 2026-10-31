import { useEffect, useState } from 'react'
import { api, num, type EngineInfo, type UsageSummary } from '../api'
import { count } from '../units'

const RANGES = [7, 30, 90]
const COLORS = ['var(--accent)', 'var(--good)', 'var(--warn)', 'var(--ink3)', 'var(--bad)']

export function ago(t: number, now = Date.now() / 1000): string {
  const s = Math.max(0, now - t)
  return s < 90 ? 'just now' : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} days ago`
}

/** Bars of written tokens, one per day or hour, stacked by model; each bar's title says the figures. */
function Bars({ cols, color, label, ends }: { cols: { key: string; label: string; models: Record<string, { completion: number }> }[]; color: (m: string) => string; label: string; ends: [string, string] }) {
  const tot = (c: (typeof cols)[number]) => Object.values(c.models).reduce((a, m) => a + m.completion, 0)
  const max = Math.max(1, ...cols.map(tot))
  return (
    <div className="ubars" role="img" aria-label={label}>
      {cols.map((c) => (
        <div key={c.key} className="ucol" title={`${c.label}: ${Object.entries(c.models).map(([m, v]) => `${m} ${num(v.completion)}`).join(', ') || 'nothing'}`}>
          {Object.entries(c.models).map(([m, v]) => <i key={m} style={{ height: `${(v.completion / max) * 100}%`, background: color(m) }} />)}
        </div>
      ))}
      <div className="uends small faint"><span>{ends[0]}</span><span>{ends[1]}</span></div>
    </div>
  )
}

function Peaks({ days, field, label }: { days: { key: string; day?: UsageSummary['days'][number] }[]; field: 'peak_decode' | 'peak_prefill'; label: string }) {
  const v = (d: (typeof days)[number]) => d.day?.[field] ?? 0
  const max = Math.max(1, ...days.map(v))
  const best = days.reduce((a, d) => (v(d) > v(a) ? d : a), days[0])
  return (
    <div className="upeak">
      <span className="small muted">{label}</span>
      <div className="ubars thin" role="img" aria-label={label}>{days.map((d) => <div key={d.key} className="ucol" title={`${d.key}: ${num(Math.round(v(d)))} tok/s`}><i style={{ height: `${(v(d) / max) * 100}%` }} /></div>)}</div>
      <span className="small num">{best && v(best) ? `${num(Math.round(v(best)))} tok/s, ${best.key}` : '—'}</span>
    </div>
  )
}

export default function MonitorUsage({ engine }: { engine: EngineInfo | null }) {
  const [days, setDays] = useState(30)
  const [u, setU] = useState<UsageSummary | null>(null)
  useEffect(() => {
    let live = true
    const load = () => api.usage(days).then((x) => { if (live) setU(x) }).catch(() => {})
    load()
    const t = setInterval(load, 60000)
    return () => { live = false; clearInterval(t) }
  }, [days])
  if (!u) return null
  const order = u.models.map((m) => m.model)
  const color = (m: string) => COLORS[Math.max(0, order.indexOf(m)) % COLORS.length]
  // every day of the range, so an idle day shows as an empty slot rather than vanishing
  const byDay = new Map(u.days.map((d) => [d.day, d]))
  const dayCols = Array.from({ length: days }, (_, i) => {
    const d = new Date(Date.now() - (days - 1 - i) * 86400000)
    const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
    return { key, label: key, models: byDay.get(key)?.models ?? {}, day: byDay.get(key) }
  })
  const byHour = new Map(u.hours.map((h) => [h.hour, h]))
  const h0 = Math.floor(Date.now() / 3600000) * 3600
  const hourCols = Array.from({ length: 48 }, (_, i) => {
    const t = h0 - (47 - i) * 3600
    return { key: String(t), label: new Date(t * 1000).toLocaleString([], { weekday: 'short', hour: '2-digit', minute: '2-digit' }), models: byHour.get(t)?.models ?? {} }
  })
  const busy = (engine?.connections ?? 0) > 0
  const last = u.last_served
  const written = u.models.reduce((a, m) => a + m.completion, 0)
  return (
    <div className="card usage">
      <div className="membar-head"><b>Usage</b>
        <span className="small muted">{busy ? 'serving now' : last ? `idle, last served ${ago(last.t)} (${last.model})` : 'nothing served yet'}</span></div>
      <div className="winsel" role="group" aria-label="Days shown">
        {RANGES.map((r) => <button key={r} type="button" aria-pressed={days === r} className={days === r ? 'active' : ''} onClick={() => setDays(r)}>{r} days</button>)}
        <span className="muted small">{u.since ? `kept by TensorFold Web since ${new Date(u.since * 1000).toLocaleDateString()}, across engine restarts` : 'counting from now'}</span>
      </div>
      {written === 0 ? <p className="small muted">No tokens written in these {days} days yet.</p> : <>
        <div className="usub small muted">Tokens written per day</div>
        <Bars cols={dayCols} color={color} label={`Tokens written per day, last ${days} days`} ends={[`${days} days ago`, 'today']} />
        <div className="usub small muted">Last 48 hours, per hour</div>
        <Bars cols={hourCols} color={color} label="Tokens written per hour, last 48 hours" ends={['48 h ago', 'now']} />
        <Peaks days={dayCols} field="peak_decode" label="Peak writing speed per day" />
        <Peaks days={dayCols} field="peak_prefill" label="Peak prompt reading per day" />
        <div style={{ overflowX: 'auto', marginTop: 12 }}><table className="tbl">
          <thead><tr><th>Model</th><th>Written</th><th>Read</th><th>Restored</th><th>Requests</th></tr></thead>
          <tbody>{u.models.map((m) => <tr key={m.model}><td><i className="ukey" style={{ background: color(m.model) }} />{m.model}</td><td>{count(m.completion)}</td><td>{count(m.prompt)}</td><td>{m.prompt ? `${Math.round((m.cached / m.prompt) * 100)}%` : '—'}</td><td>{num(m.requests)}</td></tr>)}</tbody>
        </table></div>
      </>}
    </div>
  )
}
