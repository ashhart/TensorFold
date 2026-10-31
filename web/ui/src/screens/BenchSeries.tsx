// Compare mode: two to four configurations of the served engine, measured one after the other, as a matrix.
import { useCallback, useEffect, useState } from 'react'
import { api, gb, num, type BenchRun, type EngineInfo, type Model, type SeriesConfig, type SeriesDoc } from '../api'
import type { Ctx } from '../ctx'
import { secs } from '../units'

const CONTEXTS = [null, 8192, 16384, 32768, 65536]
const OTHER = '__other__'
const tps = (x: number | null | undefined) => (x == null ? '—' : x >= 20 ? String(Math.round(x)) : x.toFixed(1))
const joules = (x: number | null | undefined) => (x == null ? '—' : x < 1 ? `${Math.round(x * 1000)} mJ` : `${x.toFixed(2)} J`)
const pct = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(Math.round(x * 100))}%`

type Row = SeriesConfig & { other: string }
const blank = (model: string): Row => ({ model, context: null, budget_fraction: null, drafts: true, label: '', other: '' })

function Matrix({ doc, onOpen, onCompare }: { doc: SeriesDoc; onOpen: (id: string) => void; onCompare: (a: string, b: string) => void }) {
  const first = doc.rows.find((r) => r.run && r.status === 'done')
  return (
    <div className="tablewrap"><table className="tbl">
      <thead><tr><th>Configuration</th><th>tok/s</th><th>without drafts</th><th>first token</th><th>{doc.mode === 'quality' ? 'right' : 'identical'}</th><th>J/token</th><th>engine memory</th><th>context admitted</th><th>against the first</th><th /></tr></thead>
      <tbody>{doc.rows.map((r) => {
        const vs = r.vs_first
        return <tr key={r.index}>
          <td>{r.label}{r.error ? <div className="small bad">{r.error}</div> : null}</td>
          <td>{tps(r.tps)}</td><td>{tps(r.tps_no_drafts)}</td><td>{secs(r.ttft)}</td>
          <td>{doc.mode === 'quality' ? (r.items ? <>{r.right}/{r.items}{vs && 'quality' in vs && vs.quality ? <small className={vs.quality.within_noise ? 'muted' : 'bad'}> {vs.quality.only_a}↔{vs.quality.only_b}{vs.quality.within_noise ? '' : ', real'}</small> : null}</> : '—') : r.checked ? `${r.exact}/${r.checked}` : '—'}</td><td>{joules(r.j_per_token)}</td><td>{gb(r.engine_peak_bytes, 1)}</td>
          <td>{r.served?.context_tokens ? num(r.served.context_tokens) : '—'}</td>
          <td>{!vs ? (r === first ? <span className="muted">reference</span> : '—') : 'refused' in vs ? <span className="small muted">{vs.refused}</span>
            : vs.speed.median_change == null ? '—' : <span className={vs.speed.within_noise ? 'muted' : vs.speed.median_change > 0 ? 'good' : 'bad'} title={vs.findings.join(' ')}>
              {pct(vs.speed.median_change)}{vs.speed.within_noise ? ', noise' : ''}</span>}</td>
          <td className="actions">{r.run && <button className="btn sm quiet" onClick={() => onOpen(r.run!)}>Open</button>}
            {r.run && first?.run && r.run !== first.run && <button className="btn sm quiet" onClick={() => onCompare(first.run!, r.run!)}>Compare</button>}</td>
        </tr>
      })}</tbody>
    </table></div>
  )
}

export default function Series({ engine, toast, stale, exactness, onOpen, onCompare }: {
  engine: EngineInfo | null; toast: Ctx['toast']; stale: boolean; exactness: boolean
  onOpen: (id: string) => void; onCompare: (a: string, b: string) => void
}) {
  const [models, setModels] = useState<Model[]>([])
  const [rows, setRows] = useState<Row[]>([])
  const [measure, setMeasure] = useState<'profile' | 'quick' | 'quality'>('profile')
  const [list, setList] = useState<SeriesDoc[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [shown, setShown] = useState<SeriesDoc | null>(null)
  const [run, setRun] = useState<BenchRun | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [asking, setAsking] = useState(false)

  const usable = models.filter((m) => m.installed && m.runs_here)
  useEffect(() => {
    api.catalog().then((c) => {
      setModels(c.models)
      const here = c.models.filter((m) => m.installed && m.runs_here).map((m) => m.id)
      const start = engine?.model_repo && here.includes(engine.model_repo) ? engine.model_repo : here[0] ?? ''
      setRows((r) => (r.length ? r : [blank(start), blank(here.find((x) => x !== start) ?? start)]))
    }).catch((e) => setErr(String(e)))
  }, [engine?.model_repo])
  const load = useCallback(() => api.seriesList().then((l) => {
    setList(l.series); setCurrent(l.current)
    setShown((s) => (s ? l.series.find((x) => x.id === s.id) ?? s : l.series[0] ?? null))
  }).catch(() => {}), [])
  useEffect(() => { load() }, [load])
  // while a comparison goes: the comparison every 2 s, and the run inside it
  useEffect(() => {
    if (!current) return
    let live = true
    const tick = async () => {
      try {
        const d = await api.series(current)
        if (!live) return
        setShown(d)
        if (d.progress?.run) setRun(await api.benchRun(d.progress.run)); else setRun(null)
        if (d.status !== 'running') { setCurrent(null); setRun(null); load(); d.restore_error ? toast(`Comparison ${d.status === 'interrupted' ? 'interrupted' : 'finished'}, but ${d.restore_error}`, 'bad') : toast(d.status === 'cancelled' ? 'Comparison cancelled; the engine is back as it was' : 'Comparison finished; the engine is back as it was') }
      } catch { /* the next tick tries again */ }
    }
    tick()
    const t = setInterval(tick, 2000)
    return () => { live = false; clearInterval(t) }
  }, [current, load, toast])

  const configs = rows.map((r) => ({ model: r.model === OTHER ? r.other.trim() : r.model, context: r.context, budget_fraction: r.budget_fraction, drafts: r.drafts, label: r.label }))
  const valid = configs.length >= 2 && configs.every((c) => c.model.includes('/'))
  const start = async () => {
    setErr(null); setAsking(false)
    try { const d = await api.seriesStart({ configs, mode: measure, exactness }); setCurrent(d.id); setShown(null) } catch (e) { setErr(String(e)) }
  }
  const set = (i: number, patch: Partial<Row>) => setRows((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)))
  const going = shown && shown.status === 'running' ? shown : null
  const remote = engine?.managed === false

  return <>
    <p className="small muted">Two to four configurations of the engine this TensorFold Web serves, measured one after the other with the same prompts; the engine restarts between them when the model, context or budget changes, and comes back to {engine?.model ?? 'how it was'} at the end. Anyone using it meanwhile is cut off.</p>
    {remote && <div className="notice info">The engine is remote: a comparison restarts it, which only TensorFold Web on its machine can do.</div>}
    {err && <div className="notice bad">{err}</div>}
    {!going && <div className="series-form">
      {rows.map((r, i) => (
        <div key={i} className="series-row">
          <b>{String.fromCharCode(65 + i)}</b>
          <select className="sel" value={r.model} onChange={(e) => set(i, { model: e.target.value })} aria-label={`Model of configuration ${i + 1}`}>
            {usable.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            <option value={OTHER}>Another downloaded checkpoint…</option>
          </select>
          {r.model === OTHER && <input className="input" value={r.other} onChange={(e) => set(i, { other: e.target.value })} placeholder="owner/name" aria-label="Checkpoint" spellCheck={false} />}
          <select className="sel" value={r.context ?? ''} onChange={(e) => set(i, { context: e.target.value ? Number(e.target.value) : null })} aria-label="Context">
            {CONTEXTS.map((c) => <option key={String(c)} value={c ?? ''}>{c ? `${num(c)} tokens` : 'Context: auto'}</option>)}
          </select>
          <select className="sel" value={r.budget_fraction ?? ''} onChange={(e) => set(i, { budget_fraction: e.target.value ? Number(e.target.value) : null })} aria-label="Memory the engine may use" title="Auto: the model's verdict decides, as in Models">
            <option value="">Memory: auto</option>{[0.7, 0.75, 0.8, 0.85, 0.9].map((f) => <option key={f} value={f}>{Math.round(f * 100)}% of memory</option>)}
          </select>
          <label className="small"><input type="checkbox" checked={r.drafts} onChange={(e) => set(i, { drafts: e.target.checked })} /> drafts</label>
          {rows.length > 2 && <button className="btn sm quiet" onClick={() => setRows((rs) => rs.filter((_, j) => j !== i))}>Remove</button>}
        </div>
      ))}
      <div className="actions">
        {rows.length < 4 && <button className="btn sm" onClick={() => setRows((rs) => [...rs, blank(usable[0]?.id ?? '')])}>Add a configuration</button>}
        <select className="sel" value={measure} onChange={(e) => setMeasure(e.target.value as 'profile' | 'quick' | 'quality')} aria-label="What to measure">
          <option value="profile">Usage profile for each (14 prompts)</option>
          <option value="quick">Quick score for each (3 prompts)</option>
          <option value="quality">Quality suite for each (228 items)</option>
        </select>
      </div>
      {measure === 'quick' && <p className="small muted">With three prompts only large, consistent differences can be told from noise; the usage profile's fourteen tell much smaller ones.</p>}
      <div className="actions">
        {asking ? <><span className="small">The engine will restart{engine?.connections ? ` and ${engine.connections} open connection${engine.connections === 1 ? '' : 's'} will be cut` : ''}. Go?</span>
          <button className="btn primary" onClick={start}>Compare {rows.length} configurations</button><button className="btn quiet" onClick={() => setAsking(false)}>Not now</button></>
          : <button className="btn primary" disabled={!valid || stale || remote || !!current} onClick={() => setAsking(true)}>Compare {rows.length} configurations</button>}
      </div>
    </div>}

    {going && <div className="liverun" aria-live="polite">
      <div className="liverun-head"><b>{going.progress?.phase === 'restoring' ? 'Putting the engine back' : `Configuration ${(going.progress?.index ?? 0) + 1} of ${going.configs.length}: ${going.rows[going.progress?.index ?? 0]?.label ?? ''}`}</b>
        <span className="muted small">{going.progress?.phase === 'engine' ? 'restarting the engine and waiting until it serves' : run ? `${run.progress.done} of ${run.progress.total} replies${run.progress.current ? `, ${run.progress.current}` : ''}` : ''}</span></div>
      <div className={`progress${run && run.progress.done ? '' : ' indeterminate'}`}><i style={run && run.progress.done ? { width: `${(run.progress.done / Math.max(1, run.progress.total)) * 100}%` } : undefined} /></div>
      <div className="liverun-foot"><span className="small muted">Each configuration is also an ordinary run in the list below.</span>
        <button className="btn sm" onClick={() => api.seriesCancel(going.id).then(() => toast('Stopping after the current reply; the engine is then put back')).catch((e) => toast(String(e), 'bad'))}>Stop</button></div>
    </div>}

    {shown && shown.rows.some((r) => r.run || r.error) && <div className="series-result">
      <div className="compare-bar"><b>{shown.status === 'running' ? 'So far' : `Comparison of ${new Date(shown.created * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}`}</b>
        {list.length > 1 && <select className="sel" value={shown.id} onChange={(e) => setShown(list.find((x) => x.id === e.target.value) ?? null)} aria-label="Earlier comparisons">
          {list.map((x) => <option key={x.id} value={x.id}>{new Date(x.created * 1000).toLocaleString()}: {x.rows.map((r) => r.label).join(' / ')}</option>)}
        </select>}
        {shown.restore_error && <span className="small bad">{shown.restore_error}</span>}
      </div>
      <Matrix doc={shown} onOpen={onOpen} onCompare={onCompare} />
      <p className="small muted">Against the first: the median change in speed prompt by prompt, and whether a sign test tells it from noise. Hover it for the details; Compare opens the full side by side.</p>
    </div>}
  </>
}
