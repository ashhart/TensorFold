// Two runs side by side: the verdicts first, then the overall figures and every prompt, with a swap.
import { useEffect, useState } from 'react'
import { api, download, gb, type CompareOverall, type ComparePair, type Comparison, type CompareWho } from '../api'
import { secs } from '../units'

const tps = (x: number | null | undefined) => (x == null ? '—' : x >= 20 ? String(Math.round(x)) : x.toFixed(1))
const joules = (x: number | null | undefined) => (x == null ? '—' : x < 1 ? `${Math.round(x * 1000)} mJ` : `${x.toFixed(2)} J`)
const pct = (x: number | null | undefined) => (x == null ? '—' : `${x >= 0 ? '+' : '−'}${Math.abs(Math.round(x * 100))}%`)
const when = (t: number) => new Date(t * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })

// for each figure: how it reads, and whether a higher value is the better one
const ROWS: { key: keyof CompareOverall; label: string; fmt: (x: number | null) => string; higher: boolean }[] = [
  { key: 'tps', label: 'Writing, tok/s', fmt: tps, higher: true },
  { key: 'tps_no_drafts', label: 'Without drafts', fmt: tps, higher: true },
  { key: 'ttft', label: 'First token', fmt: secs, higher: false },
  { key: 'j_per_token', label: 'Energy per token', fmt: joules, higher: false },
  { key: 'avg_watts', label: 'Average power', fmt: (x) => (x == null ? '—' : `${Math.round(x)} W`), higher: false },
  { key: 'engine_peak_bytes', label: 'Engine memory, peak', fmt: (x) => gb(x, 1), higher: false },
]

/** A change shown as better or worse; "±0" when it is below what the page displays. */
function Delta({ change, higher, noise = false }: { change: number | null; higher: boolean; noise?: boolean }) {
  if (change == null || !isFinite(change)) return <span className="muted">—</span>
  if (Math.abs(change) < 0.005) return <span className="muted">±0</span>
  if (noise) return <span className="muted" title="no clear difference">{pct(change)}</span>   // not coloured: the test calls it noise
  const good = change > 0 === higher
  return <span className={good ? 'good' : 'bad'}>{pct(change)}</span>
}

function swapped(c: Comparison): Comparison {
  const flip = (p: ComparePair): ComparePair => ({ ...p, a: p.b, b: p.a, change: p.change == null ? null : 1 / (1 + p.change) - 1 })
  return { ...c, a: c.b, b: c.a, pairs: c.pairs.map(flip), overall: { a: c.overall.b, b: c.overall.a },
    speed: { ...c.speed, b_faster: c.speed.b_slower, b_slower: c.speed.b_faster, median_change: c.speed.median_change == null ? null : 1 / (1 + c.speed.median_change) - 1 },
    exactness: { ...c.exactness, only_a: c.exactness.only_b, only_b: c.exactness.only_a } }
}

function Who({ w, tag }: { w: CompareWho; tag: string }) {
  return <div className="who"><b>{tag}</b><span>{w.model} · {w.engine ?? 'engine not identified'} · {w.chip ?? ''} · {when(w.created)}{w.suites_version ? <> · suites <code>{w.suites_version.digest}</code></> : null}</span></div>
}

export default function Compare({ a, b, onClose }: { a: string; b: string; onClose: () => void }) {
  const [c, setC] = useState<Comparison | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [flip, setFlip] = useState(false)
  const [onlyDiff, setOnlyDiff] = useState(true)
  useEffect(() => { setC(null); setErr(null); api.benchCompare(a, b).then(setC).catch((e) => setErr(String(e))) }, [a, b])
  if (err) return <div className="card report"><div className="notice bad">{err}</div><button className="btn sm" onClick={onClose}>Close</button></div>
  if (!c) return <div className="card report"><p className="muted">Comparing…</p></div>
  // the server's sentences name A and B as asked; swapped, the page says them from the figures it has
  const v = flip ? swapped(c) : c
  const [ida, idb] = flip ? [b, a] : [a, b]
  const prefill = v.a.mode === 'prefill'
  const graded = v.a.mode === 'quality'
  const good = graded ? 'right' : 'identical'
  const differs = (p: ComparePair) => !p.same_text || p.a.pass !== p.b.pass || (p.change != null && Math.abs(p.change) >= 0.005)
  const shown = onlyDiff ? v.pairs.filter(differs) : v.pairs
  const speedLine = v.speed.median_change == null ? null
    : `B is ${Math.abs(v.speed.median_change) < 0.005 ? 'as fast' : `${Math.abs(Math.round(v.speed.median_change * 100))}% ${v.speed.median_change > 0 ? 'faster' : 'slower'}`} at the median: ${v.speed.too_few ? 'too few prompts to tell' : v.speed.within_noise ? 'no clear difference' : 'a real difference'} (faster on ${v.speed.b_faster} of ${v.speed.b_faster + v.speed.b_slower} prompts, p = ${v.speed.p < 0.001 ? '< 0.001' : v.speed.p.toFixed(3)}).`
  return (
    <div className="card report compare">
      <div className="report-head">
        <div><h2>A against B</h2><Who w={v.a} tag="A" /><Who w={v.b} tag="B" /></div>
        <div className="actions">
          <button className="btn sm" onClick={() => setFlip((f) => !f)} title="Swap A and B">Swap</button>
          <button className="btn sm" onClick={() => download(`/api/bench/compare/report?a=${ida}&b=${idb}`, `tensorfold-compare-${ida}-${idb}.html`).catch((e) => setErr(`Not downloaded: ${e}`))}>Report (HTML)</button>
          <button className="btn sm quiet" onClick={onClose}>Close</button>
        </div>
      </div>
      {v.notes.length > 0 && <ul className="small muted">{v.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
      <ul className="findings">
        {speedLine && <li><b>Speed:</b> {speedLine}</li>}
        {v.exactness.checked > 0 && <li><b>{graded ? 'Quality' : 'Exactness'}:</b> {v.exactness.only_a === 0 && v.exactness.only_b === 0 ? `the same outcome on all ${v.exactness.checked} items.` : `${v.exactness.only_a} ${good} only in A, ${v.exactness.only_b} only in B, ${v.exactness.both} on both: ${v.exactness.too_few ? 'too few differing items to tell' : v.exactness.within_noise ? 'no clear difference' : 'a real difference'} (McNemar, p = ${v.exactness.p.toFixed(3)}).`}</li>}
        {Object.entries(v.by_suite ?? {}).filter(([, d]) => d.p < 0.05).map(([k, d]) => {
          const n = Object.keys(v.by_suite ?? {}).length
          const [oa, ob] = flip ? [d.only_b, d.only_a] : [d.only_a, d.only_b]
          return <li key={k}><b>{d.name}:</b> {oa} {good} only in A, {ob} only in B (McNemar, p = {d.p.toFixed(3)}){n > 1 ? (d.p < 0.05 / n ? `; it holds even counting the ${n} groups tested.` : `; one of ${n} groups tested, so it can be chance.`) : '.'}</li>
        })}
        <li><b>Replies:</b> {v.same_text} of {v.pairs.length} word for word the same.</li>
      </ul>
      {!prefill && <div className="tablewrap"><table className="tbl">
        <thead><tr><th /><th>A</th><th>B</th><th>B against A</th></tr></thead>
        <tbody>{ROWS.filter((r) => v.overall.a[r.key] != null || v.overall.b[r.key] != null).map((r) => {
          const x = v.overall.a[r.key], y = v.overall.b[r.key]
          return <tr key={r.key}><td>{r.label}</td><td>{r.fmt(x)}</td><td>{r.fmt(y)}</td><td><Delta change={x && y ? y / x - 1 : null} higher={r.higher} noise={v.speed.within_noise && (r.key === 'tps' || r.key === 'tps_no_drafts')} /></td></tr>
        })}
          {v.overall.a.checked ? <tr><td>Identical without drafts</td><td>{v.overall.a.exact}/{v.overall.a.checked}</td><td>{v.overall.b.exact}/{v.overall.b.checked}</td><td /></tr> : null}
        </tbody>
      </table></div>}
      <div className="compare-bar">
        <b>Prompt by prompt</b>
        <label className="small"><input type="checkbox" checked={onlyDiff} onChange={(e) => setOnlyDiff(e.target.checked)} /> Only differences ({v.pairs.filter(differs).length} of {v.pairs.length})</label>
      </div>
      {shown.length === 0 ? <p className="small muted">No difference: every prompt has the same reply, the same exactness and a speed within display precision.</p>
        : <div className="tablewrap"><table className="tbl">
          <thead><tr><th>Prompt</th><th>A {prefill ? 'prefill tok/s' : 'tok/s'}</th><th>B</th><th>change</th><th>{graded ? 'right, A / B' : 'identical w/o drafts, A / B'}</th><th>replies</th></tr></thead>
          <tbody>{shown.map((p) => {
            const [x, y] = prefill ? [p.a.prefill_tps, p.b.prefill_tps] : [p.a.tps, p.b.tps]
            const mark = (m: boolean | null) => (m == null ? '—' : m ? 'yes' : <span className="bad">no</span>)
            return <tr key={`${p.suite}/${p.prompt}`}><td>{p.suite_name}: {p.prompt}</td><td>{tps(x)}</td><td>{tps(y)}</td><td><Delta change={x && y ? y / x - 1 : null} higher noise={v.speed.within_noise} /></td>
              <td>{mark(p.a.pass)} / {mark(p.b.pass)}</td><td>{p.same_text ? 'same' : <span className="warn-text">differ</span>}</td></tr>
          })}</tbody>
        </table></div>}
      <p className="small muted">Speed: a sign test on which run was faster, prompt by prompt. Exactness: an exact McNemar test on the prompts identical with drafts off in only one run. A difference is reported as one when p is under 0.05; a median change under 2% counts as the same speed.</p>
    </div>
  )
}
