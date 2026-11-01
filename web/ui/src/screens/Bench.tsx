import { useCallback, useEffect, useMemo, useState } from 'react'
import { api, download, gb, num, type BenchBlock, type BenchHardware, type LoadLevel, type BenchMode, type BenchRun, type BenchRunRow, type BenchSuites, type SweepPoint } from '../api'
import type { Ctx } from '../ctx'
import { showSub, useSub } from '../route'
import { KL_START, KlOptions, KlReport, PeerPick, klReference, klText, useFleetPeers, type KlState } from './BenchKl'
import { copyText, shareImage } from '../shareCard'
import CustomSuites from './BenchSuites'
import Compare from './BenchCompare'
import Series from './BenchSeries'
import { ParallelPanel } from './ChatParallel'
import { clock, secs } from '../units'

const pct = (x: number | null | undefined) => (x == null ? '—' : `${x >= 0 ? '+' : ''}${Math.round(x * 100)}%`)
const tps = (x: number | null | undefined) => (x == null ? '—' : `${x.toFixed(x < 10 ? 1 : 0)}`)
const when = (t: number) => new Date(t * 1000).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
const KIND: Record<BenchMode, string> = { quick: 'Quick score', profile: 'Usage profile', prefill: 'Prefill sweep', load: 'Load sweep', quality: 'Quality', tools: 'Tool calls', kl: 'KL divergence' }
// the form's choice: a kind of run, or a comparison of configurations
type Choice = BenchMode | 'series'
// one tile per kind of run: a sign for what it measures, and what it does in a few words
const RUN: Record<Choice, { icon: string; what: string }> = {
  quick: { icon: 'M13 2L4 14h7l-1 8 9-12h-7z', what: 'Three prompts, with and without the speed-up. About a minute.' },
  profile: { icon: 'M5 20v-6M11 20V6M17 20v-9M3 20h18', what: 'Seven kinds of use, fourteen prompts: the speed of each.' },
  quality: { icon: 'M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM8 12l3 3 5-6', what: 'Are the answers right: 228 items, scored strictly.' },
  tools: { icon: 'M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.8-3.8a6 6 0 0 1-7.9 7.9l-6.9 6.9a2.1 2.1 0 0 1-3-3l6.9-6.9a6 6 0 0 1 7.9-7.9z', what: 'Does it call tools properly, as an agent would need.' },
  prefill: { icon: 'M4 6h4M4 10h8M4 14h12M4 18h16', what: 'Reading long prompts: from 1k tokens up to the window.' },
  load: { icon: 'M4 16h10v4H4zM7 10h10v4H7zM10 4h10v4H10z', what: 'Several requests at once: 1, 2, 4… up to 32.' },
  kl: { icon: 'M2 18c4 0 5-12 8-12s4 12 8 12M6 18c4 0 5-9 8-9s4 9 8 9', what: 'How far a setup drifts from a reference, token by token.' },
  series: { icon: 'M4 6h16M4 12h16M4 18h16M8 4v4M16 10v4M11 16v4', what: 'Two to four configurations measured in turn, side by side.' },
}
const kTokens = (n: number) => (n >= 1024 ? `${Math.round(n / 1024)}k` : `${n}`)
const STATUS: Record<string, string> = { running: 'Running', done: 'Done', cancelled: 'Cancelled', failed: 'Failed', interrupted: 'Interrupted by a TensorFold Web restart' }

/** Speed with and without drafts, one pair of bars per suite, on a common scale. */
function Bars({ rows, names }: { rows: [string, BenchBlock][]; names: Record<string, string> }) {
  const max = Math.max(1, ...rows.flatMap(([, b]) => [b.tps ?? 0, b.tps_no_drafts ?? 0]))
  return (
    <div className="bars" role="img" aria-label="Tokens a second per suite, with and without drafts">
      {rows.map(([id, b]) => (
        <div className="bar-row" key={id}>
          <span className="name">{names[id] ?? id}</span>
          <span className="pair">
            <i style={{ width: `${((b.tps ?? 0) / max) * 100}%` }} /><em>{tps(b.tps)}</em>
            {b.tps_no_drafts != null && <><i className="off" style={{ width: `${(b.tps_no_drafts / max) * 100}%` }} /><em className="off">{tps(b.tps_no_drafts)}</em></>}
          </span>
        </div>
      ))}
      <div className="legend"><span><i /> drafts on</span><span><i className="off" /> drafts off</span><span>tokens a second, median of the suite</span></div>
    </div>
  )
}

/** Prefill speed (bars) and time to first token (line) against the prompt size, sizes on a log scale. */
function SweepChart({ points }: { points: SweepPoint[] }) {
  const ok = points.filter((p) => p.prefill_tps != null && !p.error)
  if (!ok.length) return null
  const W = 640, H = 200, L = 44, R = 44, T = 12, B = 28
  const maxTps = Math.max(1, ...ok.map((p) => p.prefill_tps ?? 0))
  const maxT = Math.max(0.001, ...ok.map((p) => p.ttft ?? 0))
  const step = (W - L - R) / ok.length
  const x = (i: number) => L + step * (i + 0.5)
  const yT = (v: number) => T + (H - T - B) * (1 - v / maxT)
  const line = ok.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${yT(p.ttft ?? 0).toFixed(1)}`).join(' ')
  return (
    <div className="sweep">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Prefill tokens a second and time to first token by prompt size">
        {ok.map((p, i) => {
          const h = (H - T - B) * ((p.prefill_tps ?? 0) / maxTps)
          return <g key={p.size}>
            <rect x={x(i) - step * 0.3} y={H - B - h} width={step * 0.6} height={h} rx="3" fill="var(--accent)" opacity=".8" />
            <text x={x(i)} y={H - 9} textAnchor="middle">{kTokens(p.size)}</text>
            <text x={x(i)} y={H - B - h - 4} textAnchor="middle" className="v">{num(Math.round(p.prefill_tps ?? 0))}</text>
          </g>
        })}
        <path d={line} fill="none" stroke="var(--warn)" strokeWidth="2" />
        {ok.map((p, i) => <circle key={p.size} cx={x(i)} cy={yT(p.ttft ?? 0)} r="3" fill="var(--warn)" />)}
        <text x={W - R + 6} y={yT(maxT) + 4} className="v t">{secs(maxT)}</text>
      </svg>
      <div className="legend"><span><i /> prefill, tokens a second</span><span><i className="t" /> first token</span><span>prompt size in tokens</span></div>
    </div>
  )
}

const watts = (x: number | null | undefined) => (x == null ? '—' : `${x.toFixed(x < 10 ? 1 : 0)} W`)
const money = (x: number | null | undefined, cur: string) => (x == null ? '—' : new Intl.NumberFormat('en', { style: 'currency', currency: cur, maximumSignificantDigits: 2 }).format(x))
const joules = (x: number | null | undefined) => (x == null ? '—' : x < 1 ? `${Math.round(x * 1000)} mJ` : `${x.toFixed(2)} J`)
const deg = (x: number | null | undefined) => (x == null ? '—' : `${Math.round(x)} °C`)

/** Power (filled) and the hottest GPU or CPU probe (line) over the run. */
function HeatChart({ hw }: { hw: BenchHardware }) {
  const rows = hw.series.filter((r) => r[1] != null || r[3] != null || r[2] != null)
  if (rows.length < 2) return null
  const W = 640, H = 140, T = 10, B = 20
  const tEnd = Math.max(1, ...rows.map((r) => r[0] ?? 0))
  const maxW = Math.max(1, ...rows.map((r) => r[1] ?? 0))
  const x = (t: number) => (t / tEnd) * W
  const yW = (w: number) => T + (H - T - B) * (1 - w / maxW)
  const yC = (c: number) => T + (H - T - B) * (1 - c / 110)
  const pw = rows.filter((r) => r[1] != null)
  const area = pw.length > 1 ? `M${x(pw[0][0]!)},${H - B} ` + pw.map((r) => `L${x(r[0]!).toFixed(1)},${yW(r[1]!).toFixed(1)}`).join(' ') + ` L${x(pw[pw.length - 1][0]!)},${H - B} Z` : ''
  const hot = rows.map((r) => [r[0]!, Math.max(r[2] ?? 0, r[3] ?? 0)] as const).filter(([, c]) => c > 0)
  const line = hot.map(([t, c], i) => `${i ? 'L' : 'M'}${x(t).toFixed(1)},${yC(c).toFixed(1)}`).join(' ')
  return (
    <div className="sweep">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Power and temperature during the run">
        {area && <path d={area} fill="var(--accent-soft)" stroke="var(--accent)" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />}
        {line && <path d={line} fill="none" stroke="var(--warn)" strokeWidth="2" vectorEffect="non-scaling-stroke" />}
        {rows.filter((r) => r[5]).map((r) => <rect key={r[0]} x={x(r[0]) - 2} y={H - B + 4} width={4} height={6} fill="var(--bad)"><title>slowed down</title></rect>)}
      </svg>
      <div className="legend"><span><i /> power, up to {watts(maxW)}</span><span><i className="t" /> hottest probe, scale 0 to 110 °C</span>{rows.some((r) => r[5]) && <span className="bad">▮ slowed down</span>}<span>{Math.round(tEnd)} s</span></div>
    </div>
  )
}

/** In a prefill sweep the work is reading prompts, so energy goes per thousand prompt tokens there. */
function Heat({ hw, prefill }: { hw: BenchHardware; prefill: boolean }) {
  return (
    <div className="heat">
      <b>Power and heat</b>
      <dl className="figures">
        {prefill ? <>
          <div><dt>Energy per 1,000 prompt tokens</dt><dd title={hw.per_token_note}>{joules(hw.j_per_1k_prompt ?? null)}</dd></div>
          {hw.j_per_1k_prompt_above_idle != null && <div><dt>above idle</dt><dd>{joules(hw.j_per_1k_prompt_above_idle)}</dd></div>}
        </> : <>
          <div><dt>Energy per output token</dt><dd title={hw.per_token_note}>{joules(hw.j_per_token)}</dd></div>
          {hw.j_per_token_above_idle != null && <div><dt>above idle</dt><dd>{joules(hw.j_per_token_above_idle)}</dd></div>}
        </>}
        <div><dt>Average power</dt><dd>{watts(hw.avg_watts)}</dd></div>
        <div><dt>Peak</dt><dd>{watts(hw.peak_watts)}</dd></div>
        <div><dt>Idle</dt><dd>{watts(hw.idle_watts)}</dd></div>
        <div><dt>Hottest CPU</dt><dd>{deg(hw.peak_cpu_c)}</dd></div>
        <div><dt>Hottest GPU</dt><dd>{deg(hw.peak_gpu_c)}</dd></div>
        {hw.throttle && <div><dt>Slowed down</dt><dd className={hw.throttle.throttled ? 'bad' : hw.throttle.throttled === false ? 'good' : ''}>{hw.throttle.throttled == null ? 'not reported' : hw.throttle.throttled ? `yes, ${Math.round(hw.throttle.seconds ?? 0)} s` : 'no'}</dd></div>}
        {hw.cost && <div><dt>Energy cost</dt><dd>{money(hw.cost.run, hw.cost.currency)}</dd></div>}
        {hw.cost?.per_million_tokens != null && hw.j_per_token != null && <div><dt>per million tokens</dt><dd>{money(hw.cost.per_million_tokens, hw.cost.currency)}</dd></div>}
      </dl>
      <HeatChart hw={hw} />
      {hw.throttle?.reasons.length ? <p className="small">Why it slowed down: {hw.throttle.reasons.join(', ')}. Speeds measured while the machine slows itself down understate it.</p> : null}
      <p className="small muted">{hw.estimate ? `Estimate: ${hw.source} measures the ${hw.scope} only, not the whole machine. ` : `Measured for the whole machine (${hw.source}). `}Energy over the measured part of the run ({Math.round(hw.measured_s)} s, prompts read included) divided by {prefill ? 'the prompt tokens read' : `the ${num(hw.output_tokens)} tokens written`}{hw.per_token_note ? ` (per token: ${hw.per_token_note})` : ''}; idle is the machine with the model loaded and nothing asked, just before.{hw.coverage != null && hw.coverage < 0.98 ? ` Power readings cover ${Math.round(hw.coverage * 100)}% of the run; gaps over 10 s are left out, so the energy is a lower bound.` : ''}{hw.cost ? ` Cost at ${money(hw.cost.price_per_kwh, hw.cost.currency)} per kWh, set in Settings.` : ' Set an electricity price in Settings to see the cost.'}</p>
    </div>
  )
}

/** A load sweep: total speed and each request's as more are served at once, with the wait for the first token. */
function LoadReport({ levels }: { levels: LoadLevel[] }) {
  const pts = levels.filter((l) => l.total_tps != null)
  if (!pts.length) return null
  const best = pts.reduce((a, b) => ((b.total_tps ?? 0) > (a.total_tps ?? 0) ? b : a))
  const W = 640, H = 170, L = 40, B = 22
  const top = Math.max(1, ...pts.map((p) => p.total_tps ?? 0)) * 1.1
  const x = (i: number) => L + (pts.length > 1 ? i / (pts.length - 1) : 0.5) * (W - L - 16)
  const y = (v: number) => 10 + (H - B - 10) * (1 - v / top)
  const line = (k: 'total_tps' | 'per_request_tps') => pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p[k] ?? 0).toFixed(1)}`).join(' ')
  return <>
    <dl className="figures big">
      {/* "alone" is the level of one request, measured; not the first level that happens to have a figure */}
      <div><dt>Alone</dt><dd>{tps(levels.find((l) => l.concurrency === 1)?.total_tps ?? null)} tok/s</dd></div>
      <div><dt>Total at {best.concurrency} at once</dt><dd>{tps(best.total_tps)} tok/s</dd></div>
      <div><dt>Each, then</dt><dd>{tps(best.per_request_tps)} tok/s</dd></div>
      <div><dt>First token p95, then</dt><dd>{secs(best.ttft_p95)}</dd></div>
    </dl>
    <div className="sweep"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Speed by requests at once">
      <path d={line('total_tps')} fill="none" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
      <path d={line('per_request_tps')} fill="none" stroke="var(--ink3)" strokeWidth="2" strokeDasharray="4 4" vectorEffect="non-scaling-stroke" />
      {pts.map((p, i) => <g key={p.concurrency}><text x={x(i)} y={H - 6} textAnchor="middle">{p.concurrency}</text><text className="v" x={x(i)} y={y(p.total_tps ?? 0) - 8} textAnchor="middle">{tps(p.total_tps)}</text></g>)}
    </svg><div className="legend"><span><i /> total tokens a second</span><span><i className="t" /> each request's</span><span>by requests at once</span></div></div>
    <div className="tablewrap"><table className="tbl">
      <thead><tr><th>At once</th><th>total tok/s</th><th>each</th><th>first token p50</th><th>p95</th><th>requests</th></tr></thead>
      <tbody>{levels.map((l) => <tr key={l.concurrency}><td>{l.concurrency}</td><td>{tps(l.total_tps)}</td><td>{tps(l.per_request_tps)}</td><td>{secs(l.ttft_p50)}</td><td>{secs(l.ttft_p95)}</td><td>{l.requests}{l.errors ? <small className="bad"> {l.errors} failed</small> : null}</td></tr>)}</tbody>
    </table></div>
    <p className="small muted">Each level sends at least eight requests, in waves of its size, the prompts taken in turn from the usage suites, 200-token replies. Total: the tokens of a wave over its duration. The first token includes the wait for a turn.</p>
  </>
}

/** A quality run: right answers by category, and every item missed with what was expected and what was read. */
function QualityReport({ run, names }: { run: BenchRun; names: Record<string, string> }) {
  const q = run.summary?.quality
  const [all, setAll] = useState(false)
  if (!q) return null
  const missed = run.results.filter((r) => r.passed === false)
  const share = (c: { items: number; passed: number }) => `${c.passed}/${c.items}`
  return <>
    <dl className="figures big">
      <div><dt>Right</dt><dd>{q.overall.passed} of {q.overall.items}</dd></div>
      {!!q.overall.unscored && <div><dt>Not graded</dt><dd className="bad">{q.overall.unscored} <small>(request failed)</small></dd></div>}
      {Object.entries(q.categories).map(([k, c]) => <div key={k}><dt>{names[k] ?? k}</dt><dd>{share(c)}</dd></div>)}
    </dl>
    <p className="small muted">Scored strictly: the last value counts, an answer offering two fails, a reasoning block left open leaves no answer. Two quality runs compare item by item (tick them in the list below).</p>
    {missed.length > 0 && <details open={all} onToggle={(e) => setAll((e.target as HTMLDetailsElement).open)}>
      <summary>{missed.length} items missed</summary>
      <div className="tablewrap"><table className="tbl">
        <thead><tr><th>Category</th><th>Item</th><th>expected</th><th>read</th></tr></thead>
        <tbody>{missed.map((r) => <tr key={r.prompt}><td>{names[r.suite] ?? r.suite}</td><td>{r.prompt}</td><td>{r.expected ?? '—'}</td><td title={r.text}>{r.read ?? '—'}</td></tr>)}</tbody>
      </table></div>
    </details>}
  </>
}

function SweepReport({ run }: { run: BenchRun }) {
  const pts = run.summary?.sweep ?? []
  const done = pts.filter((p) => !p.error && p.prefill_tps != null)
  const last = done[done.length - 1]
  const first = done[0]
  return <>
    {last && <dl className="figures big">
      <div><dt>Prefill at {kTokens(first.size)}</dt><dd>{num(Math.round(first.prefill_tps ?? 0))} tok/s</dd></div>
      <div><dt>Prefill at {kTokens(last.size)}</dt><dd>{num(Math.round(last.prefill_tps ?? 0))} tok/s</dd></div>
      <div><dt>First token at {kTokens(last.size)}</dt><dd>{secs(last.ttft)}</dd></div>
      {run.window ? <div><dt>Served window</dt><dd>{num(run.window)}</dd></div> : null}
      {run.memory?.engine_peak_bytes ? <div><dt>Engine memory, peak</dt><dd>{gb(run.memory.engine_peak_bytes, 1)}</dd></div> : null}
    </dl>}
    <p className="small muted">One prompt per size, each opening with its own reference so the prompt cache cannot serve it; the reply is cut at 8 tokens. Prefill speed is the engine's prompt tokens over its prefill time{done.some((p) => p.estimated) ? '; this engine gives none, so ≈ marks an estimate: the first token less a measured overhead' : ''}{done.some((p) => p.cached) ? '; some tokens came from the cache, see the notes' : ''}.</p>
    <SweepChart points={pts} />
    <div className="tablewrap"><table className="tbl">
      <thead><tr><th>Size</th><th>prompt tokens</th><th>from cache</th><th>prefill</th><th>prefill tok/s</th><th>first token</th></tr></thead>
      <tbody>{pts.map((p) => <tr key={p.size}><td>{kTokens(p.size)}</td><td>{p.prompt_tokens == null ? '—' : num(p.prompt_tokens)}</td><td>{p.cached == null ? '—' : num(p.cached)}</td>
        <td>{secs(p.prefill_s)}</td><td>{p.prefill_tps == null ? '—' : <>{p.estimated ? '≈ ' : ''}{num(Math.round(p.prefill_tps))}{p.weak ? <small className="warn-text" title="the overhead is most of the time measured"> weak</small> : null}</>}</td><td>{p.error ? <span className="bad">{p.error}</span> : secs(p.ttft)}</td></tr>)}</tbody>
    </table></div>
  </>
}

const PHASE: Record<string, string> = { starting: 'Starting', idle: 'Measuring idle power', 'warm-up': 'Warming up the engine', measuring: 'Measuring' }

/** The run while it goes: what it is doing, for how long, about how long is left, and a stop that asks once. */
function LiveRun({ run, onStop }: { run: BenchRun; onStop: () => void }) {
  const [now, setNow] = useState(() => Date.now())
  const [asking, setAsking] = useState(false)
  useEffect(() => { const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t) }, [])
  // the server's clock, as of the last poll: elapsed and left are counted on it, whatever this browser's clock says
  const offset = useMemo(() => (run.server_time ? run.server_time * 1000 - Date.now() : 0), [run.server_time])
  useEffect(() => setNow(Date.now()), [offset])      // the offset taken now: the clock it applies to is now's too
  const serverNow = (now + offset) / 1000
  const p = run.progress
  const known = p.phase === 'measuring' && p.done > 0
  const left = p.eta_s != null && p.eta_at != null ? p.eta_s - (serverNow - p.eta_at) : null
  const unit = run.mode === 'prefill' ? 'sizes' : run.mode === 'load' ? 'levels' : run.mode === 'tools' ? 'scenarios' : run.mode === 'quality' ? 'items' : 'replies'
  return (
    <div className="liverun" aria-live="polite">
      <div className="liverun-head">
        <b>{p.stopping ? 'Stopping after the current reply' : PHASE[p.phase ?? ''] ?? 'Running'}{p.current && p.phase === 'measuring' && !p.stopping ? <span className="muted">: {p.current}</span> : null}</b>
        <span className="num muted">{clock(serverNow - run.created)} elapsed{known && left != null ? (left > 1 ? `, about ${clock(left)} left` : ', almost done') : ''}</span>
      </div>
      <div className={`progress${known ? '' : ' indeterminate'}`} role="progressbar" aria-valuemin={0} aria-valuemax={p.total} aria-valuenow={known ? p.done : undefined}>
        <i style={known ? { width: `${(p.done / Math.max(1, p.total)) * 100}%` } : undefined} />
      </div>
      <div className="liverun-foot">
        <span className="small muted">{p.done} of {p.total} {unit}. Chat with this engine during a run and the figures include your conversation.</span>
        {asking && !p.stopping
          ? <span className="actions"><span className="small">Stop this run? The replies so far are kept.</span><button className="btn sm danger" onClick={() => { setAsking(false); onStop() }}>Stop the run</button><button className="btn sm quiet" onClick={() => setAsking(false)}>Keep going</button></span>
          : <button className="btn sm" disabled={p.stopping} onClick={() => setAsking(true)}>Stop</button>}
      </div>
    </div>
  )
}

/** One button for the text, a menu for the image: copied where the browser allows it, downloaded otherwise. */
function Share({ run, names, toast }: { run: BenchRun; names: Record<string, string>; toast: Ctx['toast'] }) {
  const [open, setOpen] = useState(false)
  useEffect(() => {
    if (!open) return
    const close = (e: Event) => { if (e instanceof KeyboardEvent ? e.key === 'Escape' : !(e.target as HTMLElement).closest('.split')) setOpen(false) }
    document.addEventListener('mousedown', close); document.addEventListener('keydown', close)
    return () => { document.removeEventListener('mousedown', close); document.removeEventListener('keydown', close) }
  }, [open])
  const text = async () => { try { await copyText(run.share_text ?? ''); toast('Summary copied') } catch (e) { toast(String(e), 'bad') } }
  const image = async () => {
    setOpen(false)
    try { toast((await shareImage(run, names)) === 'copied' ? 'Card copied as an image' : 'This browser does not copy images here; the card was downloaded') } catch (e) { toast(String(e), 'bad') }
  }
  return (
    <span className="split">
      <button className="btn sm" onClick={text} disabled={!run.share_text}>Copy summary</button>
      <button className="btn sm" aria-haspopup="menu" aria-expanded={open} aria-label="More ways to share" onClick={() => setOpen((o) => !o)}>▾</button>
      {open && <span className="menu" role="menu">
        <button role="menuitem" onClick={image}>Card as an image</button>
        <button role="menuitem" onClick={() => { setOpen(false); download(`/api/bench/runs/${run.id}/report`, `tensorfold-bench-${run.id}.html`).catch((e) => toast(`Not downloaded: ${e}`, 'bad')) }}>Report (HTML)</button>
        <button role="menuitem" onClick={() => { setOpen(false); download(`/api/bench/runs/${run.id}/receipt`, `tensorfold-bench-${run.id}.json`).catch((e) => toast(`Not downloaded: ${e}`, 'bad')) }}>Receipt (JSON)</button>
      </span>}
    </span>
  )
}

function Report({ run, names, onDelete, toast }: { run: BenchRun; names: Record<string, string>; onDelete: () => void; toast: Ctx['toast'] }) {
  const o = run.summary?.overall
  const suites = Object.entries(run.summary?.suites ?? {}).sort((a, b) => run.suites.indexOf(a[0]) - run.suites.indexOf(b[0]))
  const timing = run.results.some((r) => r.m.timing === 'engine') ? 'measured by the engine' : 'timed by this TensorFold Web'
  const exactAll = o && o.checked > 0 && o.exact === o.checked
  return (
    <div className="card report">
      <div className="report-head">
        <div>
          <h2>{KIND[run.mode]}: {run.model}</h2>
          <p className="small muted">{when(run.created)}. {run.engine ?? 'Engine not identified'} on {run.machine?.chip ?? 'this machine'}{run.machine?.ram_bytes ? `, ${gb(run.machine.ram_bytes)}` : ''}. {STATUS[run.status] ?? run.status}.</p>
        </div>
        <div className="actions">
          {run.status !== 'running' && <Share run={run} names={names} toast={toast} />}
          <button className="btn sm quiet" onClick={onDelete} disabled={run.status === 'running'}>Delete this run</button>
        </div>
      </div>
      {run.error && <div className="notice bad">{run.error}{run.error_detail && run.error_detail !== run.error ? <details className="small"><summary>What the engine said</summary><code>{run.error_detail}</code></details> : null}</div>}
      {run.mode === 'prefill' && <SweepReport run={run} />}
      {run.mode === 'load' && <LoadReport levels={run.summary?.load ?? []} />}
      {run.mode === 'quality' && <QualityReport run={run} names={names} />}
      {run.mode === 'kl' && <KlReport run={run} />}
      {run.mode === 'tools' && run.tools && <><dl className="figures big"><div><dt>tool-eval-bench {run.tools.version}</dt><dd>{run.tools.final_score} of 100</dd></div><div><dt>Points</dt><dd>{run.tools.points}/{run.tools.max_points}</dd></div></dl>
        <QualityReport run={run} names={names} /></>}
      {(run.mode === 'quick' || run.mode === 'profile') && o && <dl className="figures big">
        <div><dt>Writing</dt><dd>{tps(o.tps)} tok/s</dd></div>
        {o.tps_no_drafts != null && <div><dt>Without drafts</dt><dd>{tps(o.tps_no_drafts)} tok/s</dd></div>}
        {o.gain != null && <div><dt>Gain from drafts</dt><dd>{pct(o.gain)}</dd></div>}
        {o.checked > 0 && <div><dt>Exact</dt><dd className={exactAll ? 'good' : 'bad'}>{o.exact} of {o.checked}</dd></div>}
        <div><dt>First token</dt><dd>{secs(o.ttft)}</dd></div>
        {o.ttft_answer != null && <div><dt>First answer token</dt><dd>{secs(o.ttft_answer)}</dd></div>}
        {run.memory?.engine_peak_bytes ? <div><dt>Engine memory, peak</dt><dd>{gb(run.memory.engine_peak_bytes, 1)}</dd></div> : null}
      </dl>}
      {run.status !== 'running' && run.findings && run.findings.length > 0 && <ul className="findings">{run.findings.map((f, i) => <li key={i}>{f}</li>)}</ul>}
      {(run.mode === 'quick' || run.mode === 'profile') && o && <p className="small muted">Medians over {o.prompts} prompts{o.tokens != null ? ` and ${num(o.tokens)} tokens` : ' (tokens not counted by the engine)'} with drafts, {timing}. {o.checked > 0 ? `Exact means the reply without drafts has the same tokens (${run.results.find((r) => r.compared_by)?.compared_by === 'token_sha' ? 'same token digest' : 'same text'}).` : 'No exactness check in this run.'}</p>}
      {(run.mode === 'quick' || run.mode === 'profile') && suites.length > 0 && <Bars rows={suites} names={names} />}
      {run.hardware && <Heat hw={run.hardware} prefill={run.mode === 'prefill'} />}
      {(run.mode === 'quick' || run.mode === 'profile') && suites.length > 0 && (
        <div className="tablewrap"><table className="tbl">
          <thead><tr><th>Suite</th><th>tok/s</th><th>no drafts</th><th>gain</th><th>first token</th><th>prefill</th><th>gaps p50 / p95</th><th>exact</th></tr></thead>
          <tbody>{suites.map(([id, b]) => (
            <tr key={id}><td>{names[id] ?? id}</td><td>{tps(b.tps)}</td><td>{tps(b.tps_no_drafts)}</td><td>{pct(b.gain)}</td><td>{secs(b.ttft)}</td><td>{secs(b.prefill_s)}</td>
              <td>{secs(b.gap_p50)} / {secs(b.gap_p95)}</td><td>{b.checked ? `${b.exact}/${b.checked}` : '—'}{b.errors ? <small className="bad"> {b.errors} failed</small> : null}</td></tr>
          ))}</tbody>
        </table></div>
      )}
      <details>
        <summary>Protocol and replies</summary>
        <dl className="kv">
          <dt>Sampling</dt><dd>temperature {run.protocol.temperature}, top_p {run.protocol.top_p}, one request at a time, {run.protocol.warmup_tokens}-token warm-up</dd>
          {run.mode === 'prefill' && <><dt>Sizes</dt><dd>{run.max_context ? `up to ${num(run.max_context)} tokens` : 'up to the served window'}{run.window ? `; window ${num(run.window)}` : ''}</dd></>}
          <dt>Thinking</dt><dd>{run.thinking
            ? <>{run.thinking.off ? 'off' : <b className="bad">still on: the figures include reasoning</b>}{run.thinking.sent ? `, sent as ${run.thinking.sent}` : ', no way of turning it off accepted'}{run.thinking.refused.length ? ` (refused: ${run.thinking.refused.join('; ')})` : ''}</>
            : run.protocol.thinking}</dd>
          {run.mode !== 'prefill' && run.suites_version && <><dt>Suites</dt><dd>version {run.suites_version.version ?? '?'}, digest <code>{run.suites_version.digest}</code>; runs compare only on the same digest</dd></>}
          <dt>Output format</dt><dd>{run.protocol.output_format}</dd>
          <dt>Gaps</dt><dd>time between stream events as TensorFold Web receives them; an event carries several tokens when drafts are kept</dd>
          {run.memory && <><dt>Memory</dt><dd>engine peak {gb(run.memory.engine_peak_bytes, 1)}, machine peak {gb(run.memory.system_peak_bytes, 1)} ({run.memory.samples} readings)</dd></>}
          <dt>TensorFold Web</dt><dd>{run.studio}</dd>
        </dl>
        {run.notes.length > 0 && <ul className="small">{run.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
        {run.mode !== 'prefill' && run.results.filter((r) => r.draft).map((r, i) => (
          <details key={i} className="reply"><summary>{names[r.suite] ?? r.suite}: {r.prompt} <span className="muted">{num(r.m.tokens)} tokens, {tps(r.m.tps)} tok/s{r.m.finish === 'length' ? ', reached max_tokens' : ''}</span></summary><pre>{r.error ?? r.text}</pre></details>
        ))}
      </details>
    </div>
  )
}

export default function Bench({ engine, toast, stale }: Ctx) {
  const [info, setInfo] = useState<BenchSuites | null>(null)
  const [runs, setRuns] = useState<BenchRunRow[]>([])
  const [busyId, setBusyId] = useState<string | null>(null)
  const [selected, setSelected] = useState<string | null>(null)
  const [run, setRun] = useState<BenchRun | null>(null)
  const [choice, setChoice] = useState<Choice>('quick')
  // the plain layer: one check, one button; the kinds of run and their options are a click away
  const [simple, setSimple] = useState(() => { try { return localStorage.getItem('tfs-bench-simple') !== 'no' } catch { return true } })
  const showMore = (more: boolean) => { setSimple(!more); try { localStorage.setItem('tfs-bench-simple', more ? 'no' : 'yes') } catch { /* no storage */ } }
  const mode: BenchMode = choice === 'series' ? 'profile' : choice
  const [upTo, setUpTo] = useState<number>(32768)
  const [atOnce, setAtOnce] = useState<number>(16)
  const [fullTools, setFullTools] = useState(false)
  const [chosen, setChosen] = useState<string[]>([])
  const [cats, setCats] = useState<string[]>([])
  const [exactness, setExactness] = useState(true)
  // what to measure: the engine this TensorFold Web serves, or another one at an address (a peer, for a comparison)
  const [aim, setAim] = useState<'served' | 'address'>('served')
  const [addr, setAddr] = useState(() => { try { return localStorage.getItem('tfs-bench-addr') ?? 'http://127.0.0.1:11434' } catch { return 'http://127.0.0.1:11434' } })
  const [addrKey, setAddrKey] = useState('')
  const [found, setFound] = useState<Awaited<ReturnType<typeof api.benchIdentify>> | null>(null)
  const [addrModel, setAddrModel] = useState('')
  const [addrLabel, setAddrLabel] = useState('')
  // a Fleet machine instead of an address typed by hand: this TensorFold Web reaches its engine, the page holds no secret
  const [addrPeer, setAddrPeer] = useState('')
  const peers = useFleetPeers(aim === 'address')
  const [klForm, setKlForm] = useState<KlState>(KL_START)
  const klRef = klReference(klForm)
  const klFixed = mode === 'kl' && klForm.fixed
  const [klLoad, setKlLoad] = useState(0)
  const check = async (peer = addrPeer) => {
    setFound(null)
    if (!peer) try { localStorage.setItem('tfs-bench-addr', addr) } catch { /* no storage */ }
    try {
      const f = peer ? await api.benchIdentifyPeer(peer) : await api.benchIdentify(addr, addrKey || undefined)
      setFound(f); setAddrModel(f.models[0] ?? '')
    } catch (e) { setFound({ url: addr, engine: null, version: null, models: [], error: String(e) }) }
  }
  const pickPeer = (p: string) => { setAddrPeer(p); setFound(null); if (p) check(p) }
  const [err, setErr] = useState<string | null>(null)
  const [starting, setStarting] = useState(false)
  const [ticked, setTicked] = useState<string[]>([])                    // runs chosen for a comparison, at most two
  const [pair, setPair] = useState<[string, string] | null>(null)

  useSub('bench', (id) => { setSelected(id); setPair(null) })   // #bench/<run>
  useEffect(() => { if (selected) showSub('bench', selected) }, [selected])
  const names: Record<string, string> = { ...Object.fromEntries([...(info?.suites ?? []), ...(info?.custom ?? [])].map((s) => [s.id, s.name])), ...(run?.suite_names ?? {}) }
  const loadSuites = useCallback(() => api.benchSuites().then(setInfo).catch((e) => setErr(String(e))), [])
  const refresh = useCallback(() => api.benchRuns().then((r) => {
    setRuns(r.runs); setBusyId(r.current)
    setSelected((s) => s ?? r.current ?? r.runs[0]?.id ?? null)
  }).catch((e) => setErr(String(e))), [])
  useEffect(() => { api.benchSuites().then((s) => { setInfo(s); setChosen(s.suites.map((x) => x.id)) }).catch((e) => setErr(String(e))); refresh() }, [refresh])
  // the selected run, read again every second while it goes
  useEffect(() => {
    if (!selected) { setRun(null); return }
    let live = true
    let t: ReturnType<typeof setTimeout>
    let pollFailed = false
    const load = () => api.benchRun(selected).then((r) => {
      if (!live) return
      if (pollFailed) { pollFailed = false; setErr(null) }
      setRun(r)
      if (r.status === 'running') t = setTimeout(load, 1000)
      else if (busyId === selected) refresh()
    }).catch((e) => { if (!live) return; pollFailed = true; setErr(String(e)); t = setTimeout(load, 2000) })   // one failed poll (a sleep) is not the end
    load()
    return () => { live = false; clearTimeout(t) }
  }, [selected, busyId, refresh])

  const servedReady = engine?.status === 'running'
  const ready = aim === 'served' ? servedReady : !!(found && !found.error && addrModel)
  const peerName = peers?.find((p) => p.id === addrPeer)?.name
  const targetName = aim === 'served' ? engine?.model ?? 'the served model' : `${addrModel || 'a model'} on ${addrPeer && peerName ? peerName : found?.engine ?? 'another engine'}`
  const served = engine?.startup?.context_tokens ?? engine?.context_length ?? null
  // the largest size the last sweep of this model read, to say how long a sweep takes here
  const lastBig = runs.find((r) => r.mode === 'prefill' && r.model === engine?.model && r.summary?.sweep?.length)?.summary?.sweep?.filter((p) => p.ttft != null && !p.error).slice(-1)[0]
  const limit = served && upTo > served ? 0 : upTo          // a size above the window means the window
  // `as` names what to run when the caller does not want the form's current choices (the plain check: always the
  // quick run on the served engine, whatever "More checks" was last set to; a state set in the same click is not seen here)
  const start = async (as: { mode?: BenchMode; aim?: 'served' | 'address'; exactness?: boolean } = {}) => {
    if (starting) return
    const m = as.mode ?? mode, a = as.aim ?? aim, ex = as.exactness ?? exactness
    setErr(null)
    setStarting(true)
    try {
      const fixed = m === 'kl' ? klText(klForm) : undefined
      const r = await api.benchStart({ mode: m, suites: m === 'profile' || m === 'load' || (m === 'kl' && fixed === undefined) ? chosen : m === 'quality' && cats.length ? cats : undefined, exactness: ex, max_context: m === 'prefill' && limit ? limit : undefined, max_concurrency: m === 'load' ? atOnce : undefined, full: m === 'tools' ? fullTools : undefined,
        target: a !== 'address' ? undefined : addrPeer ? { peer: addrPeer, model: addrModel } : { url: addr, key: addrKey || undefined, model: addrModel, label: addrLabel || undefined },
        ...(m === 'kl' ? { reference: klRef && klRef !== 'unchecked' ? klRef : undefined, kl_load: klLoad, kl_text: fixed } : {}) })
      setSelected(r.id); setBusyId(r.id); refresh()
    } catch (e) { setErr(String(e)) } finally { setStarting(false) }
  }
  const cancel = async () => { if (busyId) try { await api.benchCancel(busyId) } catch (e) { toast(String(e), 'bad') } }
  const remove = async (id: string) => {
    try { await api.benchDelete(id); setSelected(null); setRun(null); await refresh(); toast('Run deleted') } catch (e) { toast(String(e), 'bad') }
  }
  const going = run?.status === 'running' ? run : null

  return (
    <section>
      <div className="head"><div><h1>Bench</h1><p>How fast and how reliable is your AI on this machine? Measured the same way every time, so runs compare. Results stay here; the receipt is yours to share.</p></div></div>
      {err && <div className="notice bad">{err}</div>}
      {(simple ? !servedReady : !ready) && <div className="notice info">Nothing is running yet. Start an AI from <a href="#home">Home</a>, then measure it here.</div>}

      {simple ? <div className="card bench-new">
        <b>Check {engine?.model_name ?? engine?.model ?? 'the AI'}</b>
        <p className="small muted">Three short tasks (English, code, French), each written twice: once with the speed-up, once without, to check the answers are the same. About a minute.</p>
        <div className="actions">
          <button className="btn primary" aria-busy={starting} disabled={!servedReady || stale || !!busyId || starting} onClick={() => { setChoice('quick'); setAim('served'); setExactness(true); start({ mode: 'quick', aim: 'served', exactness: true }) }}>{starting ? 'Starting…' : 'Run the check'}</button>
          <button className="btn quiet" onClick={() => showMore(true)}>More checks</button>
          {busyId && !going && <span className="small muted">A run is going; open it below to follow or stop it.</span>}
        </div>
        {going && <LiveRun run={going} onStop={cancel} />}
      </div> : <div className="card bench-new">
        <div className="membar-head"><b>Kinds of run</b><button className="btn sm quiet" onClick={() => showMore(false)}>Fewer checks</button></div>
        <div className="run-tiles" role="group" aria-label="Kind of run">
          {(['quick', 'profile', 'quality', 'tools', 'prefill', 'load', 'kl', 'series'] as const).map((m) => (
            <button key={m} type="button" className="run-tile" aria-pressed={choice === m} onClick={() => setChoice(m)}>
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d={RUN[m].icon} /></svg>
              <div><b>{m === 'series' ? 'Compare configurations' : KIND[m]}</b><span>{RUN[m].what}</span></div>
            </button>
          ))}
        </div>
        {choice === 'series' ? <Series engine={engine} toast={toast} stale={stale} exactness={exactness} onOpen={(id) => { setSelected(id); setPair(null) }} onCompare={(a, b) => setPair([a, b])} /> : <>
        <div className="aim">
          <label className="small"><input type="radio" name="aim" checked={aim === 'served'} onChange={() => setAim('served')} /> The engine this TensorFold Web serves{engine?.model ? ` (${engine.model})` : ''}</label>
          <label className="small"><input type="radio" name="aim" checked={aim === 'address'} onChange={() => setAim('address')} /> Another engine at an address, to compare with</label>
          {aim === 'address' && <div className="aim-row">
            <PeerPick peers={peers} value={addrPeer} onChange={pickPeer} label="A Fleet machine" />
            {!addrPeer && <>
              <input className="input" value={addr} onChange={(e) => { setAddr(e.target.value); setFound(null) }} aria-label="Engine address" placeholder="http://127.0.0.1:11434" spellCheck={false} />
              <input className="input" type="password" value={addrKey} onChange={(e) => setAddrKey(e.target.value)} aria-label="Its API key" placeholder="API key, if it wants one" autoComplete="off" />
            </>}
            <button type="button" className="btn sm" onClick={() => check()}>Check</button>
            {found && (found.error ? <span className="small bad">{found.error}</span>
              : <span className="small">{found.engine}{found.version ? ` ${found.version}` : ''}: <select className="sel" value={addrModel} onChange={(e) => setAddrModel(e.target.value)} aria-label="Model">{found.models.map((m) => <option key={m}>{m}</option>)}</select>
                {found.engine === 'OpenAI-compatible' && <input className="input" style={{ minWidth: 140, marginLeft: 8 }} value={addrLabel} onChange={(e) => setAddrLabel(e.target.value)} placeholder="its name, e.g. mlx-lm 0.32" aria-label="Engine name" />}</span>)}
          </div>}
          {aim === 'address' && addrPeer && <p className="small muted">This TensorFold Web reaches that machine's engine through its TensorFold Web, with the token kept in Fleet: no key is typed or shown here.</p>}
          {aim === 'address' && <p className="small muted">Same protocol as for TensorFold. Its replies are compared by text with a TensorFold run; the drafts-off check is TensorFold's own. Ollama cuts a prompt longer than its context: set <code>OLLAMA_CONTEXT_LENGTH</code> on its server before a long sweep.</p>}
        </div>
        <p className="small muted">{info?.modes[mode]}{(mode === 'quick' || mode === 'profile') && exactness ? ', twice as long with the exactness check' : ''}.</p>
        {mode === 'kl' && <KlOptions load={klLoad} setLoad={setKlLoad} state={klForm} setState={setKlForm} />}
        {mode === 'load' && <label className="small">Up to <select className="sel" value={atOnce} onChange={(e) => setAtOnce(Number(e.target.value))} aria-label="Most requests at once">{[4, 8, 16, 32].map((n) => <option key={n} value={n}>{n} at once</option>)}</select> <span className="muted">The engine serves its own number at once; the others wait their turn, which shows as a longer first token.</span></label>}
        {mode === 'prefill' && info && (
          <label className="small">Up to <select className="sel" value={limit} onChange={(e) => setUpTo(Number(e.target.value))} aria-label="Largest prompt">
            {info.sizes.filter((n) => n >= 4096 && (!served || n <= served)).map((n) => <option key={n} value={n}>{num(n)} tokens</option>)}
            <option value={0}>the served window{served ? ` (${num(served)})` : ''}</option>
          </select> <span className="muted">Large sizes take minutes{lastBig ? `: ${num(lastBig.size)} tokens took ${secs(lastBig.ttft)} to read the last time here` : ''}.</span></label>
        )}
        {(mode === 'profile' || mode === 'load' || (mode === 'kl' && !klFixed)) && info && (
          <div className="filters" role="group" aria-label="Suites">
            {[...info.suites, ...info.custom.map((c) => ({ ...c, mine: true }))].map((s) => <button key={s.id} type="button" className={'mine' in s ? 'mine' : undefined} title={'mine' in s ? 'One of your suites' : undefined} aria-pressed={chosen.includes(s.id)} onClick={() => setChosen((c) => c.includes(s.id) ? c.filter((x) => x !== s.id) : [...c, s.id])}>{s.name}{'mine' in s && <small>yours</small>}</button>)}
          </div>
        )}
        {mode === 'tools' && info && (info.tools.installed
          ? <label className="small">Scenarios <select className="sel" value={fullTools ? 'full' : 'short'} onChange={(e) => setFullTools(e.target.value === 'full')} aria-label="Scenarios"><option value="short">the core 15, a few minutes</option><option value="full">the standard 69</option></select> <span className="muted">Run by tool-eval-bench (MIT), thinking off, temperature 0, seed 42.</span></label>
          : <ToolEvalInstall toast={toast} onDone={loadSuites} />)}
        {mode === 'quality' && info && <>
          <div className="filters" role="group" aria-label="Categories">
            {info.quality.categories.map((c) => <button key={c.id} type="button" aria-pressed={!cats.length || cats.includes(c.id)} onClick={() => setCats((cs) => { const now = cs.length ? cs : info.quality.categories.map((x) => x.id); return now.includes(c.id) ? now.filter((x) => x !== c.id) : [...now, c.id] })}>{c.name} <small>{c.items}</small></button>)}
          </div>
          <p className="small muted">{info.quality.about}</p>
        </>}
        {(mode === 'quick' || mode === 'profile') && <label className="toggle small"><input type="checkbox" checked={exactness} onChange={(e) => setExactness(e.target.checked)} /> Check exactness: write each reply again with the speed-up off and compare the tokens</label>}
        <div className="actions">
          <button className="btn primary" aria-busy={starting} disabled={!ready || stale || !!busyId || starting || ((mode === 'profile' || mode === 'load' || (mode === 'kl' && !klFixed)) && !chosen.length) || (mode === 'tools' && !info?.tools.installed) || (mode === 'kl' && klRef === 'unchecked') || (klFixed && !klForm.text.trim())} onClick={() => start()}>{starting ? 'Starting…' : `Run on ${targetName}`}</button>
          {busyId && !going && <span className="small muted">A run is going; open it below to follow or stop it.</span>}
        </div>
        {going && <LiveRun run={going} onStop={cancel} />}
        {info && (mode === 'quick' || mode === 'profile') && <p className="small muted" style={{ marginBottom: 0 }}>{info.about}</p>}
        </>}
      </div>}

      {!simple && choice === 'load' && engine?.status === 'running' && <div className="card" style={{ marginTop: 16 }}>
        <div className="membar-head"><b>See it live</b><span className="small muted">nothing is kept: Load sweep above is the measure</span></div>
        <p className="small muted">The same moment, several requests to {engine?.model_name ?? engine?.model}: each reply streams in its own card with its speed, and the total shows what serving several people at once gives.</p>
        <ParallelPanel model={engine?.model ?? ''} engine={engine} sampling={{ thinking: 'off', system: '', temperature: null }} />
      </div>}

      {pair ? <Compare a={pair[0]} b={pair[1]} onClose={() => setPair(null)} />
        : run && (run.status !== 'running' || run.summary) && <Report run={run} names={names} onDelete={() => remove(run.id)} toast={toast} />}

      {info && <CustomSuites suites={info.custom} builtIn={info.suites} toast={toast} onChange={loadSuites} />}

      {runs.length > 0 && (
        <div className="card">
          <div className="compare-bar"><b>Runs on this machine</b>
            <span className="small muted">{ticked.length < 2 ? `Tick two runs to compare them${ticked.length ? ' (one ticked)' : ''}.` : ''}</span>
            {ticked.length === 2 && <button className="btn sm primary" onClick={() => setPair([ticked[0], ticked[1]])}>Compare the two runs</button>}
          </div>
          <div className="tablewrap"><table className="tbl pick">
            <thead><tr><th><span className="sr-only">Compare</span></th><th>When</th><th>Kind</th><th>Model</th><th>Engine</th><th>tok/s</th><th>gain</th><th>exact</th><th>State</th></tr></thead>
            <tbody>{runs.map((r) => (
              <tr key={r.id} aria-selected={r.id === selected} onClick={() => { setSelected(r.id); setPair(null) }} tabIndex={0} onKeyDown={(e) => { if (e.key === 'Enter') { setSelected(r.id); setPair(null) } }}>
                <td onClick={(e) => e.stopPropagation()}><input type="checkbox" aria-label="Choose for a comparison" disabled={r.status === 'running'} checked={ticked.includes(r.id)}
                  onChange={(e) => setTicked((t) => e.target.checked ? [...t.filter((x) => x !== r.id), r.id].slice(-2) : t.filter((x) => x !== r.id))} /></td>
                <td>{when(r.created)}</td><td>{KIND[r.mode].toLowerCase()}</td><td>{r.model}</td><td>{r.engine ?? '—'}</td>
                <td>{r.mode === 'tools' ? (r.summary?.quality ? `${r.summary.quality.overall.passed}/${r.summary.quality.overall.items} passed` : '—') : r.mode === 'quality' ? (r.summary?.quality ? `${r.summary.quality.overall.passed}/${r.summary.quality.overall.items} right` : '—') : r.mode === 'kl' ? (r.kl?.mean_kl != null ? `KL ${r.kl.mean_kl.toFixed(4)}` : '—') : r.mode === 'load' ? (r.summary?.load?.some((l) => l.total_tps != null) ? `${tps(Math.max(...r.summary.load.filter((l) => l.total_tps != null).map((l) => l.total_tps!)))} total` : '—') : r.mode === 'prefill' ? (r.summary?.sweep?.length ? `${num(Math.round(r.summary.sweep.filter((p) => p.prefill_tps).slice(-1)[0]?.prefill_tps ?? 0))} prefill` : '—') : tps(r.summary?.overall.tps)}</td><td>{pct(r.summary?.overall.gain)}</td>
                <td>{r.summary?.overall.checked ? `${r.summary.overall.exact}/${r.summary.overall.checked}` : '—'}</td><td>{(STATUS[r.status] ?? r.status).toLowerCase()}</td>
              </tr>
            ))}</tbody>
          </table></div>
        </div>
      )}
    </section>
  )
}

/** tool-eval-bench missing: what it is, and a button that installs it for this user with uv (the command shown too). */
function ToolEvalInstall({ toast, onDone }: { toast: Ctx['toast']; onDone: () => unknown }) {
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const install = async () => {
    setBusy(true); setErr(null)
    try { const r = await api.toolEvalInstall(); toast(`tool-eval-bench ${r.version ?? ''} installed`); onDone() }
    catch (e) { setErr(String((e as Error).message ?? e)) } finally { setBusy(false) }
  }
  return (
    <div className="notice info"><div>
      Tool calls are measured by <a href="https://github.com/SeraphimSerapis/tool-eval-bench" target="_blank" rel="noreferrer">tool-eval-bench</a>, a free tool (MIT) that plays scripted conversations with mock tools and scores each one. It is not installed on this machine yet.
      <div className="actions" style={{ marginTop: 8 }}>
        <button className="btn sm primary" aria-busy={busy} disabled={busy} onClick={install}>{busy ? 'Installing… (up to five minutes)' : 'Install it'}</button>
        <span className="small muted">or in a terminal: <code>uv tool install tool-eval-bench==2.8.0</code> (the version TensorFold Web reads)</span>
      </div>
      {err && <p className="small bad" style={{ marginBottom: 0 }}>{err}</p>}
    </div></div>
  )
}
