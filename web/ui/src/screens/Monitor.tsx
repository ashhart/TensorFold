import { useEffect, useRef, useState } from 'react'
import { api, gb, num, type EngineInfo, type History } from '../api'
import { copyText } from '../shareCard'
import type { Ctx } from '../ctx'
import { clock, secs } from '../units'
import MonitorUsage from './MonitorUsage'

const WINDOWS = [{ s: 1800, label: '30 min' }, { s: 3600, label: '1 h' }, { s: 8 * 3600, label: '8 h' }]

/** Decode speed over the window, from the server's history: a reading missing for long is drawn as a gap. */
function HistoryChart({ h, window }: { h: History; window: number }) {
  const W = 400, H = 64
  const end = h.from + window
  const pts = h.t.map((t, i) => [t, h.decode_tps[i]] as const)
  const max = Math.max(1, ...pts.map(([, v]) => v ?? 0))
  const x = (t: number) => ((t - h.from) / window) * W
  const y = (v: number) => H - 4 - (v / max) * (H - 12)
  // runs of readings, broken where the engine was down or TensorFold Web did not read it
  const runs: (readonly [number, number])[][] = []
  let cur: (readonly [number, number])[] = []
  let last = -Infinity
  for (const [t, v] of pts) {
    if (v == null || t - last > 3 * h.every) { if (cur.length) runs.push(cur); cur = [] }
    if (v != null) cur.push([t, v])
    last = t
  }
  if (cur.length) runs.push(cur)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img" aria-label="Writing speed over time">
      {runs.map((r, i) => {
        const line = r.map(([t, v]) => `${x(t).toFixed(1)},${y(v).toFixed(1)}`).join(' L')
        return <g key={i}>
          <path d={`M${x(r[0][0]).toFixed(1)},${H} L${line} L${x(r[r.length - 1][0]).toFixed(1)},${H} Z`} fill="var(--accent-soft)" />
          <path d={`M${line}`} fill="none" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
        </g>
      })}
      <line x1={x(end)} x2={x(end)} y1="0" y2={H} stroke="var(--line)" />
    </svg>
  )
}

function Spark({ values }: { values: number[] }) {
  if (values.length < 2) return <svg viewBox="0 0 400 64" preserveAspectRatio="none" aria-hidden="true" />
  const max = Math.max(1, ...values)
  const pts = values.map((v, i) => [(i / (values.length - 1)) * 400, 60 - (v / max) * 52] as const)
  const line = pts.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' L')
  return (
    <svg viewBox="0 0 400 64" preserveAspectRatio="none" aria-hidden="true">
      <path d={`M0,64 L${line} L400,64 Z`} fill="var(--accent-soft)" />
      <path d={`M${line}`} fill="none" stroke="var(--accent)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}
/** This machine's memory by holder; the headroom, in GB, is what turns amber then red, not the bar. */
function MemoryBar({ m }: { m: NonNullable<EngineInfo['machine_memory']> }) {
  const engine = Math.min(m.engine ?? 0, m.total - m.available)
  const other = Math.max(0, m.total - m.available - engine)
  const pct = (x: number) => `${(x / m.total) * 100}%`
  return (
    <div className="card membar">
      <div className="membar-head"><b>Memory on this machine</b>
        <span className={`headroom ${m.headroom}`}>{gb(m.available, 1)} free{m.headroom === 'low' ? ': the engine or your apps may be stopped to make room' : m.headroom === 'tight' ? ': tight' : ''}</span></div>
      <div className="bar3" role="img" aria-label={`Engine ${gb(engine, 1)}, other ${gb(other, 1)}, free ${gb(m.available, 1)} of ${gb(m.total)}`}>
        {m.engine != null && <i className="eng" style={{ width: pct(engine) }} />}<i className="oth" style={{ width: pct(m.engine == null ? m.total - m.available : other) }} />
      </div>
      <div className="legend small"><span><i className="eng" /> engine {m.engine == null ? 'not reported' : gb(engine, 1)}</span><span><i className="oth" /> {m.engine == null ? 'in use' : 'everything else'} {gb(m.engine == null ? m.total - m.available : other, 1)}</span><span><i /> free {gb(m.available, 1)} of {gb(m.total)}</span></div>
    </div>
  )
}

const uptime = (s: number | null) => (s == null ? '' : s < 60 ? 'under a minute' : clock(s).replace(/ \d+ s$/, ''))   // 12 min, 3 h 05 min, 2 d 4 h
// TensorFold Web's own polling, which the engine logs on every request
const POLL = /"GET \/(health|v1\/models|metrics)(\?[^"]*)? HTTP/

export default function Monitor({ engine, settings, refreshEngine, toast, stale, decodeHistory }: Ctx) {
  const [win, setWin] = useState(1800)
  const [hist, setHist] = useState<History | null>(null)
  useEffect(() => {
    let live = true
    const load = () => api.history(win).then((h) => { if (live) setHist(h) }).catch(() => {})
    load()
    const t = setInterval(load, 5000)
    return () => { live = false; clearInterval(t) }
  }, [win])
  // the server's history when it has some, else this page's own readings
  const kept = hist ? hist.decode_tps.filter((v): v is number => v != null) : []
  const decode = kept.length >= 2 ? kept : decodeHistory
  const [lines, setLines] = useState<string[]>([])
  const [hidePolling, setHidePolling] = useState(true)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [streamState, setStreamState] = useState<'connecting' | 'live' | 'reconnecting'>('connecting')
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (engine?.managed === false) { setLines(['The engine runs on another machine; its log is there.']); return }
    // The browser retries a dropped stream by itself, but gives up for good on a refused one (a restarted TensorFold Web
    // asking for its token, a proxy error): then this opens a new one, waiting longer each time, up to 30 s.
    let es: EventSource | null = null
    let retry: ReturnType<typeof setTimeout> | undefined
    let failures = 0
    let live = true
    const open = () => {
      // the stream's own cookie first (an EventSource sends no header), then the stream, with nothing in its address
      api.logsSession().catch(() => {}).then(() => {
        if (!live) return
        es = new EventSource('/api/engine/logs/stream')
      // each connection starts with the server's last 50 lines, so they replace what is shown instead of repeating it
      es.onopen = () => { failures = 0; setStreamState('live'); setLines([]) }
      es.onmessage = (e) => setLines((l) => [...l.slice(-999), JSON.parse(e.data)])
      es.onerror = () => {
        setStreamState('reconnecting')
        if (es?.readyState === EventSource.CLOSED && live) {
          failures += 1
          retry = setTimeout(open, Math.min(30000, 1000 * 2 ** Math.min(5, failures)))
        }
      }
      })
    }
    open()
    return () => { live = false; clearTimeout(retry); es?.close() }
  }, [engine?.managed])
  const shown = hidePolling ? lines.filter((l) => !POLL.test(l)) : lines
  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight }, [shown.length])

  const restart = async () => {
    const model = engine?.model_repo ?? settings?.model          // the engine's own checkpoint first, never a stale setting
    if (!model) { setErr('No model is recorded to restart.'); return }
    setBusy(true); setErr(null)
    try { await api.stop(); await api.start({ model }); await refreshEngine(); toast('Engine restarting') } catch (e) { setErr(String(e)) } finally { setBusy(false) }
  }
  const stop = async () => { setBusy(true); setErr(null); try { await api.stop(); await refreshEngine(); toast('Engine stopped') } catch (e) { setErr(String(e)) } finally { setBusy(false) } }
  const copyLog = async () => { try { await copyText(shown.join('\n')); toast('Log copied') } catch { toast('The browser refused the clipboard', 'bad') } }

  const st = engine?.status ?? 'unreachable'
  const mem = engine?.memory ?? null
  const su = engine?.startup ?? {}
  const name = engine?.model_name ?? engine?.model
  const head = st === 'running' ? <><b>{name}</b> has been running{engine?.uptime_s != null && <> for {uptime(engine.uptime_s)}</>}{su.loaded_s ? <>, ready in {secs(su.loaded_s)} after its start</> : null}.</>
    : st === 'starting' ? <>Starting <b>{name ?? '…'}</b>. The first start takes longer; the log below says what it is doing.</>
    : st === 'unauthorized' ? <>A TensorFold answers at <code>{engine?.engine_url}</code> but refuses the API key this TensorFold Web holds ({engine?.error}). Enter the engine's key in <a href="#settings">Settings</a>{engine?.key_pending_restart ? ', or restart it to take the new key' : ''}.</>
    : st === 'failed' ? <>The AI stopped with a problem ({engine?.error}). The log below has the reason.</>
    : engine?.managed === false ? <>No TensorFold answers at <code>{engine?.engine_url}</code>.</> : <>Nothing is running. Start an AI from <a href="#home">Home</a> or <a href="#models">Models</a>.</>
  const peak = Math.round(Math.max(0, ...decode))
  // the average while writing: idle seconds would drag it to zero
  const writing = decode.filter((v) => v > 0)
  const busyAvg = writing.length ? Math.round(writing.reduce((a, b) => a + b, 0) / writing.length) : null
  const running = st === 'running'
  const notReported = <small className="muted">not reported by this engine</small>
  const streams = engine?.streams
  // behind a key, figures come from totals the engine adds to when a request ends: 0 then means "not finished yet"
  const pending = engine?.source === 'metrics' && (engine?.connections ?? 0) > 0 && !engine?.decode_tps
  return (
    <section>
      <div className="head">
        <div><h1>Monitor</h1><p>{head}</p></div>
        {engine?.owned && <div className="actions"><button className="btn" aria-busy={busy} disabled={busy || stale} onClick={restart}>{busy ? 'Working…' : 'Restart'}</button><button className="btn" aria-busy={busy} disabled={busy || stale} onClick={stop}>Stop</button></div>}
      </div>
      {err && <div className="notice bad">{err}</div>}
      {(engine?.findings ?? []).length > 0 && <ul className="findings-list" aria-label="Health">
        {engine!.findings!.map((f) => <li key={f.id} className={f.level}><b>{f.title}</b> <span className="ev">{f.evidence}</span><span className="hint">{f.hint}</span></li>)}
      </ul>}
      {engine?.managed && !engine.owned && st === 'running' && <div className="notice">A TensorFold answers at {engine.engine_url} but this TensorFold Web did not start it, so it cannot stop it. Stop it where it was started, or switch Settings to a remote engine.</div>}

      <div className="live">
        <div className="main">
          <div className="l">Writing replies</div>
          <div className="v">{pending ? '…' : engine?.decode_tps == null ? '—' : Math.round(engine.decode_tps)}<span>tokens a second{engine?.decode_tps ? `, about ${Math.round(engine.decode_tps * 0.75)} words` : ''}</span></div>
          <div className="s">{pending ? 'a request is running; this engine gives its speed when it ends' : engine?.decode_tps == null ? (st === 'running' ? 'no figure from this engine yet' : 'live while a reply is written') : engine?.source === 'metrics' ? `finished replies between two polls; peak ${peak} in this window` : `right now; peak ${peak} in this window`}{busyAvg != null && writing.length > 2 ? `, ${busyAvg} on average while writing` : ''}</div>
          {hist && hist.t.length >= 2 ? <HistoryChart h={hist} window={win} /> : <Spark values={decode} />}
          <div className="winsel" role="group" aria-label="Time window">
            {WINDOWS.map((w) => <button key={w.s} type="button" aria-pressed={win === w.s} className={win === w.s ? 'active' : ''} onClick={() => setWin(w.s)}>{w.label}</button>)}
            <span className="muted small">{hist && hist.t.length >= 2 ? 'kept by TensorFold Web, readings every 5 s' : 'collecting history'}</span>
          </div>
        </div>
        <dl className="side">
          <dt>Memory</dt><dd>{mem?.active != null ? gb(mem.active, 1) : mem?.footprint != null ? gb(mem.footprint, 1) : '—'}{mem?.peak != null ? <small>peak {gb(mem.peak, 1)}</small> : mem?.footprint != null ? <small>process footprint</small> : null}</dd>
          <dt>Busy with</dt><dd>{engine?.connections ?? 0}{streams ? <small>{streams.decoding} writing, {streams.prefilling} reading{streams.paused ? `, ${streams.paused} set aside` : ''}</small> : <small>{(engine?.connections ?? 0) === 1 ? 'conversation' : 'conversations'} right now</small>}{engine?.waiting ? <small>{engine.waiting} waiting for a turn</small> : null}</dd>
          {(su.prompt_cache_gib != null || engine?.prompt_cache) && <><dt>Prompt cache</dt><dd className={su.prompt_cache_gib === 0 ? 'warn' : ''}>{su.prompt_cache_gib != null ? `${su.prompt_cache_gib} GiB` : 'on'}{engine?.prompt_cache?.prompt ? <small>{num(engine.prompt_cache.restored)} of {num(engine.prompt_cache.prompt)} prompt tokens restored ({Math.round(100 * engine.prompt_cache.restored / engine.prompt_cache.prompt)}%): a later turn reads only its new tokens</small> : <small>later turns of a conversation read only their new tokens</small>}</dd></>}
          <dt>Drafts kept</dt><dd>{engine?.draft_acceptance != null ? `${Math.round(engine.draft_acceptance * 100)}%` : '—'}{engine?.draft_acceptance == null && running ? notReported : <small>the speed-up's guesses that were right, since the engine started; a rate, not a speed-up</small>}</dd>
          <dt>Conversations</dt><dd>{su.context_tokens ? `up to about ${num(Math.round(su.context_tokens / 400))} pages` : '—'}{su.context_tokens ? <small>{num(su.context_tokens)} tokens at most, as the engine said at start</small> : null}</dd>
        </dl>
      </div>
      <details className="more" style={{ marginTop: -6, marginBottom: 16 }}>
        <summary>The engine's own figures</summary>
        <dl className="side" style={{ borderLeft: 0, padding: '4px 0 0' }}>
          <dt>Reading prompts</dt><dd>{pending && !engine?.prefill_tps && !streams ? <>…<small>given when the request ends</small></> : streams?.prefilling ? <>reading now<small>{streams.prefilling} {streams.prefilling === 1 ? 'request reads its prompt' : 'requests read their prompts'}; the speed shows when it ends</small></> : engine?.prefill_tps == null ? '—' : `${num(Math.round(engine.prefill_tps))} tok/s`}{!streams?.prefilling && engine?.prefill_tps ? <small>a 16k prompt in about {Math.round(16384 / engine.prefill_tps)} s</small> : null}</dd>
          <dt>KV cache</dt><dd>{engine?.kv_fill != null ? `${Math.round(engine.kv_fill * 100)}% held` : '—'}{engine?.kv_fill == null && running ? notReported : engine?.kept_prompts != null ? <small>{engine.kept_prompts} prompts kept for reuse</small> : null}{engine?.preemptions ? <small className="bad">{num(engine.preemptions)} requests set aside to free memory</small> : null}</dd>
          <dt>Context admitted</dt><dd>{su.context_tokens ? `${num(su.context_tokens)} tokens` : '—'}{su.resident_gib ? <small>{su.resident_gib} GiB of weights resident</small> : null}{su.streams ? <small>{su.streams} requests at once, {su.left_gib} GiB left for them</small> : null}</dd>
        </dl>
        <div className="two" style={{ marginTop: 12 }}>
          <div className="card">
            <b>Exactness</b>
            <p className="small muted" style={{ margin: '6px 0 0' }}>TensorFold drafts several tokens and verifies them in one pass; it keeps a draft only when it equals the token one-token decoding would have produced. The <a href="#bench">Bench</a> checks it: it sends each prompt again with <code>"draft": false</code> and compares the tokens.</p>
          </div>
          <div className="card">
            <b>Where it listens</b>
            <dl className="kv" style={{ marginBottom: 0 }}><dt>Engine</dt><dd><code>{engine?.engine_url ?? '…'}</code></dd><dt>Managed by TensorFold Web</dt><dd>{engine?.managed ? (engine.owned ? 'yes, started here' : 'yes, but not started by this TensorFold Web') : 'no, remote'}</dd>{engine?.engine_kind && <><dt>Engine</dt><dd>{engine.engine_kind === 'native' ? 'native (TensorFold 1.0)' : 'Python line (0.6.6, frozen)'}</dd></>}{engine?.command && <><dt>Command</dt><dd><code>{engine.command.join(' ')}</code></dd></>}</dl>
          </div>
        </div>
      </details>

      {engine?.machine_memory && <MemoryBar m={engine.machine_memory} />}

      <MonitorUsage engine={engine} />

      <div className="logbox">
        <div className="bar"><b>Engine log</b>{engine?.managed !== false && <span className="state">{streamState}</span>}
          <label><input type="checkbox" checked={hidePolling} onChange={(e) => setHidePolling(e.target.checked)} /> Hide TensorFold Web's own polling</label>
          <button className="btn sm quiet" onClick={copyLog}>Copy</button></div>
        <div className="log" ref={logRef}>{shown.length ? shown.join('\n') : lines.length ? 'Only polling so far. Untick the box to see it.' : 'Nothing yet.'}</div>
      </div>

    </section>
  )
}
