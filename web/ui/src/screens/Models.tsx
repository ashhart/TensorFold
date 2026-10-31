import { useCallback, useEffect, useState } from 'react'
import { api, gb, num, shortName, type Model, type PullStatus } from '../api'
import type { Ctx } from '../ctx'
import { showSub, useSub } from '../route'
import { useWorking } from '../working'
import { ModelLogo } from '../brandLogo'

const FILTERS: { id: string; label: string; test: (m: Model) => boolean }[] = [
  { id: 'all', label: 'All', test: () => true },
  { id: 'fits', label: 'Fits here', test: (m) => m.verdict?.level === 'fits' },
  { id: 'disk', label: 'On disk', test: (m) => m.installed },
  { id: 'chat', label: 'Chat', test: (m) => m.tags.includes('chat') },
  { id: 'code', label: 'Code', test: (m) => m.tags.includes('code') },
  { id: 'vision', label: 'Reads images', test: (m) => m.tags.includes('vision') },
  { id: 'fast', label: 'Fastest', test: (m) => m.speed_hint === 'fast' || m.speed_hint === 'fastest' },
]
const pages = (tokens: number) => Math.round(tokens / 400)

export default function Models({ machine, engine, settings, refreshEngine, refreshSettings, toast, stale }: Ctx) {
  const [models, setModels] = useState<Model[] | null>(null)
  const [sel, setSel] = useState<string | null>(null)
  const [filter, setFilter] = useState('all')
  const [pulls, setPulls] = useState<Record<string, PullStatus>>({})
  const [notice, setNotice] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const { run, is } = useWorking()

  useSub('models', setSel)                                  // #models/<owner>/<name>
  useEffect(() => { if (sel) showSub('models', sel) }, [sel])
  const load = useCallback(() => api.catalog().then((c) => setModels(c.models)).catch((e) => setNotice(String(e))), [])
  useEffect(() => { load() }, [load, settings?.budget_fraction])
  useEffect(() => {              // downloads live on the server: pick them up again after a visit elsewhere
    api.pullList().then((r) => setPulls(Object.fromEntries(r.jobs.map((j) => [j.repo, j])))).catch(() => {})
  }, [])
  const refresh = () => run('refresh', async () => {
    setNotice(null)
    try { const c = await api.catalog(true); setModels(c.models); toast('Sizes refreshed from Hugging Face') } catch (e) { setNotice(`Sizes not refreshed: ${e}`) }
  })
  useEffect(() => {
    if (!models) return
    const running = models.find((m) => engine?.model && shortName(m.id).startsWith(engine.model))
    if (!sel && models.length) setSel(running?.id ?? models.find((m) => m.verdict?.level === 'fits')?.id ?? models[0].id)
  }, [models, engine?.model, sel])

  // follow downloads
  useEffect(() => {
    const active = Object.values(pulls).filter((p) => p.state === 'running')
    if (!active.length) return
    const t = setInterval(async () => {
      for (const p of active) {
        const s = await api.pullStatus(p.repo).catch(() => null)
        if (s) setPulls((old) => ({ ...old, [p.repo]: s }))
        if (s && s.state !== 'running') { load(); if (s.state === 'done') toast(`${shortName(p.repo)} downloaded`) }
      }
    }, 1000)
    return () => clearInterval(t)
  }, [pulls, load, toast])

  const current = models?.find((m) => m.id === sel) ?? null
  const isRunning = (m: Model) => !!engine?.model && engine.status !== 'stopped' && shortName(m.id).startsWith(engine.model)
  const pull = (m: Model) => run(`pull:${m.id}`, async () => {
    setNotice(null)
    for (const repo of [m.id, ...(m.drafter ? [m.drafter] : [])]) {
      const s = await api.pull(repo).catch((e) => { setNotice(String(e)); return null })
      if (s) setPulls((old) => ({ ...old, [repo]: s }))
    }
    toast(`Downloading ${m.name}: the bar in its row follows it`)
  })
  // stops a download; what reached the disk stays, and the next Download resumes it
  const stopPull = (m: Model) => run(`stop:${m.id}`, async () => {
    for (const repo of [m.id, ...(m.drafter ? [m.drafter] : [])]) {
      if (pulls[repo]?.state !== 'running') continue
      const s = await api.pullCancel(repo).catch((e) => { setNotice(String(e)); return null })
      if (s) setPulls((old) => ({ ...old, [repo]: s }))
    }
    toast(`Download of ${m.name} stopped; what came through stays, Download again resumes it`)
  })
  // one progress for a model and its drafter together
  const progressOf = (m: Model) => {
    const parts = [pulls[m.id], ...(m.drafter ? [pulls[m.drafter]] : [])].filter(Boolean) as PullStatus[]
    if (!parts.length) return null
    const done = parts.reduce((a, p) => a + p.done_bytes, 0), total = parts.reduce((a, p) => a + p.total_bytes, 0)
    const failed = parts.find((p) => p.state === 'failed')
    return { running: parts.some((p) => p.state === 'running'), failed, fraction: total ? Math.min(1, done / total) : 0 }
  }
  // one engine at a time: starting another model while one runs is asked first, in place
  const [conflict, setConflict] = useState<Model | null>(null)
  const start = async (m: Model, replace = false) => {
    const live = engine?.status === 'running' || engine?.status === 'starting'
    if (live && !engine?.owned) { setNotice(`A server already answers at ${engine?.engine_url} and this TensorFold Web did not start it. Stop it where it runs, then start ${m.name} here.`); return }
    if (live && !replace) { setConflict(m); return }
    setConflict(null); setBusy(true); setNotice(null)
    try {
      if (live) { await api.stop(); toast(`${engine?.model ?? 'The engine'} stopped`) }
      const fraction = m.verdict?.level === 'tight' ? m.verdict.budget_fraction : settings?.budget_fraction ?? null
      await api.start({ model: m.id, budget_fraction: fraction })
      await Promise.all([refreshEngine(), refreshSettings()])        // the chosen model and budget are now the settings too
      toast(`Starting ${m.name}`)
    } catch (e) { setNotice(String(e)) } finally { setBusy(false) }
  }
  const stop = async () => { setBusy(true); try { await api.stop(); await refreshEngine(); toast('Engine stopped') } catch (e) { setNotice(String(e)) } finally { setBusy(false) } }

  // the ribbon: the machine's memory, divided for the selected model
  const cap = machine?.budget.capacity_bytes ?? machine?.ram_bytes ?? 0
  const v = current?.verdict ?? null
  const weights = v?.weights_bytes ?? ((current?.size_bytes ?? 0) + (current?.drafter_bytes ?? 0))
  const engineBudget = v?.budget_bytes ?? (cap * (settings?.budget_fraction ?? machine?.budget.default_fraction ?? 0.7))
  const contextRoom = Math.max(0, v?.headroom_bytes ?? (engineBudget - weights - (machine?.budget.reserve_bytes ?? 0)))
  const working = Math.max(0, engineBudget - weights - contextRoom)
  const apps = Math.max(0, cap - engineBudget)
  const pct = (b: number) => (cap ? `${Math.max(0, Math.min(100, (b / cap) * 100))}%` : '0%')
  const wide = (b: number) => cap && b / cap > 0.11
  const unverified = v?.basis === 'cuda-unverified'
  const verdictLine = !current ? null : !current.runs_here ? <>{current.name} does not run on this machine: {current.runs_note}{v ? ` For memory alone, it ${v.level === 'fits' ? 'would fit' : v.level === 'tight' ? 'would fit, tightly' : 'would not fit'}.` : ''}</>
    : !v ? <>The size of {current.name} is not known yet.</>
    : v.level === 'no' ? <><strong>{current.name}</strong> does not fit: {v.reason}{v.needs_ram_gb ? ` A ${v.needs_ram_gb} GB machine would.` : ''}</>
    : <><strong>{current.name}</strong> {v.level === 'tight' ? 'fits, tightly' : 'fits'}: about <strong>{num(v.context_tokens)} tokens</strong> of context{unverified ? ' (a rough estimate on CUDA)' : v.basis === 'engine-floor' ? ' (from the engine\'s own memory floor)' : v.basis === 'native-measured' ? ' (the native engine sizes its own memory up to macOS\'s GPU limit)' : v.basis === 'native-unverified' ? ' (macOS\'s GPU limit could not be read: a guess at 85% of memory)' : ''}, {gb(v.left_for_apps_bytes)} left for your apps{v.level === 'tight' ? `. ${v.reason}` : '.'}</>

  const shown = models?.filter(FILTERS.find((f) => f.id === filter)!.test) ?? []
  return (
    <section>
      <div className="head">
        <div><h1>Models</h1><p>Pick the AI that runs on this machine. TensorFold Web says which ones fit, before anything downloads; the one marked <b>Recommended</b> is a good first choice.</p></div>
        <div className="actions"><button className="btn" aria-busy={is('refresh')} onClick={refresh}>{is('refresh') ? 'Asking Hugging Face…' : 'Refresh sizes'}</button></div>
      </div>
      {notice && <div className="notice">{notice}</div>}
      {conflict && <div className="notice conflict" role="alertdialog" aria-label="Another model is running">
        <span><b>{engine?.model}</b> is {engine?.status === 'starting' ? 'still loading' : 'running'}. The engine serves one model at a time: stop it, then start <b>{conflict.name}</b>?{engine?.connections ? ` ${engine.connections} ${engine.connections === 1 ? 'connection is' : 'connections are'} open and will be cut.` : ''}</span>
        <span className="actions"><button className="btn sm primary" disabled={busy} onClick={() => start(conflict, true)}>Stop it and start {conflict.name}</button><button className="btn sm quiet" onClick={() => setConflict(null)}>Keep it running</button></span>
      </div>}
      {machine && machine.backend === 'none' && <div className="notice">{machine.backend_note} Use Settings to connect to a TensorFold running elsewhere.</div>}
      {machine && machine.budget.basis === 'cuda-unverified' && <div className="notice info">Memory verdicts on CUDA machines use TensorFold's documented rule for a card's memory but have not been checked against a real run yet; treat them as rough.</div>}
      {settings?.remote && <div className="notice info">The engine is remote: this list and its verdicts describe <b>this</b> machine, and Start only works for an engine run here.</div>}

      <div className="ribbon" aria-label="Memory on this machine">
        <div className="top">
          <div><b>{machine?.chip ?? 'This machine'}</b> <span className="muted">{machine ? `${gb(machine.ram_bytes)} ${machine.unified_memory ? 'unified memory' : 'RAM'}, ${machine.os} ${machine.os_version}` : ''}</span></div>
          <div className="running">
            {engine?.status === 'running' || engine?.status === 'starting' ? (
              <><span className="chip run">{engine.status === 'running' ? 'Running' : 'Starting'}</span><ModelLogo names={[engine.model_name, engine.model]} /><b style={{ color: 'var(--ink)' }}>{engine.model_name ?? engine.model}</b>{engine.startup?.context_tokens ? <span className="num" title="The longest conversation it accepts">{num(engine.startup.context_tokens)} tokens, about {num(pages(engine.startup.context_tokens))} pages</span> : null}
                {engine.owned && <button className="btn sm" aria-busy={busy} disabled={busy || stale} onClick={stop}>Stop</button>}<a className="btn sm ghost" href="#chat">Open chat</a></>
            ) : <span>Nothing running. The AI may use up to <b style={{ color: 'var(--ink)' }}>{gb(engineBudget)}</b> of memory (<a href="#settings">change</a>).</span>}
          </div>
        </div>
        <div className="band" role="img" aria-label={`Model ${gb(weights)}, conversation memory ${gb(contextRoom)}, working room ${gb(working)}, left for your apps ${gb(apps)}`}>
          <i className="w" style={{ width: pct(weights) }}>{wide(weights) ? `The AI ${gb(weights)}` : ''}</i>
          <i className="c" style={{ width: pct(contextRoom) }}>{wide(contextRoom) ? `Conversations ${gb(contextRoom)}` : ''}</i>
          <i className="r" style={{ width: pct(working) }}>{wide(working) ? `Working room ${gb(working)}` : ''}</i>
          <i className="a">{wide(apps) ? `Your other apps ${gb(apps)}` : ''}</i>
        </div>
        <div className="legend">
          <span><i className="sw w" />The AI itself{current?.drafter ? ', with its speed-up helper' : ''}</span>
          <span><i className="sw c" />Room for long conversations</span>
          <span><i className="sw r" />Working room while it reads and writes</span>
          <span><i className="sw a" />Left for your other apps and {machine?.os ?? 'the system'}</span>
        </div>
        {verdictLine && <p className="verdict">{verdictLine}</p>}
      </div>

      <div className="filters" role="group" aria-label="Filter models">
        {FILTERS.map((f) => <button key={f.id} aria-pressed={filter === f.id} onClick={() => setFilter(f.id)}>{f.label}</button>)}
      </div>

      <div className="models">
        <div className="list" role="list" aria-label="Models">
          {!models && [0, 1, 2, 3].map((i) => <div key={i} className="mrow loading" aria-hidden="true"><div><div className="n skeleton">loading</div><div className="d skeleton">loading the catalog</div></div><div /><div /><div /></div>)}
          {models && !shown.length && <div className="mrow loading"><div className="muted">No model matches this filter.</div></div>}
          {shown.map((m) => {
            const p = progressOf(m)
            return (
              // a row is not a control itself: its name is the button that shows the model's details, and its action
              // (Download, Start) is a button of its own beside it
              <div key={m.id} role="listitem" data-selected={sel === m.id} className={`mrow ${m.runs_here ? '' : 'dim'}`} onClick={() => setSel(m.id)}>
                <div><button type="button" className="n rowpick" aria-current={sel === m.id ? 'true' : undefined} onClick={(e) => { e.stopPropagation(); setSel(m.id) }}><ModelLogo names={[m.name, m.id, m.family]} />{m.name}{m.recommended && <span className="chip fits" style={{ marginLeft: 8 }}>Recommended</span>}</button><div className="d">{m.vendor}. {m.description}</div>
                  {p?.running && <div className="progress"><i style={{ width: `${Math.round(p.fraction * 100)}%` }} /></div>}
                  {p?.failed && <div className="small" style={{ color: 'var(--bad)', marginTop: 4 }}>Download failed: {p.failed.error}</div>}</div>
                <div className="v">{gb(m.size_bytes)}<small>{m.installed ? 'on this machine' : 'to download'}</small></div>
                {/* three facts, each its own chip: does it fit in memory, can this machine's engine run it, is it on
                    disk or loaded; on a phone this column is the only one shown */}
                <div className="chipcol">
                  {m.verdict ? <span className={`chip ${m.verdict.level}`} title={m.verdict.reason}>{m.verdict.label}</span> : <span className="chip no">Size unknown</span>}
                  {!m.runs_here && <span className="chip no" title={m.runs_note ?? undefined}>Not on this engine</span>}
                  {isRunning(m) ? <span className="chip run">Loaded</span> : m.installed ? <span className="chip disk">On disk</span> : <span className="chip absent">Not downloaded</span>}
                </div>
                <div className="act" onClick={(e) => e.stopPropagation()}>
                  {isRunning(m) ? <span className="chip run">Running</span>
                    : p?.running ? <><span className="small muted num">{Math.round(p.fraction * 100)}%</span> <button className="btn sm quiet" aria-busy={is(`stop:${m.id}`)} onClick={() => stopPull(m)} aria-label={`Stop downloading ${m.name}`}>Stop</button></>
                    : p?.failed ? <button className="btn sm" disabled={!m.runs_here} onClick={() => pull(m)}>Retry</button>
                    : m.installed ? <button className="btn sm primary" aria-busy={busy && sel === m.id} disabled={busy || stale || !m.runs_here} onClick={() => { setSel(m.id); start(m) }}>{busy && sel === m.id ? 'Starting…' : 'Start'}</button>
                    : <button className="btn sm" aria-busy={is(`pull:${m.id}`)} disabled={!m.runs_here || m.verdict?.level === 'no'} onClick={() => pull(m)}>Download</button>}
                </div>
              </div>
            )
          })}
        </div>

        {current && (
          <div className="card panel">
            {!current.runs_here ? <span className="chip no">Not on this machine</span> : v ? <span className={`chip ${v.level}`}>{v.label}</span> : <span className="chip no">Size unknown</span>}
            <h2 className="modelhead"><ModelLogo names={[current.name, current.id, current.family]} size="lg" />{current.name}{current.recommended && <span className="chip fits" style={{ marginLeft: 8, verticalAlign: 'middle' }}>Recommended</span>}</h2>
            <p className="vendor">by {current.vendor}, {current.license}</p>
            <p className="small muted" style={{ margin: 0 }}>{current.description}</p>
            <dl className="kv">
              {v && v.level !== 'no' && <><dt>Conversations</dt><dd>up to about {num(pages(v.context_tokens))} pages long ({num(v.context_tokens)} tokens{unverified ? ', a rough estimate on CUDA' : ''})</dd></>}
              {isRunning(current) && engine?.startup?.context_tokens ? <><dt>Running now</dt><dd>with room for {num(pages(engine.startup.context_tokens))} pages ({num(engine.startup.context_tokens)} tokens), as the engine said at start</dd></> : null}
              {v && v.level !== 'no' && <><dt>Memory</dt><dd>it may use {gb(v.budget_bytes)}; {gb(v.left_for_apps_bytes)} stay for your other apps</dd></>}
              {v && v.level === 'tight' && <><dt>The trade</dt><dd>{v.reason}</dd></>}
              {v && v.level === 'no' && <><dt>Why not here</dt><dd>{v.reason}</dd></>}
              {v && v.level === 'no' && v.needs_ram_gb && <><dt>What it takes</dt><dd>a {v.needs_ram_gb} GB machine, or a remote engine</dd></>}
              {!current.runs_here && <><dt>Why not here</dt><dd>{current.runs_note}</dd></>}
              <dt>Good at</dt><dd>{current.tags.map((t) => <span key={t} className="tag">{t}</span>)}</dd>
              <dt>Size</dt><dd>{gb(current.size_bytes, 1)}{current.drafter ? ` plus a ${gb(current.drafter_bytes, 1)} speed-up helper` : ''}</dd>
            </dl>
            <div className="trust"><span>✓</span><span>The speed-up guesses the next words and keeps a guess only when it is exactly what the AI would have written: faster, same answers.</span></div>
            <details className="more">
              <summary>Details for the curious</summary>
              <dl className="kv">
                <dt>Checkpoint</dt><dd><code>{current.id}</code></dd>
                {current.drafter && <><dt>Drafter</dt><dd><code>{current.drafter}</code></dd></>}
                {v && !current.native && <><dt>Budget</dt><dd>{Math.round(v.budget_fraction * 100)}% of memory ({gb(v.budget_bytes)}); TensorFold's default is {Math.round((machine?.budget.default_fraction ?? 0.7) * 100)}%</dd></>}
                <dt>Runs on</dt><dd>{current.backends.map((b) => b === 'mlx' ? 'Apple silicon Macs' : b === 'cuda' ? 'NVIDIA GPUs' : b.toUpperCase()).join(' and ')}</dd>
                <dt>Engine</dt><dd>{current.native ? 'native (TensorFold 1.0); it sizes its memory itself' : 'Python line (0.6.6, frozen until this model is ported)'}</dd>
                {current.engine_info && <><dt>Engine reads</dt><dd>{[current.engine_info.family, current.engine_info.format].filter(Boolean).join(', ')}{current.engine_info.context ? `, up to ${num(current.engine_info.context)} tokens` : ''}</dd></>}
                {current.engine_info?.floor_bytes ? <><dt>Memory floor</dt><dd>{gb(current.engine_info.floor_bytes, 1)}: the weights plus the engine's runtime margin, before any context</dd></> : null}
                <dt>Command</dt><dd><code>{!current.native && v && v.budget_fraction !== (machine?.budget.default_fraction ?? 0.7) ? `${machine?.backend === 'cuda' ? 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB' : 'TENSORFOLD_MEMORY_LIMIT_GB'}=${(v.budget_bytes / 1024 ** 3).toFixed(0)} ` : ''}tensorfold serve {current.id}</code></dd>
              </dl>
            </details>
            <div className="actions">
              {isRunning(current) ? (engine?.owned && <button className="btn" aria-busy={busy} disabled={busy || stale} onClick={stop}>{busy ? 'Stopping…' : 'Stop'}</button>)
                : current.installed ? <button className="btn primary" aria-busy={busy} disabled={busy || stale || !current.runs_here} onClick={() => start(current)}>{busy ? 'Starting…' : 'Start'}</button>
                : <button className="btn primary" aria-busy={is(`pull:${current.id}`)} disabled={!current.runs_here || v?.level === 'no' || progressOf(current)?.running} onClick={() => pull(current)}>Download {gb((current.size_bytes ?? 0) + (current.drafter_bytes ?? 0))}</button>}
              {progressOf(current)?.running && <button className="btn" aria-busy={is(`stop:${current.id}`)} onClick={() => stopPull(current)}>Stop the download</button>}
              {v?.level === 'no' && <a className="btn" href="#settings">Connect a remote engine</a>}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
