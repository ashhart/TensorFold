import { useCallback, useEffect, useState } from 'react'
import { api, download, type SetupCheck, type Settings } from '../api'
import { copyText } from '../shareCard'

const MARK: Record<SetupCheck['status'], string> = { pass: '✓', warn: '!', fail: '✗', skip: '–' }
const STATUS: Record<SetupCheck['status'], string> = { pass: 'ready', warn: 'to look at', fail: 'needs fixing', skip: 'not needed here' }
// the steps of a first run, in the order a newcomer meets them
const FIRST_RUN = ['backend', 'installed', 'port', 'folder', 'model', 'engine']
const COMMAND = /^(brew|sudo|tensorfold-web|download) /

function useChecks(every = 0) {
  const [checks, setChecks] = useState<SetupCheck[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [at, setAt] = useState<Date | null>(null)
  const load = useCallback(() => {
    setBusy(true)
    return api.diagnostic().then((d) => { setChecks(d.checks); setErr(null); setAt(new Date()); return d.checks })
      .catch((e) => { setErr(String(e.message ?? e)); return null }).finally(() => setBusy(false))
  }, [])
  useEffect(() => {
    load()
    if (!every) return
    const t = setInterval(load, every)
    return () => clearInterval(t)
  }, [load, every])
  return { checks, err, busy, load, at }
}

/** Check again, said: the checks take a fraction of a second and often come back the same, so the button stays on
 *  "Checking…" long enough to be seen and a toast says what came back, changed or not. */
function useAgain(checks: SetupCheck[] | null, load: () => Promise<SetupCheck[] | null>, toast: (t: string, k?: 'ok' | 'bad') => void) {
  const [showing, setShowing] = useState(false)
  const again = async () => {
    setShowing(true)
    const before = new Map((checks ?? []).map((c) => [c.id, c.status]))
    const [after] = await Promise.all([load(), new Promise((r) => setTimeout(r, 600))])
    setShowing(false)
    if (!after) return toast('The checks could not run again', 'bad')
    const changed = after.filter((c) => before.has(c.id) && before.get(c.id) !== c.status).length
    const fix = after.filter((c) => c.status === 'fail').length, look = after.filter((c) => c.status === 'warn').length
    const state = [fix && `${fix} to fix`, look && `${look} to look at`].filter(Boolean).join(', ') || 'everything is ready'
    toast(`Checked again: ${state}${changed ? ` (${changed} changed)` : ', nothing changed'}`)
  }
  return { again, showing }
}
const checkedAt = (at: Date | null) => at ? <span className="small muted" aria-live="polite">Checked at {at.toLocaleTimeString()}</span> : null

function Fix({ text, toast }: { text: string; toast: (t: string, k?: 'ok' | 'bad') => void }) {
  if (!COMMAND.test(text) || text.startsWith('download ')) return <div className="cfix">{text}</div>
  const copy = () => copyText(text).then(() => toast('Command copied'), () => toast('Copy it by hand', 'bad'))
  return <div className="cfix"><code>{text}</code> <button className="btn ghost sm" onClick={copy} aria-label={`Copy the command ${text}`}>Copy</button></div>
}

export function CheckList({ checks, numbered, toast }: { checks: SetupCheck[]; numbered?: boolean; toast: (t: string, k?: 'ok' | 'bad') => void }) {
  const Tag = numbered ? 'ol' : 'ul'
  return (
    <Tag className="setup">
      {checks.map((c) => (
        <li key={c.id} className={c.status}>
          <span className="cmark" aria-hidden="true">{MARK[c.status]}</span>
          <div><b>{c.label}</b> <span className="sr-only">{STATUS[c.status]}</span><span className="muted">{c.detail}</span>
            {c.status !== 'pass' && c.fix && <Fix text={c.fix} toast={toast} />}</div>
        </li>
      ))}
    </Tag>
  )
}

/** The first-run list on Home: shown until every step is ready, or until it is put away. */
export function FirstRun({ settings, saveSettings, toast }: { settings: Settings | null; saveSettings: (p: Partial<Settings>) => Promise<Settings>; toast: (t: string, k?: 'ok' | 'bad') => void }) {
  const hidden = settings?.setup_hidden ?? true
  const { checks, busy, load, at } = useChecks(hidden ? 0 : 15000)
  const { again, showing } = useAgain(checks, load, toast)
  if (hidden || !checks) return null
  const steps = FIRST_RUN.map((id) => checks.find((c) => c.id === id)).filter((c): c is SetupCheck => !!c)
  const ready = steps.filter((c) => c.status === 'pass' || c.status === 'skip').length
  if (ready === steps.length) return null
  const hide = () => saveSettings({ setup_hidden: true }).then(() => toast('Put away; Settings keeps the same checks'), (e) => toast(String(e), 'bad'))
  return (
    <section className="card firstrun" aria-labelledby="firstrun-title">
      <div className="membar-head"><b id="firstrun-title">Getting started</b><span className="small muted" aria-live="polite">{ready} of {steps.length} ready</span></div>
      <p className="small muted" style={{ margin: '4px 0 0' }}>What this machine needs before it serves a model, checked here. Each step that is not ready says what to do.</p>
      <CheckList checks={steps} numbered toast={toast} />
      <div className="actions"><button className="btn" onClick={again} disabled={busy || showing} aria-busy={showing}>{showing ? 'Checking…' : 'Check again'}</button><button className="btn quiet" onClick={hide}>Hide this list</button>{checkedAt(at)}</div>
    </section>
  )
}

/** Settings: every check, and the report to send to someone helping. */
export function Diagnostic({ settings, saveSettings, toast }: { settings: Settings | null; saveSettings: (p: Partial<Settings>) => Promise<Settings>; toast: (t: string, k?: 'ok' | 'bad') => void }) {
  const { checks, err, busy, load, at } = useChecks()
  const { again, showing } = useAgain(checks, load, toast)
  const summary = () => (checks ?? []).map((c) => `${MARK[c.status]} ${c.label}: ${c.detail}${c.status !== 'pass' && c.fix ? ` (fix: ${c.fix})` : ''}`).join('\n')
  const copy = () => copyText(summary()).then(() => toast('Summary copied'), () => toast('Not copied', 'bad'))
  return (
    <div className="card" aria-labelledby="diag-title">
      <div className="membar-head"><b id="diag-title">This machine's checks</b>{checks && <span className="small muted">{checks.filter((c) => c.status === 'fail').length} to fix, {checks.filter((c) => c.status === 'warn').length} to look at</span>}</div>
      {err && <div className="notice bad">{err}</div>}
      {checks && <CheckList checks={checks} toast={toast} />}
      <div className="actions">
        <button className="btn" onClick={again} disabled={busy || showing} aria-busy={showing}>{showing ? 'Checking…' : 'Check again'}</button>
        <button className="btn" onClick={() => download('/api/diagnostic/report', 'tensorfold-web-diagnostic.json').catch((e) => toast(`Not downloaded: ${e}`, 'bad'))}>Download the diagnostic report</button>
        <button className="btn quiet" onClick={copy} disabled={!checks}>Copy a summary</button>
        {settings?.setup_hidden && <button className="btn quiet" onClick={() => saveSettings({ setup_hidden: false }).then(() => { location.hash = '#home' }, (e) => toast(String(e), 'bad'))}>Show the getting-started list</button>}
        {checkedAt(at)}
      </div>
      <p className="small muted" style={{ marginBottom: 0 }}>The report is a JSON file with these checks, the machine, the settings, the engine's state and its last 300 log lines, the fleet's names and the usage totals. Keys and tokens are taken out; machine names and paths stay.</p>
    </div>
  )
}
