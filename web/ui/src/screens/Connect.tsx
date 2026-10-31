import { useEffect, useState } from 'react'
import { api, download, num, type CheckResult, type ConnectInfo, type Model } from '../api'
import { copyText } from '../shareCard'
import type { Ctx } from '../ctx'
import { useWorking } from '../working'

const EMPTY: ConnectInfo = { engine_url: '', lan_url: null, share: false, api_key: null, key_pending_restart: false, studio_token_set: false, snippets: [] }
// each tool's logo, by its recipe id (assets/logos/README.md says where each one comes from)
const LOGOS: Record<string, string> = Object.fromEntries(Object.entries(
  import.meta.glob('../assets/logos/*.{svg,png}', { eager: true, query: '?url', import: 'default' }) as Record<string, string>,
).map(([path, url]) => [path.split('/').pop()!.replace(/\.(svg|png)$/, ''), url]))
const Logo = ({ id, size }: { id: string; size: 'sm' | 'lg' }) => LOGOS[id] ? <span className={`toollogo ${size}`} aria-hidden="true"><img src={LOGOS[id]} alt="" /></span> : null
const TOOL_KEY = 'tfs-tool'
const WHERE_KEY = 'tfs-connect-where'
// opened from this machine itself, a tool is most likely here too; opened from elsewhere, it is on that computer
const firstWhere = (): 'here' | 'away' => { try { const w = localStorage.getItem(WHERE_KEY); if (w === 'here' || w === 'away') return w } catch { /* none */ } return ['localhost', '127.0.0.1', '[::1]'].includes(location.hostname) ? 'here' : 'away' }

export default function Connect({ engine, settings, saveSettings, refreshSettings, refreshEngine, toast, stale }: Ctx) {
  const [info, setInfo] = useState<ConnectInfo>(EMPTY)
  const [tool, setTool] = useState<string>(() => { try { return localStorage.getItem(TOOL_KEY) ?? 'claude-code' } catch { return 'claude-code' } })
  const [where, setWhere] = useState<'here' | 'away'>(firstWhere)
  const [showKey, setShowKey] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [results, setResults] = useState<Record<string, { ok: boolean; checks: CheckResult[] } | 'running'>>({})
  const { run, is } = useWorking()
  const runTest = async (id: string) => {
    setResults((r) => ({ ...r, [id]: 'running' }))
    try { const out = await api.connectTest(id); setResults((r) => ({ ...r, [id]: out })) } catch (e) { setResults((r) => { const n = { ...r }; delete n[id]; return n }); toast(String(e), 'bad') }
  }
  const reload = () => api.connect(where).then(setInfo).catch((e) => setErr(String(e)))
  useEffect(() => { reload() }, [where, settings?.share, settings?.api_key_set, engine?.status, engine?.model])
  useEffect(() => { try { localStorage.setItem(WHERE_KEY, where) } catch { /* no storage */ } }, [where])
  const [models, setModels] = useState<Model[]>([])
  useEffect(() => { api.catalog().then((c) => setModels(c.models)).catch(() => {}) }, [engine?.model])
  useEffect(() => { try { localStorage.setItem(TOOL_KEY, tool) } catch { /* no storage */ } }, [tool])
  const copy = async (text: string, what = 'Copied') => {
    try { await copyText(text); toast(what) } catch { toast('The browser refused the clipboard; select the text and copy it yourself.', 'bad') }
  }
  const newKey = () => run('key', async () => {
    setErr(null)
    try { const r = await api.rotateKey(); toast(r.key_pending_restart ? 'New key set; the running engine keeps the old one until it restarts' : 'New key set; applies at the next start'); await refreshSettings(); await reload() } catch (e) { setErr(String(e)) }
  })
  // a coding agent's prompts need a long window: say so on the tool itself, with the one change that fixes it here
  const served = models.find((m) => m.id === settings?.model) ?? null
  const restartAt = (n: number) => run('window', async () => {
    try {
      await api.stop()
      await api.start({ model: settings!.model!, context: n })
      await Promise.all([refreshEngine(), refreshSettings()])
      toast(`Restarting with a ${num(n)}-token window`)
    } catch (e) { toast(String(e), 'bad') }
  })
  const WindowNotice = ({ name, need, window }: { name: string; need: number; window: number }) => {
    const most = served?.verdict?.context_tokens ?? 0
    const target = most >= 65536 ? 65536 : most >= need ? need : null
    const others = models.filter((m) => m.installed && m.runs_here && m.id !== served?.id && (m.verdict?.context_tokens ?? 0) >= need)
    const here = engine?.managed !== false && !settings?.remote && !!settings?.model
    return (
      <div className="notice"><div>
        <b>{name} needs a conversation window of {num(need)} tokens or more</b>; this machine serves {num(window)} now, so the engine would refuse its first prompt.{' '}
        {!here ? <>Raise the window where the engine runs.</>
          : target ? <><button className="btn sm primary" aria-busy={is('window')} disabled={is('window') || stale} onClick={() => restartAt(target)}>{is('window') ? 'Restarting…' : `Serve ${num(target)} tokens`}</button>{' '}
            <span className="small muted">Restarts {served?.name ?? 'the AI'} (a few seconds) and sets Longest conversation in Settings, where Auto puts it back.</span></>
          : <>{served?.name ?? 'The model running here'} cannot hold that much on this machine{others.length ? <>: start {others.map((m) => m.name).join(' or ')} from <a href="#models">Models</a>, which can.</> : '.'}</>}
      </div></div>
    )
  }
  const studioPath = `${location.origin}/engine/v1`
  const current = info.snippets.find((s) => s.id === tool) ?? info.snippets[0]
  const keyField = (id: string, key = info.api_key) => key ? (
    <div className="field"><span>API key</span><input type={showKey ? 'text' : 'password'} readOnly value={key} aria-label={`API key for ${id}`} /><button className="btn" onClick={() => setShowKey(!showKey)}>{showKey ? 'Hide' : 'Show'}</button><button className="btn" onClick={() => copy(key, 'Key copied')} aria-label={`Copy the key for ${id}`}>Copy</button></div>
  ) : null
  const host = location.hostname.split('.')[0]
  const needKey = where === 'away' && !info.relay_key
  const place = where === 'away' ? 'on another computer' : 'on this machine'
  return (
    <section>
      <div className="head"><div><h1>Connect your tools</h1><p>Use your AI in the apps you already have: say where the app runs, pick it, copy its settings.</p></div></div>
      {err && <div className="notice bad">{err}</div>}
      {info.key_pending_restart && where === 'here' && <div className="notice">A new key is set, but the running engine still uses the one it was started with. Tools on this machine keep working with that one until the engine restarts (Monitor).</div>}

      <h2 className="subhead"><span className="stepno">1</span> Where does the app run?</h2>
      <div className="wherepick" role="radiogroup" aria-label="Where the app runs">
        <button role="radio" aria-checked={where === 'here'} onClick={() => setWhere('here')}><b>On this machine</b><span>{host}, where the AI runs. The app talks to it directly.</span></button>
        <button role="radio" aria-checked={where === 'away'} onClick={() => setWhere('away')}><b>On another computer</b><span>A laptop, a phone, another server: through this TensorFold Web's address, with a key.</span></button>
      </div>

      <h2 className="subhead"><span className="stepno">2</span> Which app?</h2>
      {Object.entries(info.kinds ?? { all: 'Tools' }).map(([kind, label]) => {
        const group = info.snippets.filter((s) => !info.kinds || s.kind === kind)
        return group.length ? <div key={kind} className="toolgroup"><span className="small muted">{label}</span>
          <div className="toolbar" role="tablist" aria-label={label}>
            {group.map((s) => <button key={s.id} role="tab" aria-selected={current?.id === s.id} onClick={() => setTool(s.id)}><Logo id={s.id} size="sm" />{s.name}</button>)}
          </div></div> : null
      })}

      {current && <>
        <h2 className="subhead"><span className="stepno">3</span> Its settings</h2>
        <div className="card snippet rel" role="tabpanel">
          <h2 className="toolhead"><Logo id={current.id} size="lg" />{current.name} <span className="chip no">{current.form}</span> <span className={`chip ${where === 'away' ? 'run' : 'fits'}`}>{place}</span></h2>
          <p>{current.note}{current.where_note ? ` ${current.where_note}` : ''}</p>
          {needKey && <div className="notice"><div>From another computer the app needs a key, and none is set yet. <button className="btn sm primary" aria-busy={is('key')} onClick={newKey}>Generate a key</button> <span className="small muted">It replaces YOUR-KEY below.</span></div></div>}
          {current.min_context && info.window != null && info.window < current.min_context && <WindowNotice name={current.name} need={current.min_context} window={info.window} />}
          {(() => {
            // the key stays hidden on screen, as in the fields below; Copy takes the real one
            const secret = where === 'away' ? info.relay_key : info.api_key
            const shown = secret && !showKey ? current.snippet.split(secret).join(`${secret.slice(0, 5)}••••••••`) : current.snippet
            return <>{secret && current.snippet.includes(secret) && <button className="btn sm quiet showkey" onClick={() => setShowKey(!showKey)}>{showKey ? 'Hide the key' : 'Show the key'}</button>}<pre>{shown}</pre></>
          })()}
          <button className="btn copy" onClick={() => copy(current.snippet, `${current.name} settings copied`)}>Copy</button>
          <div className="actions" style={{ marginTop: 10 }}>
            {current.checks.length > 0 && <button className="btn" aria-busy={results[current.id] === 'running'} disabled={results[current.id] === 'running'} onClick={() => runTest(current.id)}>{results[current.id] === 'running' ? 'Testing…' : 'Test connection'}</button>}
            {current.compose && <button className="btn ghost" onClick={() => download('/api/connect/compose/open-webui', 'docker-compose.yml').catch((e) => toast(`Not downloaded: ${e}`, 'bad'))}>Download docker-compose.yml</button>}
            {current.checks.length > 0 && <span className="small muted">The test makes the calls {current.name} makes ({current.protocol === 'anthropic' ? 'Anthropic Messages' : current.protocol === 'responses' ? 'the Responses API' : 'OpenAI chat completions'}) to the AI, from this machine.</span>}
          </div>
          {(() => { const r = results[current.id]; return r && r !== 'running' ? <ul className="checks">{r.checks.map((c) => <li key={c.id} className={c.skipped ? 'muted' : c.ok ? 'good' : 'bad'}><b>{c.skipped ? '–' : c.ok ? '✓' : '✗'}</b> {c.label} <span className="muted">{c.detail}, {c.seconds} s</span></li>)}</ul> : null })()}
          <p className="small muted" style={{ marginBottom: 0 }}>{current.checked?.date ? `Checked end to end with ${current.name} itself on ${current.checked.date}, ${current.checked.machine}: ${current.checked.how}` : current.checked?.not ? `Not checked with ${current.name} itself: ${current.checked.not}` : `Not yet checked with ${current.name} itself; the test above checks the engine's side.`}</p>
        </div>
      </>}

      <h2 className="subhead">An app not in the list?</h2>
      <div className="card">
        <p>Any app that speaks the OpenAI API: give it this address and {where === 'away' ? 'this key' : info.api_key ? 'this key' : 'any key (the AI asks for none on this machine)'}, then pick the model it lists.</p>
        <div className="field"><span>Address</span><input type="text" readOnly value={where === 'away' ? studioPath : `${info.engine_url}/v1`} aria-label="Address for the app" /><button className="btn" onClick={() => copy(where === 'away' ? studioPath : `${info.engine_url}/v1`, 'Address copied')}>Copy</button></div>
        {where === 'away' ? keyField('another computer', info.relay_key) : keyField('this machine')}
        {where === 'away' && <div className="actions" style={{ marginTop: 10 }}>
          <button className="btn ghost" aria-busy={is('key')} onClick={newKey}>{info.relay_key ? 'Generate a new key' : 'Generate a key'}</button>
          {info.relay_key && <span className="small muted">A new key replaces this one: the apps holding it stop working.</span>}
        </div>}
        <details className="pmore" style={{ marginTop: 12 }}>
          <summary className="small">Advanced: let the AI answer on the local network without TensorFold Web</summary>
          <p className="small muted">For a tool on the same network that cannot use TensorFold Web's address. The AI then answers on this machine's network name too{info.relay_key || info.api_key ? ', with the key' : '; set a key first'}.</p>
          <label className="toggle small"><input type="checkbox" checked={!!settings?.share} onChange={(e) => saveSettings({ share: e.target.checked }).catch((err) => toast(`Not saved: ${err}`, 'bad'))} /> Also let the AI answer directly on the local network</label>
          {settings?.share && info.lan_url && <div className="field"><span>Network</span><input type="text" readOnly value={`${info.lan_url}/v1`} aria-label="Engine network address" /><button className="btn" onClick={() => copy(`${info.lan_url}/v1`, 'Address copied')}>Copy</button></div>}
        </details>
      </div>
    </section>
  )
}
