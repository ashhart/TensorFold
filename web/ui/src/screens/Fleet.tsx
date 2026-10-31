import { useCallback, useEffect, useState } from 'react'
import { api, gb, num, type FleetCheck, type FleetMachine } from '../api'
import type { Ctx } from '../ctx'
import { ago } from './MonitorUsage'
import { useWorking } from '../working'
import { Ask, AskName } from '../ask'
import { ModelLogo } from '../brandLogo'
import { engineSkew } from '../version'

const MARK: Record<FleetCheck['status'], string> = { pass: '✓', warn: '!', fail: '✗', skip: '–' }

function MiniMemory({ m }: { m: NonNullable<NonNullable<FleetMachine['engine']>['memory']> }) {
  const engine = Math.min(m.engine ?? 0, m.total - m.available)
  const other = Math.max(0, m.total - m.available - engine)
  const pct = (x: number) => `${(x / m.total) * 100}%`
  return (
    <div className="fmem">
      <div className="bar3" role="img" aria-label={`Engine ${gb(engine, 1)}, other ${gb(other, 1)}, free ${gb(m.available, 1)} of ${gb(m.total)}`}>
        {m.engine != null && <i className="eng" style={{ width: pct(engine) }} />}<i className="oth" style={{ width: pct(m.engine == null ? m.total - m.available : other) }} />
      </div>
      <div className="small muted"><span className={`headroom ${m.headroom}`}>{gb(m.available, 1)} free</span> of {gb(m.total)}{m.engine != null ? `, engine ${gb(engine, 1)}` : ''}</div>
    </div>
  )
}

function MachineCard({ m, here, onRemove, onToken, onOpen, working, ask, setAsk }: { m: FleetMachine; here: string | null | undefined; onRemove: () => void; onToken: (t: string) => void; onOpen: () => void; working: string | null; ask: 'remove' | 'token' | null; setAsk: (a: 'remove' | 'token' | null) => void }) {
  const e = m.engine
  const st = !m.reachable ? 'unreachable' : e?.status ?? 'unknown'
  const chip = st === 'running' ? <span className="chip run">Serving</span> : st === 'starting' ? <span className="chip tight">Starting</span>
    : st === 'unreachable' ? <span className="chip bad">Not reachable</span> : st === 'failed' ? <span className="chip bad">Engine failed</span>
    : <span className="chip no">No model</span>
  return (
    <div className="card fcard">
      <div className="fhead">
        <div><b>{m.name}</b>{m.self && <span className="tag" style={{ marginLeft: 8 }}>here</span>}
          <div className="small muted">{m.reachable ? `${m.hostname ?? ''}${m.chip ? `, ${m.chip}` : ''}` : m.url}</div></div>
        {chip}
      </div>
      {!m.reachable && <p className="small" style={{ margin: '10px 0 0' }}>{m.error}{m.error?.includes('token') && <> <button className="btn ghost sm" onClick={() => setAsk('token')}>Change the token</button></>}</p>}
      {m.reachable && <>
        {st === 'running' && e && <>
          <div className="fmodel" title={e.model ?? ''}><ModelLogo names={[e.model_name, e.model]} size="xs" />{e.model_name ?? e.model}</div>
          <div className="fbig num"><b>{e.decode_tps == null ? '—' : Math.round(e.decode_tps)}</b> <span className="muted">tokens a second now</span></div>
          <div className="small muted">{(e.connections ?? 0) > 0 ? `answering ${e.connections} ${e.connections === 1 ? 'conversation' : 'conversations'}` : e.last_served ? `idle, last used ${ago(e.last_served.t)}` : 'nothing asked yet'}</div>
          <details className="more">
            <summary>The engine's figures</summary>
            <dl className="kv">
              <dt>Connections</dt><dd className="num">{e.connections ?? 0}{e.waiting ? `, ${e.waiting} waiting` : ''}</dd>
              {e.kv_fill != null && <><dt>KV cache</dt><dd className="num">{Math.round(e.kv_fill * 100)}% held</dd></>}
              {e.context_tokens != null && <><dt>Context</dt><dd className="num">{num(e.context_tokens)} tokens</dd></>}
              {e.prompt_cache && e.prompt_cache.prompt > 0 && <><dt>Prompt cache</dt><dd className="num">{Math.round((e.prompt_cache.restored / e.prompt_cache.prompt) * 100)}% restored</dd></>}
            </dl>
          </details>
        </>}
        {e?.memory && <MiniMemory m={e.memory} />}
        {!!e?.findings.length && <ul className="ffind">{e.findings.map((f, i) => <li key={i} className={f.level}>{f.title}</li>)}</ul>}
        <div className="small faint" style={{ marginTop: 10 }}>TensorFold {m.tensorfold_version ?? 'not installed'}, TensorFold Web {m.studio_version}{!m.self && m.ms != null ? `, read in ${m.ms} ms` : ''}</div>
        {!m.self && engineSkew(here, m.tensorfold_version) && <p className="small warn-text" style={{ margin: '6px 0 0' }}>Another TensorFold release than this machine's ({here}): a reply can part from it at a near tie, so a KL or exactness check across the two compares releases as well as machines. Update both to the same release first.</p>}
      </>}
      {!m.self && ask === 'remove' && <Ask danger yes="Remove it" busy={working === `remove:${m.id}`} onYes={onRemove} onNo={() => setAsk(null)} text={<>Remove <b>{m.name}</b> from this fleet? Its TensorFold Web and its AI keep running; only this list forgets it.</>} />}
      {!m.self && ask === 'token' && <AskName text={`${m.name}'s TensorFold Web token`} initial="" ok="Save the token" onOk={onToken} onCancel={() => setAsk(null)} />}
      {!m.self && <div className="actions">
        {m.url && <button className="btn sm" aria-busy={working === `open:${m.id}`} onClick={onOpen} title={m.token_set ? 'Opens it signed in, through a one-time link' : undefined}>{working === `open:${m.id}` ? 'Opening…' : 'Open its TensorFold Web'}</button>}
        <button className="btn quiet sm" aria-busy={working === `token:${m.id}`} onClick={() => setAsk('token')}>Token</button>
        <button className="btn quiet sm" aria-busy={working === `remove:${m.id}`} onClick={() => setAsk('remove')}>Remove</button>
      </div>}
    </div>
  )
}

export default function Fleet({ toast }: Ctx) {
  const [machines, setMachines] = useState<FleetMachine[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [url, setUrl] = useState('')
  const [token, setToken] = useState('')
  const [name, setName] = useState('')
  const [checks, setChecks] = useState<FleetCheck[] | 'running' | null>(null)
  const [busy, setBusy] = useState(false)
  const { run, working } = useWorking()
  const [ask, setAsk] = useState<{ id: string; kind: 'remove' | 'token' } | null>(null)

  const load = useCallback(() => api.fleet().then((f) => { setMachines(f.machines); setErr(null) }).catch((e) => setErr(String(e.message ?? e))), [])
  useEffect(() => {
    load()
    const t = setInterval(load, 5000)
    return () => clearInterval(t)
  }, [load])

  const test = async () => {
    setChecks('running')
    try { setChecks((await api.fleetTest(url, token)).checks) } catch (e) { setChecks(null); toast(`Not tested: ${(e as Error).message}`, 'bad') }
  }
  const add = async () => {
    setBusy(true)
    try {
      const m = await api.fleetAdd({ url, token, name })
      toast(`${m.name} added`)
      setUrl(''); setToken(''); setName(''); setChecks(null)
      load()
    } catch (e) { toast((e as Error).message, 'bad'); test() } finally { setBusy(false) }
  }
  // another TensorFold Web opened signed in: this one asks it for a one-time link with the token it keeps, which the page
  // never sees; the tab opens first, so the browser does not take it for a pop-up
  const openIt = (m: FleetMachine) => run(`open:${m.id}`, async () => {
    const w = window.open('about:blank', '_blank')
    try {
      const r = await api.fleetOpen(m.id)
      if (w) w.location.href = r.url; else location.href = r.url
    } catch (e) { w?.close(); toast(String((e as Error).message ?? e), 'bad') }
  })
  const remove = async (m: FleetMachine) => {
    setAsk(null)
    await run(`remove:${m.id}`, async () => { try { await api.fleetRemove(m.id); toast(`${m.name} removed`); await load() } catch (e) { toast((e as Error).message, 'bad') } })
  }
  const retoken = async (m: FleetMachine, t: string) => {
    setAsk(null)
    await run(`token:${m.id}`, async () => { try { await api.fleetUpdate(m.id, { token: t }); toast('Token saved'); await load() } catch (e) { toast((e as Error).message, 'bad') } })
  }

  const serving = machines?.filter((m) => m.reachable && m.engine?.status === 'running') ?? []
  const total = serving.reduce((a, m) => a + (m.engine?.decode_tps ?? 0), 0)
  return (
    <>
      <div className="head">
        <div><h1>Fleet</h1>
          <p>Your machines, each with its own AI: what runs where, and a way into each one's TensorFold Web. Their tokens stay in this TensorFold Web, never in the page.</p></div>
      </div>
      {err && <div className="notice bad">{err}</div>}
      {machines && <p className="small muted num" style={{ marginTop: -6 }}>{machines.length} {machines.length === 1 ? 'machine' : 'machines'}, {serving.length} with an AI running{total ? `, ${Math.round(total)} tokens a second being written in all` : ''}.</p>}
      <div className="fgrid">
        {machines?.map((m) => <MachineCard key={m.id} m={m} here={machines.find((x) => x.self)?.tensorfold_version} working={working} ask={ask?.id === m.id ? ask.kind : null} setAsk={(k) => setAsk(k ? { id: m.id, kind: k } : null)} onRemove={() => remove(m)} onToken={(t) => retoken(m, t)} onOpen={() => openIt(m)} />)}
      </div>
      <div className="card" style={{ marginTop: 14, maxWidth: 720 }}>
        <b>Add a machine</b>
        <p className="small muted" style={{ margin: '4px 0 0' }}>Its TensorFold Web's address (the name <code>tailscale serve</code> gives it, or <code>http://host:8090</code>) and its token: <code>~/.config/tensorfold-web/web-token</code> on Linux, <code>~/Library/Application Support/TensorFold Web/web-token</code> on a Mac. No restart needed.</p>
        <div className="field"><span>Address</span><input type="text" value={url} onChange={(e) => { setUrl(e.target.value); setChecks(null) }} placeholder="http://spark.example.ts.net" aria-label="TensorFold Web address" spellCheck={false} autoComplete="off" /></div>
        <div className="field"><span>Token</span><input type="password" value={token} onChange={(e) => { setToken(e.target.value); setChecks(null) }} placeholder="its web-token" aria-label="TensorFold Web token" autoComplete="off" /></div>
        <div className="field"><span>Name</span><input type="text" value={name} onChange={(e) => setName(e.target.value)} placeholder="its hostname if left empty" aria-label="Name" maxLength={60} /></div>
        <div className="actions">
          <button className="btn" aria-busy={checks === 'running'} disabled={!url.trim() || checks === 'running'} onClick={test}>{checks === 'running' ? 'Testing…' : 'Test'}</button>
          <button className="btn primary" aria-busy={busy} disabled={!url.trim() || busy} onClick={add}>{busy ? 'Adding…' : 'Add'}</button>
        </div>
        {Array.isArray(checks) && <ul className="checks">{checks.map((c) => <li key={c.id} className={c.status === 'pass' ? 'good' : c.status === 'fail' ? 'bad' : c.status}><b>{MARK[c.status]}</b> {c.label} <span className="muted">{c.detail}{c.seconds ? `, ${c.seconds} s` : ''}</span></li>)}</ul>}
      </div>
    </>
  )
}
