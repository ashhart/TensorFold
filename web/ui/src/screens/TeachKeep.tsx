import { useCallback, useEffect, useState } from 'react'
import { api, type CheckRow, type CheckRunInfo, type CheckSet, type LivingRun, type LivingVersion } from '../api'
import { bytes } from '../units'
import { useWorking } from '../working'
import { Ask, AskName, cleanName } from '../ask'

type Toast = (t: string, k?: 'ok' | 'bad') => void
const when = (t: number) => new Date(t * 1000).toLocaleString()
const cut = (s: string | null | undefined, n = 120) => (s == null ? '—' : !s.trim() ? '(empty reply)' : s.length > n ? `${s.slice(0, n)}…` : s)

/** A copy's versions: kept before each lesson or by hand, restored or branched from, each costing what changed. */
export function Versions({ copy, busy, toast, onChange }: { copy: string; busy: boolean; toast: Toast; onChange: () => void }) {
  const [list, setList] = useState<LivingVersion[] | null>(null)
  const [store, setStore] = useState(0)
  const load = useCallback(() => api.livingVersions(copy).then((r) => { setList(r.versions); setStore(r.store_bytes) }).catch(() => setList([])), [copy])
  useEffect(() => { load() }, [load, busy])
  const { run, is } = useWorking()
  const [ask, setAsk] = useState<{ kind: 'restore' | 'remove' | 'branch'; v: LivingVersion } | { kind: 'save' } | null>(null)
  const act = (key: string, what: () => Promise<unknown>, said: string) => run(key, async () => { setAsk(null); try { await what(); toast(said); onChange(); setTimeout(load, 1500) } catch (e) { toast(String((e as Error).message ?? e), 'bad') } })
  return (
    <div className="card">
      <div className="membar-head"><b>History of {copy}</b><span className="small muted">{bytes(store, 1)} on disk for it</span></div>
      <p className="small muted">A point to go back to is kept before each lesson, and whenever you ask. Going back undoes every lesson since; the copy serves again afterwards.</p>
      {list && !list.length && <p className="small muted">Nothing yet.</p>}
      {ask?.kind === 'restore' && <Ask yes="Go back" busy={is(`restore:${ask.v.id}`)} onYes={() => act(`restore:${ask.v.id}`, () => api.livingVersionRestore(copy, ask.v.id), 'Going back: the copy stops, its files are written back, then it serves again')} onNo={() => setAsk(null)}
        text={<>Put <b>{copy}</b> back as it was at “{ask.v.label}”? The facts learned since are lost, unless you keep a point of now first.</>} />}
      {ask?.kind === 'remove' && <Ask danger yes="Remove the point" busy={is(`remove:${ask.v.id}`)} onYes={() => act(`remove:${ask.v.id}`, () => api.livingVersionRemove(copy, ask.v.id), 'Point removed')} onNo={() => setAsk(null)}
        text={<>Remove the point “{ask.v.label}”? Its files are freed unless another point holds them; there is no going back to it afterwards.</>} />}
      {ask?.kind === 'branch' && <AskName text={`A name for the copy made from “${ask.v.label}”`} initial={`${copy}-b`} ok="Make the copy" clean={cleanName} onOk={(n) => act(`branch:${ask.v.id}`, () => api.livingVersionBranch(copy, ask.v.id, n), 'Making the copy…')} onCancel={() => setAsk(null)} />}
      {ask?.kind === 'save' && <AskName text="A name for this point" initial="kept by hand" ok="Keep it" onOk={(n) => act('save', () => api.livingVersionSave(copy, n), 'Keeping a point: a few seconds, the files that changed are copied')} onCancel={() => setAsk(null)} />}
      {!!list?.length && <div className="tablewrap"><table className="tbl">
        <thead><tr><th>Point</th><th>Facts kept then</th><th>Disk</th><th /></tr></thead>
        <tbody>{[...list].reverse().map((v) => <tr key={v.id}>
          <td>{v.label}<div className="small muted">{when(v.at)}</div></td><td>{v.kept}</td><td title={`${v.changed} files differ from the point before`}>{bytes(v.new_bytes, 1)}</td>
          <td className="actions">
            <button className="btn sm" aria-busy={is(`restore:${v.id}`)} disabled={busy} onClick={() => setAsk({ kind: 'restore', v })}>Go back here</button>
            <button className="btn sm quiet" aria-busy={is(`branch:${v.id}`)} disabled={busy} onClick={() => setAsk({ kind: 'branch', v })}>Copy from here</button>
            <button className="btn sm quiet" aria-busy={is(`remove:${v.id}`)} disabled={busy} onClick={() => setAsk({ kind: 'remove', v })}>Remove</button>
          </td></tr>)}</tbody></table></div>}
      <div className="actions"><button className="btn" aria-busy={is('save')} disabled={busy} onClick={() => setAsk({ kind: 'save' })}>Keep a point now</button></div>
    </div>
  )
}

/** A copy's checks: the questions of the facts it kept, near misses and unrelated questions, asked with thinking off
 *  and on; two runs compared, with recall and the leak matrix. */
export function Checks({ copy, served, busy, toast, onChange }: { copy: string; served: boolean; busy: boolean; toast: Toast; onChange: () => void }) {
  const [set, setSet] = useState<CheckSet | null>(null)
  const [runs, setRuns] = useState<CheckRunInfo[]>([])
  const [near, setNear] = useState('')
  const [rows, setRows] = useState<CheckRow[] | null>(null)
  const load = useCallback(() => api.livingChecks(copy).then((r) => { setSet(r.set); setRuns(r.runs) }).catch(() => {}), [copy])
  useEffect(() => { load() }, [load, busy])
  const last = runs[runs.length - 1], prev = runs[runs.length - 2]
  useEffect(() => { if (last && prev) api.livingCheckCompare(copy, last.id, prev.id).then((r) => setRows(r.compare)).catch(() => setRows(null)); else setRows(null) }, [copy, last?.id, prev?.id])
  const addNear = async () => {
    if (!set || !near.trim()) return
    try { setSet(await api.livingChecksSet(copy, [...set.own, { q: near.trim(), kind: 'near' }])); setNear('') } catch (e) { toast(String((e as Error).message ?? e), 'bad') }
  }
  const dropOwn = async (q: string) => { if (!set) return; try { setSet(await api.livingChecksSet(copy, set.own.filter((o) => o.q !== q))) } catch (e) { toast(String((e as Error).message ?? e), 'bad') } }
  const { run: work, is } = useWorking()
  const run = (restart: boolean) => work(restart ? 'restart' : 'check', async () => { try { await api.livingChecksRun(copy, restart); toast(restart ? 'Restarting the copy, then asking the questions…' : 'Asking the questions… a few seconds each'); onChange() } catch (e) { toast(String((e as Error).message ?? e), 'bad') } })
  const mode = (m: 'off' | 'on') => last?.summary.modes[m]
  return (
    <div className="card">
      <b>Did {copy} remember?</b>
      <p className="small muted">The same questions before and after a lesson: the facts it learned, near misses that must not change, and unrelated questions. Run them after a restart, or whenever you like.</p>
      {set && <div className="small">
        <div><b>{set.facts.length}</b> fact questions, <b>{set.own.length}</b> of yours, <b>{set.unrelated.length}</b> unrelated.</div>
        {!!set.own.length && <ul style={{ margin: '6px 0' }}>{set.own.map((o) => <li key={o.q}>{o.q} <span className="muted">({o.kind === 'near' ? 'near miss' : o.kind === 'undone' ? 'undone: must not come back' : 'yours'})</span> <button className="btn sm quiet" onClick={() => dropOwn(o.q)} aria-label={`Remove ${o.q}`}>×</button></li>)}</ul>}
        <div className="row"><input className="input" value={near} onChange={(e) => setNear(e.target.value)} placeholder="A near miss, e.g. What is my dog called?" aria-label="Add a near-miss question" /><button className="btn sm" onClick={addNear} disabled={!near.trim()}>Add</button></div>
      </div>}
      <div className="actions">
        <button className="btn" aria-busy={is('check')} disabled={busy || !served} onClick={() => run(false)}>Ask it the questions</button>
        <button className="btn quiet" aria-busy={is('restart')} disabled={busy || !served} onClick={() => run(true)}>Restart the copy, then ask</button>
        {!served && <span className="small muted">Serve {copy} to check it.</span>}
      </div>
      {last && <dl className="figures">
        {(['off', 'on'] as const).filter((m) => mode(m)).map((m) => <div key={m}><dt>Thinking {m}</dt><dd>{mode(m)!.recalled} / {last.summary.facts} recalled<small>{mode(m)!.empty ? ` · ${mode(m)!.empty} empty` : ''}{mode(m)!.errors ? ` · ${mode(m)!.errors} errors` : ''}{mode(m)!.leaks.length ? ` · ${mode(m)!.leaks.length} leaks` : ''}{last.summary.undone ? ` · ${mode(m)!.undone_back ?? 0} of ${last.summary.undone} undone came back` : ''}</small></dd></div>)}
        <div><dt>Last run</dt><dd className="small">{last.label}<br />{when(last.at)}</dd></div>
      </dl>}
      {last && (['off', 'on'] as const).flatMap((m) => mode(m)?.leaks.map((l) => <p key={m + l.question} className="small bad">Leak (thinking {m}): “{l.question}” answered with “{l.answered_with}”</p>) ?? [])}
      {rows && prev && <>
        <p className="small" style={{ marginBottom: 4 }}>From “{prev.label}” to “{last.label}”: {rows.filter((r) => r.off?.changed || r.on?.changed).length} of {rows.length} answers changed.</p>
        <div className="tablewrap"><table className="tbl">
          <thead><tr><th>Question</th><th>Thinking off: before → after</th><th>Thinking on: before → after</th></tr></thead>
          <tbody>{rows.filter((r) => r.off?.changed || r.on?.changed || r.kind === 'fact' || r.kind === 'undone').map((r) => <tr key={r.q}>
            <td>{r.q}<div className="small muted">{r.kind === 'fact' ? 'fact' : r.kind === 'near' ? 'near miss' : r.kind === 'undone' ? 'undone by a restore' : r.kind}</div></td>
            {(['off', 'on'] as const).map((m) => <td key={m} className={`small ${r.kind === 'undone' ? (r[m]?.recalled ? 'bad' : 'good') : r[m]?.changed ? (r.kind === 'fact' ? (r[m]?.recalled ? 'good' : 'bad') : 'bad') : ''}`}>{r[m] ? (r[m]!.changed ? <>{cut(r[m]!.before)} → <b>{cut(r[m]!.after)}</b></> : `same: ${cut(r[m]!.after, 80)}`) : '—'}</td>)}
          </tr>)}</tbody></table></div>
        <p className="small muted" style={{ marginBottom: 0 }}>TensorFold Web reads words, not meaning: recall means the answer holds the words the fact adds (and does not deny them), a leak that it holds another fact's. Green: a fact now recalled. Red: a fact not recalled, or a near miss or unrelated answer that moved. Unchanged answers to unrelated questions are not listed.</p>
      </>}
    </div>
  )
}

/** What learning costs, measured with the Bench: speed and exactness per number of facts kept, quality, and how far
 *  the copy moved from the untaught model on prompts unrelated to what it learned (KL, against another machine). */
export function Measures({ copy, busy, peers, toast }: { copy: string; busy: boolean; peers: { id: string; name: string }[]; toast: Toast }) {
  const [runs, setRuns] = useState<LivingRun[]>([])
  const [peer, setPeer] = useState('')
  const load = useCallback(() => api.livingRuns(copy).then((r) => setRuns(r.runs)).catch(() => {}), [copy])
  useEffect(() => { load(); const t = setInterval(load, 10000); return () => clearInterval(t) }, [load])
  useEffect(() => { if (!peer && peers[0]) setPeer(peers[0].id) }, [peers, peer])
  const { run, is } = useWorking()
  const start = (key: string, what: () => Promise<{ id: string }>, said: string) => run(key, async () => { try { const r = await what(); toast(said); location.hash = `#bench/${r.id}` } catch (e) { toast(String((e as Error).message ?? e), 'bad') } })
  return (
    <div className="card">
      <b>What learning costs {copy}</b>
      <p className="small muted">Bench runs made while {copy} served, each with the facts it held then: speed, same answers with the speed-up off, quality. KL compares it with the untaught model on another machine, on unrelated prompts: 0 means learning changed nothing there.</p>
      <div className="actions">
        <button className="btn" aria-busy={is('quick')} disabled={busy} onClick={() => start('quick', () => api.benchStart({ mode: 'quick', exactness: true }), 'Quick score started: the Bench follows it')}>Quick score</button>
        <button className="btn" aria-busy={is('quality')} disabled={busy} onClick={() => start('quality', () => api.benchStart({ mode: 'quality', exactness: false }), 'Quality suite started (it takes a while): the Bench follows it')}>Quality suite</button>
        {!!peers.length && <>
          <select className="sel" value={peer} onChange={(e) => setPeer(e.target.value)} aria-label="Machine serving the untaught model">{peers.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
          <button className="btn" aria-busy={is('kl')} disabled={busy || !peer} onClick={() => start('kl', () => api.livingKl(peer), 'KL against the untaught model started: the Bench follows it')}>KL against it</button>
        </>}
        {!peers.length && <span className="small muted">Add a machine serving the untaught model in Fleet to measure KL against it.</span>}
      </div>
      {!!runs.length && <div className="tablewrap"><table className="tbl">
        <thead><tr><th>Run</th><th>Facts kept</th><th>Speed</th><th>Same tokens drafts off</th><th>Quality</th><th>KL</th></tr></thead>
        <tbody>{runs.map((r) => <tr key={r.id}>
          <td><a href={`#bench/${r.id}`}>{r.mode}</a><div className="small muted">{when(r.created)}{r.status !== 'done' ? ` · ${r.status}` : ''}</div></td>
          <td>{r.kept ?? '—'}</td><td>{r.tps != null ? `${Math.round(r.tps)} tok/s` : '—'}</td>
          <td>{r.checked ? `${r.exact} of ${r.checked}` : '—'}</td><td>{r.items ? `${r.right} / ${r.items}` : '—'}</td>
          <td>{r.mean_kl != null ? r.mean_kl.toFixed(4) : '—'}{r.reference ? <div className="small muted">vs {r.reference.replace(/^https?:\/\//, '').replace(/\/engine$/, '')}</div> : null}</td>
        </tr>)}</tbody></table></div>}
    </div>
  )
}

