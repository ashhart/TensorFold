import { useCallback, useEffect, useRef, useState } from 'react'
import { api, download, shortName, type GraphFact, type LessonRecord, type LivingView, type PreviewFact } from '../api'
import type { Ctx } from '../ctx'
import { bytes, secs } from '../units'
import { Checks, Measures, Versions } from './TeachKeep'
import { elapsed, useElapsed, useWorking } from '../working'
import { Ask, AskName, cleanName } from '../ask'

const STATE: Record<string, string> = { queued: 'waiting', learning: 'learning…', kept: 'kept', 'partly kept': 'partly kept', missed: 'not kept', failed: 'failed', cancelled: 'cancelled', interrupted: 'interrupted by a TensorFold Web restart', refused: 'stopped: not what was approved', 'rolled back': 'undone', learned: 'kept' }
const when = (t: number) => new Date(t * 1000).toLocaleString()
const PHASE: Record<string, string> = { 'saving a version': 'keeping a point to go back to', 'checking before': 'asking it the questions before the lesson', learning: 'learning', 'checking after': 'asking it the questions again', 'rolling back': 'undoing the lesson', done: 'done' }

/** The three steps of a lesson, the current one marked: a person always knows where they are. */
function Steps({ at }: { at: 1 | 2 | 3 }) {
  const steps = ['What should it remember?', 'Here is what I understood', 'It learns']
  return (
    <ol className="steps" aria-label="Steps">
      {steps.map((s, i) => <li key={s} aria-current={at === i + 1 ? 'step' : undefined} className={at > i + 1 ? 'done' : ''}><span className="n">{i + 1}</span>{s}</li>)}
    </ol>
  )
}
// a copy's suggested name: the model's first word, in lower case ("nemotron-living")
const suggest = (base: string, names?: Record<string, string>) => `${(names?.[base] ?? shortName(base)).split(/[\s-]/)[0].toLowerCase().replace(/[^a-z0-9]/g, '') || 'model'}-living`

/** Teach: copies of a model that learn facts into their own weights (TensorFold's Living Weights), facts found and
 *  approved before anything is learned, the lesson as it goes, and what each copy has kept. */
export default function Teach({ engine, toast, refreshEngine, stale }: Ctx) {
  const [view, setView] = useState<LivingView | null>(null)
  const [name, setName] = useState('')
  const [base, setBase] = useState('')
  const [text, setText] = useState('')
  const [facts, setFacts] = useState<(PreviewFact & { approved: boolean; edited?: boolean })[] | null>(null)
  const [previewCopy, setPreviewCopy] = useState<string | null>(null)
  const [finding, setFinding] = useState(false)
  const [busy, setBusy] = useState(false)
  const { run, is, any } = useWorking()                                // the pressed button shows it works
  // a question asked in the page, where the action is: deleting a copy, forgetting a fact, naming a fetched copy
  const [ask, setAsk] = useState<{ kind: 'remove'; name: string } | { kind: 'forget'; fact: string } | { kind: 'fetch'; name: string } | { kind: 'stopnow' } | null>(null)
  const [learned, setLearned] = useState<{ facts: GraphFact[]; lessons: LessonRecord[] } | null>(null)
  const [peer, setPeer] = useState('')
  const [theirs, setTheirs] = useState<{ name: string; base: string | null; bytes: number; taught: number }[] | null>(null)
  const [checkAround, setCheckAround] = useState(true)
  const unread = useRef(false)                                         // one toast per outage, not one per poll
  const load = useCallback(() => api.living().then((v) => { setView(v); setBase((b) => b || v.bases[0] || ''); setName((n) => n || (v.bases[0] ? suggest(v.bases[0], v.base_names) : '')); unread.current = false })
    .catch((e) => { if (!unread.current) toast(`Not read: ${e.message ?? e}`, 'bad'); unread.current = true }), [toast])
  useEffect(() => { load() }, [load])
  useEffect(() => {                                        // "Teach this" in Chat hands the text over
    try { const given = sessionStorage.getItem('tfs:teach-text'); if (given) { setText(given); sessionStorage.removeItem('tfs:teach-text') } } catch { /* no storage */ }
  }, [])
  const choring = view?.chore?.state === 'running'
  const starting = engine?.status === 'starting'
  const running = !!view?.busy || view?.creating?.state === 'copying' || choring || starting
  useEffect(() => { if (!running) return; const t = setInterval(load, 2000); return () => clearInterval(t) }, [running, load])
  useEffect(() => { load() }, [engine?.status, engine?.model_repo, load])   // a copy served or stopped elsewhere
  const served = view?.served ?? null
  useEffect(() => { if (previewCopy && served !== previewCopy) { setFacts(null); setPreviewCopy(null) } }, [served, previewCopy])   // a preview belongs to its copy
  useEffect(() => { if (served) api.livingFacts(served).then(setLearned).catch(() => setLearned(null)); else setLearned(null) }, [served, view?.job?.status, view?.job?.items.filter((i) => i.state !== 'queued' && i.state !== 'learning').length])

  const create = () => run('create', async () => {
    try { await api.livingCreate(name.trim(), base); toast('Copying the model…'); await load() } catch (e) { toast(String((e as Error).message ?? e), 'bad') }
  })
  const serve = (model: string) => run(`serve:${model}`, async () => {
    setBusy(true)
    try {
      if (engine?.owned && engine.status !== 'stopped') await api.stop()
      await api.start({ model }); await refreshEngine(); toast('Starting the copy; it learns once it serves'); await load()
    } catch (e) { toast(String((e as Error).message ?? e), 'bad') } finally { setBusy(false) }
  })
  const remove = (n: string) => run(`remove:${n}`, async () => {
    setAsk(null)
    try { await api.livingRemove(n); toast(`${n} deleted`); await load() } catch (e) { toast(String((e as Error).message ?? e), 'bad') }
  })
  const find = async () => {
    setFinding(true); setFacts(null)
    try {
      const r = await api.livingPreview(text)
      setFacts(r.facts.map((f) => ({ ...f, approved: f.supported }))); setPreviewCopy(r.copy)
      if (!r.facts.length) toast('No fact found in this text', 'bad')
    } catch (e) { toast(String((e as Error).message ?? e), 'bad') } finally { setFinding(false) }
  }
  // a fact or question edited by hand no longer has the checks the preview made on the model's wording
  const edit = (id: string, patch: Partial<PreviewFact & { approved: boolean }>) => setFacts((fs) => fs && fs.map((f) => (f.id === id ? { ...f, ...patch, edited: f.edited || 'fact' in patch || 'question' in patch } : f)))
  const approved = (facts ?? []).filter((f) => f.approved && f.fact.trim())
  const teach = () => run('teach', async () => {
    try { await api.livingTeach(approved.map((f) => ({ fact: f.fact, question: f.question })), { version_before: true, check: checkAround, copy: previewCopy ?? served ?? '' }); setFacts(null); setText(''); await load() } catch (e) { toast(String((e as Error).message ?? e), 'bad') }
  })
  const job = view?.job
  const forget = (fact: string) => run(`forget:${fact}`, async () => {
    setAsk(null)
    if (!served) return
    try { await api.livingForget(served, fact); toast('Forgetting…'); await load() } catch (e) { toast(String((e as Error).message ?? e), 'bad') }
  })
  const listTheirs = () => run('peers', async () => { try { setTheirs((await api.livingPeerCopies(peer)).copies) } catch (e) { toast(String((e as Error).message ?? e), 'bad') } })
  const fetchCopy = (n: string, as: string) => run(`fetch:${n}`, async () => {
    setAsk(null)
    try { await api.livingFetch(peer, n, cleanName(as)); toast('Fetching the files that differ from the base…'); setTheirs(null); await load() } catch (e) { toast(String((e as Error).message ?? e), 'bad') }
  })
  // the token travels in a header, never in an address: the view gets a cookie for its own path, the card a download
  const openView = async () => {
    const w = window.open('about:blank', '_blank')
    try {
      await api.livingViewSession()
      if (w) w.location.href = '/living-view/'; else location.href = '/living-view/'
    } catch (e) { w?.close(); toast(`The view did not open: ${(e as Error).message}`, 'bad') }
  }
  const downloadCard = (n: string) => run(`card:${n}`, async () => {
    try { await download(`/api/living/copies/${encodeURIComponent(n)}/card`, `${n}-model-card.md`) }
    catch (e) { toast(`Not downloaded: ${(e as Error).message}`, 'bad') }
  })
  // a fact TensorFold Web taught, found by the engine's id when the engine wrote it its own way ("The user has a dog…")
  const taughtAs = (f: GraphFact): string | undefined => {
    const kept = (learned?.lessons ?? []).filter((x) => x.outcome === 'kept' || x.outcome === 'partly kept')
    return (kept.find((x) => x.found?.some((g) => g.id === f.id)) ?? kept.find((x) => x.fact === f.text))?.fact
  }
  const jobSince = useElapsed(job?.started, job?.status === 'running')
  const choreSince = useElapsed(view?.chore?.started, view?.chore?.state === 'running')
  const copySince = useElapsed(view?.creating?.started, view?.creating?.state === 'copying')
  const learningIndex = job ? job.items.findIndex((i) => i.state === 'learning') : -1

  return (
    <section>
      <div className="head"><div><h1>Teach</h1><p>Tell your AI something it should remember: a fact about you, your work, your company. It learns it into a copy of itself, so the original stays as it was, and anything it learned can be undone.</p></div></div>
      {view?.chore?.state && <div className={`notice ${view.chore.state === 'failed' ? 'bad' : 'info'}`} role="status">{view.chore.state === 'running' ? <span><span className="pulse" />{view.chore.kind} of {view.chore.copy} is going, {elapsed(choreSince)} so far{view.chore.kind === 'a restore' || view.chore.kind === 'a branch' ? ' (a few seconds to a minute: the files that differ are written back, then the copy serves again)' : view.chore.kind === 'checks' ? ' (each question takes a few seconds)' : ''}.</span> : view.chore.state === 'failed' ? `${view.chore.kind} of ${view.chore.copy} failed: ${view.chore.error}` : `${view.chore.kind} of ${view.chore.copy} done.`}</div>}
      {view?.teachable && <div className="notice info">Learning cannot run here: {view.teachable}.{/not installed/.test(view.teachable) ? ' Copies need the native engine to be served too.' : ' Copies can still be made and served without learning.'}</div>}

      {!served && view && <div className="card">
        {text.trim() && <div className="notice info">The message from Chat is kept here: <i>“{text.trim().length > 160 ? `${text.trim().slice(0, 160)}…` : text.trim()}”</i>. {view.copies.length ? 'Serve a copy below' : 'Make a copy below, then serve it'}, and it is ready to teach.</div>}
        {view.copies.length ? <>
          <b>Which copy should learn?</b>
          <p className="small muted">A copy learns while it runs. Serve one below (the AI restarts, a few seconds), then teach it here.</p>
        </> : <>
          <b>First, a copy that can learn</b>
          <p className="small muted">Teaching changes a copy of the AI, never the original. Make one from {view.bases.length ? <b>{view.base_names?.[base] ?? shortName(base)}</b> : 'a downloaded model'} (instant on a Mac; about 18 GB to write on Linux), then serve it and teach it.</p>
          {!view.bases.length && <p className="small muted">No model that learns is on this machine yet. Only Nemotron 3.5 Lightning learns today, on a Mac or a GB10: get it from <a href="#models">Models</a>.</p>}
        </>}
        {!!view.bases.length && <div className="row" style={{ marginTop: 10 }}>
          {view.bases.length > 1 && <select className="sel" value={base} onChange={(e) => { setBase(e.target.value); setName(suggest(e.target.value, view.base_names)) }} aria-label="Model to copy">{view.bases.map((b) => <option key={b} value={b}>{view.base_names?.[b] ?? b}</option>)}</select>}
          <input className="input" value={name} onChange={(e) => setName(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '-'))} placeholder="a name for the copy" aria-label="Name of the new copy" style={{ maxWidth: 260 }} />
          <button className="btn primary" aria-busy={is('create')} disabled={!name.trim() || !base || view.creating?.state === 'copying'} onClick={create}>{is('create') ? 'Starting the copy…' : 'Make a teachable copy'}</button>
        </div>}
        {view.creating?.state === 'copying' && <div className="working"><div className="what"><span><span className="pulse" />Copying {view.base_names?.[view.creating.base ?? ''] ?? view.creating.base} to <b>{view.creating.name}</b></span><span className="muted">{elapsed(copySince)} so far</span></div><div className="progress indeterminate"><i /></div><span className="small muted">Instant on a Mac; about 18 GB to write on Linux, a minute or two.</span></div>}
        {view.creating?.state === 'failed' && <p className="small bad">The copy failed: {view.creating.error}</p>}
      </div>}

      {served && <div className="card">
        <Steps at={job && job.status === 'running' ? 3 : facts && facts.length ? 2 : 1} />
        <div className={job?.status === 'running' ? 'dimmed' : undefined} aria-disabled={job?.status === 'running' || undefined}>
        <b>1. What should {served} remember?</b>
        <p className="small muted">Write facts, or paste a short text. Short, distinct facts work best ("My cat is called Pixel", "Our team meets on Thursdays"); a long text about one subject is the weak case.</p>
        <textarea className="input" rows={5} value={text} onChange={(e) => setText(e.target.value)} placeholder={'My cat is called Pixel.\nBramblewick Robotics deploys every Thursday.'} aria-label="Facts to teach" disabled={!!view?.busy} />
        <div className="actions"><button className={`btn ${facts && facts.length ? '' : 'primary'}`} aria-busy={finding} disabled={!text.trim() || finding || !!view?.busy} onClick={find}>{finding ? 'Reading it…' : facts && facts.length ? 'Read it again' : 'Next: see what it understood'}</button>{finding && <span className="small muted">The AI reads the text and answers each question it finds; about ten seconds a fact.</span>}</div>
        {facts && !!facts.length && <>
          <b style={{ display: 'block', marginTop: 16 }}>2. Here is what it understood</b>
          <p className="small muted">Each line is a fact it found, the question it will answer with it, and what it answers today. Untick a line to leave it out; edit a fact or a question if it read you wrong.</p>
          <div className="facts">
            <div className="facts-head" aria-hidden="true"><span>Teach</span><span>The fact</span><span>The question it will answer</span></div>
            {facts.map((f) => <div key={f.id} className="factrow">
              <span className="tick"><input type="checkbox" checked={f.approved} onChange={(e) => edit(f.id, { approved: e.target.checked })} aria-label={`Teach: ${f.fact}`} /></span>
              <input className="input" value={f.fact} onChange={(e) => edit(f.id, { fact: e.target.value })} aria-label="Fact" />
              <input className="input q" value={f.question} onChange={(e) => edit(f.id, { question: e.target.value })} aria-label="Question it answers" />
              <div className="sub">About {f.about_user ? 'you' : f.subject || '—'}{f.edited ? <span> · edited by you: not checked against the text</span> : !f.supported && <span className="bad"> · the text does not state this: left out unless you tick it</span>}</div>
              <div className="now"><b>Answers today:</b> {f.edited ? <span className="muted">(asked before your edit) </span> : null}{f.now ?? '—'}</div>
            </div>)}
          </div>
          <div className="actions"><button className="btn primary" aria-busy={is('teach')} disabled={!approved.length || running || !!view?.teachable || any} onClick={teach}>{is('teach') ? 'Starting the lesson…' : `Teach ${approved.length === 1 ? 'this fact' : `these ${approved.length} facts`} (about ${approved.length * 2}–${approved.length * 3} min)`}</button>
            <label className="small muted"><input type="checkbox" checked={checkAround} onChange={(e) => setCheckAround(e.target.checked)} /> ask it the test questions before and after</label></div>
          <p className="small muted" style={{ marginBottom: 0 }}>A point to go back to is kept first, so any lesson can be undone. Chat goes on while it learns. A fact it cannot answer afterwards is taken out whole.</p>
        </>}
        </div>
      </div>}

      {view?.recovering && !job && <div className="card">
        <b>A lesson on {view.recovering} was going when TensorFold Web stopped</b>
        <p className="small muted">Nothing changes the engine or the copy until the engine shows no fact being learned, or stops. If the engine was restarted by hand, or it is not learning, stop waiting.</p>
        <div className="actions"><button className="btn" aria-busy={is('cancel')} onClick={() => run('cancel', () => api.livingCancel().then(() => toast('No longer waiting for that lesson'), (e) => toast(`Not stopped: ${e}`, 'bad')))}>Stop waiting</button></div>
      </div>}
      {job && <div className="card">
        <div className="membar-head"><b>3. {job.status === 'running' ? <><span className="pulse" />{job.copy} is learning</> : job.status === 'done' ? 'Done' : job.status === 'rolled back' ? 'Undone' : job.status === 'cancelled' ? 'Stopped' : 'It did not work'}</b><span className="small muted">{job.status === 'running' ? `${elapsed(jobSince)} so far` : job.finished ? `${job.status}, ${secs(job.finished - job.started)}` : job.status}</span></div>
        {job.status === 'running' && <div className="working">
          <div className="what"><span>{PHASE[job.phase] ?? job.phase}{learningIndex >= 0 && job.items.length > 1 ? `: fact ${learningIndex + 1} of ${job.items.length}` : ''}</span><span className="muted">a fact takes 2 to 3 minutes; the page follows by itself</span></div>
          <div className="progress indeterminate"><i /></div>
        </div>}
        {job.status === 'done' && <p className="small">{(() => { const kept = job.items.filter((i) => i.state === 'kept' || i.state === 'partly kept'); const n = job.items.length
          return kept.length === n ? `It remembers ${n === 1 ? 'it' : `all ${n}`}.` : kept.length ? `It remembers ${kept.length} of ${n}; the rest was taken out.` : 'None of it was kept: the AI tried for several rounds, then could not answer its own test questions with it, so it took the lesson out whole. The copy is as it was before.' })()}
          {job.items.some((i) => i.state === 'missed') && <span className="muted"> A short sentence about yourself, with the name said once, works best; a name that is also an everyday word, or a long text, is harder. Trying again with other words sometimes works.</span>}</p>}
        <div className="tablewrap"><table className="tbl">
          <thead><tr><th>Fact</th><th>What happened</th><th>Answered before</th><th>Answers now</th></tr></thead>
          <tbody>{job.items.map((i) => <tr key={i.id}>
            <td>{i.fact}</td>
            <td className={i.state === 'kept' ? 'good' : i.state === 'missed' || i.state === 'failed' ? 'bad' : ''}>{i.state === 'learning' ? <><span className="pulse" />learning…</> : STATE[i.state]}{i.seconds ? ` (${secs(i.seconds)})` : ''}{i.message && <div className="small muted">{i.message}</div>}</td>
            <td className="small">{i.before ?? '—'}</td><td className="small">{i.after ?? '—'}</td>
          </tr>)}</tbody></table></div>
        {job.error && <p className="small bad">{job.error}</p>}
        {view?.busy && job.status === 'running' && (job.phase === 'learning' ? <>
          {ask?.kind === 'stopnow' && <Ask danger yes="Stop now" busy={is('cancel')} onYes={() => run('cancel', () => api.livingCancel(true).then(() => { setAsk(null); toast('Stopping now') }, (e) => toast(`Not stopped: ${e}`, 'bad')))} onNo={() => setAsk(null)}
            text={<>Stop now? The fact being learned is dropped. If the AI had already kept part of it, the copy goes back to the point kept before this lesson (about 20 seconds).</>} />}
          <div className="actions">
            <button className="btn" aria-busy={is('cancel')} disabled={!!job.items.find((i) => i.state === 'learning') === false} onClick={() => run('cancel', () => api.livingCancel().then(() => toast('Stopping after the fact being learned'), (e) => toast(`Not stopped: ${e}`, 'bad')))}>{is('cancel') ? 'Stopping…' : 'Stop after this fact'}</button>
            <button className="btn quiet" onClick={() => setAsk({ kind: 'stopnow' })}>Stop now</button>
            <span className="small muted">After this fact: it finishes, the ones after it are not taught. Now: it is dropped where it is.</span>
          </div>
        </> : <div className="actions">
          <button className="btn" aria-busy={is('cancel')} onClick={() => run('cancel', () => api.livingCancel().then(() => toast(job.phase === 'checking after' ? 'Stopping the questions' : 'Stopping before it learns'), (e) => toast(`Not stopped: ${e}`, 'bad')))}>{is('cancel') ? 'Stopping…' : job.phase === 'checking after' ? 'Skip the rest of the questions' : 'Stop before it learns'}</button>
          <span className="small muted">{job.phase === 'checking after' ? 'What it learned is kept; the comparison stops here.' : 'Nothing has been learned yet: stopping here changes nothing.'}</span>
        </div>)}
      </div>}

      {view && !!view.copies.length && <div className="card">
        <b>Teachable copies</b>
        {ask?.kind === 'remove' && <Ask danger yes={`Delete ${ask.name}`} busy={is(`remove:${ask.name}`)} onYes={() => remove(ask.name)} onNo={() => setAsk(null)}
          text={<>Delete <b>{ask.name}</b> and everything it learned? The model it came from is kept.</>} />}
        <div className="tablewrap"><table className="tbl">
          <thead><tr><th>Copy</th><th>Remembers</th><th /></tr></thead>
          <tbody>{view.copies.map((c) => <tr key={c.name}>
            <td><b>{c.name}</b>{served === c.name && <span className="chip run" style={{ marginLeft: 6 }}>Running</span>}{!c.learnable && <span className="small muted"> (cannot learn: {c.family})</span>}<div className="small muted">a copy of {view.base_names?.[c.base ?? ''] ?? (c.base ? shortName(c.base) : 'a model made by hand')}</div></td>
            <td>{c.taught} {c.taught === 1 ? 'fact' : 'facts'}{c.lessons > c.taught ? <span className="small muted"> ({c.lessons - c.taught} taught and taken out)</span> : null}</td>
            <td className="actions">{served !== c.name && <button className="btn sm" aria-busy={is(`serve:${c.model}`)} disabled={busy || stale || running} onClick={() => serve(c.model)}>{is(`serve:${c.model}`) ? 'Starting…' : 'Serve'}</button>}
              <button className="btn sm quiet" aria-busy={is(`card:${c.name}`)} onClick={() => downloadCard(c.name)}>Model card</button>
              <button className="btn sm quiet" aria-busy={is(`remove:${c.name}`)} disabled={served === c.name || running} onClick={() => setAsk({ kind: 'remove', name: c.name })}>{is(`remove:${c.name}`) ? 'Deleting…' : 'Delete'}</button></td>
          </tr>)}</tbody></table></div>
        <details className="more">
          <summary>Details: where they are, another copy, a copy from another machine</summary>
          <p className="small muted">Kept in <code>{view.root}</code>: {view.copies.map((c) => `${c.name} ${bytes(c.bytes, 1)}, made ${when(c.created)}`).join('; ')}. A copy is real files, never links into the model cache, so learning can never change the downloaded model.</p>
          <div className="row">
            <select className="sel" value={base} onChange={(e) => { setBase(e.target.value); setName(suggest(e.target.value, view.base_names)) }} aria-label="Model to copy">{view.bases.map((b) => <option key={b} value={b}>{view.base_names?.[b] ?? b}</option>)}</select>
            <input className="input" value={name} onChange={(e) => setName(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, '-'))} placeholder="a name for the copy" aria-label="Name of the new copy" style={{ maxWidth: 260 }} />
            <button className="btn" aria-busy={is('create')} disabled={!name.trim() || !base || view.creating?.state === 'copying'} onClick={create}>Make another copy</button>
          </div>
          {view.creating?.state === 'copying' && <p className="small">Copying {view.base_names?.[view.creating.base ?? ''] ?? view.creating.base} to <b>{view.creating.name}</b>…</p>}
          {view.creating?.state === 'failed' && <p className="small bad">The copy failed: {view.creating.error}</p>}
          {!!view.peers.length && <div className="row">
            <span className="small">Or get a copy taught on</span>
            <select className="sel" value={peer} onChange={(e) => { setPeer(e.target.value); setTheirs(null) }} aria-label="Machine to fetch a copy from"><option value="">another machine…</option>{view.peers.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
            <button className="btn sm" aria-busy={is('peers')} disabled={!peer} onClick={listTheirs}>List its copies</button>
          </div>}
          {theirs && (theirs.length ? <ul className="small">{theirs.map((x) => <li key={x.name}><b>{x.name}</b> from {x.base ?? '?'}, {x.taught} facts kept by its TensorFold Web <button className="btn sm" aria-busy={is(`fetch:${x.name}`)} disabled={running} onClick={() => setAsk({ kind: 'fetch', name: x.name })}>Get it</button></li>)}</ul> : <p className="small muted">That machine has no teachable copy.</p>)}
          {ask?.kind === 'fetch' && <AskName text="Its name here" initial={ask.name} ok="Fetch it" clean={cleanName} onOk={(as) => fetchCopy(ask.name, as)} onCancel={() => setAsk(null)} />}
        </details>
      </div>}

      {served && <Versions copy={served} busy={running} toast={toast} onChange={load} />}
      {served && <Checks copy={served} served busy={running} toast={toast} onChange={load} />}
      {served && <Measures copy={served} busy={running} peers={view?.peers ?? []} toast={toast} />}
      {served && learned && <div className="card">
        <div className="membar-head"><b>What {served} remembers</b><button className="btn sm" aria-busy={is('view')} onClick={() => run('view', openView)}>See it in 3D</button></div>
        {ask?.kind === 'forget' && <Ask yes="Forget it" busy={is(`forget:${ask.fact}`)} onYes={() => forget(ask.fact)} onNo={() => setAsk(null)}
          text={<>Make {served} forget <b>“{ask.fact}”</b>? A point of it as it is now is kept first, so this can be undone; it goes back to before that fact's lesson, then the facts kept since are taught again (2 to 3 minutes each).</>} />}
        {!learned.facts.length && !learned.lessons.length && <p className="small muted">Nothing yet.</p>}
        {!!learned.facts.length && <div className="tablewrap"><table className="tbl">
          <thead><tr><th>Fact</th><th>State</th><th>Neurons</th><th>Checked with</th><th /></tr></thead>
          <tbody>{learned.facts.map((f) => <tr key={f.id}><td>{f.text}</td><td className={f.state === 'learned' ? 'good' : f.state === 'missed' ? 'bad' : ''}>{STATE[f.state] ?? f.state}</td><td>{f.neurons}</td><td className="small">{f.questions.slice(0, 3).join(' · ')}</td>
            <td>{(() => { const taught = f.state === 'learned' ? taughtAs(f) : undefined
              return taught && <button className="btn sm quiet" aria-busy={is(`forget:${taught}`)} disabled={running} onClick={() => setAsk({ kind: 'forget', fact: taught })}>Forget</button> })()}</td></tr>)}</tbody>
        </table></div>}
        <p className="small muted" style={{ marginBottom: 0 }}>Forget takes one fact out: the copy goes back to before that lesson, then learns the facts kept since again (2 to 3 minutes each). TensorFold Web also keeps what it taught and the answers before and after ({learned.lessons.length} {learned.lessons.length === 1 ? 'entry' : 'entries'}).</p>
      </div>}
    </section>
  )
}
