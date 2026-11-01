import { useEffect, useState } from 'react'
import { api, num, type BenchRun, type FleetMachine } from '../api'
import { engineSkew } from '../version'

// null: the same engine with drafts off; 'unchecked': an address chosen but not checked yet (the run waits);
// a Fleet machine goes by its id, this TensorFold Web reaching its engine with the key or token it keeps
export type KlReference = { url: string; key?: string; model: string } | { peer: string; model: string } | null | 'unchecked'

// the text a fixed-text run reads unless the person writes another
export const KL_TEXT = 'The lighthouse stood at the end of a narrow spit of rock, and for forty years the same family had kept its lamp burning. Every evening the keeper climbed the hundred and twelve steps, trimmed the wick, wound the clock that turned the lens, and wrote the weather in a book. Ships passing in the night never saw him, only the light, sweeping once every ten seconds across the water. When the lamp was finally made automatic, the last keeper left the book on the table, open at a page that read: wind from the west, sea calm, all well.'

/** The Fleet's machines, this one included, read once ``on`` is true (null until then, or while they are read). */
function useFleet(on: boolean): FleetMachine[] | null {
  const [all, setAll] = useState<FleetMachine[] | null>(null)
  useEffect(() => { if (on && all === null) api.fleet().then((f) => setAll(f.machines)).catch(() => setAll([])) }, [on, all])
  return all
}

/** The other machines of the Fleet, read once ``on`` is true (null until then, or while they are read). */
export function useFleetPeers(on: boolean): FleetMachine[] | null {
  const all = useFleet(on)
  return all && all.filter((m) => !m.self)
}

/** "Typed by hand" or one of the Fleet's machines, for another engine's address. */
export function PeerPick({ peers, value, onChange, label }: { peers: FleetMachine[] | null; value: string; onChange: (id: string) => void; label: string }) {
  return <select className="sel" value={value} onChange={(e) => onChange(e.target.value)} aria-label={label}>
    <option value="">Typed by hand</option>
    {peers === null ? <option disabled>Reading the Fleet…</option> : peers.map((p) => <option key={p.id} value={p.id}>{p.name}{p.engine?.model ? `: ${p.engine.model}` : p.reachable ? ', serving nothing' : ', not reachable'}</option>)}
  </select>
}

const nats = (x: number | null | undefined) => (x == null ? '—' : x === 0 ? '0' : x < 0.0001 ? x.toExponential(1) : x.toFixed(4))
const pc = (x: number | null | undefined, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`)

type Found = Awaited<ReturnType<typeof api.benchIdentify>>
// the form's whole state, kept by the Bench screen so it survives a change of mode and the run reads the same choice
// `peer`: a Fleet machine's id ('' when the address is typed by hand); `fixed`: both read `text`, every position scored
export type KlState = { kind: 'drafts' | 'address'; url: string; key: string; peer: string; found: Found | null; model: string; fixed: boolean; text: string }
export const KL_START: KlState = { kind: 'drafts', url: 'http://127.0.0.1:8000', key: '', peer: '', found: null, model: '', fixed: false, text: KL_TEXT }

export function klReference(s: KlState): KlReference {
  if (s.kind === 'drafts') return null
  if (!s.found || s.found.error || !s.model) return 'unchecked'
  return s.peer ? { peer: s.peer, model: s.model } : { url: s.found.url, key: s.key || undefined, model: s.model }
}

/** The text a fixed-text run sends, or undefined for a run on the suites' replies. */
export const klText = (s: KlState) => (s.fixed ? s.text : undefined)

/** The KL run's options: what the candidate is compared with, and whether it answers under load. */
export function KlOptions({ load, setLoad, state, setState }: { load: number; setLoad: (n: number) => void; state: KlState; setState: (f: (s: KlState) => KlState) => void }) {
  const { kind, url, key, peer, found, model, fixed, text } = state
  const fleetAll = useFleet(kind === 'address')
  const peers = fleetAll && fleetAll.filter((m) => !m.self)
  const hereVersion = fleetAll?.find((m) => m.self)?.tensorfold_version
  const peerVersion = peers?.find((m) => m.id === peer)?.tensorfold_version
  const pickKind = (k: 'drafts' | 'address') => setState((s) => ({ ...s, kind: k, found: null }))
  const edit = (patch: Partial<KlState>) => setState((s) => ({ ...s, ...patch, found: null }))
  const check = async (p = peer) => {
    const f = await (p ? api.benchIdentifyPeer(p) : api.benchIdentify(url, key || undefined)).catch((e) => ({ url, engine: null, version: null, models: [], error: String(e) }))
    // an answer for an address, key or machine changed since, or after drafts-off was chosen, is dropped
    setState((s) => (s.kind === 'address' && s.peer === p && (p || (s.url === url && s.key === key)) ? { ...s, found: f, model: f.models[0] ?? '' } : s))
  }
  const pickPeer = (p: string) => { edit({ peer: p }); if (p) check(p) }     // a machine is checked as soon as it is chosen
  return (
    <div className="aim">
      <span className="small"><b>Compared with</b></span>
      <label className="small"><input type="radio" name="klref" checked={kind === 'drafts'} onChange={() => pickKind('drafts')} /> The same engine with drafts off: exact decoding says the distributions are the same</label>
      <label className="small"><input type="radio" name="klref" checked={kind === 'address'} onChange={() => pickKind('address')} /> A reference engine at an address, such as the unquantized model on vLLM</label>
      {kind === 'address' && !found && <p className="small muted" style={{ margin: 0 }}>{peer ? 'Reading that machine\'s engine through its TensorFold Web…' : 'Check the reference before running: until then the run waits.'}</p>}
      {kind === 'address' && peer && engineSkew(hereVersion, peerVersion) && <p className="small warn-text" style={{ margin: 0 }}>That machine runs TensorFold {peerVersion}, this one {hereVersion}: a reply can part between releases at a near tie, so this KL measures the release as well as the machine. Put both on the same release for a like-for-like check.</p>}
      {kind === 'address' && <div className="aim-row">
        <PeerPick peers={peers} value={peer} onChange={pickPeer} label="A Fleet machine as the reference" />
        {!peer && <>
          <input className="input" value={url} onChange={(e) => edit({ url: e.target.value })} aria-label="Reference engine address" spellCheck={false} />
          <input className="input" type="password" value={key} onChange={(e) => edit({ key: e.target.value })} aria-label="Reference engine API key" placeholder="API key, if it wants one" autoComplete="off" />
        </>}
        <button type="button" className="btn sm" onClick={() => check()}>Check</button>
        {found && (found.error ? <span className="small bad">{found.error}</span>
          : <span className="small">{found.engine}{found.version ? ` ${found.version}` : ''}: <select className="sel" value={model} aria-label="Reference model"
              onChange={(e) => { const m = e.target.value; setState((s) => ({ ...s, model: m })) }}>{found.models.map((m) => <option key={m}>{m}</option>)}</select></span>)}
      </div>}
      {kind === 'address' && peer && <p className="small muted" style={{ margin: 0 }}>This TensorFold Web reaches that machine's engine through its TensorFold Web, with the token kept in Fleet: no key is typed or shown here.</p>}
      <span className="small"><b>What both read</b></span>
      <label className="small"><input type="radio" name="kltext" checked={!fixed} onChange={() => setState((s) => ({ ...s, fixed: false }))} /> Their own replies to the suites, compared where both wrote the same text</label>
      <label className="small"><input type="radio" name="kltext" checked={fixed} onChange={() => setState((s) => ({ ...s, fixed: true }))} /> A fixed text (every position, CUDA engines from TensorFold 1.0.8)</label>
      {fixed && <textarea className="input" rows={5} value={text} maxLength={20000} onChange={(e) => { const t = e.target.value; setState((s) => ({ ...s, text: t })) }} aria-label="The text both engines read" />}
      {fixed && <p className="small muted" style={{ margin: 0 }}>Both engines read this text once (<code>prompt_logprobs</code> on <code>/v1/completions</code>) and every position of it is compared. TensorFold on a Mac and its Python line refuse it, as other engines without it do; vLLM gives it too. Both must split the text into the same tokens.</p>}
      <label className="small">The candidate answers <select className="sel" value={load} onChange={(e) => setLoad(Number(e.target.value))} aria-label="Load while the candidate answers">
        <option value={0}>alone</option>{[3, 7, 15].map((n) => <option key={n} value={n}>with {n} other requests going</option>)}</select></label>
      {!fixed && <p className="small muted" style={{ margin: 0 }}>Both engines need log-probabilities on chat completions (TensorFold 1.0.5 on CUDA, vLLM, llama.cpp). Each position is compared where both wrote the same text, up to the first token where they part.</p>}
    </div>
  )
}

/** A KL run's figures, its largest divergences and each prompt's agreement. */
export function KlReport({ run }: { run: BenchRun }) {
  const k = run.kl
  if (!k || !k.positions) return null
  const ref = run.kl_reference
  const text = run.kl_text != null
  return (
    <>
      <p className="small muted" style={{ marginTop: 0 }}>{text ? 'Every position of a fixed text, ' : ''}{run.tags?.peer ? `${run.tags.peer}'s engine ` : ''}{text || run.tags?.peer ? 'against' : 'Against'} {ref?.kind === 'address' ? <><b>{ref.model}</b> {run.tags?.reference_peer ? <>on {run.tags.reference_peer}</> : <>at {ref.url}</>}</> : 'the same engine with drafts off'}{run.kl_load && run.kl_load_verified === false ? `; ${run.kl_load} background requests were asked, but the load is not established (see the notes), so this run does not show the candidate under load` : run.kl_load ? `, the candidate answering with ${run.kl_load} other requests going${run.kl_load_done?.full_share != null ? ` (all of them in flight ${Math.round(run.kl_load_done.full_share * 100)}% of the time; ${run.kl_load_done.ok} background replies)` : ''}` : ''}.</p>
      <dl className="figures big">
        <div><dt>Mean KL</dt><dd className={k.mean_kl === 0 ? 'good' : undefined}>{nats(k.mean_kl)} <small>nats</small></dd></div>
        <div><dt>KL p99</dt><dd>{nats(k.p99_kl)}</dd></div>
        <div><dt>Largest</dt><dd>{nats(k.max_kl)}</dd></div>
        <div><dt>Same top token</dt><dd>{pc(k.same_top)}</dd></div>
        <div><dt>{text ? 'Same distributions' : 'Identical replies'}</dt><dd>{pc(text ? k.same_distribution : k.identical, text ? 1 : 0)}</dd></div>
        <div><dt>Positions</dt><dd>{num(k.positions)}</dd></div>
      </dl>
      {k.worst.some((w) => w.kl > 0) && <div className="tablewrap"><table className="tbl">
        <thead><tr><th>Largest divergences</th><th>Token</th>{text && <th>In the text</th>}<th>{text ? 'Reference expects' : 'Reference'}</th><th>{text ? 'Candidate expects' : 'Candidate'}</th><th>KL</th></tr></thead>
        <tbody>{k.worst.filter((w) => w.kl > 0).map((w, i) => <tr key={i}><td>{w.prompt}</td><td>{w.at}</td>{text && <td><code>{JSON.stringify(w.token ?? '')}</code></td>}<td><code>{JSON.stringify(w.ref)}</code></td><td><code>{JSON.stringify(w.cand)}</code></td><td>{nats(w.kl)}</td></tr>)}</tbody>
      </table></div>}
      {!text && <div className="tablewrap"><table className="tbl">
        <thead><tr><th>Prompt</th><th>Compared</th><th>Parted at</th><th>Same top</th></tr></thead>
        <tbody>{run.results.filter((r) => r.kl).map((r) => <tr key={`${r.suite}/${r.prompt}`}><td>{r.suite}/{r.prompt}</td><td>{r.kl!.positions} of {r.m?.tokens ?? '—'}</td>
          <td>{r.kl!.differs_at == null ? (r.kl!.identical ? 'identical' : 'lengths differ') : `token ${r.kl!.differs_at}`}</td><td>{r.kl!.positions ? pc(r.kl!.agree / r.kl!.positions) : '—'}</td></tr>)}</tbody>
      </table></div>}
      {text && <details className="small"><summary>The text both read ({num(run.kl_text!.length)} characters{k.unscored ? `; ${k.unscored} positions without probabilities on one side, left out` : ''})</summary><p>{run.kl_text}</p></details>}
      <p className="small muted">A lower bound of the full KL, from the top {k.top} log-probabilities of each side: what their lists establish, nothing guessed about the rest. {text ? 'Both read the same text, so every position after the first compares; "expects" is each side\'s most likely token there.' : 'Positions after the first different token are not compared: the two replies no longer read the same text.'}</p>
    </>
  )
}
