// Replay a conversation on another configuration, and two conversations compared turn by turn.
import { useEffect, useRef, useState } from 'react'
import { api, type ReplyComparison, type ReplayProgress, type ReplayTarget } from '../api'
import type { Ctx } from '../ctx'

const pct = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(Math.round(x * 100))}%`
// B's change against A, turned round: what A gains over B ("B is 38% slower" is "A is 61% faster")
const gain = (bChange: number) => 1 / (1 + bChange) - 1
const speedWord = (x: number) => `${Math.abs(Math.round(x * 100))}% ${x >= 0 ? 'faster' : 'slower'}`
const tps = (x: number | null) => (x == null ? '—' : String(Math.round(x)))

/** Waits for the running replay to end; its conversation's id, or an error saying why it stopped. */
async function replayed(stopped: () => boolean, onProgress: (p: ReplayProgress) => void, every = 1000): Promise<string> {
  for (;;) {
    await new Promise((r) => setTimeout(r, every))
    if (stopped()) throw new Error('stopped')
    const r = await api.replayProgress().catch(() => null)
    if (!r) continue                                                      // one failed poll is not the end
    if (!r.current) throw new Error('the replay is gone: TensorFold Web may have restarted')
    onProgress(r.current)
    if (r.current.status === 'done') return r.current.id
    if (r.current.status !== 'running') throw new Error(`the replay ${r.current.status}${r.current.error ? `: ${r.current.error}` : ''}`)
  }
}

/** The speed-up checked on this conversation, in one click: its questions sent twice without randomness, with the
 *  speed-up and without it, then the two compared reply by reply. */
type Written = { thinking: string; budget: number | null; system: string; max_tokens: number | null; temperature: number | null } | null
const said = (r: NonNullable<Written>) => [r.thinking === 'off' ? 'thinking off' : r.thinking === 'on' ? `thinking on${r.budget ? `, up to ${r.budget.toLocaleString('en-US')} tokens` : ''}` : `thinking as the model decides${r.budget ? `, up to ${r.budget.toLocaleString('en-US')} tokens` : ''}`,
  r.system ? `the system prompt “${r.system.length > 60 ? `${r.system.slice(0, 60)}…` : r.system}”` : 'no system prompt',
  r.max_tokens ? `replies up to ${r.max_tokens.toLocaleString('en-US')} tokens` : null].filter(Boolean).join(', ')

export function SpeedupCheck({ id, model, onDone, onClose, toast, written }: { id: string; model: string; onDone: (withId: string, withoutId: string) => void; onClose: () => void; toast: Ctx['toast']; written: Written[] }) {
  const [step, setStep] = useState<0 | 1 | 2>(0)
  const [prog, setProg] = useState<ReplayProgress | null>(null)
  const [watch, setWatch] = useState(false)                           // the two answers shown as they are written
  // what each side wrote, turn by turn: the side being written fills from the live progress
  const [seen, setSeen] = useState<Record<1 | 2, Record<number, { q: string; text: string; reasoning: string }>>>({ 1: {}, 2: {} })
  const stop = useRef(false)
  const going = useRef(false)                                          // a replay of ours is running on the server
  const follow = (side: 1 | 2) => (p: ReplayProgress) => {
    setProg(p)
    if (p.turn != null) setSeen((s) => ({ ...s, [side]: { ...s[side], [p.turn!]: { q: p.question ?? '', text: p.live_text ?? '', reasoning: p.live_reasoning ?? '' } } }))
  }
  // closed or left mid-check: the server's replay is stopped too, so the next check is not refused as "already going"
  useEffect(() => () => { stop.current = true; if (going.current) api.replayCancel().catch(() => {}) }, [])
  const go = async () => {
    stop.current = false
    going.current = true
    try {
      setSeen({ 1: {}, 2: {} })
      setStep(1)
      setProg((await api.conversationReplay(id, { kind: 'served', drafts: true })).progress)
      const withId = await replayed(() => stop.current, follow(1), watch ? 300 : 1000)
      setStep(2)
      setProg((await api.conversationReplay(id, { kind: 'served', drafts: false })).progress)
      const withoutId = await replayed(() => stop.current, follow(2), watch ? 300 : 1000)
      going.current = false
      toast('Checked: the two answers are side by side')
      onDone(withId, withoutId)
    } catch (e) {
      going.current = false
      if (String(e).includes('stopped')) return
      toast(String((e as Error).message ?? e), 'bad'); setStep(0)
    }
  }
  const cancel = async () => { stop.current = true; going.current = false; await api.replayCancel().catch(() => {}); setStep(0); setProg(null); toast('Stopped') }
  return (
    <div className="replayform">
      <div className="membar-head"><b className="small">Verify speed-up &amp; exactness on this conversation</b><button className="btn sm quiet" onClick={onClose}>Close</button></div>
      <p className="small muted">The questions of this conversation are sent twice to {model || 'the served model'}: once with the speed-up, once without. The two answers are then put side by side. If the speed-up is exact, they are the same, token for token; the comparison also shows how much faster it was.</p>
      {(() => {
        const known = written.filter((w): w is NonNullable<Written> => !!w)
        const kinds = [...new Set(known.map(said))]
        return <p className="small">{known.length
          ? <>Each answer is written again <b>with the settings it was first written with</b> ({kinds.length === 1 ? kinds[0] : `${kinds.length} different settings across the conversation`}), whatever the options show now. One thing differs on purpose: <b>no randomness</b>{known.some((w) => w.temperature !== 0) ? ' (the original answers were drawn at random, so they are not compared; the two new ones are)' : ''}.</>
          : <>These answers were written before TensorFold Web recorded their settings: they are written again with the conversation's options as they are now, and <b>without randomness</b>.</>}
          {known.length > 0 && known.length < written.length ? ' Older answers in it, with no record, use the options as they are now.' : ''}</p>
      })()}
      {step === 0 ? <div className="actions"><button className="btn sm primary" onClick={go}>Check it</button>
        <label className="small"><input type="checkbox" checked={watch} onChange={(e) => setWatch(e.target.checked)} /> Watch the two answers being written</label></div>
        : <div className="small">
          <div className="working"><div className="what"><span><span className="pulse" />{step === 1 ? '1 of 2: with the speed-up' : '2 of 2: without the speed-up (slower)'}{prog ? `, question ${Math.min(prog.done + 1, prog.total)} of ${prog.total}` : ''}</span></div>
            <div className="progress"><i style={{ width: `${((step - 1) * 50) + (prog ? (prog.done / Math.max(1, prog.total)) * 50 : 0)}%` }} /></div></div>
          <button className="btn sm" onClick={cancel}>Stop</button>
          {watch && <div className="sides watch">
            {([1, 2] as const).map((side) => <div key={side}>
              <span className="small muted">{side === 1 ? 'With the speed-up' : 'Without the speed-up'}{step === side ? ' (being written)' : step > side ? ' (done)' : ' (next)'}</span>
              {Object.entries(seen[side]).map(([t, x]) => <div key={t}>
                {Object.keys(seen[side]).length > 1 && <div className="small"><b>{x.q.slice(0, 120)}</b></div>}
                {x.reasoning && <pre className="thinking">{x.reasoning}</pre>}
                <pre>{x.text || (step === side ? '…' : '')}</pre>
              </div>)}
            </div>)}
          </div>}
        </div>}
    </div>
  )
}

/** Where to replay: the served engine with drafts on or off, or another engine at an address. */
export function ReplayForm({ id, model, onDone, toast }: { id: string; model: string; onDone: (replayId: string) => void; toast: Ctx['toast'] }) {
  const [kind, setKind] = useState<'on' | 'off' | 'address'>('off')
  const [url, setUrl] = useState('http://127.0.0.1:8081')
  const [key, setKey] = useState('')
  const [other, setOther] = useState(model)
  const [prog, setProg] = useState<ReplayProgress | null>(null)
  // a replay already going (started here before, or from the speed-up check) shows its bar and its Stop at once
  useEffect(() => { api.replayProgress().then((r) => { if (r.current?.status === 'running') setProg(r.current) }).catch(() => {}) }, [])
  useEffect(() => {
    if (!prog || prog.status !== 'running') return
    const t = setInterval(async () => {
      const r = await api.replayProgress().catch(() => null)
      if (!r?.current) return
      setProg(r.current)
      if (r.current.status !== 'running') { clearInterval(t); if (r.current.status === 'done') { toast('Replay done'); onDone(r.current.id) } else toast(`Replay ${r.current.status}${r.current.error ? `: ${r.current.error}` : ''}`, 'bad') }
    }, 1000)
    return () => clearInterval(t)
  }, [prog?.status])  // a new poll only when a replay starts
  const go = async () => {
    const target: ReplayTarget = kind === 'address' ? { kind: 'address', url, key: key || undefined, model: other } : { kind: 'served', drafts: kind === 'on' }
    try { const r = await api.conversationReplay(id, target); setProg(r.progress) } catch (e) { toast(String(e), 'bad') }
  }
  return (
    <div className="replayform">
      <b className="small">Replay this conversation</b>
      <label className="small"><input type="radio" checked={kind === 'off'} onChange={() => setKind('off')} /> on {model || 'the served model'}, <b>without the speed-up</b> <span className="muted">(word by word: slower, and it shows the speed-up changes nothing in the answers)</span></label>
      <label className="small"><input type="radio" checked={kind === 'on'} onChange={() => setKind('on')} /> on {model || 'the served model'}, <b>with the speed-up</b> <span className="muted">(as it normally runs: to compare with the original, or after a change)</span></label>
      <label className="small"><input type="radio" checked={kind === 'address'} onChange={() => setKind('address')} /> on <b>another engine</b> <span className="muted">(Ollama, llama.cpp, another machine: the same questions, its answers and speed)</span></label>
      {kind === 'address' && <div className="aim-row">
        <input className="input" value={url} onChange={(e) => setUrl(e.target.value)} aria-label="Engine address" />
        <input className="input" type="password" value={key} onChange={(e) => setKey(e.target.value)} placeholder="its key, if any" aria-label="Its key" />
        <input className="input" value={other} onChange={(e) => setOther(e.target.value)} placeholder="model" aria-label="Model" />
      </div>}
      <p className="small muted">Each question is sent again with the original's earlier turns, without randomness (temperature 0), so the two can be compared reply by reply with “Compare its replies with…”. The speed-up has the AI guess a few words ahead and keep a guess only when it is exactly what it would have written. The replay is kept as a conversation of its own.</p>
      {prog && prog.status === 'running'
        ? <div className="small"><div className="progress"><i style={{ width: `${(prog.done / Math.max(1, prog.total)) * 100}%` }} /></div>{prog.done} of {prog.total} questions <button className="btn sm quiet" onClick={() => api.replayCancel().catch((e) => toast(`Not stopped: ${e}`, 'bad'))}>Stop</button></div>
        : <button className="btn sm primary" onClick={go}>Replay</button>}
    </div>
  )
}

/** Two conversations side by side, question by question: replies, speed, and whether the tokens are the same. */
/** Two replies side by side: what they share from the start in green, then, from where they part, the rest of each
 *  marked. The comparison is by text: replies with the same tokens are the same text, so the green part is the run
 *  of identical tokens up to the first one that differs. */
function Sides({ a, b, labels = ['A', 'B'] }: { a: string; b: string; labels?: [string, string] }) {
  let n = 0
  while (n < a.length && n < b.length && a[n] === b[n]) n++
  const same = a === b
  const side = (label: string, text: string) => (
    <div><span className="small muted">{label}</span>
      <pre><mark className="same">{text.slice(0, n)}</mark>{!same && n < text.length && <mark className="diff">{text.slice(n)}</mark>}</pre></div>
  )
  return (
    <>
      <p className="small" style={{ margin: '6px 0 0' }}>{same ? <span className="good">Identical from the first character to the last.</span>
        : n === 0 ? <span className="bad">Different from the first character.</span>
        : <><span className="good">Identical for {n.toLocaleString('en-US')} characters</span>, then <span className="bad">they part</span> (in red).</>}</p>
      <div className="sides">{side(labels[0], a)}{side(labels[1], b)}</div>
    </>
  )
}

export function ReplyCompare({ a, b, onClose, speedup, names = ['A', 'B'] }: { a: string; b: string; onClose: () => void; speedup?: boolean; names?: [string, string] }) {
  const [na, nb] = names
  const [c, setC] = useState<ReplyComparison | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const top = useRef<HTMLDivElement>(null)
  useEffect(() => { api.conversationCompare(a, b).then(setC).catch((e) => setErr(String(e))) }, [a, b])
  useEffect(() => { top.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }) }, [c, err])   // where the eye is not
  if (err) return <div ref={top} className="notice bad">{err} <button className="btn sm quiet" onClick={onClose}>Close</button></div>
  if (!c) return <p ref={top} className="muted small">Comparing the two conversations…</p>
  const sp = c.speed
  return (
    <div className="replycompare" ref={top}>
      <div className="compare-bar"><b>Two conversations, reply by reply</b><button className="btn sm" onClick={onClose}>Back to the conversation</button></div>
      <p className="small muted" style={{ margin: '4px 0 8px' }}>{speedup ? <>The same questions, written twice without randomness: on the left with speculative decoding (the speed-up: the AI drafts a few tokens ahead and keeps each only if it is the one it would have written), on the right without it, one token at a time.</> : <>The same questions, answered on the left by <b>{na}</b> and on the right by <b>{nb}</b>: it shows whether the answers are the same and which was faster.</>}</p>
      {c.turns.length > 0 && c.checked === 0 && <p className="verdict-line">Whether the tokens are the same cannot be said here: {speedup ? 'the engine did not confirm the speed-up was off on the right, or gave no token digests' : 'the engine gave no token digests, or a reply failed'}. The texts are below.</p>}
      {speedup && c.checked > 0 && <p className={`verdict-line ${c.same_tokens === c.checked ? 'good' : 'bad'}`}>{c.same_tokens === c.checked
        ? `The speed-up changed nothing: the same answer${c.checked === 1 ? '' : 's'}, token for token${c.checked < c.turns.length ? ` (${c.checked} of ${c.turns.length} could be checked)` : ''}.`
        : `The speed-up changed ${c.checked - c.same_tokens} of ${c.checked} answers checked: it is not exact here.`}{sp.median_change != null && sp.median_change < -0.005 ? ` With it, the answers came ${pct(gain(sp.median_change)).replace('+', '')} faster (${(1 / (1 + sp.median_change)).toFixed(1)}×).` : sp.median_change != null && sp.median_change > 0.005 ? ' It was not faster here.' : ''}</p>}
      {!speedup && c.checked > 0 && <p className={`verdict-line ${c.same_tokens === c.checked ? 'good' : 'bad'}`}>{c.same_tokens === c.checked
        ? `Same answers, token for token, in ${c.checked === 1 ? 'the reply checked' : `all ${c.checked} replies checked`}${c.checked < c.turns.length ? ` (of ${c.turns.length})` : ''}.`
        : `${c.checked - c.same_tokens} of ${c.checked} ${c.checked === 1 ? 'reply checked differs' : 'replies checked differ'}.`}{sp.median_change != null && Math.abs(sp.median_change) >= 0.005 ? ` On the right, ${nb} was ${speedWord(sp.median_change)} at the median.` : ''}</p>}
      <ul className="findings">
        {sp.median_change != null && (speedup
          ? <li><b>Speed:</b> {Math.abs(sp.median_change) < 0.005 ? 'as fast with speculative decoding as without' : `${pct(gain(sp.median_change))} with speculative decoding, at the median`}, {sp.too_few ? 'too few replies to call it certain' : sp.within_noise ? 'within noise' : 'a real difference'} (faster on {sp.b_slower} of {sp.b_faster + sp.b_slower} replies).</li>
          : <li><b>Speed:</b> {nb} {Math.abs(sp.median_change) < 0.005 ? `was as fast as ${na}` : `was ${speedWord(sp.median_change)} than ${na} at the median`}, {sp.within_noise ? 'within noise' : 'a real difference'} (faster on the right in {sp.b_faster} of {sp.b_faster + sp.b_slower} replies).</li>)}
        <li><b>Replies:</b> {c.same_tokens} of {c.checked} checked with the same tokens{c.checked < c.turns.length ? ` (${c.turns.length - c.checked} could not be checked)` : ''}, {c.same_text} of {c.turns.length} word for word the same.</li>
      </ul>
      {c.turns.map((t, i) => (
        <details key={i} className="turn" open={!t.same_text}>
          <summary><b>{t.question.slice(0, 90)}</b> <span className="muted small">{speedup
            ? <>{tps(t.a.tps)} tok/s with, {tps(t.b.tps)} without{t.change != null ? ` (${pct(gain(t.change))} with speculative decoding)` : ''}</>
            : <>{tps(t.a.tps)} → {tps(t.b.tps)} tok/s{t.change != null ? ` (${pct(t.change)})` : ''}</>} · {t.same_tokens ? 'same tokens' : t.same_text ? (t.same_tokens === null ? 'same text, tokens not checked' : 'same text') : 'different replies'}</span></summary>
          <Sides a={t.a.text} b={t.b.text} labels={speedup ? ['With speculative decoding (the speed-up)', 'Without it (one token at a time)'] : [na, nb]} />
        </details>
      ))}
    </div>
  )
}
