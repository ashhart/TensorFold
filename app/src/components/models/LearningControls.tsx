import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { loadEngineModel, useEngine } from '../../lib/engine'
import { LEARNING_MODEL, learningBusy, learningError, refreshLearning, useLearning } from '../../lib/learning'
import { useStore } from '../../lib/store'
import { Button, Toggle } from '../ui'

export function LearningControls({ model, activeRuns, compact = false, onGraph }: { model?: string | null; activeRuns: number; compact?: boolean; onGraph?: () => void }) {
  const tx = useAuthoredText()
  const { snapshot: engine } = useEngine()
  const { snapshot, error: readError } = useLearning(true)
  const [pending, setPending] = useState(false)
  const lock = useRef(false)
  const [error, setError] = useState<string | null>(null)
  const [lesson, setLesson] = useState('')
  const setView = useStore((s) => s.setView)
  const installed = engine?.installed.some((item) => item.repo === LEARNING_MODEL)
  const localBusy = activeRuns > 0 || engine?.status.state === 'starting'
  const working = learningBusy(snapshot)
  const blocked = pending || working || localBusy
  const eligible = !model || model === LEARNING_MODEL
  const differentLoaded = !!engine?.status.model && engine.status.model !== LEARNING_MODEL && ['ready', 'starting'].includes(engine.status.state)
  const canTeach = snapshot?.active && engine?.status.state === 'ready' && engine.status.model === LEARNING_MODEL && eligible && !blocked && !!lesson.trim() && new TextEncoder().encode(lesson.trim()).byteLength <= 8192
  const action = async (run: () => Promise<unknown>, clear = false) => {
    if (lock.current) return
    lock.current = true; setPending(true); setError(null)
    try { await run(); if (clear) setLesson(''); await refreshLearning() }
    catch (e) { setError(learningError(e)) }
    finally { lock.current = false; setPending(false) }
  }
  const openGraph = () => { onGraph?.(); setView({ name: 'models', tab: 'Learning' }) }
  return (
    <section aria-label={tx("Living Weights setup")} className={compact ? 'p-3' : 'rounded-2xl border border-line bg-card p-4 sm:p-5'}>
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2"><h2 className="text-[14px] font-semibold text-ink">{tx("Living Weights")}</h2><span className="rounded-full bg-warn/10 px-2 py-0.5 text-[12px] text-ink-2">{tx("Experimental learner")}</span></div>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Teach selected facts to a private model copy on this Mac.")}</p>
        </div>
        <Toggle label={tx("Enable Living Weights")} checked={snapshot?.enabled ?? false} disabled={!snapshot || (!snapshot.supported && !snapshot.enabled) || blocked || (!snapshot.enabled && (!eligible || !installed || differentLoaded))} onChange={(enabled) => void action(() => api.engineSetLearning(enabled, LEARNING_MODEL))} />
      </div>
      {!snapshot && !readError && <p role="status" className="mt-3 text-[13px] text-ink-2">{tx("Checking learning support…")}</p>}
      {snapshot && !snapshot.supported && <p className="mt-3 text-[13px] text-ink-2">{tx("This build or hardware does not support Living Weights.")}</p>}
      {snapshot?.supported && (!installed || !eligible) && <div className="mt-3 rounded-xl bg-card-2 p-3 text-[13px] text-ink-2"><p>{tx("Requires Nemotron 3.5 Lightning MLX 4-bit; choose it in Models before enabling.")}</p><Button size="sm" className="mt-2" onClick={() => { onGraph?.(); setView({ name: 'models', tab: 'Models' }) }}>{tx("Choose learning model")}</Button></div>}
      {snapshot?.supported && installed && eligible && !snapshot.enabled && differentLoaded && <Button size="sm" className="mt-3" disabled={blocked} onClick={() => void action(() => loadEngineModel(LEARNING_MODEL))}>{tx("Load Nemotron for learning")}</Button>}
      {snapshot?.enabled && <p className="mt-3 text-[12px] font-medium text-ink-2">{tx("Learning model: Nemotron 3.5 Lightning · ")}{snapshot.privateCopy ? tx("Private copy") : tx("Preparing copy")}</p>}
      {snapshot?.supported && <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("Only lessons you submit below are taught; chats are not ingested automatically. Changes persist in the private copy, including when learning is turned off, and cannot be undone per fact. The original download stays untouched; the copy and learning graph are local files without app-level encryption.")}</p>}
      {localBusy && <p role="status" className="mt-3 text-[13px] text-ink-2">{tx("Finish or stop local runs before changing modes or teaching.")}</p>}
      {snapshot?.state === 'preparing' && <div role="status" className="mt-3 text-[13px] text-ink-2"><p>{tx("Preparing your private model copy…")}</p>{snapshot.progress && snapshot.progress.totalBytes > 0 && <progress className="mt-2 w-full accent-accent" max={snapshot.progress.totalBytes} value={snapshot.progress.copiedBytes} aria-label={tx("Preparing learning model")} />}</div>}
      {snapshot?.enabled && !snapshot.active && snapshot.privateCopy && !working && <div className="mt-3 flex flex-wrap items-center gap-2"><p className="text-[13px] text-ink-2">{tx("Private copy ready; load it to teach.")}</p><Button size="sm" disabled={blocked} onClick={() => void action(() => loadEngineModel(LEARNING_MODEL))}>{tx("Load learning model")}</Button></div>}
      {snapshot?.enabled && !snapshot.privateCopy && !working && !pending && <p role="alert" className="mt-3 text-[13px] text-ink-2">{tx("The private model copy is missing or unverified; preserve its files before repairing it.")}</p>}
      {snapshot?.enabled && <form className="mt-4" onSubmit={(e) => { e.preventDefault(); if (canTeach) void action(() => api.engineTeach(lesson.trim()), true) }}>
        <label className="block text-[13px] font-medium text-ink" htmlFor={compact ? 'chat-learning-lesson' : 'learning-lesson'}>{tx("Teach a fact")}</label>
        <textarea id={compact ? 'chat-learning-lesson' : 'learning-lesson'} value={lesson} onChange={(e) => setLesson(e.target.value)} maxLength={8192} rows={compact ? 3 : 2} disabled={pending || working} placeholder={tx("For example: Our demo project is named Cedar.")} className="mt-2 w-full resize-y rounded-xl border border-line bg-card-2 px-3 py-2.5 text-[13px] leading-relaxed text-ink outline-none focus:border-accent disabled:opacity-60" />
        <div className="mt-2 flex flex-wrap items-center gap-2"><Button type="submit" size="sm" variant="primary" disabled={!canTeach}>{pending ? tx("Working…") : tx("Teach this fact")}</Button>{snapshot.state === 'teaching' && <Button size="sm" disabled={pending} onClick={() => void action(() => api.engineCancelTeach())}>{tx("Stop lesson")}</Button>}<span className="text-[12px] text-ink-2">{tx("Usually 1–3 minutes per fact")}</span></div>
      </form>}
      {snapshot?.job && <div role="status" aria-live="polite" className="mt-3 rounded-xl border border-line bg-card-2 px-3 py-2 text-[13px] text-ink-2"><span className="font-medium capitalize text-ink">{snapshot.job.state === 'running' ? tx("Learning") : snapshot.job.state}</span> · {snapshot.job.learned} {tx("recalled · ")}{snapshot.job.missed} {tx("missed")}{snapshot.state === 'cancelling' && <p className="mt-1">{tx("Stopping at a safe boundary; facts already kept remain.")}</p>}</div>}
      {(error || readError || snapshot?.message) && <p role={error || readError || snapshot?.state === 'failed' ? 'alert' : 'status'} className="mt-3 break-words text-[13px] text-ink-2">{error || readError || snapshot?.message}</p>}
      {readError && <Button size="sm" className="mt-2" onClick={() => void refreshLearning()}>{tx("Retry connection")}</Button>}
      {compact && <Button size="sm" variant="ghost" className="mt-3" onClick={openGraph}>{tx("View learning graph")}</Button>}
    </section>
  )
}
