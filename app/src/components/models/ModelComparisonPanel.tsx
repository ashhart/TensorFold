import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { statusText } from '../../lib/preferences/display'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { repoDisplayName } from '../../lib/turn-stats-line'
import type { EngineInstalledModel, EngineModelComparisonSnapshot } from '../../lib/types'
import { ModelsDrawer } from './ModelsDrawer'

const DEFAULT_PROMPT = 'Explain in three short sentences how a rainbow forms.'
const terminal = (snapshot: EngineModelComparisonSnapshot) => !['running', 'restoring'].includes(snapshot.state)
const message = (error: unknown) => error instanceof Error ? error.message : authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "The comparison could not finish.")
const metric = (value: number | null, unit: string) => value == null ? 'Not reported' : `${Number(value.toFixed(2))} ${unit}`

export function ModelComparisonPanel({ models, activeModel, locked, onClose }: { models: EngineInstalledModel[]; activeModel: string | null; locked: boolean; onClose: () => void }) {
  const tx = useAuthoredText()
  const available = models.filter((model) => model.fitsMemory === true)
  const first = available.find((model) => model.repo === activeModel)?.repo ?? available[0]?.repo ?? ''
  const [left, setLeft] = useState(first)
  const [right, setRight] = useState(available.find((model) => model.repo !== first)?.repo ?? '')
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [snapshot, setSnapshot] = useState<EngineModelComparisonSnapshot | null>(null)
  const [running, setRunning] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const job = useRef<string | null>(null)
  const mounted = useRef(true)
  const lastPrompt = useRef(DEFAULT_PROMPT)
  useEffect(() => {
    mounted.current = true
    const unsubscribe = api.onEngineEvent((event) => {
      if (event.kind !== 'comparison-cleared' || !mounted.current) return
      job.current = null; lastPrompt.current = DEFAULT_PROMPT
      setPrompt(DEFAULT_PROMPT); setSnapshot(null); setRunning(false); setCancelling(false); setError(tx("Comparison data was cleared with local memory."))
    })
    return () => { mounted.current = false; unsubscribe(); const id = job.current; if (id) void api.engineCancelComparison(id).catch(() => {}) }
  }, [])
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => {
      const id = job.current
      if (!id) return
      void api.engineComparisonStatus(id).then((value) => { if (mounted.current && job.current === id && value) setSnapshot((previous) => previous?.id === id && previous.revision > value.revision ? previous : value) }).catch((value) => { if (mounted.current && job.current === id) setError(message(value)) })
    }, 400)
    return () => window.clearInterval(timer)
  }, [running])
  const valid = available.some((model) => model.repo === left) && available.some((model) => model.repo === right) && left !== right && prompt.trim().length > 0 && new TextEncoder().encode(prompt).length <= 1024
  const start = async () => {
    if (locked || job.current || !valid) return
    const id = `compare_${crypto.randomUUID().replaceAll('-', '')}`
    job.current = id; lastPrompt.current = prompt; setRunning(true); setCancelling(false); setSnapshot(null); setError(null)
    try { const result = await api.engineCompareModels({ id, models: [left, right], prompt }); if (mounted.current && job.current === id) setSnapshot(result) }
    catch (value) { if (mounted.current && job.current === id) setError(message(value)) }
    finally { if (job.current === id) { job.current = null; if (mounted.current) { setRunning(false); setCancelling(false) } } }
  }
  const cancel = async () => {
    const id = job.current
    if (!id || cancelling) return
    setCancelling(true)
    try { await api.engineCancelComparison(id) }
    catch (value) { if (mounted.current && job.current === id) { setError(message(value)); setCancelling(false) } }
  }
  const close = () => {
    if (job.current) return
    if (prompt !== DEFAULT_PROMPT && prompt !== lastPrompt.current && !window.confirm(tx("Close this comparison and discard the unsent prompt?"))) return
    onClose()
  }
  return <ModelsDrawer title={tx("Compare models")} labelId="model-comparison-title" onClose={close} closeDisabled={running} description={tx("Two local models, one prompt. Read both answers and decide what works for you.")} footer={<div className="flex items-center justify-between gap-3"><button type="button" onClick={close} disabled={running} className="text-[13px] text-ink-2 disabled:opacity-40">{tx("Close")}</button>{running ? <button type="button" onClick={cancel} disabled={cancelling} className="rounded-lg border border-line px-4 py-2 text-[13px] text-ink-2 disabled:opacity-40">{cancelling ? tx("Stopping and restoring…") : tx("Cancel comparison")}</button> : <button type="button" onClick={start} disabled={locked || !valid} className="rounded-lg bg-accent px-4 py-2 text-[13px] text-accent-ink disabled:opacity-40">{tx("Compare models")}</button>}</div>}>
    <p className="text-[12px] leading-relaxed text-ink-3">{tx("Runs sequentially on your Mac with the same 256-token reply limit and temperature 0; saved context profiles can differ. No chat history, files, tools or learning are used, and your previous loaded model is restored afterwards.")}</p>
    {available.length < 2 && <p role="status" className="mt-3 rounded-lg bg-surface p-3 text-[13px] text-ink-2">{tx("Install two models confirmed to fit this Mac to compare them.")}</p>}
    {locked && !running && <p role="status" className="mt-3 text-[13px] text-ink-2">{tx("Finish local work and turn Living Weights off before starting a comparison.")}</p>}
    <div className="mt-5 grid gap-3 sm:grid-cols-2">{(['A', 'B'] as const).map((label, index) => <label key={label} className="text-[12px] text-ink-3">{tx("Model ")}{label}<select aria-label={tx("Comparison model {value0}", { value0: label })} value={index ? right : left} onChange={(event) => index ? setRight(event.target.value) : setLeft(event.target.value)} disabled={running} className="mt-1 w-full rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink"><option value="">{tx("Choose model")}</option>{available.map((model) => <option value={model.repo} title={model.repo} key={model.repo}>{model.title ?? repoDisplayName(model.repo)}</option>)}</select></label>)}</div>
    <label className="mt-4 block text-[12px] text-ink-3">{tx("Same prompt for both models")}<textarea aria-label={tx("Comparison prompt")} rows={4} value={prompt} onChange={(event) => setPrompt(event.target.value)} disabled={running} className="mt-1 w-full resize-y rounded-lg border border-line bg-card p-3 text-[13px] leading-relaxed text-ink" /></label>
    <p className="mt-1 text-[11px] text-ink-3">{new TextEncoder().encode(prompt).length} {tx("/ 1,024 bytes · Results stay in this panel and are not saved to chats.")}</p>
    {error && <p role="alert" className="mt-3 rounded-lg border border-line p-3 text-[13px] text-ink-2">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    {snapshot && <section aria-label={tx("Comparison results")} className="mt-5 space-y-3"><p role="status" aria-live="polite" className="text-[12px] text-ink-2">{snapshot.state === 'restoring' ? tx("Restoring your previous model…") : snapshot.state === 'running' ? tx("Comparison in progress…") : snapshot.state === 'cancelled' ? tx("Comparison cancelled.") : snapshot.state === 'failed' ? tx("Comparison finished with an issue.") : tx("Comparison complete.")}</p>{snapshot.rows.map((row, index) => <article key={row.model} className="rounded-xl border border-line bg-card p-4"><div className="flex items-start justify-between gap-2"><h3 title={row.model} className="min-w-0 break-words text-[13px] font-medium text-ink">{index ? tx("B") : tx("A")} · {models.find((model) => model.repo === row.model)?.title ?? repoDisplayName(row.model)}</h3><span className="shrink-0 text-[11px] text-ink-3">{statusText(row.state, tx)}</span></div>{row.result && <>{row.result.finishReason && /length|max_tokens|token_limit/.test(row.result.finishReason) && <p className="mt-2 text-[11px] text-ink-3">{tx("Reply limit reached (256 tokens); this answer may be incomplete.")}</p>}<p className="mt-3 whitespace-pre-wrap break-words text-[13px] leading-relaxed text-ink-2">{row.result.text}</p><dl className="mt-4 grid grid-cols-2 gap-2 text-[11px] text-ink-3"><div><dt>{tx("Load / setup")}</dt><dd>{metric(row.result.loadMs, 'ms')}</dd></div><div><dt>{tx("Response time")}</dt><dd>{metric(row.result.elapsedMs, 'ms')}</dd></div><div><dt>{tx("First token")}</dt><dd>{metric(row.result.firstTokenMs, 'ms')}</dd></div><div><dt>{tx("Decode speed")}</dt><dd>{metric(row.result.decodeTokensPerSecond, 'tokens/s')}</dd></div><div><dt>{tx("Output tokens")}</dt><dd>{metric(row.result.completionTokens, 'tokens')}</dd></div><div><dt>{tx("Prompt tokens")}</dt><dd>{metric(row.result.promptTokens, 'tokens')}</dd></div><div><dt>{tx("Context profile")}</dt><dd>{row.result.profile?.contextTokens?.toLocaleString(currentIntlLocale()) ?? tx("Not reported")}</dd></div></dl></>}{row.error && <p role="alert" className="mt-3 text-[12px] text-ink-2">{<OriginalDiagnostic>{row.error}</OriginalDiagnostic>}</p>}</article>)}{terminal(snapshot) && <p role={snapshot.restoreError ? 'alert' : 'status'} className="text-[12px] text-ink-3">{snapshot.restoreError ? tx("The previous model could not be restored: {value0} Choose a model to resume.", { value0: snapshot.restoreError }) : snapshot.restored ? snapshot.previousModel ? tx("Your previous model has been restored; your chat selection is unchanged.") : tx("The engine has returned to its previously unloaded state; your chat selection is unchanged.") : tx("The comparison stopped before changing the engine.")}</p>}<p className="text-[11px] text-ink-3">{tx("One short run is a sample, not a quality ranking; speed also depends on context, cache and model loading.")}</p></section>}
  </ModelsDrawer>
}
