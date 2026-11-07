import { currentIntlLocale } from '../../../lib/preferences/current-locale'
import { statusText } from '../../../lib/preferences/display'
import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import type { EngineInstalledModel } from '../../../lib/types'
import { Button, Toggle } from '../../ui'
import { ModelsDrawer } from '../ModelsDrawer'
import { BenchmarkResults } from './BenchmarkResults'
import { BenchmarkMetadata } from './BenchmarkMetadata'
import { api } from '../../../lib/ipc'
import { benchmarkActive, benchmarkBasicSummary, benchmarkDefaultModel, cachedBenchmarkModels } from './benchmark-display'
import { useModelBenchmark } from './useModelBenchmark'

export function ModelBenchmarkDrawer({ models, activeModel, preferredModel, blockedReason, onClose }: { models: EngineInstalledModel[]; activeModel: string | null; preferredModel?: string | null; blockedReason?: string | null; onClose: () => void }) {
  const tx = useAuthoredText()
  const available = cachedBenchmarkModels(models)
  const [model, setModel] = useState(() => benchmarkDefaultModel(models, preferredModel ?? activeModel))
  const [serial, setSerial] = useState(false)
  const [linkError, setLinkError] = useState('')
  const { state, controller } = useModelBenchmark()
  const selected = state.selected
  const running = benchmarkActive(state.active)
  const acting = state.action !== null
  const valid = available.some((candidate) => candidate.repo === model)
  useEffect(() => { if (!state.active && !available.some((candidate) => candidate.repo === model)) setModel(benchmarkDefaultModel(models, preferredModel ?? activeModel)) }, [models, model, preferredModel, activeModel, state.active?.id])
  useEffect(() => { if (state.active) { setModel(state.active.model); setSerial(state.active.serial) } }, [state.active?.id])
  const start = () => {
    if (blockedReason || state.loading || state.error || running || acting || !valid) return
    void controller.start({ id: `benchmark_${crypto.randomUUID().replaceAll('-', '')}`, model, serial })
  }
  return <ModelsDrawer title={tx("Benchmarks")} labelId="model-benchmark-title" description={tx("Repeatable community-v1 measurements for cached models on this Mac.").replace('community-v1', 'community-v2')} onClose={onClose} footer={<div className="flex flex-wrap items-center justify-between gap-2"><Button size="sm" variant="ghost" onClick={onClose}>{running ? tx("Close and keep running") : tx("Close")}</Button>{running ? <Button size="sm" disabled={acting || state.cancellationRequested || state.active?.state === 'restoring'} onClick={() => void controller.cancel()}>{state.cancellationRequested || state.active?.state === 'restoring' ? tx("Stopping and restoring…") : tx("Cancel benchmark")}</Button> : <Button size="sm" variant="primary" disabled={!!blockedReason || state.loading || !!state.error || acting || !valid} onClick={start}>{state.action === 'start' ? tx("Starting…") : tx("Run benchmark")}</Button>}</div>}>
    <div className="space-y-5">
      <div className="rounded-xl bg-card-2 p-3.5"><p className="text-[13px] leading-relaxed text-ink-2">{tx("Uses fixed code and chat fixtures, a 4,096-token context and 256 output tokens; 24 requests include four warm-ups and 20 measured samples at temperatures 0 and 1.")}</p><p className="mt-2 text-[12px] leading-relaxed text-ink-3">{tx("This temporarily reserves an idle app-managed engine, pauses its server and restores the previous model and profile afterward; finish chat/API work first, and no models are downloaded automatically.")}</p></div>
      {blockedReason && !running && <p role="status" className="rounded-lg border border-warn/25 bg-warn/5 p-3 text-[13px] text-warn">{blockedReason}</p>}
      <div className="space-y-3"><label className="block text-[13px] font-medium text-ink">{tx("Cached model")}<select aria-label={tx("Benchmark model")} value={model} disabled={running || acting || state.loading} onChange={(event) => setModel(event.target.value)} className="mt-1.5 w-full rounded-lg border border-line bg-card px-3 py-2.5 text-[13px] text-ink focus-visible:outline-2 focus-visible:outline-accent"><option value="">{tx("Choose a compatible cached model")}</option>{state.active && !available.some((candidate) => candidate.repo === state.active?.model) && <option value={state.active.model}>{state.active.model} {tx("· Active run model")}</option>}{available.map((candidate) => <option key={candidate.repo} value={candidate.repo}>{candidate.title ?? candidate.repo}</option>)}</select></label><div className="flex items-start justify-between gap-3 rounded-lg border border-line p-3"><div><p className="text-[13px] text-ink-2">{tx("Serial baseline")}</p><p className="mt-0.5 text-[12px] leading-relaxed text-ink-3">{tx("Use the same fixed protocol with serial execution for this run.")}</p></div><Toggle label={tx("Run a serial benchmark baseline")} checked={serial} disabled={running || acting || state.loading} onChange={setSerial} /></div>{!available.length && <p role="status" className="text-[13px] leading-relaxed text-ink-3">{tx("No compatible cached models are available; download one manually in Discover first, as linked folders and unverified memory fits are not benchmark targets.")}</p>}</div>
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line pt-4"><h3 className="text-[14px] font-medium text-ink">{running ? tx("Active run and history") : tx("Local results")}</h3><Button size="sm" variant="ghost" icon="refresh" disabled={acting || state.loading} onClick={() => void controller.refresh()}>{state.loading ? tx("Refreshing…") : tx("Refresh results")}</Button></div>
      {state.loading && <p role="status" className="text-[13px] text-ink-3">{tx("Checking the active job and saved results…")}</p>}
      {state.error && <p role="alert" className="rounded-lg border border-danger/25 bg-danger/5 p-3 text-[13px] text-danger">{<OriginalDiagnostic>{state.error}</OriginalDiagnostic>}</p>}{state.notice && <p role="status" className="text-[13px] text-ink-2">{state.notice}</p>}
      {(state.active || state.history.length > 0) && <label className="block text-[12px] text-ink-3">{tx("View a run")}<select aria-label={tx("Benchmark result history")} value={selected?.id ?? ''} disabled={acting} onChange={(event) => controller.select(event.target.value)} className="mt-1.5 w-full rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink focus-visible:outline-2 focus-visible:outline-accent">{[...(state.active ? [state.active] : []), ...state.history.filter((run) => run.id !== state.active?.id)].map((run) => <option key={run.id} value={run.id}>{run.model} · {run.serial ? tx("Serial") : tx("Standard")} · {statusText(run.state, tx)} · {new Date(run.startedAt).toLocaleString(currentIntlLocale())}</option>)}</select></label>}
      {state.active && selected?.id !== state.active.id && <div className="rounded-lg border border-line p-3"><p role="status" className="break-words text-[12px] text-ink-3">{tx("A benchmark is still active for ")}{state.active.model}{tx("; Cancel benchmark applies to that active run while you view history.")}</p><Button className="mt-2" size="sm" variant="ghost" onClick={() => controller.select(state.active!.id)}>{tx("View active run")}</Button></div>}
      {!state.loading && !selected && !state.error && <div className="rounded-xl border border-dashed border-line p-5 text-[13px] leading-relaxed text-ink-3">{tx("No benchmark results yet; choose a cached model and run the fixed protocol when the engine is idle.")}</div>}
      {selected && <><BenchmarkResults run={selected} /><BenchmarkMetadata run={selected} />{!benchmarkActive(selected) && <div className="space-y-3 border-t border-line pt-4"><div className="flex flex-wrap gap-2"><Button size="sm" icon="copy" disabled={acting} onClick={() => void controller.copy()}>{tx("Copy basic results")}</Button><Button size="sm" disabled={acting || !selected.receiptAvailable} onClick={() => void controller.export()}>{tx("Review and export JSON")}</Button></div><p className="text-[12px] leading-relaxed text-ink-3">{tx("Exports are local files; review the report before sharing it, and nothing is published automatically.")}</p><details className="text-[12px] text-ink-3"><summary className="cursor-pointer focus-visible:outline-2 focus-visible:outline-accent">{tx("Text included in basic copy")}</summary><pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-lg bg-card-2 p-3 font-sans leading-relaxed text-ink-2">{benchmarkBasicSummary(selected)}</pre></details><Button size="sm" variant="ghost" onClick={async () => { setLinkError(''); try { await api.openExternal('https://tensorfold.dev/benchmarks') } catch { setLinkError(tx("The community page could not open; try again in your browser.")) } }}>{tx("Community benchmarks")}</Button>{linkError && <p role="alert" className="text-[12px] text-danger">{linkError}</p>}</div>}</>}
      {running && <p className="text-[12px] leading-relaxed text-ink-3">{tx("You can close this drawer and reopen Benchmarks to follow or cancel the same job; closing does not cancel it.")}</p>}
    </div>
  </ModelsDrawer>
}
