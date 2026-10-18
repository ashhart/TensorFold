import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { useEngine, formatBytes } from '../../lib/engine'
import { engineBaseUrl, formatTokenRate } from '../../lib/engine-workspace'
import { repoDisplayName } from '../../lib/turn-stats-line'
import { BrandMark } from '../BrandMark'
import { Button } from '../ui'
import { useLearning, learningBusy } from '../../lib/learning'
import { CortheonToggle } from './CortheonToggle'

export function EngineRuntimePanel({ activeRuns, onApi, onWorkers, onUnload, minimal = false, compact = false, onDetails }: {
  minimal?: boolean
  compact?: boolean
  onDetails?: () => void
  activeRuns: number
  onApi: () => void
  onWorkers: () => void
  onUnload: () => void
}) {
  const tx = useAuthoredText()
  const { snapshot: learning } = useLearning()
  const { snapshot, error } = useEngine()
  const status = snapshot?.status
  const model = snapshot?.installed.find((item) => item.repo === status?.model)
  const performance = status?.state === 'ready' && snapshot?.performance?.model === status.model ? snapshot.performance : null
  const ready = status?.state === 'ready'
  const loading = status?.state === 'starting'
  const name = model?.title ?? snapshot?.suggestions.find((item) => item.repo === status?.model)?.title ?? (status?.model ? repoDisplayName(status.model) : tx("Choose a model to begin"))
  const label = !status ? tx("Checking engine") : !status.available ? tx("Engine unavailable") : ready ? tx("Ready on this Mac") : loading ? tx("Loading model") : status.state === 'failed' ? tx("Needs attention") : tx("Engine idle")
  const tone = ready ? 'bg-positive' : loading ? 'bg-warn animate-pulse' : status?.state === 'failed' || status?.available === false ? 'bg-danger' : 'bg-ink-3'
  const endpoint = engineBaseUrl(status)

  if (compact) return <section data-testid="local-runtime" aria-label={tx("Local engine")} className="mt-5 rounded-xl border border-line bg-card px-3.5 py-3">
    <div className="flex flex-wrap items-center gap-3">
      <BrandMark size={26} tile />
      <div className="min-w-0 flex-1 basis-40"><div className="flex min-w-0 items-center gap-2"><span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', tone)} /><span className="min-w-0 truncate text-[13px] font-medium text-ink">{ready || loading || status?.state === 'failed' ? name : label}</span></div>{loading && <p role="status" className="mt-1 text-[12px] text-ink-2">{tx("Loading model…")}</p>}</div>
      {ready && performance?.decodeTokensPerSecond != null && <span className="text-[12px] tabular-nums text-ink-2" title={tx("Measured on the latest app request")}>{formatTokenRate(performance.decodeTokensPerSecond)} tok/s</span>}
      <button type="button" onClick={onDetails} aria-label={tx("Engine details")} className="rounded-lg px-2 py-1.5 text-[12px] text-ink-2 hover:bg-card-2 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">{tx("Details")}</button>
    </div>
    {(error || status?.state === 'failed' && status.message) && <p role="alert" className="mt-2 break-words text-[13px] text-danger">{error || status?.message}</p>}
  </section>

  return (
    <section data-testid="local-runtime" aria-label={tx("Local engine")} className="mt-5 overflow-hidden rounded-xl border border-line bg-card">
      <div className="flex flex-wrap items-center gap-3 p-4">
        <BrandMark size={40} tile />
        <div className="min-w-0 flex-1 basis-40">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px] text-ink-2"><span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', tone)} />{label}{status?.version && <span>{"TensorFold "}{status.version}</span>}</div>
          <h2 className="mt-1 break-words text-[16px] font-medium text-ink">{name}</h2>
        </div>
        {(ready || loading) && <Button size="sm" icon="stop" onClick={onUnload} disabled={activeRuns > 0 || learningBusy(learning)} title={learningBusy(learning) ? tx("Wait for the learning checkpoint to finish saving") : activeRuns ? tx("Stop active local runs before unloading their model") : undefined}>{loading ? tx("Cancel loading") : tx("Unload")}</Button>}
      </div>
      {(error || status?.message) && <p role={error || status?.state === 'failed' ? 'alert' : 'status'} className={clsx('px-4 pb-3 text-[13px]', error || status?.state === 'failed' ? 'text-danger' : 'text-ink-2')}>{error || status?.message}</p>}
      {(ready || loading || status?.state === 'failed') ? <dl className="grid grid-cols-[repeat(auto-fit,minmax(145px,1fr))] gap-4 border-t border-line px-4 py-4">
        <Metric label={tx("Decode speed")} value={formatTokenRate(performance?.decodeTokensPerSecond)} unit={performance?.decodeTokensPerSecond != null ? 'tok/s' : undefined} detail={performance ? `Latest app request, ${new Date(performance.measuredAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : 'Measured after a request'} />
        <Metric label={tx("Prefill speed")} value={formatTokenRate(performance?.prefillTokensPerSecond)} unit={performance?.prefillTokensPerSecond != null ? 'tok/s' : undefined} detail={performance?.firstTokenMs != null ? `${(performance.firstTokenMs / 1000).toFixed(2)} s to first token` : 'No estimate shown'} />
        <Metric label={tx("Model memory")} value={model?.memoryFloorBytes != null ? formatBytes(model.memoryFloorBytes) : '—'} detail={status ? `${formatBytes(status.memoryLimitBytes)} model budget` : 'Checking this Mac'} />
        <Metric label={tx("Context window")} value={ready && status.contextTokens ? `${Math.round(status.contextTokens / 1024)}k` : '—'} detail={ready && status.contextTokens ? 'Prompt and reply combined' : 'Reported after loading'} />
      </dl> : status && <p className="border-t border-line px-4 py-3 text-[12px] text-ink-2">{formatBytes(status.memoryLimitBytes)} {tx("available to models out of ")}{formatBytes(status.totalMemoryBytes)} {tx("memory on this Mac.")}</p>}
      <CortheonToggle status={status} activeRuns={activeRuns} blocked={learningBusy(learning) || status?.state === 'starting'} />
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line bg-card-2 px-4 py-2.5 text-[12px] text-ink-2">
        <button onClick={onApi} className="min-w-0 rounded-md text-left hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">{learning?.enabled ? tx("API sharing paused during learning") : endpoint ? <span className="break-all font-mono">{endpoint}</span> : tx("Local API starts when a model is loaded")}</button>
        {!minimal && <button onClick={onWorkers} className="rounded-md px-1 py-1 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">{activeRuns ? tx("Active local runs: {count}", { count: activeRuns }) : tx("Give a worker a task")}</button>}
      </div>
      {activeRuns > 0 && <p className="border-t border-line px-4 py-2 text-[12px] text-ink-2">{tx("The loaded model is in use; stop its runs before switching or unloading.")}</p>}
      {status?.state === 'failed' && status.logTail.length > 0 && <details className="border-t border-line px-4 py-3 text-[13px] text-ink-2"><summary className="cursor-pointer">{tx("Engine details")}</summary><pre className="scroll-thin mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-all text-[12px]">{status.logTail.slice(-6).join('\n')}</pre></details>}
    </section>
  )
}

function Metric({ label, value, unit, detail }: { label: string; value: string; unit?: string; detail: string }) {
  return <div className="min-w-0"><dt className="text-[12px] text-ink-2">{label}</dt><dd className="mt-1 text-[19px] font-medium tabular-nums text-ink">{value}{unit && <span className="ml-1 text-[12px] font-normal text-ink-2">{unit}</span>}</dd><p className="mt-1 text-[12px] text-ink-2">{detail}</p></div>
}
