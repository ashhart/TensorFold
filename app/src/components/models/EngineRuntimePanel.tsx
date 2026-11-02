import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { useEngine, formatBytes } from '../../lib/engine'
import { engineBaseUrl, formatTokenRate } from '../../lib/engine-workspace'
import { repoDisplayName } from '../../lib/turn-stats-line'
import { BrandMark } from '../BrandMark'
import { Button } from '../ui'
import { useLearning, learningBusy } from '../../lib/learning'
import { CortheonToggle } from './CortheonToggle'
import { EngineUpdates } from './EngineUpdates'
import { RuntimeSummary } from './RuntimeSummary'
import { engineReadiness } from '../../lib/engine-readiness'

export function EngineRuntimePanel({ activeRuns, onApi, onWorkers, onUnload, minimal = false, compact = false, onDetails, onChat, onRetry, stopping = false }: {
  minimal?: boolean
  compact?: boolean
  onDetails?: () => void
  onChat?: () => void
  onRetry?: () => void
  stopping?: boolean
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
  const summary = engineReadiness(error ? null : snapshot)
  const ready = summary.ready
  const performance = ready && snapshot?.performance && snapshot.performance.model === status?.model ? snapshot.performance : null
  const prefill = finitePositive(performance?.prefillTokensPerSecond)
  const firstToken = finitePositive(performance?.firstTokenMs)
  const loading = !!status?.available && status.state === 'starting'
  const name = model?.title ?? snapshot?.suggestions.find((item) => item.repo === status?.model)?.title ?? (status?.model ? repoDisplayName(status.model) : tx("Choose a model to begin"))
  const label = error ? tx('Engine status unavailable') : !status ? tx("Checking engine") : !status.available ? tx("Engine unavailable") : ready ? tx("Ready locally") : loading ? tx("Loading model") : status.state === 'failed' ? tx("Needs attention") : tx("Engine idle")
  const tone = error ? 'bg-danger' : ready ? 'bg-positive' : loading ? 'bg-warn animate-pulse' : status?.state === 'failed' || status?.available === false ? 'bg-danger' : 'bg-ink-3'
  const endpoint = ready ? engineBaseUrl(status) : null

  if (compact) return <RuntimeSummary snapshot={snapshot} name={name} label={label} tone={tone} error={error}
    blocked={learningBusy(learning)} minimal={minimal} onDetails={onDetails} onChat={onChat} onApi={onApi}
    onCancel={onUnload} onRetry={onRetry} changingBlocked={activeRuns > 0 || learningBusy(learning)} stopping={stopping} />

  return (
    <section data-testid="local-runtime" aria-label={tx("Local engine")} className="mt-5 overflow-hidden rounded-xl border border-line bg-card">
      <div className="flex flex-wrap items-center gap-3 p-4">
        <BrandMark size={40} tile />
        <div className="min-w-0 flex-1 basis-40">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px] text-ink-2"><span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', tone)} />{label}{status?.version && <span>{"TensorFold "}{status.version}</span>}</div>
          <h2 className="mt-1 break-words text-[16px] font-medium text-ink">{name}</h2>
        </div>
        {(ready || loading) && <Button size="sm" icon="stop" onClick={onUnload} disabled={stopping || activeRuns > 0 || learningBusy(learning)} title={learningBusy(learning) ? tx("Wait for the learning checkpoint to finish saving") : activeRuns ? tx("Stop active local runs before unloading their model") : undefined}>{stopping ? tx('Stopping…') : loading ? tx("Cancel loading") : tx("Unload")}</Button>}
        {status?.available && status.state === 'failed' && onRetry && <Button size="sm" variant="primary" icon="refresh" onClick={onRetry} disabled={stopping || activeRuns > 0 || learningBusy(learning)}>{tx('Retry loading')}</Button>}
      </div>
      {(error || status?.message) && <p role={error || status?.state === 'failed' ? 'alert' : 'status'} className={clsx('px-4 pb-3 text-[13px]', error || status?.state === 'failed' ? 'text-danger' : 'text-ink-2')}>{error || status?.message}</p>}
      {(ready || loading || status?.state === 'failed') ? <dl className="grid grid-cols-[repeat(auto-fit,minmax(145px,1fr))] gap-4 border-t border-line px-4 py-4">
        <Metric label={tx("Decode speed")} value={formatTokenRate(summary.decode)} unit={summary.decode != null ? 'tok/s' : undefined} detail={performance && finitePositive(performance.measuredAt) ? `Latest app request, ${new Date(performance.measuredAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : 'Measured after a request'} />
        <Metric label={tx("Prefill speed")} value={formatTokenRate(prefill)} unit={prefill != null ? 'tok/s' : undefined} detail={firstToken != null ? `${(firstToken / 1000).toFixed(2)} s to first token` : 'No estimate shown'} />
        <Metric label={tx("Model memory floor")} value={formatBytes(summary.floor)} detail={status ? `${formatBytes(summary.budget)} model budget` : tx('Checking engine')} />
        <Metric label={tx("Context window")} value={summary.context != null ? `${Math.round(summary.context / 1024)}k` : '—'} detail={summary.context != null ? 'Prompt and reply combined' : 'Reported after loading'} />
      </dl> : status && <p className="border-t border-line px-4 py-3 text-[12px] text-ink-2">{formatBytes(summary.budget)} {tx("available to models out of ")}{formatBytes(finitePositive(status.totalMemoryBytes))} {tx("memory on this computer.")}</p>}
      <div className="border-t border-line p-4"><EngineUpdates /></div>
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

function finitePositive(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? value : null
}

function Metric({ label, value, unit, detail }: { label: string; value: string; unit?: string; detail: string }) {
  return <div className="min-w-0"><dt className="text-[12px] text-ink-2">{label}</dt><dd className="mt-1 text-[19px] font-medium tabular-nums text-ink">{value}{unit && <span className="ml-1 text-[12px] font-normal text-ink-2">{unit}</span>}</dd><p className="mt-1 text-[12px] text-ink-2">{detail}</p></div>
}
