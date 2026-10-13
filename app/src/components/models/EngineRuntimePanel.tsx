import clsx from 'clsx'
import { useEngine, formatBytes } from '../../lib/engine'
import { engineBaseUrl, formatTokenRate } from '../../lib/engine-workspace'
import { repoDisplayName } from '../../lib/turn-stats-line'
import { BrandMark } from '../BrandMark'
import { Button } from '../ui'
import { CortheonToggle } from './CortheonToggle'

export function EngineRuntimePanel({ activeRuns, onApi, onWorkers, onUnload }: {
  activeRuns: number
  onApi: () => void
  onWorkers: () => void
  onUnload: () => void
}) {
  const { snapshot, error } = useEngine()
  const status = snapshot?.status
  const model = snapshot?.installed.find((item) => item.repo === status?.model)
  const performance = status?.state === 'ready' && snapshot?.performance?.model === status.model ? snapshot.performance : null
  const ready = status?.state === 'ready'
  const loading = status?.state === 'starting'
  const name = snapshot?.suggestions.find((item) => item.repo === status?.model)?.title ?? (status?.model ? repoDisplayName(status.model) : 'Choose a model to begin')
  const label = !status ? 'Checking engine' : !status.available ? 'Engine unavailable' : ready ? 'Ready on this Mac' : loading ? 'Loading model' : status.state === 'failed' ? 'Needs attention' : 'Engine idle'
  const tone = ready ? 'bg-positive' : loading ? 'bg-warn animate-pulse' : status?.state === 'failed' || status?.available === false ? 'bg-danger' : 'bg-ink-3'
  const endpoint = engineBaseUrl(status)

  return (
    <section data-testid="local-runtime" aria-label="Local engine" className="mt-5 overflow-hidden rounded-xl border border-line bg-card">
      <div className="flex flex-wrap items-center gap-3 p-4">
        <BrandMark size={40} tile />
        <div className="min-w-0 flex-1 basis-40">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[12px] text-ink-2"><span className={clsx('h-1.5 w-1.5 shrink-0 rounded-full', tone)} />{label}{status?.version && <span>TensorFold {status.version}</span>}</div>
          <h2 className="mt-1 break-words text-[16px] font-medium text-ink">{name}</h2>
        </div>
        {(ready || loading) && <Button size="sm" icon="stop" onClick={onUnload} disabled={activeRuns > 0} title={activeRuns ? 'Stop active local runs before unloading their model' : undefined}>{loading ? 'Cancel loading' : 'Unload'}</Button>}
      </div>
      {(error || status?.message) && <p role={error || status?.state === 'failed' ? 'alert' : 'status'} className={clsx('px-4 pb-3 text-[13px]', error || status?.state === 'failed' ? 'text-danger' : 'text-ink-2')}>{error || status?.message}</p>}
      {(ready || loading || status?.state === 'failed') ? <dl className="grid grid-cols-[repeat(auto-fit,minmax(145px,1fr))] gap-4 border-t border-line px-4 py-4">
        <Metric label="Decode speed" value={formatTokenRate(performance?.decodeTokensPerSecond)} unit={performance?.decodeTokensPerSecond != null ? 'tok/s' : undefined} detail={performance ? `Latest app request, ${new Date(performance.measuredAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}` : 'Measured after a request'} />
        <Metric label="Prefill speed" value={formatTokenRate(performance?.prefillTokensPerSecond)} unit={performance?.prefillTokensPerSecond != null ? 'tok/s' : undefined} detail={performance?.firstTokenMs != null ? `${(performance.firstTokenMs / 1000).toFixed(2)} s to first token` : 'No estimate shown'} />
        <Metric label="Model memory" value={model?.memoryFloorBytes != null ? formatBytes(model.memoryFloorBytes) : '—'} detail={status ? `${formatBytes(status.memoryLimitBytes)} model budget` : 'Checking this Mac'} />
        <Metric label="Context window" value={ready && status.contextTokens ? `${Math.round(status.contextTokens / 1024)}k` : '—'} detail={ready && status.contextTokens ? 'Prompt and reply combined' : 'Reported after loading'} />
      </dl> : status && <p className="border-t border-line px-4 py-3 text-[12px] text-ink-2">{formatBytes(status.memoryLimitBytes)} available to models out of {formatBytes(status.totalMemoryBytes)} memory on this Mac.</p>}
      <CortheonToggle status={status} activeRuns={activeRuns} />
      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line bg-card-2 px-4 py-2.5 text-[12px] text-ink-2">
        <button onClick={onApi} className="min-w-0 rounded-md text-left hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">{endpoint ? <span className="break-all font-mono">{endpoint}</span> : 'Local API starts when a model is loaded'}</button>
        <button onClick={onWorkers} className="rounded-md px-1 py-1 hover:text-ink focus-visible:outline-2 focus-visible:outline-accent">{activeRuns ? `${activeRuns} active local run${activeRuns === 1 ? '' : 's'}` : 'Give a worker a task'}</button>
      </div>
      {activeRuns > 0 && <p className="border-t border-line px-4 py-2 text-[12px] text-ink-2">The loaded model is in use; stop its runs before switching or unloading.</p>}
      {status?.state === 'failed' && status.logTail.length > 0 && <details className="border-t border-line px-4 py-3 text-[13px] text-ink-2"><summary className="cursor-pointer">Engine details</summary><pre className="scroll-thin mt-2 max-h-32 overflow-auto whitespace-pre-wrap break-all text-[12px]">{status.logTail.slice(-6).join('\n')}</pre></details>}
    </section>
  )
}

function Metric({ label, value, unit, detail }: { label: string; value: string; unit?: string; detail: string }) {
  return <div className="min-w-0"><dt className="text-[12px] text-ink-2">{label}</dt><dd className="mt-1 text-[19px] font-medium tabular-nums text-ink">{value}{unit && <span className="ml-1 text-[12px] font-normal text-ink-2">{unit}</span>}</dd><p className="mt-1 text-[12px] text-ink-2">{detail}</p></div>
}
