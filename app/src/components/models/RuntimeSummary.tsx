import clsx from 'clsx'
import type { EngineSnapshot } from '../../lib/types'
import { engineReadiness } from '../../lib/engine-readiness'
import { formatBytes } from '../../lib/engine'
import { formatTokenRate } from '../../lib/engine-workspace'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { BrandMark } from '../BrandMark'
import { Button } from '../ui'

interface Props {
  snapshot: EngineSnapshot | null
  name: string
  label: string
  tone: string
  error: string | null
  blocked: boolean
  minimal: boolean
  onDetails?: () => void
  onChat?: () => void
  onApi: () => void
  onCancel: () => void
  onRetry?: () => void
  changingBlocked: boolean
  stopping: boolean
}

export function RuntimeSummary({ snapshot, name, label, tone, error, blocked, minimal, onDetails, onChat, onApi, onCancel, onRetry, changingBlocked, stopping }: Props) {
  const tx = useAuthoredText()
  const summary = engineReadiness(error ? null : snapshot)
  const status = snapshot?.status
  const loading = !!status?.available && status.state === 'starting'
  const failed = !!status?.available && status.state === 'failed'
  const problem = error || (status?.state === 'failed' || status?.available === false ? status.message : null)
  return <section data-testid="local-runtime" aria-label={tx('Local engine')} className="mt-5 overflow-hidden rounded-2xl border border-line bg-card shadow-card">
    <div className="flex flex-wrap items-center gap-4 px-4 py-4 sm:px-5">
      <BrandMark size={38} tile />
      <div className="min-w-0 flex-1 basis-40">
        <div className="flex flex-wrap items-center gap-2 text-[11px] font-medium text-ink-2">
          <span className={clsx('h-1.5 w-1.5 rounded-full', tone)} aria-hidden="true" />
          <span role="status">{label}</span>
          {status?.version && <span className="text-ink-3">{tx('Engine')} {status.version}</span>}
        </div>
        <h2 className="mt-1 break-words text-[16px] font-medium tracking-tight text-ink">{name}</h2>
      </div>
      <div className="flex items-center gap-2">
        {loading && <Button size="sm" icon="stop" disabled={changingBlocked || stopping} onClick={onCancel}>
          {stopping ? tx('Cancelling…') : tx('Cancel loading')}
        </Button>}
        {failed && onRetry && <Button size="sm" variant="primary" icon="refresh" disabled={changingBlocked || stopping} onClick={onRetry}>
          {tx('Retry loading')}
        </Button>}
        {summary.ready && (minimal
          ? <Button size="sm" icon="terminal" onClick={onApi}>{tx('Use local API')}</Button>
          : onChat && <Button size="sm" variant="primary" icon="message" disabled={blocked} onClick={onChat}>
            {tx('Chat with this model')}
          </Button>)}
        {onDetails && <Button size="sm" variant="ghost" aria-label={tx('Engine details')} onClick={onDetails}>{tx('Details')}</Button>}
      </div>
    </div>
    {problem && <p role="alert" className="px-5 pb-4 text-[13px] text-danger">{problem}</p>}
    {loading && <p className="px-5 pb-4 text-[13px] text-ink-2">{tx('Preparing the model in memory. You can cancel while it loads.')}</p>}
    <dl className="grid grid-cols-1 gap-3 border-t border-line bg-card-2/50 px-5 py-3 min-[680px]:grid-cols-3">
      <div className="min-w-0">
        <dt className="text-[11px] text-ink-3">{tx('Model memory floor')}</dt>
        <dd className="mt-1 text-[12px] tabular-nums text-ink-2">
          {summary.floor == null ? tx('Not reported') : formatBytes(summary.floor)}
          {summary.budget != null && <span className="text-ink-3"> / {formatBytes(summary.budget)} {tx('budget')}</span>}
        </dd>
        {summary.floorPercent != null && <div className="mt-2 h-1 max-w-48 overflow-hidden rounded-full bg-ink-5" aria-hidden="true">
          <div className="h-full rounded-full bg-accent/60" style={{ width: `${summary.floorPercent}%` }} />
        </div>}
      </div>
      <div><dt className="text-[11px] text-ink-3">{tx('Context window')}</dt>
        <dd className="mt-1 text-[12px] tabular-nums text-ink-2">{summary.context == null ? tx('Reported after loading') : `${summary.context.toLocaleString()} ${tx('tokens')}`}</dd>
      </div>
      <div><dt className="text-[11px] text-ink-3">{tx('Latest generation')}</dt>
        <dd className="mt-1 text-[12px] tabular-nums text-ink-2">{summary.decode == null ? tx('Measured after a request') : `${formatTokenRate(summary.decode)} tok/s`}</dd>
      </div>
    </dl>
  </section>
}
