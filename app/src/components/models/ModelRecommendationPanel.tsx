import { hubFailureCopy, type HubFailureCategory } from '../../../shared/hub-failure'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { formatBytes } from '../../lib/engine'
import type { ModelRecommendation } from '../../lib/model-recommendation'
import type { PullProgress } from '../../lib/types'
import { Button } from '../ui'

export function ModelRecommendationPanel({ recommendation, pull, starting, locked, checkFailed, failureKind, onStart, onDownload, onRetry, minimal = false }: {
  minimal?: boolean
  recommendation: ModelRecommendation
  pull?: PullProgress
  starting: boolean
  locked: boolean
  checkFailed: boolean
  failureKind?: HubFailureCategory | null
  onStart: (repo: string) => void
  onDownload: (repo: string) => void
  onRetry: () => void
}) {
  const tx = useAuthoredText()
  const failure = hubFailureCopy(failureKind ?? 'check')
  const model = recommendation.selected
  const downloading = !!pull && ['queued', 'running', 'verifying'].includes(pull.state)
  return (
    <section data-testid="model-recommendation" className="mt-6 rounded-xl border border-line-strong bg-card p-4">
      <p className="text-[12px] text-ink-2">{recommendation.hardwareLabel}</p>
      <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1 basis-48">
          <h2 className="text-[15px] font-medium text-ink">{model ? tx("Recommended: {value0}", { value0: model.title }) : recommendation.status === 'unsupported' ? tx("This build needs an Apple silicon Mac") : checkFailed ? tx(failure.title) : recommendation.status === 'unavailable' ? tx("No compatible recommendation yet") : tx("Finding a model for this Mac")}</h2>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{!model && checkFailed ? tx(failure.detail) : recommendation.reason}</p>
          {model && <p className="mt-2 text-[12px] text-ink-2">{formatBytes(model.memoryBytes)} {tx("memory required")}{recommendation.fallback ? tx("; {value0} uses less memory", { value0: recommendation.fallback.title }) : ''}.</p>}
        </div>
        {model && <div className="flex flex-wrap items-center gap-2"><Button size="sm" variant="primary" icon={model.action === 'load' ? minimal ? 'play' : 'message' : 'arrowDown'} onClick={() => model.action === 'load' ? onStart(model.repo) : onDownload(model.repo)} disabled={starting || downloading || locked}>{starting ? tx("Starting model…") : downloading ? pull?.state === 'queued' ? tx("Queued") : pull?.state === 'verifying' ? tx("Checking files…") : tx("Downloading…") : model.action === 'load' ? minimal ? tx("Start server") : tx("Start chatting") : tx("Download model")}</Button>
          {model.action === 'download' && recommendation.fallback?.installed && <Button size="sm" variant="default" icon="message" onClick={() => onStart(recommendation.fallback!.repo)} disabled={starting || locked}>{tx("Use installed ")}{recommendation.fallback.title}</Button>}
        </div>}
        {!model && checkFailed && recommendation.status !== 'unsupported' && <Button size="sm" icon="refresh" onClick={onRetry}>{tx("Retry checks")}</Button>}
      </div>
      {model?.action === 'download' && <p className="mt-3 border-t border-line pt-3 text-[12px] text-ink-2">{minimal ? tx("After the download completes, choose Start server to load the model and open its local API.") : tx("After the download completes, start chatting here; the engine loads the model for you.")}</p>}
    </section>
  )
}
