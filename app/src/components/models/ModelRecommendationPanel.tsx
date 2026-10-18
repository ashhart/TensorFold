import { hubFailureCopy, type HubFailureCategory } from '../../../shared/hub-failure'
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
  const failure = hubFailureCopy(failureKind ?? 'check')
  const model = recommendation.selected
  const downloading = pull?.state === 'running'
  return (
    <section data-testid="model-recommendation" className="mt-6 rounded-xl border border-line-strong bg-card p-4">
      <p className="text-[12px] text-ink-2">{recommendation.hardwareLabel}</p>
      <div className="mt-2 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1 basis-48">
          <h2 className="text-[15px] font-medium text-ink">{model ? `Recommended: ${model.title}` : recommendation.status === 'unsupported' ? 'This build needs an Apple silicon Mac' : checkFailed ? failure.title : recommendation.status === 'unavailable' ? 'No compatible recommendation yet' : 'Finding a model for this Mac'}</h2>
          <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{!model && checkFailed ? failure.detail : recommendation.reason}</p>
          {model && <p className="mt-2 text-[12px] text-ink-2">{formatBytes(model.memoryBytes)} memory required{recommendation.fallback ? `; ${recommendation.fallback.title} uses less memory` : ''}.</p>}
        </div>
        {model && <div className="flex flex-wrap items-center gap-2"><Button size="sm" variant="primary" icon={model.action === 'load' ? minimal ? 'play' : 'message' : 'arrowDown'} onClick={() => model.action === 'load' ? onStart(model.repo) : onDownload(model.repo)} disabled={starting || downloading || locked}>{starting ? 'Starting model…' : downloading ? 'Downloading…' : model.action === 'load' ? minimal ? 'Start server' : 'Start chatting' : 'Download model'}</Button>
          {model.action === 'download' && recommendation.fallback?.installed && <Button size="sm" variant="default" icon="message" onClick={() => onStart(recommendation.fallback!.repo)} disabled={starting || locked}>Use installed {recommendation.fallback.title}</Button>}
        </div>}
        {!model && checkFailed && recommendation.status !== 'unsupported' && <Button size="sm" icon="refresh" onClick={onRetry}>Retry checks</Button>}
      </div>
      {model?.action === 'download' && <p className="mt-3 border-t border-line pt-3 text-[12px] text-ink-2">{minimal ? 'After the download completes, choose Start server to load the model and open its local API.' : 'After the download completes, start chatting here; the engine loads the model for you.'}</p>}
    </section>
  )
}
