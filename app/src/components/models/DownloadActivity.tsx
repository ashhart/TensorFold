import type { EngineInstalledModel, PullProgress } from '../../lib/types'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Button } from '../ui'
import { PullRow } from './ModelRows'
import { isDownloadPending, orderedDownloads } from './download-display'

interface Props {
  pulls: PullProgress[]
  installed: EngineInstalledModel[]
  showCompleted: boolean
  locked: boolean
  minimal: boolean
  onAll: () => void
  onCancel: (repo: string) => Promise<unknown>
  onRetry: (repo: string, revision: string | null) => Promise<unknown>
  onLoad: (repo: string) => void
}

export function DownloadActivity({ pulls, installed, showCompleted, locked, minimal, onAll, onCancel, onRetry, onLoad }: Props) {
  const tx = useAuthoredText()
  const pull = orderedDownloads(pulls)[0]
  if (!pull || pull.state === 'done' && !showCompleted) return null
  const pending = pulls.filter(isDownloadPending).length
  const model = installed.find((item) => item.repo === pull.repo)
  const complete = pull.state === 'done'
  const title = complete ? tx('Download complete') : pull.state === 'failed' ? tx('Download needs attention')
    : pull.state === 'cancelled' ? tx('Download paused') : tx('Download in progress')
  return <section data-testid="models-download-activity" aria-label={tx('Current download')} className="mt-4 rounded-xl border border-line bg-card-2/50 p-3 sm:p-4">
    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
      <h2 className="text-[13px] font-medium text-ink" role="status">{title}{pending > 1 ? ` · ${pending - 1} ${tx('more in queue')}` : ''}</h2>
      <Button size="xs" variant="ghost" onClick={onAll}>{tx('All downloads')}</Button>
    </div>
    <PullRow pull={pull} onCancel={() => onCancel(pull.repo)} onRetry={() => onRetry(pull.repo, pull.revision)} />
    {complete ? <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
      <p className="max-w-lg text-[12px] leading-relaxed text-ink-2">{tx('Files are saved. Load the model when you are ready to use it.')}</p>
      <Button size="sm" variant="primary" icon="play" disabled={locked || !model || model.fitsMemory === false} onClick={() => onLoad(pull.repo)}>{minimal ? tx('Start server') : tx('Use model')}</Button>
    </div> : pull.state === 'cancelled' && <p className="mt-3 text-[12px] text-ink-2">{tx('Cached files are kept. Resume to continue this download.')}</p>}
  </section>
}
