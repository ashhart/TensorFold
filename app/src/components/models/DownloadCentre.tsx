import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import type { PullProgress } from '../../lib/types'
import { PullRow } from './ModelRows'
import { isDownloadPending, orderedDownloads } from './download-display'
import { Button } from '../ui'

export function DownloadCentre({ pulls, onCancel, onRetry }: {
  pulls: PullProgress[]
  onCancel: (repo: string) => void | Promise<unknown>
  onRetry: (repo: string, revision: string | null) => void | Promise<unknown>
}) {
  const tx = useAuthoredText()
  const [showHistory, setShowHistory] = useState(false)
  const ordered = orderedDownloads(pulls)
  const pending = ordered.filter(isDownloadPending)
  const history = ordered.filter((pull) => !isDownloadPending(pull))
  const queued = pending.filter((pull) => pull.state === 'queued').length
  const running = pending.find((pull) => pull.state !== 'queued')
  if (!pulls.length) return null
  return (
    <section className="mt-6 min-w-0" aria-labelledby="download-centre-title" data-testid="download-centre">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
        <h2 id="download-centre-title" className="text-[14px] font-medium text-ink">{tx("Downloads")}</h2>
        <span className="text-[12px] text-ink-2">{[running ? running.state === 'verifying' ? 'Verifying checkpoint' : '1 downloading' : null, queued ? `${queued} queued` : null, !pending.length ? 'No downloads running' : null].filter(Boolean).join(' · ')}</span>
      </div>
      <p className="mb-3 text-[12px] leading-relaxed text-ink-2">{tx("One download at a time, including companion models. Cached files stay available to resume; choose Load when a model is ready.")}</p>
      <div className="space-y-2">
        {[...pending, ...(showHistory ? history : history.slice(0, 3))].map((pull) => (
          <PullRow key={pull.repo} pull={pull} onCancel={() => onCancel(pull.repo)} onRetry={() => onRetry(pull.repo, pull.revision)} />
        ))}
      </div>
      {history.length > 3 && <Button size="xs" variant="ghost" className="mt-2" onClick={() => setShowHistory((current) => !current)}>{showHistory ? tx("Show fewer previous downloads") : tx("Show {value0} previous downloads", { value0: history.length - 3 })}</Button>}
    </section>
  )
}
