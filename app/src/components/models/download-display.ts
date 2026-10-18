import type { PullProgress } from '../../lib/types'
import { formatBytes } from '../../lib/engine-format'

export function isDownloadPending(pull: Pick<PullProgress, 'state'> | null | undefined): boolean {
  return pull?.state === 'queued' || pull?.state === 'running' || pull?.state === 'verifying'
}

export function downloadStateLabel(pull: PullProgress): string {
  if (pull.state === 'queued') return pull.queuePosition ? `Queued · ${pull.queuePosition} ahead` : 'Starting download'
  if (pull.state === 'running') return pull.phase.startsWith('Downloading companion') ? 'Downloading companion model' : 'Downloading'
  if (pull.state === 'verifying') return 'Verifying checkpoint'
  if (pull.state === 'done') return 'Ready to load'
  if (pull.state === 'cancelled') return 'Cancelled'
  return 'Download failed'
}

export function downloadEstimate(pull: PullProgress, now: number): string | null {
  if (pull.state !== 'running' || pull.updatedAt == null || now - pull.updatedAt > 15_000 || now < pull.updatedAt) return null
  const rate = pull.bytesPerSecond
  const seconds = pull.etaSeconds
  if (rate == null || !Number.isFinite(rate) || rate <= 0) return null
  const speed = `${formatBytes(rate)}/s`
  if (seconds == null || !Number.isFinite(seconds) || seconds <= 0) return speed
  const remaining = seconds < 60 ? `${Math.max(10, Math.ceil(seconds / 10) * 10)} sec`
    : seconds < 3600 ? `${Math.ceil(seconds / 60)} min`
      : `${Math.floor(seconds / 3600)} hr ${Math.ceil((seconds % 3600) / 60)} min`
  return `${speed} · about ${remaining} left`
}

export function orderedDownloads(pulls: PullProgress[]): PullProgress[] {
  const activeOrder = { running: 0, verifying: 0, queued: 1, done: 2, cancelled: 2, failed: 2 }
  return [...pulls].sort((left, right) => {
    const rank = activeOrder[left.state] - activeOrder[right.state]
    if (rank) return rank
    if (left.state === 'queued' && right.state === 'queued') return (left.queuePosition ?? Number.MAX_SAFE_INTEGER) - (right.queuePosition ?? Number.MAX_SAFE_INTEGER)
    return (right.finishedAt ?? right.startedAt) - (left.finishedAt ?? left.startedAt)
  })
}
