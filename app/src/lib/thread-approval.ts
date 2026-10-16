import type { Job, TimelineItem } from './types'

export function threadPendingApproval(threadId: string, timeline: TimelineItem[], live: Array<{ itemId: string; callId: string }>, jobs: Job[]) {
  const first = live[0] ?? jobs.flatMap((job) => (job.pendingApprovals ?? [])
    .filter((approval) => !approval.resolvedAt && (approval.threadId ?? job.currentThreadId ?? job.threadId) === threadId))
    .find((approval) => timeline.some((item) => item.id === approval.itemId))
  if (!first) return null
  const item = timeline.find((candidate) => candidate.id === first.itemId)
  return item ? { callId: item.approvalId ?? item.id, item } : null
}
