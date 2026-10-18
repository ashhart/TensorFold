import type { WatchJob, WatchSnapshot } from '../../../shared/watch-relay'
import { remoteDeviceStatus } from '../../agent/device-status'
import { currentSecurityRoleFromEnv } from '../../agent/security'
import { store } from '../state'
import { isActive } from './archive'
import { remoteWorkSnapshot } from './remote-snapshot'

/** Final synchronous projection, after all network awaits and directly before TLS serialization. */
export function createWatchCommandProjection(deps: {activeThreadIds?: () => readonly string[]; now?: () => number; environment?: () => Partial<Record<string, string | undefined>>} = {}) {
  return (snapshot: WatchSnapshot, sessionId: string): WatchSnapshot => {
    if (snapshot.session !== sessionId) throw new Error('paired_session_changed')
    const now = deps.now?.() ?? Date.now()
    const caps = remoteDeviceStatus('paired', currentSecurityRoleFromEnv(deps.environment?.() ?? process.env), null).capabilities
    const currentThreads = store.threads.filter(thread => isActive(thread) && thread.kind === 'task' && !thread.incognito && !thread.phoneLocalOnly && !thread.googleLocalOnly)
    const visible = new Map(currentThreads.map(thread => [thread.id, thread]))
    const active = new Set(deps.activeThreadIds?.() ?? currentThreads.filter(thread => thread.status === 'running').map(thread => thread.id))
    const jobs = snapshot.jobs.flatMap(previous => {
      const thread = visible.get(previous.id)
      if (!thread) return []
      const item = [...thread.timeline].reverse().find(entry => entry.kind === 'assistant')
      const running = active.has(thread.id)
      const state: WatchJob['state'] = running ? thread.timeline.some(entry => entry.state === 'awaiting' || entry.state === 'pending' && String(entry.kind) === 'approval') ? 'waiting' : 'running'
        : thread.status === 'error' ? 'failed' : item ? 'finished' : 'idle'
      return [{id: thread.id.slice(0, 256), title: thread.title.slice(0, 100), state, updatedAt: thread.updatedAt,
        canReply: previous.canReply && (running ? caps.taskGuide : caps.taskStart),
        latestReply: item && caps.transcriptReview && typeof item.text === 'string' ? item.text.slice(0, 1_500) : null,
        latestReplyId: item ? item.id.slice(0, 256) : null}]
    }).slice(0, 20)
    // Watch counts describe shared tasks only. Unlinked/hidden work and approvals
    // must not become an alternate way to infer private phone or archived jobs.
    const sharedWork = store.jobs.filter(job => {
      const linked = [job.threadId, job.currentThreadId, ...(job.kind === 'thread' ? [job.owner] : [])].filter((id): id is string => typeof id === 'string' && !!id)
      return !job.incognito && !job.archivedAt && linked.length > 0 && linked.every(id => visible.has(id))
    })
    const work = remoteWorkSnapshot(now)
    return {version: 1, session: sessionId, generatedAt: now, emittedAt: now, connected: snapshot.connected,
      serverName: String(store.settings.remoteControl?.serverName ?? snapshot.serverName).slice(0, 80), jobs,
      running: Math.min(work.jobs.running, sharedWork.filter(job => job.status === 'running').length),
      queued: Math.min(work.jobs.queued, sharedWork.filter(job => job.status === 'queued').length),
      approvals: Math.min(work.approvals.pending, sharedWork.reduce((sum, job) => sum + job.pendingApprovals.filter(approval => !approval.resolvedAt).length, 0)),
      transcriptReview: caps.transcriptReview, error: snapshot.error}
  }
}
