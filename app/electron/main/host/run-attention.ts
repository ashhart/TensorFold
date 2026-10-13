import type { AgentEvent } from '../../agent/types'

// Native attention for thread runs: a notification when a run finishes, fails or needs approval
// while the window is in the background, and a dock badge counting threads that need the user.
// The decisions are pure; the host supplies the Electron sinks.

export type AttentionKind = 'finished' | 'failed' | 'approval'

export interface AttentionNotice { kind: AttentionKind; threadId: string; title: string; body: string }

export function shouldNotify(input: { enabled: boolean; focused: boolean; supported: boolean; knownThread: boolean }): boolean {
  return input.enabled && !input.focused && input.supported && input.knownThread
}

export function attentionNotice(kind: AttentionKind, threadId: string, threadTitle: string | null, detail?: string | null): AttentionNotice {
  const name = threadTitle?.trim() || 'A thread'
  if (kind === 'approval') return { kind, threadId, title: 'Needs your approval', body: detail ? `${name}: run ${detail}?` : name }
  if (kind === 'failed') return { kind, threadId, title: 'Run failed', body: detail ? `${name}: ${detail}`.slice(0, 140) : name }
  return { kind, threadId, title: 'Run finished', body: name }
}

// Which threads are waiting on the user after this event. A status other than awaiting, or the
// end of the run, means the approval was answered.
export function nextNeedsYou(prev: ReadonlySet<string>, event: Pick<AgentEvent, 'type' | 'threadId'> & { status?: string }): Set<string> {
  const next = new Set(prev)
  if (event.type === 'status' && event.status === 'awaiting') return next
  if (event.type === 'status' || event.type === 'done' || event.type === 'error') next.delete(event.threadId)
  return next
}

export function badgeText(count: number): string {
  return count > 0 ? String(count) : ''
}

export interface RunAttentionDeps {
  enabled(): boolean
  focused(): boolean
  supported(): boolean
  threadTitle(threadId: string): string | null | undefined
  notify(notice: AttentionNotice): void
  setBadge(text: string): void
}

export function createRunAttention(deps: RunAttentionDeps) {
  let needsYou = new Set<string>()
  const notifiedRuns = new Set<string>()

  const publishBadge = () => { try { deps.setBadge(badgeText(needsYou.size)) } catch { /* best effort */ } }
  const maybeNotify = (notice: AttentionNotice) => {
    const title = deps.threadTitle(notice.threadId)
    if (!shouldNotify({ enabled: deps.enabled(), focused: deps.focused(), supported: deps.supported(), knownThread: title !== undefined })) return
    try { deps.notify(notice) } catch { /* notifications are best effort */ }
  }

  return {
    observe(event: AgentEvent) {
      if (event.type !== 'status' && event.type !== 'done' && event.type !== 'error') return
      const before = needsYou.size
      needsYou = nextNeedsYou(needsYou, event as Pick<AgentEvent, 'type' | 'threadId'> & { status?: string })
      if (needsYou.size !== before) publishBadge()
      if (event.type === 'done' || event.type === 'error') {
        const key = `${event.threadId}:${event.runId}`
        if (notifiedRuns.has(key)) return
        notifiedRuns.add(key)
        if (notifiedRuns.size > 500) notifiedRuns.delete(notifiedRuns.values().next().value as string)
        const title = deps.threadTitle(event.threadId) ?? null
        maybeNotify(event.type === 'done'
          ? attentionNotice('finished', event.threadId, title)
          : attentionNotice('failed', event.threadId, title, event.message))
      }
    },
    // Fired only when an approval is genuinely waiting (not auto-resolved by a saved rule).
    approvalPending(meta: { threadId: string; tool: string }) {
      if (!needsYou.has(meta.threadId)) { needsYou.add(meta.threadId); publishBadge() }
      maybeNotify(attentionNotice('approval', meta.threadId, deps.threadTitle(meta.threadId) ?? null, meta.tool))
    },
    needsYouCount: () => needsYou.size,
  }
}
