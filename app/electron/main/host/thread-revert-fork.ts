import { randomUUID } from 'node:crypto'
import { spineEligible } from '../../agent/incognito'
import type { EventSpineHost } from '../../agent/event-spine-host'
import type { ConversationState } from '../../agent/conversation-checkpoints'
import type { TurnFork } from '../../agent/turn-checkpoints'
import type { Thread } from '../../agent/types'
import { sendToRenderer } from './renderer-events'
import { store } from '../state'

export interface ThreadRevertForkDeps {
  persistThreads(): void
  // remoteServer + spineHost are forward-declared / assigned later in the bootstrap than this
  // factory; accessors keep the lookups lazy so the factory can be constructed early.
  remoteServer(): { publish(event: { kind: string; data: unknown }): void }
  spineHost(): EventSpineHost | null
}

/**
 * Apply a turn-revert's restored conversation to the live thread (VST-036): swap the
 * messages/timeline back, persist, and push the update so the renderer reflects the revert.
 *
 * Extracted verbatim from the registerWorkspaceIpc callback in main-app.ts.
 */
export function createThreadRevertApply(deps: ThreadRevertForkDeps): (threadId: string, conversation: ConversationState) => void {
  const { persistThreads, remoteServer } = deps
  return function applyThreadRevert(threadId, conversation) {
    const thread = store.threads.find((t) => t.id === threadId)
    if (!thread) return
    thread.apiMessages = conversation.apiMessages as typeof thread.apiMessages
    thread.timeline = conversation.timeline as typeof thread.timeline
    // Goal/plan were part of the snapshot: restore them too, and CLEAR them when the
    // turn being restored predates them (a revert that keeps a later goal is half-applied).
    thread.goal = typeof conversation.goal === 'string' && conversation.goal ? conversation.goal : undefined
    thread.goalStatus = (conversation.goalStatus as Thread['goalStatus']) || undefined
    thread.plan = Array.isArray(conversation.plan) ? (conversation.plan as Thread['plan']) : undefined
    // This push is an authoritative replacement, not a potentially-stale persistence
    // snapshot racing live deltas. Bump the epoch so renderer projections accept the
    // shorter timeline/apiMessages produced by a revert exactly once.
    thread.conversationRevision = (thread.conversationRevision ?? 0) + 1
    thread.updatedAt = Date.now()
    persistThreads()
    sendToRenderer('thread:update', thread)
    remoteServer().publish({ kind: 'thread', data: thread })
  }
}

/**
 * Create the forked thread from a turn fork (VST-036): seeded with the fork's conversation,
 * homed like its source (project/cwd/provider), WITHOUT the source's engine resume cursors —
 * the forked history diverges, so resuming the original harness session would desync; the
 * seeded apiMessages carry the context instead.
 *
 * Extracted verbatim from the registerWorkspaceIpc callback in main-app.ts.
 */
export function createThreadForkFactory(deps: ThreadRevertForkDeps): (sourceThreadId: string, fork: TurnFork, overrides?: { cwd?: string; notice?: string }) => Thread | null {
  const { persistThreads, remoteServer, spineHost } = deps
  return function createForkThread(sourceThreadId, fork, overrides) {
    const source = store.threads.find((t) => t.id === sourceThreadId)
    if (!source) return null
    const now = Date.now()
    const t: Thread = {
      id: randomUUID(),
      title: `Fork of ${source.title}`,
      kind: source.kind,
      projectId: source.projectId ?? null,
      // A fork worktree (VST-036) homes the fork at its own copy of the fork-point
      // snapshot; otherwise the fork shares the source folder as before. A worktree
      // fork drops the SSH workspace binding — the worktree is local by construction.
      cwd: overrides?.cwd ?? source.cwd,
      workspace: overrides?.cwd ? null : source.workspace ?? null,
      parentId: source.id,
      forkedAt: now,
      orchestratorHostId: source.orchestratorHostId,
      providerId: source.providerId,
      providerAccountId: source.providerAccountId,
      model: source.model,
      createdAt: now,
      updatedAt: now,
      status: 'idle',
      apiMessages: fork.conversation.apiMessages as Thread['apiMessages'],
      timeline: [
        ...(fork.conversation.timeline as Thread['timeline']),
        {
          id: randomUUID(),
          kind: 'notice',
          text: `Forked from “${source.title}” at turn ${fork.turn} (branch ${fork.branch}). This thread continues independently — the original is untouched.${overrides?.notice ? `\n${overrides.notice}` : ''}`,
          createdAt: now,
        },
      ],
      ...(typeof fork.conversation.goal === 'string' && fork.conversation.goal ? { goal: fork.conversation.goal } : {}),
      ...(fork.conversation.goalStatus ? { goalStatus: fork.conversation.goalStatus as Thread['goalStatus'] } : {}),
      ...(Array.isArray(fork.conversation.plan) ? { plan: fork.conversation.plan as Thread['plan'] } : {}),
      ...(source.incognito ? { incognito: true } : {}),
    }
    store.threads.unshift(t)
    persistThreads()
    if (spineEligible(t)) {
      spineHost()?.executeCommand(
        { commandId: randomUUID(), aggregate: 'thread', streamId: t.id, kind: 'thread.fork', actor: 'user' },
        () => ({ aggregate: 'thread', streamId: t.id, type: 'thread.forked', payload: { sourceThreadId, turn: fork.turn, branch: fork.branch } }),
      )
    }
    sendToRenderer('thread:update', t)
    remoteServer().publish({ kind: 'thread', data: t })
    return t
  }
}
