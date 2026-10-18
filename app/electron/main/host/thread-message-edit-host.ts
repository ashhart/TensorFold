import { assertThreadForkRestrictions } from './thread-revert-fork'
import type { Provider, Thread } from '../../agent/types'
import type { ThreadMessageEditInput } from '../../../shared/thread-message-edit'
import { assertThreadMessageEditable, prepareThreadMessageEdit } from '../../agent/message-edit'
import { saveThreadMessageEdit } from '../../store/message-edit'
import type { WorkspaceMutationGuard } from './workspace-mutation-guard'

export function createThreadMessageEditHost(deps: {
  threads(): Thread[]
  provider(thread: Thread): Provider | undefined
  aborters: Map<string, AbortController>
  assertNoPendingWork?(thread: Thread): void
  guard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  publish(thread: Thread): void
  save?: typeof saveThreadMessageEdit
}) {
  const resolve = (id: string) => {
    const thread = deps.threads().find((candidate) => candidate.id === id)
    if (!thread) throw new Error('Chat not found.')
    return thread
  }
  const validate = (thread: Thread) => {
    if (deps.aborters.has(thread.id)) throw new Error('Stop the active run before editing a sent message.')
    deps.assertNoPendingWork?.(thread)
    if (thread.timeline.some((item) => item.kind === 'cue' && item.state !== 'ok')) throw new Error('Finish or clear this chat’s pending guidance before editing its history.')
    assertThreadForkRestrictions(thread, false)
    assertThreadMessageEditable(thread, deps.provider(thread))
  }
  return {
    prepare(threadId: string, itemId: string) {
      const thread = resolve(threadId)
      validate(thread)
      return prepareThreadMessageEdit(thread, itemId)
    },
    async edit(input: ThreadMessageEditInput): Promise<Thread> {
      if (!input || typeof input !== 'object') throw new Error('Invalid message edit.')
      validate(resolve(input.threadId))
      const lease = deps.guard.reserveStart({ threadId: input.threadId })
      try {
        const thread = await (deps.save ?? saveThreadMessageEdit)(deps.threads, input, validate)
        // A closed secondary window/remote observer cannot turn a durable commit
        // into a misleading rejection. The IPC return is also authoritative.
        try { deps.publish(thread) } catch { console.warn('[message-edit] Saved history could not be broadcast to every observer.') }
        return thread
      } finally { lease.release() }
    },
  }
}
