import type { Provider, Thread } from '../../agent/types'
import { assertThreadMemoryTransport, parseThreadMemoryPolicy } from '../../../shared/thread-memory-policy'
import { saveThreadMemoryPolicy } from '../../store'
import type { WorkspaceMutationGuard } from './workspace-mutation-guard'

/** Shared by IPC and acceptance tests: no renderer optimism or per-run overrides. */
export function createThreadMemoryPolicyHost(deps: {
  threads(): Thread[]
  provider(thread: Thread): Provider | undefined
  aborters: Map<string, AbortController>
  guard: Pick<WorkspaceMutationGuard, 'reserveStart'>
  publish(thread: Thread): void
  save?: typeof saveThreadMemoryPolicy
}) {
  return async (id: string, value: unknown, expectedRevision: number): Promise<Thread> => {
    if (value === undefined) throw new Error('Choose both saved-memory options explicitly.')
    const requested = parseThreadMemoryPolicy(value)
    const validate = (thread: Thread) => {
      if (thread.status === 'running' || deps.aborters.has(thread.id)) throw new Error('Stop this run before changing its memory choices.')
      assertThreadMemoryTransport({ ...thread, memoryPolicy: requested }, deps.provider(thread))
    }
    const target = deps.threads().find((t) => t.id === id)
    if (!target) throw new Error('Chat not found.')
    validate(target)
    const lease = deps.guard.reserveStart({ threadId: id })
    try {
      const thread = await (deps.save ?? saveThreadMemoryPolicy)(deps.threads, id, requested, expectedRevision, validate)
      deps.publish(thread)
      return thread
    } finally { lease.release() }
  }
}
