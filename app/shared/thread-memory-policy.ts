import type { Provider, Room, Thread, ThreadMemoryPolicy } from '../electron/agent/types'

type MemoryThread = Pick<Thread, 'memoryPolicy' | 'incognito'>

/** Absent legacy choices preserve ordinary saved-chat behavior. Present choices are exact. */
export function parseThreadMemoryPolicy(value: unknown): ThreadMemoryPolicy {
  if (value === undefined) return { useExisting: true, contribute: true }
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Memory choices must contain two boolean values.')
  const input = value as Record<string, unknown>
  if (Object.keys(input).length !== 2 || !Object.hasOwn(input, 'useExisting') || !Object.hasOwn(input, 'contribute') || typeof input.useExisting !== 'boolean' || typeof input.contribute !== 'boolean') throw new Error('Memory choices must contain only useExisting and contribute booleans.')
  return { useExisting: input.useExisting, contribute: input.contribute }
}

/** A captured run/parent choice can narrow live state, never expand it. */
export function effectiveThreadMemoryPolicy(thread: MemoryThread, captured?: ThreadMemoryPolicy): ThreadMemoryPolicy {
  const saved = parseThreadMemoryPolicy(thread.memoryPolicy)
  const run = captured === undefined ? saved : parseThreadMemoryPolicy(captured)
  return { useExisting: saved.useExisting && run.useExisting, contribute: !thread.incognito && saved.contribute && run.contribute }
}

export function captureThreadMemoryPolicy(thread: MemoryThread, inherited?: ThreadMemoryPolicy): Readonly<ThreadMemoryPolicy> {
  return Object.freeze(effectiveThreadMemoryPolicy(thread, inherited))
}

export function threadMemoryPolicyRevision(thread: Pick<Thread, 'memoryPolicyRevision'>): number {
  const value = thread.memoryPolicyRevision
  if (value === undefined) return 0
  if (!Number.isSafeInteger(value) || value < 0) throw new Error('The saved memory-choice revision is invalid.')
  return value
}

export function threadMemoryUnsupportedReason(thread: Pick<Thread, 'engine' | 'orchestratorHostId'>, provider: Pick<Provider, 'transport'> | undefined): string | null {
  if (thread.orchestratorHostId) return 'This remote host does not advertise saved-memory policy enforcement. Choose This Mac for this chat.'
  if (thread.engine && thread.engine !== 'native') return 'Saved-memory controls require TensorFold’s native API-model engine.'
  if (!provider || provider.transport != null && provider.transport !== 'chat-completions') return 'Saved-memory controls require an API-based model. CLI providers keep their own context.'
  return null
}

/** Existing unrestricted CLI conversations stay compatible; requested restrictions fail closed. */
export function assertThreadMemoryTransport(thread: MemoryThread & Pick<Thread, 'engine' | 'orchestratorHostId'>, provider: Pick<Provider, 'transport'> | undefined, requested?: ThreadMemoryPolicy): void {
  const policy = effectiveThreadMemoryPolicy(thread, requested)
  if (policy.useExisting && policy.contribute) return
  const reason = threadMemoryUnsupportedReason(thread, provider)
  if (reason) throw new Error(reason)
}

/** Only an unchanged, live owner may file a proposal after asynchronous work. */
export function canContributeThreadMemory(thread: Thread, captured: ThreadMemoryPolicy, revision: number, signal?: AbortSignal): boolean {
  return !signal?.aborted && !thread.archivedAt && threadMemoryPolicyRevision(thread) === revision && effectiveThreadMemoryPolicy(thread, captured).contribute
}
export function assertSavedMemoryRead(thread: MemoryThread, captured?: ThreadMemoryPolicy): void {
  if (!effectiveThreadMemoryPolicy(thread, captured).useExisting) throw new Error('Use saved memory is off for this chat. Prior chats and saved memories are unavailable.')
}
export function assertMemoryContribution(thread: MemoryThread, captured?: ThreadMemoryPolicy): void {
  if (!effectiveThreadMemoryPolicy(thread, captured).contribute) throw new Error('Propose new memories is off for this chat. Learned skill and memory proposals are unavailable.')
}

/** Display damaged metadata without treating unknown choices as enabled or disabled. */
export function inspectThreadMemoryPolicy(thread: MemoryThread & Pick<Thread, 'memoryPolicyRevision'>): { policy: ThreadMemoryPolicy | null; revision: number | null; problem: string | null } {
  let revision: number
  try { revision = threadMemoryPolicyRevision(thread) }
  catch { return { policy: null, revision: null, problem: 'The saved memory-choice revision is damaged. Start a new chat to choose memory options; this history remains available.' } }
  try { return { policy: parseThreadMemoryPolicy(thread.memoryPolicy), revision, problem: null } }
  catch { return { policy: null, revision, problem: 'Saved memory choices need review. Save both choices explicitly before continuing this chat.' } }
}

const operationalRoomEventKinds = new Set(['task', 'status', 'notice', 'error', 'assign'])
export function isOperationalRoomEventKind(value: unknown): boolean {
  return typeof value === 'string' && operationalRoomEventKinds.has(value)
}

/** Keep explicit task/status records, not previous agent conversations or scratch memory. */
export function roomWithoutSavedRecall(room: Room): Room {
  return { ...room, memory: [], memorySummary: '', sharedMemory: undefined, finalText: '', events: (room.events ?? []).filter((event) => isOperationalRoomEventKind(event.kind)) }
}
