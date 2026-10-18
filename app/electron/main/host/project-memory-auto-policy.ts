import { isActiveProjectMemoryEntry } from '../../agent/project-memory'
import { authoredMemoryMessages, harvestMemoryCandidates } from '../../agent/memory-harvest'
import type { ProjectMemoryEntry, Thread } from '../../agent/types'

/** Suppress an exact replay without muting later turns or updated source files. */
export function projectMemoryAutoSourceSuppressed(entries: ProjectMemoryEntry[], source: { projectId: string; sourceThreadId?: string | null; sourceTurn?: number | null; sourceRef?: string | null }): boolean {
  return entries.some((entry) => entry.projectId === source.projectId && !isActiveProjectMemoryEntry(entry)
    && ((!!source.sourceThreadId && Number.isSafeInteger(source.sourceTurn) && source.sourceTurn! > 0 && entry.sourceThreadId === source.sourceThreadId && entry.sourceTurn === source.sourceTurn)
      || (!!source.sourceRef && /^(?:connector|connector-event):[^:]+$/.test(source.sourceRef) && entry.sourceType === 'connector' && entry.sourceRef === source.sourceRef)))
}

/** Retain authored user turns, except exact origins of facts currently in Trash. */
export function projectMemoryAutoMessages(entries: ProjectMemoryEntry[], thread: Thread): Array<{ role: string; content: string; sourceTurn: number }> {
  let sourceTurn = 0
  const messages: Array<{ role: string; content: string; sourceTurn: number }> = []
  for (const item of thread.timeline ?? []) {
    if (item.kind === 'user') sourceTurn++
    if (thread.projectId && projectMemoryAutoSourceSuppressed(entries, { projectId: thread.projectId, sourceThreadId: thread.id, sourceTurn })) continue
    for (const message of authoredMemoryMessages({ ...thread, timeline: [item] })) {
      messages.push({ role: message.role, content: String(message.content ?? ''), sourceTurn })
    }
  }
  return messages
}

/** Keep original turn provenance when old history is salvaged during a later run. */
export function projectMemoryAutoCandidateTurn(messages: ReturnType<typeof projectMemoryAutoMessages>, text: string): number | null {
  for (const message of [...messages].reverse()) {
    if (harvestMemoryCandidates([message], Number.MAX_SAFE_INTEGER).includes(text)) return message.sourceTurn
  }
  return null
}
