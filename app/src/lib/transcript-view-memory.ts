import type { TranscriptPosition } from './transcript-scroll'
import type { TranscriptWindowState } from './transcript-window'

interface TranscriptViewState {
  position?: TranscriptPosition
  window?: TranscriptWindowState
}

// Keep only viewport coordinates, row IDs and loaded ranges for this renderer session.
export function createTranscriptViewMemory(limit = 64) {
  const entries = new Map<string, TranscriptViewState>()
  return {
    get(threadId: string): TranscriptViewState | undefined { return entries.get(threadId) },
    set(threadId: string, patch: Partial<TranscriptViewState>): void {
      const next = { ...entries.get(threadId), ...patch }
      entries.delete(threadId)
      entries.set(threadId, next)
      while (entries.size > limit) entries.delete(entries.keys().next().value!)
    },
  }
}

export const transcriptViewMemory = createTranscriptViewMemory()
