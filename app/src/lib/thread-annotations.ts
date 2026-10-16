// Thread annotations, renderer side (VST-040): pinned messages + freeform notes over the
// serialized `Thread.annotations` blob. Mirrors the tested host core
// (electron/agent/thread-annotations.ts) — the host owns pruning on revert; the renderer
// does the interactive pin/unpin/toggle and persists via patchThread. Kept renderer-local so
// the panel needn't import host runtime across the type boundary.

export interface PinnedMessage {
  id: string
  itemId: string
  label: string | null
  done: boolean
  createdAt: number
}

export interface ThreadAnnotations {
  pinned: PinnedMessage[]
  notes: string
}

export function emptyAnnotations(): ThreadAnnotations {
  return { pinned: [], notes: '' }
}

// Parse the serialized blob defensively — a corrupt/partial value yields empty, never throws.
export function parseAnnotations(raw: string | null | undefined): ThreadAnnotations {
  if (!raw) return emptyAnnotations()
  try {
    const parsed = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object') return emptyAnnotations()
    const pinned = Array.isArray(parsed.pinned)
      ? parsed.pinned.filter((p: unknown): p is PinnedMessage => !!p && typeof (p as PinnedMessage).itemId === 'string')
        .map((p: PinnedMessage) => ({ id: String(p.id ?? p.itemId), itemId: p.itemId, label: typeof p.label === 'string' ? p.label : null, done: p.done === true, createdAt: typeof p.createdAt === 'number' ? p.createdAt : 0 }))
      : []
    return { pinned, notes: typeof parsed.notes === 'string' ? parsed.notes : '' }
  } catch {
    return emptyAnnotations()
  }
}

export function serializeAnnotations(state: ThreadAnnotations): string {
  return JSON.stringify(state)
}

// Pin idempotently by itemId (a message is pinned once); repinning updates its label.
export function pinMessage(state: ThreadAnnotations, itemId: string, label: string | null, now: number, newId: () => string): ThreadAnnotations {
  if (state.pinned.some((p) => p.itemId === itemId)) {
    return { ...state, pinned: state.pinned.map((p) => p.itemId === itemId ? { ...p, label: label ?? p.label } : p) }
  }
  return { ...state, pinned: [...state.pinned, { id: newId(), itemId, label, done: false, createdAt: now }] }
}

export function unpinMessage(state: ThreadAnnotations, itemId: string): ThreadAnnotations {
  return { ...state, pinned: state.pinned.filter((p) => p.itemId !== itemId) }
}

export function togglePinnedDone(state: ThreadAnnotations, itemId: string): ThreadAnnotations {
  return { ...state, pinned: state.pinned.map((p) => p.itemId === itemId ? { ...p, done: !p.done } : p) }
}

export function isPinned(state: ThreadAnnotations, itemId: string): boolean {
  return state.pinned.some((p) => p.itemId === itemId)
}

// Drop pins pointing at items a revert removed (keeps the panel honest after a rewind).
export function pruneAnnotations(state: ThreadAnnotations, liveItemIds: Iterable<string>): ThreadAnnotations {
  const live = new Set(liveItemIds)
  const pinned = state.pinned.filter((p) => live.has(p.itemId))
  return pinned.length === state.pinned.length ? state : { ...state, pinned }
}
