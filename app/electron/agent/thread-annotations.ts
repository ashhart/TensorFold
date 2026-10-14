// Thread annotations (VST-040): pinned messages (with labels + done state), thread markers,
// and freeform notes. Pure reducers over a plain-JSON annotation record so the host persists it and the
// renderer renders it; no I/O, deterministic, serialize/restore round-trips.

export interface PinnedMessage {
  id: string
  // The timeline item this pin points at (so the UI can scroll/highlight it).
  itemId: string
  label: string | null
  done: boolean
  createdAt: number
}

export interface ThreadMarker {
  id: string
  itemId: string
  label: string
  done: boolean
  createdAt: number
}

export interface ThreadAnnotations {
  pinned: PinnedMessage[]
  markers: ThreadMarker[]
  notes: string
}

export function emptyAnnotations(): ThreadAnnotations {
  return { pinned: [], markers: [], notes: '' }
}

export interface AnnotationDeps {
  now: () => number
  newId: () => string
}

// ---- pinned messages ----

// Pin a timeline item. Idempotent by itemId — pinning the same message twice updates its
// label instead of creating a duplicate pin (a message is pinned or not, once).
export function pinMessage(state: ThreadAnnotations, itemId: string, label: string | null, deps: AnnotationDeps): ThreadAnnotations {
  const existing = state.pinned.find((pin) => pin.itemId === itemId)
  if (existing) {
    return { ...state, pinned: state.pinned.map((pin) => pin.itemId === itemId ? { ...pin, label: label ?? pin.label } : pin) }
  }
  const pin: PinnedMessage = { id: deps.newId(), itemId, label: label ?? null, done: false, createdAt: deps.now() }
  return { ...state, pinned: [...state.pinned, pin] }
}

export function unpinMessage(state: ThreadAnnotations, id: string): ThreadAnnotations {
  return { ...state, pinned: state.pinned.filter((pin) => pin.id !== id) }
}

export function setPinnedLabel(state: ThreadAnnotations, id: string, label: string | null): ThreadAnnotations {
  return { ...state, pinned: state.pinned.map((pin) => pin.id === id ? { ...pin, label: label ?? null } : pin) }
}

export function setPinnedDone(state: ThreadAnnotations, id: string, done: boolean): ThreadAnnotations {
  return { ...state, pinned: state.pinned.map((pin) => pin.id === id ? { ...pin, done } : pin) }
}

// ---- markers ----

export function addMarker(state: ThreadAnnotations, itemId: string, label: string, deps: AnnotationDeps): ThreadAnnotations {
  const marker: ThreadMarker = { id: deps.newId(), itemId, label: label.trim() || 'Marker', done: false, createdAt: deps.now() }
  return { ...state, markers: [...state.markers, marker] }
}

export function removeMarker(state: ThreadAnnotations, id: string): ThreadAnnotations {
  return { ...state, markers: state.markers.filter((marker) => marker.id !== id) }
}

export function setMarkerDone(state: ThreadAnnotations, id: string, done: boolean): ThreadAnnotations {
  return { ...state, markers: state.markers.map((marker) => marker.id === id ? { ...marker, done } : marker) }
}

export function setMarkerLabel(state: ThreadAnnotations, id: string, label: string): ThreadAnnotations {
  return { ...state, markers: state.markers.map((marker) => marker.id === id ? { ...marker, label: label.trim() || marker.label } : marker) }
}

// ---- notes ----

export function setNotes(state: ThreadAnnotations, notes: string): ThreadAnnotations {
  return { ...state, notes: typeof notes === 'string' ? notes : '' }
}

// ---- pruning + persistence ----

// Drop pins/markers whose target timeline item no longer exists (e.g. after a turn revert
// truncated the timeline), so a re-render never dangles a pin over a deleted message.
export function pruneAnnotations(state: ThreadAnnotations, liveItemIds: Iterable<string>): ThreadAnnotations {
  const live = new Set(liveItemIds)
  return {
    pinned: state.pinned.filter((pin) => live.has(pin.itemId)),
    markers: state.markers.filter((marker) => live.has(marker.itemId)),
    notes: state.notes,
  }
}

export function serializeAnnotations(state: ThreadAnnotations): string {
  return JSON.stringify(state)
}

// Restore defensively: a corrupt/absent blob yields empty annotations, and each record is
// normalized so a hand-edited or older file can't crash the renderer.
export function restoreAnnotations(raw: string | null | undefined): ThreadAnnotations {
  if (!raw) return emptyAnnotations()
  let parsed: Partial<ThreadAnnotations>
  try { parsed = JSON.parse(raw) } catch { return emptyAnnotations() }
  const pinned = Array.isArray(parsed.pinned) ? parsed.pinned.filter((pin) => pin && typeof pin.id === 'string' && typeof pin.itemId === 'string').map((pin) => ({
    id: pin.id, itemId: pin.itemId, label: typeof pin.label === 'string' ? pin.label : null, done: !!pin.done, createdAt: Number(pin.createdAt) || 0,
  })) : []
  const markers = Array.isArray(parsed.markers) ? parsed.markers.filter((marker) => marker && typeof marker.id === 'string' && typeof marker.itemId === 'string').map((marker) => ({
    id: marker.id, itemId: marker.itemId, label: typeof marker.label === 'string' ? marker.label : 'Marker', done: !!marker.done, createdAt: Number(marker.createdAt) || 0,
  })) : []
  return { pinned, markers, notes: typeof parsed.notes === 'string' ? parsed.notes : '' }
}
