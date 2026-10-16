// Terminal multiplex + flow control (VST-039): the pure core behind ack-based
// terminal output, replay-safe ANSI-aware scrollback capping, and the
// workspace layout presets (single / two-columns / two-rows / top-main / left-main / four).
// No PTY, no I/O — the host feeds PTY bytes in and reads replay/backpressure decisions out,
// so all three are deterministically testable.

// ---- 1. ack-based output buffer (flow control + reconnect replay) ----

export interface OutputChunk {
  seq: number
  data: string
}

export interface ReplaySince {
  chunks: OutputChunk[]
  // True when the requested cursor fell BELOW the retained window (the buffer rolled past it
  // — slow/absent consumer, or a long disconnect): the client must clear and take the live
  // tail instead of trusting a partial replay. Mirrors the spine's snapshotRequired.
  gap: boolean
  // The latest sequence in the buffer (the client's new cursor after consuming `chunks`).
  head: number
}

// A bounded, sequence-numbered output ring. Two forces evict the oldest chunks: an ack
// (consumer confirmed receipt) and the byte cap (backpressure safety, so an unacking client
// can't grow the buffer without bound). Replaying before the retained window reports a gap.
export class TerminalOutputBuffer {
  private chunks: OutputChunk[] = []
  private nextSeq = 1
  private ackedThrough = 0
  private bytes = 0
  private readonly maxBytes: number

  constructor(options: { maxBytes?: number } = {}) {
    this.maxBytes = Math.max(1, options.maxBytes ?? 256 * 1024)
  }

  get head(): number {
    return this.nextSeq - 1
  }

  get unackedBytes(): number {
    let total = 0
    for (const chunk of this.chunks) if (chunk.seq > this.ackedThrough) total += chunk.data.length
    return total
  }

  get retainedBytes(): number {
    return this.bytes
  }

  // Append PTY output; assigns the next sequence and evicts the oldest chunks past maxBytes.
  append(data: string): OutputChunk {
    const chunk: OutputChunk = { seq: this.nextSeq++, data }
    this.chunks.push(chunk)
    this.bytes += data.length
    this.evictToCap()
    return chunk
  }

  // Consumer confirms receipt through `seq`: drop everything at or below it (it will never be
  // replayed again), freeing the buffer. Idempotent + monotonic (a stale lower ack is a no-op).
  ack(seq: number): void {
    if (seq <= this.ackedThrough) return
    this.ackedThrough = Math.min(seq, this.head)
    this.dropAcked()
  }

  // Replay everything after `cursor` for a reconnecting client. gap=true when the cursor is
  // older than the oldest retained chunk (data was evicted) → the client re-baselines.
  since(cursor: number): ReplaySince {
    const oldest = this.chunks.length ? this.chunks[0].seq : this.head + 1
    const gap = cursor + 1 < oldest && cursor < this.head
    return { chunks: this.chunks.filter((chunk) => chunk.seq > cursor), gap, head: this.head }
  }

  // Backpressure: is the un-acked backlog past the threshold? The host uses this to pause
  // reading the PTY (or warn) rather than let a wedged client force unbounded eviction.
  backpressured(thresholdBytes: number): boolean {
    return this.unackedBytes >= thresholdBytes
  }

  private dropAcked(): void {
    while (this.chunks.length && this.chunks[0].seq <= this.ackedThrough) {
      this.bytes -= this.chunks[0].data.length
      this.chunks.shift()
    }
  }

  private evictToCap(): void {
    while (this.bytes > this.maxBytes && this.chunks.length > 1) {
      this.bytes -= this.chunks[0].data.length
      this.chunks.shift()
    }
  }
}

// ---- 2. ANSI-aware scrollback capping ----

// eslint-disable-next-line no-control-regex
const ANSI_ESCAPE = /\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])/g

// Trim `text` from the FRONT to at most maxBytes, but never cut inside an ANSI escape
// sequence — a half-escape corrupts every subsequent line's rendering. If the naive cut point
// lands inside an escape, advance it to the end of that sequence. Returns the retained tail.
export function capScrollbackAnsiSafe(text: string, maxBytes: number): string {
  if (text.length <= maxBytes) return text
  const naiveCut = text.length - maxBytes
  // Find any escape sequence straddling the cut point and move the cut to its end.
  ANSI_ESCAPE.lastIndex = 0
  let cut = naiveCut
  let match: RegExpExecArray | null
  while ((match = ANSI_ESCAPE.exec(text))) {
    const start = match.index
    const end = start + match[0].length
    if (start < naiveCut && end > naiveCut) { cut = end; break }
    if (start >= naiveCut) break
  }
  return text.slice(cut)
}

// ---- 3. layout presets ----

export type LayoutPreset = 'single' | 'two-columns' | 'two-rows' | 'top-main' | 'left-main' | 'four'

export interface Pane {
  id: string
  terminalId: string | null
  label: string | null
  suspended: boolean
}

export interface LayoutState {
  preset: LayoutPreset
  panes: Pane[]
  activePaneId: string
}

const PRESET_SLOTS: Record<LayoutPreset, number> = {
  'single': 1,
  'two-columns': 2,
  'two-rows': 2,
  'top-main': 3,
  'left-main': 3,
  'four': 4,
}

export function presetSlots(preset: LayoutPreset): number {
  return PRESET_SLOTS[preset] ?? 1
}

let paneCounter = 0
function makePane(newId: () => string): Pane {
  return { id: newId(), terminalId: null, label: null, suspended: false }
}

export function initLayout(newId: () => string = () => `pane-${++paneCounter}`): LayoutState {
  const pane = makePane(newId)
  return { preset: 'single', panes: [pane], activePaneId: pane.id }
}

// Switch preset: grow (append empty panes) or shrink (drop trailing panes, reassigning active
// if it was dropped) to the preset's slot count. Preserves existing panes' terminal bindings.
export function applyPreset(state: LayoutState, preset: LayoutPreset, newId: () => string = () => `pane-${++paneCounter}`): LayoutState {
  const want = presetSlots(preset)
  const panes = state.panes.slice(0, want)
  while (panes.length < want) panes.push(makePane(newId))
  const activePaneId = panes.some((pane) => pane.id === state.activePaneId) ? state.activePaneId : panes[0].id
  return { preset, panes, activePaneId }
}

// Split: move to the next-larger preset that fits one more pane; the new pane becomes active.
export function splitPane(state: LayoutState, newId: () => string = () => `pane-${++paneCounter}`): LayoutState {
  const order: LayoutPreset[] = ['single', 'two-columns', 'top-main', 'four']
  const next = order.find((preset) => presetSlots(preset) > state.panes.length) ?? state.preset
  const grown = applyPreset(state, next, newId)
  const fresh = grown.panes.find((pane) => !state.panes.some((old) => old.id === pane.id))
  return { ...grown, activePaneId: fresh?.id ?? grown.activePaneId }
}

export function assignTerminal(state: LayoutState, paneId: string, terminalId: string | null, label?: string | null): LayoutState {
  return { ...state, panes: state.panes.map((pane) => pane.id === paneId ? { ...pane, terminalId, label: label ?? pane.label } : pane) }
}

export function setActivePane(state: LayoutState, paneId: string): LayoutState {
  return state.panes.some((pane) => pane.id === paneId) ? { ...state, activePaneId: paneId } : state
}

// Suspend/resume a pane's terminal (keeps the slot + binding; the PTY is paused, not closed).
export function suspendPane(state: LayoutState, paneId: string, suspended: boolean): LayoutState {
  return { ...state, panes: state.panes.map((pane) => pane.id === paneId ? { ...pane, suspended } : pane) }
}

export function renamePane(state: LayoutState, paneId: string, label: string): LayoutState {
  return { ...state, panes: state.panes.map((pane) => pane.id === paneId ? { ...pane, label: label.trim() || null } : pane) }
}

// Persist to plain JSON and restore (layout persistence). Restore drops panes whose terminal
// is no longer live (`liveTerminalIds`), then normalizes the pane count back to the preset so
// a reload never shows a dead/blank slot as bound.
export function serializeLayout(state: LayoutState): string {
  return JSON.stringify(state)
}

export function restoreLayout(raw: string, liveTerminalIds: Iterable<string>, newId: () => string = () => `pane-${++paneCounter}`): LayoutState {
  const live = new Set(liveTerminalIds)
  let parsed: LayoutState
  try {
    parsed = JSON.parse(raw)
  } catch {
    return initLayout(newId)
  }
  if (!parsed || !Array.isArray(parsed.panes) || !PRESET_SLOTS[parsed.preset]) return initLayout(newId)
  const panes = parsed.panes.map((pane) => ({
    id: typeof pane.id === 'string' ? pane.id : newId(),
    terminalId: pane.terminalId && live.has(pane.terminalId) ? pane.terminalId : null,
    label: typeof pane.label === 'string' ? pane.label : null,
    suspended: !!pane.suspended,
  }))
  const normalized = applyPreset({ preset: parsed.preset, panes, activePaneId: parsed.activePaneId ?? panes[0]?.id ?? '' }, parsed.preset, newId)
  return normalized
}
