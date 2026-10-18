import type { TimelineItem } from '../../lib/types'
import type { ThreadAnnotations } from '../../lib/thread-annotations'

export type OutlineKind = 'message' | 'answer' | 'outcome' | 'pin'
export interface ConversationOutlineEntry { itemId: string; kind: OutlineKind; label: string; ordinal: number }
export const OUTLINE_PAGE_SIZE = 60
export const OUTLINE_SCAN_LIMIT = 50_000

const snippet = (text: string) => text.slice(0, 2000).replace(/\s+/g, ' ').trim().slice(0, 180)

/** No generated summaries, extra storage or model calls; every row names an active exact ID. */
export function buildConversationOutline(timeline: readonly TimelineItem[], annotations: ThreadAnnotations, options: { activeAssistantId?: string } = {}) {
  const entries: ConversationOutlineEntry[] = []
  const pins = new Map(annotations.pinned.map(pin => [pin.itemId, pin]))
  let lastAnswer = -1
  const add = (item: TimelineItem, kind: OutlineKind, ordinal: number, content = '') => {
    const pin = pins.get(item.id)
    entries.push({ itemId: item.id, kind: pin ? 'pin' : kind, label: snippet(pin?.label || content || item.text || item.title || item.output || ''), ordinal })
  }
  const bounded = Math.min(timeline.length, OUTLINE_SCAN_LIMIT)
  // The final assistant before the next user is an answer only when no later tool/reasoning remains.
  for (let index = 0; index < bounded; index++) {
    const item = timeline[index]
    if (item.kind === 'user') {
      if (lastAnswer >= 0 && timeline[lastAnswer].id !== options.activeAssistantId) add(timeline[lastAnswer], 'answer', lastAnswer)
      lastAnswer = -1
      add(item, 'message', index)
    } else if (item.kind === 'assistant') {
      if (pins.has(item.id)) { add(item, 'pin', index); lastAnswer = -1 }
      else lastAnswer = index
    } else {
      if (item.kind === 'tool' || item.kind === 'reasoning') lastAnswer = -1
      if (pins.has(item.id)) add(item, 'pin', index)
      else if (item.kind === 'error' || item.kind === 'tool' && (item.state === 'error' || item.state === 'denied'
        || (item.state === 'ok' && (!!item.changedPaths?.length || !!item.diff?.length || !!item.artifacts?.outputFiles)))) {
        add(item, 'outcome', index, item.title || item.command || item.text || item.output || item.tool || '')
      }
    }
  }
  if (lastAnswer >= 0 && timeline[lastAnswer].id !== options.activeAssistantId) add(timeline[lastAnswer], 'answer', lastAnswer)
  entries.sort((a, b) => a.ordinal - b.ordinal)
  return { entries, truncated: timeline.length > bounded }
}

export function outlinePage(entries: readonly ConversationOutlineEntry[], input: { kind: 'all' | OutlineKind; query: string; page: number }) {
  const query = input.query.normalize('NFKC').toLocaleLowerCase().trim()
  const filtered = entries.filter(entry => (input.kind === 'all' || input.kind === entry.kind) && (!query || entry.label.normalize('NFKC').toLocaleLowerCase().includes(query)))
  const pages = Math.max(1, Math.ceil(filtered.length / OUTLINE_PAGE_SIZE))
  const page = Math.min(pages - 1, Math.max(0, Math.floor(input.page) || 0))
  return { entries: filtered.slice(page * OUTLINE_PAGE_SIZE, (page + 1) * OUTLINE_PAGE_SIZE), total: filtered.length, page, pages }
}
