import { useLayoutEffect, useRef, useState, type RefObject } from 'react'
import type { TurnRow } from '../../lib/turn-blocks'
import { includeTranscriptRange, restoreTranscriptRanges, transcriptSegments, TRANSCRIPT_PAGE, type TranscriptRange } from '../../lib/transcript-window'
import { transcriptViewMemory } from '../../lib/transcript-view-memory'

/** Pages add DOM without evicting selected text; streaming appends never rebuild older pages. */
export function useTranscriptWindow(threadId: string, rows: TurnRow[], targetId: string | null, scrollRef: RefObject<HTMLDivElement | null>) {
  const [state, setState] = useState<{ id: string; ranges: TranscriptRange[]; total: number }>(() => ({ id: threadId, ranges: restoreTranscriptRanges(transcriptViewMemory.get(threadId)?.window, rows.length), total: rows.length }))
  const previousHeight = useRef<{ id: string; height: number; top: number } | null>(null)
  const own = state.id === threadId
  // Explicit edits/reverts can replace a long branch with a shorter active timeline.
  let ranges = restoreTranscriptRanges(own ? state : transcriptViewMemory.get(threadId)?.window, rows.length)
  const target = targetId ? rows.findIndex(row => row.type === 'item' ? row.item.id === targetId : row.type === 'work' && row.turn.work.some(item => item.id === targetId)) : -1
  if (target >= 0) ranges = includeTranscriptRange(ranges, Math.max(0, target - 15), Math.min(rows.length, target + TRANSCRIPT_PAGE - 15), rows.length)
  useLayoutEffect(() => {
    if (state.id !== threadId || state.total !== rows.length || JSON.stringify(state.ranges) !== JSON.stringify(ranges)) setState({ id: threadId, ranges, total: rows.length })
    transcriptViewMemory.set(threadId, { window: { ranges, total: rows.length } })
    const anchor = previousHeight.current, element = scrollRef.current
    if (anchor && anchor.id === threadId && element) element.scrollTop = anchor.top + element.scrollHeight - anchor.height
    previousHeight.current = null
  })
  const load = (start: number, end: number) => {
    const element = scrollRef.current
    if (element) previousHeight.current = { id: threadId, height: element.scrollHeight, top: element.scrollTop }
    setState({ id: threadId, ranges: includeTranscriptRange(ranges, start, end, rows.length), total: rows.length })
  }
  return { segments: transcriptSegments(rows.length, ranges), load, pageSize: TRANSCRIPT_PAGE }
}
