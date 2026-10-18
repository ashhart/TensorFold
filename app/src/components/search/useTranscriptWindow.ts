import { useLayoutEffect, useRef, useState, type RefObject } from 'react'
import type { TurnRow } from '../../lib/turn-blocks'
import { includeTranscriptRange, transcriptSegments, TRANSCRIPT_PAGE, type TranscriptRange } from '../../lib/transcript-window'

/** Pages add DOM without evicting selected text; streaming appends never rebuild older pages. */
export function useTranscriptWindow(threadId: string, rows: TurnRow[], targetId: string | null, scrollRef: RefObject<HTMLDivElement | null>) {
  const [state, setState] = useState<{ id: string; ranges: TranscriptRange[]; total: number }>(() => ({ id: threadId, ranges: [{ start: Math.max(0, rows.length - TRANSCRIPT_PAGE), end: rows.length }], total: rows.length }))
  const previousHeight = useRef<{ height: number; top: number } | null>(null)
  const initial = [{ start: Math.max(0, rows.length - TRANSCRIPT_PAGE), end: rows.length }]
  const own = state.id === threadId
  // Explicit edits/reverts can replace a long branch with a shorter active timeline.
  let ranges = own && rows.length >= state.total ? state.ranges : initial
  if (own && rows.length > state.total) ranges = includeTranscriptRange(ranges, state.total, rows.length, rows.length)
  const target = targetId ? rows.findIndex(row => row.type === 'item' ? row.item.id === targetId : row.type === 'work' && row.turn.work.some(item => item.id === targetId)) : -1
  if (target >= 0) ranges = includeTranscriptRange(ranges, Math.max(0, target - 15), Math.min(rows.length, target + TRANSCRIPT_PAGE - 15), rows.length)
  useLayoutEffect(() => {
    if (state.id !== threadId || state.total !== rows.length || JSON.stringify(state.ranges) !== JSON.stringify(ranges)) setState({ id: threadId, ranges, total: rows.length })
    const anchor = previousHeight.current, element = scrollRef.current
    if (anchor && element) { element.scrollTop = anchor.top + element.scrollHeight - anchor.height; previousHeight.current = null }
  })
  const load = (start: number, end: number) => {
    const element = scrollRef.current
    if (element) previousHeight.current = { height: element.scrollHeight, top: element.scrollTop }
    setState({ id: threadId, ranges: includeTranscriptRange(ranges, start, end, rows.length), total: rows.length })
  }
  return { segments: transcriptSegments(rows.length, ranges), load, pageSize: TRANSCRIPT_PAGE }
}
