export interface TranscriptRange { start: number; end: number }
export const TRANSCRIPT_PAGE = 150

export interface TranscriptWindowState { ranges: TranscriptRange[]; total: number }

export function restoreTranscriptRanges(saved: TranscriptWindowState | undefined, total: number): TranscriptRange[] {
  if (!saved || saved.total === 0 || total < saved.total) return [{ start: Math.max(0, total - TRANSCRIPT_PAGE), end: total }]
  return includeTranscriptRange(saved.ranges, saved.total, total, total)
}

export function includeTranscriptRange(ranges: TranscriptRange[], start: number, end: number, total: number): TranscriptRange[] {
  const sorted = [...ranges, { start, end }].map(range => ({ start: Math.max(0, Math.min(total, range.start)), end: Math.max(0, Math.min(total, range.end)) }))
    .filter(range => range.start < range.end).sort((a, b) => a.start - b.start)
  const result: TranscriptRange[] = []
  for (const range of sorted) {
    const last = result.at(-1)
    if (last && range.start <= last.end) last.end = Math.max(last.end, range.end)
    else result.push({ ...range })
  }
  return result
}
export function transcriptSegments(total: number, ranges: TranscriptRange[]): Array<TranscriptRange & { kind: 'visible' | 'gap' }> {
  const segments: Array<TranscriptRange & { kind: 'visible' | 'gap' }> = []
  let offset = 0
  for (const range of includeTranscriptRange(ranges, 0, 0, total)) {
    if (range.start > offset) segments.push({ kind: 'gap', start: offset, end: range.start })
    segments.push({ kind: 'visible', ...range }); offset = range.end
  }
  if (offset < total) segments.push({ kind: 'gap', start: offset, end: total })
  return segments
}
