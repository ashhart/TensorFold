export interface TranscriptPosition {
  top: number
  following: boolean
  anchor?: { id: string; offset: number }
}

export function transcriptAtEnd(element: HTMLElement): boolean {
  return element.scrollHeight - element.scrollTop - element.clientHeight <= 24
}

export function captureTranscriptPosition(element: HTMLElement, following: boolean): TranscriptPosition {
  const rows = element.querySelectorAll<HTMLElement>('[data-transcript-row]')
  const viewport = element.getBoundingClientRect()
  let low = 0, high = rows.length
  while (low < high) {
    const middle = Math.floor((low + high) / 2)
    if (rows[middle].getBoundingClientRect().bottom <= viewport.top + 1) low = middle + 1
    else high = middle
  }
  const row = rows[low]
  const id = row?.dataset.transcriptRow
  return {
    top: element.scrollTop,
    following,
    ...(id ? { anchor: { id, offset: row.getBoundingClientRect().top - viewport.top } } : {}),
  }
}

export function restoreTranscriptPosition(element: HTMLElement, saved?: TranscriptPosition): void {
  const maximum = Math.max(0, element.scrollHeight - element.clientHeight)
  if (!saved || saved.following) { element.scrollTop = maximum; return }
  const row = saved.anchor ? Array.from(element.querySelectorAll<HTMLElement>('[data-transcript-row]'))
    .find((candidate) => candidate.dataset.transcriptRow === saved.anchor!.id) : undefined
  const top = row && saved.anchor
    ? element.scrollTop + row.getBoundingClientRect().top - element.getBoundingClientRect().top - saved.anchor.offset
    : saved.top
  element.scrollTop = Math.max(0, Math.min(maximum, top))
}
