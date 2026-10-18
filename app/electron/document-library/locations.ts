import type { DocumentLocation, DocumentSection } from '../../shared/document-library'

const CHUNK = 1_800
const OVERLAP = 256
const MAX_SECTIONS = 10_000
export function documentSections(content: string, extension: string): DocumentSection[] {
  if (extension === '.pdf') return splitSection(content, { kind: 'extracted', reason: 'PDF text extraction has no verified page mapping; page coordinates are unavailable.', charStart: 0, charEnd: content.length })
  if (extension === '.csv' || extension === '.tsv') return delimitedSections(content, extension === '.csv' ? ',' : '\t')
  const headings: Array<{ offset: number; title?: string }> = [{ offset: 0 }]
  for (const match of content.matchAll(/^#{1,6}\s+(.+)$/gm)) {
    if (headings.length >= MAX_SECTIONS) throw new Error('Document has too many sections; split the file into smaller documents and retry.')
    if (match.index === 0) headings[0].title = match[1].trim()
    else headings.push({ offset: match.index!, title: match[1].trim() })
  }
  const sections: DocumentSection[] = []
  let consumed = 0
  let startLine = 1
  for (let i = 0; i < headings.length; i++) {
    const heading = headings[i]
    const boundary = headings[i + 1]?.offset ?? content.length
    startLine += (content.slice(consumed, heading.offset).match(/\n/g)?.length ?? 0)
    consumed = heading.offset
    let lineStart = startLine
    for (let offset = heading.offset; offset < boundary; offset += CHUNK - OVERLAP) {
      const end = Math.min(boundary, offset + CHUNK)
      const text = content.slice(offset, end)
      const newlines = text.match(/\n/g)?.length ?? 0
      sections.push({ id: `text-${offset}-${end}`, text, location: { kind: 'text', section: heading.title, lineStart, lineEnd: lineStart + Math.max(0, newlines - (text.endsWith('\n') ? 1 : 0)), charStart: offset, charEnd: end } })
      if (sections.length > MAX_SECTIONS) throw new Error('Document has too many indexed passages; split the file and retry.')
      if (end === boundary) break
      lineStart += (content.slice(offset, end - OVERLAP).match(/\n/g)?.length ?? 0)
    }
  }
  return sections
}

/** CSV coordinates count logical records, including the header, even for quoted newlines. */
function delimitedSections(content: string, delimiter: string): DocumentSection[] {
  const sections: DocumentSection[] = []
  let quoted = false
  let start = 0
  let row = 1
  let columns = 1
  for (let i = 0; i <= content.length; i++) {
    const ch = content[i]
    if (ch === '"') {
      if (quoted && content[i + 1] === '"') i++
      else quoted = !quoted
    }
    if (!quoted && ch === delimiter) columns++
    if ((!quoted && ch === '\n') || i === content.length) {
      if (quoted) throw new Error('Delimited table has an unterminated quoted field.')
      if (i > start) {
        const previous = sections[sections.length - 1]
        if (sections.length >= 1_000 && previous?.location.kind === 'table' && previous.text.length + i - start + 1 <= CHUNK && previous.location.charEnd + 1 === start) {
          previous.text += `\n${content.slice(start, i)}`
          previous.location.charEnd = i
          previous.location.rowEnd = row
          previous.location.columnEnd = Math.max(previous.location.columnEnd, columns)
          previous.id = `table-${previous.location.charStart}-${i}`
        } else sections.push(...splitSection(content.slice(start, i), { kind: 'table', rowStart: row, rowEnd: row, columnStart: 1, columnEnd: columns, charStart: start, charEnd: i }))
        if (sections.length > MAX_SECTIONS) throw new Error('Table has too many indexed passages; split the file and retry.')
      }
      row++
      columns = 1
      start = i + 1
    }
  }
  return sections
}

export function splitSection(text: string, location: DocumentLocation): DocumentSection[] {
  const sections: DocumentSection[] = []
  for (let offset = 0; offset < text.length; offset += CHUNK - OVERLAP) {
    const end = Math.min(text.length, offset + CHUNK)
    const next = { ...location, charStart: location.charStart + offset, charEnd: location.charStart + end }
    sections.push({ id: `${location.kind}-${next.charStart}-${next.charEnd}`, text: text.slice(offset, end), location: next })
    if (end === text.length) break
  }
  return sections
}
