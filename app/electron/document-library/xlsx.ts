import path from 'node:path'
import { parseXlsxSharedStrings, readZipEntries, xlsxCellValue, xmlAttr } from '../agent/document-parser'
import { splitSection } from './locations'
import type { DocumentSection } from '../../shared/document-library'

/** Keeps real sheet/row/cell references rather than assigning flattened row numbers. */
export function extractWorkbook(bytes: Buffer, maxBytes: number): { content: string; sections: DocumentSection[] } {
  const entries = readZipEntries(bytes, maxBytes)
  const shared = parseXlsxSharedStrings(entries.get('xl/sharedStrings.xml')?.toString('utf8') ?? '')
  const workbook = entries.get('xl/workbook.xml')?.toString('utf8') ?? ''
  const relationships = entries.get('xl/_rels/workbook.xml.rels')?.toString('utf8') ?? ''
  const paths = new Map<string, string>()
  for (const match of relationships.matchAll(/<Relationship\b([^>]*)\/?\s*>/g)) {
    const id = xmlAttr(match[1], 'Id')
    const target = xmlAttr(match[1], 'Target')
    if (!id || !target || xmlAttr(match[1], 'TargetMode') === 'External') continue
    const resolved = path.posix.normalize(target.startsWith('/') ? target.slice(1) : `xl/${target}`)
    if (/^xl\/worksheets\/[^/]+\.xml$/.test(resolved)) paths.set(id, resolved)
  }
  const sheets = [...workbook.matchAll(/<sheet\b([^>]*)\/?\s*>/g)].map(match => ({ name: xmlAttr(match[1], 'name') ?? '', file: paths.get(xmlAttr(match[1], 'r:id') ?? '') }))
  if (!sheets.length || sheets.some(sheet => !sheet.file)) throw new Error('Workbook sheet relationships are unavailable; exact sheet coordinates cannot be verified.')
  let content = ''
  const sections: DocumentSection[] = []
  for (const sheet of sheets) {
    const xml = entries.get(sheet.file!)?.toString('utf8')
    if (!xml) throw new Error(`Workbook sheet ${sheet.name || sheet.file} is missing.`)
    for (const row of xml.matchAll(/<row\b([^>]*)>([\s\S]*?)<\/row>/g)) {
      const number = Number(xmlAttr(row[1], 'r'))
      if (!Number.isSafeInteger(number) || number < 1 || number > 1_048_576) throw new Error('Workbook row coordinates are unavailable or out of bounds.')
      const values: string[] = []
      let firstColumn = Infinity
      let lastColumn = 0
      for (const cell of row[2].matchAll(/<c\b([^>]*)>([\s\S]*?)<\/c>/g)) {
        const reference = xmlAttr(cell[1], 'r')
        const match = /^([A-Z]{1,3})(\d+)$/.exec(reference ?? '')
        if (!match || Number(match[2]) !== number) throw new Error('Workbook cell coordinates are unavailable.')
        const column = [...match[1]].reduce((total, letter) => total * 26 + letter.charCodeAt(0) - 64, 0)
        if (column > 16_384) throw new Error('Workbook column exceeds supported bounds.')
        firstColumn = Math.min(firstColumn, column)
        lastColumn = Math.max(lastColumn, column)
        values[column - 1] = xlsxCellValue(cell[1], cell[2], shared)
      }
      if (!values.length) continue
      const text = values.join('\t')
      const start = content.length
      content += `${text}\n`
      const previous = sections[sections.length - 1]
      if (sections.length >= 1_000 && previous?.location.kind === 'table' && previous.location.sheetPath === sheet.file && previous.text.length + text.length + 1 <= 1_800 && previous.location.rowEnd < number) {
        previous.text += `\n${text}`
        previous.location.rowEnd = number
        previous.location.columnStart = Math.min(previous.location.columnStart, firstColumn)
        previous.location.columnEnd = Math.max(previous.location.columnEnd, lastColumn)
        previous.location.charEnd = start + text.length
        previous.location.range = `${columnName(previous.location.columnStart)}${previous.location.rowStart}:${columnName(previous.location.columnEnd)}${number}`
        previous.id = `table-${previous.location.charStart}-${previous.location.charEnd}`
      } else sections.push(...splitSection(text, { kind: 'table', sheet: sheet.name || undefined, sheetPath: sheet.file, rowStart: number, rowEnd: number, columnStart: firstColumn, columnEnd: lastColumn, range: `${columnName(firstColumn)}${number}:${columnName(lastColumn)}${number}`, charStart: start, charEnd: start + text.length }))
      if (sections.length > 10_000) throw new Error('Workbook has too many indexed passages; split the file and retry.')
      if (Buffer.byteLength(content, 'utf8') > maxBytes) throw new Error('Workbook extracted content exceeds its configured byte limit.')
    }
  }
  return { content, sections }
}
function columnName(value: number): string {
  let out = ''
  while (value > 0) { value--; out = String.fromCharCode(65 + value % 26) + out; value = Math.floor(value / 26) }
  return out
}
