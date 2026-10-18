import { createHash } from 'node:crypto'
import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { inflateRawSync, inflateSync } from 'node:zlib'
import type { DocumentReadResult, TableItem, TableReadResult } from './types'

const MAX_ARCHIVE_ENTRIES = 256
const MAX_EXPANDED_DOCUMENT_BYTES = 8_000_000
const TEXT_EXTENSIONS = new Set([
  '.md',
  '.mdx',
  '.txt',
  '.rst',
  '.adoc',
  '.csv',
  '.tsv',
  '.json',
  '.jsonl',
  '.yaml',
  '.yml',
  '.toml',
  '.xml',
  '.html',
  '.css',
  '.js',
  '.jsx',
  '.ts',
  '.tsx',
  '.py',
  '.rb',
  '.go',
  '.rs',
  '.java',
  '.c',
  '.cpp',
  '.h',
  '.hpp',
])
const PDF_EXTENSIONS = new Set(['.pdf'])
const TABLE_EXTENSIONS = new Set(['.csv', '.tsv', '.xlsx'])
const BLOCKED_NAMES = new Set(['.env', '.npmrc', '.netrc', 'id_rsa', 'id_ed25519'])
const BLOCKED_EXTENSIONS = new Set(['.pem', '.key', '.p12', '.pfx', '.crt', '.cer', '.sqlite', '.db', '.png', '.jpg', '.jpeg', '.gif', '.webp', '.xls', '.docx', '.pptx'])

export async function readSupportedDocument(file: string, maxBytes: number, suppliedBytes?: Buffer): Promise<string | null> {
  const ext = normalizeExt(path.extname(file))
  const bytes = suppliedBytes ?? await readFileBytesBounded(file, maxBytes)
  if (PDF_EXTENSIONS.has(ext)) return extractPdfText(bytes, Math.min(maxBytes, MAX_EXPANDED_DOCUMENT_BYTES))
  if (ext === '.xlsx') return extractXlsxText(bytes, Math.min(maxBytes, MAX_EXPANDED_DOCUMENT_BYTES))
  try { return new TextDecoder('utf-8', { fatal: true }).decode(bytes) }
  catch { throw new Error('Text file is not valid UTF-8; convert it to UTF-8 and retry.') }
}

export async function readFileBytesBounded(file: string, maxBytes: number): Promise<Buffer> {
  const handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    const info = await handle.stat()
    if (!info.isFile() || info.size > maxBytes) throw new Error('Document exceeds its configured byte limit.')
    const chunks: Buffer[] = []
    let total = 0
    while (true) {
      const chunk = Buffer.allocUnsafe(Math.min(64 * 1024, maxBytes - total + 1))
      const { bytesRead } = await handle.read(chunk, 0, chunk.length, null)
      if (!bytesRead) break
      total += bytesRead
      if (total > maxBytes) throw new Error('Document exceeds its configured byte limit.')
      chunks.push(chunk.subarray(0, bytesRead))
    }
    return Buffer.concat(chunks, total)
  } finally {
    await handle.close()
  }
}

export function looksLikeSupportedDocument(file: string): boolean {
  const base = path.basename(file).toLowerCase()
  const ext = normalizeExt(path.extname(file))
  return !BLOCKED_NAMES.has(base) && !BLOCKED_EXTENSIONS.has(ext) && (TEXT_EXTENSIONS.has(ext) || PDF_EXTENSIONS.has(ext) || TABLE_EXTENSIONS.has(ext))
}

export function tableFromDocument(doc: DocumentReadResult): TableReadResult | null {
  if (!isTableExtension(doc.extension)) return null
  const delimiter: TableReadResult['delimiter'] = doc.extension === '.csv' ? ',' : '\t'
  const parsed = parseDelimitedTable(doc.content, delimiter)
  if (parsed.length < 2) return null
  const columnCount = Math.max(...parsed.map((row) => row.length))
  const columns = uniqueColumns(padRow(parsed[0], columnCount).map((value, index) => value.trim() || `Column ${index + 1}`))
  const rows = parsed.slice(1)
    .filter((row) => row.some((cell) => cell.trim()))
    .map((row) => padRow(row, columnCount))
  if (!rows.length || !columns.length) return null
  const numericColumns = summarizeNumericColumns(columns, rows)
  return {
    ...doc,
    title: doc.title === doc.content.split('\n')[0]?.trim() ? path.basename(doc.path) : doc.title,
    excerpt: tableExcerpt(columns, rows),
    rowCount: rows.length,
    columnCount,
    columns,
    delimiter,
    numericColumns,
    rows,
  }
}

function parseDelimitedTable(content: string, delimiter: ',' | '\t'): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  for (let i = 0; i < content.length; i++) {
    const ch = content[i]
    if (quoted) {
      if (ch === '"') {
        if (content[i + 1] === '"') {
          cell += '"'
          i += 1
        } else {
          quoted = false
        }
      } else {
        cell += ch
      }
      continue
    }
    if (ch === '"') {
      quoted = true
    } else if (ch === delimiter) {
      row.push(cell)
      cell = ''
    } else if (ch === '\n') {
      row.push(cell)
      rows.push(row)
      row = []
      cell = ''
    } else if (ch !== '\r') {
      cell += ch
    }
  }
  row.push(cell)
  if (row.some((value) => value.trim()) || rows.length === 0) rows.push(row)
  return rows
}

function uniqueColumns(columns: string[]): string[] {
  const seen = new Map<string, number>()
  return columns.map((column, index) => {
    const base = column || `Column ${index + 1}`
    const count = seen.get(base) ?? 0
    seen.set(base, count + 1)
    return count ? `${base} ${count + 1}` : base
  })
}

function padRow(row: string[], length: number): string[] {
  return Array.from({ length }, (_v, index) => row[index] ?? '')
}

function summarizeNumericColumns(columns: string[], rows: string[][]): TableItem['numericColumns'] {
  return columns.map((name, index) => {
    const values = rows
      .map((row) => parseTableNumber(row[index]))
      .filter((value): value is number => value != null)
    if (!values.length) return null
    const min = Math.min(...values)
    const max = Math.max(...values)
    const average = values.reduce((sum, value) => sum + value, 0) / values.length
    return { name, count: values.length, min, max, average }
  }).filter((item): item is TableItem['numericColumns'][number] => !!item)
}

function parseTableNumber(value: string): number | null {
  const cleaned = value.trim().replace(/[$£€,%]/g, '').replace(/,/g, '')
  if (!cleaned || !/^-?\d+(?:\.\d+)?$/.test(cleaned)) return null
  const n = Number(cleaned)
  return Number.isFinite(n) ? n : null
}

function tableExcerpt(columns: string[], rows: string[][]): string {
  const preview = [columns, ...rows.slice(0, 2)]
  return preview.map((row) => row.slice(0, 6).join(' | ')).join('; ').slice(0, 280)
}

export function formatColumnSummary(column: TableItem['numericColumns'][number]): string {
  return `${column.name}: min ${formatTableNumber(column.min)}, max ${formatTableNumber(column.max)}, avg ${formatTableNumber(column.average)}`
}

function formatTableNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, '')
}

export function isTableExtension(ext: string): boolean {
  return TABLE_EXTENSIONS.has(normalizeExt(ext))
}

function extractXlsxText(buffer: Buffer, maxExpandedBytes: number): string | null {
  const entries = readZipEntries(buffer, maxExpandedBytes)
  const sheetPath = firstXlsxSheetPath(entries)
  const sheet = sheetPath ? entries.get(sheetPath) : entries.get('xl/worksheets/sheet1.xml')
  if (!sheet) return null
  const sharedStrings = parseXlsxSharedStrings(entries.get('xl/sharedStrings.xml')?.toString('utf8') ?? '')
  const rows = parseXlsxSheetRows(sheet.toString('utf8'), sharedStrings)
  if (!rows.length) return null
  return rows.map((row) => row.join('\t')).join('\n')
}

export function readZipEntries(buffer: Buffer, maxExpandedBytes = MAX_EXPANDED_DOCUMENT_BYTES): Map<string, Buffer> {
  if (!Number.isSafeInteger(maxExpandedBytes) || maxExpandedBytes < 1) throw new Error('Archive expansion limit must be a positive integer.')
  const entries = new Map<string, Buffer>()
  let offset = 0
  let expandedBytes = 0
  let entryCount = 0
  while (offset + 30 <= buffer.length) {
    const sig = buffer.readUInt32LE(offset)
    if (sig === 0x02014b50 || sig === 0x06054b50) break
    if (sig !== 0x04034b50) {
      offset += 1
      continue
    }
    const flags = buffer.readUInt16LE(offset + 6)
    const method = buffer.readUInt16LE(offset + 8)
    let compressedSize = buffer.readUInt32LE(offset + 18)
    const fileNameLength = buffer.readUInt16LE(offset + 26)
    const extraLength = buffer.readUInt16LE(offset + 28)
    const nameStart = offset + 30
    const dataStart = nameStart + fileNameLength + extraLength
    if (dataStart > buffer.length) break
    // General-purpose bit 3: the sizes aren't in the local header (they live in a trailing
    // data descriptor). Streaming ZIP writers — which many real .xlsx files use — set this
    // and store compressedSize=0, so we'd otherwise read an empty buffer and skip the sheet.
    // Recover the compressed extent by scanning to the next data-descriptor / header signature.
    const hasDataDescriptor = (flags & 0x08) !== 0 || compressedSize === 0
    let dataEnd: number
    let nextOffset: number
    if (hasDataDescriptor) {
      const found = scanToDataExtent(buffer, dataStart)
      if (!found) break
      dataEnd = found.dataEnd
      nextOffset = found.nextOffset
      compressedSize = dataEnd - dataStart
    } else {
      dataEnd = dataStart + compressedSize
      nextOffset = dataEnd
    }
    if (dataEnd > buffer.length) break
    entryCount += 1
    if (entryCount > MAX_ARCHIVE_ENTRIES) throw new Error(`Document archive exceeds the ${MAX_ARCHIVE_ENTRIES}-entry limit.`)
    const name = buffer.subarray(nameStart, nameStart + fileNameLength).toString('utf8').replace(/^\/+/, '')
    const data = buffer.subarray(dataStart, dataEnd)
    try {
      const remaining = maxExpandedBytes - expandedBytes
      if (remaining < 1) throw new ArchiveExpansionLimitError(maxExpandedBytes)
      const value = method === 0
        ? Buffer.from(data)
        : method === 8
          ? inflateRawSync(data, { maxOutputLength: remaining })
          : null
      if (value) {
        if (value.byteLength > remaining) throw new ArchiveExpansionLimitError(maxExpandedBytes)
        expandedBytes += value.byteLength
        entries.set(name, value)
      }
    } catch (error) {
      if (error instanceof ArchiveExpansionLimitError || isZlibOutputLimit(error)) throw new ArchiveExpansionLimitError(maxExpandedBytes)
      // Ignore individual malformed entries; another sheet entry may still be readable.
    }
    offset = nextOffset
  }
  return entries
}

class ArchiveExpansionLimitError extends Error {
  constructor(limit: number) {
    super(`Document archive exceeds the ${limit}-byte expanded-content limit.`)
  }
}

function isZlibOutputLimit(error: unknown): boolean {
  const value = error as { code?: string; message?: string }
  return value?.code === 'ERR_BUFFER_TOO_LARGE' || /maxOutputLength|buffer larger than/i.test(String(value?.message ?? ''))
}

// For a data-descriptor entry (local header sizes unknown), find where the compressed
// data ends by scanning forward for the next ZIP signature: a data descriptor
// (PK\x07\x08), the next local file header (PK\x03\x04), or the central directory
// (PK\x01\x02). Returns the byte index where the data ends and where the next local
// header parse should resume (past the 16-byte data descriptor when one is present).
function scanToDataExtent(buffer: Buffer, dataStart: number): { dataEnd: number; nextOffset: number } | null {
  for (let i = dataStart; i + 4 <= buffer.length; i++) {
    const sig = buffer.readUInt32LE(i)
    if (sig === 0x08074b50) {
      // Optional-signature data descriptor: signature(4) + crc(4) + compSize(4) + uncompSize(4).
      return { dataEnd: i, nextOffset: i + 16 }
    }
    if (sig === 0x04034b50 || sig === 0x02014b50) {
      // No data-descriptor signature was written; the descriptor (crc+sizes, 12 bytes) sits
      // just before this header. Back up past it so the data extent excludes the descriptor.
      const dataEnd = Math.max(dataStart, i - 12)
      return { dataEnd, nextOffset: i }
    }
  }
  return null
}

function firstXlsxSheetPath(entries: Map<string, Buffer>): string | null {
  const workbook = entries.get('xl/workbook.xml')?.toString('utf8')
  const rels = entries.get('xl/_rels/workbook.xml.rels')?.toString('utf8')
  if (!workbook || !rels) return null
  const sheet = /<sheet\b[^>]*\br:id="([^"]+)"/.exec(workbook)
  const rid = sheet?.[1]
  if (!rid) return null
  const relRe = /<Relationship\b([^>]*)\/?>/g
  let match: RegExpExecArray | null
  while ((match = relRe.exec(rels))) {
    if (xmlAttr(match[1], 'Id') !== rid) continue
    const target = xmlAttr(match[1], 'Target')
    if (!target) return null
    return target.startsWith('xl/') ? target : `xl/${target.replace(/^\/+/, '')}`
  }
  return null
}

export function parseXlsxSharedStrings(xml: string): string[] {
  const strings: string[] = []
  const siRe = /<si\b[\s\S]*?<\/si>/g
  let si: RegExpExecArray | null
  while ((si = siRe.exec(xml))) {
    const parts: string[] = []
    const textRe = /<t\b[^>]*>([\s\S]*?)<\/t>/g
    let text: RegExpExecArray | null
    while ((text = textRe.exec(si[0]))) parts.push(xmlDecode(text[1]))
    strings.push(parts.join(''))
  }
  return strings
}

function parseXlsxSheetRows(xml: string, sharedStrings: string[]): string[][] {
  const rows: string[][] = []
  const rowRe = /<row\b[^>]*>([\s\S]*?)<\/row>/g
  let rowMatch: RegExpExecArray | null
  while ((rowMatch = rowRe.exec(xml))) {
    const row: string[] = []
    const cellRe = /<c\b([^>]*)>([\s\S]*?)<\/c>/g
    let cellMatch: RegExpExecArray | null
    while ((cellMatch = cellRe.exec(rowMatch[1]))) {
      const attrs = cellMatch[1]
      const body = cellMatch[2]
      const ref = xmlAttr(attrs, 'r')
      const col = ref ? xlsxColumnIndex(ref) : row.length
      row[col] = xlsxCellValue(attrs, body, sharedStrings)
    }
    if (row.some((cell) => String(cell ?? '').trim())) rows.push(padRow(row, row.length))
  }
  return rows
}

export function xlsxCellValue(attrs: string, body: string, sharedStrings: string[]): string {
  const type = xmlAttr(attrs, 't')
  if (type === 'inlineStr') {
    return [...body.matchAll(/<t\b[^>]*>([\s\S]*?)<\/t>/g)].map((match) => xmlDecode(match[1])).join('')
  }
  const raw = /<v\b[^>]*>([\s\S]*?)<\/v>/.exec(body)?.[1] ?? ''
  const value = xmlDecode(raw)
  if (type === 's') return sharedStrings[Number(value)] ?? ''
  if (type === 'b') return value === '1' ? 'TRUE' : value === '0' ? 'FALSE' : value
  return value
}

function xlsxColumnIndex(ref: string): number {
  const letters = /^[A-Z]+/i.exec(ref)?.[0]?.toUpperCase() ?? ''
  let n = 0
  for (const ch of letters) n = n * 26 + ch.charCodeAt(0) - 64
  return Math.max(0, n - 1)
}

export function xmlAttr(attrs: string, name: string): string | null {
  const match = new RegExp(`\\b${name}="([^"]*)"`).exec(attrs)
  return match ? xmlDecode(match[1]) : null
}

function xmlDecode(value: string): string {
  return value
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, '&')
}

// A PDF is actually encrypted only when its trailer references an /Encrypt dictionary
// via an indirect reference, i.e. `/Encrypt N G R`. The bare literal "/Encrypt" can also
// appear inside content streams, object data, or a PDF that merely documents PDF
// internals, so matching the literal anywhere wrongly rejects readable, non-encrypted
// files. Requiring the indirect-reference form keeps the gate tight.
export function isEncryptedPdf(binary: string): boolean {
  return /\/Encrypt\s+\d+\s+\d+\s+R\b/.test(binary)
}

function extractPdfText(buffer: Buffer, maxExpandedBytes: number): string | null {
  const binary = buffer.toString('latin1')
  if (isEncryptedPdf(binary)) throw new Error('Encrypted PDF extraction is not supported.')

  const streamTexts = extractPdfStreams(buffer, binary, maxExpandedBytes)
    .map(extractPdfContentText)
    .filter(Boolean)

  const text = normalizeExtractedPdfText(streamTexts.join('\n'))
  return text || null
}

function extractPdfStreams(buffer: Buffer, binary: string, maxExpandedBytes: number): string[] {
  const streams: string[] = []
  let expandedBytes = 0
  const streamRe = /<<(?:[\s\S]{0,4000}?)>>\s*stream\r?\n/g
  let match: RegExpExecArray | null
  while ((match = streamRe.exec(binary))) {
    const dict = match[0]
    const start = streamRe.lastIndex
    const end = binary.indexOf('endstream', start)
    if (end < 0) break
    let bytes = buffer.subarray(start, end)
    while (bytes.length && (bytes[bytes.length - 1] === 0x0a || bytes[bytes.length - 1] === 0x0d)) bytes = bytes.subarray(0, bytes.length - 1)
    if (/\/Filter\s*(?:\[?\s*)\/FlateDecode\b/.test(dict)) {
      try {
        const remaining = maxExpandedBytes - expandedBytes
        if (remaining < 1) throw new ArchiveExpansionLimitError(maxExpandedBytes)
        const inflated = inflateSync(bytes, { maxOutputLength: remaining })
        expandedBytes += inflated.byteLength
        streams.push(inflated.toString('latin1'))
      } catch (error) {
        if (error instanceof ArchiveExpansionLimitError || isZlibOutputLimit(error)) throw new ArchiveExpansionLimitError(maxExpandedBytes)
        // Ignore malformed or unsupported streams; other streams may still contain useful text.
      }
    } else if (!/\/Filter\b/.test(dict)) {
      if (expandedBytes + bytes.byteLength > maxExpandedBytes) throw new ArchiveExpansionLimitError(maxExpandedBytes)
      expandedBytes += bytes.byteLength
      streams.push(bytes.toString('latin1'))
    }
    streamRe.lastIndex = end + 'endstream'.length
  }
  return streams
}

function extractPdfContentText(content: string): string {
  const blocks = content.match(/\bBT\b[\s\S]*?\bET\b/g) ?? [content]
  const parts: string[] = []
  for (const block of blocks) {
    let arrayMatch: RegExpExecArray | null
    const arrayRe = /\[((?:\s|.|\r|\n)*?)\]\s*TJ\b/g
    while ((arrayMatch = arrayRe.exec(block))) {
      const value = extractPdfStrings(arrayMatch[1]).join('')
      if (value) parts.push(value)
    }

    let stringMatch: RegExpExecArray | null
    const stringRe = /(\((?:\\.|[^\\)])*\)|<[\da-fA-F\s]+>)\s*(?:Tj|'|")\b/g
    while ((stringMatch = stringRe.exec(block))) {
      const value = decodePdfStringToken(stringMatch[1])
      if (value) parts.push(value)
    }
  }
  return parts.join('\n')
}

function extractPdfStrings(value: string): string[] {
  const strings: string[] = []
  let match: RegExpExecArray | null
  const tokenRe = /\((?:\\.|[^\\)])*\)|<[\da-fA-F\s]+>/g
  while ((match = tokenRe.exec(value))) {
    const decoded = decodePdfStringToken(match[0])
    if (decoded) strings.push(decoded)
  }
  return strings
}

function decodePdfStringToken(token: string): string {
  if (token.startsWith('<')) return decodePdfHexString(token.slice(1, -1))
  return decodePdfLiteralString(token.slice(1, -1))
}

function decodePdfLiteralString(value: string): string {
  let out = ''
  for (let i = 0; i < value.length; i++) {
    const ch = value[i]
    if (ch !== '\\') {
      out += ch
      continue
    }
    const next = value[++i]
    if (next == null) break
    if (next === 'n') out += '\n'
    else if (next === 'r') out += '\r'
    else if (next === 't') out += '\t'
    else if (next === 'b') out += '\b'
    else if (next === 'f') out += '\f'
    else if (next === '(' || next === ')' || next === '\\') out += next
    else if (next === '\r' && value[i + 1] === '\n') i += 1
    else if (next === '\n' || next === '\r') {
      // Escaped physical line break; no character is emitted.
    } else if (/[0-7]/.test(next)) {
      let octal = next
      for (let j = 0; j < 2 && /[0-7]/.test(value[i + 1] ?? ''); j++) octal += value[++i]
      out += String.fromCharCode(parseInt(octal, 8))
    } else {
      out += next
    }
  }
  return decodePdfEncodedString(out)
}

function decodePdfHexString(value: string): string {
  const hex = value.replace(/\s+/g, '')
  const bytes: number[] = []
  for (let i = 0; i < hex.length; i += 2) {
    bytes.push(parseInt(hex.slice(i, i + 2).padEnd(2, '0'), 16))
  }
  return decodePdfEncodedBytes(bytes)
}

function decodePdfEncodedString(value: string): string {
  return decodePdfEncodedBytes(Array.from(value, (ch) => ch.charCodeAt(0) & 0xff))
}

function decodePdfEncodedBytes(bytes: number[]): string {
  if (bytes.length >= 2 && bytes[0] === 0xfe && bytes[1] === 0xff) {
    let out = ''
    for (let i = 2; i + 1 < bytes.length; i += 2) out += String.fromCharCode((bytes[i] << 8) | bytes[i + 1])
    return out
  }
  if (bytes.length >= 2 && bytes[0] === 0xff && bytes[1] === 0xfe) {
    let out = ''
    for (let i = 2; i + 1 < bytes.length; i += 2) out += String.fromCharCode(bytes[i] | (bytes[i + 1] << 8))
    return out
  }
  return Buffer.from(bytes).toString('latin1')
}

function normalizeExtractedPdfText(value: string): string {
  return value
    .replace(/\u0000/g, '')
    .replace(/[ \t\f\v]+/g, ' ')
    .replace(/[ \t]*\n[ \t]*/g, '\n')
    .replace(/\n{3,}/g, '\n\n')
    .trim()
}

export function titleFromContent(content: string): string {
  const line = content.split('\n').find((item) => item.trim())
  if (!line) return ''
  return line.replace(/^#+\s*/, '').trim().slice(0, 120)
}

export function excerpt(content: string): string {
  return content.replace(/\s+/g, ' ').trim().slice(0, 280)
}

export function stableDocumentId(sourceId: string, relativePath: string): string {
  return createHash('sha256').update(`${sourceId}|${relativePath}`).digest('hex').slice(0, 16)
}

export function normalizeDocumentId(id: string): string {
  return String(id || '').trim().split(':').pop() || ''
}

export function normalizeExt(ext: string): string {
  return ext.startsWith('.') ? ext.toLowerCase() : `.${ext.toLowerCase()}`
}

export function isInside(root: string, file: string): boolean {
  const rel = path.relative(root, file)
  return rel === '' || (!!rel && !rel.startsWith('..') && !path.isAbsolute(rel))
}
