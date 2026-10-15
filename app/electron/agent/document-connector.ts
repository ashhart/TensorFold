import { createHash, randomUUID } from 'node:crypto'
import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { inflateRawSync, inflateSync } from 'node:zlib'
import type {
  ConnectorEvent,
  ConnectorSourceRef,
  DocumentConnectorSettings,
  DocumentItem,
  DocumentListOptions,
  DocumentReadResult,
  DocumentSourceConfig,
  DocumentSummaryResult,
  DocumentSyncResult,
  TableItem,
  TableListOptions,
  TableReadResult,
  TableSummaryResult,
} from './types'
import { ensureSummaryCitations } from './summary-citations'

const MAX_SCAN_FILES = 1_500
const MAX_SCAN_DEPTH = 8
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

export function makeDocumentSource(target: string, now = Date.now()): DocumentSourceConfig {
  return {
    id: `doc_${randomUUID().slice(0, 8)}`,
    label: path.basename(target) || 'Documents',
    path: target,
    kind: looksLikeSupportedDocument(target) ? 'file' : 'dir',
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
  }
}

export async function scanDocumentItems(settings: DocumentConnectorSettings, opts: DocumentListOptions = {}): Promise<{ documents: DocumentItem[]; sources: DocumentSourceConfig[] }> {
  const now = Date.now()
  const sources = settings.sources.map((source) => ({ ...source }))
  const maxDocuments = Math.max(25, Math.min(settings.maxDocuments || 500, 5_000))
  const maxBytes = Math.max(8_192, Math.min(settings.maxDocumentBytes || 1_000_000, 8_000_000))
  const documents: DocumentItem[] = []

  for (const source of sources) {
    if (!source.enabled) continue
    try {
      const files = await collectDocumentFiles(source.path, source.kind)
      for (const file of files.slice(0, MAX_SCAN_FILES)) {
        try {
          const item = await readDocumentFile(source, file, maxBytes)
          if (item) documents.push(publicDocument(item))
        } catch {
          // Ignore malformed/oversize individual files; source errors are for unreadable roots.
        }
      }
      source.lastSyncAt = now
      source.lastError = null
      source.updatedAt = now
    } catch (e: any) {
      source.lastError = String(e?.message ?? e)
      source.updatedAt = now
    }
  }

  let filtered = documents.sort((a, b) => (b.modifiedAt ?? 0) - (a.modifiedAt ?? 0))
  if (opts.path?.trim()) {
    const needle = opts.path.trim().toLowerCase()
    filtered = filtered.filter((item) => item.relativePath.toLowerCase().includes(needle) || item.path.toLowerCase().includes(needle))
  }
  if (opts.extension?.trim()) {
    const ext = normalizeExt(opts.extension)
    filtered = filtered.filter((item) => item.extension === ext)
  }
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    filtered = filtered.filter((item) => [item.title, item.relativePath, item.excerpt].some((value) => value.toLowerCase().includes(q)))
  }
  const limit = Math.max(1, Math.min(Number(opts.limit) || maxDocuments, maxDocuments))
  return { documents: filtered.slice(0, limit), sources }
}

export async function readDocumentItem(settings: DocumentConnectorSettings, id: string): Promise<DocumentReadResult> {
  const target = normalizeDocumentId(id)
  const maxBytes = Math.max(8_192, Math.min(settings.maxDocumentBytes || 1_000_000, 8_000_000))
  for (const source of settings.sources.filter((s) => s.enabled)) {
    const files = await collectDocumentFiles(source.path, source.kind)
    for (const file of files.slice(0, MAX_SCAN_FILES)) {
      let item: DocumentReadResult | null = null
      try {
        item = await readDocumentFile(source, file, maxBytes)
      } catch {
        // Skip malformed/oversize/encrypted individual files so an unrelated bad
        // file (e.g. an encrypted PDF sorting before the target) does not abort the read.
        continue
      }
      if (item && (item.id === target || item.relativePath === id || item.path === id)) return item
    }
  }
  throw new Error('Document not found. Sync the connector and try again.')
}

export async function summarizeDocuments(settings: DocumentConnectorSettings, opts: DocumentListOptions = {}): Promise<DocumentSummaryResult> {
  const { documents } = await scanDocumentItems(settings, { ...opts, limit: opts.limit ?? 12 })
  if (!documents.length) return ensureSummaryCitations({ summary: 'No documents found for the current filters.\n\nSources: none.', sources: [] })
  const sources = documents.map(documentSourceRef)
  const lines = documents.map((doc) => `- [${documentSourceId(doc)}] ${documentSummaryLabel(doc)}: ${doc.excerpt || '(no excerpt)'}`)
  return ensureSummaryCitations({
    summary: `${documents.length} document${documents.length === 1 ? '' : 's'} matched.\n\n${lines.join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  })
}

export async function scanTableItems(settings: DocumentConnectorSettings, opts: TableListOptions = {}): Promise<{ tables: TableItem[]; sources: DocumentSourceConfig[] }> {
  const { documents, sources } = await scanDocumentItems(settings, { ...opts, query: undefined, extension: undefined, limit: settings.maxDocuments })
  let tables: TableItem[] = []
  for (const doc of documents) {
    if (!isTableExtension(doc.extension)) continue
    if (opts.extension?.trim() && doc.extension !== normalizeExt(opts.extension)) continue
    try {
      const read = await readDocumentItem(settings, doc.id)
      const table = tableFromDocument(read)
      if (table) tables.push(publicTable(table))
    } catch {
      // Ignore malformed table files; source-level failures are reported by document scan.
    }
  }
  if (opts.path?.trim()) {
    const needle = opts.path.trim().toLowerCase()
    tables = tables.filter((table) => table.relativePath.toLowerCase().includes(needle) || table.path.toLowerCase().includes(needle))
  }
  if (opts.query?.trim()) {
    const q = opts.query.trim().toLowerCase()
    tables = tables.filter((table) => [table.title, table.relativePath, table.excerpt, table.columns.join(' ')].some((value) => value.toLowerCase().includes(q)))
  }
  const maxTables = Math.max(1, Math.min(settings.maxDocuments || 500, 5_000))
  const limit = Math.max(1, Math.min(Number(opts.limit) || maxTables, maxTables))
  return { tables: tables.slice(0, limit), sources }
}

export async function readTableItem(settings: DocumentConnectorSettings, id: string): Promise<TableReadResult> {
  const doc = await readDocumentItem(settings, id)
  const table = tableFromDocument(doc)
  if (!table) throw new Error('Table not found. Use a local CSV or TSV document source.')
  return table
}

export async function summarizeTables(settings: DocumentConnectorSettings, opts: TableListOptions = {}): Promise<TableSummaryResult> {
  const { tables } = await scanTableItems(settings, { ...opts, limit: opts.limit ?? 12 })
  if (!tables.length) return ensureSummaryCitations({ summary: 'No tables found for the current filters.\n\nSources: none.', sources: [] })
  const sources = tables.map(documentSourceRef)
  const lines = tables.map((table) => {
    const stats = table.numericColumns.length
      ? ` Numeric columns: ${table.numericColumns.slice(0, 4).map(formatColumnSummary).join('; ')}.`
      : ' Numeric columns: none detected.'
    return `- [${documentSourceId(table)}] ${documentSummaryLabel(table)}: ${table.rowCount} rows x ${table.columnCount} columns (${table.columns.join(', ')}).${stats}`
  })
  return ensureSummaryCitations({
    summary: `${tables.length} table${tables.length === 1 ? '' : 's'} matched.\n\n${lines.join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  })
}

export async function syncDocumentConnector(settings: DocumentConnectorSettings, existingEvents: ConnectorEvent[]): Promise<DocumentSyncResult & { events: ConnectorEvent[] }> {
  const syncedAt = Date.now()
  const { documents, sources } = await scanDocumentItems(settings, { limit: settings.maxDocuments })
  const known = new Set(existingEvents.map((event) => event.threadKey))
  const events: ConnectorEvent[] = []
  for (const doc of documents.slice(0, 50)) {
    const threadKey = documentSourceId(doc)
    if (known.has(threadKey)) continue
    events.push(documentEventFromItem(doc, syncedAt))
  }
  return { documents, sources, events, syncedAt }
}

export function documentSourceId(doc: Pick<DocumentItem, 'sourceId' | 'id'>): string {
  return `docs:${doc.sourceId}:${doc.id}`
}

export function documentSourceRef(doc: DocumentItem): ConnectorSourceRef {
  return {
    id: documentSourceId(doc),
    connectorId: 'docs',
    sourceId: doc.sourceId,
    itemId: doc.id,
    title: doc.title || doc.relativePath,
    path: doc.path,
    excerpt: doc.excerpt,
    createdAt: doc.modifiedAt,
  }
}

function documentSummaryLabel(doc: DocumentItem): string {
  if (doc.title && doc.title !== doc.relativePath) return `${doc.title} (${doc.relativePath})`
  return doc.title || doc.relativePath
}

async function collectDocumentFiles(root: string, kind: DocumentSourceConfig['kind']): Promise<string[]> {
  const stat = await fs.stat(root)
  if (kind === 'file' || stat.isFile()) {
    if (!looksLikeSupportedDocument(root)) throw new Error('Document source is not a supported local document.')
    return [root]
  }
  if (!stat.isDirectory()) throw new Error('Document source is not a folder.')
  const rootReal = await fs.realpath(root)
  const files: string[] = []
  const walk = async (dir: string, depth: number) => {
    if (depth > MAX_SCAN_DEPTH || files.length >= MAX_SCAN_FILES) return
    const entries = await fs.readdir(dir, { withFileTypes: true }).catch(() => [])
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (files.length >= MAX_SCAN_FILES) break
      if (entry.name === '.git' || entry.name === 'node_modules' || entry.name === '.vontra') continue
      const full = path.join(dir, entry.name)
      if (entry.isSymbolicLink()) continue
      if (entry.isDirectory()) {
        if (entry.name.startsWith('.') && entry.name !== '.github') continue
        await walk(full, depth + 1)
      } else if (entry.isFile() && looksLikeSupportedDocument(entry.name)) {
        const real = await fs.realpath(full).catch(() => null)
        if (real && isInside(rootReal, real)) files.push(full)
      }
    }
  }
  await walk(root, 0)
  return files
}

async function readDocumentFile(source: DocumentSourceConfig, file: string, maxBytes: number): Promise<DocumentReadResult | null> {
  if (!looksLikeSupportedDocument(file)) return null
  const stat = await fs.stat(file)
  if (!stat.isFile() || stat.size > maxBytes) return null
  const content = await readSupportedDocument(file, maxBytes)
  if (!content || content.includes('\u0000')) return null
  if (Buffer.byteLength(content, 'utf8') > maxBytes) return null
  const normalized = content.replace(/\r\n/g, '\n').trim()
  const relativePath = path.relative(source.kind === 'file' ? path.dirname(source.path) : source.path, file) || path.basename(file)
  const title = titleFromContent(normalized) || path.basename(file)
  return {
    id: stableDocumentId(source.id, relativePath),
    sourceId: source.id,
    sourceLabel: source.label,
    path: file,
    relativePath,
    title,
    excerpt: excerpt(normalized),
    size: stat.size,
    modifiedAt: stat.mtimeMs,
    extension: normalizeExt(path.extname(file)),
    content: normalized,
  }
}

function publicDocument(doc: DocumentReadResult): DocumentItem {
  const { content: _content, ...rest } = doc
  return rest
}

function publicTable(table: TableReadResult): TableItem {
  const { rows: _rows, ...rest } = table
  return rest
}

function documentEventFromItem(doc: DocumentItem, now: number): ConnectorEvent {
  return {
    id: randomUUID(),
    connectorId: 'docs',
    pluginName: 'Documents',
    action: 'document_indexed',
    direction: 'inbound',
    classification: 'document',
    priority: 'normal',
    trustLevel: 'suggest',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: null,
    resolvedBy: null,
    title: doc.title || doc.relativePath,
    summary: `${doc.relativePath}: ${doc.excerpt}`,
    actor: doc.sourceLabel,
    threadKey: documentSourceId(doc),
    payloadKeys: ['path', 'title', 'excerpt', 'modified_at', 'messageId'],
    // doc.id is a stable hash of sourceId+relativePath (stableDocumentId), so this rides in
    // the payload as messageId for rememberConnectorEvents' insert-time dedup, mirroring
    // email/Slack/Discord. Without it, two concurrent syncs (manual trigger overlapping an
    // auto-sync timer) can both miss each other's in-flight events and double-ingest.
    payload: { messageId: documentSourceId(doc) },
    sources: [documentSourceRef(doc)],
    createdAt: now,
  }
}

async function readSupportedDocument(file: string, maxBytes: number): Promise<string | null> {
  const ext = normalizeExt(path.extname(file))
  const bytes = await readFileBytesBounded(file, maxBytes)
  if (PDF_EXTENSIONS.has(ext)) return extractPdfText(bytes, Math.min(maxBytes, MAX_EXPANDED_DOCUMENT_BYTES))
  if (ext === '.xlsx') return extractXlsxText(bytes, Math.min(maxBytes, MAX_EXPANDED_DOCUMENT_BYTES))
  return bytes.toString('utf8')
}

async function readFileBytesBounded(file: string, maxBytes: number): Promise<Buffer> {
  const handle = await fs.open(file, constants.O_RDONLY | constants.O_NOFOLLOW)
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

function looksLikeSupportedDocument(file: string): boolean {
  const base = path.basename(file).toLowerCase()
  const ext = normalizeExt(path.extname(file))
  return !BLOCKED_NAMES.has(base) && !BLOCKED_EXTENSIONS.has(ext) && (TEXT_EXTENSIONS.has(ext) || PDF_EXTENSIONS.has(ext) || TABLE_EXTENSIONS.has(ext))
}

function tableFromDocument(doc: DocumentReadResult): TableReadResult | null {
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

function formatColumnSummary(column: TableItem['numericColumns'][number]): string {
  return `${column.name}: min ${formatTableNumber(column.min)}, max ${formatTableNumber(column.max)}, avg ${formatTableNumber(column.average)}`
}

function formatTableNumber(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(2).replace(/\.?0+$/, '')
}

function isTableExtension(ext: string): boolean {
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

function parseXlsxSharedStrings(xml: string): string[] {
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

function xlsxCellValue(attrs: string, body: string, sharedStrings: string[]): string {
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

function xmlAttr(attrs: string, name: string): string | null {
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

function titleFromContent(content: string): string {
  const line = content.split('\n').find((item) => item.trim())
  if (!line) return ''
  return line.replace(/^#+\s*/, '').trim().slice(0, 120)
}

function excerpt(content: string): string {
  return content.replace(/\s+/g, ' ').trim().slice(0, 280)
}

function stableDocumentId(sourceId: string, relativePath: string): string {
  return createHash('sha256').update(`${sourceId}|${relativePath}`).digest('hex').slice(0, 16)
}

function normalizeDocumentId(id: string): string {
  return String(id || '').trim().split(':').pop() || ''
}

function normalizeExt(ext: string): string {
  return ext.startsWith('.') ? ext.toLowerCase() : `.${ext.toLowerCase()}`
}

function isInside(root: string, file: string): boolean {
  const rel = path.relative(root, file)
  return rel === '' || (!!rel && !rel.startsWith('..') && !path.isAbsolute(rel))
}
