import { randomUUID } from 'node:crypto'
import path from 'node:path'
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

import { formatColumnSummary, isTableExtension, looksLikeSupportedDocument, normalizeDocumentId, normalizeExt, tableFromDocument } from './document-parser'
import { collectDocumentFiles, readDocumentFile, scanDocumentSnapshot } from '../document-library/scanner'
export { readZipEntries, isEncryptedPdf } from './document-parser'

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
  const { documents: complete, sources } = await scanDocumentSnapshot(settings)
  const documents = complete.map(publicDocument)
  const maxDocuments = Math.max(1, Math.min(settings.maxDocuments || 500, 5_000))

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
    const q = opts.query.trim().normalize('NFKC').toLowerCase().slice(0, 256)
    const fullText = new Map(complete.map(item => [item.id, item.content]))
    filtered = filtered.filter(item => [item.title, item.relativePath, fullText.get(item.id) ?? ''].some(value => value.normalize('NFKC').toLowerCase().includes(q)))
      .map(item => {
        const matchingSection = complete.find(doc => doc.id === item.id)?.sections?.find(section => section.text.normalize('NFKC').toLowerCase().includes(q))
        if (!matchingSection) return item
        const index = matchingSection.text.normalize('NFKC').toLowerCase().indexOf(q)
        return { ...item, excerpt: matchingSection.text.slice(Math.max(0, index - 80), index + 200).replace(/\s+/g, ' ') }
      })
  }
  const limit = Math.max(1, Math.min(Number(opts.limit) || maxDocuments, maxDocuments))
  return { documents: filtered.slice(0, limit), sources }
}

export async function readDocumentItem(settings: DocumentConnectorSettings, id: string): Promise<DocumentReadResult> {
  const target = normalizeDocumentId(id)
  const maxBytes = Math.max(8_192, Math.min(settings.maxDocumentBytes || 1_000_000, 8_000_000))
  for (const source of settings.sources.filter((s) => s.enabled)) {
    const files = await collectDocumentFiles(source.path, source.kind).catch(() => [])
    for (const file of files.slice(0, 1_500)) {
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
      // Per-file failures are retained by the document scan; other tables remain usable.
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

export async function syncDocumentConnector(settings: DocumentConnectorSettings, existingEvents: ConnectorEvent[], indexed?: { documents: DocumentItem[]; sources: DocumentSourceConfig[] }): Promise<DocumentSyncResult & { events: ConnectorEvent[] }> {
  const syncedAt = Date.now()
  const { documents, sources } = indexed ?? await scanDocumentItems(settings, { limit: settings.maxDocuments })
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

function publicDocument(doc: DocumentReadResult): DocumentItem {
  const { content: _content, sections: _sections, revision: _revision, locationNote: _locationNote, ...rest } = doc
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
