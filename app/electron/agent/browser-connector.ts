import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type {
  BrowserConnectorSettings,
  BrowserItem,
  BrowserListOptions,
  BrowserReadResult,
  BrowserSnapshotInput,
  BrowserSourceConfig,
  BrowserSummaryResult,
  BrowserSyncResult,
  ConnectorEvent,
  ConnectorSourceRef,
} from './types'
import { ensureSummaryCitations } from './summary-citations'

const MAX_EXCERPT = 360

export function makeBrowserCurrentTabSource(input: BrowserSnapshotInput, now = Date.now()): BrowserSourceConfig {
  const url = normalizeBrowserUrl(input.url)
  return {
    id: `browser_${randomUUID().slice(0, 8)}`,
    label: cleanText(input.title, url.hostname, 90) || url.hostname,
    kind: 'current_tab',
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
    url: url.href,
    title: cleanText(input.title, url.hostname, 160) || url.hostname,
    excerpt: cleanText(input.excerpt, '', MAX_EXCERPT),
    capturedAt: normalizeTime(input.capturedAt, now),
  }
}

export function makeBrowserHistorySource(target: string, now = Date.now()): BrowserSourceConfig {
  return {
    id: `browser_${randomUUID().slice(0, 8)}`,
    label: path.basename(target) || 'Browser history export',
    kind: 'history_export',
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
    path: target,
  }
}

export async function scanBrowserItems(settings: BrowserConnectorSettings, opts: BrowserListOptions = {}): Promise<{ items: BrowserItem[]; sources: BrowserSourceConfig[] }> {
  const now = Date.now()
  const sources = settings.sources.map((source) => ({ ...source }))
  const maxItems = Math.max(1, Math.min(settings.maxItems || 300, 5_000))
  const items: BrowserItem[] = []

  for (const source of sources) {
    if (!source.enabled) continue
    try {
      if (source.kind === 'current_tab') {
        const item = itemFromCurrentTab(source)
        if (item) items.push(item)
      } else if (source.kind === 'history_export') {
        items.push(...await itemsFromHistoryExport(source, maxItems))
      }
      source.lastSyncAt = now
      source.lastError = null
      source.updatedAt = now
    } catch (e: any) {
      source.lastError = String(e?.message ?? e)
      source.updatedAt = now
    }
  }

  let filtered = items.sort((a, b) => (b.capturedAt ?? 0) - (a.capturedAt ?? 0))
  if (opts.query?.trim()) {
    const query = opts.query.trim().toLowerCase()
    filtered = filtered.filter((item) => [item.title, item.url, item.excerpt, item.sourceLabel].some((value) => value.toLowerCase().includes(query)))
  }
  const limit = Math.max(1, Math.min(Number(opts.limit) || maxItems, maxItems))
  return { items: filtered.slice(0, limit), sources }
}

export async function readBrowserItem(settings: BrowserConnectorSettings, id: string): Promise<BrowserReadResult> {
  const target = normalizeBrowserId(id)
  const { items } = await scanBrowserItems(settings, { limit: settings.maxItems || 300 })
  const item = items.find((candidate) => candidate.id === target || candidate.url === id || browserSourceId(candidate) === id)
  if (!item) throw new Error('Browser source item not found. Sync the connector and try again.')
  const source = browserSourceRef(item)
  return {
    ...item,
    content: [
      `ID: ${item.id}`,
      `Source ID: ${source.id}`,
      `Source: ${item.sourceLabel}`,
      `Kind: ${item.kind === 'current_tab' ? 'Current tab' : 'History export'}`,
      `URL: ${item.url}`,
      `Captured: ${item.capturedAt ? new Date(item.capturedAt).toLocaleString() : 'unknown'}`,
      item.path ? `Export: ${item.path}` : '',
      '',
      item.title,
      item.excerpt,
      '',
      'Sources:',
      `- ${source.id} | ${source.title}`,
    ].filter((line) => line !== '').join('\n'),
  }
}

export async function summarizeBrowser(settings: BrowserConnectorSettings, opts: BrowserListOptions = {}): Promise<BrowserSummaryResult> {
  const { items } = await scanBrowserItems(settings, { ...opts, limit: opts.limit ?? 12 })
  if (!items.length) return ensureSummaryCitations({ summary: 'No browser sources found for the current filters.\n\nSources: none.', sources: [] })
  const sources = items.map(browserSourceRef)
  const lines = items.map((item) => `- [${browserSourceId(item)}] ${item.title || item.url}: ${item.excerpt || item.url}`)
  return ensureSummaryCitations({
    summary: `${items.length} browser source${items.length === 1 ? '' : 's'} matched.\n\n${lines.join('\n')}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
    sources,
  })
}

export async function syncBrowserConnector(settings: BrowserConnectorSettings, existingEvents: ConnectorEvent[]): Promise<BrowserSyncResult> {
  const syncedAt = Date.now()
  const { items, sources } = await scanBrowserItems(settings, { limit: settings.maxItems || 300 })
  const known = new Set(existingEvents.map((event) => event.threadKey))
  const events: ConnectorEvent[] = []
  for (const item of items.slice(0, 50)) {
    const threadKey = browserSourceId(item)
    if (known.has(threadKey)) continue
    events.push(browserEventFromItem(item, syncedAt))
  }
  return { items, sources, events, syncedAt }
}

export function browserSourceId(item: Pick<BrowserItem, 'sourceId' | 'id'>): string {
  return `browser:${item.sourceId}:${item.id}`
}

export function browserSourceRef(item: BrowserItem): ConnectorSourceRef {
  return {
    id: browserSourceId(item),
    connectorId: 'browser',
    sourceId: item.sourceId,
    itemId: item.id,
    title: item.title || item.url,
    path: item.url,
    excerpt: item.excerpt,
    createdAt: item.capturedAt,
  }
}

function itemFromCurrentTab(source: BrowserSourceConfig): BrowserItem | null {
  if (!source.url) return null
  const url = normalizeBrowserUrl(source.url)
  const title = cleanText(source.title, url.hostname, 160) || url.hostname
  const excerpt = cleanText(source.excerpt, '', MAX_EXCERPT)
  const capturedAt = normalizeTime(source.capturedAt, source.updatedAt || source.createdAt || null)
  return {
    id: stableBrowserItemId(source.id, url.href, title, capturedAt),
    sourceId: source.id,
    sourceLabel: source.label,
    kind: 'current_tab',
    url: url.href,
    title,
    excerpt,
    capturedAt,
  }
}

async function itemsFromHistoryExport(source: BrowserSourceConfig, maxItems: number): Promise<BrowserItem[]> {
  if (!source.path) throw new Error('Browser history source has no export path.')
  const raw = await fs.readFile(source.path, 'utf8')
  const parsed = JSON.parse(raw)
  const records = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.items)
      ? parsed.items
      : Array.isArray(parsed?.history)
        ? parsed.history
        : Array.isArray(parsed?.visits)
          ? parsed.visits
          : []
  if (!Array.isArray(records)) throw new Error('Browser history export must be a JSON array or object with items/history/visits.')
  const items: BrowserItem[] = []
  // Walk the full export (not a pre-sliced prefix): invalid entries (no url/href) must not
  // consume the maxItems*N budget before validity is checked, or a large run of invalid
  // records near the start can starve out valid ones further in. The loop still bails as
  // soon as maxItems valid items are collected, so this stays bounded.
  for (const record of records) {
    if (!record || typeof record !== 'object') continue
    const rawUrl = String(record.url ?? record.href ?? '').trim()
    if (!rawUrl) continue
    let url: URL
    try {
      url = normalizeBrowserUrl(rawUrl)
    } catch {
      continue
    }
    const title = cleanText(record.title ?? record.name, url.hostname, 160) || url.hostname
    const excerpt = cleanText(record.excerpt ?? record.text ?? record.snippet ?? record.content, '', MAX_EXCERPT)
    const capturedAt = normalizeTime(record.visitedAt ?? record.lastVisitTime ?? record.last_visit_time ?? record.time ?? record.date, null)
    items.push({
      id: stableBrowserItemId(source.id, url.href, title, capturedAt),
      sourceId: source.id,
      sourceLabel: source.label,
      kind: 'history_export',
      url: url.href,
      title,
      excerpt,
      capturedAt,
      path: source.path,
    })
    if (items.length >= maxItems) break
  }
  return items
}

function browserEventFromItem(item: BrowserItem, syncedAt: number): ConnectorEvent {
  return {
    id: randomUUID(),
    connectorId: 'browser',
    pluginName: 'Browser',
    action: 'browser_source_seen',
    direction: 'inbound',
    classification: 'document',
    priority: 'low',
    trustLevel: 'notify',
    requiresApproval: false,
    approvalId: null,
    approvalStatus: null,
    resolvedAt: null,
    resolvedBy: null,
    title: item.title || item.url,
    summary: item.excerpt || item.url,
    actor: item.sourceLabel,
    threadKey: browserSourceId(item),
    payloadKeys: ['source', 'url', 'title', 'messageId'],
    // item.id is a stable hash of sourceId+url+title+capturedAt (stableBrowserItemId), so this
    // rides in the payload as messageId for rememberConnectorEvents' insert-time dedup,
    // mirroring email/Slack/Discord. Without it, two concurrent syncs (manual trigger
    // overlapping an auto-sync timer) can both miss each other's in-flight events and
    // double-ingest.
    payload: { messageId: browserSourceId(item) },
    sources: [browserSourceRef(item)],
    createdAt: item.capturedAt ?? syncedAt,
  }
}

function normalizeBrowserUrl(raw: string): URL {
  const value = String(raw || '').trim()
  if (!value) throw new Error('Browser source URL is required.')
  if (value.length > 2_000) throw new Error('Browser source URL is too long.')
  const url = new URL(value)
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Browser source URLs must use http or https.')
  url.username = ''
  url.password = ''
  return url
}

function normalizeBrowserId(id: string): string {
  return String(id || '').replace(/^browser:[^:]+:/, '')
}

function stableBrowserItemId(sourceId: string, url: string, title: string, capturedAt: number | null): string {
  return createHash('sha1').update(`${sourceId}\0${url}\0${title}\0${capturedAt ?? ''}`).digest('hex').slice(0, 16)
}

// Milliseconds between the WebKit/Chrome epoch (1601-01-01) and the Unix epoch (1970-01-01).
const WEBKIT_EPOCH_OFFSET_MS = 11_644_473_600_000

export function normalizeTime(value: unknown, fallback: number | null): number | null {
  if (value == null || value === '') return fallback
  if (typeof value === 'number' && Number.isFinite(value)) {
    // Chrome/WebKit history (last_visit_time) stores MICROSECONDS since 1601-01-01, i.e.
    // values around 1.3e16 today. Detect that epoch (anything past 1970 in WebKit-µs is
    // >= the offset expressed in µs) and convert to Unix ms; otherwise treat large values
    // as Unix microseconds (÷1000) and the rest as Unix milliseconds.
    if (value >= WEBKIT_EPOCH_OFFSET_MS * 1000) return Math.floor(value / 1000 - WEBKIT_EPOCH_OFFSET_MS)
    return value > 10_000_000_000_000 ? Math.floor(value / 1000) : Math.floor(value)
  }
  if (typeof value === 'string') {
    const numeric = Number(value)
    if (Number.isFinite(numeric) && value.trim()) return normalizeTime(numeric, fallback)
    const parsed = Date.parse(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

function cleanText(value: unknown, fallback = '', max = 160): string {
  return String(value ?? fallback).replace(/\s+/g, ' ').trim().slice(0, max)
}
