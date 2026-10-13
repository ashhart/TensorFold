import { createHash } from 'node:crypto'
import { existsSync, realpathSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { browserSourceRef, scanBrowserItems } from './browser-connector.ts'
import { calendarSourceRef, scanCalendarEvents } from './calendar-connector.ts'
import { documentSourceRef, scanDocumentItems, scanTableItems } from './document-connector.ts'
import { emailSourceRef, scanEmailMessages } from './email-connector.ts'
import { scanSlackMessages, slackSourceRef } from './slack-connector.ts'
import { contentText } from './content.ts'
import type {
  BrowserConnectorSettings,
  CalendarConnectorSettings,
  ConnectorEvent,
  ConnectorSourceRef,
  DocumentConnectorSettings,
  EmailConnectorSettings,
  Mission,
  ProjectMemoryEntry,
  Room,
  SlackConnectorSettings,
  Thread,
  Ticket,
  TimelineItem,
} from './types'

export const RETRIEVAL_INDEX_SCHEMA = 'vontra.retrieval-index/1'
export const RETRIEVAL_KINDS = ['code', 'doc', 'chat', 'email', 'summary', 'artifact'] as const

export type RetrievalKind = typeof RETRIEVAL_KINDS[number]

export interface RetrievalRecord {
  id: string
  kind: RetrievalKind
  title: string
  text: string
  sourceIds: string[]
  path?: string | null
  createdAt?: number | null
  metadata?: Record<string, string | number | boolean | null>
}

export interface RetrievalIndexStats {
  total: number
  byKind: Record<RetrievalKind, number>
  sourceIds: number
}

export interface RetrievalIndex {
  schema: typeof RETRIEVAL_INDEX_SCHEMA
  generatedAt: number
  records: RetrievalRecord[]
  stats: RetrievalIndexStats
}

export interface RetrievalSearchOptions {
  query?: string | null
  kinds?: RetrievalKind[] | string | null
  limit?: number | null
}

export interface RetrievalSearchHit {
  record: RetrievalRecord
  score: number
  matchedTerms: string[]
}

export interface RetrievalSearchResult {
  query: string
  kinds: RetrievalKind[]
  generatedAt: number
  total: number
  hits: RetrievalSearchHit[]
  stats: RetrievalIndexStats
}

export interface RetrievalCodeFileInput {
  path: string
  content: string
  modifiedAt?: number | null
}

export interface RetrievalArtifactInput {
  id: string
  title?: string | null
  text?: string | null
  sourceIds?: string[]
  path?: string | null
  createdAt?: number | null
}

export interface RetrievalIndexInput {
  records?: Partial<RetrievalRecord>[]
  codeFiles?: RetrievalCodeFileInput[]
  documents?: ConnectorSourceRef[]
  connectorEvents?: ConnectorEvent[]
  threads?: Thread[]
  missions?: Mission[]
  rooms?: Room[]
  tickets?: Ticket[]
  projectMemory?: ProjectMemoryEntry[]
  artifacts?: RetrievalArtifactInput[]
}

export interface WorkspaceRetrievalIndexInput extends RetrievalIndexInput {
  cwd: string
  path?: string | null
  query?: string | null
  kinds?: RetrievalKind[] | string | null
  maxCodeFiles?: number | null
  maxConnectorItems?: number | null
  emailSettings?: EmailConnectorSettings
  slackSettings?: SlackConnectorSettings
  calendarSettings?: CalendarConnectorSettings
  documentSettings?: DocumentConnectorSettings
  browserSettings?: BrowserConnectorSettings
}

const CODE_EXTENSIONS = new Set([
  '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs',
  '.py', '.rb', '.go', '.rs', '.java', '.kt', '.swift',
  '.c', '.cc', '.cpp', '.h', '.hpp', '.cs', '.php',
  '.sh', '.zsh', '.bash', '.sql', '.graphql',
])
const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'dist-electron', '.next', 'build', '.cache', 'vendor', '.venv'])
const BLOCKED_NAMES = new Set(['.env', '.npmrc', '.netrc', 'id_rsa', 'id_ed25519'])
const BLOCKED_EXTENSIONS = new Set(['.pem', '.key', '.p12', '.pfx', '.crt', '.cer', '.sqlite', '.db'])
const MAX_RECORD_TEXT = 4_000
const MAX_CODE_BYTES = 240_000
const DEFAULT_LIMIT = 12
const DEFAULT_MAX_CODE_FILES = 300
const DEFAULT_MAX_CONNECTOR_ITEMS = 80

export function buildRetrievalIndex(input: RetrievalIndexInput = {}, now = Date.now()): RetrievalIndex {
  const records = new Map<string, RetrievalRecord>()
  const add = (record: Partial<RetrievalRecord> & { kind?: unknown }) => {
    const normalized = normalizeRetrievalRecord(record)
    if (!normalized) return
    const existing = records.get(normalized.id)
    records.set(normalized.id, existing ? mergeRecords(existing, normalized) : normalized)
  }

  for (const record of input.records ?? []) add(record)
  for (const file of input.codeFiles ?? []) add(codeRecord(file))
  for (const source of input.documents ?? []) add(sourceRecord('doc', source))
  for (const event of input.connectorEvents ?? []) add(connectorEventRecord(event))
  for (const thread of input.threads ?? []) addThreadRecords(thread, add)
  for (const mission of input.missions ?? []) addMissionRecords(mission, add)
  for (const room of input.rooms ?? []) addRoomRecords(room, add)
  for (const ticket of input.tickets ?? []) addTicketRecords(ticket, add)
  for (const memory of input.projectMemory ?? []) add(projectMemoryRecord(memory))
  for (const artifact of input.artifacts ?? []) add(artifactRecord(artifact))

  const list = [...records.values()].sort((a, b) => (b.createdAt ?? 0) - (a.createdAt ?? 0) || a.kind.localeCompare(b.kind) || a.title.localeCompare(b.title))
  return {
    schema: RETRIEVAL_INDEX_SCHEMA,
    generatedAt: now,
    records: list,
    stats: indexStats(list),
  }
}

export async function buildWorkspaceRetrievalIndex(input: WorkspaceRetrievalIndexInput): Promise<RetrievalIndex> {
  const kinds = normalizeKinds(input.kinds)
  const limit = Math.max(1, Math.min(input.maxConnectorItems ?? DEFAULT_MAX_CONNECTOR_ITEMS, 500))
  const query = clean(input.query)
  const data: RetrievalIndexInput = {
    records: input.records,
    connectorEvents: input.connectorEvents,
    threads: input.threads,
    missions: input.missions,
    rooms: input.rooms,
    tickets: input.tickets,
    projectMemory: input.projectMemory,
    artifacts: input.artifacts,
  }

  if (kinds.includes('code')) {
    data.codeFiles = await collectCodeFiles(input.cwd, input.path ?? undefined, input.maxCodeFiles ?? DEFAULT_MAX_CODE_FILES)
  }

  if (kinds.includes('doc') && input.documentSettings) {
    const documents = await scanDocumentItems(input.documentSettings, { query: query || undefined, limit })
    const tables = await scanTableItems(input.documentSettings, { query: query || undefined, limit })
    data.documents = [
      ...documents.documents.map(documentSourceRef),
      ...tables.tables.map(documentSourceRef),
      ...(data.documents ?? []),
    ]
  }

  if (kinds.includes('email') && input.emailSettings) {
    const { messages } = await scanEmailMessages(input.emailSettings, { query: query || undefined, limit })
    data.documents = [
      ...(data.documents ?? []),
      ...messages.map(emailSourceRef).map((source) => ({ ...source, connectorId: 'email' as const })),
    ]
  }

  if (kinds.includes('chat') && input.slackSettings) {
    const { messages } = await scanSlackMessages(input.slackSettings, { query: query || undefined, limit })
    data.documents = [
      ...(data.documents ?? []),
      ...messages.map(slackSourceRef),
    ]
  }

  if (kinds.includes('summary') && input.calendarSettings) {
    const { events } = await scanCalendarEvents(input.calendarSettings, { query: query || undefined, limit })
    data.documents = [
      ...(data.documents ?? []),
      ...events.map(calendarSourceRef),
    ]
  }

  if (kinds.includes('doc') && input.browserSettings) {
    const { items } = await scanBrowserItems(input.browserSettings, { query: query || undefined, limit })
    data.documents = [
      ...(data.documents ?? []),
      ...items.map(browserSourceRef),
    ]
  }

  return buildRetrievalIndex(data)
}

export function searchRetrievalIndex(index: RetrievalIndex, options: RetrievalSearchOptions = {}): RetrievalSearchResult {
  const query = clean(options.query)
  const kinds = normalizeKinds(options.kinds)
  const terms = tokenize(query)
  const limit = Math.max(1, Math.min(Number(options.limit) || DEFAULT_LIMIT, 50))
  const now = Date.now()
  const hits = index.records
    .filter((record) => kinds.includes(record.kind))
    .map((record) => scoreRecord(record, query, terms, now))
    .filter((hit) => !query || hit.score > 0)
    .sort((a, b) => b.score - a.score || (b.record.createdAt ?? 0) - (a.record.createdAt ?? 0) || a.record.title.localeCompare(b.record.title))
    .slice(0, limit)
  return {
    query,
    kinds,
    generatedAt: index.generatedAt,
    total: index.records.length,
    hits,
    stats: index.stats,
  }
}

export function formatRetrievalSearchResult(result: RetrievalSearchResult): string {
  const counts = RETRIEVAL_KINDS
    .map((kind) => `${kind}:${result.stats.byKind[kind]}`)
    .join(' ')
  const header = [
    `Retrieval index: ${result.stats.total} records (${counts}); ${result.stats.sourceIds} source ids.`,
    `Query: ${result.query || '(none)'}`,
    `Kinds: ${result.kinds.join(', ')}`,
  ]
  if (!result.hits.length) return `${header.join('\n')}\n\nNo retrieval hits matched.`
  const lines = result.hits.map((hit, index) => {
    const record = hit.record
    return [
      `${index + 1}. [${record.kind}] ${record.title} (score ${hit.score})`,
      record.path ? `   Path: ${record.path}` : '',
      record.sourceIds.length ? `   Sources: ${record.sourceIds.join(', ')}` : '',
      `   ${excerpt(record.text, result.query, 260)}`,
    ].filter(Boolean).join('\n')
  })
  return `${header.join('\n')}\n\nResults:\n${lines.join('\n\n')}`
}

function addThreadRecords(thread: Thread, add: (record: Partial<RetrievalRecord> & { kind?: unknown }) => void): void {
  for (const [index, message] of (thread.apiMessages ?? []).entries()) {
    const messageText = contentText(message.content)
    if (message.role === 'tool' || !messageText.trim()) continue
    add({
      id: `thread:${thread.id}:message:${index}`,
      kind: 'chat',
      title: `${thread.title} ${message.role}`,
      text: messageText,
      sourceIds: [`thread:${thread.id}:message:${index}`],
      path: thread.cwd,
      createdAt: thread.createdAt,
    })
  }

  for (const item of thread.timeline ?? []) {
    const text = [item.text, item.output, item.meta].filter(Boolean).join('\n')
    if (text.trim()) {
      add({
        id: `thread:${thread.id}:timeline:${item.id}`,
        kind: timelineRetrievalKind(item),
        title: item.title || item.tool || `${thread.title} ${item.kind}`,
        text,
        sourceIds: [`thread:${thread.id}:timeline:${item.id}`],
        path: item.path ?? null,
        createdAt: item.createdAt ?? thread.updatedAt,
      })
    }
    for (const record of artifactRecordsFromTimelineItem(thread, item)) add(record)
  }

  for (const compaction of thread.compactions ?? []) {
    add({
      id: `thread:${thread.id}:compaction:${compaction.id}`,
      kind: 'summary',
      title: `${thread.title} compaction`,
      text: compaction.summaryExcerpt,
      sourceIds: [`thread:${thread.id}:compaction:${compaction.id}`],
      path: thread.cwd,
      createdAt: compaction.at,
      metadata: { trigger: compaction.trigger, messagesFolded: compaction.messagesFolded },
    })
  }
}

function addMissionRecords(mission: Mission, add: (record: Partial<RetrievalRecord> & { kind?: unknown }) => void): void {
  for (const checkpoint of mission.checkpoints ?? []) {
    add({
      id: `mission:${mission.id}:checkpoint:${checkpoint.id}`,
      kind: 'summary',
      title: `${mission.title}: ${checkpoint.label}`,
      text: checkpoint.summary,
      sourceIds: [`mission:${mission.id}:checkpoint:${checkpoint.id}`, ...(checkpoint.artifactRefs ?? []).map((ref) => `artifact:${ref}`)],
      path: mission.cwd ?? null,
      createdAt: checkpoint.createdAt,
      metadata: { missionStatus: checkpoint.status },
    })
    for (const ref of checkpoint.artifactRefs ?? []) {
      add({
        id: `artifact:${ref}`,
        kind: 'artifact',
        title: `Mission artifact ${ref}`,
        text: `${mission.title}: ${checkpoint.summary}`,
        sourceIds: [`artifact:${ref}`, `mission:${mission.id}:checkpoint:${checkpoint.id}`],
        path: mission.cwd ?? null,
        createdAt: checkpoint.createdAt,
      })
    }
  }

  for (const item of mission.scratchMemory ?? []) {
    add({
      id: `mission:${mission.id}:scratch:${item.id}`,
      kind: 'summary',
      title: `${mission.title} scratch ${item.kind}`,
      text: item.text,
      sourceIds: [`mission:${mission.id}:scratch:${item.id}`],
      path: mission.cwd ?? null,
      createdAt: item.createdAt,
    })
  }
}

function addRoomRecords(room: Room, add: (record: Partial<RetrievalRecord> & { kind?: unknown }) => void): void {
  if (room.memorySummary?.trim()) {
    add({
      id: `room:${room.id}:memory-summary`,
      kind: 'summary',
      title: `${room.name} memory summary`,
      text: room.memorySummary,
      sourceIds: [`room:${room.id}:memory-summary`],
      path: room.workspaceCwd ?? room.rootCwd ?? null,
      createdAt: room.updatedAt,
    })
  }
  for (const memory of room.memory ?? []) {
    add({
      id: `room:${room.id}:memory:${memory.id}`,
      kind: 'summary',
      title: `${room.name} ${memory.kind}`,
      text: memory.text,
      sourceIds: [`room:${room.id}:memory:${memory.id}`],
      path: room.workspaceCwd ?? room.rootCwd ?? null,
      createdAt: memory.createdAt,
    })
  }
  for (const event of room.events ?? []) {
    add({
      id: `room:${room.id}:event:${event.id}`,
      kind: event.kind === 'tool' ? 'artifact' : 'chat',
      title: `${room.name} ${event.seat}`,
      text: [event.kind, event.tool, event.text].filter(Boolean).join(' '),
      sourceIds: [`room:${room.id}:event:${event.id}`],
      path: event.path ?? room.workspaceCwd ?? room.rootCwd ?? null,
      createdAt: event.createdAt,
    })
  }
}

function addTicketRecords(ticket: Ticket, add: (record: Partial<RetrievalRecord> & { kind?: unknown }) => void): void {
  for (const evidence of ticket.evidence ?? []) {
    add({
      id: `ticket:${ticket.id}:evidence:${evidence.id}`,
      kind: 'artifact',
      title: `${ticket.title}: ${evidence.kind}`,
      text: evidence.summary,
      sourceIds: [`ticket:${ticket.id}:evidence:${evidence.id}`, evidence.ref].filter((value): value is string => !!value),
      path: evidence.ref ?? null,
      createdAt: evidence.createdAt,
      metadata: { ticketId: ticket.id, evidenceKind: evidence.kind },
    })
  }
}

function codeRecord(file: RetrievalCodeFileInput): Partial<RetrievalRecord> {
  const filePath = normalizeSlash(file.path)
  return {
    id: `code:${filePath}`,
    kind: 'code',
    title: filePath,
    text: file.content,
    sourceIds: [`code:${filePath}`],
    path: filePath,
    createdAt: file.modifiedAt ?? null,
  }
}

function sourceRecord(kind: RetrievalKind, source: ConnectorSourceRef): Partial<RetrievalRecord> {
  const mappedKind = connectorKind(source.connectorId, kind)
  return {
    id: source.id,
    kind: mappedKind,
    title: source.title || source.id,
    text: [source.title, source.excerpt, source.path].filter(Boolean).join('\n'),
    sourceIds: [source.id],
    path: source.path ?? null,
    createdAt: source.createdAt ?? null,
    metadata: { connectorId: source.connectorId, sourceId: source.sourceId ?? null, itemId: source.itemId ?? null },
  }
}

function connectorEventRecord(event: ConnectorEvent): Partial<RetrievalRecord> {
  const sources = event.sources ?? []
  const sourceIds = sources.length ? sources.map((source) => source.id) : [`connector-event:${event.id}`]
  return {
    id: `connector-event:${event.id}`,
    kind: connectorKind(event.connectorId, event.connectorId === 'email' ? 'email' : 'summary'),
    title: event.title,
    text: [
      event.pluginName,
      event.actor,
      event.threadKey,
      event.summary,
      ...sources.map((source) => [source.title, source.excerpt, source.path].filter(Boolean).join(' ')),
    ].filter(Boolean).join('\n'),
    sourceIds,
    path: sources.find((source) => source.path)?.path ?? null,
    createdAt: event.createdAt,
    metadata: { connectorId: event.connectorId, action: event.action, priority: event.priority },
  }
}

function projectMemoryRecord(memory: ProjectMemoryEntry): Partial<RetrievalRecord> {
  return {
    id: `memory:${memory.id}`,
    kind: 'summary',
    title: `${memory.kind} memory`,
    text: memory.text,
    sourceIds: normalizeSourceIds(memory.sourceIds, memory.sourceRef ? [memory.sourceRef] : [`memory:${memory.id}`]),
    path: memory.sourceType === 'file' ? memory.sourceRef : null,
    createdAt: memory.updatedAt,
    metadata: { projectId: memory.projectId, sourceType: memory.sourceType, confidence: memory.confidence, privacyScope: memory.privacyScope },
  }
}

function artifactRecord(artifact: RetrievalArtifactInput): Partial<RetrievalRecord> {
  const id = clean(artifact.id) || `artifact:${hashText(`${artifact.title ?? ''}:${artifact.text ?? ''}`)}`
  return {
    id: id.startsWith('artifact:') ? id : `artifact:${id}`,
    kind: 'artifact',
    title: clean(artifact.title) || id,
    text: clean(artifact.text) || clean(artifact.title) || id,
    sourceIds: normalizeSourceIds(artifact.sourceIds, [`artifact:${id.replace(/^artifact:/, '')}`]),
    path: artifact.path ?? null,
    createdAt: artifact.createdAt ?? null,
  }
}

function artifactRecordsFromTimelineItem(thread: Thread, item: TimelineItem): Array<Partial<RetrievalRecord>> {
  if (item.kind !== 'tool') return []
  const records: Array<Partial<RetrievalRecord>> = []
  const artifactId = artifactIdFromToolItem(item)
  if (artifactId) {
    records.push({
      id: `artifact:${artifactId}`,
      kind: 'artifact',
      title: item.title || item.tool || artifactId,
      text: [item.output, item.meta, item.command].filter(Boolean).join('\n'),
      sourceIds: [`artifact:${artifactId}`, `thread:${thread.id}:timeline:${item.id}`],
      path: item.path ?? null,
      createdAt: item.createdAt ?? thread.updatedAt,
    })
  }
  if (item.diff?.length || item.path) {
    const id = `thread:${thread.id}:artifact:${item.id}`
    records.push({
      id,
      kind: 'artifact',
      title: item.title || item.path || item.tool || 'Tool artifact',
      text: [item.output, item.meta, item.diff?.map((line) => line.text).join('\n')].filter(Boolean).join('\n'),
      sourceIds: [id, `thread:${thread.id}:timeline:${item.id}`],
      path: item.path ?? null,
      createdAt: item.createdAt ?? thread.updatedAt,
    })
  }
  return records
}

async function collectCodeFiles(cwd: string, scopedPath: string | undefined, maxFiles: number): Promise<RetrievalCodeFileInput[]> {
  const root = safeResolveWorkspace(cwd, scopedPath || '.')
  const files: RetrievalCodeFileInput[] = []
  const walk = async (dir: string, depth: number) => {
    if (depth > 16 || files.length >= maxFiles) return
    let entries: any[] = []
    try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
    for (const entry of entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))) {
      if (files.length >= maxFiles) break
      if (entry.name.startsWith('.') || SKIP_DIRS.has(entry.name)) continue
      if (entry.isSymbolicLink()) continue
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        await walk(full, depth + 1)
        continue
      }
      if (!isCodeFile(entry.name)) continue
      try {
        const stat = await fs.stat(full)
        if (!stat.isFile() || stat.size > MAX_CODE_BYTES) continue
        const content = await fs.readFile(full, 'utf8')
        files.push({ path: normalizeSlash(path.relative(path.resolve(cwd), full)), content, modifiedAt: stat.mtimeMs })
      } catch {
        // Ignore binary/unreadable files.
      }
    }
  }
  await walk(root, 0)
  return files
}

function safeResolveWorkspace(cwd: string, relPath: string): string {
  const root = path.resolve(cwd)
  const realRoot = realpathSync.native(root)
  const target = path.resolve(root, relPath || '.')
  if (target !== root && !target.startsWith(root + path.sep)) throw new Error(`Retrieval path "${relPath}" escapes the workspace.`)
  let probe = target
  while (!existsSync(probe)) {
    const parent = path.dirname(probe)
    if (parent === probe) break
    probe = parent
  }
  const realProbe = realpathSync.native(probe)
  if (realProbe !== realRoot && !realProbe.startsWith(realRoot + path.sep)) throw new Error(`Retrieval path "${relPath}" escapes the workspace through a symlink.`)
  return target
}

function normalizeRetrievalRecord(record: Partial<RetrievalRecord> & { kind?: unknown }): RetrievalRecord | null {
  const kind = normalizeKind(record.kind)
  if (!kind) return null
  const title = clean(record.title) || clean(record.path) || kind
  const text = compact(clean(record.text), MAX_RECORD_TEXT)
  if (!text) return null
  const sourceIds = normalizeSourceIds(record.sourceIds, record.id ? [String(record.id)] : undefined)
  const id = clean(record.id) || `${kind}:${hashText(`${title}:${text}`)}`
  return {
    id,
    kind,
    title,
    text,
    sourceIds,
    path: record.path ?? null,
    createdAt: finite(record.createdAt) ?? null,
    metadata: normalizeMetadata(record.metadata),
  }
}

function mergeRecords(a: RetrievalRecord, b: RetrievalRecord): RetrievalRecord {
  return {
    ...a,
    title: a.title || b.title,
    text: compact(`${a.text}\n${b.text}`, MAX_RECORD_TEXT),
    sourceIds: unique([...a.sourceIds, ...b.sourceIds]),
    path: a.path ?? b.path ?? null,
    createdAt: Math.max(a.createdAt ?? 0, b.createdAt ?? 0) || a.createdAt || b.createdAt || null,
    metadata: { ...(a.metadata ?? {}), ...(b.metadata ?? {}) },
  }
}

// Recency decay: a fresh record keeps its full score; an old one decays toward a floor
// so stale content ranks below comparably-matching recent content. Half-life in days.
export function recencyWeight(createdAt: number | null | undefined, now: number, halfLifeDays = 30, floor = 0.5): number {
  if (!createdAt || createdAt <= 0) return floor
  const ageDays = Math.max(0, (now - createdAt) / 86_400_000)
  return floor + (1 - floor) * Math.pow(2, -ageDays / halfLifeDays)
}

function scoreRecord(record: RetrievalRecord, query: string, terms: string[], now = Date.now()): RetrievalSearchHit {
  if (!query) return { record, score: Math.max(1, Math.round((record.createdAt ?? 0) / 1_000_000_000)), matchedTerms: [] }
  const title = record.title.toLowerCase()
  const text = record.text.toLowerCase()
  const pathText = String(record.path ?? '').toLowerCase()
  const sourceText = record.sourceIds.join(' ').toLowerCase()
  const phrase = query.toLowerCase()
  let score = 0
  if (title.includes(phrase)) score += 18
  if (text.includes(phrase)) score += 10
  if (pathText.includes(phrase)) score += 5
  if (sourceText.includes(phrase)) score += 6
  const matchedTerms: string[] = []
  for (const term of terms) {
    let termScore = 0
    if (title.includes(term)) termScore += 6
    if (text.includes(term)) termScore += countMatches(text, term)
    if (pathText.includes(term)) termScore += 3
    if (sourceText.includes(term)) termScore += 4
    if (termScore > 0) {
      matchedTerms.push(term)
      score += termScore
    }
  }
  // Apply recency decay so equally-matching records favour the fresher one.
  if (score > 0) score = Math.max(1, Math.round(score * recencyWeight(record.createdAt, now)))
  return { record, score, matchedTerms: unique(matchedTerms) }
}

function indexStats(records: RetrievalRecord[]): RetrievalIndexStats {
  const byKind = emptyKindCounts()
  const sourceIds = new Set<string>()
  for (const record of records) {
    byKind[record.kind]++
    for (const sourceId of record.sourceIds) sourceIds.add(sourceId)
  }
  return { total: records.length, byKind, sourceIds: sourceIds.size }
}

function emptyKindCounts(): Record<RetrievalKind, number> {
  return { code: 0, doc: 0, chat: 0, email: 0, summary: 0, artifact: 0 }
}

function connectorKind(connectorId: string, fallback: RetrievalKind): RetrievalKind {
  if (connectorId === 'email') return 'email'
  if (connectorId === 'slack' || connectorId === 'telegram' || connectorId === 'whatsapp' || connectorId === 'imessage') return 'chat'
  if (connectorId === 'docs' || connectorId === 'browser') return 'doc'
  if (connectorId === 'calendar') return 'summary'
  return fallback
}

function timelineRetrievalKind(item: TimelineItem): RetrievalKind {
  if (item.kind === 'tool') return 'artifact'
  if (item.kind === 'user' || item.kind === 'assistant' || item.kind === 'notice') return 'chat'
  return 'summary'
}

function artifactIdFromToolItem(item: TimelineItem): string | null {
  const text = [item.meta, item.output].filter(Boolean).join('\n')
  return /(?:artifacts=|Artifact:\s*)([A-Za-z0-9._:-]+)/.exec(text)?.[1] ?? null
}

function normalizeKinds(value: RetrievalSearchOptions['kinds']): RetrievalKind[] {
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[,;\s]+/)
      : []
  const kinds = unique(raw.map((item) => normalizeKind(item)).filter((item): item is RetrievalKind => !!item))
  return kinds.length ? kinds : [...RETRIEVAL_KINDS]
}

function normalizeKind(value: unknown): RetrievalKind | null {
  const text = clean(value).toLowerCase()
  if (text === 'docs' || text === 'document' || text === 'documents' || text === 'browser') return 'doc'
  if (text === 'messages' || text === 'message' || text === 'slack') return 'chat'
  if (text === 'emails' || text === 'mail') return 'email'
  if (text === 'summaries' || text === 'memory' || text === 'calendar') return 'summary'
  if (text === 'artifacts' || text === 'evidence') return 'artifact'
  return (RETRIEVAL_KINDS as readonly string[]).includes(text) ? text as RetrievalKind : null
}

function normalizeSourceIds(value: unknown, fallback: string[] = []): string[] {
  const raw = Array.isArray(value)
    ? value
    : typeof value === 'string'
      ? value.split(/[,;\n]/)
      : []
  const ids = raw.map((item) => clean(item)).filter(Boolean)
  for (const id of fallback.map((item) => clean(item)).filter(Boolean)) {
    if (!ids.includes(id)) ids.push(id)
  }
  return unique(ids)
}

function isCodeFile(name: string): boolean {
  const ext = path.extname(name).toLowerCase()
  return CODE_EXTENSIONS.has(ext) && !BLOCKED_NAMES.has(name) && !BLOCKED_EXTENSIONS.has(ext)
}

function tokenize(query: string): string[] {
  return unique(query.toLowerCase().split(/[^a-z0-9_./:-]+/).map((item) => item.trim()).filter((item) => item.length >= 2)).slice(0, 12)
}

function excerpt(text: string, query: string, max: number): string {
  const flat = text.replace(/\s+/g, ' ').trim()
  if (!query) return flat.slice(0, max)
  const lower = flat.toLowerCase()
  const terms = tokenize(query)
  const index = terms.map((term) => lower.indexOf(term)).filter((pos) => pos >= 0).sort((a, b) => a - b)[0] ?? 0
  const start = Math.max(0, index - 60)
  const slice = flat.slice(start, start + max)
  return `${start > 0 ? '...' : ''}${slice}${start + max < flat.length ? '...' : ''}`
}

function compact(value: string, max: number): string {
  const text = value.replace(/\s+/g, ' ').trim()
  return text.length <= max ? text : `${text.slice(0, max - 3).trim()}...`
}

function clean(value: unknown): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

function normalizeSlash(value: string): string {
  return value.replace(/\\/g, '/').replace(/^\.\/+/, '')
}

function countMatches(haystack: string, needle: string): number {
  let count = 0
  let index = 0
  while ((index = haystack.indexOf(needle, index)) >= 0) {
    count++
    index += needle.length
  }
  return count
}

function finite(value: unknown): number | null {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function normalizeMetadata(value: unknown): Record<string, string | number | boolean | null> | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined
  const out: Record<string, string | number | boolean | null> = {}
  for (const [key, raw] of Object.entries(value)) {
    if (raw == null || typeof raw === 'string' || typeof raw === 'number' || typeof raw === 'boolean') out[key] = raw ?? null
  }
  return Object.keys(out).length ? out : undefined
}

function hashText(value: string): string {
  return createHash('sha256').update(value).digest('hex').slice(0, 12)
}

function unique<T>(values: T[]): T[] {
  return [...new Set(values)]
}
