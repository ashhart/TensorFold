import { isOperationalRoomEventKind, roomWithoutSavedRecall } from '../../shared/thread-memory-policy'
import { browserSourceRef, scanBrowserItems } from './browser-connector.ts'
import { calendarSourceRef, scanCalendarEvents } from './calendar-connector.ts'
import { documentSourceRef, scanDocumentItems, scanTableItems } from './document-connector.ts'
import { emailSourceRef, scanEmailMessages } from './email-connector.ts'
import { scanSlackMessages, slackSourceRef } from './slack-connector.ts'
import { contentText } from './content.ts'
import { canRecallThread } from './session-recall'
import { isActiveProjectMemoryEntry } from './project-memory'
import type {
  ConnectorEvent,
  ConnectorSourceRef,
  Mission,
  ProjectMemoryEntry,
  Room,
  Thread,
  Ticket,
  TimelineItem,
} from './types'

import { RETRIEVAL_INDEX_SCHEMA } from './retrieval-types'
import type {
  RetrievalArtifactInput, RetrievalCodeFileInput, RetrievalIndex, RetrievalIndexInput,
  RetrievalKind, RetrievalRecord, WorkspaceRetrievalIndexInput,
} from './retrieval-types'
import { collectCodeFiles } from './retrieval-code'
import {
  clean, hashText, indexStats, mergeRecords, normalizeKinds, normalizeRetrievalRecord,
  normalizeSlash, normalizeSourceIds,
} from './retrieval-records'
export * from './retrieval-types'
export { searchRetrievalIndex, formatRetrievalSearchResult, recencyWeight } from './retrieval-search'

const DEFAULT_MAX_CODE_FILES = 300
const DEFAULT_MAX_CONNECTOR_ITEMS = 80

function visibleThreadRecord(record: Partial<RetrievalRecord>, visibleThreadIds: Set<string>, currentThreadId?: string, visibleMissionIds?: Set<string>): boolean {
  const threadSources = [record.id, ...record.sourceIds ?? []].filter((id): id is string => typeof id === 'string' && id.startsWith('thread:'))
  const missionSources = [record.id, ...record.sourceIds ?? []].filter((id): id is string => typeof id === 'string' && id.startsWith('mission:'))
  return threadSources.every((id) => visibleThreadIds.has(id.split(':')[1]))
    && (!visibleMissionIds || missionSources.every((id) => visibleMissionIds.has(id.split(':')[1])))
    && !((record.metadata?.phoneLocalOnly === true || record.metadata?.googleLocalOnly === true) && record.metadata.threadId !== currentThreadId)
}

// Revalidate sources after asynchronous collection before exposing source text.
export function filterRetrievalThreadSources(index: RetrievalIndex, threads: Thread[], currentThreadId?: string, missions: Mission[] = []): RetrievalIndex {
  const ids = new Set(threads.filter((thread) => canRecallThread(thread, currentThreadId)).map((thread) => thread.id))
  const missionIds = new Set(missions.filter((mission) => !mission.threadId || ids.has(mission.threadId)).map((mission) => mission.id))
  const records = index.records.filter((record) => visibleThreadRecord(record, ids, currentThreadId, missionIds))
  return { ...index, records, stats: { ...indexStats(records), ...(index.stats.codeScan ? { codeScan: index.stats.codeScan } : {}) } }
}

function projectMemorySourceId(memory: Pick<ProjectMemoryEntry, 'projectId' | 'id'>): string {
  return `project-memory:${encodeURIComponent(memory.projectId)}:${encodeURIComponent(memory.id)}`
}

function projectMemoryRecordVisible(record: Partial<RetrievalRecord>, entries: ProjectMemoryEntry[]): boolean {
  const sources = [record.id, ...record.sourceIds ?? []].filter((id): id is string => typeof id === 'string' && /^(?:project-memory|memory):/.test(id))
  return sources.every((source) => {
    const matches = source.startsWith('project-memory:')
      ? entries.filter((entry) => projectMemorySourceId(entry) === source)
      : entries.filter((entry) => `memory:${entry.id}` === source && (typeof record.metadata?.projectId !== 'string' || entry.projectId === record.metadata.projectId))
    // Old id-only records are unsafe if the id exists in more than one project.
    return matches.length === 1 && isActiveProjectMemoryEntry(matches[0])
  })
}

// Cached records retain memory provenance and must pass the current store.
export function filterRetrievalProjectMemorySources(index: RetrievalIndex, entries: ProjectMemoryEntry[]): RetrievalIndex {
  const records = index.records.filter((record) => projectMemoryRecordVisible(record, entries))
  return { ...index, records, stats: { ...indexStats(records), ...(index.stats.codeScan ? { codeScan: index.stats.codeScan } : {}) } }
}

export function buildRetrievalIndex(input: RetrievalIndexInput = {}, now = Date.now()): RetrievalIndex {
  const records = new Map<string, RetrievalRecord>()
  const visibleThreadIds = new Set((input.threads ?? []).filter((thread) => canRecallThread(thread, input.currentThreadId)).map((thread) => thread.id))
  const visibleMissions = (input.missions ?? []).filter((mission) => !mission.threadId || visibleThreadIds.has(mission.threadId))
  const visibleMissionIds = new Set(visibleMissions.map((mission) => mission.id))
  const add = (record: Partial<RetrievalRecord> & { kind?: unknown }) => {
    // Prebuilt artifacts keep provenance; changing their kind cannot bypass live thread policy.
    if (!visibleThreadRecord(record, visibleThreadIds, input.currentThreadId, visibleMissionIds)) return
    if (!projectMemoryRecordVisible(record, input.projectMemory ?? [])) return
    if (input.savedMemoryAllowed === false && [record.id, ...record.sourceIds ?? []].some((id) => typeof id === 'string' && (/^(?:project-memory|memory):/.test(id) || /^room:[^:]+:memory(?:-|:)/.test(id) || /^room:[^:]+:event:/.test(id) && !isOperationalRoomEventKind(record.metadata?.roomEventKind) || /^thread:/.test(id) && !id.startsWith(`thread:${input.currentThreadId}:`)))) return
    const normalized = normalizeRetrievalRecord(record)
    if (!normalized) return
    const existing = records.get(normalized.id)
    records.set(normalized.id, existing ? mergeRecords(existing, normalized) : normalized)
  }

  for (const record of input.records ?? []) add(record)
  for (const file of input.codeFiles ?? []) add(codeRecord(file))
  for (const source of input.documents ?? []) add(sourceRecord('doc', source))
  for (const event of input.connectorEvents ?? []) add(connectorEventRecord(event))
  for (const thread of input.threads ?? []) if (canRecallThread(thread, input.currentThreadId) && (input.savedMemoryAllowed !== false || thread.id === input.currentThreadId)) addThreadRecords(thread, add)
  for (const mission of visibleMissions) addMissionRecords(mission, (record) => add({ ...record, sourceIds: [...record.sourceIds ?? [], ...(mission.threadId ? [`thread:${mission.threadId}:mission:${mission.id}`] : [])] }))
  for (const room of input.rooms ?? []) addRoomRecords(input.savedMemoryAllowed === false ? roomWithoutSavedRecall(room) : room, add, input.savedMemoryAllowed !== false)
  for (const ticket of input.tickets ?? []) addTicketRecords(ticket, add)
  if (input.savedMemoryAllowed !== false) for (const memory of input.projectMemory ?? []) if (isActiveProjectMemoryEntry(memory)) add(projectMemoryRecord(memory))
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
  let codeScan: RetrievalIndex['stats']['codeScan']
  const data: RetrievalIndexInput = {
    savedMemoryAllowed: input.savedMemoryAllowed,
    currentThreadId: input.currentThreadId,
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
    const collected = await collectCodeFiles(input.cwd, input.path ?? undefined, input.maxCodeFiles ?? DEFAULT_MAX_CODE_FILES)
    data.codeFiles = collected.files
    codeScan = collected.scan
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

  if (input.recallThreads) data.threads = input.recallThreads()
  if (input.recallProjectMemory) data.projectMemory = input.savedMemoryAllowed === false ? [] : input.recallProjectMemory()
  const index = buildRetrievalIndex(data)
  if (codeScan) index.stats.codeScan = codeScan
  return index
}

function addThreadRecords(thread: Thread, add: (record: Partial<RetrievalRecord> & { kind?: unknown }) => void): void {
  if (thread.incognito) return
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

function addRoomRecords(room: Room, add: (record: Partial<RetrievalRecord> & { kind?: unknown }) => void, savedMemoryAllowed = true): void {
  if (savedMemoryAllowed && room.memorySummary?.trim()) {
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
  for (const memory of savedMemoryAllowed ? room.memory ?? [] : []) {
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
      metadata: { roomEventKind: event.kind },
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
  const sourceId = projectMemorySourceId(memory)
  return {
    id: sourceId,
    kind: 'summary',
    title: `${memory.kind} memory`,
    text: memory.text,
    sourceIds: normalizeSourceIds([sourceId, ...(memory.sourceIds ?? []), ...(memory.sourceRef ? [memory.sourceRef] : [])], [sourceId]),
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

function unique<T>(values: T[]): T[] {
  return [...new Set(values)]
}
