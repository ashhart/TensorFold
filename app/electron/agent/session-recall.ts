import { contentText } from './content.ts'
import type { ApiMessage, Thread, TimelineItem } from './types'

export interface SessionSearchOptions {
  query: string
  projectId?: string | null
  currentThreadId?: string | null
  includeCurrent?: boolean
  includeArchived?: boolean
  limit?: number
}

export interface SessionSearchMatch {
  threadId: string
  title: string
  kind: Thread['kind']
  projectId: string | null
  updatedAt: number
  score: number
  excerpt: string
  sourceIds: string[]
}

export interface SessionSearchResult {
  query: string
  matches: SessionSearchMatch[]
  omitted: number
}

export interface SessionReadOptions {
  threadId: string
  maxMessages?: number
}

export interface SessionReadMessage {
  role: string
  content: string
  sourceId: string
}

export interface SessionReadResult {
  thread: Thread | null
  messages: SessionReadMessage[]
  omittedBefore: number
}

export function searchSessionThreads(threads: Thread[], options: SessionSearchOptions): SessionSearchResult {
  const query = compact(options.query)
  const terms = queryTerms(query)
  const limit = clampLimit(options.limit, 12, 1, 50)
  if (!terms.length) return { query, matches: [], omitted: 0 }

  const candidates = threads.filter((thread) => {
    if (thread.incognito) return false
    if (!options.includeCurrent && options.currentThreadId && thread.id === options.currentThreadId) return false
    if (!options.includeArchived && thread.archivedAt) return false
    if (options.projectId && thread.projectId !== options.projectId) return false
    return true
  })
  const matches = candidates
    .map((thread) => scoreThread(thread, terms))
    .filter((match): match is SessionSearchMatch => !!match)
    .sort((a, b) => b.score - a.score || b.updatedAt - a.updatedAt)

  return { query, matches: matches.slice(0, limit), omitted: Math.max(0, matches.length - limit) }
}

export function readSessionThread(threads: Thread[], options: SessionReadOptions): SessionReadResult {
  const threadId = String(options.threadId ?? '').trim()
  const thread = threads.find((item) => item.id === threadId && !item.incognito) ?? null
  if (!thread) return { thread: null, messages: [], omittedBefore: 0 }

  const allMessages = (thread.apiMessages ?? [])
    .map((message, index) => messageToReadMessage(thread, message, index))
    .filter((message): message is SessionReadMessage => !!message)
  const maxMessages = clampLimit(options.maxMessages, 12, 1, 80)
  if (allMessages.length <= maxMessages) return { thread, messages: allMessages, omittedBefore: 0 }

  const omittedBefore = allMessages.length - maxMessages
  return {
    thread,
    messages: [
      { role: 'notice', content: `truncated ${omittedBefore} earlier messages`, sourceId: `thread:${thread.id}:truncated` },
      ...allMessages.slice(-maxMessages),
    ],
    omittedBefore,
  }
}

export function formatSessionSearchResult(result: SessionSearchResult): string {
  if (!result.matches.length) return `No prior local sessions matched "${result.query}".`
  const lines = result.matches.map((match, index) => [
    `${index + 1}. ${match.title} (${match.kind}, updated ${new Date(match.updatedAt).toISOString()})`,
    `   thread: ${match.threadId}${match.projectId ? ` · project: ${match.projectId}` : ''}`,
    `   excerpt: ${match.excerpt}`,
    `   sources: ${match.sourceIds.join(', ')}`,
  ].join('\n'))
  if (result.omitted) lines.push(`…${result.omitted} more matches omitted.`)
  return lines.join('\n\n')
}

export function formatSessionReadResult(result: SessionReadResult): string {
  if (!result.thread) return 'No prior local session found for that thread id.'
  const header = [
    `Session: ${result.thread.title}`,
    `Thread: ${result.thread.id}`,
    `Kind: ${result.thread.kind}`,
    `Updated: ${new Date(result.thread.updatedAt).toISOString()}`,
  ].join('\n')
  const body = result.messages.map((message) => [
    `[${message.role}] ${message.sourceId}`,
    message.content,
  ].join('\n')).join('\n\n---\n\n')
  return `${header}\n\n${body}`
}

function scoreThread(thread: Thread, terms: string[]): SessionSearchMatch | null {
  const snippets: Array<{ text: string; sourceId: string; weight: number }> = []
  if (thread.title) snippets.push({ text: thread.title, sourceId: `thread:${thread.id}:title`, weight: 3 })
  for (const [index, message] of (thread.apiMessages ?? []).entries()) {
    const text = messageText(message)
    if (text) snippets.push({ text, sourceId: `thread:${thread.id}:message:${index}`, weight: 2 })
  }
  for (const item of thread.timeline ?? []) {
    const text = timelineText(item)
    if (text) snippets.push({ text, sourceId: `thread:${thread.id}:timeline:${item.id}`, weight: 1 })
  }

  let score = 0
  const sourceIds: string[] = []
  const excerpts: string[] = []
  for (const snippet of snippets) {
    const haystack = snippet.text.toLowerCase()
    const hits = terms.filter((term) => haystack.includes(term)).length
    if (!hits) continue
    score += hits * snippet.weight
    sourceIds.push(snippet.sourceId)
    if (excerpts.length < 2) excerpts.push(excerpt(snippet.text, terms))
  }
  if (!score) return null

  return {
    threadId: thread.id,
    title: thread.title || 'Untitled session',
    kind: thread.kind,
    projectId: thread.projectId ?? null,
    updatedAt: thread.updatedAt || thread.createdAt || 0,
    score,
    excerpt: excerpts.join(' … '),
    sourceIds: [...new Set(sourceIds)].slice(0, 6),
  }
}

function messageToReadMessage(thread: Thread, message: ApiMessage, index: number): SessionReadMessage | null {
  const text = messageText(message)
  if (!text) return null
  return {
    role: message.role,
    content: text,
    sourceId: `thread:${thread.id}:message:${index}`,
  }
}

function messageText(message: ApiMessage): string {
  return compact(contentText(message.content))
}

function timelineText(item: TimelineItem): string {
  return compact([
    item.title,
    item.tool,
    item.text,
    item.command,
    item.meta,
  ].filter(Boolean).join(' '))
}

function queryTerms(query: string): string[] {
  return [...new Set(query.toLowerCase().split(/[^a-z0-9_/-]+/).map((term) => term.trim()).filter((term) => term.length >= 2))]
}

function excerpt(text: string, terms: string[], max = 220): string {
  const compacted = compact(text)
  const lower = compacted.toLowerCase()
  const firstHit = terms.map((term) => lower.indexOf(term)).filter((index) => index >= 0).sort((a, b) => a - b)[0] ?? 0
  const start = Math.max(0, firstHit - 60)
  const slice = compacted.slice(start, start + max)
  return `${start > 0 ? '…' : ''}${slice}${start + max < compacted.length ? '…' : ''}`
}

function compact(value: string): string {
  return String(value ?? '').replace(/\s+/g, ' ').trim()
}

function clampLimit(value: unknown, fallback: number, min: number, max: number): number {
  const parsed = Math.floor(Number(value))
  if (!Number.isFinite(parsed)) return fallback
  return Math.max(min, Math.min(max, parsed))
}
