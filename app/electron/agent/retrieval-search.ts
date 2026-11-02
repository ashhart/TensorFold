import { RETRIEVAL_KINDS } from './retrieval-types'
import type { RetrievalIndex, RetrievalRecord, RetrievalSearchHit, RetrievalSearchOptions, RetrievalSearchResult } from './retrieval-types'
import { clean, normalizeKinds, unique } from './retrieval-records'

const DEFAULT_LIMIT = 12

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
  const scan = result.stats.codeScan
  if (scan) {
    header.push(`Code scan (${scan.scope}): read ${scan.readFiles} of ${scan.discoveredFiles} discovered files; visited ${scan.visitedDirectories} directories and ${scan.visitedEntries} entries.`)
    if (scan.skippedFiles) header.push(`Code files skipped as oversized, non-regular, or unreadable: ${scan.skippedFiles}.`)
    if (scan.truncated) {
      const reasons = scan.limits.map(limit => ({
        files: `${scan.maxFiles}-file read budget`, directories: `${scan.maxDirectories}-directory budget`,
        entries: `${scan.maxEntries}-entry budget`, directory_entries: `${scan.maxEntriesPerDirectory}-entry directory budget`,
        depth: `${scan.maxDepth}-level depth budget`,
      })[limit])
      if (scan.unreadableDirectories) reasons.push(`${scan.unreadableDirectories} unreadable directories`)
      header.push(`Partial code coverage: ${reasons.join(', ')}. Narrow the path to search the relevant source tree.`)
    }
  }
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

function countMatches(haystack: string, needle: string): number {
  let count = 0
  let index = 0
  while ((index = haystack.indexOf(needle, index)) >= 0) {
    count++
    index += needle.length
  }
  return count
}
