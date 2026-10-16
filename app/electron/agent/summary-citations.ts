import type { ConnectorSourceRef } from './types'

export interface SourceCitedSummary {
  summary: string
  sources: ConnectorSourceRef[]
}

export function ensureSummaryCitations<T extends SourceCitedSummary>(result: T): T {
  const summary = result.summary.trim()
  const sources = result.sources ?? []
  if (!sources.length) {
    return {
      ...result,
      summary: hasSourcesSection(summary) ? summary : `${summary}\n\nSources: none.`,
    }
  }
  const missing = sources.filter((source) => !summary.includes(source.id))
  if (!missing.length && hasSourcesSection(summary)) return { ...result, summary }
  const lines = missing.length ? missing.map(formatSourceLine) : sources.map(formatSourceLine)
  return {
    ...result,
    summary: `${summary}\n\n${hasSourcesSection(summary) ? 'Additional sources:' : 'Sources:'}\n${lines.join('\n')}`,
  }
}

function hasSourcesSection(summary: string): boolean {
  return /(^|\n)Sources:/i.test(summary)
}

function formatSourceLine(source: ConnectorSourceRef): string {
  const detail = [source.title, source.excerpt].map((value) => value?.trim()).filter(Boolean).join(' | ')
  return `- ${source.id}${detail ? ` | ${detail}` : ''}`
}
