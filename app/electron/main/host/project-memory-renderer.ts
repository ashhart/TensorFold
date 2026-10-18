import path from 'node:path'
import { statSync } from 'node:fs'
import {
  detectProjectMemoryWarnings,
  formatProjectMemoryEntryForPrompt,
  formatProjectMemoryWarnings,
  scopeProjectMemoryEntries,
  type ProjectMemorySourceSnapshot,
} from '../../agent/project-memory'
import type { Project, ProjectMemoryEntry } from '../../agent/types'
import { selectRelevantMemory } from '../../agent/memory-harvest'
import { store } from '../state'

export function projectMemorySourceSnapshots(project: Project, entries: ProjectMemoryEntry[]): Record<string, ProjectMemorySourceSnapshot> {
  const sources: Record<string, ProjectMemorySourceSnapshot> = {}
  for (const entry of scopeProjectMemoryEntries(entries, project.id)) {
    const ref = entry.sourceRef?.trim()
    if (!ref || entry.sourceType !== 'file' || sources[ref]) continue
    // File refs belong to the remote host. A local lookalike must not make stale remote
    // memory appear freshly verified.
    if (project.workspace?.kind === 'ssh') {
      sources[ref] = { sourceRef: ref, exists: false, updatedAt: null }
      continue
    }
    const resolved = path.resolve(project.cwd, ref)
    const relative = path.relative(project.cwd, resolved)
    if (relative.startsWith('..') || path.isAbsolute(relative)) {
      sources[ref] = { sourceRef: ref, exists: false, updatedAt: null }
      continue
    }
    try {
      const stat = statSync(resolved)
      sources[ref] = { sourceRef: ref, exists: stat.isFile(), updatedAt: stat.mtimeMs }
    } catch {
      sources[ref] = { sourceRef: ref, exists: false, updatedAt: null }
    }
  }
  return sources
}

export function renderProjectMemoryBucket(project: Project | undefined, query?: string): string {
  if (!project) return ''
  let entries = scopeProjectMemoryEntries(store.projectMemory, project.id)
  // Query-driven recall: with a lot of memory + a current query, inject only the
  // entries relevant to this turn instead of the whole bucket (tighter, on-topic).
  if (query && query.trim() && entries.length > 6) {
    const relevant = selectRelevantMemory(entries.map((e) => ({ id: e.id, text: e.text })), query, 8)
    if (relevant.length) { const keep = new Set(relevant.map((r) => r.id)); entries = entries.filter((e) => keep.has(e.id)) }
  }
  const parts: string[] = []
  const warnings = detectProjectMemoryWarnings(entries, projectMemorySourceSnapshots(project, entries))
  if (warnings.length) {
    parts.push([
      'MEMORY WARNINGS — stale or unverifiable durable memory is reference-only. Prefer live files, user instructions, and security policy.',
      formatProjectMemoryWarnings(warnings),
    ].join('\n'))
  }
  if (project.memory?.trim()) parts.push(project.memory.trim())
  if (entries.length) {
    const byKind = entries.map(formatProjectMemoryEntryForPrompt)
    parts.push(byKind.join('\n'))
  }
  return parts.join('\n\n')
}
