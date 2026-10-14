// Diff-view depth (VST-040): the pure logic behind the diff panel's per-turn navigation,
// file-tree filter, whitespace-ignore toggle, and "ask why this changed" chat reference.
// Composes diff.ts; no rendering — the panel calls these and draws the result.

import { computeDiff } from './diff'
import type { DiffLine, TimelineItem } from './types'

// ---- whitespace-ignore diff ----

// Diff two texts ignoring pure-whitespace changes: lines that differ only in leading/trailing
// or collapsed inner whitespace are treated as equal, so reindentation/trailing-space churn
// doesn't flood the diff. Implemented by diffing whitespace-normalized text, which drops
// whitespace-only edits from the hunk set.
export function computeDiffIgnoringWhitespace(oldText: string, newText: string, context = 3): DiffLine[] {
  const norm = (text: string) => text.split('\n').map((line) => line.replace(/\s+/g, ' ').trim()).join('\n')
  return computeDiff(norm(oldText), norm(newText), context)
}

export function computeDiffView(oldText: string, newText: string, options: { ignoreWhitespace?: boolean; context?: number } = {}): DiffLine[] {
  return options.ignoreWhitespace
    ? computeDiffIgnoringWhitespace(oldText, newText, options.context)
    : computeDiff(oldText, newText, options.context)
}

// ---- per-turn diff navigation ----

// A turn that produced file edits — the nav model for "jump to the diff at turn N".
export interface DiffTurn {
  turn: number
  itemId: string
  files: string[]
}

const WRITE_TOOLS = new Set(['write_file', 'edit_file', 'apply_patch', 'patch_file'])

// Extract the turns that produced diffs from a timeline, in order. Turn numbers count user
// messages (matching threadTurnNumber), so a diff is attributed to the turn it happened in.
export function diffTurns(timeline: TimelineItem[]): DiffTurn[] {
  const out: DiffTurn[] = []
  let turn = 0
  for (const item of timeline) {
    if (item.kind === 'user') turn++
    const isWrite = (item.tool && WRITE_TOOLS.has(item.tool)) || (item.diff?.length ?? 0) > 0
    if (item.kind === 'tool' && isWrite) {
      const path = item.path ?? (typeof item.args?.path === 'string' ? item.args.path : null) ?? (typeof item.args?.file === 'string' ? item.args.file : null)
      const files = path ? [path] : []
      out.push({ turn: Math.max(1, turn), itemId: item.id, files })
    }
  }
  return out
}

// Next/previous diff turn relative to a current turn (for the panel's ◀ ▶ turn stepper).
// Returns null at the ends so the UI can disable the arrows.
export function stepDiffTurn(turns: DiffTurn[], current: number, direction: 'next' | 'prev'): DiffTurn | null {
  const ordered = [...turns].sort((a, b) => a.turn - b.turn)
  if (direction === 'next') return ordered.find((t) => t.turn > current) ?? null
  return [...ordered].reverse().find((t) => t.turn < current) ?? null
}

// ---- file tree + filter ----

export interface DiffFileNode {
  name: string
  path: string
  isDir: boolean
  children: DiffFileNode[]
}

// Group changed file paths into a directory tree for the panel's file sidebar. A case-
// insensitive substring `filter` narrows to matching leaf paths (keeping their ancestors).
export function buildDiffFileTree(paths: string[], filter = ''): DiffFileNode[] {
  const needle = filter.trim().toLowerCase()
  const kept = needle ? paths.filter((path) => path.toLowerCase().includes(needle)) : paths
  const root: DiffFileNode = { name: '', path: '', isDir: true, children: [] }
  for (const path of [...new Set(kept)].sort()) {
    const parts = path.split('/').filter(Boolean)
    let node = root
    let acc = ''
    parts.forEach((part, index) => {
      acc = acc ? `${acc}/${part}` : part
      const isLeaf = index === parts.length - 1
      let child = node.children.find((c) => c.name === part)
      if (!child) {
        child = { name: part, path: acc, isDir: !isLeaf, children: [] }
        node.children.push(child)
      }
      node = child
    })
  }
  return root.children
}

// ---- "ask why this changed" chat reference ----

export interface AskWhyReference {
  // A composer-insertable prompt referencing a specific file/hunk, so the user can ask the
  // model to explain a change without hand-copying the diff.
  prompt: string
  file: string
  turn: number | null
}

function hunkText(diff: DiffLine[], maxLines = 40): string {
  const lines: string[] = []
  for (const line of diff) {
    if (line.type === 'hunk') lines.push(line.text)
    else if (line.type === 'add') lines.push(`+ ${line.text}`)
    else if (line.type === 'del') lines.push(`- ${line.text}`)
    else lines.push(`  ${line.text}`)
    if (lines.length >= maxLines) { lines.push('… (truncated)'); break }
  }
  return lines.join('\n')
}

// Build the chat reference for a diff hunk. The prompt embeds a bounded excerpt so the model
// sees exactly what's being asked about; empty diffs still produce a sensible file-level ask.
export function buildAskWhyReference(input: { file: string; diff: DiffLine[]; turn?: number | null }): AskWhyReference {
  const file = input.file.trim() || 'this file'
  const fence = '```'
  const excerpt = input.diff.length ? `\n\n${fence}diff\n${hunkText(input.diff)}\n${fence}` : ''
  const turnRef = input.turn != null ? ` (turn ${input.turn})` : ''
  return {
    prompt: `Why did ${file} change${turnRef}? Explain the reasoning behind this edit:${excerpt}`,
    file,
    turn: input.turn ?? null,
  }
}
