// Run: jiti src/lib/diff-words.test.ts
//
// Word-level / intra-line diff for the renderer's DiffView (VST-045). The main
// process emits a flat DiffLine[] (line-level only); this helper runs jsdiff's
// diffWords on an adjacent del+add pair to highlight *what* changed within the
// line, not just which lines changed. Pure + deterministic; no IPC, no
// persistence — computed on render.
import { diffWordsWithSpace } from 'diff'
import type { DiffLine } from '../lib/types'

export interface WordSpan {
  text: string
  /** true when this span is part of the word-level change (added or removed). */
  changed: boolean
}

export interface LineWordDiff {
  /** removed-side spans (for the del line); empty if unchanged. */
  removed: WordSpan[]
  /** added-side spans (for the add line); empty if unchanged. */
  added: WordSpan[]
}

/**
 * Compute word-level spans for one adjacent del+add line pair. Returns null if
 * the two lines are identical (no intra-line change to show) so the caller can
 * fall back to plain rendering. Uses diffWordsWithSpace so leading/trailing
 * whitespace differences are treated as their own tokens (preserves indentation
 * changes as visible spans rather than merging them into adjacent words).
 */
export function diffWordsInLine(oldLine: string, newLine: string): LineWordDiff | null {
  if (oldLine === newLine) return null
  // Guard against pathological inputs: very long lines produce huge part arrays
  // and the value here is readability of normal source edits, not whole-file
  // regurgitation. Above the cap, fall back to line-level rendering (null).
  if (oldLine.length > 4000 || newLine.length > 4000) return null

  const parts = diffWordsWithSpace(oldLine, newLine)
  const removed: WordSpan[] = []
  const added: WordSpan[] = []
  for (const part of parts) {
    if (!part.value) continue
    if (part.added) added.push({ text: part.value, changed: true })
    else if (part.removed) removed.push({ text: part.value, changed: true })
    else {
      // A shared (unchanged) span belongs to both sides as context.
      removed.push({ text: part.value, changed: false })
      added.push({ text: part.value, changed: false })
    }
  }
  // If diffWords produced no changed spans (e.g. only a trailing-space diff got
  // collapsed), treat as unchanged so the caller renders plain.
  const hasChange = removed.some((s) => s.changed) || added.some((s) => s.changed)
  if (!hasChange) return null
  return { removed, added }
}

/**
 * Walk a DiffLine[] and pair adjacent del+add lines so the renderer can
 * highlight intra-line word changes. Returns a map keyed by the del/add line
 * index (in the input array) to its word-level spans. Lines not in the map get
 * plain rendering. A del followed by an add is paired; lone dels/adds (pure
 * insertions or deletions) are skipped (no pairing partner to diff against).
 */
export function pairWordDiffs(diff: DiffLine[]): Map<number, LineWordDiff> {
  const out = new Map<number, LineWordDiff>()
  for (let i = 0; i < diff.length; i++) {
    const cur = diff[i]
    const next = diff[i + 1]
    if (cur.type === 'del' && next?.type === 'add') {
      const wd = diffWordsInLine(cur.text, next.text)
      if (wd) {
        out.set(i, wd)
        out.set(i + 1, wd)
      }
    }
  }
  return out
}
