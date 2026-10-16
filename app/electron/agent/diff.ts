import { diffLines } from 'diff'
import type { DiffLine } from './types'

// Line-based diff using jsdiff's Myers implementation (O(ND), linear-space
// backtrack — no O(n*m) DP table). Produces GitHub-style unified hunks for
// display, preserving the DiffLine[] contract every consumer (tools, diff-view,
// inline-review, renderer) depends on: { type, text, oldNo?, newNo? } with
// 1-based line numbers and `hunk` rows carrying `@@ … @@` headers.
//
// jsdiff's diffLines splits on `\n` and each part's `value` keeps its trailing
// newline (except possibly the last). We normalize so the last line is not
// dropped whether or not the source ended with `\n`.

// Threshold above which the `context`-collapse walk is skipped in favor of a
// straight prefix/suffix trim. The collapse is O(parts) so this is about
// avoiding needless work on very large changes, not a memory guard (Myers is
// already linear-space). Well below any pathological threshold.
const MAX_COLLAPSE_PARTS = 200_000

export function computeDiff(oldText: string, newText: string, context = 3): DiffLine[] {
  // Normalize so both inputs end with a trailing newline when non-empty.
  // jsdiff's diffLines tokenizes a line as "text\n", so a final line with no
  // trailing newline is a distinct token from the same text with one. That
  // mismatch turns a pure append ("a\nb" -> "a\nb\nc") into a spurious
  // removed+readded churn on the unchanged last line. Ensuring both sides end
  // with `\n` makes the tokenizer see consistent complete-line tokens.
  const a = oldText.length && !oldText.endsWith('\n') ? oldText + '\n' : oldText
  const b = newText.length && !newText.endsWith('\n') ? newText + '\n' : newText
  const parts = diffLines(a, b)
  if (parts.length === 0) return []

  // Expand into typed ops with 1-based line numbers. jsdiff emits consecutive
  // removed/added parts independently; map each to del/add per source line.
  type Op = { t: 'eq' | 'del' | 'add'; text: string; oldNo: number; newNo: number }
  const ops: Op[] = []
  let oldNo = 1
  let newNo = 1
  for (const part of parts) {
    // Split the part's value into display lines. A trailing `\n` produces an
    // empty final element which we drop (it's the line-break, not a line).
    const raw = part.value
    const lines = raw.length ? raw.replace(/\n$/, '').split('\n') : []
    for (const text of lines) {
      if (part.added) {
        ops.push({ t: 'add', text, oldNo: 0, newNo: newNo++ })
      } else if (part.removed) {
        ops.push({ t: 'del', text, oldNo: oldNo++, newNo: 0 })
      } else {
        ops.push({ t: 'eq', text, oldNo: oldNo++, newNo: newNo++ })
      }
    }
    // A removed+added pair that the engine kept separate still consumes lines on
    // each side independently (handled above). An empty-valued part contributes
    // nothing, which is correct.
  }

  // Collapse unchanged regions to `context` lines around changes, emitting
  // `@@ -oldStart,… +newStart,… @@` hunk headers between groups. Same shape as
  // the prior LCS impl's collapse so display output is stable.
  if (ops.length > MAX_COLLAPSE_PARTS) return collapseCoarse(ops, context)

  const keep = new Array<boolean>(ops.length).fill(false)
  for (let i = 0; i < ops.length; i++) {
    if (ops[i].t !== 'eq') {
      for (let k = Math.max(0, i - context); k <= Math.min(ops.length - 1, i + context); k++) keep[k] = true
    }
  }

  const lines: DiffLine[] = []
  let lastKept = -2
  for (let i = 0; i < ops.length; i++) {
    if (!keep[i]) continue
    if (i > lastKept + 1) {
      const op = ops[i]
      const oldStart = op.t !== 'add' ? op.oldNo : (op.newNo)
      const newStart = op.t !== 'del' ? op.newNo : (op.oldNo)
      lines.push({ type: 'hunk', text: `@@ -${oldStart},0 +${newStart},0 @@` })
    }
    const op = ops[i]
    lines.push({
      type: op.t === 'eq' ? 'ctx' : op.t === 'del' ? 'del' : 'add',
      text: op.text,
      oldNo: op.t !== 'add' ? op.oldNo : undefined,
      newNo: op.t !== 'del' ? op.newNo : undefined,
    })
    lastKept = i
  }
  return lines
}

// Linear-time fallback for pathological inputs: trims common prefix/suffix,
// keeps `context` lines of each, renders the differing middle as deletions
// followed by additions. Same semantics as the prior coarseDiff so behavior on
// very large changes is unchanged.
function collapseCoarse(ops: Array<{ t: 'eq' | 'del' | 'add'; text: string; oldNo: number; newNo: number }>, context: number): DiffLine[] {
  const n = ops.length
  let pre = 0
  while (pre < n && ops[pre].t === 'eq') pre++
  let suf = 0
  while (suf < n - pre && ops[n - 1 - suf].t === 'eq') suf++

  const lines: DiffLine[] = []
  const ctxStart = Math.max(0, pre - context)
  const firstOp = ops[ctxStart]
  lines.push({ type: 'hunk', text: `@@ -${firstOp.oldNo || firstOp.newNo},0 +${firstOp.newNo || firstOp.oldNo},0 @@` })
  for (let k = ctxStart; k < pre; k++) lines.push({ type: 'ctx', text: ops[k].text, oldNo: ops[k].oldNo, newNo: ops[k].newNo })
  for (let k = pre; k < n - suf; k++) {
    const op = ops[k]
    if (op.t === 'del') lines.push({ type: 'del', text: op.text, oldNo: op.oldNo })
    else if (op.t === 'add') lines.push({ type: 'add', text: op.text, newNo: op.newNo })
  }
  for (let k = 0; k < Math.min(context, suf); k++) {
    const op = ops[n - suf + k]
    lines.push({ type: 'ctx', text: op.text, oldNo: op.oldNo, newNo: op.newNo })
  }
  return lines
}

export function diffStat(diff: DiffLine[]): { added: number; removed: number } {
  let added = 0
  let removed = 0
  for (const l of diff) {
    if (l.type === 'add') added++
    else if (l.type === 'del') removed++
  }
  return { added, removed }
}
