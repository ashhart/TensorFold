import { promises as fs } from 'node:fs'
import path from 'node:path'
import { computeDiff } from './diff'
import type { DiffLine } from './types'
import { rel, safeResolve } from './tool-paths'
import { blockedWriteReason } from './command-policy'
import { atomicWriteWorkspaceFile, workspaceContentSha256 } from './atomic-file-write'
import { canonicalWorkspacePath } from './workspace-path-policy'

export const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'dist-electron', '.next', 'build', '.cache', 'vendor', '.venv'])
export const MAX_GLOB_RESULTS = 500
export const DEFAULT_GLOB_RESULTS = 100
const MAX_PATCH_BYTES = 256_000
const MAX_PATCH_FILES = 20
const MAX_REPLACEMENTS = 40

function normalizeSlash(value: string): string {
  return value.replace(/\\/g, '/').replace(/^\.\/+/, '')
}

function assertWritablePath(relPath: string): void {
  const blocked = blockedWriteReason(relPath)
  if (blocked) throw new Error(`Blocked by TensorFold safety policy: ${blocked}.`)
}

// Replace the first occurrence of a literal `needle` with a literal `replacement`.
// String.prototype.replace expands $&, $`, $', $1..$9 and $$ in the replacement ARGUMENT
// even when the pattern is a plain string — so passing model/user-supplied text directly
// silently corrupts any edit whose new text contains those tokens. A function replacer
// disables that interpretation, so the bytes are written verbatim.
export function replaceLiteral(haystack: string, needle: string, replacement: string): string {
  return haystack.replace(needle, () => replacement)
}

function escapeRegExp(value: string): string {
  return value.replace(/[|\\{}()[\]^$+?.]/g, '\\$&')
}

export function validateRelativePattern(pattern: string): string {
  const normalized = normalizeSlash(pattern.trim())
  if (!normalized) throw new Error('Glob pattern is required.')
  if (path.isAbsolute(normalized) || normalized.split('/').includes('..')) {
    throw new Error(`Glob pattern "${pattern}" escapes the workspace.`)
  }
  return normalized
}

function globToRegExp(pattern: string): RegExp {
  let out = '^'
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i]
    if (ch === '*') {
      if (pattern[i + 1] === '*') {
        i++
        if (pattern[i + 1] === '/') {
          i++
          out += '(?:[^/]+/)*'
        } else {
          out += '.*'
        }
      } else {
        out += '[^/]*'
      }
      continue
    }
    if (ch === '?') {
      out += '[^/]'
      continue
    }
    if (ch === '[') {
      const end = pattern.indexOf(']', i + 1)
      if (end > i + 1) {
        const body = pattern.slice(i + 1, end).replace(/\\/g, '\\\\')
        out += `[${body}]`
        i = end
        continue
      }
    }
    out += escapeRegExp(ch)
  }
  out += '$'
  return new RegExp(out)
}

function globMatches(pattern: string, fileRel: string): boolean {
  const normalized = normalizeSlash(fileRel)
  const re = globToRegExp(pattern)
  if (re.test(normalized)) return true
  return !pattern.includes('/') && re.test(path.posix.basename(normalized))
}

function wantsHidden(pattern: string): boolean {
  return pattern.split('/').some((part) => part.startsWith('.') && part !== '.' && part !== '..')
}

export async function collectFiles(cwd: string, base: string, pattern: string, max: number): Promise<string[]> {
  const hits: string[] = []
  const includeHidden = wantsHidden(pattern)
  const walk = async (dir: string, depth: number) => {
    if (depth > 24 || hits.length >= max) return
    let entries: any[] = []
    try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
    const sorted = entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))
    for (const e of sorted) {
      if (hits.length >= max) break
      if ((!includeHidden && e.name.startsWith('.')) || SKIP_DIRS.has(e.name)) continue
      const full = path.join(dir, e.name)
      if (e.isDirectory()) {
        await walk(full, depth + 1)
        continue
      }
      const relativeToBase = normalizeSlash(path.relative(base, full))
      const relativeToRoot = normalizeSlash(rel(cwd, full))
      if (globMatches(pattern, relativeToBase) || globMatches(pattern, relativeToRoot)) hits.push(relativeToRoot)
    }
  }
  await walk(base, 0)
  return [...new Set(hits)].sort((a, b) => a.localeCompare(b)).slice(0, max)
}

type ReplacementHunk = { path?: string; old_string: string; new_string: string }
type PatchHunk = { oldText: string; newText: string; altOldText?: string; altNewText?: string }
type PatchOperation =
  | { kind: 'add'; filePath: string; content: string }
  | { kind: 'delete'; filePath: string }
  | { kind: 'update'; filePath: string; hunks: PatchHunk[]; moveTo?: string }
type AppliedChange = { filePath: string; finalPath: string; before: string; after: string; kind: 'add' | 'delete' | 'update' | 'move' }

function countOccurrences(haystack: string, needle: string): number {
  if (!needle) return 0
  let count = 0
  let idx = 0
  while ((idx = haystack.indexOf(needle, idx)) !== -1) {
    count++
    idx += needle.length
  }
  return count
}

function assertSmallPatch(text: string) {
  if (Buffer.byteLength(text, 'utf8') > MAX_PATCH_BYTES) {
    throw new Error(`Patch is too large. Keep patches under ${MAX_PATCH_BYTES} bytes.`)
  }
}

export function parseApplyPatchText(patch: string): PatchOperation[] {
  assertSmallPatch(patch)
  const lines = patch.trimEnd().split(/\r?\n/)
  let i = 0
  if (lines[i] === '*** Begin Patch') i++
  const ops: PatchOperation[] = []

  const atBoundary = () => i >= lines.length || lines[i].startsWith('*** ')

  while (i < lines.length) {
    const line = lines[i]
    if (line === '*** End Patch') break

    if (line.startsWith('*** Add File: ')) {
      const filePath = line.slice('*** Add File: '.length).trim()
      i++
      const content: string[] = []
      while (!atBoundary()) {
        if (!lines[i].startsWith('+')) throw new Error(`Invalid add-file line in ${filePath}; expected lines to start with "+".`)
        content.push(lines[i].slice(1))
        i++
      }
      ops.push({ kind: 'add', filePath, content: content.length ? `${content.join('\n')}\n` : '' })
      continue
    }

    if (line.startsWith('*** Delete File: ')) {
      ops.push({ kind: 'delete', filePath: line.slice('*** Delete File: '.length).trim() })
      i++
      continue
    }

    if (line.startsWith('*** Update File: ')) {
      const filePath = line.slice('*** Update File: '.length).trim()
      i++
      let moveTo: string | undefined
      if (lines[i]?.startsWith('*** Move to: ')) {
        moveTo = lines[i].slice('*** Move to: '.length).trim()
        i++
      }
      const hunks: PatchHunk[] = []
      while (i < lines.length && !lines[i].startsWith('*** ')) {
        if (!lines[i].startsWith('@@')) throw new Error(`Invalid update hunk in ${filePath}; expected "@@".`)
        i++
        const oldLines: string[] = []
        const newLines: string[] = []
        while (i < lines.length && !lines[i].startsWith('@@') && !lines[i].startsWith('*** ')) {
          const hunkLine = lines[i]
          if (hunkLine.startsWith('\\')) {
            i++
            continue
          }
          if (hunkLine.startsWith(' ')) {
            oldLines.push(hunkLine.slice(1))
            newLines.push(hunkLine.slice(1))
          } else if (hunkLine.startsWith('-')) {
            oldLines.push(hunkLine.slice(1))
          } else if (hunkLine.startsWith('+')) {
            newLines.push(hunkLine.slice(1))
          } else {
            throw new Error(`Invalid patch line in ${filePath}; expected " ", "-", or "+".`)
          }
          i++
        }
        const oldText = oldLines.length ? `${oldLines.join('\n')}\n` : ''
        if (!oldText) throw new Error(`Update hunk in ${filePath} must include context or removed lines.`)
        hunks.push({
          oldText,
          newText: newLines.length ? `${newLines.join('\n')}\n` : '',
          altOldText: oldLines.join('\n'),
          altNewText: newLines.join('\n'),
        })
      }
      if (!hunks.length && !moveTo) throw new Error(`Update file ${filePath} has no hunks.`)
      ops.push({ kind: 'update', filePath, hunks, moveTo })
      continue
    }

    if (!line.trim()) {
      i++
      continue
    }
    throw new Error(`Unsupported patch line: ${line}`)
  }

  if (!ops.length) throw new Error('Patch contained no file operations.')
  if (ops.length > MAX_PATCH_FILES) throw new Error(`Patch touches too many files. Limit is ${MAX_PATCH_FILES}.`)
  return ops
}

export function replacementHunksFromArgs(args: Record<string, any>): ReplacementHunk[] {
  if (Array.isArray(args.replacements)) {
    if (args.replacements.length > MAX_REPLACEMENTS) throw new Error(`Too many replacements. Limit is ${MAX_REPLACEMENTS}.`)
    return args.replacements.map((r: any) => ({
      path: typeof r?.path === 'string' ? r.path : args.path,
      old_string: String(r?.old_string ?? ''),
      new_string: String(r?.new_string ?? ''),
    }))
  }
  if (typeof args.old_string === 'string' && typeof args.new_string === 'string') {
    return [{ path: args.path, old_string: args.old_string, new_string: args.new_string }]
  }
  return []
}

export async function stageReplacementHunks(cwd: string, hunks: ReplacementHunk[]): Promise<AppliedChange[]> {
  if (!hunks.length) throw new Error('Provide either patch text or search/replace hunks.')
  const grouped = new Map<string, ReplacementHunk[]>()
  for (const hunk of hunks) {
    if (!hunk.path) throw new Error('Each replacement needs a path.')
    if (!hunk.old_string) throw new Error(`Replacement for ${hunk.path} has an empty old_string.`)
    const abs = safeResolve(cwd, hunk.path)
    await canonicalWorkspacePath(cwd, abs, 'write')
    const key = rel(cwd, abs)
    assertWritablePath(key)
    grouped.set(key, [...(grouped.get(key) ?? []), hunk])
  }
  if (grouped.size > MAX_PATCH_FILES) throw new Error(`Patch touches too many files. Limit is ${MAX_PATCH_FILES}.`)

  const changes: AppliedChange[] = []
  for (const [filePath, fileHunks] of grouped) {
    const abs = safeResolve(cwd, filePath)
    const before = await fs.readFile(abs, 'utf8')
    let after = before
    for (const hunk of fileHunks) {
      const count = countOccurrences(after, hunk.old_string)
      if (count === 0) throw new Error(`String not found in ${filePath}.`)
      if (count > 1) throw new Error(`String is not unique (${count} matches) in ${filePath}.`)
      after = replaceLiteral(after, hunk.old_string, hunk.new_string)
    }
    changes.push({ filePath, finalPath: filePath, before, after, kind: 'update' })
  }
  return changes
}

export async function stagePatchOperations(cwd: string, ops: PatchOperation[]): Promise<AppliedChange[]> {
  const seenSources = new Set<string>()
  const seenTargets = new Set<string>()
  const changes: AppliedChange[] = []

  for (const op of ops) {
    const file = safeResolve(cwd, op.filePath)
    await canonicalWorkspacePath(cwd, file, 'write')
    const filePath = rel(cwd, file)
    assertWritablePath(filePath)
    if (seenSources.has(filePath)) throw new Error(`Patch repeats file ${filePath}; combine hunks under one operation.`)
    seenSources.add(filePath)

    if (op.kind === 'add') {
      try {
        // lstat (not access) so an existing SYMLINK at the target counts as "exists" and is
        // rejected, rather than access() following a dangling link and reporting ENOENT —
        // which would then add a file THROUGH the link to its out-of-workspace destination.
        await fs.lstat(file)
        throw new Error(`Cannot add ${filePath}; file already exists.`)
      } catch (e: any) {
        if (e?.code !== 'ENOENT') throw e
      }
      if (seenTargets.has(filePath)) throw new Error(`Patch writes ${filePath} more than once.`)
      seenTargets.add(filePath)
      changes.push({ filePath, finalPath: filePath, before: '', after: op.content, kind: 'add' })
      continue
    }

    if (op.kind === 'delete') {
      const before = await fs.readFile(file, 'utf8')
      if (seenTargets.has(filePath)) throw new Error(`Patch writes ${filePath} more than once.`)
      seenTargets.add(filePath)
      changes.push({ filePath, finalPath: filePath, before, after: '', kind: 'delete' })
      continue
    }

    const before = await fs.readFile(file, 'utf8')
    let after = before
    // Hunks are built with LF line endings; a CRLF file would never match. Convert hunk text to
    // CRLF for CRLF files so multi-line hunks match and the file's line endings are preserved.
    const usesCrlf = /\r\n/.test(before)
    const toFileEol = (s: string) => (usesCrlf ? s.replace(/\n/g, '\r\n') : s)
    for (const hunk of op.hunks) {
      let oldText = toFileEol(hunk.oldText)
      let newText = toFileEol(hunk.newText)
      let count = countOccurrences(after, oldText)
      if (count === 0 && hunk.altOldText) {
        const alt = toFileEol(hunk.altOldText)
        const altCount = countOccurrences(after, alt)
        if (altCount > 0) {
          oldText = alt
          newText = toFileEol(hunk.altNewText ?? '')
          count = altCount
        }
      }
      if (count === 0) throw new Error(`Patch hunk not found in ${filePath}.`)
      if (count > 1) throw new Error(`Patch hunk is not unique (${count} matches) in ${filePath}.`)
      after = replaceLiteral(after, oldText, newText)
    }

    const final = op.moveTo ? safeResolve(cwd, op.moveTo) : file
    await canonicalWorkspacePath(cwd, final, 'write')
    const finalPath = rel(cwd, final)
    assertWritablePath(finalPath)
    if (op.moveTo) {
      try {
        // lstat treats dangling symlinks as existing. A move is create-only at its
        // destination and must not silently replace any concurrently-owned entry.
        await fs.lstat(final)
        throw new Error(`Cannot move ${filePath} to ${finalPath}; target already exists.`)
      } catch (e: any) {
        if (e?.code !== 'ENOENT') throw e
      }
    }
    if (seenTargets.has(finalPath)) throw new Error(`Patch writes ${finalPath} more than once.`)
    seenTargets.add(finalPath)
    changes.push({ filePath, finalPath, before, after, kind: op.moveTo ? 'move' : 'update' })
  }

  return changes
}

export function combinedDiff(changes: AppliedChange[]): DiffLine[] {
  const diff: DiffLine[] = []
  for (const change of changes) {
    if (diff.length) diff.push({ type: 'hunk', text: '' })
    const title = change.filePath === change.finalPath ? change.finalPath : `${change.filePath} -> ${change.finalPath}`
    diff.push({ type: 'hunk', text: `*** ${title}` })
    diff.push(...computeDiff(change.before, change.after))
  }
  return diff
}

async function assertStagedContentUnchanged(file: string, staged: string): Promise<void> {
  let current: Buffer
  try { current = await fs.readFile(file) } catch {
    throw new Error(`Target "${file}" changed after the patch was staged; refusing to overwrite concurrent work.`)
  }
  if (workspaceContentSha256(current) !== workspaceContentSha256(staged)) {
    throw new Error(`Target "${file}" changed after the patch was staged; refusing to overwrite concurrent work.`)
  }
}

async function unlinkIfContentMatches(file: string, expectedSha256: string): Promise<void> {
  const current = await fs.readFile(file)
  if (workspaceContentSha256(current) !== expectedSha256) {
    throw new Error(`Target "${file}" changed during patch rollback; preserving concurrent work.`)
  }
  await fs.unlink(file)
}

export async function writeAppliedChanges(cwd: string, changes: AppliedChange[]) {
  // Snapshot every path the patch will touch, then apply; if any write fails mid-batch, restore the
  // snapshots so a multi-file patch is all-or-nothing rather than left half-applied.
  const touched = new Set<string>()
  for (const change of changes) {
    assertWritablePath(change.filePath)
    assertWritablePath(change.finalPath)
    touched.add(safeResolve(cwd, change.filePath))
    touched.add(safeResolve(cwd, change.finalPath))
  }
  const snapshots = new Map<string, Buffer | null>()
  for (const p of touched) {
    await canonicalWorkspacePath(cwd, p, 'write')
    snapshots.set(p, await fs.readFile(p).catch(() => null))
  }
  // Expected post-mutation digest (`null` means this call removed the path). Used
  // as an optimistic rollback preflight so ordinary concurrent edits are preserved.
  // The portable local-filesystem limitation is documented in security-operations.md.
  const mutated = new Map<string, string | null>()

  try {
    for (const change of changes) {
      const from = safeResolve(cwd, change.filePath)
      const to = safeResolve(cwd, change.finalPath)
      await canonicalWorkspacePath(cwd, from, 'write')
      await canonicalWorkspacePath(cwd, to, 'write')
      if (change.kind === 'delete') {
        await assertStagedContentUnchanged(from, change.before)
        await fs.unlink(from)
        mutated.set(from, null)
        continue
      }
      if (change.kind === 'move') {
        await assertStagedContentUnchanged(from, change.before)
        await atomicWriteWorkspaceFile(cwd, to, change.after, { ifExists: 'error' })
        mutated.set(to, workspaceContentSha256(change.after))
        await assertStagedContentUnchanged(from, change.before)
        await fs.unlink(from)
        mutated.set(from, null)
        continue
      }
      await atomicWriteWorkspaceFile(cwd, to, change.after, change.kind === 'add'
        ? { ifExists: 'error' }
        : { expectedSha256: workspaceContentSha256(change.before) })
      mutated.set(to, workspaceContentSha256(change.after))
    }
  } catch (e) {
    // Best-effort rollback only paths this call actually mutated. Rolling back every
    // staged path would overwrite a concurrent file that caused a create-only add/move
    // to fail before TensorFold touched it.
    for (const [p, expected] of [...mutated].reverse()) {
      const prev = snapshots.get(p) ?? null
      if (prev == null) {
        if (expected) await unlinkIfContentMatches(p, expected).catch(() => {})
      } else if (expected == null) {
        await atomicWriteWorkspaceFile(cwd, p, prev, { ifExists: 'error' }).catch(() => {})
      } else {
        await atomicWriteWorkspaceFile(cwd, p, prev, { expectedSha256: expected }).catch(() => {})
      }
    }
    throw e
  }
}

export function patchSummary(changes: AppliedChange[]): string {
  return changes.map((change) => {
    if (change.kind === 'add') return `Created ${change.finalPath}`
    if (change.kind === 'delete') return `Deleted ${change.filePath}`
    if (change.kind === 'move') return `Moved ${change.filePath} -> ${change.finalPath}`
    return `Edited ${change.finalPath}`
  }).join('\n')
}
