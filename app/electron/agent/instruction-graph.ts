// Instruction Graph (build.md P0).
//
// Resolves `@import` lines inside project instruction files (AGENTS.md, provider
// overlays, imported docs) into a single auditable graph: shared truth + overlays,
// recursive imports with cycle detection, ignored local overrides, token estimates,
// and the exact resolvedContent the model will receive.
//
// Safety: every imported path goes through blockedReadReason (credential denylist)
// THEN safeResolve (workspace jail incl. symlink realpath check) before any read,
// so `@../../.ssh/id_rsa`, `@/etc/passwd`, `@.env`, and symlink escapes never read.
import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { safeResolve } from './tools'
import { blockedReadReason } from './command-policy'
import { estimateTokens } from './compaction'
import type { InstructionNode, InstructionNodeKind, ResolvedInstructionGraph } from './types'

// Preserve loop.ts root-file precedence.
const ROOT_CANDIDATES = ['AGENTS.md', 'CLAUDE.md', '.vontra/AGENTS.md']
const IGNORE_FILE = '.vontra/instructions-ignore' // newline list of relpaths to mark ignored
// Safety caps — terminate fan-out and guard against token-bombs.
const MAX_NODES = 64
const MAX_TOTAL_BYTES = 256 * 1024
const PER_NODE_CAP = 16 * 1024
const MAX_IMPORTS_PER_FILE = 32 // cap @import lines parsed from one file (fan-out guard)
const MODEL_CHAR_BUDGET = 8000 // resolvedContent budget — preview shows EXACTLY what the model receives
// Instruction imports are documentation by intent — only doc/text files may ever be
// inlined. This is the primary secret-ingestion guard: config/credential carriers
// (.npmrc, *.pem, *.tfvars, .git/config, serviceAccount.json, …) are not doc files,
// so they can never be pulled into the system prompt even though they live in-workspace.
const DOC_EXTS = new Set(['.md', '.markdown', '.mdx', '.txt', '.text'])
// A line that is exactly `@<relpath>`, optionally followed by a `# comment`.
const IMPORT_RE = /^\s*@([^\s#]+)\s*(?:#.*)?$/

const tokensFor = (content: string): number => estimateTokens([{ role: 'system', content }])
const norm = (rel: string): string => path.normalize(rel).replace(/^\.\//, '').replace(/\\/g, '/')

async function readInstructionPrefix(abs: string, maxBytes = PER_NODE_CAP): Promise<string> {
  const handle = await fs.open(abs, constants.O_RDONLY | constants.O_NOFOLLOW)
  try {
    const info = await handle.stat()
    if (!info.isFile()) throw new Error('not a regular file')
    const buffer = Buffer.allocUnsafe(Math.min(maxBytes, Math.max(1, info.size)))
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)
    return buffer.subarray(0, bytesRead).toString('utf8')
  } finally {
    await handle.close()
  }
}

interface WorkNode extends InstructionNode {
  blocked: boolean // placeholder for a denied/escaped/missing import — kept for the preview, excluded from model content
}

interface Ctx {
  cwd: string
  nodes: WorkNode[]
  byKey: Map<string, WorkNode> // dedup by resolved abs path (or sentinel for placeholders)
  onStack: Set<string> // abs paths on the current DFS path → cycle detection
  cycles: string[]
  totalBytes: number
  ignore: Set<string>
}

function placeholder(ctx: Ctx, key: string, relSpec: string, kind: InstructionNodeKind, content: string): void {
  if (ctx.byKey.has(key)) return
  if (ctx.nodes.length >= MAX_NODES) return // honor the node cap for placeholders too (else a hostile repo balloons the graph)
  const node: WorkNode = { id: key, kind, path: relSpec, content, imports: [], ignored: false, tokenEstimate: 0, blocked: true }
  ctx.byKey.set(key, node)
  ctx.nodes.push(node)
}

// Recursive resolve of one @import spec. Never throws; records a placeholder node
// for blocked / escaped / missing imports so the preview shows exactly what happened.
async function resolveNode(ctx: Ctx, rawSpec: string, kind: InstructionNodeKind): Promise<void> {
  const relSpec = norm(rawSpec.replace(/^@/, ''))
  if (!relSpec) return

  // 1. Credential denylist FIRST — before any path resolution or read.
  const denied = blockedReadReason(relSpec)
  if (denied) { placeholder(ctx, `blk:${relSpec}`, relSpec, kind, `Blocked by TensorFold safety policy: ${denied}.`); return }

  // 2. Jail to the workspace (throws on ../ , absolute, or symlink escape).
  let abs: string
  try {
    abs = safeResolve(ctx.cwd, relSpec)
  } catch (err: any) {
    placeholder(ctx, `esc:${relSpec}`, relSpec, kind, `[skipped: ${err?.message || 'escapes the workspace'}]`)
    return
  }

  // 2b. Documentation-only: refuse to inline anything that isn't a doc/text file.
  //     This is the real secret-ingestion guard — config/credential carriers that the
  //     narrow denylist misses (.npmrc, *.pem, *.tfvars, .git/config, *.json secrets…)
  //     are simply not importable, even though they sit inside the workspace.
  if (!DOC_EXTS.has(path.extname(relSpec).toLowerCase())) {
    placeholder(ctx, `ext:${relSpec}`, relSpec, kind, '[skipped: only .md / .markdown / .txt instruction files can be imported]')
    return
  }

  // 3. Cycle FIRST — a node still on the DFS stack is a back-edge. (Must precede the
  //    dedup check: a cyclic target is also already in byKey, so checking dedup first
  //    would silently absorb the cycle instead of recording it.)
  if (ctx.onStack.has(abs)) { ctx.cycles.push(`${relSpec} (cycle)`); return }
  // 4. Dedup (diamond imports) — already fully-resolved file, referenced again, not re-read.
  if (ctx.byKey.has(abs)) return
  // 5. Caps — bound node count and total bytes.
  if (ctx.nodes.length >= MAX_NODES || ctx.totalBytes >= MAX_TOTAL_BYTES) {
    placeholder(ctx, `cap:${relSpec}`, relSpec, kind, '[skipped: instruction graph size cap reached]')
    return
  }

  // 6. Read (skip symlinks defensively; tolerate missing/unreadable files).
  let raw: string
  try {
    const st = await fs.lstat(abs)
    if (st.isSymbolicLink() || !st.isFile()) { placeholder(ctx, abs, relSpec, kind, '[skipped: not a regular file]'); return }
    raw = await readInstructionPrefix(abs)
  } catch {
    placeholder(ctx, abs, relSpec, kind, `[skipped: ${relSpec} not found]`)
    return
  }
  ctx.totalBytes += raw.length

  // 7. Parse @import lines; strip them from the node's own content (children inline as their own nodes).
  const imports: string[] = []
  const bodyLines: string[] = []
  for (const line of raw.split('\n')) {
    const m = IMPORT_RE.exec(line)
    if (m) { if (imports.length < MAX_IMPORTS_PER_FILE) imports.push(m[1]) } // strip every @import line from the body; only resolve up to the per-file cap
    else bodyLines.push(line)
  }
  const content = bodyLines.join('\n').trim()

  const node: WorkNode = {
    id: abs,
    kind,
    path: relSpec,
    content,
    imports,
    ignored: ctx.ignore.has(relSpec),
    tokenEstimate: tokensFor(raw),
    blocked: false,
  }
  ctx.byKey.set(abs, node)
  ctx.nodes.push(node)

  // 8. Recurse into imports (children are always 'imported' regardless of parent kind).
  ctx.onStack.add(abs)
  for (const imp of imports) await resolveNode(ctx, imp, 'imported')
  ctx.onStack.delete(abs)
}

async function loadIgnoreSet(cwd: string): Promise<Set<string>> {
  try {
    const abs = safeResolve(cwd, IGNORE_FILE)
    const raw = await readInstructionPrefix(abs, 64 * 1024)
    return new Set(raw.split('\n').map((l) => norm(l.trim())).filter((l) => l && !l.startsWith('#')))
  } catch {
    return new Set()
  }
}

async function firstRootCandidate(cwd: string): Promise<string | null> {
  for (const cand of ROOT_CANDIDATES) {
    try {
      const abs = safeResolve(cwd, cand)
      const st = await fs.lstat(abs)
      if (st.isSymbolicLink() || !st.isFile()) continue
      const raw = await readInstructionPrefix(abs)
      if (raw.trim()) return cand
    } catch { /* try next */ }
  }
  return null
}

async function providerOverlay(cwd: string, providerId: string | null | undefined): Promise<string | null> {
  if (!providerId) return null
  const safeId = providerId.replace(/[^\w.-]/g, '')
  if (!safeId) return null
  for (const cand of [`.vontra/${safeId}.md`, `.vontra/AGENTS.${safeId}.md`]) {
    try {
      const abs = safeResolve(cwd, cand)
      const st = await fs.lstat(abs)
      if (st.isFile() && !st.isSymbolicLink() && (await readInstructionPrefix(abs)).trim()) return cand
    } catch { /* try next */ }
  }
  return null
}

// --- Nested / hierarchical instruction files (Codex/Cursor/Amp parity) ---------
// Beyond the single root file, walk the workspace for directory-scoped AGENTS.md / CLAUDE.md
// (one per subdirectory, AGENTS.md preferred). They are resolved shallow→deep so the most
// specific guidance is read LAST — a single-blob approximation of "nearest-file-wins" (recency
// gives the deeper, more-specific file precedence in the model's eyes). All the same safety as
// imports applies (each goes through resolveNode: jail, doc-only, symlink skip, caps).
const MAX_NESTED_DEPTH = 6
const MAX_NESTED_DIRS = 48 // total directories scanned — bounds fan-out on a huge monorepo
const NESTED_SKIP = new Set(['node_modules', '.git', 'dist', 'dist-electron', '.next', 'build', '.cache', 'vendor', '.venv', 'release', 'coverage', 'out', 'tmp'])
const NESTED_NAMES = ['AGENTS.md', 'CLAUDE.md'] // AGENTS.md wins when both are present in a dir

// Discover directory-scoped instruction files BELOW the root (depth >= 1). Returns workspace-
// relative paths sorted shallow→deep then alphabetically. Never throws; skips symlinks and
// ignored paths; bounded by depth and total directories scanned.
async function discoverNestedInstructionFiles(cwd: string, ignore: Set<string>): Promise<string[]> {
  const found: string[] = []
  let dirsVisited = 0
  async function walk(absDir: string, relDir: string, depth: number): Promise<void> {
    if (depth > MAX_NESTED_DEPTH || dirsVisited >= MAX_NESTED_DIRS) return
    let entries: Array<{ name: string; isFile(): boolean; isDirectory(): boolean }>
    try { entries = await fs.readdir(absDir, { withFileTypes: true }) as any } catch { return }
    dirsVisited++
    if (depth >= 1) {
      for (const name of NESTED_NAMES) {
        const rel = norm(path.join(relDir, name))
        if (ignore.has(rel)) continue
        if (entries.some((e) => e.isFile() && e.name === name)) { found.push(rel); break }
      }
    }
    for (const e of entries) {
      // isDirectory() is false for a symlink entry (reported as a symlink), so symlinked dirs
      // are never descended into — the same defensive stance the read path takes.
      if (!e.isDirectory() || e.name.startsWith('.') || NESTED_SKIP.has(e.name)) continue
      await walk(path.join(absDir, e.name), path.join(relDir, e.name), depth + 1)
    }
  }
  try { await walk(cwd, '', 0) } catch { /* best effort */ }
  found.sort((a, b) => (a.split('/').length - b.split('/').length) || a.localeCompare(b))
  return found
}

// Build the full resolved instruction graph for a workspace.
export async function buildInstructionGraph(cwd: string, providerId?: string | null): Promise<ResolvedInstructionGraph> {
  const empty: ResolvedInstructionGraph = { rootPath: cwd || '', providerId: providerId ?? null, nodes: [], cycles: [], resolvedContent: '', tokenEstimate: 0 }
  if (!cwd) return empty

  const ctx: Ctx = { cwd, nodes: [], byKey: new Map(), onStack: new Set(), cycles: [], totalBytes: 0, ignore: await loadIgnoreSet(cwd) }

  const root = await firstRootCandidate(cwd)
  if (root) await resolveNode(ctx, root, 'shared')
  const overlay = await providerOverlay(cwd, providerId)
  if (overlay) await resolveNode(ctx, overlay, 'provider')
  // Directory-scoped instruction files below the root, shallow→deep (deeper = read last = wins).
  const nested = await discoverNestedInstructionFiles(cwd, ctx.ignore)
  const nestedSet = new Set(nested)
  for (const rel of nested) await resolveNode(ctx, rel, 'imported')

  // Assemble what the model receives: real (non-blocked, non-ignored) nodes, in DFS order,
  // each prefixed with a provenance header. Shared root first, its imports inline-after, overlay,
  // then the directory-scoped nested files (labelled with the directory they apply to).
  let resolvedContent = ctx.nodes
    .filter((n) => !n.blocked && !n.ignored && n.content)
    .map((n) => {
      if (nestedSet.has(n.path)) {
        const dir = n.path.includes('/') ? n.path.slice(0, n.path.lastIndexOf('/')) : '.'
        return `<!-- scoped instructions for ${dir}/ (${n.path}) -->\n${n.content}`
      }
      return `<!-- ${n.kind}: ${n.path} -->\n${n.content}`
    })
    .join('\n\n')
  // Cap at the model budget so resolvedContent IS exactly what the loop sends (preview == model input).
  if (resolvedContent.length > MODEL_CHAR_BUDGET) resolvedContent = resolvedContent.slice(0, MODEL_CHAR_BUDGET) + '\n…(truncated)'

  const nodes: InstructionNode[] = ctx.nodes.map(({ blocked: _blocked, ...n }) => n)
  return { rootPath: cwd, providerId: providerId ?? null, nodes, cycles: ctx.cycles, resolvedContent, tokenEstimate: tokensFor(resolvedContent) }
}

// Resolve a single @import spec (with recursion) to inlined text. Never throws;
// returns a visible placeholder for blocked/escaped/missing so callers/tests can
// assert no secret content leaked. (Unit-test entrypoint.)
export async function resolveImports(spec: string, opts: { cwd: string }): Promise<string> {
  const ctx: Ctx = { cwd: opts.cwd, nodes: [], byKey: new Map(), onStack: new Set(), cycles: [], totalBytes: 0, ignore: new Set() }
  await resolveNode(ctx, spec, 'imported')
  return ctx.nodes.map((n) => n.content).join('\n\n')
}

// Drop-in replacement for loop.ts readProjectInstructions — same { file, text } | null shape.
export async function resolveInstructionGraph(cwd: string, providerId?: string | null): Promise<{ file: string; text: string } | null> {
  const g = await buildInstructionGraph(cwd, providerId)
  if (!g.resolvedContent.trim()) return null
  const importCount = g.nodes.filter((n) => n.kind === 'imported').length
  const rootLabel = g.nodes[0]?.path ?? 'AGENTS.md'
  const file = importCount ? `instruction graph (${rootLabel} + ${importCount} import${importCount > 1 ? 's' : ''})` : rootLabel
  return { file, text: g.resolvedContent } // already capped to MODEL_CHAR_BUDGET — preview shows the same string
}
