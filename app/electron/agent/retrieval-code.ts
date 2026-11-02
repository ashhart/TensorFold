import { existsSync, realpathSync, type Dirent } from 'node:fs'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { RetrievalCodeFileInput, RetrievalCodeScan, RetrievalCodeScanLimit } from './retrieval-types'
import { normalizeSlash } from './retrieval-records'

const CODE_EXTENSIONS = new Set([
  '.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs',
  '.py', '.rb', '.go', '.rs', '.java', '.kt', '.swift',
  '.c', '.cc', '.cpp', '.h', '.hpp', '.cs', '.php',
  '.zig', '.zon', '.metal', '.cu', '.cuh',
  '.sh', '.zsh', '.bash', '.sql', '.graphql',
])
const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'dist-electron', '.next', 'build', '.cache', 'vendor', '.venv'])
const BLOCKED_NAMES = new Set(['.env', '.npmrc', '.netrc', 'id_rsa', 'id_ed25519'])
const BLOCKED_EXTENSIONS = new Set(['.pem', '.key', '.p12', '.pfx', '.crt', '.cer', '.sqlite', '.db'])
const MAX_CODE_BYTES = 240_000
const MAX_DIRECTORIES = 512
const MAX_ENTRIES = 16_384
const MAX_ENTRIES_PER_DIRECTORY = 2_048
const MAX_DEPTH = 16

interface CodeCollection {
  files: RetrievalCodeFileInput[]
  scan: RetrievalCodeScan
}

export async function collectCodeFiles(cwd: string, scopedPath: string | undefined, maxFiles: number): Promise<CodeCollection> {
  maxFiles = Number.isFinite(maxFiles) ? Math.max(1, Math.min(Math.floor(maxFiles), 3_000)) : 300
  const root = safeResolveWorkspace(cwd, scopedPath || '.')
  const files: RetrievalCodeFileInput[] = []
  const limits = new Set<RetrievalCodeScanLimit>()
  const scan: RetrievalCodeScan = {
    scope: normalizeSlash(path.relative(path.resolve(cwd), root)) || '.',
    readFiles: 0, discoveredFiles: 0, visitedDirectories: 0, visitedEntries: 0,
    skippedFiles: 0, unreadableDirectories: 0,
    maxFiles, maxDirectories: MAX_DIRECTORIES, maxEntries: MAX_ENTRIES,
    maxEntriesPerDirectory: MAX_ENTRIES_PER_DIRECTORY, maxDepth: MAX_DEPTH,
    limits: [], truncated: false,
  }
  const directories = [{ dir: root, depth: 0 }]
  const branches = new Map<string, string[]>()

  // Discover bounded directory batches first so one large app folder cannot exhaust the file-read budget.
  for (let cursor = 0; cursor < directories.length; cursor++) {
    if (scan.visitedEntries >= MAX_ENTRIES) { limits.add('entries'); break }
    const { dir, depth } = directories[cursor]
    scan.visitedDirectories++
    const entries: Dirent[] = []
    try {
      const handle = await fs.opendir(dir)
      for await (const entry of handle) {
        if (scan.visitedEntries >= MAX_ENTRIES) { limits.add('entries'); break }
        if (entries.length >= MAX_ENTRIES_PER_DIRECTORY) { limits.add('directory_entries'); break }
        scan.visitedEntries++
        entries.push(entry)
      }
    } catch { scan.unreadableDirectories++ }
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (entry.name.startsWith('.') || SKIP_DIRS.has(entry.name)) continue
      if (entry.isSymbolicLink()) continue
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        if (depth >= MAX_DEPTH) { limits.add('depth'); continue }
        if (directories.length >= MAX_DIRECTORIES) { limits.add('directories'); continue }
        directories.push({ dir: full, depth: depth + 1 })
        continue
      }
      if (!isCodeFile(entry.name)) continue
      const relative = path.relative(root, full).split(path.sep)
      const branch = relative.length > 1 ? relative[0] : '.'
      const candidates = branches.get(branch) ?? []
      candidates.push(full)
      branches.set(branch, candidates)
      scan.discoveredFiles++
    }
  }

  // Round-robin across scope branches gives native code a share even when app contains hundreds of candidates.
  const queues = [...branches].sort(([a], [b]) => a.localeCompare(b)).map(([, candidates]) => ({ candidates, cursor: 0 }))
  let remaining = scan.discoveredFiles
  while (remaining > 0 && files.length < maxFiles) {
    for (const queue of queues) {
      if (files.length >= maxFiles) break
      const full = queue.candidates[queue.cursor++]
      if (!full) continue
      remaining--
      try {
        const stat = await fs.lstat(full)
        if (!stat.isFile() || stat.size > MAX_CODE_BYTES) { scan.skippedFiles++; continue }
        const content = await fs.readFile(full, 'utf8')
        files.push({ path: normalizeSlash(path.relative(path.resolve(cwd), full)), content, modifiedAt: stat.mtimeMs })
      } catch { scan.skippedFiles++ }
    }
  }
  if (remaining > 0) limits.add('files')
  scan.readFiles = files.length
  scan.limits = [...limits].sort()
  scan.truncated = limits.size > 0 || scan.unreadableDirectories > 0
  return { files, scan }
}

function safeResolveWorkspace(cwd: string, relPath: string): string {
  const root = path.resolve(cwd)
  const realRoot = realpathSync.native(root)
  const target = path.resolve(root, relPath || '.')
  if (target !== root && !target.startsWith(root + path.sep)) throw new Error(`Retrieval path "${relPath}" escapes the workspace.`)
  let probe = target
  while (!existsSync(probe)) {
    const parent = path.dirname(probe)
    if (parent === probe) break
    probe = parent
  }
  const realProbe = realpathSync.native(probe)
  if (realProbe !== realRoot && !realProbe.startsWith(realRoot + path.sep)) throw new Error(`Retrieval path "${relPath}" escapes the workspace through a symlink.`)
  return target
}

function isCodeFile(name: string): boolean {
  const ext = path.extname(name).toLowerCase()
  return CODE_EXTENSIONS.has(ext) && !BLOCKED_NAMES.has(name) && !BLOCKED_EXTENSIONS.has(ext)
}
