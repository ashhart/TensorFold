import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { lstatSync, readlinkSync, readdirSync, readFileSync, realpathSync, statSync } from 'node:fs'
import path from 'node:path'
import { hostGitEnv } from './host-git-env'
import { redactCommandStreams } from './command-output-redaction'
import { SECRET_MASK } from './security'
import { redactSecretValues } from './secret-detection'
import { SANDBOX_ARTIFACT_SCHEMA } from './sandbox-types'
import type {
  SandboxArtifactBaseline, SandboxArtifactBundle, SandboxArtifactDiff, SandboxFileArtifact,
  SandboxLogArtifact, SandboxWorkspaceSnapshot, SandboxedCommandResult,
} from './sandbox-types'
import type { ToolArtifactSummary } from './types'

const INPUT_SNAPSHOT_PATHS = [
  'package.json',
  'package-lock.json',
  'pnpm-lock.yaml',
  'yarn.lock',
  'bun.lockb',
  'tsconfig.json',
  'vite.config.ts',
]
const OUTPUT_ARTIFACT_PATHS = [
  'package.json',
  'package-lock.json',
  'npm-shrinkwrap.json',
  '.vontra-artifacts',
  'artifacts',
  'screenshots',
  'test-results',
  'playwright-report',
  'coverage',
  'dist',
]
const ARTIFACT_SKIP_DIRS = new Set(['.git', 'node_modules'])
const MAX_ARTIFACT_FILES = 120
const MAX_HASH_BYTES = 10 * 1024 * 1024
const SCREENSHOT_RE = /\.(?:png|jpe?g|webp|gif)$/i

export function captureSandboxArtifactBaseline(cwd: string, now = Date.now(), captureWorkspace = true): SandboxArtifactBaseline {
  const root = realpathSync(cwd)
  return {
    inputSnapshot: collectArtifactFiles(root, INPUT_SNAPSHOT_PATHS),
    outputFiles: collectArtifactFiles(root, OUTPUT_ARTIFACT_PATHS),
    capturedAt: now,
    ...(captureWorkspace ? { workspaceSnapshot: captureWorkspaceSnapshot(root) ?? undefined } : {}),
  }
}

export function buildSandboxArtifactBundle(input: {
  cwd: string
  command: string
  result: Pick<SandboxedCommandResult, 'stdout' | 'stderr' | 'profile' | 'implementation'>
  baseline?: SandboxArtifactBaseline
  now?: number
}): SandboxArtifactBundle {
  const now = input.now ?? Date.now()
  const root = realpathSync(input.cwd)
  const baseline = input.baseline ?? captureSandboxArtifactBaseline(root, now)
  const outputFiles = collectArtifactFiles(root, OUTPUT_ARTIFACT_PATHS)
  const screenshots = outputFiles.filter((file) => SCREENSHOT_RE.test(file.path))
  const diffs = diffArtifactFiles(baseline.outputFiles, outputFiles)
  const commandStreams = redactCommandStreams(input.result.stdout, input.result.stderr)
  const logs = ([
    sandboxLogArtifact('stdout', commandStreams.stdout),
    sandboxLogArtifact('stderr', commandStreams.stderr),
  ] as SandboxLogArtifact[]).filter((log) => log.bytes > 0)

  return {
    schema: SANDBOX_ARTIFACT_SCHEMA,
    id: `sbox-${Math.max(0, Math.floor(now)).toString(36)}`,
    command: input.command,
    cwd: root,
    profile: input.result.profile,
    implementation: input.result.implementation,
    inputSnapshot: baseline.inputSnapshot,
    outputFiles,
    screenshots,
    logs,
    diffs,
    patch: buildArtifactPatch(baseline.outputFiles, outputFiles, diffs),
    createdAt: now,
  }
}

const ARTIFACT_SUMMARY_PATHS_MAX = 20

export function summarizeSandboxArtifactBundle(bundle: SandboxArtifactBundle): ToolArtifactSummary {
  return {
    id: bundle.id,
    profile: bundle.profile,
    implementation: bundle.implementation,
    inputFiles: bundle.inputSnapshot.length,
    outputFiles: bundle.outputFiles.length,
    screenshots: bundle.screenshots.slice(0, ARTIFACT_SUMMARY_PATHS_MAX).map((file) => file.path),
    diffs: bundle.diffs.slice(0, ARTIFACT_SUMMARY_PATHS_MAX).map((diff) => ({ path: diff.path, status: diff.status })),
    patch: !!bundle.patch,
    logBytes: bundle.logs.reduce((sum, log) => sum + log.bytes, 0),
  }
}

export function attachSandboxArtifacts(command: string, cwd: string, result: SandboxedCommandResult, baseline: SandboxArtifactBaseline): SandboxedCommandResult {
  const streams = redactCommandStreams(result.stdout, result.stderr)
  const sanitized = { ...result, ...streams }
  return {
    ...sanitized,
    artifacts: buildSandboxArtifactBundle({ cwd, command, result: sanitized, baseline }),
    changedPaths: changedWorkspacePaths(cwd, baseline.workspaceSnapshot),
  }
}

function gitOutput(args: string[], cwd: string): Buffer | null {
  const result = spawnSync('git', args, {
    cwd,
    encoding: 'buffer',
    maxBuffer: 32 * 1024 * 1024,
    timeout: 20_000,
    env: hostGitEnv({ ...process.env, GIT_OPTIONAL_LOCKS: '0' }),
  })
  return result.status === 0 && Buffer.isBuffer(result.stdout) ? result.stdout : null
}

function captureWorkspaceSnapshot(cwd: string): SandboxWorkspaceSnapshot | null {
  const rootRaw = gitOutput(['rev-parse', '--show-toplevel'], cwd)
  if (!rootRaw) return null
  const rootText = rootRaw.toString('utf8').trim()
  if (!rootText) return null
  let root: string
  try { root = realpathSync(rootText) } catch { return null }
  const listed = gitOutput(['ls-files', '-c', '-o', '--exclude-standard', '-z'], root)
  if (!listed) return null
  const files: Array<{ path: string; fingerprint: string }> = []
  for (const rel of listed.toString('utf8').split('\0').filter(Boolean)) {
    if (rel.startsWith('/') || rel.split('/').includes('..')) continue
    const full = path.join(root, ...rel.split('/'))
    let stat
    try { stat = lstatSync(full) } catch { continue }
    let fingerprint: string
    if (stat.isSymbolicLink()) {
      try { fingerprint = `link:${workspaceContentFingerprint(readlinkSync(full))}` } catch { continue }
    } else if (stat.isFile()) {
      try {
        fingerprint = stat.size <= MAX_HASH_BYTES
          ? `file:${workspaceContentFingerprint(readFileSync(full))}`
          : `large:${stat.size}:${stat.mtimeMs}:${stat.ctimeMs}`
      } catch { continue }
    } else {
      continue
    }
    files.push({ path: rel.replace(/\\/g, '/'), fingerprint })
  }
  return { root, files: files.sort((a, b) => a.path.localeCompare(b.path)) }
}

function workspaceContentFingerprint(content: string | Buffer): string {
  return createHash('sha256').update(content).digest('hex')
}

function changedWorkspacePaths(cwd: string, before: SandboxWorkspaceSnapshot | undefined): string[] {
  if (!before) return []
  const after = captureWorkspaceSnapshot(before.root)
  if (!after || after.root !== before.root) return []
  const previous = new Map(before.files.map((file) => [file.path, file.fingerprint]))
  const current = new Map(after.files.map((file) => [file.path, file.fingerprint]))
  const changed = new Set<string>()
  for (const [file, fingerprint] of current) if (previous.get(file) !== fingerprint) changed.add(file)
  for (const file of previous.keys()) if (!current.has(file)) changed.add(file)
  return [...changed].sort().map((repoPath) => {
    const relative = path.relative(cwd, path.join(before.root, ...repoPath.split('/')))
    return relative.split(path.sep).join('/') || '.'
  })
}

function collectArtifactFiles(cwd: string, roots: string[]): SandboxFileArtifact[] {
  const out: SandboxFileArtifact[] = []
  for (const root of roots) {
    if (out.length >= MAX_ARTIFACT_FILES) break
    collectArtifactPath(cwd, path.join(cwd, root), out)
  }
  return out.sort((a, b) => a.path.localeCompare(b.path))
}

function collectArtifactPath(cwd: string, fullPath: string, out: SandboxFileArtifact[]): void {
  if (out.length >= MAX_ARTIFACT_FILES) return
  let stat
  try { stat = statSync(fullPath) } catch { return }
  if (stat.isDirectory()) {
    const name = path.basename(fullPath)
    if (ARTIFACT_SKIP_DIRS.has(name)) return
    let entries
    try { entries = readdirSync(fullPath, { withFileTypes: true }) } catch { return }
    for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
      if (out.length >= MAX_ARTIFACT_FILES) break
      collectArtifactPath(cwd, path.join(fullPath, entry.name), out)
    }
    return
  }
  if (!stat.isFile()) return
  const relPath = normalizeArtifactPath(cwd, fullPath)
  if (!relPath) return
  out.push({
    path: relPath,
    bytes: stat.size,
    sha256: hashArtifactFile(fullPath, stat.size),
    ...artifactText(fullPath, stat.size),
  })
}

function normalizeArtifactPath(cwd: string, fullPath: string): string | null {
  const relative = path.relative(cwd, fullPath)
  if (!relative || relative.startsWith('..') || path.isAbsolute(relative)) return null
  return relative.split(path.sep).join('/')
}

function hashArtifactFile(fullPath: string, bytes: number): string {
  const hash = createHash('sha256')
  if (bytes > MAX_HASH_BYTES) {
    hash.update(`large:${bytes}:${statSync(fullPath).mtimeMs}`)
  } else {
    hash.update(readFileSync(fullPath))
  }
  return hash.digest('hex')
}

function artifactText(fullPath: string, bytes: number): { text?: string } {
  if (bytes > 200_000) return {}
  let buf
  try { buf = readFileSync(fullPath) } catch { return {} }
  if (buf.includes(0)) return {}
  const { text } = redactSecretValues(buf.toString('utf8'), SECRET_MASK)
  return { text }
}

function sandboxLogArtifact(kind: SandboxLogArtifact['kind'], text: string): SandboxLogArtifact {
  return {
    kind,
    bytes: Buffer.byteLength(text),
    preview: text.replace(/\s+/g, ' ').trim().slice(0, 400),
  }
}

function diffArtifactFiles(before: SandboxFileArtifact[], after: SandboxFileArtifact[]): SandboxArtifactDiff[] {
  const previous = new Map(before.map((file) => [file.path, file]))
  const current = new Map(after.map((file) => [file.path, file]))
  const diffs: SandboxArtifactDiff[] = []
  for (const file of after) {
    const old = previous.get(file.path)
    if (!old) {
      diffs.push({ path: file.path, status: 'added', afterSha256: file.sha256, bytes: file.bytes })
    } else if (old.sha256 !== file.sha256 || old.bytes !== file.bytes) {
      diffs.push({ path: file.path, status: 'changed', beforeSha256: old.sha256, afterSha256: file.sha256, bytes: file.bytes })
    }
  }
  for (const file of before) {
    if (!current.has(file.path)) diffs.push({ path: file.path, status: 'deleted', beforeSha256: file.sha256, bytes: file.bytes })
  }
  return diffs.sort((a, b) => a.path.localeCompare(b.path))
}

function buildArtifactPatch(before: SandboxFileArtifact[], after: SandboxFileArtifact[], diffs: SandboxArtifactDiff[]): string | null {
  const previous = new Map(before.map((file) => [file.path, file]))
  const current = new Map(after.map((file) => [file.path, file]))
  const patches: string[] = []
  for (const diff of diffs) {
    const oldFile = previous.get(diff.path)
    const newFile = current.get(diff.path)
    const oldText = diff.status === 'added' ? '' : oldFile?.text
    const newText = diff.status === 'deleted' ? '' : newFile?.text
    if (oldText == null || newText == null) continue
    patches.push(formatArtifactPatch(diff.path, diff.status, oldText, newText))
  }
  return patches.length ? patches.join('\n') : null
}

function formatArtifactPatch(pathName: string, status: SandboxArtifactDiff['status'], oldText: string, newText: string): string {
  const oldLines = patchLines(oldText)
  const newLines = patchLines(newText)
  return [
    `diff --git a/${pathName} b/${pathName}`,
    `--- ${status === 'added' ? '/dev/null' : `a/${pathName}`}`,
    `+++ ${status === 'deleted' ? '/dev/null' : `b/${pathName}`}`,
    '@@',
    ...oldLines.map((line) => `-${line}`),
    ...newLines.map((line) => `+${line}`),
  ].join('\n')
}

function patchLines(text: string): string[] {
  const lines = text.replace(/\r\n/g, '\n').split('\n')
  if (lines.at(-1) === '') lines.pop()
  return lines
}
