import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { lstatSync, readlinkSync, readFileSync, realpathSync } from 'node:fs'
import path from 'node:path'
import { hostGitEnv } from './host-git-env'

export interface SandboxWorkspaceSnapshot {
  root: string
  files: Array<{ path: string; fingerprint: string }>
}

export const MAX_HASH_BYTES = 10 * 1024 * 1024

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

export function captureWorkspaceSnapshot(cwd: string): SandboxWorkspaceSnapshot | null {
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

export function changedWorkspacePaths(cwd: string, before: SandboxWorkspaceSnapshot | undefined): string[] {
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
