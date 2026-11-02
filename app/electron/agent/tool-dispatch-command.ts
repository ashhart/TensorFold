import path from 'node:path'
import { randomUUID } from 'node:crypto'
import { blockedCommandReason, commandNetworkReason } from './command-policy'
import { commandOutputWithRedactionNotice, redactCommandStreams } from './command-output-redaction'
import { summarizeSandboxArtifactBundle, runSandboxedCommand, sandboxProfileForCommand, taskSandboxNetworkOverride } from './sandbox'
import { type WorkspaceBackend } from './workspace-backend'
import { isRemote, shArg, workspaceBackend } from './tool-workspace'
import type { ToolContext, ToolResult } from './tool-context'

interface RemoteCommandSnapshot { root: string; prefix: string; tree: string }

async function captureRemoteCommandSnapshot(backend: WorkspaceBackend, cwd: string): Promise<RemoteCommandSnapshot | null> {
  const token = randomUUID().replace(/-/g, '')
  const script = [
    'set -eu',
    'root=$(git rev-parse --show-toplevel 2>/dev/null) || exit 3',
    'prefix=$(git rev-parse --show-prefix 2>/dev/null) || exit 3',
    'common=$(cd "$root" && git rev-parse --git-common-dir 2>/dev/null) || exit 3',
    'case "$common" in /*) ;; *) common="$root/$common" ;; esac',
    `idx="$common/vontra-command-index-${token}"`,
    'trap \'rm -f "$idx"\' EXIT HUP INT TERM',
    'if git rev-parse --verify --quiet HEAD >/dev/null; then GIT_INDEX_FILE="$idx" git read-tree HEAD; fi',
    'cd "$root"',
    'GIT_INDEX_FILE="$idx" git add -A -- .',
    'tree=$(GIT_INDEX_FILE="$idx" git write-tree)',
    'printf \'%s\\0%s\\0%s\\0\' "$root" "$prefix" "$tree"',
  ].join('; ')
  const result = await backend.exec(script, { cwd, timeoutMs: 25_000 })
  if (result.code !== 0) return null
  const fields = result.stdout.split('\0')
  const [root, prefix, tree] = fields
  if (!root || !/^[0-9a-f]{40,64}$/i.test(tree ?? '')) return null
  return { root, prefix: String(prefix ?? '').replace(/\\/g, '/').replace(/^\/+|\/+$/g, ''), tree }
}

async function remoteCommandChangedPaths(
  backend: WorkspaceBackend,
  cwd: string,
  before: RemoteCommandSnapshot | null,
): Promise<string[]> {
  if (!before) return []
  const after = await captureRemoteCommandSnapshot(backend, cwd)
  if (!after || after.root !== before.root) return []
  const diff = await backend.exec(
    `git diff --name-only -z --no-renames ${shArg(before.tree)} ${shArg(after.tree)} --`,
    { cwd: before.root, timeoutMs: 25_000 },
  )
  if (diff.code !== 0) return []
  const base = `/${before.prefix}`.replace(/\/+$/, '') || '/'
  return [...new Set(diff.stdout.split('\0').filter(Boolean).map((repoPath) =>
    path.posix.relative(base, `/${repoPath.replace(/^\/+/, '')}`) || '.',
  ))].sort()
}

// Command dispatch preserves sandbox selection, output redaction and mutation evidence.
export async function executeCommandTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  switch (name) {
    case 'run_command': {
      const command: string = args.command ?? ''
      const blocked = blockedCommandReason(command)
      if (blocked) return { ok: false, output: `Blocked by TensorFold safety policy: ${blocked}.`, command, meta: 'blocked' }
      const networkReason = ctx.networkCapabilityAllowed === false ? commandNetworkReason(command) : null
      if (networkReason) return { ok: false, output: `Blocked by ticket capability policy: ${networkReason}; add the network capability to run it.`, command, meta: 'blocked' }
      const profile = sandboxProfileForCommand(command)
      if (isRemote(ctx)) {
        const startedRemote = Date.now()
        const backend = workspaceBackend(ctx)
        const mutationBaseline = profile === 'read-only' ? null : await captureRemoteCommandSnapshot(backend, ctx.cwd).catch(() => null)
        const res = await backend.exec(command, { cwd: ctx.cwd, timeoutMs: 120_000, signal: ctx.signal })
        const changedPaths = await remoteCommandChangedPaths(backend, ctx.cwd, mutationBaseline).catch(() => [])
        const msRemote = Date.now() - startedRemote
        const streams = redactCommandStreams(res.stdout, res.stderr)
        const out = [streams.stdout, streams.stderr].filter(Boolean).join('\n').trim()
        return {
          ok: res.code === 0,
          output: commandOutputWithRedactionNotice(out || (res.code === 0 ? '(no output)' : `Exited with code ${res.code}`), streams.redactionCount),
          command,
          meta: `exit ${res.code} · ${msRemote}ms · ssh${streams.redactionCount ? ` · redacted=${streams.redactionCount}` : ''}`,
          changedPaths,
        }
      }
      const started = Date.now()
      const result = await runSandboxedCommand(command, {
        cwd: ctx.cwd,
        profile,
        // Only a command already classified as network-bearing can use the task
        // network toggle. Opaque build/interpreter profiles remain network-denied.
        allowNetwork: taskSandboxNetworkOverride(profile, ctx.allowSandboxNetwork),
        timeoutMs: 120_000,
        maxBuffer: 8 * 1024 * 1024,
        signal: ctx.signal,
      })
      const ms = Date.now() - started
      const out = [result.stdout, result.stderr].filter(Boolean).join('\n').trim()
      const redactionCount = result.redactionCount ?? 0
      // The artifact record rides as metadata for the UI; the model sees only the command output.
      return {
        ok: result.code === 0,
        output: commandOutputWithRedactionNotice(out || (result.code === 0 ? '(no output)' : `Exited with code ${result.code}`), redactionCount),
        command,
        meta: `exit ${result.code} · ${ms}ms · sandbox=${result.profile} · ${result.implementation}${result.artifacts ? ` · artifacts=${result.artifacts.id}` : ''}${redactionCount ? ` · redacted=${redactionCount}` : ''}`,
        sandboxArtifacts: result.artifacts,
        ...(result.artifacts ? { artifacts: summarizeSandboxArtifactBundle(result.artifacts) } : {}),
        changedPaths: [...new Set([
          ...(result.changedPaths ?? []),
          ...(result.artifacts?.diffs.map((entry) => entry.path) ?? []),
        ])],
      }
    }
    default: return { ok: false, output: `Unknown tool: ${name}` }
  }
}
