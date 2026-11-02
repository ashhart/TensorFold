import { promises as fs } from 'node:fs'
import { createHash } from 'node:crypto'
import path from 'node:path'
import { canonicalMcpManifest, verifyConfiguredManifest } from './mcp-process'
import { insideWorkspace, protectedGitPath, snapshotWorkspaceFile } from './isolated-workspace-files'
import { resolveCliBinary } from './cli-path'
import type { IsolatedWorkspace } from './isolated-workspace'
import type { McpServerConfig } from './types'

export async function verifyPrivateMcpLaunch(config: McpServerConfig, workspace: IsolatedWorkspace, views: Array<{ source: string; target: string }>): Promise<void> {
  if (!config.manifestDigest) return
  const checked = await verifyConfiguredManifest(config)
  if (!checked.ok || checked.digest !== config.manifestDigest) throw new Error('MCP manifest or launch files changed during isolation preparation.')
  const manifest = JSON.parse(await fs.readFile(config.manifestPath!, 'utf8'))
  if (createHash('sha256').update(canonicalMcpManifest(manifest)).digest('hex') !== config.manifestDigest) throw new Error('MCP manifest changed during isolation preparation.')
  const command = config.command.includes('/') || config.command.includes(path.sep) ? path.resolve(workspace.root, config.command)
    : resolveCliBinary(null, [config.command], { env: { ...process.env, ...config.env } })
  const pinned = [...(manifest.launchFiles ?? []), ...(manifest.commandSha256 ? [{ path: command, sha256: manifest.commandSha256 }] : [])]
  for (const entry of pinned) {
    const source = path.resolve(workspace.root, entry.path)
    if (protectedGitPath(source)) throw new Error('Pinned MCP launch code cannot reside in mutable Git metadata.')
    const view = views.find(view => insideWorkspace(view.source, source))
    const privateFile = insideWorkspace(workspace.root, source) ? path.join(workspace.workspace, path.relative(workspace.root, source))
      : view ? path.join(view.target, path.relative(view.source, source)) : source
    if ((await snapshotWorkspaceFile(privateFile)).sha256 !== entry.sha256) throw new Error('Private MCP launch code does not match its signed inventory.')
  }
}
