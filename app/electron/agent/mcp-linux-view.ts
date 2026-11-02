import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { resolveCliBinary } from './cli-path'
import { linuxSandboxReadRoots } from './linux-sandbox'
import { insideWorkspace, scanWorkspace, snapshotWorkspaceFile } from './isolated-workspace-files'
import type { IsolatedWorkspace } from './isolated-workspace'
import type { McpServerConfig } from './types'

export async function prepareLinuxMcpViews(config: McpServerConfig, workspace: IsolatedWorkspace): Promise<Array<{ source: string; target: string }>> {
  const command = config.command.includes('/') ? path.resolve(workspace.root, config.command)
    : resolveCliBinary(null, [config.command], { env: { ...process.env, ...config.env } })
  if (!path.isAbsolute(command)) throw new Error('MCP command could not be resolved to a local executable.')
  const views: Array<{ source: string; target: string }> = []
  let bytes = 0, files = 0
  const candidates = [path.dirname(command)]
  for (const argument of config.args) {
    if (!argument || argument.startsWith('-')) continue
    const candidate = path.resolve(workspace.root, argument)
    const stat = await fs.stat(candidate).catch(() => null)
    if (!stat) continue
    candidates.push(stat.isDirectory() || !/\.(?:[cm]?js|py|rb|sh|bash|zsh|so)$/i.test(candidate) ? candidate : path.dirname(candidate))
  }
  for (const input of candidates) {
    const source = path.resolve(input)
    const canonical = await fs.realpath(source)
    if (insideWorkspace(workspace.root, source) || views.some(view => insideWorkspace(view.source, source))) continue
    const home = await fs.realpath(os.homedir()), temporary = await fs.realpath(os.tmpdir())
    if ([source, canonical].some(root => insideWorkspace(root, home) || insideWorkspace(root, workspace.root) || root === temporary)) {
      throw new Error('MCP runtime roots cannot expose a home, temporary root or workspace ancestor.')
    }
    if (['/usr', '/bin', '/sbin', '/lib', '/lib64', '/nix/store'].some(root => insideWorkspace(root, source))) continue
    if (views.length >= 32) throw new Error('Too many MCP runtime roots.')
    const target = path.join(workspace.container, 'runtimes', String(views.length))
    await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 })
    if ((await fs.stat(canonical)).isFile()) {
      bytes += (await snapshotWorkspaceFile(canonical, target)).size
      files++
    } else {
      await fs.mkdir(target, { mode: 0o700 })
      const copied = await scanWorkspace(canonical, target, { externalReadRoots: linuxSandboxReadRoots() })
      files += copied.size
      for (const entry of copied.values()) if (entry.kind === 'file') bytes += entry.size
    }
    if (bytes > 2 * 1024 * 1024 * 1024 || files > 100000) throw new Error('MCP runtime copies exceed the isolation limit.')
    views.push({ source, target })
  }
  return views
}
