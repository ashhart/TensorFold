import { writeFileSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { canUseLinuxSandbox, linuxSandboxReadRoots } from './linux-sandbox'
import { canUseWindowsSandbox, windowsSandboxLaunch, type WindowsSandboxLaunch } from './windows-sandbox'
import { prepareIsolatedWorkspace, publishIsolatedWorkspace, removeIsolatedWorkspace, type IsolatedWorkspace } from './isolated-workspace'
import { sameWorkspaceContent, scanWorkspace } from './isolated-workspace-files'
import { sandboxCommandEnv } from './sandbox-process'
import type { McpServerConfig } from './types'
import type { ChildProcess } from 'node:child_process'
import { prepareLinuxMcpViews } from './mcp-linux-view'
import { verifyPrivateMcpLaunch } from './mcp-isolation-manifest'

export async function waitForMcpSandbox(proc: ChildProcess, readyFile?: string): Promise<void> {
  if (!readyFile) return
  const deadline = Date.now() + 60000
  while (Date.now() < deadline) {
    if (proc.exitCode !== null || proc.signalCode !== null) throw new Error('Windows MCP isolation exited before becoming ready.')
    if (await fs.stat(readyFile).then(stat => stat.isFile(), () => false)) return
    await new Promise(resolve => setTimeout(resolve, 50))
  }
  throw new Error('Windows MCP isolation did not become ready within its startup limit.')
}

export async function disposeMcpIsolation(isolation: McpIsolation | undefined, proc?: ChildProcess): Promise<void> {
  if (!isolation) return
  isolation.cancel()
  if (proc && proc.exitCode === null && proc.signalCode === null) await new Promise<void>(resolve => proc.once('close', () => resolve()))
  await isolation.dispose()
}

export class McpIsolation {
  private queue: Promise<unknown> = Promise.resolve()
  private stopped = false
  private controller = new AbortController()
  constructor(readonly workspace: IsolatedWorkspace, readonly windowsLaunch?: WindowsSandboxLaunch, readonly readViews?: Array<{ source: string; target: string }>) {}

  async hostChanged(): Promise<boolean> {
    const current = await scanWorkspace(this.workspace.root, undefined, { externalReadRoots: this.workspace.externalReadRoots })
    return [...new Set([...current.keys(), ...this.workspace.original.keys()])]
      .some(key => !sameWorkspaceContent(current.get(key), this.workspace.original.get(key)))
  }

  async call<T>(action: () => Promise<T>, assertCurrent: () => void): Promise<T> {
    const previous = this.queue
    const pending = previous.catch(() => {}).then(async () => {
      if (this.stopped) throw new Error('MCP isolated session stopped.')
      assertCurrent()
      const result = await action()
      assertCurrent()
      if (this.stopped) throw new Error('MCP stopped before workspace publication.')
      await publishIsolatedWorkspace(this.workspace, this.controller.signal)
      return result
    })
    this.queue = pending
    return pending
  }

  cancel(): void {
    this.stopped = true
    this.controller.abort()
    if (this.windowsLaunch) {
      try { writeFileSync(this.windowsLaunch.cancelFile, 'cancel', { flag: 'wx' }) } catch {}
    }
  }

  async dispose(): Promise<void> {
    this.cancel()
    await this.queue.catch(() => {})
    await removeIsolatedWorkspace(this.workspace)
  }
}

export async function prepareMcpIsolation(config: McpServerConfig): Promise<McpIsolation | undefined> {
  if (config.launchMode === 'docker_gateway') return undefined
  if (!canUseLinuxSandbox() && !canUseWindowsSandbox()) return undefined
  const workspace = await prepareIsolatedWorkspace(path.resolve(config.cwd || process.cwd()), process.platform === 'linux' ? linuxSandboxReadRoots() : [])
  try {
    const launch = process.platform === 'win32' ? await windowsSandboxLaunch({ layout: workspace,
      command: config.command, args: config.args, allowNetwork: false, allowWrite: true,
      allowPathRuntimes: !!config.env?.PATH,
      env: { ...sandboxCommandEnv(workspace.workspace), ...config.env } }) : undefined
    const readViews = process.platform === 'linux' ? await prepareLinuxMcpViews(config, workspace) : launch?.readViews ?? []
    await verifyPrivateMcpLaunch(config, workspace, readViews)
    return new McpIsolation(workspace, launch, readViews)
  } catch (error) { await removeIsolatedWorkspace(workspace); throw error }
}
