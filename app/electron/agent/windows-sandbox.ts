import path from 'node:path'
import { windowsSandboxHelper } from './windows-sandbox-helper'
import { windowsRuntimeView } from './windows-sandbox-runtime'
import type { IsolatedWorkspace } from './isolated-workspace'
export { canUseWindowsSandbox } from './windows-sandbox-helper'

export interface WindowsSandboxLaunch {
  command: string
  args: string[]
  env: NodeJS.ProcessEnv
  cwd: string
  cancelFile: string
  readyFile: string
  readViews: Array<{ source: string; target: string }>
}

export function windowsSandboxArguments(input: {
  root: string; workspace: string; scratch: string; command: string; args: string[];
  allowNetwork: boolean; allowWrite: boolean; parentPid: number; timeoutMs: number;
}): string[] {
  if (!Number.isInteger(input.parentPid) || input.parentPid <= 0 || !Number.isInteger(input.timeoutMs) || input.timeoutMs < 0) {
    throw new Error('Invalid Windows sandbox process limits.')
  }
  for (const directory of [input.root, input.workspace, input.scratch]) {
    if (!/^[a-z]:\\/i.test(directory) || directory.startsWith('\\') || directory.includes('\0')) throw new Error('Windows isolation needs local absolute paths.')
  }
  return ['--run', input.root, input.workspace, input.scratch, input.allowNetwork ? '1' : '0', input.allowWrite ? '1' : '0',
    String(input.timeoutMs), String(input.parentPid), 'abi1', '--', input.command, ...input.args]
}

export async function windowsSandboxLaunch(input: {
  layout: IsolatedWorkspace; command: string; args: string[]; env: NodeJS.ProcessEnv;
  allowNetwork: boolean; allowWrite: boolean; timeoutMs?: number; allowPathRuntimes?: boolean;
}): Promise<WindowsSandboxLaunch> {
  const helper = windowsSandboxHelper()
  const runtime = await windowsRuntimeView(input.layout, input.command, input.args, input.env, input.allowPathRuntimes)
  let command = runtime.command, args = runtime.args
  if (/\.(?:cmd|bat)$/i.test(command)) {
    if ([command, ...args].some(value => /["%\r\n]/.test(value))) throw new Error('Use a native executable for MCP arguments requiring shell expansion.')
    args = ['/d', '/s', '/c', [command, ...args].map(value => `"${value}"`).join(' ')]
    command = runtime.env.ComSpec!
  }
  return {
    command: helper.binary, readViews: runtime.readViews,
    args: windowsSandboxArguments({ root: input.layout.container, workspace: input.layout.workspace, scratch: input.layout.scratch,
      command, args, allowNetwork: input.allowNetwork, allowWrite: input.allowWrite, timeoutMs: input.timeoutMs ?? 0, parentPid: process.pid }),
    env: runtime.env, cwd: input.layout.workspace, cancelFile: path.join(input.layout.container, 'cancel'), readyFile: path.join(input.layout.container, 'ready'),
  }
}
