import { linuxSandboxCommand, linuxSandboxReadRoots } from './linux-sandbox'
import { windowsSandboxLaunch } from './windows-sandbox'
import { prepareIsolatedWorkspace, publishIsolatedWorkspace, removeIsolatedWorkspace, type IsolatedWorkspace } from './isolated-workspace'
import { sandboxCommandEnv, runSandboxProcess } from './sandbox-process'
import type { SandboxProfileId, SandboxedCommandOptions, SandboxedCommandResult } from './sandbox-types'

export async function runPortableSandboxCommand(argv: string[], options: SandboxedCommandOptions & {
  platform: 'linux' | 'win32'; profile: SandboxProfileId; allowWrite: boolean; privateInstall: boolean;
}): Promise<SandboxedCommandResult> {
  let layout: IsolatedWorkspace | undefined
  const implementation = options.platform === 'linux' ? 'linux-bubblewrap' : 'windows-appcontainer'
  try {
    if (options.signal?.aborted) return { code: 130, stdout: '', stderr: 'Cancelled before sandbox launch.', sandboxed: true, profile: options.profile, implementation }
    if (options.platform === 'win32' || (options.allowWrite && !options.privateInstall)) {
      layout = await prepareIsolatedWorkspace(options.cwd, options.platform === 'linux' ? linuxSandboxReadRoots() : [])
    }
    const launch = options.platform === 'linux'
      ? { command: '', args: linuxSandboxCommand({ command: argv[0], args: argv.slice(1), cwd: options.cwd,
        workspaceSource: layout?.workspace, allowNetwork: options.allowNetwork === true, writableStage: options.allowWrite,
        protectedPaths: layout ? [...layout.original.keys()].filter(relative => /(^|\/)\.git$/i.test(relative)) : [] }),
      cwd: options.cwd, env: undefined, cancelFile: undefined }
      : await windowsSandboxLaunch({ layout: layout!, command: argv[0], args: argv.slice(1), env: sandboxCommandEnv(options.cwd),
        allowNetwork: options.allowNetwork === true, allowWrite: options.allowWrite, timeoutMs: options.timeoutMs ?? 120000 })
    const result = await runSandboxProcess(options.platform === 'linux' ? launch.args : [launch.command, ...launch.args], {
      cwd: launch.cwd, timeoutMs: options.timeoutMs, maxBuffer: options.maxBuffer, signal: options.signal,
      sandboxed: true, profile: options.profile, implementation,
      privateTemp: options.platform === 'linux' ? '/tmp' : layout!.scratch, env: launch.env, cancelFile: launch.cancelFile,
    })
    if (layout && options.allowWrite && !options.signal?.aborted && ![124, 126, 130].includes(result.code)) {
      try { result.changedPaths = await publishIsolatedWorkspace(layout, options.signal) } catch (error: any) {
        return { ...result, code: 126, stderr: `${result.stderr}\nIsolated publication refused or stopped: ${error.message}`.trim() }
      }
    }
    return result
  } catch (error: any) {
    return { code: 126, stdout: '', stderr: `Process isolation refused: ${error.message}`, sandboxed: true, profile: options.profile, implementation }
  } finally { if (layout) await removeIsolatedWorkspace(layout) }
}
