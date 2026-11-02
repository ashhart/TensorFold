import { promises as fs } from 'node:fs'
import { parsePackageInstallCommand, packageInstallShellCommand } from './package-install-command'
import {
  installEntryIdentity, preparePackageInstallLayout, sameInstallEntry,
  type PackageInstallLayout,
} from './package-install-layout'
import { publishPackageInstallation, type PackagePublicationHooks } from './package-install-transaction'
import type { SandboxedCommandResult } from './sandbox-types'

export interface IsolatedPackageRunner {
  (input: { command: string; cwd: string; workspace: string }): Promise<SandboxedCommandResult>
}

const activeInstalls = new Map<string, Promise<void>>()

async function withInstallLock<T>(root: string, action: () => Promise<T>): Promise<T> {
  const previous = activeInstalls.get(root) ?? Promise.resolve()
  let release!: () => void
  const pending = new Promise<void>((resolve) => { release = resolve })
  const queued = previous.then(() => pending)
  activeInstalls.set(root, queued)
  await previous
  try { return await action() } finally {
    release()
    if (activeInstalls.get(root) === queued) activeInstalls.delete(root)
  }
}

function failure(message: string, template?: SandboxedCommandResult, cancelled = false): SandboxedCommandResult {
  return {
    code: cancelled ? 130 : 1,
    stdout: template?.stdout ?? '',
    stderr: [template?.stderr, message].filter(Boolean).join('\n'),
    sandboxed: template?.sandboxed ?? false,
    profile: 'package-install',
    implementation: template?.implementation ?? 'unsandboxed',
  }
}

async function removeStage(layout: PackageInstallLayout): Promise<void> {
  for (let attempt = 0; attempt < 4; attempt++) {
    const current = await installEntryIdentity(layout.stage)
    if (!current) return
    if (!sameInstallEntry(layout.stageIdentity, current)) throw new Error('The installation stage identity changed.')
    try {
      await fs.rm(layout.stage, { recursive: true, force: true })
      return
    } catch (error: any) {
      if (attempt === 3 || !['EBUSY', 'ENOTEMPTY', 'EPERM', 'EMFILE', 'ENFILE'].includes(error?.code)) throw error
      await new Promise((resolve) => setTimeout(resolve, 40 * (attempt + 1)))
    }
  }
}

export async function installWorkspacePackages(
  command: string,
  cwd: string,
  run: IsolatedPackageRunner,
  hooks: PackagePublicationHooks = {},
): Promise<SandboxedCommandResult> {
  let plan
  try { plan = parsePackageInstallCommand(command) } catch (error: any) {
    return failure(String(error?.message ?? error))
  }
  const root = await fs.realpath(cwd)
  return withInstallLock(root, async () => {
    let layout: PackageInstallLayout | undefined
    let executed: SandboxedCommandResult | undefined
    let outcome: SandboxedCommandResult | undefined
    const finish = (result: SandboxedCommandResult) => { outcome = result; return result }
    try {
      if (hooks.signal?.aborted) return finish(failure('Dependency installation was cancelled.', undefined, true))
      layout = await preparePackageInstallLayout(root)
      if (plan.operation === 'ci' && !layout.files.get('package-lock.json') && !layout.files.get('npm-shrinkwrap.json')) {
        return finish(failure('npm ci needs an existing npm lockfile; run npm install first.'))
      }
      executed = await run({ command: packageInstallShellCommand(plan, layout.stage), cwd: layout.stage, workspace: root })
      if (executed.code !== 0) return finish(executed)
      if (hooks.signal?.aborted) return finish(failure('Dependency installation was cancelled before publication.', executed, true))
      const published = await publishPackageInstallation(layout, hooks)
      return finish({
        ...executed,
        stdout: [
          executed.stdout.trim(),
          `Dependencies were installed in ${root}.`,
          'Lifecycle scripts were skipped. Run a separately approved npm rebuild if native dependencies need setup.',
        ].filter(Boolean).join('\n'),
        stderr: [executed.stderr, ...published.warnings].filter(Boolean).join('\n'),
        changedPaths: published.changedPaths,
      })
    } catch (error: any) {
      return finish(failure(`Dependencies were not installed: ${error?.message ?? error}`, executed, hooks.signal?.aborted))
    } finally {
      if (layout) {
        try { await removeStage(layout) } catch (error: any) {
          if (outcome) outcome.stderr += `\nAn installation stage could not be removed: ${layout.stage}. ${error?.message ?? error}`
        }
      }
    }
  })
}
