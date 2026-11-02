import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import { windowsSandboxHelper } from './windows-sandbox-helper'
import { prepareWindowsGitView } from './windows-git-view'
import { windowsLocalInput } from './windows-local-input'
import { snapshotWindowsWorkspace } from './windows-workspace-snapshot'
import { insideWorkspace, scanWorkspace, sameWorkspaceContent, type WorkspaceInventory } from './isolated-workspace-files'
export { publishIsolatedWorkspace } from './isolated-workspace-publish'

export interface IsolatedWorkspace {
  root: string
  container: string
  workspace: string
  scratch: string
  original: WorkspaceInventory
  initial: WorkspaceInventory
  rootIdentity: { dev: number; ino: number }
  containerIdentity: { dev: number; ino: number }
  externalReadRoots: string[]
}

export async function prepareIsolatedWorkspace(input: string, externalReadRoots: string[] = []): Promise<IsolatedWorkspace> {
  if (process.platform === 'win32' && (!await windowsLocalInput(input) || !await windowsLocalInput(os.tmpdir()))) {
    throw new Error('Windows isolation needs local project and temporary directories without reparse-point ancestors.')
  }
  const root = await fs.realpath(input)
  if (root === path.parse(root).root || insideWorkspace(root, await fs.realpath(os.homedir()))) throw new Error('Choose a project directory for isolation.')
  const temporary = await fs.realpath(os.tmpdir())
  if (insideWorkspace(root, temporary)) throw new Error('The workspace cannot contain the isolation staging directory.')
  const container = await fs.mkdtemp(path.join(temporary, 'tensorfold-isolation-'))
  const workspace = path.join(container, 'workspace'), scratch = path.join(container, 'scratch')
  try {
    await fs.chmod(container, 0o700)
    if (process.platform === 'win32') await promisify(execFile)(windowsSandboxHelper().binary, ['--prepare-directory', container], { timeout: 5000, windowsHide: true })
    await fs.mkdir(workspace, { mode: 0o700 }); await fs.mkdir(scratch, { mode: 0o700 })
    const rootIdentity = await fs.lstat(root), containerIdentity = await fs.lstat(container)
    if (process.platform === 'win32') await snapshotWindowsWorkspace(root, workspace, true)
    const original = await scanWorkspace(root, process.platform === 'win32' ? undefined : workspace, { externalReadRoots })
    if (process.platform === 'win32') {
      const copied = await scanWorkspace(workspace)
      for (const [relative, entry] of original) if (entry.kind !== 'link' && !sameWorkspaceContent(entry, copied.get(relative))) {
        throw new Error('Workspace changed while its private Windows copy was prepared.')
      }
    }
    if (process.platform === 'win32') await prepareWindowsGitView(root, workspace, container, original)
    const initial = await scanWorkspace(workspace, undefined, { externalReadRoots, virtualRoot: root })
    return { root, container, workspace, scratch, original, initial, rootIdentity, containerIdentity, externalReadRoots }
  } catch (error) { await fs.rm(container, { recursive: true, force: true }); throw error }
}

export async function removeIsolatedWorkspace(layout: IsolatedWorkspace): Promise<void> {
  const current = await fs.lstat(layout.container).catch(error => { if (error.code !== 'ENOENT') throw error; return null })
  if (!current) return
  if (!current.isDirectory() || current.isSymbolicLink() || current.dev !== layout.containerIdentity.dev || current.ino !== layout.containerIdentity.ino) {
    throw new Error('Isolation staging identity changed; cleanup refused.')
  }
  const makeRemovable = async (directory: string): Promise<void> => {
    const stat = await fs.lstat(directory)
    if (!stat.isDirectory() || stat.isSymbolicLink()) return
    await fs.chmod(directory, 0o700)
    for (const entry of await fs.readdir(directory)) await makeRemovable(path.join(directory, entry))
  }
  await makeRemovable(layout.container)
  await fs.rm(layout.container, { recursive: true, force: true, maxRetries: 4, retryDelay: 80 })
}
