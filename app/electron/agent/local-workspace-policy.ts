import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { WorkspaceLocation } from './types'

export interface RemoteWorkspaceScope { confineToHome?: boolean }

function deny(message: string): never {
  throw Object.assign(new Error(message), { code: 403 })
}

export async function canonicalHomeDirectory(target: string, home = os.homedir(), allowHome = true): Promise<string> {
  if (!target || !path.isAbsolute(target)) deny('Paired workspaces require an absolute directory inside the home folder.')
  const [canonical, canonicalHome] = await Promise.all([fs.realpath(target), fs.realpath(home)])
  if (canonical === path.parse(canonical).root || (canonical !== canonicalHome && !canonical.startsWith(canonicalHome + path.sep))) {
    deny('Directory access is limited to the home folder for paired devices.')
  }
  if (!allowHome && canonical === canonicalHome) deny('Choose a project directory below the home folder.')
  if (!(await fs.stat(canonical)).isDirectory()) deny('Workspace path is not a directory.')
  return canonical
}

/** Produce one authoritative local path, rejecting ambiguous cwd/workspace bindings. */
export async function resolvePairedWorkspace(cwd: string, workspace?: WorkspaceLocation | null, home = os.homedir()): Promise<string> {
  if (workspace && workspace.kind !== 'local') deny('Paired sessions are limited to local home workspaces.')
  const canonical = await canonicalHomeDirectory(cwd, home, false)
  if (workspace) {
    const effective = await canonicalHomeDirectory(workspace.path, home, false)
    if (effective !== canonical) deny('Project cwd and local workspace path must identify the same directory.')
  }
  return canonical
}
