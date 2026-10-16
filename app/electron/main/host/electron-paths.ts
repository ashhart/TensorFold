import { chmodSync, lstatSync, mkdirSync } from 'node:fs'
import type { promises as fsPromises } from 'node:fs'
import path from 'node:path'

export const ISOLATED_ELECTRON_PATH_NAMES = [
  'appData',
  'userData',
  'sessionData',
  'temp',
  'logs',
  'crashDumps',
] as const

export type IsolatedElectronPathName = typeof ISOLATED_ELECTRON_PATH_NAMES[number]

export interface IsolatedElectronPathPlan {
  root: string
  runtimeRoot: string
  paths: Record<IsolatedElectronPathName, string>
}

export interface ElectronPathSetter {
  setPath(name: IsolatedElectronPathName, value: string): void
}

interface PrivateDirectoryIo {
  mkdirSync: typeof mkdirSync
  chmodSync: typeof chmodSync
  lstatSync: typeof lstatSync
  uid: number | null
}

export function planIsolatedElectronPaths(vontraHome: string | null | undefined): IsolatedElectronPathPlan | null {
  const configured = String(vontraHome ?? '').trim()
  if (!configured) return null
  if (configured.includes('\0')) throw new Error('VONTRA_HOME contains an invalid null byte.')
  const root = path.resolve(configured)
  if (root === path.parse(root).root) throw new Error('VONTRA_HOME cannot be the filesystem root.')
  const runtimeRoot = path.join(root, 'electron-runtime')
  return {
    root,
    runtimeRoot,
    paths: {
      appData: path.join(runtimeRoot, 'app-data'),
      userData: path.join(runtimeRoot, 'user-data'),
      sessionData: path.join(runtimeRoot, 'session-data'),
      temp: path.join(runtimeRoot, 'temp'),
      logs: path.join(runtimeRoot, 'logs'),
      crashDumps: path.join(runtimeRoot, 'crash-dumps'),
    },
  }
}

/**
 * Establish every Electron/Chromium-owned writable path before app.ready. In
 * particular, sessionData owns cookies, local storage, network state, and the
 * Chromium HTTP cache; Electron has no separate supported `cache` PathName.
 */
export function applyIsolatedElectronPathPlan(
  plan: IsolatedElectronPathPlan | null,
  appPaths: ElectronPathSetter,
  overrides: Partial<PrivateDirectoryIo> = {},
): void {
  if (!plan) return
  const io: PrivateDirectoryIo = {
    mkdirSync: overrides.mkdirSync ?? mkdirSync,
    chmodSync: overrides.chmodSync ?? chmodSync,
    lstatSync: overrides.lstatSync ?? lstatSync,
    uid: overrides.uid === undefined
      ? (typeof process.getuid === 'function' ? process.getuid() : null)
      : overrides.uid,
  }
  const directories = [plan.root, plan.runtimeRoot, ...Object.values(plan.paths)]
  for (const directory of directories) ensurePrivateDirectory(directory, io)
  // Set paths only after every directory passed validation, so a filesystem failure
  // cannot leave Electron half-configured and silently mixing isolated and real paths.
  for (const name of ISOLATED_ELECTRON_PATH_NAMES) appPaths.setPath(name, plan.paths[name])
}

export interface LegacyUserDataMigrationOptions {
  isolationPlan: IsolatedElectronPathPlan | null
  appName: string
  legacyAppNames: readonly string[]
  appPaths: {
    getPath(name: 'appData'): string
    setPath(name: 'userData', value: string): void
  }
  files: Pick<typeof fsPromises, 'stat' | 'mkdir' | 'cp'>
}

export async function migrateLegacyElectronUserData(options: LegacyUserDataMigrationOptions): Promise<'isolated' | 'migrated' | 'absent'> {
  // Crucially return before getPath('appData'), setPath(), stat(), mkdir(), or cp().
  // A configured VONTRA_HOME was already applied before main-app imported.
  if (options.isolationPlan) return 'isolated'
  const appData = options.appPaths.getPath('appData')
  const current = path.join(appData, options.appName)
  options.appPaths.setPath('userData', current)
  for (const name of options.legacyAppNames) {
    const legacy = path.join(appData, name)
    if (legacy === current) continue
    const info = await options.files.stat(legacy).catch(() => null)
    if (!info?.isDirectory()) continue
    await options.files.mkdir(current, { recursive: true })
    await options.files.cp(legacy, current, { recursive: true, force: false, errorOnExist: false }).catch(() => {})
    return 'migrated'
  }
  return 'absent'
}

function ensurePrivateDirectory(directory: string, io: PrivateDirectoryIo): void {
  let before
  try { before = io.lstatSync(directory) } catch (error) {
    if ((error as NodeJS.ErrnoException)?.code !== 'ENOENT') throw error
    io.mkdirSync(directory, { recursive: true, mode: 0o700 })
    before = io.lstatSync(directory)
  }
  // Inspect before chmod: chmod follows symlinks and could otherwise mutate an
  // attacker-selected target outside VONTRA_HOME.
  if (!before.isDirectory() || before.isSymbolicLink()) {
    throw new Error(`Refusing non-directory or symlinked isolated Electron path: ${directory}`)
  }
  io.chmodSync(directory, 0o700)
  const secured = io.lstatSync(directory)
  if (!secured.isDirectory() || secured.isSymbolicLink() || secured.dev !== before.dev || secured.ino !== before.ino) {
    throw new Error(`Isolated Electron path identity changed while securing it: ${directory}`)
  }
  if ((secured.mode & 0o777) !== 0o700) throw new Error(`Isolated Electron path is not mode 0700: ${directory}`)
  if (io.uid !== null && secured.uid !== io.uid) throw new Error(`Isolated Electron path is not owned by the current user: ${directory}`)
}
