// Read-only preflight for Squirrel's global per-user namespace; VONTRA_HOME does not isolate it.
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import { execFile } from 'node:child_process'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

const execute = promisify(execFile)
const APP_ID = 'dev.tensorfold.studio'

async function exists(filename) {
  try { await fs.lstat(filename); return true } catch (error) { if (error.code === 'ENOENT') return false; throw error }
}

async function readDirectory(filename) {
  try { return await fs.readdir(filename) } catch (error) { if (error.code === 'ENOENT') return []; throw error }
}

async function readDomain(run, args) {
  try { await run('/usr/bin/defaults', args); return true } catch (error) {
    if (/does not exist|not found/i.test(error.stderr ?? '')) return false
    throw new Error('Cannot verify that the native updater preference domain is absent; smoke refused.')
  }
}

export async function captureNativeUpdateNamespace(deps = {}) {
  const taskHome = deps.taskHome ?? os.homedir(), run = deps.run ?? execute
  const library = path.join(taskHome, 'Library')
  const byHost = await readDirectory(path.join(library, 'Preferences', 'ByHost'))
  let activeShipItJob = false
  try { await run('/bin/launchctl', ['print', `gui/${process.getuid()}/${APP_ID}.ShipIt`]); activeShipItJob = true } catch (error) {
    if (!/could not find.*service|not found|no such process/i.test(error.stderr ?? '')) throw new Error('Cannot verify that the native updater job is absent; smoke refused.')
  }
  const processes = await run('/bin/ps', ['-axo', 'pid=,comm='])
  const studioProcesses = processes.stdout.split('\n').flatMap((line) => {
    const match = /^\s*(\d+)\s+(.*)$/.exec(line)
    return match && (/^TensorFold(?: Studio)?$/.test(match[2]) || /\/Contents\/MacOS\/TensorFold(?: Studio)?$/.test(match[2])) ? [Number(match[1])] : []
  })
  return {
    appCacheExists: await exists(path.join(library, 'Caches', APP_ID)),
    shipItCacheExists: await exists(path.join(library, 'Caches', `${APP_ID}.ShipIt`)),
    shipItPreferenceFileExists: await exists(path.join(library, 'Preferences', `${APP_ID}.ShipIt.plist`)),
    shipItByHostFiles: byHost.filter((name) => name.startsWith(`${APP_ID}.ShipIt.`) && name.endsWith('.plist')).length,
    shipItPreferenceDomainExists: await readDomain(run, ['read', `${APP_ID}.ShipIt`]),
    shipItCurrentHostDomainExists: await readDomain(run, ['-currentHost', 'read', `${APP_ID}.ShipIt`]),
    activeShipItJob,
    studioProcesses,
  }
}

export function assertNativeUpdateNamespaceClear(snapshot, ownPid = null) {
  if (snapshot.appCacheExists || snapshot.shipItCacheExists || snapshot.shipItPreferenceFileExists || snapshot.shipItByHostFiles || snapshot.shipItPreferenceDomainExists || snapshot.shipItCurrentHostDomainExists || snapshot.activeShipItJob || snapshot.studioProcesses.some((pid) => pid !== ownPid)) {
    throw new Error('An existing native Studio updater cache, state, job or another app instance is present; smoke will not consume, overwrite or reset it.')
  }
}

export async function captureCocoaPreferences(deps = {}) {
  const taskHome = deps.taskHome ?? os.homedir()
  const directory = path.join(taskHome, 'Library', 'Preferences')
  const filenames = [`${APP_ID}.plist`, ...(await readDirectory(path.join(directory, 'ByHost'))).filter((name) => name.startsWith(`${APP_ID}.`) && !name.startsWith(`${APP_ID}.ShipIt.`) && name.endsWith('.plist')).map((name) => path.join('ByHost', name))]
  const snapshot = {}
  for (const name of filenames.sort()) {
    const filename = path.join(directory, name)
    try {
      const info = await fs.lstat(filename, { bigint: true })
      if (!info.isFile() || info.isSymbolicLink() || info.uid !== BigInt(process.getuid())) throw new Error('Cannot safely snapshot existing Studio Cocoa preferences; smoke refused.')
      const bytes = await fs.readFile(filename)
      snapshot[name] = { sha256: createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length, device: String(info.dev), inode: String(info.ino), mode: String(info.mode), modified: String(info.mtimeNs) }
    } catch (error) { if (error.code !== 'ENOENT') throw error }
  }
  return snapshot
}

export async function assertCocoaPreferencesUnchanged(before, deps = {}) {
  const after = await captureCocoaPreferences(deps)
  if (JSON.stringify(before) !== JSON.stringify(after)) throw new Error('Existing Studio Cocoa/ByHost preferences changed; smoke stopped without restoring or resetting user data.')
}
