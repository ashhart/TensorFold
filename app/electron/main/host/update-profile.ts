import { closeSync, constants, fstatSync, lstatSync, openSync, readSync, realpathSync } from 'node:fs'
import path from 'node:path'

export const PRIVATE_UPDATE_PROFILE_FILE = '.tensorfold-update-profile.json'
export const PRIVATE_UPDATE_PROFILE_PREFIX = 'vontra-packaged-smoke-'

export interface PrivateUpdateProfile { root: string; appBundle: string; debugPort: number | null }

// Squirrel relaunches via LaunchServices without our environment/arguments. Only an explicitly
// prepared private test-copy bundle can recover a non-default profile without that environment.
function validatePrivateUpdateProfile(execPath: string): PrivateUpdateProfile | null {
  let fd: number | null = null
  try {
    if (typeof process.getuid !== 'function') return null
    const appBundle = path.resolve(execPath, '..', '..', '..')
    if (!appBundle.endsWith('.app')) return null
    const parent = path.dirname(appBundle)
    if (!path.basename(parent).startsWith(PRIVATE_UPDATE_PROFILE_PREFIX) || path.dirname(parent) !== realpathSync('/tmp') || realpathSync(parent) !== parent) return null
    const directory = lstatSync(parent)
    if (!directory.isDirectory() || directory.isSymbolicLink() || directory.uid !== process.getuid() || (directory.mode & 0o777) !== 0o700) return null
    if (realpathSync(execPath) !== path.resolve(execPath) || !realpathSync(execPath).startsWith(`${appBundle}${path.sep}`) || !lstatSync(appBundle).isDirectory()) return null
    const marker = path.join(parent, PRIVATE_UPDATE_PROFILE_FILE)
    if (lstatSync(marker).isSymbolicLink()) return null
    fd = openSync(marker, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0))
    const stat = fstatSync(fd)
    if (!stat.isFile() || stat.uid !== process.getuid() || (stat.mode & 0o777) !== 0o600 || stat.size > 4096) return null
    const buffer = Buffer.alloc(4097)
    const size = readSync(fd, buffer, 0, buffer.length, 0)
    if (size > 4096) return null
    const markerData = JSON.parse(buffer.subarray(0, size).toString('utf8'))
    if (!markerData || Array.isArray(markerData) || Object.keys(markerData).sort().join(',') !== 'appBundle,debugPort,root,schemaVersion') return null
    if (markerData.schemaVersion !== 1 || markerData.root !== parent || markerData.appBundle !== appBundle || realpathSync(appBundle) !== appBundle) return null
    const debugPort = markerData.debugPort
    if (debugPort != null && (!Number.isInteger(debugPort) || debugPort < 1024 || debugPort > 65535)) return null
    return { root: parent, appBundle, debugPort: debugPort ?? null }
  } catch { return null } finally { if (fd != null) closeSync(fd) }
}

export function privateUpdateRelaunchProfile(execPath: string, requireRelaunchRecovery = false): PrivateUpdateProfile | null {
  const profile = validatePrivateUpdateProfile(execPath)
  let recognized = false
  try {
    const appBundle = path.resolve(execPath, '..', '..', '..'), parent = path.dirname(appBundle)
    recognized = appBundle.endsWith('.app') && path.basename(parent).startsWith(PRIVATE_UPDATE_PROFILE_PREFIX) && path.dirname(parent) === realpathSync('/tmp')
  } catch { /* ordinary apps have no private qualification profile */ }
  if (requireRelaunchRecovery && recognized && !profile) throw new Error('The private update profile could not be verified; TensorFold will not open another profile.')
  return profile
}
