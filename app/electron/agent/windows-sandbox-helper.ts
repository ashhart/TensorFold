import { createHash } from 'node:crypto'
import { existsSync, readFileSync, statSync } from 'node:fs'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import os from 'node:os'

export interface WindowsSandboxHelper { binary: string; sha256: string }

export function verifyWindowsSandboxHelper(binary: string, manifestFile: string): WindowsSandboxHelper {
  const manifest = JSON.parse(readFileSync(manifestFile, 'utf8'))
  const stat = statSync(binary)
  if (!stat.isFile() || stat.size > 8 * 1024 * 1024 || manifest.abi !== 1 || manifest.platform !== 'win32' || manifest.arch !== 'x64') {
    throw new Error('Windows sandbox helper has an unsupported manifest or format.')
  }
  const bytes = readFileSync(binary)
  if (bytes.toString('ascii', 0, 2) !== 'MZ' || createHash('sha256').update(bytes).digest('hex') !== manifest.sha256) {
    throw new Error('Windows sandbox helper integrity verification failed.')
  }
  return { binary, sha256: manifest.sha256 }
}

export function windowsSandboxHelper(): WindowsSandboxHelper {
  const resources = (process as NodeJS.Process & { resourcesPath?: string }).resourcesPath
  if (resources && existsSync(path.join(resources, 'app.asar'))) {
    return verifyWindowsSandboxHelper(path.join(resources, 'sandbox/TensorFoldSandbox.exe'), path.join(resources, 'app.asar/sandbox-native.json'))
  }
  const candidates = ['../../build/sandbox/windows-x64', '../build/sandbox/windows-x64'].map(relative => path.resolve(__dirname, relative))
  const root = candidates.find(candidate => existsSync(path.join(candidate, 'manifest.json')))
  if (!root) throw new Error('Build the Windows sandbox helper before starting isolated commands.')
  return verifyWindowsSandboxHelper(path.join(root, 'TensorFoldSandbox.exe'), path.join(root, 'manifest.json'))
}

let probe: { sha256: string; available: boolean } | undefined
export function canUseWindowsSandbox(): boolean {
  if (process.platform !== 'win32' || process.arch !== 'x64') return false
  try {
    const helper = windowsSandboxHelper()
    if (probe?.sha256 === helper.sha256) return probe.available
    const result = spawnSync(helper.binary, ['--probe'], {
      timeout: 15000, encoding: 'utf8', maxBuffer: 64000, windowsHide: true,
      env: { SystemRoot: process.env.SystemRoot || 'C:\\Windows', TEMP: os.tmpdir(), TMP: os.tmpdir() },
    })
    const evidence = result.status === 0 ? JSON.parse(result.stdout) : null
    const available = evidence?.abi === 1 && evidence.appContainer === true && evidence.lessPrivileged === true
    probe = { sha256: helper.sha256, available }
    return available
  } catch { return false }
}
