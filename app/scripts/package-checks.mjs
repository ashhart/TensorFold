// Packaging checks that sit beside the release gate: the staged engine before electron-builder runs,
// the engine inside the packaged app afterwards, and the Electron fuse wire.
import { execFile } from 'node:child_process'
import { lstat, readFile, realpath } from 'node:fs/promises'
import path from 'node:path'
import { promisify } from 'node:util'
import { readEngineManifest, stagedEngineIsCurrent } from './fetch-engine.mjs'

const exec = promisify(execFile)
const CODESIGN = '/usr/bin/codesign'
const FUSE_SENTINEL = Buffer.from('dL7pKGdnNz796PbbjQWNKmHXBZaB9tsX', 'ascii')
// electron-builder's electronFuses keys in fuse-wire order (FuseV1Options 0..7).
export const FUSE_ORDER = [
  'runAsNode',
  'enableCookieEncryption',
  'enableNodeOptionsEnvironmentVariable',
  'enableNodeCliInspectArguments',
  'enableEmbeddedAsarIntegrityValidation',
  'onlyLoadAppFromAsar',
  'loadBrowserProcessSpecificV8Snapshot',
  'grantFileProtocolExtraPrivileges',
]

export async function assertStagedEngine(root) {
  const manifest = await readEngineManifest(root)
  if (!await stagedEngineIsCurrent(path.join(root, 'build'), manifest)) {
    throw new Error(`build/engine does not hold the verified TensorFold engine ${manifest.version}; run node scripts/fetch-engine.mjs first.`)
  }
  return manifest
}

export function packagedEnginePaths(app, manifest) {
  const engineRoot = path.join(app, 'Contents', 'Resources', 'engine')
  return { engineRoot, files: (manifest.executables ?? []).map((rel) => path.join(engineRoot, rel)), binary: path.join(engineRoot, 'bin', 'tensorfold-native') }
}

function output(result) {
  return `${result?.stdout ?? ''}\n${result?.stderr ?? ''}`
}

// A bundled executable carries the pinned Developer ID signer and team and the hardened runtime.
async function assertDeveloperIdSigned(file, label, release, run) {
  await run(CODESIGN, ['--verify', '--strict', '--verbose=2', file])
  const lines = output(await run(CODESIGN, ['--display', '--verbose=4', file])).split(/\r?\n/).map((line) => line.trim())
  const authority = lines.find((line) => line.startsWith('Authority=Developer ID Application:')) ?? ''
  const team = lines.find((line) => line.startsWith('TeamIdentifier=')) ?? ''
  if (authority !== `Authority=${release.identity}` || team !== `TeamIdentifier=${release.teamId}` || lines.includes('Signature=adhoc')) {
    throw new Error(`${label} is not signed with the repository-pinned Developer ID identity (${release.identity}, team ${release.teamId}).`)
  }
  if (!lines.some((line) => /^CodeDirectory .*flags=0x[0-9a-f]+\([^)]*\bruntime\b/.test(line))) {
    throw new Error(`${label} is not signed with the hardened runtime.`)
  }
}

// The computer-use helper is a regular file at Contents/Resources/computer-use, signed like the engine in
// release mode, and passes its own self-test (pure checks; it never sends input).
export async function verifyPackagedComputerHelper(app, { release = null, run = exec } = {}) {
  const helper = path.join(app, 'Contents', 'Resources', 'computer-use', 'tensorfold-computer')
  let info
  try { info = await lstat(helper) } catch { throw new Error(`${app} has no computer-use helper at Contents/Resources/computer-use.`) }
  if (info.isSymbolicLink() || !info.isFile()) throw new Error('The computer-use helper must be a regular file.')
  if (release) await assertDeveloperIdSigned(helper, 'The computer-use helper', release, run)
  const report = output(await run(helper, ['--self-test'])).trim().split(/\r?\n/).filter(Boolean).at(-1) ?? ''
  let parsed = null
  try { parsed = JSON.parse(report) } catch { /* reported below */ }
  if (parsed?.ok !== true) throw new Error(`The computer-use helper failed its self-test: ${report.slice(0, 120)}`)
  return { helper, checks: parsed.checks }
}

// The engine must sit at Contents/Resources/engine/{bin,lib} as real files, print the pinned version,
// and in release mode carry the exact Developer ID signer, team and hardened runtime.
export async function verifyPackagedEngine(app, { manifest, release = null, run = exec }) {
  if (!manifest?.version) throw new Error('Engine verification requires the pinned engine manifest.')
  const { engineRoot, files, binary } = packagedEnginePaths(app, manifest)
  const canonicalRoot = await realpath(engineRoot).catch(() => null)
  if (!canonicalRoot) throw new Error(`${app} has no bundled engine at Contents/Resources/engine.`)
  for (const file of files) {
    let info
    try { info = await lstat(file) } catch { throw new Error(`${app} is missing the bundled engine file ${path.relative(engineRoot, file)}.`) }
    if (info.isSymbolicLink() || !info.isFile()) throw new Error(`Bundled engine file ${path.relative(engineRoot, file)} must be a regular file.`)
    if (path.relative(canonicalRoot, await realpath(file)).startsWith('..')) throw new Error(`Bundled engine file ${file} escapes the engine folder.`)
  }
  if (release) {
    for (const file of files) await assertDeveloperIdSigned(file, `Bundled engine file ${path.relative(engineRoot, file)}`, release, run)
  }
  const version = output(await run(binary, ['--version'])).trim()
  if (version !== `tensorfold-native ${manifest.version}`) throw new Error(`Bundled engine reports "${version.slice(0, 80)}", expected tensorfold-native ${manifest.version}.`)
  return { engineRoot, version }
}

export async function readFuseWire(app) {
  const framework = path.join(app, 'Contents', 'Frameworks', 'Electron Framework.framework', 'Electron Framework')
  const bytes = await readFile(framework)
  const at = bytes.indexOf(FUSE_SENTINEL)
  if (at < 0) throw new Error(`No Electron fuse wire found in ${framework}.`)
  if (bytes.lastIndexOf(FUSE_SENTINEL) !== at) throw new Error('Universal Electron frameworks are not checked by this gate.')
  const position = at + FUSE_SENTINEL.length
  if (bytes[position] !== 1) throw new Error(`Unexpected Electron fuse wire version ${bytes[position]}.`)
  const length = bytes[position + 1]
  const wire = {}
  for (let i = 0; i < Math.min(length, FUSE_ORDER.length); i++) {
    const state = bytes[position + 2 + i]
    wire[FUSE_ORDER[i]] = state === 0x31 ? true : state === 0x30 ? false : state === 0x72 ? 'removed' : `unknown(${state})`
  }
  return wire
}

export async function assertFuses(app, expected) {
  if (!expected || typeof expected !== 'object') throw new Error('package.json build.electronFuses is required.')
  const wire = await readFuseWire(app)
  for (const [key, value] of Object.entries(expected)) {
    if (!FUSE_ORDER.includes(key)) continue
    if (wire[key] !== value) throw new Error(`Electron fuse ${key} reads ${wire[key]} in ${app}; expected ${value}.`)
  }
  return wire
}
