#!/usr/bin/env node
import { execFile, spawn } from 'node:child_process'
import { chmod, lstat, mkdir, mkdtemp, readFile, readdir, realpath, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { assertProductionMacSigner, assertProductionUpdateTrustRoot, readReleaseTrustConfig } from './update-signing-trust.mjs'
import { assertFuses, assertStagedEngine, verifyPackagedComputerHelper, verifyPackagedEngine } from './package-checks.mjs'
import { assertPackagedMacOSMinimum } from './macos-runtime-minimum.mjs'

const exec = promisify(execFile)
const CODESIGN = '/usr/bin/codesign'
const DITTO = '/usr/bin/ditto'
const HDIUTIL = '/usr/bin/hdiutil'
const MOUNT = '/sbin/mount'
const PLUTIL = '/usr/bin/plutil'
const SPCTL = '/usr/sbin/spctl'
const XCRUN = '/usr/bin/xcrun'

export function validateMacReleaseEnvironment(env, expectedSigner) {
  const expectedIdentity = String(expectedSigner?.identity || '').trim()
  const expectedTeamId = String(expectedSigner?.teamId || '').trim()
  if (!expectedSigner?.productionReady || !expectedIdentity || !/^[A-Z0-9]{10}$/.test(expectedTeamId)) {
    throw new Error('Release packaging requires a production-ready repository-pinned macOS signer.')
  }
  const identity = String(env.CSC_NAME || '').trim()
  const identityMatch = /^Developer ID Application: .+ \(([A-Z0-9]{10})\)$/.exec(identity)
  if (!identityMatch) {
    throw new Error('Release packaging requires CSC_NAME="Developer ID Application: Legal Name (TEAMID)"; undefined, ad-hoc, Apple Development, and generic identities are refused.')
  }
  const teamId = String(env.APPLE_TEAM_ID || '').trim()
  if (!teamId) throw new Error('Release packaging requires APPLE_TEAM_ID to match the repository-pinned Developer ID team.')
  if (identity !== expectedIdentity || identityMatch[1] !== expectedTeamId || teamId !== expectedTeamId) {
    throw new Error(`CSC_NAME and APPLE_TEAM_ID must exactly match the repository-pinned macOS signer (${expectedIdentity}, team ${expectedTeamId}).`)
  }
  if (String(env.CSC_IDENTITY_AUTO_DISCOVERY || '').toLowerCase() === 'false') {
    throw new Error('Release packaging cannot disable code-signing identity discovery.')
  }

  const apiKey = all(env, ['APPLE_API_KEY', 'APPLE_API_KEY_ID', 'APPLE_API_ISSUER'])
  const keychainProfile = String(env.APPLE_KEYCHAIN_PROFILE || '').trim()
  if (env.APPLE_ID || env.APPLE_APP_SPECIFIC_PASSWORD) {
    throw new Error('Release packaging refuses raw Apple ID notarization credentials because notarytool and build failures can expose the app-specific password in process arguments or logs. Pre-store credentials in a notarytool keychain profile (APPLE_KEYCHAIN_PROFILE, optional APPLE_KEYCHAIN) or use the App Store Connect API-key variables.')
  }
  if (!keychainProfile && !apiKey) {
    throw new Error('Release packaging requires notarization credentials: APPLE_KEYCHAIN_PROFILE (and optional APPLE_KEYCHAIN), or APPLE_API_KEY + APPLE_API_KEY_ID + APPLE_API_ISSUER.')
  }
  return { identity, teamId, notarization: keychainProfile ? 'keychain-profile' : 'api-key' }
}

export async function validateMacReleasePreflight(env, root = fileURLToPath(new URL('..', import.meta.url))) {
  const trustConfig = await readReleaseTrustConfig(root)
  const updateTrust = await assertProductionUpdateTrustRoot(root, trustConfig)
  const expectedSigner = await assertProductionMacSigner(root, trustConfig)
  const release = validateMacReleaseEnvironment(env, expectedSigner)
  return { ...release, updateTrust }
}

export async function validateMacPackagingPreflight(mode, env, root = fileURLToPath(new URL('..', import.meta.url))) {
  if (mode === 'dev') return null
  if (mode === 'release') return validateMacReleasePreflight(env, root)
  throw new Error('--mode must be dev or release.')
}

export function builderArgs(mode, targets, identity = '') {
  if (mode !== 'dev' && mode !== 'release') throw new Error('--mode must be dev or release.')
  const allowed = new Set(['dir', 'dmg', 'zip'])
  if (!targets.length || targets.some((target) => !allowed.has(target))) throw new Error('--targets must contain dir, dmg, and/or zip.')
  const args = ['--mac', ...targets, '--publish', 'never']
  // Flipping fuses breaks the unsigned dev build's ad-hoc signature, so dev re-signs it ad hoc.
  if (mode === 'dev') args.push('--config.forceCodeSigning=false', '--config.mac.identity=null', '--config.mac.notarize=false', '--config.electronFuses.resetAdHocDarwinSignature=true')
  else args.push('--config.forceCodeSigning=true', `--config.mac.identity=${identity.replace(/^Developer ID Application: /, '')}`, '--config.mac.notarize=true')
  return args
}

export function expectedMacArtifacts(outputDir, metadata) {
  const productName = String(metadata.productName || '').trim()
  const version = String(metadata.version || '').trim()
  const arch = String(metadata.arch || '').trim()
  if (!productName || !version || !arch) throw new Error('Product name, version, and architecture are required to scope release artifacts.')
  if ([productName, version, arch].some((value) => /[\\/\0\r\n]/.test(value) || value === '.' || value === '..')) {
    throw new Error('Product name, version, and architecture must be safe artifact-name components.')
  }
  if (!/^[0-9A-Za-z][0-9A-Za-z._+-]{0,127}$/.test(version)) throw new Error('Version contains unsupported artifact-name characters.')
  if (!['arm64', 'x64', 'universal'].includes(arch)) throw new Error('Architecture must be arm64, x64, or universal.')
  const appFolder = arch === 'x64' ? 'mac' : `mac-${arch}`
  const stem = `${productName}-${version}-${arch}`
  return {
    apps: [path.join(outputDir, appFolder, `${productName}.app`)],
    dmgs: [path.join(outputDir, `${stem}.dmg`)],
    zips: [path.join(outputDir, `${stem}.zip`)],
  }
}

export async function cleanExpectedMacArtifacts(artifacts, targets) {
  const selected = [
    ...artifacts.apps,
    ...(targets.includes('dmg') ? artifacts.dmgs : []),
    ...(targets.includes('zip') ? artifacts.zips : []),
  ]
  await Promise.all(selected.map((artifact) => rm(artifact, { recursive: true, force: true })))
}

export async function verifyMacReleaseArtifacts(outputDir, targets, metadata, release, run = exec) {
  const expectedIdentity = String(release?.identity || '').trim()
  const expectedTeamId = String(release?.teamId || '').trim()
  const identityMatch = /^Developer ID Application: .+ \(([A-Z0-9]{10})\)$/.exec(expectedIdentity)
  if (!identityMatch || identityMatch[1] !== expectedTeamId) {
    throw new Error('Release verification requires the exact repository-pinned Developer ID Application identity and matching team ID.')
  }
  const allowedTargets = new Set(['dir', 'dmg', 'zip'])
  if (!targets.length || targets.some((target) => !allowedTargets.has(target))) throw new Error('Release verification targets must contain dir, dmg, and/or zip.')
  const expectedAppId = String(metadata?.appId || '').trim()
  if (!expectedAppId) throw new Error('Release verification requires the repository app ID.')
  const artifacts = expectedMacArtifacts(outputDir, metadata)
  const baseline = await verifySignedMacApp(artifacts.apps[0], metadata, release, run)
  if (targets.includes('dmg')) for (const dmg of artifacts.dmgs) await verifyDmgApp(dmg, metadata, release, baseline, run)
  if (targets.includes('zip')) for (const zip of artifacts.zips) await verifyZipApp(zip, metadata, release, baseline, run)
  return artifacts
}

async function verifySignedMacApp(app, metadata, release, run, containmentRoot = null) {
  await requireAppBundle(app, containmentRoot)
  await run(CODESIGN, ['--verify', '--deep', '--strict', '--verbose=2', app])
  const detail = await run(CODESIGN, ['--display', '--verbose=4', app])
  const signature = commandOutput(detail)
  const lines = signature.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
  const developerAuthority = lines.find((line) => line.startsWith('Authority=Developer ID Application:')) ?? ''
  const teamIdentifier = lines.find((line) => line.startsWith('TeamIdentifier=')) ?? ''
  const cdHashLines = lines.filter((line) => line.startsWith('CDHash='))
  const cdHash = cdHashLines[0]?.slice('CDHash='.length).trim().toLowerCase() ?? ''
  if (developerAuthority !== `Authority=${release.identity}` || teamIdentifier !== `TeamIdentifier=${release.teamId}` || lines.includes('Signature=adhoc')) {
    throw new Error(`${app} is not signed with the repository-pinned Developer ID identity (${release.identity}, team ${release.teamId}).`)
  }
  if (cdHashLines.length !== 1 || !/^(?:[a-f0-9]{40}|[a-f0-9]{64})$/.test(cdHash)) {
    throw new Error(`${app} did not expose exactly one valid signed-code CDHash.`)
  }
  await verifyPackagedEngine(app, { manifest: metadata.engine, release, run })
  await verifyPackagedComputerHelper(app, { release, run })

  const infoPlist = path.join(app, 'Contents', 'Info.plist')
  await requireArtifact(infoPlist, 'app Info.plist')
  const bundleIdentifier = await plistValue(infoPlist, 'CFBundleIdentifier', run)
  const shortVersion = await plistValue(infoPlist, 'CFBundleShortVersionString', run)
  const bundleVersion = await plistValue(infoPlist, 'CFBundleVersion', run)
  if (bundleIdentifier !== metadata.appId) throw new Error(`${app} has bundle identifier ${bundleIdentifier || '(missing)'}; expected ${metadata.appId}.`)
  if (shortVersion !== metadata.version) throw new Error(`${app} has version ${shortVersion || '(missing)'}; expected ${metadata.version}.`)
  if (!bundleVersion) throw new Error(`${app} has no CFBundleVersion.`)
  await assertPackagedMacOSMinimum(app, { run })

  const assessment = await run(SPCTL, ['--assess', '--type', 'execute', '--verbose=4', app])
  if (!/source=Notarized Developer ID/i.test(commandOutput(assessment))) {
    throw new Error(`${app} did not pass a notarized Developer ID Gatekeeper assessment.`)
  }
  return { app, cdHash, bundleIdentifier, shortVersion, bundleVersion }
}

async function verifyZipApp(zip, metadata, release, baseline, run) {
  await requireArtifact(zip, 'ZIP')
  return withTemporaryDirectory('vontra-release-zip-', async (extractionRoot) => {
    // `ditto` is the macOS-native ZIP extractor and preserves the bundle metadata/signatures that
    // must be assessed. Any invalid archive or extraction failure is fatal.
    await run(DITTO, ['-x', '-k', '--noqtn', zip, extractionRoot])
    const containedApp = path.join(extractionRoot, `${metadata.productName}.app`)
    await assertNoCompetingTopLevelApps(extractionRoot, path.basename(containedApp))
    const descriptor = await verifySignedMacApp(containedApp, metadata, release, run, extractionRoot)
    assertSameSignedApp(descriptor, baseline, `ZIP ${zip}`)
    return descriptor
  })
}

async function verifyDmgApp(dmg, metadata, release, baseline, run) {
  await requireArtifact(dmg, 'DMG')
  await run(XCRUN, ['stapler', 'validate', dmg])
  const temporaryRoot = await makePrivateTemporaryDirectory('vontra-release-dmg-')
  const mountPoint = path.join(temporaryRoot, 'mount')
  await mkdir(mountPoint, { mode: 0o700 })
  let attached = false
  let value
  let operationError = null
  let detachError = null
  try {
    await run(HDIUTIL, ['attach', '-readonly', '-nobrowse', '-noautoopen', '-owners', 'on', '-mountpoint', mountPoint, dmg])
    attached = true
    const containedApp = path.join(mountPoint, `${metadata.productName}.app`)
    await assertNoCompetingTopLevelApps(mountPoint, path.basename(containedApp))
    const descriptor = await verifySignedMacApp(containedApp, metadata, release, run, mountPoint)
    assertSameSignedApp(descriptor, baseline, `DMG ${dmg}`)
    value = descriptor
  } catch (error) {
    operationError = error
    // hdiutil can fail after the filesystem was attached (for example while reporting
    // metadata). Before deleting the mountpoint, ask the kernel whether it became active.
    // If that check itself fails, preserve the directory and fail closed for manual cleanup.
    if (!attached) {
      try {
        attached = await isActiveMountPoint(mountPoint, run)
      } catch (mountCheckError) {
        detachError = new Error(`Could not determine whether the failed DMG attach left ${mountPoint} mounted: ${errorMessage(mountCheckError)}`)
      }
    }
  }

  if (attached && !detachError) {
    try {
      await run(HDIUTIL, ['detach', mountPoint])
    } catch (firstError) {
      try {
        await run(HDIUTIL, ['detach', '-force', mountPoint])
      } catch (forceError) {
        detachError = new AggregateError([firstError, forceError], `Could not detach release-verification mount ${mountPoint}; it was left intact for safe manual cleanup.`)
      }
    }
  }
  if (attached && !detachError) {
    try {
      if (await isActiveMountPoint(mountPoint, run)) {
        detachError = new Error(`hdiutil reported success but release-verification mount ${mountPoint} is still active; it was left intact for safe manual cleanup.`)
      }
    } catch (mountCheckError) {
      detachError = new Error(`Could not confirm that release-verification mount ${mountPoint} detached: ${errorMessage(mountCheckError)}`)
    }
  }
  if (!detachError) {
    try {
      await rm(temporaryRoot, { recursive: true, force: true })
    } catch (error) {
      detachError = error
    }
  }
  if (operationError && detachError) throw new AggregateError([operationError, detachError], `DMG verification and cleanup both failed for ${dmg}.`)
  if (operationError) throw operationError
  if (detachError) throw detachError
  return value
}

async function withTemporaryDirectory(prefix, operation) {
  const temporaryRoot = await makePrivateTemporaryDirectory(prefix)
  let value
  let operationError = null
  try {
    value = await operation(temporaryRoot)
  } catch (error) {
    operationError = error
  }
  let cleanupError = null
  try {
    await rm(temporaryRoot, { recursive: true, force: true })
  } catch (error) {
    cleanupError = error
  }
  if (operationError && cleanupError) throw new AggregateError([operationError, cleanupError], `Release verification and temporary-directory cleanup both failed for ${temporaryRoot}.`)
  if (operationError) throw operationError
  if (cleanupError) throw cleanupError
  return value
}

async function makePrivateTemporaryDirectory(prefix) {
  const temporaryRoot = await mkdtemp(path.join(os.tmpdir(), prefix))
  try {
    await chmod(temporaryRoot, 0o700)
    const info = await lstat(temporaryRoot)
    if (!info.isDirectory() || info.isSymbolicLink() || (typeof process.getuid === 'function' && info.uid !== process.getuid())) {
      throw new Error('Could not create an owner-controlled release-verification directory.')
    }
    return temporaryRoot
  } catch (error) {
    await rm(temporaryRoot, { recursive: true, force: true }).catch(() => {})
    throw error
  }
}

async function isActiveMountPoint(mountPoint, run) {
  const canonical = await realpath(mountPoint)
  const mounted = commandOutput(await run(MOUNT, []))
  return mounted.split(/\r?\n/).some((line) => line.includes(` on ${canonical} (`))
}

function assertSameSignedApp(candidate, baseline, label) {
  for (const field of ['cdHash', 'bundleIdentifier', 'shortVersion', 'bundleVersion']) {
    if (candidate[field] !== baseline[field]) {
      throw new Error(`${label} does not contain the exact verified app: ${field} differs from ${baseline.app}.`)
    }
  }
}

async function assertNoCompetingTopLevelApps(root, expectedName) {
  const competing = (await readdir(root, { withFileTypes: true }))
    .map((entry) => entry.name)
    .filter((name) => name.toLowerCase().endsWith('.app') && name !== expectedName)
  if (competing.length) throw new Error(`Release archive contains unexpected top-level app bundle(s): ${competing.join(', ')}.`)
}

async function plistValue(infoPlist, key, run) {
  const result = await run(PLUTIL, ['-extract', key, 'raw', '-o', '-', infoPlist])
  return String(result?.stdout ?? '').trim()
}

function commandOutput(result) {
  return `${result?.stdout ?? ''}\n${result?.stderr ?? ''}`
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

export async function notarizeMacDmgArtifacts(outputDir, release, env, metadata, run = exec) {
  const { dmgs } = expectedMacArtifacts(outputDir, metadata)
  await requireArtifact(dmgs[0], 'DMG')
  for (const dmg of dmgs) {
    const auth = release.notarization === 'keychain-profile'
      ? [...(env.APPLE_KEYCHAIN ? ['--keychain', env.APPLE_KEYCHAIN] : []), '--keychain-profile', env.APPLE_KEYCHAIN_PROFILE]
      : ['--key', env.APPLE_API_KEY, '--key-id', env.APPLE_API_KEY_ID, '--issuer', env.APPLE_API_ISSUER]
    try {
      await run(XCRUN, ['notarytool', 'submit', dmg, '--wait', ...auth])
    } catch {
      // Never propagate execFile's error message: Node includes the full argv,
      // which can expose credentials if this function is ever called by a legacy
      // or injected runner. Supported production modes contain no raw password,
      // but keep the error boundary redacted as defense in depth.
      throw new Error(`Apple notarization submission failed for ${path.basename(dmg)}. Review the protected notarytool/CI logs.`)
    }
    await run(XCRUN, ['stapler', 'staple', dmg])
  }
  return dmgs
}

async function main() {
  if (process.platform !== 'darwin') throw new Error('macOS packaging must run on macOS.')
  const options = parseArgs(process.argv.slice(2))
  const mode = String(options.mode || '')
  const targets = String(options.targets || '').split(',').map((item) => item.trim()).filter(Boolean)
  const root = fileURLToPath(new URL('..', import.meta.url))
  // A notarized app that cannot authenticate its own update channel is not a releasable build.
  // Run this before cleaning outputs or spawning electron-builder.
  const release = await validateMacPackagingPreflight(mode, process.env, root)
  const engine = await assertStagedEngine(root)
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  const metadata = {
    productName: pkg?.build?.productName || pkg?.productName || pkg?.name,
    version: pkg?.version,
    arch: process.arch,
    appId: pkg?.build?.appId,
    engine,
  }
  const outputDir = path.join(root, pkg?.build?.directories?.output || 'release')
  const expected = expectedMacArtifacts(outputDir, metadata)
  const executable = path.join(root, 'node_modules', '.bin', 'electron-builder')
  const args = builderArgs(mode, targets, release?.identity)
  const env = mode === 'dev'
    ? { ...process.env, CSC_IDENTITY_AUTO_DISCOVERY: 'false' }
    : { ...process.env, CSC_NAME: release.identity.replace(/^Developer ID Application: /, '') }

  // A prior build with the same version must never satisfy this invocation's verification.
  // Remove only the exact current version/architecture outputs; unrelated historical
  // artifacts remain untouched.
  await cleanExpectedMacArtifacts(expected, targets)
  await new Promise((resolve, reject) => {
    const child = spawn(executable, args, { cwd: root, env, stdio: 'inherit' })
    child.once('error', reject)
    child.once('exit', (code, signal) => code === 0 ? resolve() : reject(new Error(`electron-builder failed (${signal || code}).`)))
  })
  await assertFuses(expected.apps[0], pkg?.build?.electronFuses)
  await assertPackagedMacOSMinimum(expected.apps[0])
  if (mode === 'release') {
    if (targets.includes('dmg')) await notarizeMacDmgArtifacts(outputDir, release, process.env, metadata)
    await verifyMacReleaseArtifacts(outputDir, targets, metadata, release)
    console.log(`Verified Developer ID signature, notarization and the bundled engine (${release.notarization}, team ${release.teamId}).`)
  } else {
    const { version } = await verifyPackagedEngine(expected.apps[0], { manifest: engine })
    await verifyPackagedComputerHelper(expected.apps[0])
    console.log(`Bundled ${version}; fuses match package.json.`)
    console.log('Created explicit unsigned development package; do not distribute it.')
  }
}

function parseArgs(argv) {
  const result = {}
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) continue
    const key = argv[i].slice(2)
    result[key] = argv[i + 1] && !argv[i + 1].startsWith('--') ? argv[++i] : '1'
  }
  return result
}

function all(env, names) {
  return names.every((name) => String(env[name] || '').trim())
}

async function requireArtifact(artifact, label, directory = false) {
  let info
  try { info = await lstat(artifact) } catch { throw new Error(`Release verification found no current ${label} at ${artifact}.`) }
  if (info.isSymbolicLink()) throw new Error(`Release verification refuses a symlinked ${label} at ${artifact}.`)
  if (directory) {
    if (!info.isDirectory()) throw new Error(`Release verification expected a ${label} directory at ${artifact}.`)
    return
  }
  if (!info.isFile() || info.size <= 0) throw new Error(`Release verification found an empty or invalid current ${label} at ${artifact}.`)
}

async function requireAppBundle(app, containmentRoot = null) {
  let info
  try { info = await lstat(app) } catch { throw new Error(`Release verification found no current app bundle at ${app}.`) }
  if (info.isSymbolicLink() || !info.isDirectory()) throw new Error(`Release verification expected a non-symlink app bundle directory at ${app}.`)
  if (containmentRoot) {
    const expectedName = path.basename(app)
    const topLevelApps = (await readdir(containmentRoot, { withFileTypes: true }))
      .filter((entry) => /\.app$/i.test(entry.name))
      .map((entry) => entry.name)
    if (topLevelApps.length !== 1 || topLevelApps[0] !== expectedName) {
      throw new Error(`Release archive must contain exactly one top-level ${expectedName} app bundle.`)
    }
    const canonicalRoot = await realpath(containmentRoot)
    const canonicalApp = await realpath(app)
    const relative = path.relative(canonicalRoot, canonicalApp)
    if (relative !== expectedName || path.isAbsolute(relative)) {
      throw new Error(`Release archive app escaped its private verification root: ${app}.`)
    }
  }
}

const invoked = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : ''
if (invoked === import.meta.url) {
  main().catch((error) => {
    console.error(`[package-mac] ${error?.message || error}`)
    process.exitCode = 1
  })
}
