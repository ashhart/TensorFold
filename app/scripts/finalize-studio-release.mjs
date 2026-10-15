#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { createHash, createPrivateKey } from 'node:crypto'
import { copyFile, lstat, mkdir, readFile, realpath, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { assertProductionMacSigner, assertProductionUpdateTrustRoot, assertSigningKeyMatchesTrustRoot, readReleaseTrustConfig } from './update-signing-trust.mjs'
import { assertFuses } from './package-checks.mjs'
import { expectedMacArtifacts, verifyMacReleaseArtifacts } from './package-mac.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))

export function parseFinalizationArgs(argv) {
  if (argv.length !== 2 || argv[0] !== '--artifacts' || !argv[1] || argv[1].startsWith('--')) throw new Error('Use --artifacts <directory containing the notarized app, stapled DMG and ZIP>.')
  return path.resolve(argv[1])
}

export async function finalizeStudioRelease(sourceRoot) {
  if (process.platform !== 'darwin' || process.arch !== 'arm64') throw new Error('Finalize this release on Apple silicon macOS.')
  const info = await lstat(sourceRoot)
  if (!info.isDirectory() || info.isSymbolicLink() || process.getuid && info.uid !== process.getuid()) throw new Error('Artifact directory must be a locally owned real directory.')
  const trust = await readReleaseTrustConfig(root)
  await assertProductionUpdateTrustRoot(root, trust)
  const signer = await assertProductionMacSigner(root, trust)
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  const metadata = { productName: pkg.build.productName, version: pkg.version, arch: 'arm64', appId: pkg.build.appId, engine: JSON.parse(await readFile(path.join(root, 'build', 'engine.json'), 'utf8')) }
  const source = expectedMacArtifacts(await realpath(sourceRoot), metadata)
  await assertFuses(source.apps[0], pkg.build.electronFuses)
  await verifyMacReleaseArtifacts(sourceRoot, ['dir', 'dmg', 'zip'], metadata, signer)
  const keyFile = path.join(os.homedir(), '.config', 'tensorfold-studio', 'release', 'update-signing-ed25519.pem')
  const keyInfo = await lstat(keyFile)
  if (!keyInfo.isFile() || keyInfo.isSymbolicLink() || keyInfo.mode & 0o077 || process.getuid && keyInfo.uid !== process.getuid()) throw new Error('Update-signing key must remain in its owner-only protected file.')
  assertSigningKeyMatchesTrustRoot(createPrivateKey(await readFile(keyFile)), trust.update.ed25519PublicKey)
  const notes = path.join(root, 'docs', 'beta', `${pkg.version}.md`)
  await lstat(notes)
  const destinationRoot = path.join(root, pkg.build.directories.output)
  const destination = expectedMacArtifacts(destinationRoot, metadata)
  function run(command, args) {
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, { cwd: root, stdio: 'inherit' })
      child.once('error', reject)
      child.once('exit', code => code === 0 ? resolve() : reject(new Error(`${path.basename(command)} failed; finalization stopped.`)))
    })
  }
  const digest = async file => createHash('sha256').update(await readFile(file)).digest('hex')
  const hashes = await Promise.all([digest(source.dmgs[0]), digest(source.zips[0])])
  if (path.resolve(sourceRoot) !== path.resolve(destinationRoot)) {
    await mkdir(path.dirname(destination.apps[0]), { recursive: true })
    await rm(destination.apps[0], { recursive: true, force: true })
    await run('/usr/bin/ditto', ['--norsrc', '--noextattr', source.apps[0], destination.apps[0]])
    await copyFile(source.dmgs[0], destination.dmgs[0])
    await copyFile(source.zips[0], destination.zips[0])
  }
  if (await digest(destination.dmgs[0]) !== hashes[0] || await digest(destination.zips[0]) !== hashes[1]) throw new Error('Release artifacts changed during finalization.')
  await verifyMacReleaseArtifacts(destinationRoot, ['dir', 'dmg', 'zip'], metadata, signer)
  await run(process.execPath, [path.join(root, 'scripts', 'smoke-packaged-mac.mjs')])
  await run(process.execPath, [path.join(root, 'scripts', 'make-update-manifest.mjs'), '--version', pkg.version, '--artifact', destination.dmgs[0], '--download-url', `https://tensorfold.dev/studio/TensorFold-${pkg.version}-arm64.dmg`, '--update-zip', destination.zips[0], '--update-zip-url', `https://tensorfold.dev/studio/TensorFold-${pkg.version}-arm64.zip`, '--notes', notes, '--out', path.join(destinationRoot, 'update.json'), '--private-key', keyFile])
  console.log('Notarized release finalized and manifest signed locally; artifact/privacy clearance and hosting remain separate.')
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  finalizeStudioRelease(parseFinalizationArgs(process.argv.slice(2))).catch(error => { console.error(error.message); process.exitCode = 1 })
}
