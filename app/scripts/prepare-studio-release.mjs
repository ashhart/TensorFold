#!/usr/bin/env node
import { spawnSync } from 'node:child_process'
import { readFile, stat } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { createPrivateKey } from 'node:crypto'
import { assertProductionMacSigner, assertProductionUpdateTrustRoot, assertSigningKeyMatchesTrustRoot, readReleaseTrustConfig } from './update-signing-trust.mjs'
import { packageWithXcodeAccount } from './package-mac-xcode.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))
if (process.platform !== 'darwin' || process.arch !== 'arm64') throw new Error('Prepare this release on an Apple silicon Mac.')
const trust = await readReleaseTrustConfig(root)
await assertProductionUpdateTrustRoot(root, trust)
const signer = await assertProductionMacSigner(root, trust)
const profile = process.env.APPLE_KEYCHAIN_PROFILE || 'tensorfold-notary'
if (!/^[a-zA-Z0-9_.-]{1,80}$/.test(profile)) throw new Error('Expected a local notarization keychain profile name.')
if (process.env.APPLE_ID || process.env.APPLE_APP_SPECIFIC_PASSWORD) throw new Error('Use a protected notarization keychain profile, not raw Apple credential environment variables.')

// Check credentials before packaging removes any previous local artifacts.
const args = ['notarytool', 'history', '--keychain-profile', profile, '--output-format', 'json', ...(process.env.APPLE_KEYCHAIN ? ['--keychain', process.env.APPLE_KEYCHAIN] : [])]
const credentials = spawnSync('xcrun', args, { encoding: 'utf8', timeout: 30000, maxBuffer: 1024 * 1024 })
const useXcode = credentials.status !== 0 || !!credentials.error
if (useXcode && process.env.APPLE_KEYCHAIN_PROFILE) throw new Error('The explicitly selected notarization profile is not usable; repair that profile or omit APPLE_KEYCHAIN_PROFILE to use the existing Xcode account.')
const key = path.join(os.homedir(), '.config', 'tensorfold-studio', 'release', 'update-signing-ed25519.pem')
if (!(await stat(key)).isFile()) throw new Error('The protected update-signing key is unavailable.')
assertSigningKeyMatchesTrustRoot(createPrivateKey(await readFile(key)), trust.update.ed25519PublicKey)
const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
const artifact = path.join(root, 'release', `${pkg.build.productName}-${pkg.version}-arm64.dmg`)
const updateZip = path.join(root, 'release', `${pkg.build.productName}-${pkg.version}-arm64.zip`)
const notes = path.join(root, 'docs', 'beta', `${pkg.version}.md`)
await stat(notes)
const env = { ...process.env, CSC_NAME: signer.identity, APPLE_TEAM_ID: signer.teamId, APPLE_KEYCHAIN_PROFILE: profile }
const npm = process.env.npm_execpath
if (!npm) throw new Error('Run this helper using npm run release:prepare.')
function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, env, stdio: 'inherit' })
  if (result.error || result.status !== 0) throw new Error('Release preparation stopped because a required check failed.')
}
run(process.execPath, [npm, 'run', 'verify'])
if (useXcode) {
  run(process.execPath, [npm, 'run', 'build'])
  run(process.execPath, [path.join(root, 'scripts', 'fetch-engine.mjs')])
  run(process.execPath, [path.join(root, 'scripts', 'build-computer-helper.mjs')])
  await packageWithXcodeAccount()
} else run(process.execPath, [npm, 'run', 'dist:mac'])
run(process.execPath, [npm, 'run', 'smoke:mac:packaged'])
run(process.execPath, [path.join(root, 'scripts', 'make-update-manifest.mjs'), '--version', pkg.version, '--artifact', artifact, '--download-url', `https://tensorfold.dev/studio/TensorFold-${pkg.version}-arm64.dmg`, '--update-zip', updateZip, '--update-zip-url', `https://tensorfold.dev/studio/TensorFold-${pkg.version}-arm64.zip`, '--notes', notes, '--out', path.join(root, 'release', 'update.json'), '--private-key', key])
console.log('Signed release and update manifest prepared locally; public hosting is the next release step.')
