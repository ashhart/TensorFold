#!/usr/bin/env node
import { spawn } from 'node:child_process'
import { chmod, copyFile, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { assertProductionMacSigner, assertProductionUpdateTrustRoot, readReleaseTrustConfig } from './update-signing-trust.mjs'
import { assertFuses, assertStagedEngine, verifyPackagedComputerHelper, verifyPackagedEngine } from './package-checks.mjs'
import { expectedMacArtifacts, verifyMacReleaseArtifacts } from './package-mac.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))
const xml = value => String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;')
const plist = body => `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>${body}</dict></plist>\n`
const string = (key, value) => `<key>${xml(key)}</key><string>${xml(value)}</string>`

export function archiveInfoPlist(metadata, signer, date = new Date()) {
  expectedMacArtifacts('/tmp', metadata)
  if (!/^Developer ID Application: .+ \([A-Z0-9]{10}\)$/.test(signer.identity) || !/^[A-Z0-9]{10}$/.test(signer.teamId) || !signer.identity.endsWith(`(${signer.teamId})`)) throw new Error('Archive requires the exact pinned Developer ID signer.')
  return plist(`<key>ArchiveVersion</key><integer>2</integer><key>CreationDate</key><date>${date.toISOString().replace(/\.\d{3}Z$/, 'Z')}</date>${string('Name', metadata.productName)}${string('SchemeName', metadata.productName)}<key>ApplicationProperties</key><dict>${string('ApplicationPath', `Applications/${metadata.productName}.app`)}${string('CFBundleIdentifier', metadata.appId)}${string('CFBundleShortVersionString', metadata.version)}${string('CFBundleVersion', metadata.version)}${string('SigningIdentity', signer.identity)}${string('Team', signer.teamId)}</dict>`)
}

export function xcodeExportOptionsPlist(signer) {
  return plist(`${string('method', 'developer-id')}${string('destination', 'upload')}${string('teamID', signer.teamId)}${string('signingStyle', 'manual')}${string('signingCertificate', signer.identity)}`)
}

export function isXcodeNotarizationProcessing(code, output) {
  return code === 65 && /error: Archive "[^"\r\n]+\.xcarchive" is processing and not ready for distribution\./.test(output)
}

export function isStaplerTicketPending(code, output) {
  return code === 65 && /Record not found/.test(output)
}

export async function packageWithXcodeAccount() {
  if (process.platform !== 'darwin' || process.arch !== 'arm64') throw new Error('Xcode release packaging requires Apple silicon macOS.')
  if (process.env.APPLE_ID || process.env.APPLE_APP_SPECIFIC_PASSWORD) throw new Error('Use the account already signed into Xcode; raw Apple credentials are refused.')
  const trust = await readReleaseTrustConfig(root)
  await assertProductionUpdateTrustRoot(root, trust)
  const signer = await assertProductionMacSigner(root, trust)
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  const metadata = { productName: pkg.build.productName, version: pkg.version, appId: pkg.build.appId, arch: 'arm64', engine: await assertStagedEngine(root) }
  const work = await mkdtemp('/tmp/studio-xcode-release-')
  await chmod(work, 0o700)
  const output = path.join(work, 'artifacts')
  const artifacts = expectedMacArtifacts(output, metadata)
  const env = { ...process.env, CSC_NAME: signer.identity.replace(/^Developer ID Application: /, '') }
  delete env.CSC_IDENTITY_AUTO_DISCOVERY
  function run(command, args, processingKind = null) {
    return new Promise((resolve, reject) => {
      const child = spawn(command, args, { cwd: root, env, stdio: processingKind ? ['ignore', 'pipe', 'pipe'] : 'inherit' })
      let output = ''
      if (processingKind) for (const stream of [child.stdout, child.stderr]) stream.on('data', chunk => {
        output = (output + chunk.toString()).slice(-65536)
        process.stdout.write(chunk)
      })
      child.once('error', reject)
      child.once('exit', (code, signal) => {
        if (code === 0) return resolve(true)
        if (processingKind === 'app' && isXcodeNotarizationProcessing(code, output) || processingKind === 'container' && isStaplerTicketPending(code, output)) return resolve(false)
        reject(new Error(`${path.basename(command)} failed (${signal || code}); private staging is preserved at ${work}.`))
      })
    })
  }
  const builder = path.join(root, 'node_modules', '.bin', 'electron-builder')
  const builderOptions = ['--publish', 'never', '--config.forceCodeSigning=true', `--config.mac.identity=${env.CSC_NAME}`, '--config.mac.notarize=false', `--config.directories.output=${output}`]
  // Signing without notarization is private staging, never a distributable result.
  await run(builder, ['--mac', 'dir', ...builderOptions])
  await assertFuses(artifacts.apps[0], pkg.build.electronFuses)
  await verifyPackagedEngine(artifacts.apps[0], { manifest: metadata.engine, release: signer })
  await verifyPackagedComputerHelper(artifacts.apps[0], { release: signer })
  const exportOptions = path.join(work, 'ExportOptions.plist')
  await writeFile(exportOptions, xcodeExportOptionsPlist(signer), { mode: 0o600 })
  async function archiveApp(name, includeDmg = false) {
    const archive = path.join(work, `${name}.xcarchive`)
    const app = path.join(archive, 'Products', 'Applications', `${metadata.productName}.app`)
    await mkdir(path.dirname(app), { recursive: true, mode: 0o700 })
    await run('/usr/bin/ditto', [artifacts.apps[0], app])
    if (includeDmg) {
      const resources = path.join(app, 'Contents', 'Resources', 'notarization-artifacts')
      await mkdir(resources, { mode: 0o700 })
      // Apple's notary service issues tickets for every nested container. This private carrier
      // lets Xcode's existing account submit the exact DMG; no account secret is extracted.
      // Copy container bytes only: DMG Finder/resource-fork attributes cannot be signed as
      // resources of the carrier app. The public DMG and its bytes remain unchanged.
      await copyFile(artifacts.dmgs[0], path.join(resources, path.basename(artifacts.dmgs[0])))
      await run('/usr/bin/codesign', ['--force', '--options', 'runtime', '--timestamp', '--entitlements', path.join(root, pkg.build.mac.entitlements), '--sign', signer.identity, app])
    }
    await run('/usr/bin/codesign', ['--verify', '--deep', '--strict', app])
    await writeFile(path.join(archive, 'Info.plist'), archiveInfoPlist(metadata, signer), { mode: 0o600 })
    const exported = path.join(work, `${name}-notarized`)
    await run('/usr/bin/xcodebuild', ['-exportArchive', '-archivePath', archive, '-exportOptionsPlist', exportOptions, '-exportPath', exported, '-allowProvisioningUpdates'])
    const deadline = Date.now() + 60 * 60_000
    const command = includeDmg ? '/usr/bin/xcrun' : '/usr/bin/xcodebuild'
    const args = includeDmg ? ['stapler', 'staple', artifacts.dmgs[0]] : ['-exportNotarizedApp', '-archivePath', archive, '-exportPath', exported]
    while (!await run(command, args, includeDmg ? 'container' : 'app')) {
      if (Date.now() >= deadline) throw new Error(`Apple has not finished notarization; private staging is preserved at ${work}.`)
      await new Promise(resolve => setTimeout(resolve, 30_000))
    }
    if (includeDmg) return null
    const notarized = path.join(exported, `${metadata.productName}.app`)
    await run('/usr/bin/xcrun', ['stapler', 'validate', notarized])
    return notarized
  }
  const notarizedApp = await archiveApp('Studio')
  // Xcode re-signs on export, so the exported notarized app becomes the release baseline.
  await rm(artifacts.apps[0], { recursive: true, force: true })
  // Keep signed bytes/ticket while omitting export-dialog file-access and resource xattrs.
  // macOS may regenerate its ordinary provenance marker; system protections stay enabled.
  await run('/usr/bin/ditto', ['--norsrc', '--noextattr', notarizedApp, artifacts.apps[0]])
  await assertFuses(artifacts.apps[0], pkg.build.electronFuses)
  await verifyMacReleaseArtifacts(output, ['dir'], metadata, signer)
  await run(builder, ['--prepackaged', artifacts.apps[0], '--mac', 'dmg', 'zip', ...builderOptions])
  await archiveApp('Studio-containers', true)
  await verifyMacReleaseArtifacts(output, ['dir', 'dmg', 'zip'], metadata, signer)
  const destinationRoot = path.join(root, pkg.build.directories.output)
  const destination = expectedMacArtifacts(destinationRoot, metadata)
  await mkdir(path.dirname(destination.apps[0]), { recursive: true })
  await rm(destination.apps[0], { recursive: true, force: true })
  await run('/usr/bin/ditto', [artifacts.apps[0], destination.apps[0]])
  await run('/usr/bin/ditto', [artifacts.dmgs[0], destination.dmgs[0]])
  await run('/usr/bin/ditto', [artifacts.zips[0], destination.zips[0]])
  await verifyMacReleaseArtifacts(destinationRoot, ['dir', 'dmg', 'zip'], metadata, signer)
  console.log(`Developer ID app, DMG and ZIP passed notarization, signature, bundled engine and fuse checks; private staging: ${work}`)
  return destination
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  packageWithXcodeAccount().catch(error => { console.error(error.message); process.exitCode = 1 })
}
