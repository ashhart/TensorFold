import assert from 'node:assert/strict'
import { generateKeyPairSync } from 'node:crypto'
import { mkdir, mkdtemp, readFile, realpath, rm, stat, symlink, writeFile } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  builderArgs,
  cleanExpectedMacArtifacts,
  expectedMacArtifacts,
  notarizeMacDmgArtifacts,
  validateMacPackagingPreflight,
  validateMacReleaseEnvironment,
  validateMacReleasePreflight,
  verifyMacReleaseArtifacts,
} from './package-mac.mjs'
import { readReleaseTrustConfig } from './update-signing-trust.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))
const engineManifest = { version: '1.0.7', executables: ['bin/tensorfold-native', 'lib/libtfgrammar.dylib'] }
const identity = 'Developer ID Application: TensorFold Studio Test Fixture (AB12CD34EF)'
const teamId = 'AB12CD34EF'
const signer = { identity, teamId, productionReady: true }
const release = { identity, teamId, productionReady: true, notarization: 'keychain-profile' }
const validReleaseEnv = {
  CSC_NAME: identity,
  APPLE_TEAM_ID: teamId,
  APPLE_KEYCHAIN_PROFILE: 'vontra-notary',
}

// The checked-in metadata pins the release owner's signer and update root together. It is one
// canonical strict-JSON source for both runtime and preflight.
const checkedInTrust = await readReleaseTrustConfig(root)
assert.match(checkedInTrust.macos.identity, /^Developer ID Application: .+ \([A-Z0-9]{10}\)$/)
assert.ok(checkedInTrust.macos.identity.endsWith(`(${checkedInTrust.macos.teamId})`))
assert.equal(checkedInTrust.macos.productionReady, true)
assert.equal(checkedInTrust.update.productionReady, true)
assert.doesNotMatch(checkedInTrust.update.id, /placeholder|orphan/)

assert.throws(() => validateMacReleaseEnvironment({}, null), /production-ready repository-pinned macOS signer/)
assert.throws(() => validateMacReleaseEnvironment({}, signer), /requires CSC_NAME/)
for (const bad of ['-', 'adhoc', `Apple Development: TensorFold Studio Test Fixture (${teamId})`, 'Developer ID Application']) {
  assert.throws(() => validateMacReleaseEnvironment({ CSC_NAME: bad }, signer), /requires CSC_NAME/)
}
assert.throws(
  () => validateMacReleaseEnvironment({ CSC_NAME: identity, APPLE_TEAM_ID: teamId }, signer),
  /requires notarization credentials/,
)
assert.throws(
  () => validateMacReleaseEnvironment({ ...validReleaseEnv, APPLE_ID: 'release@example.test', APPLE_APP_SPECIFIC_PASSWORD: 'NOT_A_REAL_SECRET' }, signer),
  /refuses raw Apple ID notarization credentials/,
)
assert.equal(validateMacReleaseEnvironment(validReleaseEnv, signer).notarization, 'keychain-profile')
assert.throws(
  () => validateMacReleaseEnvironment({ ...validReleaseEnv, APPLE_TEAM_ID: 'ZZZZZZZZZZ' }, signer),
  /must exactly match the repository-pinned macOS signer/,
)
assert.throws(
  () => validateMacReleaseEnvironment({ ...validReleaseEnv, CSC_IDENTITY_AUTO_DISCOVERY: 'false' }, signer),
  /cannot disable code-signing identity discovery/,
)
assert.equal(validateMacReleaseEnvironment({
  CSC_NAME: identity,
  APPLE_TEAM_ID: teamId,
  APPLE_API_KEY: '/secrets/AuthKey.p8',
  APPLE_API_KEY_ID: 'KEY123',
  APPLE_API_ISSUER: 'issuer-uuid',
}, signer).notarization, 'api-key')

// The orphaned development update root that shipped before the release owner pinned a real one.
const ORPHAN_UPDATE = {
  id: 'vontra-update-ed25519-placeholder-orphan-2026-07',
  productionReady: false,
  ed25519PublicKey: '-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEAjVdaa7CV5FeFOo6NVqocd+85ZQmSScNe5t7muxGqmB4=\n-----END PUBLIC KEY-----',
}
const orphanRoot = await mkdtemp(path.join(os.tmpdir(), 'vontra-orphan-release-trust-'))
try {
  await mkdir(path.join(orphanRoot, 'build'), { recursive: true })
  await writeTrust(orphanRoot, { schemaVersion: 1, update: ORPHAN_UPDATE, macos: signer })
  await assert.rejects(
    () => validateMacReleasePreflight(validReleaseEnv, orphanRoot),
    /Production packaging is blocked.*PRODUCTION_READY is false.*placeholder\/orphan/i,
    'valid signing and notarization credentials cannot bypass an orphan update root',
  )
} finally {
  await rm(orphanRoot, { recursive: true, force: true })
}
assert.equal(await validateMacPackagingPreflight('dev', {}, root), null, 'explicit development packaging remains allowed with placeholder release metadata')

const readyRoot = await mkdtemp(path.join(os.tmpdir(), 'vontra-ready-release-trust-'))
try {
  await mkdir(path.join(readyRoot, 'build'), { recursive: true })
  const readyMac = { identity, teamId, productionReady: true }
  const relabelledOrphan = {
    schemaVersion: 1,
    update: {
      ...ORPHAN_UPDATE,
      id: 'vontra-update-ed25519-org-release-test-fixture',
      productionReady: true,
    },
    macos: readyMac,
  }
  await writeTrust(readyRoot, relabelledOrphan)
  await assert.rejects(
    () => validateMacReleasePreflight(validReleaseEnv, readyRoot),
    /known orphaned development key/i,
    'metadata cannot relabel the known orphan public-key bytes as production-ready',
  )

  const { publicKey: organizationPublicKey } = generateKeyPairSync('ed25519')
  const organizationPublicPem = organizationPublicKey.export({ type: 'spki', format: 'pem' }).toString().trim()
  const readyUpdate = {
    id: 'vontra-update-ed25519-org-release-test-fixture',
    productionReady: true,
    ed25519PublicKey: organizationPublicPem,
  }
  const PLACEHOLDER_MAC = { identity: 'Developer ID Application: REPLACE_WITH_ORGANIZATION_LEGAL_NAME (XXXXXXXXXX)', teamId: 'XXXXXXXXXX', productionReady: false }
  await writeTrust(readyRoot, { schemaVersion: 1, update: readyUpdate, macos: PLACEHOLDER_MAC })
  await assert.rejects(
    () => validateMacReleasePreflight(validReleaseEnv, readyRoot),
    /repository-pinned macOS signer.*PRODUCTION_READY is false.*placeholder/i,
    'a ready update root cannot bypass placeholder Developer ID metadata',
  )

  await writeTrust(readyRoot, { schemaVersion: 1, update: readyUpdate, macos: readyMac })
  const ready = await validateMacReleasePreflight(validReleaseEnv, readyRoot)
  assert.equal(ready.identity, identity)
  assert.equal(ready.teamId, teamId)
  assert.equal(ready.updateTrust.productionReady, true)
  assert.match(ready.updateTrust.fingerprint, /^[a-f0-9]{64}$/)

  await writeTrust(readyRoot, {
    schemaVersion: 1,
    update: readyUpdate,
    macos: { identity, teamId: 'ZZZZZZZZZZ', productionReady: true },
  })
  await assert.rejects(
    () => readReleaseTrustConfig(readyRoot),
    /identity and macos\.teamId must identify the same Apple team/,
    'checked-in identity and team metadata cannot disagree',
  )

  const duplicateKeySource = `${JSON.stringify({ schemaVersion: 1, update: readyUpdate, macos: readyMac }, null, 2).replace(
    '"schemaVersion": 1,',
    '"schemaVersion": 1,\n  "schemaVersion": 1,',
  )}\n`
  await writeFile(path.join(readyRoot, 'build/release-trust.json'), duplicateKeySource)
  await assert.rejects(
    () => readReleaseTrustConfig(readyRoot),
    /canonical strict-JSON serialization/,
    'duplicate JSON keys cannot create a shadow trust configuration',
  )

  await writeFile(path.join(readyRoot, 'build/release-trust.json'), '{"schemaVersion":1,')
  await assert.rejects(
    () => readReleaseTrustConfig(readyRoot),
    /must be strict JSON/,
    'malformed release metadata fails before packaging',
  )

  await writeTrust(readyRoot, { schemaVersion: 1, update: readyUpdate, macos: readyMac, decoyUpdatePublicKey: organizationPublicPem })
  await assert.rejects(
    () => readReleaseTrustConfig(readyRoot),
    /must contain exactly: schemaVersion, update, macos/,
    'unknown decoy metadata cannot impersonate an effective release field',
  )
} finally {
  await rm(readyRoot, { recursive: true, force: true })
}

assert.deepEqual(builderArgs('dev', ['dir']), [
  '--mac', 'dir', '--publish', 'never', '--config.forceCodeSigning=false', '--config.mac.identity=null', '--config.mac.notarize=false', '--config.electronFuses.resetAdHocDarwinSignature=true',
])
assert.deepEqual(builderArgs('release', ['dmg', 'zip'], identity), [
  '--mac', 'dmg', 'zip', '--publish', 'never', '--config.forceCodeSigning=true', `--config.mac.identity=${identity.replace(/^Developer ID Application: /, '')}`, '--config.mac.notarize=true',
])
assert.throws(() => expectedMacArtifacts('/tmp/output', { productName: '../escape', version: '1.0.0', arch: 'arm64' }), /safe artifact-name components/)
assert.throws(() => expectedMacArtifacts('/tmp/output', { productName: 'TensorFold Studio', version: '1.0.0', arch: 'mips' }), /Architecture must be/)

const tmp = await mkdtemp(path.join(os.tmpdir(), 'vontra-package-mac-'))
try {
  const metadata = { productName: 'TensorFold Studio', version: '1.0.0', arch: 'arm64', appId: 'com.vontra.app', engine: engineManifest }
  const expected = expectedMacArtifacts(tmp, metadata)
  const app = expected.apps[0]
  const dmg = expected.dmgs[0]
  const zip = expected.zips[0]
  await createFixtureApp(path.dirname(app), metadata)
  await writeFile(dmg, 'fixture')
  await writeFile(zip, 'fixture')

  // Stale artifacts from another release/architecture must never be verified, notarized,
  // mounted, extracted, or accepted as a substitute for the current invocation.
  const staleApp = path.join(tmp, 'mac', 'Old TensorFold-Studio.app')
  const staleDmg = path.join(tmp, 'TensorFold Studio-0.1.0-x64.dmg')
  await mkdir(staleApp, { recursive: true })
  await writeFile(staleDmg, 'stale')

  const normal = createFakeRun(metadata)
  const found = await verifyMacReleaseArtifacts(tmp, ['dmg', 'zip'], metadata, release, normal.run)
  assert.deepEqual(found.apps, [app])
  assert.deepEqual(found.dmgs, [dmg])
  assert.deepEqual(found.zips, [zip])
  assert.equal(normal.activeMounts.size, 0, 'successful DMG verification leaves no mount active')
  for (const temporaryRoot of normal.temporaryRoots) {
    await assert.rejects(() => stat(temporaryRoot), { code: 'ENOENT' }, 'verification temporary roots are removed')
  }
  assert.equal(normal.calls.some((call) => call.includes(staleApp) || call.includes(staleDmg)), false, 'stale release artifacts are out of scope')
  assert.ok(normal.calls.some((call) => tool(call[0]) === 'codesign' && call.includes('--strict')))
  assert.ok(normal.calls.some((call) => tool(call[0]) === 'xcrun' && call.includes('-show-build')), 'release verification inspects native OS requirements')
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, createFakeRun(metadata, { nativeMinimum: '14.0' }).run),
    /requires macOS 14.0, but the app declares 13.3/,
  )
  assert.ok(normal.calls.some((call) => tool(call[0]) === 'spctl'))
  assert.ok(normal.calls.some((call) => tool(call[0]) === 'xcrun' && call.includes('validate')))
  const attach = normal.calls.find((call) => tool(call[0]) === 'hdiutil' && call[1] === 'attach')
  assert.ok(attach?.includes('-readonly') && attach.includes('-nobrowse') && attach.includes('-noautoopen'), 'DMG is mounted read-only without UI side effects')
  assert.ok(normal.calls.some((call) => tool(call[0]) === 'hdiutil' && call[1] === 'detach'), 'DMG mount is detached')
  assert.ok(normal.calls.some((call) => tool(call[0]) === 'ditto' && call.includes('--noqtn')), 'ZIP is extracted with the native metadata-preserving path')

  for (const container of ['zip', 'dmg']) {
    const mismatch = createFakeRun(metadata, {
      cdHashFor: (candidate) => candidate.includes(`vontra-release-${container}-`) ? 'b'.repeat(40) : 'a'.repeat(40),
    })
    await assert.rejects(
      () => verifyMacReleaseArtifacts(tmp, [container], metadata, release, mismatch.run),
      /does not contain the exact verified app: cdHash differs/,
      `${container.toUpperCase()} must contain the exact already-verified app CDHash`,
    )
    assert.equal(mismatch.activeMounts.size, 0)
    for (const temporaryRoot of mismatch.temporaryRoots) await assert.rejects(() => stat(temporaryRoot), { code: 'ENOENT' })
  }

  for (const container of ['zip', 'dmg']) {
    const competing = createFakeRun(metadata, { competingApp: true })
    await assert.rejects(
      () => verifyMacReleaseArtifacts(tmp, [container], metadata, release, competing.run),
      /unexpected top-level app bundle|must contain exactly one top-level TensorFold Studio\.app app bundle/,
      `${container.toUpperCase()} cannot smuggle a competing top-level app beside the expected bundle`,
    )
    assert.equal(competing.activeMounts.size, 0)
    for (const temporaryRoot of competing.temporaryRoots) await assert.rejects(() => stat(temporaryRoot), { code: 'ENOENT' })
  }

  const invalidZip = createFakeRun(metadata, { failZipExtraction: true })
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['zip'], metadata, release, invalidZip.run),
    /invalid ZIP fixture/,
    'an invalid ZIP is a release failure',
  )
  for (const temporaryRoot of invalidZip.temporaryRoots) {
    await assert.rejects(() => stat(temporaryRoot), { code: 'ENOENT' }, 'invalid ZIP extraction roots are still removed')
  }

  const partialAttach = createFakeRun(metadata, { failAfterAttach: true })
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['dmg'], metadata, release, partialAttach.run),
    /hdiutil attach failed after mounting/,
    'an attach command that fails after mounting remains a packaging failure',
  )
  assert.equal(partialAttach.activeMounts.size, 0, 'a partially successful attach is detected through /sbin/mount and detached')
  assert.ok(partialAttach.calls.some((call) => tool(call[0]) === 'mount'), 'kernel mount state is checked after partial attach')
  assert.ok(partialAttach.calls.some((call) => tool(call[0]) === 'hdiutil' && call[1] === 'detach'), 'partial attach is detached')
  for (const temporaryRoot of partialAttach.temporaryRoots) await assert.rejects(() => stat(temporaryRoot), { code: 'ENOENT' })

  const detachFailure = createFakeRun(metadata, { failDetach: true })
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['dmg'], metadata, release, detachFailure.run),
    /Could not detach release-verification mount/,
    'normal and force detach failure is surfaced instead of recursively deleting through a live mount',
  )
  assert.equal(detachFailure.activeMounts.size, 1, 'failed detach remains represented as active')
  for (const temporaryRoot of detachFailure.temporaryRoots) {
    await stat(temporaryRoot) // fail-closed cleanup deliberately preserves an un-detachable mount root
    detachFailure.activeMounts.clear() // fake-only mount bookkeeping; no real mount exists in this regression
    await rm(temporaryRoot, { recursive: true, force: true })
  }

  const wrongIdentity = createFakeRun(metadata, { identityFor: () => 'Developer ID Application: Different Org (ZZZZZZZZZZ)' })
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, wrongIdentity.run),
    /not signed with the repository-pinned Developer ID identity/i,
    'a valid Developer ID signature from a different organization cannot satisfy the configured release identity',
  )
  const wrongTeam = createFakeRun(metadata, { teamFor: () => 'ZZZZZZZZZZ' })
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, wrongTeam.run),
    /not signed with the repository-pinned Developer ID identity/i,
    'codesign TeamIdentifier must independently match the configured release team',
  )
  const duplicateCdHash = createFakeRun(metadata, { duplicateCdHash: true })
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, duplicateCdHash.run),
    /exactly one valid signed-code CDHash/,
    'ambiguous observed CDHash output fails closed',
  )

  const dmgCalls = []
  await notarizeMacDmgArtifacts(tmp, { notarization: 'api-key' }, {
    APPLE_API_KEY: '/secrets/AuthKey.p8', APPLE_API_KEY_ID: 'KEY123', APPLE_API_ISSUER: 'issuer-uuid',
  }, metadata, async (command, args) => { dmgCalls.push([command, ...args]); return { stdout: '', stderr: '' } })
  assert.ok(dmgCalls.some((call) => tool(call[0]) === 'xcrun' && call[1] === 'notarytool' && call.includes('--wait')))
  assert.ok(dmgCalls.some((call) => tool(call[0]) === 'xcrun' && call[1] === 'stapler' && call[2] === 'staple'))
  assert.equal(dmgCalls.some((call) => call.includes(staleDmg)), false, 'stale DMGs are never submitted for notarization')

  const profileCalls = []
  await notarizeMacDmgArtifacts(tmp, { notarization: 'keychain-profile' }, {
    APPLE_KEYCHAIN: '/Users/release/Library/Keychains/release.keychain-db',
    APPLE_KEYCHAIN_PROFILE: 'vontra-notary',
  }, metadata, async (command, args) => { profileCalls.push([command, ...args]); return { stdout: '', stderr: '' } })
  const profileSubmit = profileCalls.find((call) => tool(call[0]) === 'xcrun' && call[1] === 'notarytool')
  assert.ok(profileSubmit?.includes('--keychain-profile') && profileSubmit.includes('vontra-notary'))
  assert.equal(profileSubmit?.includes('--password'), false, 'notarization never places an app-specific password in argv')

  await assert.rejects(
    () => notarizeMacDmgArtifacts(tmp, { notarization: 'keychain-profile' }, {
      APPLE_KEYCHAIN_PROFILE: 'vontra-notary',
    }, metadata, async () => { throw new Error('Command failed: xcrun notarytool --password NOT_A_REAL_SECRET') }),
    (error) => error instanceof Error && /Apple notarization submission failed/.test(error.message) && !error.message.includes('NOT_A_REAL_SECRET'),
    'notarytool failures cross a fixed redacted error boundary',
  )

  await rm(zip)
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['zip'], metadata, release, createFakeRun(metadata).run),
    /no current ZIP/,
    'requested ZIP target must actually exist for this version and architecture',
  )
  await symlink(staleDmg, zip)
  await assert.rejects(
    () => verifyMacReleaseArtifacts(tmp, ['zip'], metadata, release, createFakeRun(metadata).run),
    /refuses a symlinked ZIP/,
    'a symlink cannot substitute an expected release artifact',
  )
  await rm(zip)
  await writeFile(zip, 'fixture')

  await cleanExpectedMacArtifacts(expected, ['dmg', 'zip'])
  await assert.rejects(() => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, createFakeRun(metadata).run), /no current app bundle/)
  await stat(staleDmg) // cleanup leaves stale historical artifacts alone

  // The bundled engine is part of the release gate: exact signer, team, hardened runtime, version.
  await createFixtureApp(path.dirname(app), metadata)
  const engineCalls = normal.calls.filter((call) => call.some((arg) => typeof arg === 'string' && arg.includes('/Contents/Resources/engine/')))
  assert.ok(engineCalls.some((call) => tool(call[0]) === 'codesign' && call.includes('--strict') && call.at(-1).endsWith('bin/tensorfold-native')), 'engine binary gets codesign --verify --strict')
  assert.ok(engineCalls.some((call) => tool(call[0]) === 'codesign' && call.includes('--strict') && call.at(-1).endsWith('lib/libtfgrammar.dylib')), 'grammar dylib gets codesign --verify --strict')
  assert.ok(engineCalls.some((call) => call[0].endsWith('bin/tensorfold-native') && call[1] === '--version'), 'engine version is read back')
  const engineTeam = createFakeRun(metadata, { teamFor: (candidate) => candidate.includes('/engine/') ? 'ZZZZZZZZZZ' : teamId })
  await assert.rejects(() => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, engineTeam.run), /Bundled engine file bin\/tensorfold-native is not signed with the repository-pinned/)
  const noRuntime = createFakeRun(metadata, { runtimeFlag: false })
  await assert.rejects(() => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, noRuntime.run), /not signed with the hardened runtime/)
  const oldEngine = createFakeRun(metadata, { engineVersion: 'tensorfold-native 1.0.6' })
  await assert.rejects(() => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, oldEngine.run), /expected tensorfold-native 1.0.7/)
  await assert.rejects(() => verifyMacReleaseArtifacts(tmp, ['dir'], { ...metadata, engine: undefined }, release, createFakeRun(metadata).run), /requires the pinned engine manifest/)
  assert.ok(normal.calls.some((call) => tool(call[0]) === 'codesign' && call.includes('--strict') && String(call.at(-1)).endsWith('computer-use/tensorfold-computer')), 'computer-use helper gets codesign --verify --strict')
  await assert.rejects(() => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, createFakeRun(metadata, { helperSelfTest: '{"ok":false}\n' }).run), /computer-use helper failed its self-test/)
  await rm(path.join(app, 'Contents', 'Resources', 'engine', 'lib', 'libtfgrammar.dylib'))
  await assert.rejects(() => verifyMacReleaseArtifacts(tmp, ['dir'], metadata, release, createFakeRun(metadata).run), /missing the bundled engine file lib\/libtfgrammar.dylib/)
} finally {
  await rm(tmp, { recursive: true, force: true })
}

console.log('mac release gate tests passed')

async function writeTrust(targetRoot, trust) {
  await writeFile(path.join(targetRoot, 'build/release-trust.json'), `${JSON.stringify(trust, null, 2)}\n`)
}

async function createFixtureApp(parent, metadata) {
  const app = path.join(parent, `${metadata.productName}.app`)
  await mkdir(path.join(app, 'Contents'), { recursive: true })
  await writeFile(path.join(app, 'Contents', 'Info.plist'), 'fixture plist')
  for (const rel of engineManifest.executables) {
    const file = path.join(app, 'Contents', 'Resources', 'engine', rel)
    await mkdir(path.dirname(file), { recursive: true })
    await writeFile(file, Buffer.concat([Buffer.from('cffaedfe', 'hex'), Buffer.from('fixture engine file')]))
  }
  const helper = path.join(app, 'Contents', 'Resources', 'computer-use', 'tensorfold-computer')
  await mkdir(path.dirname(helper), { recursive: true })
  await writeFile(helper, 'fixture computer-use helper')
  return app
}

function createFakeRun(metadata, options = {}) {
  const calls = []
  const temporaryRoots = new Set()
  const activeMounts = new Set()
  const run = async (command, args) => {
    calls.push([command, ...args])
    const name = tool(command)
    if (name === 'hdiutil' && args[0] === 'attach') {
      const mountPoint = args[args.indexOf('-mountpoint') + 1]
      temporaryRoots.add(path.dirname(mountPoint))
      await createFixtureApp(mountPoint, metadata)
      if (options.competingApp) {
        await mkdir(path.join(mountPoint, 'Competing.app', 'Contents'), { recursive: true })
      }
      activeMounts.add(await realpath(mountPoint))
      if (options.failAfterAttach) throw new Error('hdiutil attach failed after mounting')
      return { stdout: '', stderr: '' }
    }
    if (name === 'hdiutil' && args[0] === 'detach') {
      if (options.failDetach) throw new Error('fixture detach failed')
      activeMounts.delete(await realpath(args.at(-1)))
      return { stdout: '', stderr: '' }
    }
    if (name === 'mount') {
      return {
        stdout: [...activeMounts].map((mountPoint) => `fixture@/dev/disk-test on ${mountPoint} (apfs, local, read-only, nobrowse)`).join('\n'),
        stderr: '',
      }
    }
    if (name === 'ditto') {
      const extractionRoot = args.at(-1)
      temporaryRoots.add(extractionRoot)
      if (options.failZipExtraction) throw new Error('invalid ZIP fixture')
      await createFixtureApp(extractionRoot, metadata)
      if (options.competingApp) {
        await mkdir(path.join(extractionRoot, 'Competing.app', 'Contents'), { recursive: true })
      }
      return { stdout: '', stderr: '' }
    }
    if (name === 'codesign' && args.includes('--display')) {
      const candidate = args.at(-1)
      const observedIdentity = options.identityFor?.(candidate) ?? identity
      const observedTeam = options.teamFor?.(candidate) ?? teamId
      const cdHash = options.cdHashFor?.(candidate) ?? 'a'.repeat(40)
      const duplicate = options.duplicateCdHash ? `CDHash=${'b'.repeat(40)}\n` : ''
      const flags = options.runtimeFlag === false ? '0x0(none)' : '0x10000(runtime)'
      return { stdout: '', stderr: `CodeDirectory v=20500 size=100 flags=${flags} hashes=1+7 location=embedded\nAuthority=${observedIdentity}\nTeamIdentifier=${observedTeam}\nCDHash=${cdHash}\n${duplicate}` }
    }
    if (name === 'tensorfold-computer' && args[0] === '--self-test') return { stdout: options.helperSelfTest ?? '{"ok":true,"checks":217}\n', stderr: '' }
    if (name === 'tensorfold-native' && args[0] === '--version') return { stdout: `${options.engineVersion ?? 'tensorfold-native 1.0.7'}\n`, stderr: '' }
    if (name === 'plutil') {
      const key = args[1]
      const values = {
        CFBundleIdentifier: metadata.appId,
        CFBundleShortVersionString: metadata.version,
        CFBundleVersion: '100',
        LSMinimumSystemVersion: '13.3',
      }
      return { stdout: `${values[key]}\n`, stderr: '' }
    }
    if (name === 'xcrun' && args[0] === 'vtool') return { stdout: `Load command 1\n cmd LC_BUILD_VERSION\n platform MACOS\n minos ${options.nativeMinimum ?? '13.3'}\n sdk 26.5\n`, stderr: '' }
    if (name === 'spctl') return { stdout: '', stderr: 'accepted\nsource=Notarized Developer ID\n' }
    return { stdout: '', stderr: '' }
  }
  return { run, calls, temporaryRoots, activeMounts }
}

function tool(command) {
  return path.basename(command)
}
