import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { ORPHANED_DEVELOPMENT_UPDATE_TRUST_SPKI_SHA256, publicSpkiSha256, readReleaseTrustConfig } from './update-signing-trust.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))

async function main() {
  const pkg = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8'))
  assert.equal(pkg.main, 'dist-electron/main.js')
  assert.equal(pkg.packageManager, 'npm@10.9.8')
  assert.deepEqual(pkg.engines, { node: '>=22 <23', npm: '10.9.8' })
  assert.equal(pkg.dependencies?.undici, '7.30.0', 'DNS pinning must ship as a direct production dependency')
  assert.deepEqual(pkg.devEngines?.packageManager, { name: 'npm', version: '10.9.8', onFail: 'error' })
  assert.equal(pkg.build?.appId, 'dev.tensorfold.studio')
  assert.equal(pkg.build?.productName, 'TensorFold')
  assert.equal(pkg.build?.directories?.buildResources, 'build')
  assert.equal(pkg.build?.directories?.output, 'release')
  assert.equal(pkg.scripts?.preinstall, 'node scripts/check-package-manager.mjs')
  assert.equal(pkg.scripts?.build, 'npm run install:check && node scripts/clean-build-output.mjs && vite build', 'build directly refuses hybrid dependencies and removes stale output even when npm lifecycle hooks are disabled')
  assert.ok(pkg.scripts?.dev?.startsWith('npm run install:check &&'), 'dev directly enforces the package-manager guard')
  assert.ok(pkg.scripts?.start?.startsWith('npm run install:check &&'), 'start directly enforces the package-manager guard')
  assert.equal(pkg.scripts?.prebuild, undefined, 'security checks do not rely on suppressed npm pre-hooks')
  assert.ok(pkg.scripts?.['test:agent']?.startsWith('node scripts/check-package-manager.mjs'), 'direct guarded-suite runs also reject a hybrid dependency tree')
  assert.ok(pkg.scripts?.['pack:mac:dev']?.includes('--mode dev --targets dir'))
  assert.equal(pkg.scripts?.['smoke:mac:packaged'], 'node scripts/smoke-packaged-mac.mjs', 'packaged acceptance uses the profile-safe smoke harness')
  assert.ok(pkg.scripts?.['pack:mac']?.includes('--mode release --targets dir'))
  assert.ok(pkg.scripts?.['dist:mac']?.includes('--mode release --targets dmg,zip'))
  assert.ok(pkg.scripts?.['start:headless']?.includes('npm run build && electron . --headless-server'))
  assert.ok(pkg.scripts?.['release:manifest']?.includes('scripts/make-update-manifest.mjs'))
  assert.equal(pkg.scripts?.['deps:trusted-scripts'], 'node node_modules/electron/install.js && npm rebuild --ignore-scripts=false @lydell/node-pty', 'only the reviewed Electron downloader and package-scoped native rebuild may run explicitly')
  assert.deepEqual(pkg.build?.files, ['dist/**/*', 'dist-electron/**/*', 'package.json', '!node_modules/kokoro-js/dist/kokoro.web.js', '!node_modules/kokoro-js/dist/kokoro.web.js.map', '!node_modules/**/pino/pretty-demo.png', '!node_modules/**/.claude{,/**/*}'])
  assert.equal(pkg.name, 'tensorfold-studio')
  assert.equal(pkg.build?.asar, true)
  assert.equal(pkg.build?.forceCodeSigning, true)

  const releaseTrust = await readReleaseTrustConfig(root)
  // The beta establishes custody: the release owner's Developer ID and a fresh update root are pinned.
  assert.equal(releaseTrust.update.productionReady, true, 'the pinned update root is production-ready')
  assert.equal(releaseTrust.macos.productionReady, true, 'the pinned macOS signer is production-ready')
  assert.equal(/^Developer ID Application: .+ \(([A-Z0-9]{10})\)$/.exec(releaseTrust.macos.identity)?.[1], releaseTrust.macos.teamId, 'identity and Team ID name the same Apple team')
  assert.doesNotMatch(releaseTrust.macos.identity, /REPLACE_WITH|placeholder/i)
  assert.notEqual(publicSpkiSha256(releaseTrust.update.ed25519PublicKey), ORPHANED_DEVELOPMENT_UPDATE_TRUST_SPKI_SHA256, 'the orphaned development key is gone')

  const runtimeUpdateTrust = await readFile(path.join(root, 'electron/agent/update-trust.ts'), 'utf8')
  assert.match(runtimeUpdateTrust, /import releaseTrust from ['"]\.\.\/\.\.\/build\/release-trust\.json['"]/, 'runtime consumes the same strict trust object as release preflight')
  assert.doesNotMatch(runtimeUpdateTrust, /BEGIN PUBLIC KEY/, 'runtime source cannot carry a second effective public-key copy')

  const packageMac = await readFile(path.join(root, 'scripts/package-mac.mjs'), 'utf8')
  assert.match(packageMac, /readReleaseTrustConfig/, 'release preflight reads the strict shared trust configuration')
  assert.match(packageMac, /CSC_NAME and APPLE_TEAM_ID must exactly match the repository-pinned macOS signer/, 'release env cannot self-attest an arbitrary publisher')
  assert.match(packageMac, /CDHash/, 'release verification binds archive apps to the verified app code hash')
  assert.match(packageMac, /hdiutil[\s\S]*attach[\s\S]*-readonly/, 'release verification mounts DMGs read-only')
  assert.match(packageMac, /ditto[\s\S]*-x[\s\S]*-k/, 'release verification extracts ZIPs')

  const mac = pkg.build?.mac ?? {}
  assert.equal(mac.icon, 'build/icon.icns')
  assert.equal(mac.entitlements, 'build/entitlements.mac.plist')
  assert.equal(mac.entitlementsInherit, 'build/entitlements.mac.plist')
  assert.equal(mac.hardenedRuntime, true)
  assert.match(mac.extendInfo?.NSMicrophoneUsageDescription ?? '', /microphone/i)
  assert.equal(mac.identity, 'Developer ID Application', 'release config cannot silently fall back to undefined/ad-hoc identity')
  assert.equal(mac.notarize, true)
  assert.ok(Array.isArray(mac.target) && mac.target.includes('dmg') && mac.target.includes('zip'))

  const icns = await readFile(path.join(root, mac.icon))
  assert.equal(icns.subarray(0, 4).toString('ascii'), 'icns')

  const png = await readFile(path.join(root, 'build/icon.png'))
  assert.deepEqual([...png.subarray(0, 8)], [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])

  const entitlements = await readFile(path.join(root, mac.entitlements), 'utf8')
  assert.match(entitlements, /com\.apple\.security\.cs\.allow-jit/)
  assert.match(entitlements, /com\.apple\.security\.device\.audio-input/)
  assert.doesNotMatch(entitlements, /com\.apple\.security\.cs\.allow-unsigned-executable-memory/)
  assert.doesNotMatch(entitlements, /com\.apple\.security\.cs\.disable-library-validation/)

  const npmrc = await readFile(path.join(root, '.npmrc'), 'utf8')
  assert.match(npmrc, /^package-lock=true$/m)
  assert.match(npmrc, /^engine-strict=true$/m)
  assert.match(npmrc, /^legacy-peer-deps=false$/m)
  assert.match(npmrc, /^install-links=false$/m)
  assert.match(npmrc, /^ignore-scripts=true$/m)

  const cleanInstallVerifier = await readFile(path.join(root, 'scripts/verify-clean-install.mjs'), 'utf8')
  assert.match(cleanInstallVerifier, /['"]ci['"]/, 'clean-install verification executes npm ci')
  assert.doesNotMatch(cleanInstallVerifier, /--dry-run/, 'clean-install verification must materialize and validate an isolated dependency tree')
  assert.match(cleanInstallVerifier, /node_modules.*\.package-lock\.json/s, 'the isolated install is checked for a realized dependency tree')
  assert.match(cleanInstallVerifier, /rebuild.*--ignore-scripts=false.*electron.*@lydell\/node-pty/s, 'only the allowlisted native package scripts are rebuilt')
  assert.match(cleanInstallVerifier, /require\('electron'\).*require\('@lydell\/node-pty'\)/s, 'the clean native runtime is smoke-tested')

  const testRunner = await readFile(path.join(root, 'scripts/run-tests.mjs'), 'utf8')
  const testCoverageGuard = await readFile(path.join(root, 'scripts/check-test-agent-coverage.mjs'), 'utf8')
  assert.match(testRunner, /testRoots\s*=\s*\[[^\]]*['"]electron['"][^\]]*['"]src['"][^\]]*['"]scripts['"]/s, 'the runner auto-discovers script regressions as well as app tests')
  assert.match(testCoverageGuard, /scannedDirs\s*=\s*\[[^\]]*['"]electron['"][^\]]*['"]src['"][^\]]*['"]scripts['"]/s, 'the orphan-test guard covers the same script root as the runner')

  const readme = await readFile(path.join(root, 'README.md'), 'utf8')
  assert.match(readme, /npm ci[\s\S]*npm run deps:trusted-scripts[\s\S]*npm run dev/, 'the documented normal install path explicitly runs only the trusted native setup before dev')
  assert.match(readme, /built-in server is always loopback-only/i, 'README describes the executable headless bind policy')
  assert.match(readme, /VONTRA_HEADLESS_BIND_HOST[^\n]*accepts only `127\.0\.0\.1`/i, 'README cannot imply that the built-in server accepts a LAN bind')
  assert.doesNotMatch(readme, /VONTRA_HEADLESS_BIND_HOST=0\.0\.0\.0/, 'README never advertises a bind value rejected by the runtime')
  assert.match(readme, /smoke:mac:packaged[\s\S]*mode-0700 temporary `VONTRA_HOME`[\s\S]*accepts no profile-path\s+override/i, 'README documents the isolated packaged smoke boundary')
  assert.match(readme, /never repurposes\s+`HOME` for isolation/i, 'README warns against the incident-prone HOME isolation pattern')
  assert.match(readme, /Before Electron becomes ready[\s\S]*`sessionData` \(including Chromium cookies and cache\)[\s\S]*legacy application-data\s+migration is skipped/i, 'README records full Electron-owned path isolation')

  const securityOperations = await readFile(path.join(root, 'docs/security-operations.md'), 'utf8')
  assert.match(securityOperations, /local desktop GUI starts as `owner`/i, 'operations guide records the trusted local-GUI launch role')
  assert.match(securityOperations, /Headless\/server launches default to `operator`/i, 'operations guide records the safer headless launch role')
  assert.match(securityOperations, /built-in server is loopback-only/i, 'operations guide does not confuse an external proxy with TensorFold Studio bind authority')
  assert.doesNotMatch(securityOperations, /Default runtime role is `operator`/, 'operations guide must distinguish local desktop and headless defaults')
  assert.match(securityOperations, /smoke:mac:packaged[\s\S]*mode-0700 temporary `VONTRA_HOME`[\s\S]*Never use `HOME=\.\.\.` as profile isolation/i, 'operations guide requires the safe packaged smoke harness')
  assert.match(securityOperations, /VONTRA_HOME` is applied before `app\.whenReady\(\)`[\s\S]*real app-data legacy migration is a no-op/i, 'operations guide records early Electron path isolation')
  assert.match(securityOperations, /optimistic concurrency preflight, not a true\s+filesystem compare-and-swap/i, 'local check-to-rename concurrency limit remains explicit')
  assert.match(securityOperations, /SFTP v3 provides no server-atomic digest CAS/i, 'SSH write preflights are not overstated as atomic CAS')

  const appUpdates = await readFile(path.join(root, 'docs/app-updates.md'), 'utf8')
  assert.match(appUpdates, /sole production feed is hardcoded as `https:\/\/tensorfold\.dev\/studio\/update\.json`/i, 'update guide records the immutable canonical feed')
  assert.match(appUpdates, /SHA-256 checksum and an Ed25519 signature/i, 'update guide requires signed checksum metadata')
  assert.match(appUpdates, /rejects its immutable SPKI fingerprint even if its mutable ID\/readiness\s+labels are changed/i, 'update guide records the known-orphan fingerprint gate')
  assert.match(appUpdates, /single strict-JSON source imported by the runtime and read by\s+release preflight/i, 'update guide records the single effective trust source')
  assert.match(appUpdates, /CSC_NAME` and\s+`APPLE_TEAM_ID` to exactly equal the repository pin/i, 'update guide records the pinned macOS publisher boundary')
  assert.match(appUpdates, /DMG is validated and\s+mounted read-only[\s\S]*ZIP is extracted[\s\S]*CDHash/i, 'update guide records contained-app verification')
  assert.match(appUpdates, /device\/inode identity, size, and hash before calling `shell\.openPath`/i, 'update guide describes local artifact re-verification')
  assert.match(appUpdates, /availability has\s+not been verified/i, 'update guide keeps the unverified production feed deployment prerequisite visible')
  assert.doesNotMatch(appUpdates, /Settings:\s*`autoCheckUpdates`\s+and\s+`updateFeedUrl`/i, 'update guide cannot advertise the retired configurable feed')
  assert.doesNotMatch(appUpdates, /replace or augment this detector/i, 'update guide cannot describe the verified downloader as future work')

  console.log('package config tests passed')
}

void main()
