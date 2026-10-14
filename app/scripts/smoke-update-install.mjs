#!/usr/bin/env node
// Real native replacement/relaunch, only in an owner-held disposable copy of a signed app.
import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import { createRequire } from 'node:module'
import { mkdir, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { promisify } from 'node:util'
import { createIsolatedSmokeProfile, removeExactSmokeProfile, selectUnusedLoopbackPort } from './smoke-packaged-mac.mjs'
import { qualificationEnvironment } from './qualify-packaged-memory.mjs'
import { assertCocoaPreferencesUnchanged, assertNativeUpdateNamespaceClear, captureCocoaPreferences, captureNativeUpdateNamespace } from './squirrel-smoke-namespace.mjs'

const root = fileURLToPath(new URL('..', import.meta.url))
const require = createRequire(import.meta.url)
const run = promisify(execFile)

export function parseUpdateSmokeArgs(argv) {
  const options = {}
  for (let index = 0; index < argv.length; index += 2) {
    const key = argv[index]
    if (!['--from-app', '--target-app', '--expected-version', '--out', '--playwright-module'].includes(key) || !argv[index + 1] || argv[index + 1].startsWith('--')) throw new Error('Use --from-app, --target-app, --expected-version, --out and optional --playwright-module; the canonical feed cannot be overridden.')
    options[key.slice(2)] = argv[index + 1]
  }
  if (!options['from-app'] || !options['target-app'] || !options['expected-version'] || !options.out) throw new Error('Two signed app paths, the expected target version and private evidence directory are required.')
  for (const key of ['from-app', 'target-app']) if (!path.resolve(options[key]).endsWith('.app')) throw new Error('Update smoke inputs must be packaged .app bundles.')
  return options
}

async function assertSignedBundle(app, signer) {
  await run('/usr/bin/codesign', ['--verify', '--deep', '--strict', app])
  const detail = await run('/usr/bin/codesign', ['--display', '--verbose=4', app])
  const lines = detail.stderr.split('\n').map((line) => line.trim())
  assert.ok(lines.includes(`Authority=${signer.identity}`) && lines.includes(`TeamIdentifier=${signer.teamId}`) && !lines.includes('Signature=adhoc'), 'both builds must carry the pinned Developer ID signature')
  const version = await run('/usr/bin/plutil', ['-extract', 'CFBundleShortVersionString', 'raw', '-o', '-', path.join(app, 'Contents', 'Info.plist')])
  const codeDirectoryHash = lines.find((line) => line.startsWith('CDHash='))?.slice(7)
  assert.match(codeDirectoryHash ?? '', /^[a-f0-9]{40,64}$/, 'a signed bundle must expose its CodeDirectory identity')
  return { version: version.stdout.trim(), codeDirectoryHash }
}

async function assertBootstrap(app) {
  // A target lacking recovery would relaunch the real default profile instead of the fixture.
  const asar = await readFile(path.join(app, 'Contents', 'Resources', 'app.asar'))
  assert.ok(asar.includes(Buffer.from('.tensorfold-update-profile.json')), 'both signed apps must contain the guarded private-profile relaunch recovery')
}

async function connect(chromium, port, timeoutMs = 30_000) {
  const until = Date.now() + timeoutMs
  for (;;) {
    try {
      const browser = await chromium.connectOverCDP(`http://127.0.0.1:${port}`, { timeout: 1000 })
      const context = browser.contexts()[0]
      const page = context.pages()[0] || await context.waitForEvent('page', { timeout: 10_000 })
      await page.waitForSelector('body.ready', { timeout: 30_000 })
      return { browser, page }
    } catch {
      if (Date.now() >= until) throw new Error('The isolated updated app did not relaunch its guarded loopback test endpoint.')
      await new Promise((resolve) => setTimeout(resolve, 200))
    }
  }
}

export async function smokeUpdateInstall(options) {
  if (process.platform !== 'darwin') throw new Error('Native Squirrel replacement/relaunch smoke requires macOS.')
  const { chromium } = require(options['playwright-module'] || 'playwright')
  const { macos } = JSON.parse(await readFile(path.join(root, 'build/release-trust.json'), 'utf8'))
  const fromApp = path.resolve(options['from-app']), targetApp = path.resolve(options['target-app'])
  const fromIdentity = await assertSignedBundle(fromApp, macos)
  const fromVersion = fromIdentity.version
  const targetIdentity = await assertSignedBundle(targetApp, macos)
  assert.equal(targetIdentity.version, options['expected-version'])
  await run('/usr/sbin/spctl', ['--assess', '--type', 'execute', '--verbose=4', targetApp])
  await assertBootstrap(fromApp); await assertBootstrap(targetApp)
  const nativeNamespaceBefore = await captureNativeUpdateNamespace()
  assertNativeUpdateNamespaceClear(nativeNamespaceBefore)
  const cocoaPreferencesBefore = await captureCocoaPreferences()
  const profile = await createIsolatedSmokeProfile()
  // Use the target filename from the start: Squirrel renames a bundle to its new
  // product filename, while the deliberately strict private-profile marker binds
  // an exact path. The old signed source bytes/version remain unchanged.
  const appBundle = path.join(profile.root, path.basename(targetApp))
  const readExecutable = async bundle => (await run('/usr/bin/plutil', ['-extract', 'CFBundleExecutable', 'raw', '-o', '-', path.join(bundle, 'Contents', 'Info.plist')])).stdout.trim()
  const fromExecutableName = await readExecutable(fromApp), targetExecutableName = await readExecutable(targetApp)
  for (const name of [fromExecutableName, targetExecutableName]) assert.ok(name && !/[\\/\0\r\n]/.test(name), 'signed app executable must be a safe filename')
  const port = await selectUnusedLoopbackPort()
  let connection, child
  let step = 'launch'
  let nativeInstallAttempted = false, smokeCompleted = false
  try {
    await run('/usr/bin/ditto', ['--noqtn', fromApp, appBundle])
    await writeFile(path.join(profile.root, '.tensorfold-update-profile.json'), JSON.stringify({ schemaVersion: 1, root: profile.root, appBundle, debugPort: port }), { mode: 0o600 })
    const env = qualificationEnvironment(process.env, profile.root, port, 64)
    // This smoke does not alter the model budget; it tests updater preservation/relaunch only.
    delete env.TENSORFOLD_STUDIO_MEMORY_BUDGET_BYTES
    const executable = path.join(appBundle, 'Contents', 'MacOS', fromExecutableName)
    // Copying can take time: the global namespace must still be empty at actual launch.
    assertNativeUpdateNamespaceClear(await captureNativeUpdateNamespace())
    await assertCocoaPreferencesUnchanged(cocoaPreferencesBefore)
    child = spawn(executable, [], { env, stdio: 'ignore' })
    connection = await connect(chromium, port)
    await assertCocoaPreferencesUnchanged(cocoaPreferencesBefore)
    const { page } = connection
    step = 'canonical feed and native support'
    const checked = await page.evaluate(() => window.tensorfoldStudio.checkForUpdates())
    assert.equal(checked.currentVersion, fromVersion)
    assert.equal(checked.latestVersion, options['expected-version'])
    assert.equal(checked.state, 'available', 'the real canonical signed feed must offer the expected newer target')
    assert.equal(checked.signatureVerified, true, 'the canonical feed must verify against the embedded production root')
    assert.equal(checked.automaticInstall?.supported, true, checked.automaticInstall?.reason ?? 'the signed private fixture must support automatic installation')
    step = 'preservation fixture'
    const thread = await page.evaluate(async () => {
      const bridge = window.tensorfoldStudio, settings = await bridge.getSettings()
      await bridge.saveSettings({ ...settings, temperature: 0.125, autoCheckUpdates: false,
        providers: [...settings.providers, { id: 'upgrade-fixture-provider', name: 'Upgrade fixture', kind: 'cloud', baseUrl: 'https://api.example.com/v1', apiKey: 'synthetic-upgrade-fixture-key', models: [] }] })
      localStorage.setItem('tensorfold-rename-preservation-fixture', 'retained')
      const created = await bridge.createThread('', 'chat')
      await bridge.renameThread(created.id, 'Upgrade preservation fixture')
      return created
    })
    const projectFolder = path.join(profile.root, 'preserved-project')
    await mkdir(projectFolder)
    await writeFile(path.join(projectFolder, 'README.md'), 'Preserved update project fixture.\n')
    const project = await page.evaluate((cwd) => window.tensorfoldStudio.createProject({ name: 'Upgrade preserved project', cwd, workspace: { kind: 'local', path: cwd }, executionTarget: 'local' }), projectFolder)
    // Saving update preferences intentionally resets a prior check; use a fresh signed check.
    const beforeDownload = await page.evaluate(() => window.tensorfoldStudio.checkForUpdates())
    assert.equal(beforeDownload.state, 'available', 'a fresh signed check must follow fixture settings changes')
    assert.equal(beforeDownload.latestVersion, options['expected-version'])
    assert.equal(beforeDownload.signatureVerified, true)
    step = 'verified ZIP download'
    assert.equal((await page.evaluate(() => window.tensorfoldStudio.downloadAppUpdate())).ok, true, 'the canonical ZIP download must complete and verify')
    assert.equal((await page.evaluate(() => window.tensorfoldStudio.getUpdateStatus())).state, 'ready')

    // A real active native run must be rejected before any native installer staging or quit.
    step = 'active work refusal'
    await page.evaluate(async (id) => {
      const bridge = window.tensorfoldStudio, snapshot = await bridge.engineSnapshot()
      const model = snapshot.installed.find((entry) => entry.fitsMemory === true)
      if (!model) throw new Error('Active-run install guard smoke needs a cached compatible model; it does not download model weights.')
      await bridge.configureThread(id, 'tensorfold', model.repo, null)
      await bridge.startRun({ threadId: id, userText: 'Write a long explanation of how to maintain a git repository; do not use tools or modify any files.', approvalMode: 'readonly' })
    }, thread.id)
    const refused = await page.evaluate(() => window.tensorfoldStudio.installAppUpdate())
    assert.equal(refused.ok, false, 'active work must keep the current app open')
    const activeWorkRefusal = await page.evaluate(() => window.tensorfoldStudio.getUpdateStatus())
    assert.equal(activeWorkRefusal.state, 'ready', 'refusal must retain the verified ready update')
    assert.match(activeWorkRefusal.error ?? '', /Finish or stop active workers/, 'the refusal must be caused by active-work safety')
    assert.equal(child.exitCode, null)
    await page.evaluate((id) => window.tensorfoldStudio.cancelRun(id), thread.id)
    let settled = false
    for (let attempt = 0; attempt < 120; attempt++) {
      const idle = await page.evaluate(async (id) => {
        const bridge = window.tensorfoldStudio
        const [threads, snapshot] = await Promise.all([bridge.getThreads(), bridge.engineSnapshot()])
        return threads.find((item) => item.id === id)?.status !== 'running' && snapshot.status.state !== 'starting'
      }, thread.id)
      if (idle) { settled = true; break }
      await page.waitForTimeout(100)
    }
    assert.equal(settled, true, 'cancelled work and any model startup must settle before the positive install trial')
    await page.waitForTimeout(300)
    await assertCocoaPreferencesUnchanged(cocoaPreferencesBefore)
    const namespaceAtInstall = await captureNativeUpdateNamespace()
    assertNativeUpdateNamespaceClear(namespaceAtInstall, child.pid)
    const exit = new Promise((resolve) => child.once('exit', resolve))
    nativeInstallAttempted = true
    step = 'native installation and relaunch'
    // The successful call may lose its renderer response as macOS closes the old app.
    const install = page.evaluate(() => window.tensorfoldStudio.installAppUpdate()).catch(() => null)
    const outcome = await Promise.race([exit.then(() => 'quit'), install.then((value) => value?.ok === false ? 'refused' : 'handoff'), new Promise((resolve) => setTimeout(() => resolve('timeout'), 60_000))])
    assert.notEqual(outcome, 'refused', 'idle preservation must allow the verified update handoff')
    assert.notEqual(outcome, 'timeout', 'native staging/quit must complete')
    if (child.exitCode === null) await Promise.race([exit, new Promise((_, reject) => setTimeout(() => reject(new Error('The original app did not quit after the update handoff.')), 30_000))])
    await connection.browser.close()
    connection = await connect(chromium, port, 60_000)
    const preserved = await connection.page.evaluate(async ({ id, projectId }) => ({
      version: (await window.tensorfoldStudio.getUpdateStatus()).currentVersion,
      thread: (await window.tensorfoldStudio.getThreads()).find((item) => item.id === id)?.title,
      project: (await window.tensorfoldStudio.getProjects()).find((item) => item.id === projectId)?.name,
      temperature: (await window.tensorfoldStudio.getSettings()).temperature,
      fixtureCredentialPreserved: (await window.tensorfoldStudio.getSettings()).providers.find((item) => item.id === 'upgrade-fixture-provider')?.apiKey === '••••••••',
      rendererPreferencePreserved: localStorage.getItem('tensorfold-rename-preservation-fixture') === 'retained',
      appTitle: document.title,
    }), { id: thread.id, projectId: project.id })
    assert.equal(preserved.version, options['expected-version'])
    assert.equal(preserved.thread, 'Upgrade preservation fixture')
    assert.equal(preserved.temperature, 0.125)
    assert.equal(preserved.project, 'Upgrade preserved project')
    assert.equal(preserved.fixtureCredentialPreserved, true, 'the renderer must retain a masked stored fixture credential after the app rename/update')
    assert.equal(preserved.rendererPreferencePreserved, true, 'Chromium local storage must survive the app rename/update')
    assert.deepEqual(await assertSignedBundle(appBundle, macos), targetIdentity, 'the actual copied bundle was replaced with the exact signed published target')
    await run('/usr/sbin/spctl', ['--assess', '--type', 'execute', '--verbose=4', appBundle])
    await run('/usr/bin/xcrun', ['stapler', 'validate', appBundle])
    await assertCocoaPreferencesUnchanged(cocoaPreferencesBefore)
    const nativeNamespaceAfterRelaunch = await captureNativeUpdateNamespace()
    await mkdir(options.out, { recursive: true, mode: 0o700 })
    await connection.page.screenshot({ path: path.join(options.out, 'native-relaunched.png'), scale: 'css' })
    const report = { evidenceKind: 'real Developer ID native Squirrel replacement and automatic relaunch in an isolated copy', fromVersion, fromBuildKind: 'Exact signed source bundle supplied for this run; source version and CodeDirectory identity recorded', fromCodeDirectoryHash: fromIdentity.codeDirectoryHash, historicalBeta1Upgrade: 'The historical public beta.1 requires its documented manual first upgrade.', targetVersion: preserved.version, targetBuildKind: 'exact published signed, notarized and stapled target from the canonical root-verified feed', targetCodeDirectoryHash: targetIdentity.codeDirectoryHash, targetZipSha256: beforeDownload.automaticUpdate?.checksum.value, signatureVerified: true, activeWorkBlocked: true, activeWorkRefusalReason: activeWorkRefusal.error, threadPreserved: true, projectPreserved: true, settingsPreserved: true, storedCredentialRemainsMasked: preserved.fixtureCredentialPreserved, rendererPreferencePreserved: preserved.rendererPreferencePreserved, targetAppTitle: preserved.appTitle, fromExecutableName, targetExecutableName, profileIsolated: true, globalNativeNamespaceLimitation: 'Squirrel caches and ShipIt job remain global per user/production bundle ID; all prior native state was required absent before launch and at the install handoff.', nativeNamespaceBefore, namespaceAtInstall, nativeNamespaceAfterRelaunch, existingCocoaPreferencesUnchanged: true, nativeCacheResetOrDeleted: false }
    await writeFile(path.join(options.out, 'native-update.json'), `${JSON.stringify(report, null, 2)}\n`, { mode: 0o600 })
    smokeCompleted = true
    return report
  } catch (error) {
    await mkdir(options.out, { recursive: true, mode: 0o700 })
    let updateStatus = null
    try { updateStatus = await connection?.page.evaluate(() => window.tensorfoldStudio.getUpdateStatus()) } catch { /* the native renderer may already have quit */ }
    await writeFile(path.join(options.out, 'native-update-failure.json'), `${JSON.stringify({ step, error: error.message, nativeInstallAttempted, profile: profile.root, updateStatus, nativeCacheResetOrDeleted: false }, null, 2)}\n`, { mode: 0o600 })
    throw new Error(`${step}: ${error.message}`)
  } finally {
    try { await connection?.page.evaluate(() => window.tensorfoldStudio.engineUnload()) } catch { /* only the private process remains */ }
    // Squirrel's relaunched process is not our child: identify it by this exact private executable.
    const executables = [fromExecutableName, targetExecutableName].map(name => path.join(appBundle, 'Contents', 'MacOS', name))
    const processes = await run('/bin/ps', ['-axo', 'pid=,command='])
    for (const line of processes.stdout.split('\n')) {
      const match = /^\s*(\d+)\s+(.*)$/.exec(line)
      if (executables.includes(match?.[2])) { try { process.kill(Number(match[1]), 'SIGTERM') } catch { /* already exited */ } }
    }
    await connection?.browser.close()
    if (child?.exitCode === null && child.signalCode === null) child.kill('SIGTERM')
    await new Promise((resolve) => setTimeout(resolve, 300))
    if (!nativeInstallAttempted) await removeExactSmokeProfile(profile)
    else {
      // Do not remove a target/descriptor while an uncertain native transaction may still use it.
      await mkdir(options.out, { recursive: true, mode: 0o700 })
      const nativeNamespaceAfterStop = await captureNativeUpdateNamespace()
      const hasNativeFootprint = Object.entries(nativeNamespaceAfterStop).some(([key, value]) => key === 'studioProcesses' ? value.length > 0 : !!value)
      if (smokeCompleted && !hasNativeFootprint) await removeExactSmokeProfile(profile)
      const retained = !smokeCompleted || hasNativeFootprint
      await writeFile(path.join(options.out, smokeCompleted ? 'native-update-footprint.json' : 'native-update-incomplete.json'), `${JSON.stringify({ profileRetained: retained ? profile.root : null, profileIdentity: { device: String(profile.identity.dev), inode: String(profile.identity.ino), uid: String(profile.identity.uid) }, reason: retained ? 'Retain the exact private target and descriptor until newly created native state and job references receive ownership-checked review.' : 'No native global footprint remains after the completed update.', nativeNamespaceAfterStop, nativeCacheResetOrDeleted: false }, null, 2)}\n`, { mode: 0o600 })
    }
  }
}

if (process.argv[1] && pathToFileURL(path.resolve(process.argv[1])).href === import.meta.url) {
  smokeUpdateInstall(parseUpdateSmokeArgs(process.argv.slice(2))).then((report) => console.log(JSON.stringify(report))).catch((error) => { console.error(error.message); process.exitCode = 1 })
}
