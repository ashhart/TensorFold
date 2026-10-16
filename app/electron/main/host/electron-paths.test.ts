import assert from 'node:assert/strict'
import { chmodSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, symlinkSync, writeFileSync } from 'node:fs'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { applyIsolatedElectronPathPlan, ISOLATED_ELECTRON_PATH_NAMES, migrateLegacyElectronUserData, planIsolatedElectronPaths } from './electron-paths'

assert.equal(planIsolatedElectronPaths(undefined), null)
assert.equal(planIsolatedElectronPaths('   '), null)
assert.throws(() => planIsolatedElectronPaths(path.parse(process.cwd()).root), /filesystem root/)

const parent = mkdtempSync(path.join(os.tmpdir(), 'vontra-electron-paths-'))
try {
  const configuredHome = path.join(parent, 'profile', '..', 'profile')
  const plan = planIsolatedElectronPaths(configuredHome)!
  assert.equal(plan.root, path.join(parent, 'profile'))
  assert.equal(plan.paths.userData, path.join(plan.root, 'electron-runtime', 'user-data'))
  assert.equal(plan.paths.sessionData, path.join(plan.root, 'electron-runtime', 'session-data'))
  assert.notEqual(plan.paths.sessionData, plan.paths.userData, 'Chromium session/cache state has its own isolated path')

  const setCalls: Array<[string, string]> = []
  applyIsolatedElectronPathPlan(plan, { setPath: (name, value) => setCalls.push([name, value]) })
  assert.deepEqual(setCalls.map(([name]) => name), [...ISOLATED_ELECTRON_PATH_NAMES])
  for (const directory of [plan.root, plan.runtimeRoot, ...Object.values(plan.paths)]) {
    const info = lstatSync(directory)
    assert.equal(info.isDirectory(), true)
    assert.equal(info.isSymbolicLink(), false)
    assert.equal(info.mode & 0o777, 0o700)
  }

  let isolatedPathCalls = 0
  let isolatedFileCalls = 0
  assert.equal(await migrateLegacyElectronUserData({
    isolationPlan: plan,
    appName: 'TensorFold Studio',
    legacyAppNames: ['Vontra Studio'],
    appPaths: {
      getPath: () => { isolatedPathCalls++; throw new Error('real appData must not be read') },
      setPath: () => { isolatedPathCalls++; throw new Error('late userData mutation must not run') },
    },
    files: {
      stat: async () => { isolatedFileCalls++; throw new Error('legacy stat must not run') },
      mkdir: async () => { isolatedFileCalls++; throw new Error('legacy mkdir must not run') },
      cp: async () => { isolatedFileCalls++; throw new Error('legacy copy must not run') },
    },
  }), 'isolated')
  assert.equal(isolatedPathCalls, 0, 'isolation bypasses all real appData/userData APIs')
  assert.equal(isolatedFileCalls, 0, 'isolation bypasses all legacy filesystem migration')

  const appData = path.join(parent, 'ordinary-app-data')
  const legacy = path.join(appData, 'Vontra Studio')
  mkdirSync(legacy, { recursive: true })
  writeFileSync(path.join(legacy, 'fixture.txt'), 'legacy')
  const ordinarySetCalls: Array<[string, string]> = []
  assert.equal(await migrateLegacyElectronUserData({
    isolationPlan: null,
    appName: 'TensorFold Studio',
    legacyAppNames: ['Vontra Studio'],
    appPaths: {
      getPath: (name) => { assert.equal(name, 'appData'); return appData },
      setPath: (name, value) => ordinarySetCalls.push([name, value]),
    },
    files: fs,
  }), 'migrated')
  assert.deepEqual(ordinarySetCalls, [['userData', path.join(appData, 'TensorFold Studio')]])
  assert.equal(readFileSync(path.join(appData, 'TensorFold Studio', 'fixture.txt'), 'utf8'), 'legacy')

  // The rename preserves old profile data and never replaces current settings.
  const rebrandAppData = path.join(parent, 'rebrand-app-data')
  const previousApp = path.join(rebrandAppData, 'Vontra')
  const renamedApp = path.join(rebrandAppData, 'TensorFold Studio')
  mkdirSync(previousApp, { recursive: true })
  mkdirSync(renamedApp, { recursive: true })
  writeFileSync(path.join(previousApp, 'settings.json'), '{"theme":"old"}')
  writeFileSync(path.join(previousApp, 'threads.json'), '[{"id":"preserved"}]')
  writeFileSync(path.join(renamedApp, 'settings.json'), '{"theme":"current"}')
  assert.equal(await migrateLegacyElectronUserData({
    isolationPlan: null,
    appName: 'TensorFold Studio',
    legacyAppNames: ['Vontra', 'vontra'],
    appPaths: { getPath: () => rebrandAppData, setPath: () => {} },
    files: fs,
  }), 'migrated')
  assert.equal(readFileSync(path.join(renamedApp, 'settings.json'), 'utf8'), '{"theme":"current"}')
  assert.equal(readFileSync(path.join(renamedApp, 'threads.json'), 'utf8'), '[{"id":"preserved"}]')
  assert.equal(readFileSync(path.join(previousApp, 'threads.json'), 'utf8'), '[{"id":"preserved"}]')

  const symlinkTarget = path.join(parent, 'symlink-target')
  const symlinkHome = path.join(parent, 'symlink-home')
  mkdirSync(symlinkTarget)
  chmodSync(symlinkTarget, 0o755)
  symlinkSync(symlinkTarget, symlinkHome, 'dir')
  const symlinkPlan = planIsolatedElectronPaths(symlinkHome)!
  assert.throws(
    () => applyIsolatedElectronPathPlan(symlinkPlan, { setPath: () => assert.fail('no Electron path may be set after validation failure') }),
    /symlinked isolated Electron path/,
  )
  assert.equal(statSync(symlinkTarget).mode & 0o777, 0o755, 'symlink target is rejected before chmod')

  const mainEntry = readFileSync(path.join(process.cwd(), 'electron/main.ts'), 'utf8')
  assert.ok(
    mainEntry.indexOf("import './main/early-electron-paths'") >= 0 &&
    mainEntry.indexOf("import './main/early-electron-paths'") < mainEntry.indexOf("import './main-app'"),
    'the isolated path bootstrap imports before the application and any session initialization',
  )
  assert.ok(
    mainEntry.indexOf("import './main/early-electron-paths'") < mainEntry.indexOf("import './main/single-instance'") &&
    mainEntry.indexOf("import './main/single-instance'") < mainEntry.indexOf("import './main-app'"),
    'the per-profile single-instance lock is taken after path isolation and before the app loads',
  )
  const earlyBootstrap = readFileSync(path.join(process.cwd(), 'electron/main/early-electron-paths.ts'), 'utf8')
  assert.match(earlyBootstrap, /export \{ APP_NAME, PROFILE_APP_NAME \} from '\.\/host\/app-identity'/)
  assert.match(earlyBootstrap, /configureAppIdentity\(app, Boolean\(isolatedElectronPathPlan\)\)/)
  assert.match(earlyBootstrap, /LEGACY_APP_NAMES = \[.*'Vontra'/)
  assert.doesNotMatch(earlyBootstrap, /getPath\s*\(/, 'early isolation never needs a real Electron path lookup')
  assert.match(earlyBootstrap, /process\.env\.VONTRA_HOME\s*=\s*isolatedElectronPathPlan\.root/, 'store and Electron paths share the canonical isolated root')
  const bootHelpers = readFileSync(path.join(process.cwd(), 'electron/main/host/boot-helpers.ts'), 'utf8')
  assert.match(bootHelpers, /migrateLegacyElectronUserData\(\{[\s\S]*?isolationPlan:\s*isolatedElectronPathPlan/, 'legacy migration receives the already-applied isolation plan')
  assert.match(bootHelpers, /appName: PROFILE_APP_NAME/, 'the public rename keeps legacy profile migration at the same storage root')
} finally {
  rmSync(parent, { recursive: true, force: true })
}

console.log('early isolated Electron path tests passed')
