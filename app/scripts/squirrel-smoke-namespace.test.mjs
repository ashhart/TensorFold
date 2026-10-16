import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { assertCocoaPreferencesUnchanged, assertNativeUpdateNamespaceClear, captureCocoaPreferences, captureNativeUpdateNamespace } from './squirrel-smoke-namespace.mjs'

const taskHome = await fs.mkdtemp(path.join(os.tmpdir(), 'squirrel-namespace-test-'))
const absent = () => Object.assign(new Error('absent'), { stderr: 'The domain does not exist. Could not find service.' })
const run = async (command) => { if (command === '/bin/ps') return { stdout: '100 /fixture.app/Contents/MacOS/TensorFold Studio\n' }; throw absent() }
try {
  const clear = await captureNativeUpdateNamespace({ taskHome, run })
  assert.doesNotThrow(() => assertNativeUpdateNamespaceClear(clear, 100))
  assert.throws(() => assertNativeUpdateNamespaceClear(clear), /another app instance/)
  const renamed = await captureNativeUpdateNamespace({ taskHome, run: async (command) => {
    if (command === '/bin/ps') return { stdout: '100 /fixture.app/Contents/MacOS/TensorFold Studio\n101 /new.app/Contents/MacOS/TensorFold\n' }
    throw absent()
  } })
  assert.deepEqual(renamed.studioProcesses, [100, 101], 'both names share the same updater/profile namespace during the rename')
  assert.throws(() => assertNativeUpdateNamespaceClear(renamed, 100), /another app instance/)
  for (const key of ['appCacheExists', 'shipItCacheExists', 'shipItPreferenceFileExists', 'shipItByHostFiles', 'shipItPreferenceDomainExists', 'shipItCurrentHostDomainExists', 'activeShipItJob']) assert.throws(() => assertNativeUpdateNamespaceClear({ ...clear, [key]: true }, 100), /will not consume/)
  const directory = path.join(taskHome, 'Library', 'Preferences')
  await fs.mkdir(directory, { recursive: true })
  const preferences = path.join(directory, 'dev.tensorfold.studio.plist')
  await fs.writeFile(preferences, 'original preferences')
  const before = await captureCocoaPreferences({ taskHome })
  await assertCocoaPreferencesUnchanged(before, { taskHome })
  await fs.writeFile(preferences, 'changed preferences')
  await assert.rejects(() => assertCocoaPreferencesUnchanged(before, { taskHome }), /without restoring/)
  assert.equal(await fs.readFile(preferences, 'utf8'), 'changed preferences', 'the guard never resets or restores user preference files')
} finally { await fs.rm(taskHome, { recursive: true, force: true }) }
console.log('Native update smoke refuses global updater state and preserves Cocoa preference snapshots read-only')
