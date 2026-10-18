import assert from 'node:assert/strict'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { spawnSync } from 'node:child_process'

// Exercise the production Swift policy/state machine/store with injected
// network and Keychain adapters; never access the real phone account or device.
if (process.platform !== 'darwin') {
  console.log('Native watch relay regression requires macOS Swift/Security; run this check on the Mac build host.')
} else {
  const directory = mkdtempSync(path.join(tmpdir(), 'tensorfold-watch-native-'))
  try {
    const sources = ['WatchBackgroundPolicy.swift', 'WatchBackgroundRelay.swift', 'WatchBackgroundConfigurationStore.swift']
      .map((name) => path.join('mobile/modules/tensorfold-watch-connectivity/ios', name))
    const executable = path.join(directory, 'native-core-tests')
    const compile = spawnSync('xcrun', ['swiftc', ...sources, 'mobile/modules/tensorfold-watch-connectivity/tests/main.swift', '-o', executable], {encoding: 'utf8', timeout: 60_000})
    assert.equal(compile.status, 0, `Swift compile failed: ${compile.stderr || compile.error?.message}`)
    const test = spawnSync(executable, [], {encoding: 'utf8', timeout: 15_000})
    assert.equal(test.status, 0, `Native relay regression failed: ${test.stderr || test.error?.message}`)
    assert.match(test.stdout, /PASS \d+ native background relay and consent assertions/)
    console.log(test.stdout.trim())
  } finally { rmSync(directory, {recursive: true, force: true}) }
}
