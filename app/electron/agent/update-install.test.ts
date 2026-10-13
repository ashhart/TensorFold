import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { EventEmitter } from 'node:events'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import http from 'node:http'
import { assertPinnedMacSignature, installVerifiedUpdate, stageNativeUpdate, startVerifiedUpdateFeed, type NativeUpdateDriver, type UpdateInstallSafety } from './update-install'
import { downloadVerifiedUpdate } from './update-download'

const trust = { identity: 'Developer ID Application: Example Publisher (ABCDEFGHIJ)', teamId: 'ABCDEFGHIJ', productionReady: true }
const validSignature = `Authority=${trust.identity}\nAuthority=Developer ID Certification Authority\nTeamIdentifier=${trust.teamId}`

class Driver extends EventEmitter implements NativeUpdateDriver {
  outcome = 'update-downloaded'
  feedUrl: string | null = null
  installs = 0
  setFeedURL({ url }: { url: string; serverType: 'default' }) { this.feedUrl = url }
  checkForUpdates() { queueMicrotask(() => this.emit(this.outcome)) }
  quitAndInstall() { this.installs++ }
}

async function main() {
  assert.doesNotThrow(() => assertPinnedMacSignature(validSignature, trust))
  assert.throws(() => assertPinnedMacSignature(`${validSignature}\nSignature=adhoc`, trust), /production Developer ID/)
  assert.throws(() => assertPinnedMacSignature(validSignature.replace('TeamIdentifier=ABCDEFGHIJ', 'TeamIdentifier=ZZZZZZZZZZ'), trust), /production Developer ID/)
  assert.throws(() => assertPinnedMacSignature(validSignature, { ...trust, productionReady: false }), /production Developer ID/)

  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-update-install-'))
  const bytes = Buffer.from('verified ZIP fixture')
  const checksum = { algorithm: 'sha256' as const, value: createHash('sha256').update(bytes).digest('hex') }
  let feed: Awaited<ReturnType<typeof startVerifiedUpdateFeed>> | null = null
  try {
    const download = await downloadVerifiedUpdate({ downloadUrl: 'https://example.com/update.zip', checksum, manifestSignatureVerified: true, downloadRoot: root }, {
      lookup: async () => [{ address: '93.184.216.34', family: 4 }], fetchImpl: (async () => new Response(bytes)) as typeof fetch,
    })
    feed = await startVerifiedUpdateFeed({ download, downloadRoot: root, version: '1.0.1', releaseName: null, releaseNotes: null })
    const url = new URL(feed.url)
    assert.equal(url.hostname, '127.0.0.1', 'the native updater feed binds only to loopback')
    assert.equal((await fetch(`${url.origin}/manifest`)).status, 404, 'the feed requires its random private path')
    const foreignHostStatus = await new Promise<number | undefined>((resolve, reject) => { const request = http.get(feed!.url, { headers: { Host: 'hostile.example' } }, (response) => { response.resume(); resolve(response.statusCode) }); request.on('error', reject) })
    assert.equal(foreignHostStatus, 403, 'foreign Host headers cannot reach the private feed')
    assert.equal((await fetch(feed.url, { method: 'POST' })).status, 403)
    const manifest = await (await fetch(feed.url)).json()
    assert.equal(manifest.sha256, checksum.value)
    assert.equal(manifest.size, bytes.length)
    assert.equal(new URL(manifest.url).origin, url.origin)
    assert.deepEqual(Buffer.from(await (await fetch(manifest.url)).arrayBuffer()), bytes, 'Squirrel receives only the verified local ZIP bytes')
    await fs.writeFile(download.path, 'tampered ZIP')
    assert.equal((await fetch(manifest.url)).status, 500, 'a changed ZIP is blocked before the native updater receives it')
  } finally { await feed?.close(); await fs.rm(root, { recursive: true, force: true }) }

  const driver = new Driver()
  await stageNativeUpdate(driver, 'http://127.0.0.1:1/private/manifest', 100)
  assert.equal(driver.installs, 0, 'staging alone never quits the app')
  assert.equal(driver.listenerCount('error'), 0)
  driver.outcome = 'error'
  await assert.rejects(() => stageNativeUpdate(driver, 'http://127.0.0.1:1/private/manifest', 100), /could not stage/)
  driver.outcome = 'update-not-available'
  await assert.rejects(() => stageNativeUpdate(driver, 'http://127.0.0.1:1/private/manifest', 100), /did not stage/)
  assert.equal(driver.installs, 0)

  const order: string[] = []
  let released = 0
  let active = true
  const safety: UpdateInstallSafety = {
    assertIdle: () => { order.push('idle'); if (active) throw new Error('Finish active work before installing.') },
    acquireLease: async () => { order.push('lease'); return { release: () => { released++; order.push('release') } } },
    preserve: async () => { order.push('preserve') },
  }
  const installDriver = new Driver()
  const input = {
    driver: installDriver, safety,
    openFeed: async () => { order.push('feed'); return { url: 'http://127.0.0.1:1/private/manifest', close: async () => { order.push('close') } } },
    stage: async () => { order.push('stage') },
  }
  await assert.rejects(() => installVerifiedUpdate(input), /Finish active work/)
  assert.equal(installDriver.installs, 0)
  assert.deepEqual(order, ['idle'], 'active work prevents even staging the installer')
  active = false
  order.length = 0
  await assert.rejects(() => installVerifiedUpdate({ ...input, safety: { ...safety, preserve: async () => { throw new Error('Disk full') } } }), /Disk full/)
  assert.equal(installDriver.installs, 0, 'preservation failure never restarts the app')
  assert.equal(released, 1, 'failed preparation releases the admission gate')
  assert.equal(order.includes('stage'), false)
  order.length = 0
  await installVerifiedUpdate(input)
  assert.equal(installDriver.installs, 1, 'only verified, staged and preserved idle work can install/restart')
  assert.deepEqual(order, ['idle', 'lease', 'idle', 'preserve', 'idle', 'feed', 'stage', 'idle', 'preserve', 'idle', 'close'])
  assert.equal(released, 1, 'successful handoff retains the global admission gate until process exit')
  installDriver.emit('error', new Error('native relaunch failed'))
  assert.equal(released, 2, 'asynchronous native handoff failure unlocks workspace admission when the app stays open')
  console.log('secure automatic update feed and restart safety tests passed')
}

main().catch((error) => { console.error(error); process.exitCode = 1 })
