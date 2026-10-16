import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { createHash } from 'node:crypto'
import { BotVmInstaller, assertManagedRuntimeAvailable, stopManagedRuntimeServices } from './installer'
import { BotVmService, type VmRunner } from './service'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-managed-setup-test-'))
try {
  let waiting = false, entered = false, startFails = false
  const runner: VmRunner = async (binary, args) => {
    if (binary.endsWith('launchctl')) return { code: 1, stdout: '', stderr: '' }
    if (args[0] === 'system' && args[1] === 'version') return { code: 0, stdout: '[{"appName":"container","version":"1.5.0"}]', stderr: '' }
    if (args[0] === 'system' && args[1] === 'start') return { code: startFails ? 1 : 0, stdout: '', stderr: '' }
    return { code: 0, stdout: '[]', stderr: '' }
  }
  class FixtureInstaller extends BotVmInstaller {
    async install(signal?: AbortSignal) {
      entered = true
      if (waiting) await new Promise<void>((_resolve, reject) => { if (signal?.aborted) reject(new Error('VM setup cancelled.')); else signal?.addEventListener('abort', () => reject(new Error('VM setup cancelled.')), { once: true }) })
      const runtime = this.paths(); await fs.mkdir(path.dirname(runtime.binary), { recursive: true }); await fs.mkdir(runtime.appRoot, { recursive: true }); await fs.writeFile(runtime.binary, 'fixture runtime', { mode: 0o700 }); return runtime
    }
    async verifyInstalled() {}
  }
  const fixtureOptions = { platform: 'darwin' as const, architecture: 'arm64', osVersion: '26.5.2', runner, verifyRuntime: async () => {}, installerFactory: (home: string, run: VmRunner) => new FixtureInstaller(home, run) }
  const home = path.join(root, 'rollback'), prior = path.join(root, 'previous', 'container')
  await fs.mkdir(path.dirname(prior)); await fs.writeFile(prior, 'previous runtime', { mode: 0o700 })
  const previous = { version: 1, binary: prior, sha256: createHash('sha256').update('previous runtime').digest('hex') }
  await fs.mkdir(path.join(home, 'bot-vms'), { recursive: true }); await fs.writeFile(path.join(home, 'bot-vms', 'runtime.json'), JSON.stringify(previous))
  startFails = true
  const rollback = new BotVmService({ ...fixtureOptions, appHome: home })
  await assert.rejects(rollback.managedSetup('writer'), /services could not start/)
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(home, 'bot-vms', 'runtime.json'), 'utf8')), previous, 'failed service activation restores the earlier runtime selection')
  assert.equal(rollback.setupStatus().active, false)
  assert.equal(rollback.setupStatus().phase, 'error')
  startFails = false; waiting = true; entered = false
  const cancel = new BotVmService({ ...fixtureOptions, appHome: path.join(root, 'cancel'), runtimePath: prior })
  const pending = cancel.managedSetup('writer')
  while (!entered) await new Promise((resolve) => setTimeout(resolve, 1))
  assert.equal(cancel.setupStatus().active, true); assert.equal(cancel.setupStatus().botId, 'writer')
  const stopped = cancel.stop('writer')
  await assert.rejects(pending, /cancelled/); await stopped
  assert.equal(cancel.setupStatus().active, false)
  assert.equal(cancel.setupStatus().phase, 'cancelled')
  const fresh = new BotVmService({ ...fixtureOptions, appHome: path.join(root, 'queued'), runtimePath: prior })
  const queued = fresh.managedSetup('writer'); const queuedStop = fresh.stop('writer')
  await assert.rejects(queued, /cancelled/); await queuedStop
  assert.equal(fresh.setupStatus().active, false, 'even cancellation before the worker microtask cannot boot the bot later')
  let bootouts = 0
  const runtime = { installRoot: '/fictional-owned/runtime', appRoot: '/fictional-owned/data', binary: '/fictional-owned/runtime/bin/container' }
  const foreign: VmRunner = async (_binary, args) => { if (args[0] === 'bootout') bootouts++; return { code: 0, stdout: 'program = /fictional-other/runtime/bin/container-apiserver\nCONTAINER_APP_ROOT => /fictional-other/data', stderr: '' } }
  await assert.rejects(assertManagedRuntimeAvailable(runtime, foreign), /Another Apple container runtime/)
  await assert.rejects(stopManagedRuntimeServices(runtime, foreign), /Another Apple container runtime/)
  assert.equal(bootouts, 0, 'foreign user services are never stopped')
  const live = new Set(['com.apple.container.apiserver'])
  const owned: VmRunner = async (_binary, args) => {
    const label = args[1].split('/').at(-1)!
    if (args[0] === 'bootout') { bootouts++; live.delete(label); return { code: 0, stdout: '', stderr: '' } }
    return live.has(label) ? { code: 0, stdout: `program = ${runtime.installRoot}/bin/container-apiserver\nCONTAINER_APP_ROOT => ${runtime.appRoot}`, stderr: '' } : { code: 1, stdout: '', stderr: '' }
  }
  await stopManagedRuntimeServices(runtime, owned); assert.equal(bootouts, 1); assert.equal(live.size, 0)
  let removalPending = false, delayedPrints = 0
  const delayedOwned: VmRunner = async (_binary, args) => {
    if (!args[1].endsWith('/com.apple.container.apiserver')) return { code: 1, stdout: '', stderr: '' }
    if (args[0] === 'bootout') { removalPending = true; return { code: 0, stdout: '', stderr: '' } }
    if (removalPending && ++delayedPrints > 1) return { code: 1, stdout: '', stderr: '' }
    return { code: 0, stdout: `program = ${runtime.installRoot}/bin/container-apiserver\nCONTAINER_APP_ROOT => ${runtime.appRoot}`, stderr: '' }
  }
  await stopManagedRuntimeServices(runtime, delayedOwned)
  assert.ok(delayedPrints > 1, 'shutdown waits for asynchronous launchd removal rather than reporting a false failure')
  console.log('managed setup activation rollback, pause cancellation and exact owned-service cleanup passed')
} finally { await fs.rm(root, { recursive: true, force: true }) }
