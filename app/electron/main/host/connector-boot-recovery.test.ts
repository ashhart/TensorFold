import assert from 'node:assert/strict'
import path from 'node:path'
import { bootAppState } from './boot-sequence'

async function exercise(failRecovery: boolean) {
  const order: string[] = []
  const store: Record<string, any> = {}
  const noop = () => {}
  const asyncNoop = async () => {}
  const configured: Record<string, any> = {
    store, path, HEADLESS_MODE: false,
    fs: {}, app: { on: noop }, BrowserWindow: { getAllWindows: () => [] },
    vontraHomeDir: () => '/synthetic-no-filesystem-access', vontraPath: () => '/synthetic-no-filesystem-access',
    createEventSpineHost: async () => ({}),
    loadSettings: async () => ({ defaultCwd: '', remoteControl: {} }),
    loadPairingSessions: async () => null,
    loadConnectorEvents: async () => { order.push('loaded'); return [] },
    recoverConnectorDeliveries: async () => { order.push('recovered'); if (failRecovery) throw new Error('Synthetic recovery storage failure') },
    checkpoints: { restorePersisted: asyncNoop }, providerStatusRuntime: { setStatus: noop },
    mcpHost: { setConfigs: noop, connectEnabled: asyncNoop }, webhookServer: { syncFromEndpoints: asyncNoop },
    remoteServer: { syncFromSettings: async () => { order.push('remote') } },
    createWindow: () => { order.push('window') }, startScheduler: () => { order.push('scheduler') },
    writeTensorFoldStudioStateJson: asyncNoop, purgeLegacyUserDataState: asyncNoop, cleanupMediaCache: asyncNoop,
  }
  const deps = new Proxy(configured, { get: (target, key: string) => key in target ? target[key] : key.startsWith('load') ? async () => [] : noop })
  if (failRecovery) await assert.rejects(() => bootAppState(deps as any), /recovery storage failure/)
  else await bootAppState(deps as any)
  return order
}

const success = await exercise(false)
assert.ok(success.indexOf('recovered') > success.indexOf('loaded'))
for (const consumer of ['window', 'remote', 'scheduler']) assert.ok(success.indexOf(consumer) > success.indexOf('recovered'), `${consumer} must not precede durable interrupted-send recovery`)
assert.deepEqual(await exercise(true), ['loaded', 'recovered'], 'Recovery storage failure must prevent window, remote and scheduler admission')
