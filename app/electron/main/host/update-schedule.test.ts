import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { mock } from 'node:test'
import ts from 'typescript'

const requireBuiltin = createRequire(__filename)
const source = readFileSync(new URL('./updates.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
}).outputText

function updater() {
  const settings = { autoCheckUpdates: true }
  const store = { settings, updateStatus: { state: 'idle' } }
  const requests: string[] = []
  const feed = 'https://synthetic.invalid/feed'
  const imports: Record<string, unknown> = {
    electron: { app: { getVersion: () => '1.0.0', isPackaged: false } },
    '../../../build/release-trust.json': {},
    '../../agent/update-checker': {
      emptyUpdateStatus: () => ({ state: 'idle' }),
      checkingUpdateStatus: () => ({ state: 'checking' }),
      checkUpdateFeed: async (url: string) => {
        requests.push(url)
        return { state: 'current' }
      },
    },
    '../../agent/update-feed': { resolveUpdateFeedUrl: () => feed },
    '../../agent/security': { assertUpdateFeedUrl: (url: string) => assert.equal(url, feed) },
    '../../store': {},
    '../state': { store },
    './audit': {},
    '../../agent/update-download': {},
    '../../agent/update-trust': {},
    '../../agent/update-install': {},
    './ipc': {},
    './update-profile': {},
  }
  const loaded = { exports: {} as typeof import('./updates') }
  new Function('require', 'module', 'exports', compiled)((id: string) => {
    if (id.startsWith('node:')) return requireBuiltin(id)
    assert.ok(Object.hasOwn(imports, id), `Reviewed updater import ${id}`)
    return imports[id]
  }, loaded, loaded.exports)
  return { ...loaded.exports, settings, requests, feed }
}

async function advance(ms: number) {
  mock.timers.tick(ms)
  // Let the synthetic feed response finish before advancing to another due callback.
  for (let i = 0; i < 12; i++) await Promise.resolve()
}

async function withClock(run: () => Promise<void>) {
  mock.timers.enable({ apis: ['setTimeout', 'setInterval'] })
  try { await run() }
  finally { mock.timers.reset() }
}

await withClock(async () => {
  const update = updater()
  update.startUpdateChecks()
  await advance(100)
  update.settings.autoCheckUpdates = false
  update.startUpdateChecks()
  await advance(6 * 60 * 60_000)
  assert.deepEqual(update.requests, [], 'turning automatic checks off cancels the pending first check and interval')
  await update.checkForAppUpdate()
  assert.deepEqual(update.requests, [update.feed], 'a manual check still works when automatic checks are off')
})

await withClock(async () => {
  const update = updater()
  update.startUpdateChecks()
  await advance(100)
  // Settings are persisted before the asynchronous post-save work reschedules checks.
  update.settings.autoCheckUpdates = false
  await advance(1500)
  assert.deepEqual(update.requests, [], 'the first callback honors saved Off before rescheduling')
  await advance(6 * 60 * 60_000)
  assert.deepEqual(update.requests, [], 'the periodic callback also honors saved Off before rescheduling')
})

await withClock(async () => {
  const update = updater()
  update.startUpdateChecks()
  await advance(100)
  update.startUpdateChecks()
  await advance(100)
  update.startUpdateChecks()
  await advance(1300)
  await advance(100)
  assert.deepEqual(update.requests, [], 'staggered settings saves cancel the older first checks')
  await advance(100)
  assert.deepEqual(update.requests, [update.feed], 'only the latest first check reaches the feed')
  await advance(6 * 60 * 60_000 - 1500)
  assert.deepEqual(update.requests, [update.feed, update.feed], 'automatic checks keep their six-hour interval')
})

console.log('Automatic update checks honor saved preferences and coalesce settings reschedules')
