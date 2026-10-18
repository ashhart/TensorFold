import { reviewedLanguageImport } from '../../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import * as controllerModule from './benchmark-controller'
import { fixtureRun } from './benchmark-test-fixtures'

const originalTimeout = globalThis.setTimeout, originalClear = globalThis.clearTimeout
const timers = new Map<number, () => Promise<void>>()
let timerId = 0
;(globalThis as any).setTimeout = (callback: () => Promise<void>) => { const id = ++timerId; timers.set(id, callback); return id }
;(globalThis as any).clearTimeout = (id: number) => { timers.delete(id) }
const slots: any[] = []
let cursor = 0
let queued: Array<() => void> = []
const calls: any[] = []
let latest: ReturnType<typeof fixtureRun> | null = null
const react = {
  useState: (initial: any) => { const index = cursor++; if (!(index in slots)) slots[index] = initial; return [slots[index], (next: any) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }] },
  useRef: (initial: unknown) => { const index = cursor++; if (!(index in slots)) slots[index] = { current: initial }; return slots[index] },
  useEffect: (work: () => any, deps: unknown[]) => { const index = cursor++; const old = slots[index]; if (!old || deps.some((value, i) => value !== old.deps[i])) queued.push(() => { old?.cleanup?.(); slots[index] = { deps, cleanup: work() } }) },
}
const api = {
  engineBenchmarkStatus: async (id?: string) => { calls.push(['status', id]); return latest }, engineBenchmarkHistory: async () => { calls.push(['history']); return latest && latest.state === 'complete' ? [latest] : [] },
  engineStartBenchmark: async (input: any) => { calls.push(['start', input]); latest = fixtureRun({ ...input, serial: !!input.serial, state: 'preparing' }); return latest },
  engineCancelBenchmark: async (id: string) => { calls.push(['cancel', id]); return true }, engineExportBenchmark: async (id: string) => { calls.push(['export', id]); return { saved: false } },
  writeClipboard: async (text: string) => { calls.push(['copy', text]); return { ok: true } },
}
const compiled = ts.transpileModule(readFileSync(path.join(__dirname, 'useModelBenchmark.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const imports: Record<string, unknown> = { react, '../../../lib/ipc': { api }, './benchmark-controller': controllerModule }
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', compiled)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports); return imports[name] }, loaded, loaded.exports)
const render = () => { cursor = 0; queued = []; const value = loaded.exports.useModelBenchmark(); for (const work of queued) work(); return value }
async function settle() { for (let i = 0; i < 12; i++) await Promise.resolve() }
try {
  let hook = render(); await settle(); hook = render()
  assert.deepEqual(calls.map((call) => call[0]), ['status', 'history'])
  assert.equal(timers.size, 0, 'idle history does not create a polling loop')
  await hook.controller.start({ id: 'benchmark_hook_fixture_012345', model: 'fixture/cached-model', serial: false }); hook = render()
  assert.equal(timers.size, 1, 'one active job owns one serial poll timer')
  const poll = timers.values().next().value!; timers.clear(); await poll(); hook = render()
  assert.equal(timers.size, 1, 'the next poll is scheduled after the previous read completes')
  await hook.controller.cancel()
  assert.deepEqual(calls.filter((call) => call[0] === 'cancel'), [['cancel', latest!.id]])
  latest = fixtureRun({ id: latest!.id, revision: 10 })
  const finish = timers.values().next().value!; timers.clear(); await finish(); hook = render()
  assert.equal(timers.size, 0, 'terminal status ends the polling loop')
  await hook.controller.export(); await hook.controller.copy()
  assert.deepEqual(calls.find((call) => call[0] === 'export'), ['export', latest.id])
  assert.match(calls.find((call) => call[0] === 'copy')[1], /community-v1/)
  assert.match(hook.controller.snapshot().notice, /copied/)
  slots.forEach((slot) => slot?.cleanup?.())
  assert.equal(calls.filter((call) => call[0] === 'cancel').length, 1, 'hook cleanup does not invoke engine cancellation')
  console.log('Benchmark hook API adapter, read-only mount, one poll timer, exact cancel/export/copy routing and no cancel on cleanup passed')
} finally { slots.forEach((slot) => slot?.cleanup?.()); globalThis.setTimeout = originalTimeout; globalThis.clearTimeout = originalClear }
