// Run: jiti src/lib/terminal-layout.test.ts
//
// Terminal workspace layout presets (VST-039), renderer side: slot counts, grid styles,
// area mapping, per-project persistence.
import assert from 'node:assert/strict'
import { isLayoutPreset, layoutGridStyle, loadLayoutPreset, paneGridArea, presetSlots, saveLayoutPreset } from './terminal-layout'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('terminal-layout tests:')

// localStorage shim.
const store = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    getItem: (k: string) => store.get(k) ?? null,
    setItem: (k: string, v: string) => { store.set(k, v) },
    removeItem: (k: string) => { store.delete(k) },
  },
})

{
  assert.equal(presetSlots('single'), 1)
  assert.equal(presetSlots('two-columns'), 2)
  assert.equal(presetSlots('top-main'), 3)
  assert.equal(presetSlots('four'), 4)
  assert.equal(isLayoutPreset('four'), true)
  assert.equal(isLayoutPreset('nope'), false)
  assert.equal(isLayoutPreset(3), false)
  ok('presetSlots + isLayoutPreset')
}

{
  assert.deepEqual(layoutGridStyle('two-columns'), { gridTemplateColumns: '1fr 1fr', gridTemplateRows: '1fr' })
  assert.match(String(layoutGridStyle('top-main').gridTemplateAreas), /main main/)
  assert.match(String(layoutGridStyle('left-main').gridTemplateAreas), /main a/)
  // top/left-main map pane 0 → the big area, the rest to the stack.
  assert.equal(paneGridArea('top-main', 0), 'main')
  assert.equal(paneGridArea('top-main', 1), 'a')
  assert.equal(paneGridArea('top-main', 2), 'b')
  assert.equal(paneGridArea('two-columns', 0), '', 'even presets flow, no named area')
  ok('layoutGridStyle + paneGridArea: even grids vs main+stack areas')
}

{
  assert.equal(loadLayoutPreset('proj-1'), 'single', 'default before any save')
  saveLayoutPreset('proj-1', 'four')
  saveLayoutPreset('proj-2', 'left-main')
  assert.equal(loadLayoutPreset('proj-1'), 'four')
  assert.equal(loadLayoutPreset('proj-2'), 'left-main')
  assert.equal(loadLayoutPreset('proj-unknown'), 'single', 'unknown project → default')
  // A garbage value in storage falls back to single, never throws.
  store.set('vontra.terminal.layout', '{"proj-1":"bogus"}')
  assert.equal(loadLayoutPreset('proj-1'), 'single')
  store.set('vontra.terminal.layout', 'not json')
  assert.equal(loadLayoutPreset('proj-1'), 'single')
  ok('load/saveLayoutPreset: per-project persistence, garbage-safe')
}

console.log(`  ${passed} passed`)
