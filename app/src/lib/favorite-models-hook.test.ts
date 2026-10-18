import { reviewedLanguageImport } from '../components/preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { webcrypto } from 'node:crypto'
import ts from 'typescript'

const original = Object.fromEntries(['window', 'location', 'localStorage', 'crypto'].map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]))
const values = new Map<string, string>()
const surface = Object.assign(new EventTarget(), { tensorfoldStudio: { platform: 'darwin' }, vontra: undefined }) as any
let credential: string | null = null
let ownerId: string | null = null
let readsFail = false, writes = 0
let dirty = false
let cursor = 0
let effects: (() => void)[] = []
const slots: any[] = []
const same = (a: any[] | undefined, b: any[] | undefined) => !!a && !!b && a.length === b.length && a.every((value, index) => Object.is(value, b[index]))
const react = {
  useState(initial: any) { const index = cursor++; if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial; return [slots[index], (value: any) => { const next = typeof value === 'function' ? value(slots[index]) : value; if (!Object.is(next, slots[index])) { slots[index] = next; dirty = true } }] },
  useRef(initial: any) { const index = cursor++; if (!(index in slots)) slots[index] = { current: initial }; return slots[index] },
  useMemo(build: any, dependencies: any[]) { const index = cursor++; if (!same(slots[index]?.dependencies, dependencies)) slots[index] = { value: build(), dependencies }; return slots[index].value },
  useCallback(callback: any, dependencies: any[]) { return react.useMemo(() => callback, dependencies) },
  useEffect(effect: any, dependencies: any[]) {
    const index = cursor++
    if (same(slots[index]?.dependencies, dependencies)) return
    effects.push(() => { slots[index]?.cleanup?.(); slots[index] = { dependencies, cleanup: effect() } })
  },
  useSyncExternalStore(subscribe: any, get: any) {
    const index = cursor++
    if (slots[index]?.subscribe !== subscribe) { slots[index]?.cleanup?.(); slots[index] = { subscribe, cleanup: subscribe(() => { dirty = true }) } }
    return get()
  },
}
const module = { exports: {} as any }
const reviewedImports: Record<string, any> = { react, './remote-pairing': { pairedToken: () => credential } }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'favorite-models.ts'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, reviewedImports); if (reviewed.matched) return reviewed.value;  assert.ok(id in reviewedImports, id); return reviewedImports[id] }, module, module.exports)
const { useFavoriteModels } = module.exports
const routeA = { providerId: 'provider-a', accountId: null, model: 'model-a' }
const routeB = { providerId: 'provider-a', accountId: 'work', model: 'model-a' }
let current: any
function render() { cursor = 0; dirty = false; current = useFavoriteModels(ownerId); const scheduled = effects; effects = []; for (const effect of scheduled) effect(); return current }
async function settle() { for (let index = 0; index < 20; index++) { await new Promise((resolve) => setTimeout(resolve, 1)); if (dirty) render() } return current }
function event(key: string | null) { const signal = new Event('storage'); Object.defineProperty(signal, 'key', { value: key }); surface.dispatchEvent(signal) }
try {
  Object.defineProperty(globalThis, 'window', { configurable: true, value: surface })
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { origin: 'https://host-one.example.test' } })
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: (key: string) => { if (readsFail) throw new Error('Synthetic getter failure'); return values.get(key) ?? null }, setItem: (key: string, value: string) => { writes++; values.set(key, value) } } })
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: webcrypto })
  render(); assert.equal(current.ready, false)
  await settle(); assert.equal(current.ready, true)
  assert.equal(current.toggle(routeA), true); render()
  assert.equal(current.isFavorite(routeA), true)
  const oldLocalToggle = current.toggle
  ownerId = 'remote-owner'
  render(); assert.equal(current.ready, false); assert.deepEqual(current.favorites, [])
  assert.equal(oldLocalToggle(routeB), false, 'a callback captured on another owning host cannot change its preferences after navigation')
  await settle(); assert.equal(current.ready, true); assert.deepEqual(current.favorites, [])
  current.toggle(routeB); render(); assert.equal(current.isFavorite(routeA), false); assert.equal(current.isFavorite(routeB), true)
  ownerId = null; render(); await settle(); assert.equal(current.isFavorite(routeA), true); assert.equal(current.isFavorite(routeB), false)
  const desktopKey = [...values.keys()].find((key) => key.includes('["desktop",null]'))!
  values.set(desktopKey, JSON.stringify([routeB])); event(desktopKey); render()
  assert.equal(current.ready, true, 'ordinary favourite storage events do not reset host/session scope')
  assert.equal(current.isFavorite(routeB), true); assert.equal(current.isFavorite(routeA), false)
  const previousWrites = writes, previousSaved = values.get(desktopKey)
  readsFail = true
  assert.equal(current.toggle(routeA), false); render()
  assert.match(current.error, /could not be read.*unchanged/)
  assert.equal(current.isFavorite(routeB), true, 'the hook keeps its last known view during a transient getter failure')
  assert.equal(current.isFavorite(routeA), false)
  assert.equal(writes, previousWrites); assert.equal(values.get(desktopKey), previousSaved, 'visible read failure blocks an otherwise working setter')
  readsFail = false

  surface.tensorfoldStudio = undefined
  credential = 'synthetic-device-token-one'
  render(); assert.deepEqual(current.favorites, []); await settle()
  assert.equal(current.ready, true); current.toggle(routeA); render(); assert.equal(current.isFavorite(routeA), true)
  const oldPairToggle = current.toggle
  credential = 'synthetic-device-token-two'
  render(); assert.equal(current.ready, false); assert.deepEqual(current.favorites, [])
  assert.equal(oldPairToggle(routeB), false)
  await settle(); assert.equal(current.ready, true); assert.deepEqual(current.favorites, [])
  current.toggle(routeB); render(); assert.equal(current.isFavorite(routeB), true)
  surface.dispatchEvent(new Event('vontra:paired-auth-failed'))
  render(); await settle(); assert.equal(current.ready, false); assert.deepEqual(current.favorites, [])
  assert.match(current.error, /unavailable/)
  assert.equal(current.toggle(routeA), false)
  credential = null; event('vontra.paired.token'); render(); await settle()
  assert.equal(current.ready, false, 'a revoked/cleared pairing never falls back into browser demo favourites')
  credential = 'synthetic-device-token-three'; event('vontra.paired.token'); render(); await settle()
  assert.equal(current.ready, true); assert.deepEqual(current.favorites, [])
  current.toggle(routeA); render()
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { origin: 'https://host-two.example.test' } })
  render(); await settle(); assert.deepEqual(current.favorites, [])
  assert.ok([...values.keys()].every((key) => !key.includes('synthetic-device-token')), 'no paired credential is copied to preference keys')

  let resolveHash!: (value: ArrayBuffer) => void
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: { subtle: { digest: () => new Promise<ArrayBuffer>((resolve) => { resolveHash = resolve }) } } })
  credential = 'synthetic-delayed-token'; render(); assert.equal(current.ready, false)
  const staleHash = resolveHash
  surface.tensorfoldStudio = { platform: 'darwin' }; credential = null; render(); await settle()
  assert.equal(current.isFavorite(routeB), true)
  staleHash(new Uint8Array(32).buffer); await settle()
  assert.equal(current.isFavorite(routeB), true, 'a delayed old pairing hash cannot replace a newer desktop scope')
  assert.equal(current.isFavorite(routeA), false)
  console.log('Actual favourites hook hides stale host/account routes immediately, rejects old callbacks, responds to cross-window updates and revocation, and discards delayed pairing work.')
} finally {
  for (const slot of slots) slot?.cleanup?.()
  for (const [name, descriptor] of Object.entries(original)) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else delete (globalThis as any)[name] }
}
