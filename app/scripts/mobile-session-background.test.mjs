import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const ts = require('typescript')
const root = path.resolve('mobile/src/lib')
const saved = { baseUrl: 'https://example.invalid', token: 'fictional-device-token', sessionId: 'fictional-session', label: 'Test phone' }
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no }); return { promise, resolve, reject } }

// Execute the production provider, including effects and event callbacks, without a
// phone, real Keychain, network, native module or token. Hook slots retain state
// and dependencies across renders; async event completions trigger rerenders.
function harness({ read = async () => JSON.stringify(saved), nativeDisable = async () => {}, pairing = async () => saved, write = async () => {}, appState = 'background' } = {}) {
  const slots = []; let cursor = 0; let pending = []; let dirty = true; let value; let appListener
  const calls = { reads: 0, deletes: 0, writes: [], disables: 0, pairs: 0, events: [] }
  const same = (a, b) => a && b && a.length === b.length && a.every((x, i) => Object.is(x, b[i]))
  const React = {
    createContext: () => ({ Provider: 'provider' }),
    useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial; return [slots[i], next => { slots[i] = typeof next === 'function' ? next(slots[i]) : next; dirty = true }] },
    useRef(initial) { const i = cursor++; return slots[i] ??= { current: initial } },
    useMemo(fn, deps) { const i = cursor++; if (!slots[i] || !same(slots[i].deps, deps)) slots[i] = { deps, value: fn() }; return slots[i].value },
    useCallback(fn, deps) { return React.useMemo(() => fn, deps) },
    useEffect(fn, deps) { const i = cursor++; if (!slots[i] || !same(slots[i].deps, deps)) { const previous = slots[i]; slots[i] = { deps }; pending.push(() => { previous?.cleanup?.(); slots[i].cleanup = fn() }) } },
  }
  const mocks = {
    react: React,
    'react/jsx-runtime': { jsx: (_type, props) => { value = props.value; return props }, jsxs: (_type, props) => props },
    'react-native': { Platform: { OS: 'ios' }, AppState: { currentState: appState, addEventListener: (_event, listener) => { appListener = listener; return { remove() {} } } }, Linking: { getInitialURL: async () => null, addEventListener: () => ({ remove() {} }) } },
    'expo-secure-store': { WHEN_UNLOCKED_THIS_DEVICE_ONLY: 'unlocked-device-only', getItemAsync: async () => { calls.reads++; return read() }, deleteItemAsync: async () => { calls.deletes++; calls.events.push('delete') }, setItemAsync: async (_key, raw, opts) => { calls.writes.push({ raw, opts }); calls.events.push('write'); await write() } },
    './pinned-fetch': { fetchForPin: () => () => { throw Error('Unexpected network') } },
    './watch-native': { disableBackgroundWatch: async () => { calls.disables++; calls.events.push('disable'); await nativeDisable() } },
    './client': { validateHost: value => { if (!value.startsWith('https://')) throw Error('Invalid host'); return value }, HostError: class HostError extends Error {}, StudioClient: class StudioClient { static async pair(...args) { calls.pairs++; calls.events.push('pair'); return pairing(...args) } async host() { return { snapshot: {}, status: null } } } },
  }
  const cache = new Map()
  function load(file) {
    if (cache.has(file)) return cache.get(file)
    const module = { exports: {} }; cache.set(file, module.exports)
    const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX } }).outputText
    const localRequire = id => id in mocks ? mocks[id] : load(path.resolve(path.dirname(file), `${id}.ts`))
    vm.runInNewContext(`(function(require,module,exports){${compiled}\n})`, { URL, AbortController, Date, setTimeout: () => 1, clearTimeout() {}, __DEV__: false })(localRequire, module, module.exports)
    return module.exports
  }
  const { SessionProvider } = load(path.join(root, 'session.tsx'))
  function render() { if (!dirty) return; dirty = false; cursor = 0; SessionProvider({ children: null }); const effects = pending; pending = []; for (const effect of effects) effect() }
  async function settle() { for (let i = 0; i < 12; i++) { render(); await Promise.resolve() } render() }
  render()
  return { calls, settle, get value() { render(); return value }, active() { appListener('active'); render() }, background() { appListener('background'); render() } }
}

{
  let locked = true
  const h = harness({ read: async () => { if (locked) throw Error('Protected data unavailable'); return JSON.stringify(saved) } })
  await h.settle()
  assert.equal(h.calls.deletes, 0, 'Locked background wake must never delete the saved phone token')
  assert.equal(h.calls.disables, 0, 'Locked background wake must leave the native watch relay running')
  assert.equal(h.value.session, null)
  assert.match(h.value.error, /unlock|available|temporarily/i)
  locked = false; h.active(); await h.settle()
  assert.equal(h.calls.reads, 2, 'Unlock should retry the deferred read')
  assert.equal(h.value.session.token, saved.token)
}
{
  for (const raw of [null, '{broken', JSON.stringify({ ...saved, token: '' })]) {
    const h = harness({ read: async () => raw }); await h.settle()
    assert.equal(h.calls.deletes, 0, 'Invalid data is retained for explicit repair')
    assert.equal(h.calls.disables, 1, 'Successfully read absent/invalid session fails closed for native relay')
    assert.equal(h.value.session, null)
  }
}
{
  const read = deferred(); const h = harness({ read: () => read.promise })
  const forgotten = h.value.forget(); await forgotten; read.resolve(JSON.stringify(saved)); await h.settle()
  assert.equal(h.value.session, null, 'Late restore cannot resurrect an explicitly forgotten session')
  assert.equal(h.calls.deletes, 1); assert.equal(h.calls.disables, 1)
}
{
  const read = deferred(); const next = { ...saved, token: 'fictional-new-token', sessionId: 'fictional-new-session' }
  const h = harness({ read: () => read.promise, pairing: async () => next })
  await h.value.pair('https://example.invalid', 'fictional-onetime-code'); read.resolve(JSON.stringify(saved)); await h.settle()
  assert.equal(h.value.session.token, next.token, 'Late restore cannot overwrite a new pairing')
  assert.deepEqual(h.calls.events, ['disable', 'pair', 'write'])
  assert.equal(h.calls.writes[0].opts.keychainAccessible, 'unlocked-device-only')
}
{
  const h = harness({ nativeDisable: async () => { throw Error('Native Keychain unavailable') } }); await h.settle()
  await h.value.forget(); await h.settle()
  assert.equal(h.calls.deletes, 1, 'Phone token must still clear if native relay clear fails')
  assert.match(h.value.error, /watch|revoke/i, 'Failed relay cleanup must remain visible')
  await assert.rejects(h.value.pair('https://example.invalid', 'fictional-onetime-code'), /watch|revoke/i)
  assert.equal(h.calls.pairs, 0, 'New pairing is blocked until old relay access is cleared')
}
{
  const pairing = deferred(); const h = harness({ pairing: () => pairing.promise }); await h.settle()
  const pending = h.value.pair('https://example.invalid', 'fictional-onetime-code'); await h.settle()
  await h.value.forget(); pairing.resolve(saved)
  await assert.rejects(pending, /changed|cancel/i); await h.settle()
  assert.equal(h.value.session, null)
  assert.equal(h.calls.writes.length, 0, 'Late pairing must not persist a token after forget')
}
{
  const pendingWrite = deferred(); const h = harness({ write: () => pendingWrite.promise }); await h.settle()
  const pendingPair = h.value.pair('https://example.invalid', 'fictional-onetime-code'); await h.settle()
  assert.equal(h.calls.writes.length, 1)
  const forgetting = h.value.forget(); await h.settle()
  assert.equal(h.calls.deletes, 0, 'Forget waits for an in-flight secure write before deletion')
  pendingWrite.resolve(); await forgetting
  await assert.rejects(pendingPair, /changed|cancel/i); await h.settle()
  assert.equal(h.calls.deletes, 1); assert.equal(h.value.session, null)
  assert.equal(h.calls.events.at(-1), 'delete')
}
{
  const read = deferred(); const h = harness({ read: () => read.promise })
  h.active(); read.resolve(JSON.stringify(saved)); await h.settle()
  assert.equal(h.value.session.token, saved.token, 'Foreground polling cancellation must not invalidate identity restoration')
}
{
  const h = harness({ read: async () => null, nativeDisable: async () => { throw Error('Native clear failed') } })
  await h.settle(); assert.equal(h.value.ready, true)
  assert.match(h.value.error, /watch|revoke/i)
  assert.equal(h.calls.deletes, 0)
}
console.log('mobile session background production-provider regressions passed')
