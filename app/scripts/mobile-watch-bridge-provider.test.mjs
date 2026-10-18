import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const ts = require('typescript')
const appRoot = path.resolve('.')
const sourceRoot = path.join(appRoot, 'mobile/src/lib')
const device = { baseUrl: 'https://example.invalid', token: 'fictional-watch-token', sessionId: 'fixture-session-a', label: 'Fixture phone', certificatePin: 'a'.repeat(64) }
const caps = { taskStart: true, taskGuide: true, taskStop: true, approvalResolve: true, transcriptReview: true, changesReview: true, engineControl: false }
const host = { snapshot: { serverName: 'Fixture Mac', activeThreadIds: ['fixture-job'], threads: [{ id: 'fixture-job', title: 'Fixture job', kind: 'task', providerId: 'fixture-provider', model: 'fixture-model', status: 'running', updatedAt: 100, timeline: [{ id: 'fixture-message', kind: 'assistant', text: 'Fixture response' }] }], work: { jobs: { running: 1, queued: 0 }, approvals: { pending: 0 } } }, status: { capabilities: caps, engine: null } }
const sessionState = (overrides = {}) => ({ session: device, ready: true, host, error: null, lastSynced: 100, ...overrides })
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no }); return { promise, resolve, reject } }

function harness({ session = sessionState(), appState = 'active', configured = device.sessionId, status, configure, oldNative = false } = {}) {
  const slots = []; let cursor = 0; let pending = []; let dirty = true; let mounted = true; let value; let appListener
  let currentSession = session; let nativeIdentity = configured
  const calls = { disables: 0, statuses: 0, configured: [], published: [], afterUnmountUpdates: 0 }
  const same = (a, b) => a && b && a.length === b.length && a.every((x, i) => Object.is(x, b[i]))
  const React = {
    createContext: () => ({ Provider: 'provider' }),
    useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial; return [slots[i], next => { if (!mounted) calls.afterUnmountUpdates++; slots[i] = typeof next === 'function' ? next(slots[i]) : next; dirty = true }] },
    useRef(initial) { const i = cursor++; return slots[i] ??= { current: initial } },
    useEffect(fn, deps) { const i = cursor++; if (!slots[i] || !same(slots[i].deps, deps)) { const previous = slots[i]; slots[i] = { deps }; pending.push(() => { previous?.cleanup?.(); slots[i].cleanup = fn() }) } },
  }
  const port = {
    async backgroundStatus() { calls.statuses++; return status ? status(calls.statuses) : { configured: !!nativeIdentity, sessionId: nativeIdentity } },
    async configureBackground(input) { calls.configured.push(input); if (configure) await configure(input); nativeIdentity = input.sessionId },
    async disable() { calls.disables++; nativeIdentity = null },
    async publish(payload) { calls.published.push(JSON.parse(payload)) },
  }
  const mocks = {
    react: React,
    'react/jsx-runtime': { jsx: (_type, props) => { value = props.value; return props }, jsxs: (_type, props) => props },
    'react-native': { Platform: { OS: 'ios' }, AppState: { currentState: appState, addEventListener: (_event, listener) => { appListener = listener; return { remove() {} } } } },
    expo: { requireOptionalNativeModule: () => oldNative ? { publish() {}, addListener() {} } : port },
    './session': { useSession: () => currentSession },
  }
  const cache = new Map()
  function load(file) {
    if (cache.has(file)) return cache.get(file)
    const module = { exports: {} }; cache.set(file, module.exports)
    const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX } }).outputText
    const localRequire = id => {
      if (id in mocks) return mocks[id]
      // Exercise the real shared snapshot implementation without loading the
      // unrelated foreground relay adapter's network client dependency.
      if (id === './watch-relay') return load(path.join(appRoot, 'shared/watch-relay.ts'))
      return load(path.resolve(path.dirname(file), `${id}.ts`))
    }
    vm.runInNewContext(`(function(require,module,exports){${compiled}\n})`, { Date, Promise, Set })(localRequire, module, module.exports)
    return module.exports
  }
  const { WatchBridgeProvider } = load(path.join(sourceRoot, 'watch-bridge.tsx'))
  const { backgroundWatch } = load(path.join(sourceRoot, 'watch-native.ts'))
  function render() { if (!dirty || !mounted) return; dirty = false; cursor = 0; WatchBridgeProvider({ children: null }); const effects = pending; pending = []; for (const effect of effects) effect() }
  async function settle() { for (let i = 0; i < 24; i++) { render(); await Promise.resolve() } render() }
  render()
  return {
    calls, settle, backgroundWatch,
    get value() { render(); return value }, get nativeIdentity() { return nativeIdentity },
    setSession(next) { currentSession = next; dirty = true; render() },
    setNativeIdentity(next) { nativeIdentity = next },
    active() { appListener('active'); render() }, background() { appListener('background'); render() },
    unmount() { mounted = false; for (const slot of slots) slot?.cleanup?.() },
  }
}

{
  const h = harness({ session: sessionState({ session: null, ready: false, host: null }), appState: 'background' })
  await h.settle()
  assert.equal(h.calls.statuses, 0); assert.equal(h.calls.disables, 0)
  h.setSession(sessionState({ session: null, ready: true, host: null, error: 'Protected data temporarily unavailable' })); await h.settle()
  assert.equal(h.calls.disables, 0, 'Locked/deferred React bootstrap must preserve durable native consent')
  assert.equal(h.nativeIdentity, device.sessionId)
  h.unmount(); assert.equal(h.calls.disables, 0, 'Unmount must not revoke native watch access')
}
{
  const h = harness(); await h.settle()
  assert.equal(h.value.enabled, true); assert.equal(h.value.available, true)
  assert.equal(h.calls.published.length, 1)
  assert.equal(h.calls.published[0].session, device.sessionId)
  assert.equal(h.calls.published[0].jobs[0].id, 'fixture-job')
  assert(!JSON.stringify(h.calls.published).includes(device.token), 'Snapshot must not publish the phone token')
  h.background(); await h.settle()
  const count = h.calls.published.length
  h.setSession(sessionState({ lastSynced: 200 })); await h.settle()
  assert.equal(h.calls.published.length, count, 'Background UI does not publish foreground snapshots')
  assert.equal(h.calls.disables, 0, 'Backgrounding must preserve native relay configuration')
  h.active(); await h.settle(); assert(h.calls.published.length > count)
  h.unmount(); assert.equal(h.calls.disables, 0); assert.equal(h.nativeIdentity, device.sessionId)
}
{
  const h = harness(); await h.settle(); const count = h.calls.published.length
  h.setSession(sessionState({ error: 'Host connection revoked', lastSynced: 200 })); await h.settle()
  assert.equal(h.calls.published.length, count, 'Unauthenticated/error state must not publish')
  h.setNativeIdentity('fixture-session-b')
  h.setSession(sessionState({ lastSynced: 300 })); await h.settle()
  assert.equal(h.calls.published.length, count, 'Native identity guard must reject stale UI account publications')
}
{
  const granting = deferred(); const h = harness({ configured: null, configure: () => granting.promise }); await h.settle()
  const enable = h.value.setEnabled(true); await h.settle()
  assert.equal(h.value.busy, true)
  const revoked = h.backgroundWatch.disable(); await h.settle()
  assert.equal(h.value.enabled, false); assert.equal(h.value.busy, false)
  granting.resolve(); await enable; await revoked; await h.settle()
  assert.equal(h.value.enabled, false, 'Late enable cannot undo an explicit revoke in mounted settings')
  assert.equal(h.nativeIdentity, null)
}
{
  const oldStatus = deferred(); const newStatus = deferred()
  const h = harness({ session: sessionState({ host: null }), status: count => count === 1 ? oldStatus.promise : newStatus.promise })
  await h.settle()
  const next = { ...device, sessionId: 'fixture-session-b', token: 'fictional-other-token' }
  h.setSession(sessionState({ session: next, host: null })); await h.settle()
  oldStatus.resolve({ configured: true, sessionId: device.sessionId }); await h.settle()
  assert.equal(h.value.enabled, false, 'Old identity restore completion must never show consent for the new account')
  newStatus.resolve({ configured: true, sessionId: next.sessionId }); await h.settle()
  assert.equal(h.value.enabled, true)
  assert.equal(h.calls.disables, 0)
}
{
  const h = harness({ oldNative: true }); await h.settle()
  assert.equal(h.value.available, false); assert.equal(h.value.enabled, false)
  await h.value.setEnabled(true); await h.settle()
  assert.equal(h.value.enabled, false)
  assert.match(h.value.error, /native iPhone build/i)
  assert.equal(h.calls.configured.length, 0); assert.equal(h.calls.published.length, 0); assert.equal(h.calls.disables, 0)
}
{
  const pendingStatus = deferred()
  const h = harness({ session: sessionState({ host: null }), status: () => pendingStatus.promise })
  await h.settle(); h.unmount()
  pendingStatus.resolve({ configured: true, sessionId: device.sessionId }); await h.settle()
  assert.equal(h.calls.afterUnmountUpdates, 0, 'Late restore must not write state into an unmounted watch settings provider')
  assert.equal(h.calls.disables, 0, 'Unmounting during restore must preserve durable native access')
  assert.equal(h.nativeIdentity, device.sessionId)
}
console.log('Production WatchBridgeProvider + native controller regressions passed')
