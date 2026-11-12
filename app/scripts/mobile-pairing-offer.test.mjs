import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import vm from 'node:vm'
import { createRequire } from 'node:module'
const require = createRequire(import.meta.url)
const ts = require('typescript')
const src = path.resolve('mobile/src')
const saved = { baseUrl: 'https://example.invalid', token: 'fictional-device-token', sessionId: 'fictional-session', label: 'Test phone' }
const outsideHost = 'https://outside.invalid'
const link = `${outsideHost}/#pair=fictional-onetime-code`
const deepLink = `tensorfold-mobile://pair?link=${encodeURIComponent(link)}`

// Hook slots for one component: state, refs and effect dependencies persist across its renders.
function hooks() {
  const slots = []; let cursor = 0; let pending = []; let dirty = true; let last
  const same = (a, b) => a && b && a.length === b.length && a.every((x, i) => Object.is(x, b[i]))
  const React = {
    createContext: () => ({ Provider: 'provider' }),
    useState(initial) { const i = cursor++; if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial; return [slots[i], next => { slots[i] = typeof next === 'function' ? next(slots[i]) : next; dirty = true }] },
    useRef(initial) { const i = cursor++; return slots[i] ??= { current: initial } },
    useMemo(fn, deps) { const i = cursor++; if (!slots[i] || !same(slots[i].deps, deps)) slots[i] = { deps, value: fn() }; return slots[i].value },
    useCallback(fn, deps) { return React.useMemo(() => fn, deps) },
    useEffect(fn, deps) { const i = cursor++; if (!slots[i] || !same(slots[i].deps, deps)) { const previous = slots[i]; slots[i] = { deps }; pending.push(() => { previous?.cleanup?.(); slots[i].cleanup = fn() }) } },
  }
  return { React, touch() { dirty = true }, run(component) { for (let i = 0; dirty && i < 20; i++) { dirty = false; cursor = 0; last = component(); const effects = pending; pending = []; for (const effect of effects) effect() } return last } }
}

// Runs production TypeScript with the given modules replaced; every other relative import is the real file.
function loader(mocks) {
  const cache = new Map()
  return function load(file) {
    if (cache.has(file)) return cache.get(file)
    const module = { exports: {} }; cache.set(file, module.exports)
    const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, jsx: ts.JsxEmit.ReactJSX } }).outputText
    const real = id => { const base = path.resolve(path.dirname(file), id); return load(fs.existsSync(`${base}.ts`) ? `${base}.ts` : `${base}.tsx`) }
    vm.runInNewContext(`(function(require,module,exports){${compiled}\n})`, { URL, URLSearchParams, AbortController, TextDecoder, Date, setTimeout: () => 1, clearTimeout() {}, __DEV__: false })(id => id in mocks ? mocks[id] : real(id), module, module.exports)
    return module.exports
  }
}

const nodes = tree => tree == null || typeof tree !== 'object' ? [] : Array.isArray(tree) ? tree.flatMap(nodes) : [tree, ...nodes(tree.props?.children)]
const words = tree => typeof tree === 'string' ? [tree] : tree == null || typeof tree !== 'object' ? [] : Array.isArray(tree) ? tree.flatMap(words) : words(tree.props?.children)

// The production session provider and pairing screen on a phone with no network, Keychain or native module.
// `stored` is the saved device session, `initialUrl` the link that launched the app.
function phone({ stored = null, initialUrl = null } = {}) {
  const provider = hooks(); const form = hooks(); let value; let tree; let urlListener
  const requests = []
  const answer = { ok: true, status: 200, url: '', headers: { get: name => name === 'content-type' ? 'application/json' : null }, text: async () => JSON.stringify({ token: 'fictional-new-token', sessionId: 'fictional-new-session' }) }
  const { SessionProvider } = loader({
    react: provider.React,
    'react/jsx-runtime': { jsx: (_type, props) => { value = props.value; return props } },
    'react-native': { Platform: { OS: 'ios' }, AppState: { currentState: 'background', addEventListener: () => ({ remove() {} }) }, Linking: { getInitialURL: async () => initialUrl, addEventListener: (_event, listener) => { urlListener = listener; return { remove() {} } } } },
    'expo-secure-store': { WHEN_UNLOCKED_THIS_DEVICE_ONLY: 'unlocked-device-only', getItemAsync: async () => stored, deleteItemAsync: async () => {}, setItemAsync: async () => {} },
    './pinned-fetch': { fetchForPin: () => async url => { requests.push(String(url)); return answer } },
    './watch-native': { disableBackgroundWatch: async () => {} },
  })(path.join(src, 'lib/session.tsx'))
  const element = (type, props) => ({ type, props })
  const { Pairing } = loader({
    react: form.React,
    'react/jsx-runtime': { jsx: element, jsxs: element, Fragment: 'Fragment' },
    'react-native': { KeyboardAvoidingView: 'KeyboardAvoidingView', Platform: { OS: 'ios' }, ScrollView: 'ScrollView', View: 'View' },
    'expo-router': { router: { replace() {}, push() {} } },
    'react-native-safe-area-context': { SafeAreaView: 'SafeAreaView' },
    '../lib/session': { useSession: () => value },
    './ui': { Body: 'Body', Brand: 'Brand', Button: 'Button', Field: 'Field', Heading: 'Heading', Notice: 'Notice', layout: { page: {}, content: {}, card: {} }, useColors: () => ({}) },
  })(path.join(src, 'components/Pairing.tsx'))
  const find = id => nodes(tree).find(node => node.props?.testID === id)
  return {
    requests, find,
    get value() { return value },
    // Renders both until nothing changes; a macrotask between rounds lets every pending promise finish.
    async settle() { for (let i = 0; i < 4; i++) { provider.run(() => SessionProvider({ children: null })); form.touch(); tree = form.run(() => Pairing()); await new Promise(resolve => setImmediate(resolve)) } },
    open(url) { urlListener({ url }) },
    field: id => find(id).props.value,
    type(id, text) { find(id).props.onChangeText(text) },
    press(id) { find(id).props.onPress() },
    shows: text => words(tree).includes(text),
  }
}

{
  // The app is launched by a link another app or page opened, on a phone that is not paired.
  const p = phone({ initialUrl: deepLink }); await p.settle()
  assert.equal(p.value.pairingLink, '', 'A link from outside the app must not become the pairing link by itself')
  assert.equal(p.field('pair-code'), '', 'A link from outside the app must not fill in the one-time code')
  assert.equal(p.field('pair-host'), '', 'A link from outside the app must not fill in the host')
  assert.equal(p.value.offer.baseUrl, outsideHost)
  assert.ok(p.find('pair-offer'), 'The pairing screen asks about the link')
  assert.ok(p.shows(outsideHost), 'The question names the host the link asks this phone to connect to')
  assert.deepEqual(p.requests, [], 'Nothing is sent to the host before the user answers')
  p.press('pair-offer-accept'); await p.settle()
  assert.equal(p.value.offer, null); assert.equal(p.find('pair-offer'), undefined)
  assert.equal(p.field('pair-code'), link, 'An accepted link fills in the form')
  assert.equal(p.field('pair-host'), outsideHost)
  assert.deepEqual(p.requests, [], 'Accepting the link still leaves Connect to the user')
  p.press('pair-submit'); await p.settle()
  assert.deepEqual(p.requests, [`${outsideHost}/v1/auth/pair`])
  assert.equal(p.value.session.baseUrl, outsideHost)
}
{
  // The link arrives while the app is open, and the user turns it down.
  const p = phone(); await p.settle()
  p.open(deepLink); await p.settle()
  assert.equal(p.value.offer.link, link); assert.ok(p.shows(outsideHost))
  p.press('pair-offer-decline'); await p.settle()
  assert.equal(p.value.offer, null); assert.equal(p.find('pair-offer'), undefined)
  assert.equal(p.value.pairingLink, ''); assert.equal(p.field('pair-code'), ''); assert.equal(p.field('pair-host'), '')
}
{
  // What the user typed stays until they accept the link.
  const p = phone(); await p.settle()
  p.type('pair-host', saved.baseUrl); p.type('pair-code', 'fictional-typed-code'); await p.settle()
  p.open(deepLink); await p.settle()
  assert.equal(p.field('pair-host'), saved.baseUrl, 'A link from outside the app must not replace the host the user typed')
  assert.equal(p.field('pair-code'), 'fictional-typed-code')
  // Pairing with the typed host answers the question: the link is gone, and forgetting the host does not bring it back.
  p.press('pair-submit'); await p.settle()
  assert.deepEqual(p.requests, [`${saved.baseUrl}/v1/auth/pair`])
  assert.equal(p.value.session.baseUrl, saved.baseUrl); assert.equal(p.value.offer, null)
  await p.value.forget(); await p.settle()
  assert.equal(p.value.offer, null, 'A link the user never accepted must not return after the phone forgets its host')
}
{
  // A paired phone takes no pairing link from outside, at launch or later, and none is waiting once it forgets its host.
  const p = phone({ stored: JSON.stringify(saved), initialUrl: deepLink }); await p.settle()
  assert.equal(p.value.session.token, saved.token); assert.equal(p.value.offer, null); assert.equal(p.value.pairingLink, '')
  p.open(deepLink); await p.settle()
  assert.equal(p.value.offer, null); assert.equal(p.value.pairingLink, '')
  await p.value.forget(); await p.settle()
  assert.equal(p.value.session, null); assert.equal(p.value.offer, null)
  assert.equal(p.find('pair-offer'), undefined); assert.equal(p.field('pair-code'), ''); assert.equal(p.field('pair-host'), '')
}
{
  // A link that could not pair is shown with its reason and no way to accept it.
  const unusable = { [`${outsideHost}/#pair=fictional-code&expires=1`]: /expired/i, 'http://outside.invalid/#pair=fictional-code': /https/i, [`${outsideHost}/#pair=fictional-code&pin=short`]: /pin/i, 'ftp://outside.invalid/#pair=fictional-code': /invalid/i }
  for (const [bad, reason] of Object.entries(unusable)) for (const url of [bad, `tensorfold-mobile://pair?link=${encodeURIComponent(bad)}`]) {
    const p = phone(); await p.settle()
    p.open(url); await p.settle()
    assert.match(p.value.offer.problem, reason, url)
    assert.ok(p.shows(p.value.offer.problem), 'The pairing screen says why the link cannot be used')
    assert.equal(p.find('pair-offer-accept'), undefined); assert.equal(p.value.pairingLink, ''); assert.equal(p.field('pair-code'), '')
    p.press('pair-offer-decline'); await p.settle()
    assert.equal(p.value.offer, null)
  }
}
{
  // Links that are not pairing links ask nothing: a route inside the app, a link with no code, and text that is no URL.
  for (const url of ['tensorfold-mobile://host', 'tensorfold-mobile://pair?link=fictional-code', `tensorfold-mobile://pair?link=${encodeURIComponent(outsideHost)}`, outsideHost, 'not a link']) {
    const p = phone(); await p.settle()
    p.open(url); await p.settle()
    assert.equal(p.value.offer, null, url); assert.equal(p.value.pairingLink, ''); assert.equal(p.find('pair-offer'), undefined)
  }
}
console.log('mobile pairing: a link from outside the app is only an offer until the user accepts the host it names')
