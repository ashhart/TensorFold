import assert from 'node:assert/strict'
import { createServer, type Server } from 'node:http'
import { IosWdaProvider, iosPng, normalizeIosEndpoint, parseIosSource } from './ios-wda'

const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jVSkAAAAASUVORK5CYII='
const source = '<?xml version="1.0"?><AppiumAUT><XCUIElementTypeApplication x="0" y="0" width="390" height="844"><XCUIElementTypeButton label="Open &amp; close" enabled="true" visible="true" x="20" y="40" width="100" height="50"/><XCUIElementTypeSecureTextField label="private-password" value="do-not-return" x="0" y="100" width="200" height="30" visible="false"/></XCUIElementTypeApplication></AppiumAUT>'

async function server(handler: (method: string, url: string, body: any) => { status?: number; headers?: Record<string, string>; value?: unknown; sessionId?: string | null; raw?: string; delay?: number }): Promise<{ server: Server; endpoint: string; calls: Array<{ method: string; url: string; body: any }> }> {
  const calls: Array<{ method: string; url: string; body: any }> = []
  const instance = createServer((req, res) => {
    let input = ''
    req.on('data', (chunk) => { input += chunk })
    req.on('end', () => {
      const call = { method: req.method!, url: req.url!, body: input ? JSON.parse(input) : undefined }
      calls.push(call)
      const result = handler(call.method, call.url, call.body)
      const send = () => {
        if (res.destroyed) return
        res.writeHead(result.status ?? 200, { 'Content-Type': 'application/json', ...result.headers })
        res.end(result.raw ?? JSON.stringify({ value: result.value ?? null, sessionId: result.sessionId ?? (call.url.startsWith('/session/ours') ? 'ours' : null) }))
      }
      if (result.delay) setTimeout(send, result.delay); else send()
    })
  })
  await new Promise<void>((resolve) => instance.listen(0, '127.0.0.1', resolve))
  const address = instance.address()
  assert.ok(address && typeof address !== 'string')
  return { server: instance, endpoint: `http://127.0.0.1:${address.port}`, calls }
}
function baseline(_method: string, url: string) {
  if (url === '/status') return { value: { ready: true, os: { name: 'iOS', version: '27.0' } } }
  if (url === '/wda/locked') return { value: false }
  if (url === '/session') return { value: { sessionId: 'ours' }, sessionId: 'ours' }
  if (url.endsWith('/wda/screen')) return { value: { screenSize: { width: 390, height: 844 } } }
  if (url.endsWith('/source')) return { value: source }
  if (url.endsWith('/screenshot')) return { value: png }
  if (url.endsWith('/wda/activeAppInfo')) return { value: { name: 'Sample', bundleId: 'com.example.sample', processArguments: { env: { SECRET: 'omit' } } } }
  return { value: null }
}
async function close(instance: Server) { instance.closeAllConnections(); await new Promise<void>((resolve) => instance.close(() => resolve())) }

async function main() {
  assert.equal(normalizeIosEndpoint('http://localhost:8100/'), 'http://127.0.0.1:8100')
  assert.equal(normalizeIosEndpoint('http://192.168.1.10:8100', 'wifi'), 'http://192.168.1.10:8100')
  assert.equal(normalizeIosEndpoint('https://[fd00::1]:8100', 'wifi'), 'https://[fd00::1]:8100')
  for (const endpoint of ['https://example.com:8100', 'http://8.8.8.8:8100', 'http://169.254.169.254:8100', 'http://127.0.0.1:22', 'http://127.0.0.1:8100/path', 'http://127.0.0.1:8100/?x=y', 'http://user:secret@127.0.0.1:8100', 'http://127.0.0.1:8100#x', 'http://2130706433:8100', 'http://0177.0.0.1:8100', 'http://[::ffff:127.0.0.1]:8100']) assert.throws(() => normalizeIosEndpoint(endpoint, 'wifi'))
  assert.throws(() => normalizeIosEndpoint('http://10.0.0.2:8100', 'usb'), /loopback forward/)
  assert.equal(iosPng(png).width, 1)
  assert.throws(() => iosPng(Buffer.from('not png').toString('base64')), /PNG|invalid/)
  assert.throws(() => iosPng(`data:image/png;base64,${png}`), /invalid/)
  assert.throws(() => parseIosSource('<!DOCTYPE x [<!ENTITY x SYSTEM "file:///secret">]>', { width: 390, height: 844 }), /unsupported/)
  const parsed = parseIosSource(source.replace('visible="false"', 'visible="true"'), { width: 390, height: 844 })
  assert.equal(parsed.nodes[0].text, 'Open & close')
  assert.equal(parsed.nodes[1].text, '[secure field]')
  assert.ok(!JSON.stringify(parsed).includes('private-password'))
  assert.ok(!JSON.stringify(parsed).includes('do-not-return'))
  assert.equal(parseIosSource('<XCUIElementTypeButton x="-2" y="0" width="10" height="20" label="outside"/>', { width: 100, height: 100 }).nodes.length, 0)
  const focusedSource = '<XCUIElementTypeTextField label="Sample entry" focused="true" enabled="true" visible="true" x="20" y="100" width="200" height="30"/>'
  assert.equal(parseIosSource(focusedSource, { width: 390, height: 844 }).nodes[0].focused, true, 'source reports an explicit focus attribute without inferring absent focus')
  assert.equal(parseIosSource(focusedSource.replace('focused="true"', ''), { width: 390, height: 844 }).nodes[0].focused, undefined)
  assert.equal(parseIosSource('<XCUIElementTypeTextView x="20" y="100" width="200" height="30"/>', { width: 390, height: 844 }).nodes[0].role, 'TextView', 'empty editable text views stay available for focus verification')

  const normal = await server(baseline)
  const provider = new IosWdaProvider({ endpoint: normal.endpoint })
  try {
    assert.equal((await provider.connect()).osVersion, '27.0')
    const observed = await provider.observe()
    assert.equal(observed.dimensions.width, 390)
    assert.equal(observed.nodes.length, 1)
    assert.deepEqual(observed.app, { name: 'Sample', bundleId: 'com.example.sample' })
    assert.ok(!JSON.stringify(observed).includes('SECRET'))
    await provider.act({ action: 'tap', x: 20, y: 30 }, observed.dimensions)
    assert.deepEqual(normal.calls.at(-1), { method: 'POST', url: '/session/ours/wda/tap', body: { x: 20, y: 30 } })
    await provider.act({ action: 'swipe', fromX: 20, fromY: 300, toX: 20, toY: 100 }, observed.dimensions)
    assert.equal(normal.calls.at(-1)?.url, '/session/ours/wda/dragfromtoforduration')
    await provider.act({ action: 'home' }, observed.dimensions)
    assert.equal(normal.calls.at(-1)?.url, '/wda/homescreen')
    await provider.act({ action: 'launch', bundleId: 'com.example.sample' }, observed.dimensions)
    assert.equal(normal.calls.at(-1)?.url, '/session/ours/wda/apps/activate')
    await assert.rejects(() => provider.act({ action: 'tap', x: 390, y: 20 }, observed.dimensions), /inside/)
    await assert.rejects(() => provider.act({ action: 'launch', bundleId: '../../wda/unlock' }, observed.dimensions), /bundle identifier/)
    await assert.rejects(() => provider.act({ action: 'type', text: 'password' }, observed.dimensions), /secure fields/)
    const mutations = normal.calls.filter((call) => call.method === 'POST').length
    await assert.rejects(() => provider.act({ action: 'tap', x: 1, y: 1 }, { width: 1, height: 1 }), /changed size/)
    assert.equal(normal.calls.filter((call) => call.method === 'POST').length, mutations)
    await provider.disconnect()
    assert.equal(normal.calls.at(-1)?.method, 'DELETE')
    assert.equal(normal.calls.at(-1)?.url, '/session/ours')
    const count = normal.calls.length
    await assert.rejects(() => provider.observe(), /Connect your iPhone/)
    assert.equal(normal.calls.length, count, 'disconnected observation does not contact the endpoint')
  } finally { await provider.disconnect(); await close(normal.server) }

  const busy = await server((method, url) => url === '/status' ? { ...baseline(method, url), sessionId: 'somebody-else' } : baseline(method, url))
  try {
    await assert.rejects(() => new IosWdaProvider({ endpoint: busy.endpoint }).connect(), /another active session/)
    assert.deepEqual(busy.calls.map((call) => call.url), ['/status'])
  } finally { await close(busy.server) }

  const locked = await server((method, url) => url === '/wda/locked' ? { value: true } : baseline(method, url))
  try {
    await assert.rejects(() => new IosWdaProvider({ endpoint: locked.endpoint }).connect(), /Unlock your iPhone yourself/)
    assert.ok(!locked.calls.some((call) => call.method === 'POST'))
  } finally { await close(locked.server) }

  const secure = await server((method, url) => url.endsWith('/source') ? { value: source.replace('visible="false"', 'visible="true"') } : baseline(method, url))
  const secureProvider = new IosWdaProvider({ endpoint: secure.endpoint })
  try {
    await secureProvider.connect()
    await assert.rejects(() => secureProvider.observe(), /secure fields/)
    assert.ok(!secure.calls.some((call) => call.url.endsWith('/screenshot')), 'secure trees are refused before a screenshot is requested')
  } finally { await secureProvider.disconnect(); await close(secure.server) }

  const invalidResponses: Array<{ status?: number; headers?: Record<string, string>; raw?: string; value?: unknown }> = [{ status: 302, headers: { Location: 'https://example.com/' } }, { raw: '{"value":', status: 200 }, { headers: { 'Content-Length': String(1024 * 1024) }, raw: '' }, { value: { ready: true, os: { name: 'Android' } } }]
  for (const response of invalidResponses) {
    const invalid = await server(() => response)
    try {
      await assert.rejects(() => new IosWdaProvider({ endpoint: invalid.endpoint }).connect())
      assert.equal(invalid.calls.length, 1)
    } finally { await close(invalid.server) }
  }

  const slow = await server((method, url) => ({ ...baseline(method, url), delay: 200 }))
  try {
    await assert.rejects(() => new IosWdaProvider({ endpoint: slow.endpoint, timeoutMs: 100 }).connect(), /did not respond in time/)
    const controller = new AbortController()
    const running = new IosWdaProvider({ endpoint: slow.endpoint }).connect(controller.signal)
    setTimeout(() => controller.abort(), 20)
    await assert.rejects(() => running, /stopped/)
    const own = new IosWdaProvider({ endpoint: slow.endpoint })
    const pending = own.connect()
    await new Promise((resolve) => setTimeout(resolve, 20))
    await own.disconnect()
    await assert.rejects(() => pending, /stopped/)
  } finally { await close(slow.server) }
  const creating = await server((method, url) => ({ ...baseline(method, url), ...(url === '/session' ? { delay: 120 } : {}) }))
  try {
    const own = new IosWdaProvider({ endpoint: creating.endpoint })
    const controller = new AbortController()
    const result = own.connect(controller.signal)
    const rejected = assert.rejects(() => result, /stopped/)
    while (!creating.calls.some((call) => call.url === '/session')) await new Promise((resolve) => setTimeout(resolve, 5))
    controller.abort()
    await own.disconnect()
    await rejected
    assert.ok(creating.calls.some((call) => call.method === 'DELETE' && call.url === '/session/ours'), 'a cancelled creation releases only the session ID it created')
    assert.ok(!creating.calls.some((call) => call.url.endsWith('/wda/screen') || call.url.endsWith('/source') || call.url.endsWith('/screenshot')), 'cancellation during creation performs cleanup without reading the phone')
  } finally { await close(creating.server) }

  const focusedRoutes = (method: string, url: string) => {
    if (url.endsWith('/source')) return { value: focusedSource }
    if (url.endsWith('/element/active')) return { value: { 'element-6066-11e4-a52e-4f735466cecf': 'active-entry', ELEMENT: 'active-entry' } }
    if (url.endsWith('/element/active-entry/name')) return { value: 'XCUIElementTypeTextField' }
    if (url.endsWith('/element/active-entry/rect')) return { value: { x: 20, y: 100, width: 200, height: 30 } }
    if (url.endsWith('/element/active-entry/enabled') || url.endsWith('/element/active-entry/displayed')) return { value: true }
    return baseline(method, url)
  }
  const focused = await server(focusedRoutes)
  const focusedProvider = new IosWdaProvider({ endpoint: focused.endpoint })
  try {
    await focusedProvider.connect()
    const observed = await focusedProvider.observe()
    assert.equal(observed.nodes[0].focused, true, 'a unique live enabled visible keyboard-focused field allows typing')
    await focusedProvider.act({ action: 'type', text: 'Synthetic sample' }, observed.dimensions)
    assert.equal(focused.calls.at(-1)?.url, '/session/ours/wda/keys')
    assert.deepEqual(focused.calls.at(-1)?.body, { value: [...'Synthetic sample'] })
    assert.ok(!focused.calls.some((call) => /\/(?:text|attribute\/value)$/.test(call.url)), 'focus verification never requests field values')
  } finally { await focusedProvider.disconnect(); await close(focused.server) }

  for (const refusal of ['missing-active', 'duplicate-tree', 'disabled', 'hidden', 'wrong-type', 'malformed-id', 'changed-focus', 'different-bounds'] as const) {
    let activeReads = 0
    const refused = await server((method, url) => {
      if (url.endsWith('/element/active')) {
        activeReads++
        if (refusal === 'missing-active') return { status: 404, value: { error: 'no such element' } }
        if (refusal === 'malformed-id') return { value: { ELEMENT: '../../wda/unlock' } }
        if (refusal === 'changed-focus' && activeReads % 2 === 0) return { value: { ELEMENT: 'other-entry' } }
      }
      if (refusal === 'duplicate-tree' && url.endsWith('/source')) return { value: focusedSource + focusedSource }
      if (refusal === 'disabled' && url.endsWith('/enabled')) return { value: false }
      if (refusal === 'hidden' && url.endsWith('/displayed')) return { value: false }
      if (refusal === 'wrong-type' && url.endsWith('/name')) return { value: 'XCUIElementTypeButton' }
      if (refusal === 'different-bounds' && url.endsWith('/rect')) return { value: { x: 100, y: 100, width: 200, height: 30 } }
      return focusedRoutes(method, url)
    })
    const refusedProvider = new IosWdaProvider({ endpoint: refused.endpoint })
    try {
      await refusedProvider.connect()
      const observed = await refusedProvider.observe()
      assert.ok(observed.nodes.every((node) => node.focused !== true), `${refusal} must not grant typing from XML focus alone`)
      await assert.rejects(() => refusedProvider.act({ action: 'type', text: 'Synthetic sample' }, observed.dimensions), /no verified focused text field/)
      assert.ok(!refused.calls.some((call) => call.url.endsWith('/wda/keys') || call.url.includes('..')), `${refusal} must not send keys or unsafe URL paths`)
    } finally { await refusedProvider.disconnect(); await close(refused.server) }
  }
  const activeSecure = await server((method, url) => url.endsWith('/element/active-entry/name') ? { value: 'XCUIElementTypeSecureTextField' } : focusedRoutes(method, url))
  const activeSecureProvider = new IosWdaProvider({ endpoint: activeSecure.endpoint })
  try {
    await activeSecureProvider.connect()
    await assert.rejects(() => activeSecureProvider.observe(), /secure fields/)
    assert.ok(!activeSecure.calls.some((call) => call.url.endsWith('/screenshot')), 'an active secure field is refused before a screenshot even if tree focus is misleading')
  } finally { await activeSecureProvider.disconnect(); await close(activeSecure.server) }
  console.log('iOS WebDriverAgent provider tests passed')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
