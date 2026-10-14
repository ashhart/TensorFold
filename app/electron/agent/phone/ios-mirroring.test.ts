import assert from 'node:assert/strict'
import { IosMirroringProvider, type PhoneNativeHelper } from './ios-mirroring'

const token = '12345678-1234-1234-1234-123456789abc'
const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jVSkAAAAASUVORK5CYII='

async function main() {
  const calls: Array<{ command: string; params?: Record<string, unknown> }> = []
  let disposed = 0
  let response: Record<string, object> = {
    phone_status: { available: true, accessibility: true, screenRecording: true },
    phone_connect: { token },
    phone_observe: { width: 1, height: 1, base64: png, nodes: [{ role: 'text', text: 'Sample', bounds: { x: 0, y: 0, width: 1, height: 1 }, enabled: true, secure: false }], truncated: false },
  }
  const helper: PhoneNativeHelper = {
    request: async <T extends object>(command: string, params?: Record<string, unknown>) => { calls.push({ command, params }); return (response[command] ?? {}) as T },
    dispose: () => { disposed++ },
  }
  const provider = new IosMirroringProvider({ helper: () => helper, platform: 'darwin' })
  assert.equal((await provider.status()).available, true)
  assert.ok(!calls.some((call) => call.command === 'request_permissions'), 'status never prompts for permissions')
  await provider.requestPermissions({ screenRecording: true })
  assert.deepEqual(calls.find((call) => call.command === 'request_permissions')?.params, { accessibility: false, screenRecording: true })
  await assert.rejects(() => provider.observe(), /Connect iPhone Mirroring/)
  assert.equal((await provider.connect()).transport, 'wifi')
  const observed = await provider.observe()
  assert.equal(observed.screenshot.mimeType, 'image/png')
  assert.equal(observed.nodes[0].text, 'Sample')
  await provider.act({ action: 'tap', x: 0, y: 0 }, observed.dimensions)
  assert.deepEqual(calls.at(-1), { command: 'phone_input', params: { token, width: 1, height: 1, action: 'tap', x: 0, y: 0 } })
  await provider.act({ action: 'swipe', fromX: 0, fromY: 0, toX: 0, toY: 0, durationMs: 300 }, observed.dimensions)
  assert.deepEqual(calls.at(-1)?.params, { token, width: 1, height: 1, action: 'swipe', x: 0, y: 0, toX: 0, toY: 0, durationMs: 300 })
  await provider.act({ action: 'home' }, observed.dimensions)
  await provider.act({ action: 'enter' }, observed.dimensions)
  const before = calls.length
  await assert.rejects(() => provider.act({ action: 'launch', bundleId: 'com.example.sample' }, observed.dimensions), /Home Screen/)
  assert.equal(calls.length, before, 'unsupported actions never reach the native helper')
  response = { ...response, phone_observe: { width: 1, height: 1, base64: png, nodes: [{ role: 'text', text: 'Secret', bounds: { x: 0, y: 0, width: 1, height: 1 }, enabled: true, secure: true }], truncated: false } }
  await assert.rejects(() => provider.observe(), /secure fields/)
  response = { ...response, phone_observe: { width: 9000, height: 1, base64: png, nodes: [] } }
  await assert.rejects(() => provider.observe(), /invalid screen observation/)
  await provider.disconnect()
  assert.ok(calls.some((call) => call.command === 'phone_disconnect' && call.params?.token === token))
  assert.ok(disposed > 0)
  await assert.rejects(() => provider.act({ action: 'home' }, observed.dimensions), /Connect iPhone Mirroring/)
  const other = new IosMirroringProvider({ helper: () => { throw new Error('must not run') }, platform: 'linux' })
  assert.equal((await other.status()).available, false)
  await assert.rejects(() => other.connect(), /supported Macs/)

  let release: (value: object) => void = () => undefined
  let abortDisposals = 0
  const slowHelper: PhoneNativeHelper = {
    request: <T extends object>(command: string) => command === 'phone_connect' ? Promise.resolve({ token } as unknown as T) : new Promise<T>((resolve) => { release = (value) => resolve(value as T) }),
    dispose: () => { abortDisposals++ },
  }
  const stopped = new IosMirroringProvider({ helper: () => slowHelper, platform: 'darwin' })
  await stopped.connect()
  const control = new AbortController()
  const pending = stopped.observe(control.signal)
  control.abort()
  await assert.rejects(() => pending, /stopped/)
  assert.equal(abortDisposals, 1, 'abort terminates the dedicated helper instead of leaving input queued')
  release({ width: 1, height: 1, base64: png, nodes: [], truncated: false })
  await assert.rejects(() => stopped.observe(), /Connect iPhone Mirroring/, 'late native reply cannot restore a revoked connection')
  await stopped.disconnect()
  console.log('iOS scoped Mirroring provider tests passed')
}
main().catch((error) => { console.error(error); process.exitCode = 1 })
