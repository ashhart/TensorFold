import { request as httpRequest } from 'node:http'
import { request as httpsRequest } from 'node:https'
import { isIP } from 'node:net'

// Original client for the documented WebDriverAgent protocol. The user owns and
// starts the signed runner; TensorFold never installs it, unlocks a phone, or
// takes over another client's session. All content stays in memory here.
export interface IosDimensions { width: number; height: number }
export interface IosNode {
  role: string
  text: string
  bounds: { x: number; y: number; width: number; height: number }
  enabled: boolean
  secure: boolean
  focused?: boolean
}
export interface IosObservation {
  dimensions: IosDimensions
  screenshot: { mimeType: 'image/png'; base64: string; width: number; height: number }
  nodes: IosNode[]
  app?: { name: string; bundleId: string }
  truncated: boolean
}
export type IosAction =
  | { action: 'tap'; x: number; y: number }
  | { action: 'swipe'; fromX: number; fromY: number; toX: number; toY: number; durationMs?: number }
  | { action: 'type'; text: string }
  | { action: 'home' }
  | { action: 'launch'; bundleId: string }
export interface IosConnection { platform: 'ios'; label: string; transport: 'usb' | 'wifi'; osVersion?: string }
export class IosPhoneError extends Error {
  constructor(readonly code: 'endpoint' | 'connection' | 'protocol' | 'busy' | 'locked' | 'cancelled' | 'timeout' | 'stale' | 'input' | 'disconnected' | 'secure', message: string) {
    super(message)
    this.name = 'IosPhoneError'
  }
}

const RESPONSE_LIMIT = 768 * 1024
const IMAGE_RESPONSE_LIMIT = 12 * 1024 * 1024
const SOURCE_LIMIT = 512 * 1024
const IMAGE_LIMIT = 8 * 1024 * 1024
const NODE_LIMIT = 240
const PNG_SIGNATURE = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10])
const SESSION_ID = /^[A-Za-z0-9_-]{1,128}$/
const BUNDLE_ID = /^[A-Za-z0-9][A-Za-z0-9-]*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)+$/

function object(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
}
function text(value: unknown, limit = 240): string {
  return typeof value === 'string' ? value.replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/g, '').slice(0, limit) : ''
}
function cancelled(): IosPhoneError { return new IosPhoneError('cancelled', 'Phone control was stopped.') }
function assertSignal(signal?: AbortSignal): void { if (signal?.aborted) throw cancelled() }

export function normalizeIosEndpoint(input: string, transport: 'usb' | 'wifi' = 'usb'): string {
  if (typeof input !== 'string' || input.length > 256) throw new IosPhoneError('endpoint', 'Enter a loopback or private numeric IP WebDriverAgent endpoint with its port.')
  // Validate the authority before WHATWG URL normalization to reject decimal,
  // octal and hexadecimal host aliases, credentials, paths, and DNS rebinding.
  const match = /^(https?):\/\/(\[[a-fA-F0-9:]+\]|localhost|[0-9.]+):(\d{1,5})\/?$/.exec(input.trim())
  if (!match) throw new IosPhoneError('endpoint', 'Use http://127.0.0.1:8100 or an explicit private numeric IP and port; paths, credentials and DNS names are not supported.')
  let host = match[2].replace(/^\[|\]$/g, '')
  if (host === 'localhost') host = '127.0.0.1'
  const port = Number(match[3])
  if (port < 1024 || port > 65535) throw new IosPhoneError('endpoint', 'The phone bridge port must be between 1024 and 65535.')
  const family = isIP(host)
  const octets = family === 4 ? host.split('.').map(Number) : []
  const loopback = host === '::1' || family === 4 && octets[0] === 127
  const privateAddress = family === 4 && (octets[0] === 10 || octets[0] === 192 && octets[1] === 168 || octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31)
    || family === 6 && /^f[cd]/i.test(host) && !host.includes('.')
  if (!family || !loopback && !privateAddress || transport === 'usb' && !loopback) throw new IosPhoneError('endpoint', transport === 'usb' ? 'USB requires an existing loopback forward to your signed WebDriverAgent runner.' : 'Wi-Fi requires your paired phone on a private numeric IP; public and link-local endpoints are blocked.')
  return `${match[1]}://${family === 6 ? `[${host}]` : host}:${port}`
}

export function iosPng(base64: unknown): IosObservation['screenshot'] {
  if (typeof base64 !== 'string' || base64.length > Math.ceil(IMAGE_LIMIT / 3) * 4 || base64.length % 4 !== 0 || !/^[A-Za-z0-9+/]*={0,2}$/.test(base64)) throw new IosPhoneError('protocol', 'The phone bridge returned an invalid or oversized screenshot.')
  const bytes = Buffer.from(base64, 'base64')
  if (bytes.length < 45 || bytes.length > IMAGE_LIMIT || !bytes.subarray(0, 8).equals(PNG_SIGNATURE) || bytes.readUInt32BE(8) !== 13 || bytes.toString('ascii', 12, 16) !== 'IHDR') throw new IosPhoneError('protocol', 'The phone bridge did not return a PNG screenshot.')
  const width = bytes.readUInt32BE(16)
  const height = bytes.readUInt32BE(20)
  if (width < 1 || height < 1 || width > 8192 || height > 8192 || width * height > 24_000_000) throw new IosPhoneError('protocol', 'The phone screenshot dimensions exceed the supported limit.')
  return { mimeType: 'image/png', base64, width, height }
}

function xmlText(input: string): string {
  return input.replace(/&(?:amp|lt|gt|quot|apos|#\d{1,7}|#x[\da-fA-F]{1,6});/g, (entity) => {
    const fixed: Record<string, string> = { '&amp;': '&', '&lt;': '<', '&gt;': '>', '&quot;': '"', '&apos;': "'" }
    if (fixed[entity]) return fixed[entity]
    const code = entity.startsWith('&#x') ? Number.parseInt(entity.slice(3, -1), 16) : Number(entity.slice(2, -1))
    return Number.isInteger(code) && code > 0 && code <= 0x10ffff && !(code >= 0xd800 && code <= 0xdfff) ? String.fromCodePoint(code) : ''
  })
}

// Read only bounded attributes from XCUIElement tags; no XML engine, DTD,
// external entities, file accesses, recursive expansion or live selectors.
export function parseIosSource(source: unknown, dimensions: IosDimensions): { nodes: IosNode[]; truncated: boolean } {
  if (typeof source !== 'string' || Buffer.byteLength(source) > SOURCE_LIMIT || /<!DOCTYPE|<!ENTITY/i.test(source)) throw new IosPhoneError('protocol', 'The phone bridge returned an unsupported or oversized accessibility tree.')
  const nodes: IosNode[] = []
  let truncated = false
  for (const match of source.matchAll(/<XCUIElementType([A-Za-z]{1,50})\b([^<>]{0,16384})\/?>/g)) {
    const attrs: Record<string, string> = Object.create(null)
    for (const attr of match[2].matchAll(/([A-Za-z][A-Za-z0-9_-]{0,40})\s*=\s*(?:"([^"<>]*)"|'([^'<>]*)')/g)) attrs[attr[1]] = xmlText(attr[2] ?? attr[3] ?? '')
    if (attrs.visible === 'false' || attrs.visible === '0') continue
    const x = Number(attrs.x), y = Number(attrs.y), width = Number(attrs.width), height = Number(attrs.height)
    if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0 || x < 0 || y < 0 || x + width > dimensions.width + 2 || y + height > dimensions.height + 2) continue
    const secure = /SecureTextField/i.test(match[1])
    const label = secure ? '[secure field]' : text(attrs.label || attrs.name || attrs.value)
    if (!label && !/Button|TextField|TextView|Switch|Cell|Link/.test(match[1])) continue
    if (nodes.length >= NODE_LIMIT) { truncated = true; break }
    nodes.push({ role: match[1], text: label, bounds: { x, y, width, height }, enabled: attrs.enabled !== 'false' && attrs.enabled !== '0', secure, ...(['true', 'false', '1', '0'].includes(attrs.focused) ? { focused: attrs.focused === 'true' || attrs.focused === '1' } : {}) })
  }
  return { nodes, truncated }
}

function viewport(value: unknown): IosDimensions {
  const result = object(object(value)?.screenSize)
  const width = result?.width, height = result?.height
  if (typeof width !== 'number' || typeof height !== 'number' || !Number.isFinite(width) || !Number.isFinite(height) || width < 1 || height < 1 || width > 4096 || height > 4096) throw new IosPhoneError('protocol', 'The phone bridge returned invalid screen dimensions.')
  return { width, height }
}
function point(x: number, y: number, dimensions: IosDimensions): void {
  if (!Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x >= dimensions.width || y >= dimensions.height) throw new IosPhoneError('input', 'Use coordinates inside the latest phone screenshot.')
}

interface WdaReply { value: unknown; sessionId?: unknown }
interface WdaOptions { endpoint: string; transport?: 'usb' | 'wifi'; timeoutMs?: number }

export class IosWdaProvider {
  readonly endpoint: string
  readonly transport: 'usb' | 'wifi'
  private sessionId: string | null = null
  private lifetime = new AbortController()
  private inFlight = false
  private creating: Promise<WdaReply> | null = null
  private cleaning: Promise<void> | null = null
  private readonly timeoutMs: number

  constructor(options: WdaOptions) {
    this.transport = options.transport ?? 'usb'
    this.endpoint = normalizeIosEndpoint(options.endpoint, this.transport)
    this.timeoutMs = Math.max(100, Math.min(options.timeoutMs ?? 15_000, 30_000))
  }

  async connect(signal?: AbortSignal): Promise<IosConnection> {
    if (this.sessionId || this.inFlight) throw new IosPhoneError('busy', 'This phone bridge already has a TensorFold connection.')
    this.lifetime = new AbortController()
    signal = AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])])
    return this.exclusive(async () => {
      const reply = await this.request('GET', '/status', undefined, signal)
      const status = object(reply.value)
      const os = object(status?.os)
      if (status?.ready !== true || os?.name !== 'iOS') throw new IosPhoneError('protocol', 'This endpoint is not a ready iOS WebDriverAgent runner.')
      if (typeof reply.sessionId === 'string' && reply.sessionId) throw new IosPhoneError('busy', 'WebDriverAgent has another active session; finish it before connecting TensorFold.')
      await this.requireUnlocked(signal)
      assertSignal(signal)
      // Creation is the only operation whose reply must be read after Stop:
      // the server assigns the ID, and we need that ID to delete our session.
      // No application is launched, terminated, unlocked or read by this call.
      this.creating = this.request('POST', '/session', { capabilities: { alwaysMatch: { shouldTerminateApp: false, forceAppLaunch: false, shouldWaitForQuiescence: false }, firstMatch: [{}] } }, undefined, undefined, undefined, true)
      let created: WdaReply
      try { created = await this.creating } finally { this.creating = null }
      const nested = object(created.value)
      const id = typeof nested?.sessionId === 'string' ? nested.sessionId : created.sessionId
      if (typeof id !== 'string' || !SESSION_ID.test(id)) throw new IosPhoneError('protocol', 'WebDriverAgent did not create a valid TensorFold session.')
      if (signal?.aborted) { await this.releaseSession(id); throw cancelled() }
      this.sessionId = id
      try {
        assertSignal(signal)
        viewport((await this.request('GET', this.path('/wda/screen'), undefined, signal)).value)
      } catch (error) {
        this.sessionId = null
        await this.releaseSession(id)
        throw error
      }
      return { platform: 'ios', label: 'iPhone', transport: this.transport, osVersion: text(os.version, 32) || undefined }
    })
  }

  async observe(signal?: AbortSignal): Promise<IosObservation> {
    signal = AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])])
    return this.exclusive(async () => {
      this.path('')
      await this.requireUnlocked(signal)
      const dimensions = viewport((await this.request('GET', this.path('/wda/screen'), undefined, signal)).value)
      const tree = parseIosSource((await this.request('GET', this.path('/source'), undefined, signal)).value, dimensions)
      if (tree.nodes.some((node) => node.secure)) throw new IosPhoneError('secure', 'Complete secure fields directly on your phone, then refresh its screen.')
      await this.verifyKeyboardFocus(tree.nodes, signal)
      const screenshot = iosPng((await this.request('GET', this.path('/screenshot'), undefined, signal, undefined, IMAGE_RESPONSE_LIMIT)).value)
      const current = object((await this.request('GET', this.path('/wda/activeAppInfo'), undefined, signal)).value)
      const after = viewport((await this.request('GET', this.path('/wda/screen'), undefined, signal)).value)
      if (dimensions.width !== after.width || dimensions.height !== after.height) throw new IosPhoneError('stale', 'The phone rotated during capture; refresh its screen before acting.')
      const bundleId = text(current?.bundleId, 255)
      return { dimensions, screenshot, ...tree, app: BUNDLE_ID.test(bundleId) ? { name: text(current?.name, 120), bundleId } : undefined }
    })
  }

  async act(action: IosAction, dimensions: IosDimensions, signal?: AbortSignal): Promise<void> {
    signal = AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])])
    return this.exclusive(async () => {
      const base = this.path('')
      await this.requireUnlocked(signal)
      const fresh = viewport((await this.request('GET', `${base}/wda/screen`, undefined, signal)).value)
      if (fresh.width !== dimensions.width || fresh.height !== dimensions.height) throw new IosPhoneError('stale', 'The phone screen changed size; refresh it before acting.')
      if (action.action === 'tap') {
        point(action.x, action.y, fresh)
        await this.request('POST', `${base}/wda/tap`, { x: action.x, y: action.y }, signal)
      } else if (action.action === 'swipe') {
        point(action.fromX, action.fromY, fresh)
        point(action.toX, action.toY, fresh)
        const durationMs = action.durationMs ?? 400
        if (!Number.isFinite(durationMs) || durationMs < 100 || durationMs > 2000) throw new IosPhoneError('input', 'Swipe duration must be between 100 and 2000 milliseconds.')
        await this.request('POST', `${base}/wda/dragfromtoforduration`, { fromX: action.fromX, fromY: action.fromY, toX: action.toX, toY: action.toY, duration: durationMs / 1000 }, signal)
      } else if (action.action === 'type') {
        if (typeof action.text !== 'string' || !action.text || action.text.length > 4000 || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\ue000-\uf8ff]/.test(action.text)) throw new IosPhoneError('input', 'Type between 1 and 4000 characters without control or WebDriver key codes.')
        // A fresh tree is the only evidence available without broad queries.
        // Refuse when it exposes a secure field; passwords stay with the owner.
        const source = (await this.request('GET', `${base}/source`, undefined, signal)).value
        if (typeof source !== 'string' || /XCUIElementTypeSecureTextField\b/.test(source)) throw new IosPhoneError('secure', 'Enter passwords and other secure fields directly on your phone.')
        const tree = parseIosSource(source, fresh)
        await this.verifyKeyboardFocus(tree.nodes, signal)
        if (!tree.nodes.some((node) => node.focused === true && node.enabled && !node.secure && /^(TextField|TextView)$/.test(node.role))) throw new IosPhoneError('input', 'The phone has no verified focused text field; select a field and refresh, or type directly on your phone.')
        await this.request('POST', `${base}/wda/keys`, { value: [...action.text] }, signal)
      } else if (action.action === 'home') {
        // Homescreen is a documented sessionless route, but is only reachable
        // after this provider has an owned, unlocked session.
        await this.request('POST', '/wda/homescreen', {}, signal)
      } else if (action.action === 'launch') {
        if (typeof action.bundleId !== 'string' || action.bundleId.length > 255 || !BUNDLE_ID.test(action.bundleId)) throw new IosPhoneError('input', 'Launch requires a valid installed iPhone app bundle identifier.')
        await this.request('POST', `${base}/wda/apps/activate`, { bundleId: action.bundleId }, signal)
      } else throw new IosPhoneError('input', 'This iPhone action is not supported.')
    })
  }

  async disconnect(): Promise<void> {
    const id = this.sessionId
    const creating = this.creating
    this.sessionId = null
    this.lifetime.abort()
    if (id) await this.releaseSession(id)
    if (creating) await creating.catch(() => undefined)
    if (this.cleaning) await this.cleaning
  }

  private path(suffix: string): string {
    if (!this.sessionId) throw new IosPhoneError('disconnected', 'Connect your iPhone before controlling it.')
    return `/session/${this.sessionId}${suffix}`
  }
  private async verifyKeyboardFocus(nodes: IosNode[], signal?: AbortSignal): Promise<void> {
    // iOS XML `focused` can describe UI focus and is often absent. WDA's
    // dedicated active-element route specifically reports hasKeyboardFocus.
    // Verify its type and live rectangle against one visible tree node;
    // never infer focus from a selected field, label, keyboard or tap history.
    for (const node of nodes) node.focused = false
    const timeout = Math.min(this.timeoutMs, 2_000)
    try {
      const activeRoute = this.path('/element/active')
      const active = object((await this.request('GET', activeRoute, undefined, signal, timeout)).value)
      const id = active?.['element-6066-11e4-a52e-4f735466cecf'] ?? active?.ELEMENT
      if (typeof id !== 'string' || !SESSION_ID.test(id) || active?.ELEMENT !== undefined && active.ELEMENT !== id) return
      const element = this.path(`/element/${id}`)
      const name = (await this.request('GET', `${element}/name`, undefined, signal, timeout)).value
      if (typeof name !== 'string') return
      if (/SecureTextField/.test(name)) throw new IosPhoneError('secure', 'Complete secure fields directly on your phone, then refresh its screen.')
      const role = name.replace(/^XCUIElementType/, '')
      if (!/^(TextField|TextView)$/.test(role)) return
      const rect = object((await this.request('GET', `${element}/rect`, undefined, signal, timeout)).value)
      if (!rect || !['x', 'y', 'width', 'height'].every((key) => typeof rect[key] === 'number' && Number.isFinite(rect[key]))) return
      const enabled = (await this.request('GET', `${element}/enabled`, undefined, signal, timeout)).value
      const visible = (await this.request('GET', `${element}/displayed`, undefined, signal, timeout)).value
      if (enabled !== true || visible !== true) return
      const matching = nodes.filter((node) => node.role === role && node.enabled && !node.secure && (['x', 'y', 'width', 'height'] as const).every((key) => Math.abs(node.bounds[key] - (rect[key] as number)) <= 1))
      if (matching.length !== 1) return
      // A focus change during the fixed read sequence must not grant typing.
      const after = object((await this.request('GET', activeRoute, undefined, signal, timeout)).value)
      if ((after?.['element-6066-11e4-a52e-4f735466cecf'] ?? after?.ELEMENT) !== id || after?.ELEMENT !== undefined && after.ELEMENT !== id) return
      matching[0].focused = true
    } catch (error) {
      if (signal?.aborted) throw cancelled()
      if (error instanceof IosPhoneError && (error.code === 'secure' || error.code === 'cancelled')) throw error
      // Older runners and screens without a keyboard-focused element may
      // return no-such-element/unsupported. Unknown focus keeps typing closed.
    }
  }
  private async requireUnlocked(signal?: AbortSignal): Promise<void> {
    const state = (await this.request('GET', '/wda/locked', undefined, signal)).value
    if (state !== false) throw new IosPhoneError(state === true ? 'locked' : 'protocol', state === true ? 'Unlock your iPhone yourself before using the developer bridge.' : 'WebDriverAgent did not report a valid phone lock state.')
  }
  private async exclusive<T>(run: () => Promise<T>): Promise<T> {
    if (this.inFlight) throw new IosPhoneError('busy', 'Wait for the current phone operation to finish.')
    this.inFlight = true
    try { return await run() } finally { this.inFlight = false }
  }
  private releaseSession(id: string): Promise<void> {
    const pending = this.request('DELETE', `/session/${id}`, undefined, undefined, 1_500, undefined, true).then(() => undefined, () => undefined)
    this.cleaning = pending
    void pending.finally(() => { if (this.cleaning === pending) this.cleaning = null })
    return pending
  }
  private request(method: 'GET' | 'POST' | 'DELETE', route: string, body?: unknown, signal?: AbortSignal, timeoutMs = this.timeoutMs, limit = RESPONSE_LIMIT, cleanupOnly = false): Promise<WdaReply> {
    assertSignal(signal)
    const combined = cleanupOnly ? new AbortController().signal : AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])])
    assertSignal(combined)
    const target = new URL(route, this.endpoint)
    const data = body === undefined ? undefined : Buffer.from(JSON.stringify(body))
    return new Promise((resolve, reject) => {
      let settled = false
      const finish = (error: Error | null, reply?: WdaReply) => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        combined.removeEventListener('abort', abort)
        if (error) reject(error); else resolve(reply!)
      }
      const req = (target.protocol === 'https:' ? httpsRequest : httpRequest)(target, {
        method, agent: false,
        headers: { Accept: 'application/json', 'Accept-Encoding': 'identity', ...(data ? { 'Content-Type': 'application/json', 'Content-Length': String(data.length) } : {}) },
      }, (res) => {
        const status = res.statusCode ?? 0
        if (status < 200 || status >= 300) {
          res.destroy()
          finish(new IosPhoneError('protocol', status >= 300 && status < 400 ? 'The phone bridge attempted a redirect, which is blocked.' : `WebDriverAgent refused the operation with HTTP ${status}.`))
          return
        }
        if (res.headers['content-encoding'] && res.headers['content-encoding'] !== 'identity' || !String(res.headers['content-type'] ?? '').toLowerCase().includes('application/json')) {
          res.destroy(); finish(new IosPhoneError('protocol', 'The phone bridge returned an unsupported response format.')); return
        }
        const advertised = Number(res.headers['content-length'])
        if (Number.isFinite(advertised) && advertised > limit) { res.destroy(); finish(new IosPhoneError('protocol', 'The phone bridge response exceeds the size limit.')); return }
        const chunks: Buffer[] = []
        let size = 0
        res.on('data', (chunk: Buffer) => {
          size += chunk.length
          if (size > limit) { res.destroy(); finish(new IosPhoneError('protocol', 'The phone bridge response exceeds the size limit.')); return }
          chunks.push(chunk)
        })
        res.on('error', () => finish(new IosPhoneError('connection', 'The phone bridge connection closed before the response completed.')))
        res.on('end', () => {
          try {
            const reply = object(JSON.parse(Buffer.concat(chunks).toString('utf8')))
            if (!reply || !Object.hasOwn(reply, 'value') || object(reply.value)?.error) throw new Error('protocol')
            if (method !== 'DELETE' && route.startsWith('/session/') && reply.sessionId !== this.sessionId) throw new Error('session changed')
            finish(null, reply as unknown as WdaReply)
          } catch { finish(new IosPhoneError('protocol', 'WebDriverAgent returned an invalid response or could not perform the operation.')) }
        })
      })
      const abort = () => { req.destroy(); finish(cancelled()) }
      const timer = setTimeout(() => { req.destroy(); finish(new IosPhoneError('timeout', 'The phone bridge did not respond in time; check the runner and connection.')) }, timeoutMs)
      combined.addEventListener('abort', abort, { once: true })
      req.on('error', () => finish(new IosPhoneError('connection', 'Cannot reach your signed WebDriverAgent runner; check the USB forward or paired Wi-Fi endpoint.')))
      req.end(data)
      if (combined.aborted) abort()
    })
  }
}
