import { IosPhoneError, iosPng, type IosAction, type IosConnection, type IosDimensions, type IosNode, type IosObservation } from './ios-wda'

// A dedicated native helper is required. Stopping a phone request terminates
// its helper so an interrupted input can never run later from a queued line.
export interface PhoneNativeHelper {
  request<T extends object>(cmd: string, params?: Record<string, unknown>, timeoutMs?: number): Promise<T>
  dispose(): void
}
export interface IosMirroringStatus { available: boolean; accessibility: boolean; screenRecording: boolean }
interface NativeObservation {
  width: number
  height: number
  base64: string
  nodes: IosNode[]
  truncated: boolean
  warnings?: string[]
}
export interface IosMirroringOptions { helper: () => PhoneNativeHelper; platform?: string }

export class IosMirroringProvider {
  private token: string | null = null
  private lifetime = new AbortController()
  private inFlight = false
  private readonly platform: string
  constructor(private readonly options: IosMirroringOptions) { this.platform = options.platform ?? process.platform }

  async status(): Promise<IosMirroringStatus> {
    if (this.platform !== 'darwin') return { available: false, accessibility: false, screenRecording: false }
    const status = await this.options.helper().request<IosMirroringStatus>('phone_status')
    if (typeof status.available !== 'boolean' || typeof status.accessibility !== 'boolean' || typeof status.screenRecording !== 'boolean') throw new IosPhoneError('protocol', 'The native phone helper returned invalid permission status.')
    return status
  }

  // The caller exposes this only behind a user-clicked permissions button.
  async requestPermissions(which: { accessibility?: boolean; screenRecording?: boolean }): Promise<IosMirroringStatus> {
    if (this.platform !== 'darwin') return this.status()
    if (which.accessibility !== true && which.screenRecording !== true) throw new IosPhoneError('input', 'Choose the macOS permission to request.')
    await this.options.helper().request('request_permissions', { accessibility: which.accessibility === true, screenRecording: which.screenRecording === true })
    return this.status()
  }

  async connect(signal?: AbortSignal): Promise<IosConnection> {
    if (this.token || this.inFlight) throw new IosPhoneError('busy', 'Disconnect the current iPhone Mirroring session before reconnecting.')
    if (this.platform !== 'darwin') throw new IosPhoneError('connection', 'Apple iPhone Mirroring is available on supported Macs only.')
    this.lifetime = new AbortController()
    const operation = this.signal(signal)
    return this.exclusive(async () => {
      const result = await this.request<{ token: string }>('phone_connect', {}, operation)
      if (typeof result.token !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(result.token)) throw new IosPhoneError('protocol', 'The native phone helper did not return a valid connection token.')
      if (operation.aborted) throw new IosPhoneError('cancelled', 'Phone control was stopped.')
      this.token = result.token
      return { platform: 'ios', label: 'iPhone', transport: 'wifi' }
    })
  }

  async observe(signal?: AbortSignal): Promise<IosObservation> {
    const token = this.currentToken()
    const operation = this.signal(signal)
    return this.exclusive(async () => {
      const result = await this.request<NativeObservation>('phone_observe', { token }, operation)
      const screenshot = iosPng(result.base64)
      if (!Number.isInteger(result.width) || !Number.isInteger(result.height) || result.width !== screenshot.width || result.height !== screenshot.height || !Array.isArray(result.nodes) || result.nodes.length > 240) throw new IosPhoneError('protocol', 'The native phone helper returned an invalid screen observation.')
      const dimensions = { width: result.width, height: result.height }
      const nodes = result.nodes.map((node) => {
        if (!node || typeof node.text !== 'string' || node.text.length > 240 || node.role !== 'text' || typeof node.secure !== 'boolean' || !node.bounds || !Object.values(node.bounds).every((value) => typeof value === 'number' && Number.isFinite(value)) || node.bounds.x < 0 || node.bounds.y < 0 || node.bounds.width <= 0 || node.bounds.height <= 0 || node.bounds.x + node.bounds.width > dimensions.width + 2 || node.bounds.y + node.bounds.height > dimensions.height + 2) throw new IosPhoneError('protocol', 'The native phone helper returned invalid screen text bounds.')
        return node
      })
      if (nodes.some((node) => node.secure)) throw new IosPhoneError('secure', 'Complete secure fields directly on your phone, then refresh its screen.')
      return { dimensions, screenshot, nodes, truncated: result.truncated === true }
    })
  }

  async act(action: IosAction | { action: 'enter' }, dimensions: IosDimensions, signal?: AbortSignal): Promise<void> {
    const token = this.currentToken()
    const operation = this.signal(signal)
    const input: Record<string, unknown> = { token, width: dimensions.width, height: dimensions.height, action: action.action }
    if (action.action === 'tap') Object.assign(input, { x: action.x, y: action.y })
    else if (action.action === 'swipe') Object.assign(input, { x: action.fromX, y: action.fromY, toX: action.toX, toY: action.toY, durationMs: action.durationMs ?? 400 })
    else if (action.action === 'type') input.text = action.text
    else if (action.action !== 'home' && action.action !== 'enter') throw new IosPhoneError('input', 'iPhone Mirroring supports tap, swipe, typing, Home and Enter; open other iPhone apps from its Home Screen.')
    return this.exclusive(async () => { await this.request('phone_input', input, operation) })
  }

  async disconnect(): Promise<void> {
    const token = this.token
    this.token = null
    const wasBusy = this.inFlight
    this.lifetime.abort()
    if (wasBusy) { this.options.helper().dispose(); return }
    if (token) await this.options.helper().request('phone_disconnect', { token }, 1500).catch(() => undefined)
    // The helper contains the native pin, observation and token, so teardown
    // also clears those when a disconnect reply was lost or malformed.
    this.options.helper().dispose()
  }

  private currentToken(): string {
    if (!this.token) throw new IosPhoneError('disconnected', 'Connect iPhone Mirroring before reading or controlling it.')
    return this.token
  }
  private signal(signal?: AbortSignal): AbortSignal { return AbortSignal.any([this.lifetime.signal, ...(signal ? [signal] : [])]) }
  private async exclusive<T>(run: () => Promise<T>): Promise<T> {
    if (this.inFlight) throw new IosPhoneError('busy', 'Wait for the current phone operation to finish.')
    this.inFlight = true
    try { return await run() } finally { this.inFlight = false }
  }
  private async request<T extends object>(command: string, params: Record<string, unknown>, signal: AbortSignal): Promise<T> {
    if (signal.aborted) throw new IosPhoneError('cancelled', 'Phone control was stopped.')
    const helper = this.options.helper()
    let rejectStopped: (error: Error) => void = () => undefined
    const stopped = new Promise<never>((_resolve, reject) => { rejectStopped = reject })
    const abort = () => {
      this.token = null
      helper.dispose()
      rejectStopped(new IosPhoneError('cancelled', 'Phone control was stopped.'))
    }
    signal.addEventListener('abort', abort, { once: true })
    try {
      const result = await Promise.race([helper.request<T>(command, params, 12_000), stopped])
      if (signal.aborted) throw new IosPhoneError('cancelled', 'Phone control was stopped.')
      return result
    } finally { signal.removeEventListener('abort', abort) }
  }
}
