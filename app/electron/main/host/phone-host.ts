import { app } from 'electron'
import { createHmac, randomBytes, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { AndroidPhoneAdapter, type AndroidAction } from '../../agent/phone/android'
import { IosWdaProvider, type IosAction, type IosObservation } from '../../agent/phone/ios-wda'
import { IosMirroringProvider } from '../../agent/phone/ios-mirroring'
import { ComputerHelper } from '../../agent/computer-use/helper'
import { PhoneService, type PhoneBackend, type PhoneRawObservation } from '../../agent/phone/service'
import type { PhoneAction, PhoneDeveloperConnectInput, PhoneDevice, PhonePairInput, PhoneSnapshot } from '../../agent/types'
import { APP_BUNDLE_ID } from './app-identity'

export interface PhoneHostOptions {
  appHome: string
  supported: boolean
  authorizeThread(threadId: string): void
  markChatLocal(threadId: string): Promise<void>
  cancelThread(threadId: string): void
  emit(snapshot: PhoneSnapshot): void
}
function androidAction(action: PhoneAction): AndroidAction {
  switch (action.action) {
    case 'tap': return { kind: 'tap', x: action.x, y: action.y }
    case 'swipe': return { kind: 'swipe', x1: action.x, y1: action.y, x2: action.toX, y2: action.toY, durationMs: action.durationMs }
    case 'type': return { kind: 'type', text: action.text }
    case 'launch': return { kind: 'launch', packageName: action.app }
    default: return { kind: 'key', key: action.action }
  }
}
function iosAction(action: PhoneAction): IosAction {
  switch (action.action) {
    case 'tap': return action
    case 'swipe': return { action: 'swipe', fromX: action.x, fromY: action.y, toX: action.toX, toY: action.toY, durationMs: action.durationMs }
    case 'type': return action
    case 'home': return { action: 'home' }
    case 'launch': return { action: 'launch', bundleId: action.app }
    default: throw new Error('This iPhone connection does not support Back or Enter; use the phone’s visible controls.')
  }
}
function iosRaw(raw: IosObservation): PhoneRawObservation {
  return {
    width: raw.dimensions.width, height: raw.dimensions.height,
    image: `data:image/png;base64,${raw.screenshot.base64}`,
    nodes: raw.nodes.filter((node) => node.enabled).map((node) => ({ text: node.text, description: '', role: node.role, bounds: node.bounds, editable: /TextField|TextView/.test(node.role), secure: node.secure, enabled: node.enabled, focused: (node as { focused?: boolean }).focused === true, interactive: /Button|TextField|TextView/.test(node.role) })),
    app: raw.app?.bundleId ?? raw.app?.name ?? null,
    warnings: raw.truncated ? ['Only the first screen controls are shown; refresh after navigating.'] : [],
  }
}

export function createPhoneHost(options: PhoneHostOptions) {
  const salt = randomBytes(32)
  const serials = new Map<string, string>()
  let adapter = new AndroidPhoneAdapter()
  let helper: ComputerHelper | null = null
  const mirror = new IosMirroringProvider({ helper: () => helper ??= new ComputerHelper({ binaryPath: app.isPackaged ? path.join(process.resourcesPath, 'computer-use', 'tensorfold-computer') : path.join(app.getAppPath(), 'build', 'computer-use', 'tensorfold-computer'), ownBundleId: APP_BUNDLE_ID }) })
  let configured = false
  const configFile = path.join(options.appHome, 'phone', 'config.json')
  const assertSupported = () => { if (!options.supported) throw new Error('Phone control is available only in the local desktop app.') }
  const configure = async () => {
    if (configured) return
    let doc: any
    try {
      const stat = await fs.lstat(configFile)
      if (!stat.isFile() || stat.isSymbolicLink() || stat.size > 16_384) throw new Error('Invalid phone configuration.')
      doc = JSON.parse(await fs.readFile(configFile, 'utf8'))
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw new Error('Phone tool settings are unreadable; preserve them and choose ADB again to repair this configuration.')
    }
    if (doc != null) {
      if (doc.version !== 1 || typeof doc.adbPath !== 'string' || doc.adbPath.length > 4096 || !path.isAbsolute(doc.adbPath) || doc.adbPath.includes('\0')) throw new Error('Saved ADB settings are invalid; choose its executable again.')
      adapter = new AndroidPhoneAdapter({ adbPath: doc.adbPath })
    }
    configured = true
  }
  const iosBackend = (provider: IosWdaProvider | IosMirroringProvider): PhoneBackend => ({
    observe: async (signal) => iosRaw(await provider.observe(signal)),
    act: async (action, dimensions, signal) => { await provider.act(iosAction(action), dimensions, signal) },
    disconnect: async () => { await provider.disconnect() },
  })
  const service = new PhoneService({
    authorizeThread: options.authorizeThread, cancelThread: options.cancelThread, emit: options.emit,
    discover: async (signal) => {
      assertSupported(); await configure()
      const resolved = await adapter.resolve(signal)
      const devices: PhoneDevice[] = []
      serials.clear()
      if (resolved) for (const device of await adapter.devices(signal)) {
        const id = `android:${createHmac('sha256', salt).update(device.serial).digest('hex')}`
        serials.set(id, device.serial)
        devices.push({ id, name: device.name, platform: 'android', transport: device.transport, state: device.state === 'unavailable' ? 'setup-required' : device.state, detail: device.reason ?? null })
      }
      let available = false
      if (process.platform === 'darwin') available = await fs.access('/System/Applications/iPhone Mirroring.app').then(() => true, () => false)
      if (available) devices.push({ id: 'ios-mirroring', name: 'iPhone Mirroring', platform: 'ios', transport: 'mirroring', state: 'ready', detail: 'Open and pair your iPhone in Apple’s iPhone Mirroring app first; TensorFold captures only its phone window.' })
      return { adb: { available: !!resolved, path: resolved?.path ?? null, version: resolved?.version ?? null }, devices, mirroring: { available, detail: available ? 'Apple pairing, Accessibility and Screen Recording are required before phone control.' : 'Apple iPhone Mirroring is not installed on this Mac; a signed developer bridge is another connection option.' } }
    },
    connect: async (id, signal) => {
      assertSupported()
      if (id === 'ios-mirroring') { await mirror.connect(signal); return iosBackend(mirror) }
      const serial = serials.get(id)
      if (!serial || !(await adapter.devices(signal)).some((d) => d.serial === serial && d.state === 'ready')) throw new Error('The selected Android phone is no longer ready; refresh devices and reconnect it.')
      const connectedAdapter = adapter
      return {
        observe: async (inner) => {
          const raw = await connectedAdapter.observe(serial, inner)
          return { width: raw.width, height: raw.height, image: `data:image/png;base64,${raw.png.toString('base64')}`, nodes: raw.nodes.filter((node) => node.enabled).map((node) => ({ text: node.text, description: node.description, role: node.className, bounds: node.bounds, editable: node.editable, secure: node.password, enabled: node.enabled, focused: node.focused, interactive: node.clickable })), app: raw.activePackage ?? null, warnings: raw.warnings }
        },
        act: async (action, dimensions, inner) => { await connectedAdapter.act(serial, androidAction(action), dimensions, inner) },
        // Closing TensorFold's view does not revoke an ADB pairing used by other apps.
        disconnect: async () => {},
      }
    },
    connectDeveloper: async (input, signal) => {
      assertSupported()
      const provider = new IosWdaProvider(input)
      try {
        const connection = await provider.connect(signal)
        return { device: { id: `ios-developer:${randomUUID()}`, name: connection.label || 'iPhone developer bridge', platform: 'ios', transport: input.transport, state: 'ready', detail: 'Your signed WebDriverAgent runner' }, backend: iosBackend(provider) }
      } catch (error) { await provider.disconnect().catch(() => {}); throw error }
    },
  })
  return {
    service,
    async grantChat(id: string): Promise<PhoneSnapshot> {
      const sessionId = service.snapshot().session?.id
      if (!sessionId) throw new Error('Connect a phone before granting this chat access.')
      options.authorizeThread(id)
      await options.markChatLocal(id)
      options.authorizeThread(id)
      if (service.snapshot().session?.id !== sessionId) throw new Error('The connected phone changed while saving; review it and allow this chat again.')
      return service.grant(id)
    },
    async setAdbPath(selected: string): Promise<PhoneSnapshot> {
      assertSupported()
      if (service.snapshot().session || service.snapshot().busy) throw new Error('Disconnect the phone before changing Android tools.')
      if (!['adb', 'adb.exe'].includes(path.basename(selected)) || !path.isAbsolute(selected)) throw new Error('Choose the adb executable inside Android Platform Tools.')
      await service.manage(async (signal) => {
      if (service.snapshot().session) throw new Error('Disconnect the phone before changing Android tools.')
      const candidate = new AndroidPhoneAdapter({ adbPath: selected })
      if (!await candidate.resolve(signal)) throw new Error('That executable did not identify itself as Android Debug Bridge.')
      const dir = path.dirname(configFile)
      await fs.mkdir(dir, { recursive: true, mode: 0o700 })
      const stat = await fs.lstat(dir)
      if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('The phone settings directory is invalid.')
      const temporary = path.join(dir, `.config-${randomUUID()}.tmp`)
      try { await fs.writeFile(temporary, JSON.stringify({ version: 1, adbPath: selected }), { mode: 0o600, flag: 'wx' }); await fs.rename(temporary, configFile) } finally { await fs.rm(temporary, { force: true }) }
      configured = true; adapter = candidate
      })
      return service.refresh()
    },
    async pairWifi(input: PhonePairInput): Promise<PhoneSnapshot> {
      assertSupported(); await configure()
      if (service.snapshot().busy) throw new Error('Wait for the current phone operation before pairing.')
      await service.manage((signal) => adapter.pair(input, signal))
      return service.refresh()
    },
    async connectWifi(endpoint: string): Promise<PhoneSnapshot> {
      assertSupported(); await configure()
      if (service.snapshot().busy) throw new Error('Wait for the current phone operation before connecting.')
      await service.manage((signal) => adapter.connect(endpoint, signal))
      return service.refresh()
    },
    async permissions(): Promise<PhoneSnapshot> {
      assertSupported()
      if (service.snapshot().busy) throw new Error('Wait for the current phone operation before checking permissions.')
      await service.manage(async () => { await mirror.requestPermissions({ accessibility: true, screenRecording: true }) })
      return service.refresh()
    },
    async shutdown(): Promise<void> { await service.shutdown(); helper?.dispose(); helper = null },
  }
}

export type PhoneHost = ReturnType<typeof createPhoneHost>
let host: PhoneHost | null = null
export function initializePhoneHost(options: PhoneHostOptions): PhoneHost { return host ??= createPhoneHost(options) }
export function phoneHost(): PhoneHost | null { return host }
