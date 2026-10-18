import { lstat, readFile, rename, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { randomBytes } from 'node:crypto'
import { createMobileBridge, validateMobileBridgeConfig } from '../../agent/mobile-bridge'
import { ensureMobilePrivateDirectory, loadMobileBridgeCertificate } from '../../agent/mobile-bridge-certificate'
import type { WatchSpeechBroker } from '../../agent/watch-speech'
import type { WatchSnapshot } from '../../../shared/watch-relay'
import type { MobileBridgeConfig } from '../../agent/types/mobile-bridge'
import { requireSecurityRole } from './audit'
import { handle } from './ipc'

export interface MobileBridgeRuntimeDeps {
  profileRoot: () => string
  watchSpeech?: WatchSpeechBroker
  projectWatchSnapshot?: (snapshot: WatchSnapshot, sessionId: string) => WatchSnapshot
  setWatchSpeech?: (sessionId: string, enabled: boolean) => Promise<boolean>
  remoteServer: {
    status(): { url: string | null }
    syncFromSettings(): Promise<void>
    issuePairingLink(label: string): Promise<{ credential: string; expiresAt: number }>
  }
  ownerToken: () => string
  authorizeSessionToken: (token: string) => { ok: boolean; sessionId?: string }
  /** Called only by an explicit owner enable action, never by background restoration. */
  enableLoopback: () => Promise<void>
  audit?: (kind: string, data: Record<string, unknown>) => void
  reserveOperation?: () => { release(): void }
}

export function createMobileBridgeRuntime(deps: MobileBridgeRuntimeDeps) {
  const bridge = createMobileBridge({
    requireOwner: () => requireSecurityRole('owner', 'manage the private mobile connection'),
    loopbackUrl: () => deps.remoteServer.status().url,
    ownerToken: deps.ownerToken,
    authorizeSessionToken: deps.authorizeSessionToken,
    issuePairingLink: label => deps.remoteServer.issuePairingLink(label),
    loadCertificate: host => loadMobileBridgeCertificate(deps.profileRoot(), host),
    audit: deps.audit,
    reserveRequestOperation: deps.reserveOperation,
    watchSpeech: deps.watchSpeech,
    projectWatchSnapshot: deps.projectWatchSnapshot,
  })
  const configPath = () => path.join(deps.profileRoot(), 'mobile-bridge', 'config.json')
  let operations: Promise<unknown> = Promise.resolve(), activeOperations = 0
  function serial<T>(action: () => Promise<T>): Promise<T> {
    const guarded = async () => {
      const reservation = deps.reserveOperation?.()
      activeOperations++
      try { return await action() }
      finally { activeOperations--; reservation?.release() }
    }
    const result = operations.then(guarded, guarded)
    operations = result.catch(() => {})
    return result
  }

  async function save(config: MobileBridgeConfig): Promise<void> {
    const directory = path.dirname(configPath())
    await ensureMobilePrivateDirectory(directory)
    try {
      const info = await lstat(configPath())
      if (!info.isFile() || info.isSymbolicLink() || process.getuid && info.uid !== process.getuid()) throw new Error('Mobile connection configuration is not privately owned.')
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
    const temporary = path.join(directory, `.config-${randomBytes(8).toString('hex')}.json`)
    await writeFile(temporary, `${JSON.stringify(config)}\n`, { mode: 0o600, flag: 'wx' })
    await rename(temporary, configPath())
  }

  async function initialize(): Promise<void> {
    try {
      const info = await lstat(configPath())
      if (!info.isFile() || info.isSymbolicLink() || info.size > 1024 || process.getuid && info.uid !== process.getuid() || info.mode & 0o077) return
      const config = JSON.parse(await readFile(configPath(), 'utf8')) as MobileBridgeConfig
      if (config.enabled !== true) return
      await serial(() => bridge.configure(config, true))
    } catch { /* A disabled/invalid profile cannot expose a listener during boot. Status carries start errors. */ }
  }

  function registerIpc(): void {
    handle('mobile:setDeviceWatchSpeech', async (_event, input: unknown) => {
      requireSecurityRole('owner', 'allow Kokoro audio for a paired phone and watch')
      if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Choose a paired device and whether to enable watch speech.')
      const value = input as Record<string, unknown>
      if (Object.keys(value).some(key => !['sessionId', 'enabled'].includes(key)) || typeof value.sessionId !== 'string' || !value.sessionId || value.sessionId.length > 256 || typeof value.enabled !== 'boolean') throw new Error('Choose a paired device and whether to enable watch speech.')
      if (!deps.setWatchSpeech) throw new Error('Kokoro watch permissions are unavailable in this build.')
      return serial(async () => {
        if (!value.enabled) deps.watchSpeech?.revoke(value.sessionId as string)
        return {enabled: await deps.setWatchSpeech!(value.sessionId as string, value.enabled as boolean)}
      })
    })
    handle('mobile:bridgeStatus', async () => {
      requireSecurityRole('owner', 'inspect the private mobile connection')
      return bridge.status()
    })
    handle('mobile:configureBridge', async (_event, input: unknown) => {
      requireSecurityRole('owner', 'configure the private mobile connection')
      if (!input || typeof input !== 'object' || Array.isArray(input)) throw new Error('Mobile connection configuration is invalid.')
      const value = input as Record<string, unknown>
      if (Object.keys(value).some(key => !['enabled', 'host', 'port'].includes(key)) || typeof value.enabled !== 'boolean' || typeof value.host !== 'string' || typeof value.port !== 'number') throw new Error('Mobile connection configuration is invalid.')
      const config: MobileBridgeConfig = { version: 1, enabled: value.enabled, host: value.host, port: value.port }
      return serial(async () => {
        if (config.enabled) {
          // Reject a public/stale interface before enabling the local control plane.
          validateMobileBridgeConfig(config, bridge.status().interfaces)
          await deps.enableLoopback()
          await deps.remoteServer.syncFromSettings()
        }
        const status = await bridge.configure(config)
        try { await save(config) }
        catch {
          await bridge.stop()
          throw new Error('The mobile connection could not be saved; it has been stopped.')
        }
        return status
      })
    })
    handle('mobile:issuePairing', async (_event, label: unknown) => {
      requireSecurityRole('owner', 'pair a phone with TensorFold')
      if (label !== undefined && (typeof label !== 'string' || label.length > 120)) throw new Error('Use a phone name of up to 120 characters.')
      return serial(() => bridge.issuePairing(typeof label === 'string' ? label : 'Paired phone'))
    })
  }

  return { initialize, registerIpc, stop: () => bridge.stop(), status: () => bridge.status(), activeOperations: () => activeOperations + bridge.activeOperations() }
}
