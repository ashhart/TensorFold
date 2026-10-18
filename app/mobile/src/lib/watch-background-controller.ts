import type { DeviceSession } from './client'

export interface BackgroundWatchPort {
  configureBackground(input: { baseUrl: string; token: string; sessionId: string; certificatePin: string }): Promise<void>
  backgroundStatus(): Promise<{ configured: boolean; sessionId?: string; reason?: string }>
  disable(): Promise<void>
  publish(payload: string): Promise<void>
}

/** Native Keychain configuration is the single durable watch-sharing consent. */
export function createWatchBackgroundController(port: BackgroundWatchPort | null) {
  let revision = 0
  let tail: Promise<unknown> = Promise.resolve()
  const revoked = new Set<() => void>()
  const enqueue = <T,>(action: () => Promise<T>): Promise<T> => {
    const result = tail.then(action, action)
    tail = result.catch(() => {})
    return result
  }
  return {
    available: !!port,
    onRevoke(callback: () => void): () => void { revoked.add(callback); return () => { revoked.delete(callback) } },
    async restore(sessionId: string): Promise<{ enabled: boolean; reason?: string }> {
      const mine = revision
      return enqueue(async () => {
        if (!port) return { enabled: false }
        const status = await port.backgroundStatus()
        if (mine !== revision) return { enabled: false }
        if (status.configured && status.sessionId !== sessionId) { await port.disable(); return { enabled: false } }
        return { enabled: status.configured && status.sessionId === sessionId, ...(status.reason ? { reason: status.reason } : {}) }
      })
    },
    set(session: DeviceSession, enabled: boolean): Promise<boolean> {
      const mine = ++revision
      if (!enabled) void port?.disable().catch(() => {})
      const captured = { baseUrl: session.baseUrl, token: session.token, sessionId: session.sessionId, certificatePin: session.certificatePin ?? '' }
      return enqueue(async () => {
        if (mine !== revision) return false
        if (!port) throw new Error('Background watch access requires a new native iPhone build with the watch companion.')
        if (!enabled) { await port.disable(); return false }
        if (!/^https:\/\//i.test(captured.baseUrl) || !/^[a-fA-F0-9]{64}$/.test(captured.certificatePin)) {
          await port.disable()
          throw new Error('Pair using the private HTTPS QR code on your Mac before enabling background watch access.')
        }
        try {
          await port.configureBackground(captured)
          if (mine !== revision) { await port.disable(); return false }
          return true
        } catch {
          await port.disable()
          throw new Error('Background watch access could not be saved; sharing remains disabled.')
        }
      })
    },
    disable(): Promise<void> {
      ++revision
      for (const listener of revoked) { try { listener() } catch { /* Native revocation must proceed even if a view is gone. */ } }
      void port?.disable().catch(() => {})
      return enqueue(async () => { await port?.disable() })
    },
    publish(sessionId: string, payload: string): Promise<void> {
      const mine = revision
      return enqueue(async () => {
        if (!port || mine !== revision) return
        const status = await port.backgroundStatus()
        if (mine === revision && status.configured && status.sessionId === sessionId) await port.publish(payload)
      })
    },
  }
}
