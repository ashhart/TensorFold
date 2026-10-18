import type { TensorFoldStudioApi } from '../../../shared/studio-api'
import type { Settings, WhatsAppBridgeRuntimeSnapshot, WhatsAppBridgeSource } from '../types'

type WhatsAppDemoApi = Pick<TensorFoldStudioApi, 'addWhatsAppSource' | 'updateWhatsAppSource' | 'removeWhatsAppSource' | 'startWhatsAppPairing' | 'disconnectWhatsApp' | 'whatsappRuntime' | 'onWhatsAppRuntime' | 'onWhatsAppQr' | 'onWhatsAppStatus'>
const unavailable = 'WhatsApp pairing is unavailable in browser demo mode.'

/** Editable demonstration accounts; no phone connection, credentials or QR data are created. */
export function createMockWhatsAppApi(settings: () => Settings, id: () => string): WhatsAppDemoApi {
  const runtime = new Map<string, WhatsAppBridgeRuntimeSnapshot>()
  const listeners = new Set<Parameters<TensorFoldStudioApi['onWhatsAppRuntime']>[0]>()
  const statusListeners = new Set<Parameters<TensorFoldStudioApi['onWhatsAppStatus']>[0]>()
  const snapshot = () => structuredClone(settings().whatsapp)
  const source = (sourceId: string) => {
    const found = settings().whatsapp.sources.find(item => item.id === sourceId)
    if (!found) throw new Error('WhatsApp demo account not found.')
    return found
  }
  const publish = (account: WhatsAppBridgeSource, phase: WhatsAppBridgeRuntimeSnapshot['phase'], error: string | null = null) => {
    const state: WhatsAppBridgeRuntimeSnapshot = {
      sourceId: account.id, phase, qr: null, qrExpiresAt: null, phoneNumber: null,
      reconnectAttempt: 0, nextReconnectAt: null, lastError: error, updatedAt: Date.now(),
    }
    runtime.set(account.id, state)
    for (const callback of listeners) callback(structuredClone(state))
    for (const callback of statusListeners) callback({ sourceId: account.id, paired: false, phoneNumber: null, credentialRotatedAt: null, enabled: account.enabled, lastError: error })
  }
  return {
    addWhatsAppSource: async label => {
      const now = Date.now()
      const account: WhatsAppBridgeSource = {
        id: `demo-whatsapp-${id()}`, label: label.trim().slice(0, 120) || 'WhatsApp account', enabled: false,
        allowedChatJids: [], allowSelfChat: false, phoneNumber: null, paired: false,
        lastQrAt: null, lastEventAt: null, lastError: null, credentialRotatedAt: null, createdAt: now, updatedAt: now,
      }
      settings().whatsapp.sources.push(account)
      publish(account, 'disconnected')
      return snapshot()
    },
    updateWhatsAppSource: async (sourceId, patch) => {
      const account = source(sourceId)
      const next = { ...account }
      const keys = Object.keys(patch)
      if (keys.some(key => !['label', 'enabled', 'allowedChatJids', 'allowSelfChat'].includes(key))) throw new Error('Unsupported WhatsApp demo account field.')
      if (patch.label !== undefined) {
        if (typeof patch.label !== 'string' || !patch.label.trim() || patch.label.length > 120) throw new Error('Use a WhatsApp account label of 1 to 120 characters.')
        next.label = patch.label.trim()
      }
      for (const key of ['enabled', 'allowSelfChat'] as const) {
        if (patch[key] !== undefined) {
          if (typeof patch[key] !== 'boolean') throw new Error('Invalid WhatsApp account toggle.')
          next[key] = patch[key]
        }
      }
      if (patch.allowedChatJids !== undefined) {
        if (!Array.isArray(patch.allowedChatJids) || patch.allowedChatJids.length > 200 || patch.allowedChatJids.some(value => typeof value !== 'string' || value.length > 128)) throw new Error('Invalid WhatsApp conversation selection.')
        next.allowedChatJids = [...new Set(patch.allowedChatJids)]
      }
      Object.assign(account, next, { updatedAt: Date.now() })
      publish(account, 'disconnected')
      return snapshot()
    },
    removeWhatsAppSource: async sourceId => {
      source(sourceId)
      settings().whatsapp.sources = settings().whatsapp.sources.filter(account => account.id !== sourceId)
      runtime.delete(sourceId)
      return snapshot()
    },
    startWhatsAppPairing: async sourceId => {
      const account = source(sourceId)
      account.lastError = unavailable
      publish(account, 'error', unavailable)
      return { started: false }
    },
    disconnectWhatsApp: async sourceId => {
      const account = source(sourceId)
      account.enabled = false
      account.lastError = null
      publish(account, 'disconnected')
      return { ok: true }
    },
    whatsappRuntime: async () => settings().whatsapp.sources.map(account => structuredClone(runtime.get(account.id) ?? {
      sourceId: account.id, phase: 'disconnected', qr: null, qrExpiresAt: null, phoneNumber: null,
      reconnectAttempt: 0, nextReconnectAt: null, lastError: null, updatedAt: account.updatedAt,
    })),
    onWhatsAppRuntime: callback => { listeners.add(callback); return () => { listeners.delete(callback) } },
    onWhatsAppStatus: callback => { statusListeners.add(callback); return () => { statusListeners.delete(callback) } },
    onWhatsAppQr: () => () => {},
  }
}
