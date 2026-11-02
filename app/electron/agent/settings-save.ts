import type { RemoteOrchestratorProfile, Settings } from './types'
import { validateWhatsAppSourcePatch } from './whatsapp-bridge'
import { blueBubblesIdentity } from './bluebubbles-connector'

type RemoteUrlNormalizer = (raw: string) => Promise<string>

// Account membership and identity are owned by sign-in and disconnect IPC, not stale settings drafts.
export function normalizeMicrosoftMailSettingsForSave(next: Settings, current: Settings): Settings['email'] {
  const accounts = current.email.sources.filter(source => source.kind === 'microsoft_graph')
  const ids = new Set(accounts.map(source => source.id))
  if (next.email.sources.some(source => source.kind === 'microsoft_graph' && !ids.has(source.id))) throw new Error('Microsoft accounts must be connected with Microsoft sign-in.')
  return { ...next.email, sources: [...next.email.sources.filter(source => source.kind !== 'microsoft_graph' && !ids.has(source.id)), ...accounts] }
}

export function normalizeIMessageSettingsForSave(next: Settings, current: Settings): Settings['imessage'] {
  const settings = next.imessage
  if (settings.transport !== undefined && !['local', 'bluebubbles'].includes(settings.transport)) throw new Error('Choose local iMessage or BlueBubbles.')
  for (const value of [settings.allowedChatIds, settings.blueBubblesUrl, settings.blueBubblesPassword]) {
    if (value !== undefined && (typeof value !== 'string' || value.length > 20_000 || value.includes('\0'))) throw new Error('Invalid iMessage connection settings.')
  }
  const changed = blueBubblesIdentity(settings) !== blueBubblesIdentity(current.imessage)
  return { ...settings, cursor: changed ? undefined : current.imessage.cursor,
    bridgeCursorIdentity: changed ? undefined : current.imessage.bridgeCursorIdentity,
    lastSyncAt: changed ? null : current.imessage.lastSyncAt, lastError: changed ? null : current.imessage.lastError }
}

// Credential-bearing WhatsApp rows may only be removed by the dedicated shred workflow.
export function assertDedicatedWhatsAppSourceRemoval(next: Settings, current: Settings): void {
  const incoming = new Set((next.whatsapp?.sources ?? []).map((source) => source.id))
  const removed = (current.whatsapp?.sources ?? []).filter((source) => !incoming.has(source.id))
  if (removed.length) {
    throw new Error('WhatsApp accounts must be removed with the dedicated Remove account control so linked-device credentials are confirmed and shredded.')
  }
}

// Whole-settings drafts never own linked-device identity, lifecycle, or contact grants.
export function normalizeWhatsAppSettingsForSave(next: Settings, current: Settings): Settings['whatsapp'] {
  assertDedicatedWhatsAppSourceRemoval(next, current)
  const existing = new Set(current.whatsapp.sources.map((source) => source.id))
  for (const source of next.whatsapp?.sources ?? []) {
    if (!existing.has(source.id)) throw new Error('WhatsApp accounts must be added with the dedicated Add account control.')
    validateWhatsAppSourcePatch({ label: source.label, enabled: source.enabled, ...(source.allowedChatJids !== undefined ? { allowedChatJids: source.allowedChatJids } : {}), ...(source.allowSelfChat !== undefined ? { allowSelfChat: source.allowSelfChat } : {}) })
  }

  return current.whatsapp
}

function remoteClientNeedsValidation(next: Settings, current: Settings): boolean {
  const a = next.remoteClient
  const b = current.remoteClient
  return a.serverUrl !== b.serverUrl || a.authToken !== b.authToken || (a.enabled && !b.enabled)
}

function remoteHostNeedsValidation(next: RemoteOrchestratorProfile, current: RemoteOrchestratorProfile | undefined): boolean {
  return !current ||
    next.serverUrl !== current.serverUrl ||
    next.authToken !== current.authToken ||
    (next.enabled && !current.enabled)
}

export async function normalizeChangedRemoteDestinations(
  next: Settings,
  current: Settings,
  normalize: RemoteUrlNormalizer,
): Promise<Settings> {
  let remoteClient = { ...next.remoteClient }
  if (remoteClient.serverUrl && remoteClient.authToken && remoteClientNeedsValidation(next, current)) {
    remoteClient = { ...remoteClient, serverUrl: await normalize(remoteClient.serverUrl) }
  }

  const currentHosts = new Map(current.remoteOrchestrators.map((host) => [host.id, host]))
  const remoteOrchestrators = await Promise.all(next.remoteOrchestrators.map(async (host) => {
    if (!host.serverUrl || !host.authToken || !remoteHostNeedsValidation(host, currentHosts.get(host.id))) return host
    return { ...host, serverUrl: await normalize(host.serverUrl) }
  }))

  return { ...next, remoteClient, remoteOrchestrators }
}
