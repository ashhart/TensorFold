import type { RemoteOrchestratorProfile, Settings } from './types'
import { validateWhatsAppSourcePatch } from './whatsapp-bridge'

type RemoteUrlNormalizer = (raw: string) => Promise<string>

/** Credential-bearing WhatsApp rows may only be removed by the dedicated shred workflow. */
export function assertDedicatedWhatsAppSourceRemoval(next: Settings, current: Settings): void {
  const incoming = new Set((next.whatsapp?.sources ?? []).map((source) => source.id))
  const removed = (current.whatsapp?.sources ?? []).filter((source) => !incoming.has(source.id))
  if (removed.length) {
    throw new Error('WhatsApp accounts must be removed with the dedicated Remove account control so linked-device credentials are confirmed and shredded.')
  }
}

/** Whole-settings drafts never own linked-device identity, lifecycle, or contact grants. */
export function normalizeWhatsAppSettingsForSave(next: Settings, current: Settings): Settings['whatsapp'] {
  assertDedicatedWhatsAppSourceRemoval(next, current)
  const existing = new Set(current.whatsapp.sources.map((source) => source.id))
  for (const source of next.whatsapp?.sources ?? []) {
    if (!existing.has(source.id)) throw new Error('WhatsApp accounts must be added with the dedicated Add account control.')
    validateWhatsAppSourcePatch({ label: source.label, enabled: source.enabled, ...(source.allowedChatJids !== undefined ? { allowedChatJids: source.allowedChatJids } : {}), ...(source.allowSelfChat !== undefined ? { allowSelfChat: source.allowSelfChat } : {}) })
  }
  // Preserve the latest complete rows, including saved contact scopes and disconnect intent.
  // All editable account fields now have a dedicated owner-gated patch transaction.
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

/**
 * Validate only remote destinations changed by this settings transaction.
 *
 * Whole-settings saves are used for theme, provider, and connector edits. Revalidating
 * an unchanged legacy/offline remote URL on every one of those saves makes that stale
 * row a global write lock. A changed/newly-enabled credential-bearing destination still
 * goes through the full DNS/SSRF normalizer before it reaches the live store.
 */
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
