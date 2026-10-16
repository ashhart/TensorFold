import { mergeProviderStatus, normalizeCodexSnapshot, type PersistedProviderStatus } from '../../agent/provider-status'

interface ProviderStatusRuntimeDeps {
  saveProviderStatus(state: PersistedProviderStatus): Promise<void>
  sendToRenderer(channel: string, payload: unknown): void
}

/**
 * Provider rate-limit status (VST-037): real codex account windows persisted per
 * provider/account so the model picker + readiness doctor show live pace/ETA.
 *
 * In-memory map restored at boot; each recorded snapshot merges and fire-and-forget
 * persists behind a single write tail. The purge flow can quiesce writes (stop
 * persisting + await the tail), clear the map, and resume.
 */
export function createProviderStatusRuntime(deps: ProviderStatusRuntimeDeps) {
  let providerStatus: PersistedProviderStatus = { version: 1, statuses: {} }
  let providerStatusWriteTail: Promise<void> = Promise.resolve()
  let providerStatusWritesQuiesced = false

  function status(): PersistedProviderStatus {
    return providerStatus
  }

  // Restore last-known windows so the picker/doctor show something before the next codex
  // turn refreshes them. Used by the boot restore path.
  function setStatus(next: PersistedProviderStatus): void {
    providerStatus = next
  }

  async function quiesce(): Promise<void> {
    providerStatusWritesQuiesced = true
    await providerStatusWriteTail.catch(() => {})
  }

  function resume(): void {
    providerStatusWritesQuiesced = false
  }

  function record(providerId: string, accountId: string | null, rawSnapshot: unknown): void {
    if (providerStatusWritesQuiesced) return
    const raw = normalizeCodexSnapshot(rawSnapshot)
    if (!raw) return
    providerStatus = mergeProviderStatus(providerStatus, { providerId, accountId, raw, updatedAt: Date.now() })
    const snapshot = structuredClone(providerStatus)
    const next = providerStatusWriteTail.catch(() => {}).then(() => deps.saveProviderStatus(snapshot))
    providerStatusWriteTail = next.catch(() => {})
    deps.sendToRenderer('providerStatus:update', providerStatus)
  }

  async function clear(): Promise<void> {
    providerStatus = { version: 1, statuses: {} }
    await deps.saveProviderStatus(providerStatus)
    deps.sendToRenderer('providerStatus:update', providerStatus)
  }

  return { status, setStatus, record, quiesce, clear, resume }
}
