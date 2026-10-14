import type { EventSpineHost } from '../../../agent/event-spine-host'
import type { SpineAggregate } from '../../../agent/event-spine'
import { requireSecurityRole } from '../audit'
import { handle } from '../ipc'

export interface SpineIpcDeps {
  spine: () => EventSpineHost | null
}

export function registerSpineIpc(deps: SpineIpcDeps) {
  const { spine } = deps

// ---- ordered event spine (VST-034) ----
// Cursor-based replay for renderer/remote projectors + command-receipt lookups. Reads are
// operator-gated (payloads carry titles/ids); renderer cursors are namespaced so a client
// can never move a host-internal projector.
handle('spine:readSince', async (_e, input: { since?: number; aggregate?: SpineAggregate; streamId?: string; limit?: number }) => {
  requireSecurityRole('operator', 'read the event spine')
  const host = spine()
  if (!host) return { events: [], hasMore: false, snapshotRequired: false }
  const limit = Math.min(Math.max(1, Math.floor(Number(input?.limit ?? 500)) || 500), 2000)
  const since = Number(input?.since ?? 0)
  return host.readSince(Number.isFinite(since) ? since : 0, { aggregate: input?.aggregate, streamId: input?.streamId, limit })
})

handle('spine:receipt', async (_e, commandId: string) => {
  requireSecurityRole('operator', 'read command receipts')
  const id = String(commandId ?? '').trim()
  return id ? spine()?.receipt(id) ?? null : null
})

handle('spine:advanceCursor', async (_e, input: { projector?: string; sequence?: number }) => {
  requireSecurityRole('operator', 'advance a spine projector cursor')
  const projector = String(input?.projector ?? '').trim()
  const sequence = Number(input?.sequence ?? NaN)
  if (!projector || !Number.isFinite(sequence) || sequence < 0) throw new Error('A projector name and a non-negative sequence are required.')
  return spine()?.advanceCursor(`renderer:${projector}`, Math.floor(sequence)) ?? 0
})

}
