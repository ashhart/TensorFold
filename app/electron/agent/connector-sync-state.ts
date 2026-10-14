import type { SyncCursor } from './types'

type SyncOwnedSourceState = {
  id: string
  lastSyncAt?: number | null
  lastError?: string | null
  cursor?: SyncCursor
}

const hasOwn = (value: object, key: PropertyKey) => Object.prototype.hasOwnProperty.call(value, key)

/**
 * Apply only fields owned by a connector scan to the latest settings snapshot.
 *
 * A scan starts from an older source object and may finish after the user changed a
 * label, token, path, allowlist, or even removed the source. Replacing `sources` with
 * the scan result rolls those edits back. This merge behaves like a small CAS: current
 * membership/configuration wins, while cursor/error/timestamp advances from the scan.
 */
export function mergeConnectorSyncSources<T extends SyncOwnedSourceState>(
  current: T[],
  synced: ReadonlyArray<SyncOwnedSourceState>,
): T[] {
  const updates = new Map(synced.map((source) => [source.id, source]))
  return current.map((source) => {
    const update = updates.get(source.id)
    if (!update) return source
    return {
      ...source,
      ...(hasOwn(update, 'lastSyncAt') ? { lastSyncAt: update.lastSyncAt } : {}),
      ...(hasOwn(update, 'lastError') ? { lastError: update.lastError } : {}),
      ...(hasOwn(update, 'cursor') ? { cursor: update.cursor } : {}),
    }
  })
}
