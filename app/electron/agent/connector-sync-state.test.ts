import assert from 'node:assert/strict'
import { mergeConnectorSyncSources } from './connector-sync-state.ts'
import type { SyncCursor } from './types'

interface Source {
  id: string
  label: string
  token: string
  lastSyncAt: number | null
  lastError: string | null
  cursor?: SyncCursor
}

const current: Source[] = [
  { id: 'kept', label: 'Edited while syncing', token: 'new-token', lastSyncAt: 10, lastError: 'old', cursor: { position: 1, lastId: 'one', updatedAt: 10 } },
  { id: 'added', label: 'Added while syncing', token: 'added-token', lastSyncAt: null, lastError: null },
]
const staleScan: Source[] = [
  { id: 'kept', label: 'Stale label', token: 'old-token', lastSyncAt: 20, lastError: null, cursor: { position: 2, lastId: 'two', updatedAt: 20 } },
  { id: 'removed', label: 'Deleted while syncing', token: 'gone', lastSyncAt: 20, lastError: null },
]

const merged = mergeConnectorSyncSources(current, staleScan)
assert.equal(merged[0].label, 'Edited while syncing', 'user-edited configuration wins over the stale scan snapshot')
assert.equal(merged[0].token, 'new-token', 'a stale scan cannot roll back a credential')
assert.equal(merged[0].lastSyncAt, 20)
assert.equal(merged[0].lastError, null)
assert.deepEqual(merged[0].cursor, { position: 2, lastId: 'two', updatedAt: 20 })
assert.equal(merged.some((source) => source.id === 'removed'), false, 'a scan cannot resurrect a source removed while it ran')
assert.equal(merged.some((source) => source.id === 'added'), true, 'a scan preserves a source added while it ran')

console.log('connector sync state merge tests passed')
