// Transient confirmation for connector "Sources" actions. The fix for "I click
// Sync and nothing happens, no error": a successful sync that finds nothing still
// resolves cleanly (disabled/empty/already-synced sources), so the only state
// change was an invisible lastSyncAt. These helpers produce an explicit, visible
// success message so the user sees the action confirmed — distinct from the
// already-working global error toast.

export interface ConnectorStatus {
  state: 'ok' | 'error'
  msg: string
}

export function plural(word: string, n: number): string {
  return n === 1 ? word : `${word}s`
}

// e.g. formatSyncSummary(1, 0, 'message') => "Scanned 1 source · 0 messages"
export function formatSyncSummary(sourceCount: number, itemCount: number, noun: string): string {
  if (sourceCount <= 0) return 'No sources to scan'
  return `Scanned ${sourceCount} ${plural('source', sourceCount)} · ${itemCount} ${plural(noun, itemCount)}`
}
