import type { ClusterWorkspaceApi, ClusterWorkspaceSnapshot } from '../../../shared/cluster-workspace'
export interface ClusterClientState { snapshot: ClusterWorkspaceSnapshot | null; pending: string | null; error: string; loading: boolean }
function snapshot(value: unknown): value is ClusterWorkspaceSnapshot {
  if (!value || typeof value !== 'object') return false
  const raw = value as ClusterWorkspaceSnapshot
  return Number.isSafeInteger(raw.revision) && raw.revision >= 0 && ['hosts', 'plans', 'jobs', 'deployments'].every(key => Array.isArray(raw[key as keyof ClusterWorkspaceSnapshot]) && (raw[key as 'hosts'] as unknown[]).length <= 10_000)
}
/** Polls are read-only; they cannot replace newer action results or retry an uncertain command. */
export function createClusterClient(port: Pick<ClusterWorkspaceApi, 'clusterWorkspace'>) {
  let state: ClusterClientState = { snapshot: null, pending: null, error: '', loading: true }
  let live = true, read = 0, mutation = 0
  const listeners = new Set<(state: ClusterClientState) => void>()
  const emit = (patch: Partial<ClusterClientState>) => { state = { ...state, ...patch }; if (live) listeners.forEach(listener => listener(state)) }
  const accept = (value: unknown) => {
    if (!snapshot(value)) throw new Error('Invalid cluster workspace response.')
    if (state.snapshot && value.revision < state.snapshot.revision) return
    emit({ snapshot: value })
  }
  return {
    snapshot: () => state,
    activateView() { if (!live) { live = true; ++mutation; ++read; emit({ pending: null, loading: true }) } },
    subscribe(listener: (state: ClusterClientState) => void) { listeners.add(listener); listener(state); return () => { listeners.delete(listener) } },
    async refresh(silent = false) {
      if (!live || state.pending) return
      const own = ++read, version = mutation
      if (!silent) emit({ loading: true, error: '' })
      try { const value = await port.clusterWorkspace(); if (live && own === read && version === mutation) { accept(value); emit({ loading: false }) } }
      catch (error) { if (live && own === read && version === mutation) emit({ loading: false, ...(!silent || !state.snapshot ? { error: error instanceof Error ? error.message : String(error) } : {}) }) }
    },
    async execute<T>(label: string, action: () => Promise<T>): Promise<T | undefined> {
      if (!live || state.pending) return undefined
      const own = ++mutation; ++read; emit({ pending: label, error: '' })
      try {
        const value = await action()
        if (!live || own !== mutation) return undefined
        if (snapshot(value)) accept(value)
        return value
      } catch (error) { if (live && own === mutation) emit({ error: error instanceof Error ? error.message : String(error) }); return undefined }
      finally { if (live && own === mutation) emit({ pending: null, loading: false }) }
    },
    dispose() { live = false; ++mutation; ++read; listeners.clear() },
  }
}
export type ClusterClient = ReturnType<typeof createClusterClient>
