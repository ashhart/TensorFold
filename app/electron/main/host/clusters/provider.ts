import type { Provider } from '../../../agent/types'
/** Managed endpoints are promoted only after main-process deployment and canary checks. */
export function preserveManagedComputeProviders(incoming: Provider[], current: Provider[]): Provider[] {
  return incoming.map(provider => {
    const existing = current.find(row => row.id === provider.id)
    if (!provider.id.startsWith('cluster:') && !provider.remoteCompute && !existing?.remoteCompute) return provider
    if (!existing?.remoteCompute || !existing.id.startsWith('cluster:')) throw new Error('Use the verified compute endpoint action to add a managed provider.')
    const { name: _name, ...original } = existing, { name: _editedName, ...edited } = provider
    if (JSON.stringify(original) !== JSON.stringify(edited)) throw new Error('Reconcile the managed endpoint in Compute and clusters before changing its connection.')
    return { ...existing, name: provider.name }
  })
}
