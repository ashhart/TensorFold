import type { Provider } from '../../../agent/types'
/**
 * Providers whose connection was created and verified in the main process: cluster deployments
 * and donation cards. The renderer may rename them, but never rewrite the URL, token or model
 * list — those go through their own actions.
 */
const isManagedId = (id: string) => id.startsWith('cluster:') || id.startsWith('donation:')
/** The fields that make up a managed connection; everything else is display or policy metadata. */
const CONNECTION_FIELDS = ['baseUrl', 'apiKey', 'models', 'transport', 'auth', 'remoteCompute', 'kind', 'source'] as const
function connectionSignature(provider: Provider): string {
  return JSON.stringify(CONNECTION_FIELDS.map((field) => {
    const value = (provider as unknown as Record<string, unknown>)[field]
    return field === 'models' ? [...((value as string[] | undefined) ?? [])] : value ?? null
  }))
}
/** Managed endpoints are promoted only after main-process deployment and canary checks. */
export function preserveManagedComputeProviders(incoming: Provider[], current: Provider[]): Provider[] {
  return incoming.map(provider => {
    const existing = current.find(row => row.id === provider.id)
    if (!isManagedId(provider.id) && !provider.remoteCompute && !existing?.remoteCompute) return provider
    if (!existing?.remoteCompute || !isManagedId(existing.id)) throw new Error('Use the verified compute endpoint action to add a managed provider.')
    if (connectionSignature(existing) !== connectionSignature(provider)) throw new Error('Reconcile the managed endpoint in Compute and clusters before changing its connection.')
    return { ...existing, name: provider.name }
  })
}
