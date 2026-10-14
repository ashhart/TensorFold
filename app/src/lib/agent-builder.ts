import type { Provider } from './types'
import { providerModelGroups } from './provider-accounts'

// Kept outside the TSX component so the model-source summary has a fast regression seam.
export function agentBuilderModelSourceCount(providers: Provider[]): number {
  return providers.reduce((count, provider) => {
    return count + providerModelGroups(provider).filter((group) => group.models.length > 0).length
  }, 0)
}
