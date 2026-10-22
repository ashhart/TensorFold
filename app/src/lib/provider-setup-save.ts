import type { Provider, Settings } from './types'

type ProviderSettings = Pick<Settings, 'providers' | 'activeProviderId' | 'activeModel'>

export async function persistProviderSetup<T extends ProviderSettings>(
  draft: T,
  provider: Provider,
  models: string[] | undefined,
  save: (next: T) => Promise<void>,
): Promise<Provider> {
  const nextProvider = models?.length ? { ...provider, models, discoveredModels: models } : provider
  const next = {
    ...draft,
    providers: draft.providers.some((item) => item.id === nextProvider.id)
      ? draft.providers.map((item) => item.id === nextProvider.id ? nextProvider : item)
      : [...draft.providers, nextProvider],
    activeProviderId: draft.activeProviderId ?? nextProvider.id,
    activeModel: draft.activeProviderId ? draft.activeModel : (nextProvider.models[0] ?? draft.activeModel),
  }
  await save(next)
  return nextProvider
}
