// Pure resolver for the Automations editor model <select>. Extracted so the binding
// between a saved automation's (providerId, providerAccountId, model) and the rendered
// <option> set is unit-testable without a DOM.
//
// Why this exists: the option set is built by providerModelGroups(), which (a) DROPS the
// null "default login / provider key" group whenever a provider has named accounts, and
// (b) canonicalizes CLI model ids (e.g. opus-4.8 -> opus). A controlled <select> value
// rebuilt naively from the raw draft fields can therefore reference an option that is never
// rendered; the browser then silently selects its FIRST <option> (the provider's default
// model). To the user this looks like "the model I picked reverted to the default after
// reload", and the displayed default would overwrite the pinned model on the next save.
import type { Provider } from './types'
import { providerModelGroups, type ProviderModelGroup } from './provider-accounts'

export function modelChoiceValue(providerId: string, accountId: string | null | undefined, model: string): string {
  return JSON.stringify([providerId, accountId ?? null, model])
}

export function parseModelChoice(value: string): { providerId: string; providerAccountId: string | null; model: string } | null {
  try {
    const [providerId, providerAccountId, model] = JSON.parse(value)
    if (typeof providerId === 'string' && (typeof providerAccountId === 'string' || providerAccountId === null) && typeof model === 'string') {
      return { providerId, providerAccountId, model }
    }
  } catch {
    // Ignore malformed select values.
  }
  return null
}

export interface ModelGroupEntry { provider: Provider; group: ProviderModelGroup }

export interface ModelSelectState {
  /** provider backing the current draft (falls back to providers[0] when the saved id is gone) */
  provider: Provider | undefined
  /** flattened option groups across every provider */
  modelGroups: ModelGroupEntry[]
  /** the option group that represents the saved model, or null if none is rendered */
  selectedGroup: ModelGroupEntry | null
  /** value to bind to <select value=...>; always matches a rendered option when a model is set */
  selectedModelValue: string
  /** non-empty when the saved model has no real option and must be shown as a synthetic one */
  orphanModelValue: string
}

export function resolveModelSelect(
  providers: Provider[],
  draft: { providerId: string | null; providerAccountId?: string | null; model: string | null },
): ModelSelectState {
  const provider = providers.find((p) => p.id === draft.providerId) ?? providers[0]
  const modelGroups: ModelGroupEntry[] = providers.flatMap((p) => providerModelGroups(p).map((group) => ({ provider: p, group })))
  const accountId = draft.providerAccountId ?? null
  const model = draft.model

  // Prefer the exact (provider + account + model) option; fall back to the same provider's
  // group that lists this model (covers the dropped null-default-account case).
  const selectedGroup = model
    ? (modelGroups.find(({ provider: p, group }) => p.id === draft.providerId && group.accountId === accountId && group.models.includes(model))
      ?? modelGroups.find(({ provider: p, group }) => p.id === draft.providerId && group.models.includes(model))
      ?? null)
    : null

  const selectedModelValue = selectedGroup
    ? modelChoiceValue(selectedGroup.provider.id, selectedGroup.group.accountId, model!)
    : (provider && model ? modelChoiceValue(provider.id, accountId, model) : '')

  const orphanModelValue = model && !selectedGroup ? selectedModelValue : ''

  return { provider, modelGroups, selectedGroup, selectedModelValue, orphanModelValue }
}
