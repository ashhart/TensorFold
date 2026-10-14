import type { Provider, Seat } from './types'
import { resolveProviderModel, selectProviderAccount } from './provider-accounts'

// Cross-model delegation routing (Traycer-style): a single spawn binds provider/account/model,
// and every override field is optional. Null or absent means "inherit the caller's route".

export interface DelegationRouteBase {
  providerId: string
  // Undefined is preserved (vs null) because runtimeProvider treats an undefined account as
  // "fall back to the active/default account" while null means "no account".
  providerAccountId?: string | null
  model: string
}

export interface DelegationRouteOverride {
  targetSeat?: string | null
  providerId?: string | null
  accountId?: string | null
  model?: string | null
}

export type DelegationRouteResult =
  | { route: DelegationRouteBase; notice?: string; error?: never }
  | { route?: never; notice?: never; error: string }

function clean(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

// Resolve the effective route for a delegated sub-agent. With no override fields the base
// route is returned untouched (exact inherit). `targetSeat` (rooms only) adopts that seat's
// provider/account/model as the base before explicit providerId/accountId/model apply.
// `localOnly` (project confinement) rejects any resolved route whose provider is not local.
export function resolveDelegationRoute(
  providers: Provider[],
  base: DelegationRouteBase,
  override: DelegationRouteOverride,
  seats?: Seat[],
  localOnly?: boolean,
): DelegationRouteResult {
  const targetSeat = clean(override.targetSeat)
  const providerId = clean(override.providerId)
  const accountId = clean(override.accountId)
  const model = clean(override.model)
  if (!targetSeat && !providerId && !accountId && !model) return { route: { ...base } }

  let route = { ...base }
  if (targetSeat) {
    const seat = seats?.find((candidate) => candidate.name === targetSeat)
    if (!seat) return { error: `unknown seat "${targetSeat}"` }
    route = { providerId: seat.providerId, providerAccountId: seat.providerAccountId, model: seat.model }
  }
  // A same-provider override keeps the inherited account; only a cross-provider switch
  // resets it (the account id is meaningless on the new provider).
  if (providerId && providerId !== route.providerId) route = { providerId, model: route.model }
  const provider = providers.find((candidate) => candidate.id === route.providerId)
  if (!provider) return { error: `unknown provider "${route.providerId}"` }
  if (localOnly && provider.kind !== 'local') return { error: `provider "${provider.name}" must be a local provider for this local-only project` }
  if (accountId) {
    if (!selectProviderAccount(provider, accountId)) return { error: `unknown account "${accountId}" for provider "${provider.name}"` }
    route.providerAccountId = accountId
  }
  const inheritedModel = route.model
  const resolvedModel = resolveProviderModel(provider, route.providerAccountId, model, inheritedModel) ?? inheritedModel
  route.model = resolvedModel
  // resolveProviderModel silently falls back to the provider's first choice when neither the
  // requested nor the inherited model is offered; surface that downgrade instead of hiding it.
  const notice = resolvedModel !== model && resolvedModel !== inheritedModel
    ? `Model fallback: provider "${provider.name}" does not offer the requested model, so the sub-agent will use "${resolvedModel}".`
    : undefined
  return notice ? { route, notice } : { route }
}
