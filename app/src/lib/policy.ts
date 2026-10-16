import type { Project, Provider } from './types'
import { providerAccountChoices } from './provider-accounts'

// A project's model policy (e.g. "Local only" for sensitive work) is a guardrail
// that narrows which providers a chat or room seat may use, and supplies a default.

export function allowedProviders(providers: Provider[], project?: Project | null): Provider[] {
  if (project?.localOnly) {
    return providers.filter((p) => p.kind === 'local')
  }
  return providers
}

// The provider+model a new chat/seat should start with inside this project.
export function projectDefaultModel(project: Project | null | undefined, providers: Provider[]): { providerId: string | null; providerAccountId: string | null; model: string | null } {
  const allowed = allowedProviders(providers, project)
  const p = allowed.find((x) => x.id === project?.defaultProviderId) || allowed[0] || null
  const accounts = p ? providerAccountChoices(p) : []
  const requestedAccountId = project?.defaultProviderAccountId ?? p?.defaultAccountId ?? null
  const account = accounts.find((choice) => choice.id === requestedAccountId) ?? accounts[0] ?? null
  const models = account?.models?.length ? account.models : (p?.models ?? [])
  const model = project?.defaultModel && models.includes(project.defaultModel) ? project.defaultModel : (models[0] ?? null)
  return { providerId: p?.id ?? null, providerAccountId: account?.id ?? null, model }
}

// Is this provider/model allowed under the project's policy?
export function violatesPolicy(project: Project | null | undefined, provider: Provider | undefined): boolean {
  return !!project?.localOnly && !!provider && provider.kind !== 'local'
}
