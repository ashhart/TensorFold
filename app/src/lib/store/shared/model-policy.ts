import { api } from '../../ipc'
import { allowedProviders, projectDefaultModel } from '../../policy'
import { resolveProviderModel } from '../../provider-accounts'
import { resolveSurfaceModel } from '../../assistant-model'
import type { Provider, Project, Room, Seat, Settings, Thread } from '../../types'
import type { AppState } from '../types'

export function selectedAccountId(provider: Provider | null | undefined, accountId?: string | null): string | null {
  const accounts = (provider?.accounts ?? []).filter((account) => account.enabled !== false)
  if (!accounts.length) return null
  if (accountId === null) return null
  if (accountId && accounts.some((account) => account.id === accountId)) return accountId
  if (accountId === undefined && provider?.defaultAccountId && accounts.some((account) => account.id === provider.defaultAccountId)) return provider.defaultAccountId
  return null
}

export function projectDefaults(project: Project | null | undefined): Pick<Thread, 'cwd' | 'workspace'> {
  const workspace = project?.workspace ?? null
  return { cwd: project?.cwd || workspace?.path || '', workspace }
}

export function runErrorMessage(e: unknown): string {
  return String((e as any)?.message ?? e ?? 'Run failed to start.')
}

export function defaultRoomSeats(settings: Settings | null, project: Project | null | undefined): Seat[] {
  if (!settings) return []
  const providers = allowedProviders(settings.providers, project).filter((provider) => provider.models.length > 0)
  const defaults = projectDefaultModel(project ?? null, settings.providers)
  const preferred = defaults.providerId ? providers.find((provider) => provider.id === defaults.providerId) : null
  const provider = preferred ?? providers[0]
  if (!provider) return []
  const accountId = selectedAccountId(provider, defaults.providerAccountId)
  const model = resolveProviderModel(provider, accountId, defaults.model ?? '') ?? provider.models[0]
  if (!model) return []
  const base = crypto.randomUUID()
  const mk = (suffix: string, name: string, role: string, lane: Seat['lane'], roleProfileId: Seat['roleProfileId']): Seat => ({
    id: `${base}-${suffix}`,
    name,
    role,
    lane,
    roleProfileId,
    fanOutBudget: 'single',
    providerId: provider.id,
    providerAccountId: accountId,
    model,
  })
  return [
    mk('planner', 'Planner', 'Plan the work, create executable tickets, and keep acceptance checks explicit.', 'planning', 'planner'),
    mk('builder', 'Builder', 'Build from room tickets and report file/test evidence.', 'development', 'builder'),
    mk('reviewer', 'Reviewer', 'Review worker evidence, risks, tests, and final gate blockers.', 'review', 'reviewer'),
  ]
}

export function roomProblemFromTicket(ticket: import('../../types').Ticket): string {
  const evidence = ticket.requiredEvidence?.length ? `\nRequired evidence:\n${ticket.requiredEvidence.map((item) => `- ${item}`).join('\n')}` : ''
  const acceptance = ticket.acceptanceCriteria?.length ? `\nAcceptance criteria:\n${ticket.acceptanceCriteria.map((item) => `- ${item}`).join('\n')}` : ''
  const caps = ticket.allowedCapabilities?.length ? `\nAllowed capabilities: ${ticket.allowedCapabilities.join(', ')}` : ''
  return [
    `Kanban ticket: ${ticket.title}`,
    ticket.body ? `\nDetails:\n${ticket.body}` : '',
    `\nType: ${ticket.type}`,
    `Priority: ${ticket.priority}`,
    `Effort: ${ticket.effort}`,
    `Risk: ${ticket.risk ?? 'low'}`,
    caps,
    evidence,
    acceptance,
    'Create any needed room tickets first, then start build/review work through the room supervisor.',
  ].filter(Boolean).join('\n')
}

// New chats/tasks inherit their project's model policy (e.g. a local-only project
// starts on its default local model), persisted so the headless run honours it too.
export function applyProjectModel(get: () => AppState, s: Settings | null, projectId: string | null | undefined, t: Thread) {
  if (!s) return
  const project = projectId ? get().projects.find((p) => p.id === projectId) ?? null : null
  const surface = resolveSurfaceModel(s, t.kind === 'chat' ? 'chat' : 'code')
  let providerId = surface.providerId
  let providerAccountId: string | null | undefined = surface.providerAccountId ?? undefined
  let model = surface.model
  if (project && (project.localOnly || project.defaultProviderId)) {
    const d = projectDefaultModel(project, s.providers)
    if (d.providerId) { providerId = d.providerId; model = d.model }
    providerAccountId = d.providerAccountId ?? providerAccountId
  }
  const provider = s.providers.find((p) => p.id === providerId)
  providerAccountId = selectedAccountId(provider, providerAccountId)
  model = provider ? (resolveProviderModel(provider, providerAccountId, model) ?? model) : model
  t.providerId = providerId; t.providerAccountId = providerAccountId; t.model = model
  if (providerId) api.configureThread(t.id, providerId, model ?? '', providerAccountId).catch(() => {})
}
