import { randomUUID } from 'node:crypto'
import { persistSettings } from './persistence'
import { checkCodexCli } from '../../agent/codex-cli'
import { checkClaudeCli } from '../../agent/claude-cli'
import { defaultCliModels } from '../../agent/cli-models'
import { builtinClis, cliToolStatuses } from '../../agent/cli-registry'
import { fetchAiGatewayModels, isAiGatewayProvider } from '../../agent/model-gateway'
import { contextsFromModelsPayload, fetchLmStudioModels, fetchOllamaModels, fetchOpenAiCompatibleModelsDetailed, isLmStudioProvider, isOllamaProvider, isOpenRouterProvider } from '../../agent/local-providers'
import { recordHostedContext, recordHostedContexts } from '../../agent/hosted-context'
import { providerWithAccount, resolveProviderModel, selectProviderAccount } from '../../agent/provider-accounts'
import { auditPreview, isSecretMask, maskedSecretUrlError } from '../../agent/security'
import { createProviderFetch } from '../../agent/secure-fetch'
import { readBoundedJsonResponse } from '../../agent/bounded-response'
import { redactText } from '../../agent/export-redaction'
import { ModelRouter, type ModelExhaustionSignal, type ModelRouteAudit, type ModelRouteDecision, type ModelRouteState } from '../../agent/model-router'
import type {
  KanbanPhaseModel,
  ModelFailoverResetPolicy,
  ModelRouterCooldownSnapshot,
  ModelRouterHistoryEntry,
  ModelRouterSnapshot,
  Project,
  Provider,
  Thread,
  TicketWorkOptions,
} from '../../agent/types'
import { store } from '../state'
import { isEngineProvider } from '../../agent/engine/provider'
import { engineService } from './engine-runtime'
import { auditEvent } from './audit'

const OPENROUTER_PRIORITY_MODELS = [
  'openrouter/free',
  'openrouter/auto',
  'openrouter/fusion',
  'z-ai/glm-5.2',
  'nvidia/nemotron-3-ultra-550b-a55b:free',
  'nvidia/nemotron-3-super-120b-a12b:free',
  'qwen/qwen3-coder:free',
  'openai/gpt-oss-120b:free',
]

export const modelRouter = new ModelRouter({
  getState: () => store.modelRouteState,
  setState: (state) => { store.modelRouteState = state },
})

export function selectedProviderAccountId(provider: Provider, accountId?: string | null): string | null {
  const effectiveAccountId = accountId === undefined
    ? (store.settings.activeProviderAccountId ?? provider.defaultAccountId ?? null)
    : accountId
  return selectProviderAccount(provider, effectiveAccountId)?.id ?? null
}

export function runtimeProvider(provider: Provider, accountId?: string | null): Provider {
  const effectiveAccountId = accountId === undefined
    ? (store.settings.activeProviderAccountId ?? provider.defaultAccountId ?? null)
    : accountId
  return providerWithAccount(provider, effectiveAccountId)
}

function providerUsesLocalCompatibleModels(provider: Provider): boolean {
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') return false
  // The built-in engine's models come from `tensorfold-native models`, not a /models endpoint.
  if (isEngineProvider(provider)) return false
  if (provider.kind === 'local') return true
  return provider.source === 'omlx' || isOllamaProvider(provider) || isLmStudioProvider(provider)
}

export function rememberProviderModels(baseProviderId: string, accountId: string | null, models: string[]): void {
  const clean = Array.from(new Set(models.map((model) => model.trim()).filter(Boolean)))
  if (!clean.length) return
  const index = store.settings.providers.findIndex((provider) => provider.id === baseProviderId)
  if (index < 0) return
  const provider = store.settings.providers[index]
  if (accountId) {
    const accounts = (provider.accounts ?? []).map((account) => account.id === accountId ? { ...account, models: clean, updatedAt: Date.now() } : account)
    store.settings.providers[index] = { ...provider, accounts }
  } else {
    store.settings.providers[index] = { ...provider, models: clean }
  }
  if (store.settings.activeProviderId === baseProviderId && !clean.includes(store.settings.activeModel ?? '')) {
    store.settings = { ...store.settings, activeModel: clean[0] ?? store.settings.activeModel, providers: store.settings.providers }
  }
  persistSettings('providerModels')
}

export async function runtimeProviderWithFreshModels(baseProvider: Provider, accountId?: string | null): Promise<Provider> {
  let provider = runtimeProvider(baseProvider, accountId)
  if (!providerUsesLocalCompatibleModels(provider)) return provider
  try {
    const listed = await fetchOpenAiCompatibleModelsDetailed(provider.baseUrl, provider.apiKey, 1500)
    if (listed.models.length) {
      // The provider's own answer carries each model's window (vLLM's max_model_len); keep it.
      recordHostedContexts(baseProvider.id, listed.models, listed.contextTokens)
      rememberProviderModels(baseProvider.id, accountId ?? null, listed.models)
      provider = { ...provider, models: listed.models }
    }
  } catch {
    // Keep the saved model list when a local server does not expose /models.
  }
  return provider
}

async function fetchOpenRouterModels(provider: Provider, fetchImpl: typeof fetch): Promise<string[]> {
  const url = provider.baseUrl.replace(/\/+$/, '') + '/models?sort=pricing-low-to-high&supported_parameters=tools'
  const res = await fetchImpl(url, {
    headers: {
      ...(provider.apiKey ? { Authorization: `Bearer ${provider.apiKey}` } : {}),
      'HTTP-Referer': 'https://vontra.local',
      'X-Title': 'TensorFold',
    },
    signal: AbortSignal.timeout(10_000),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText || ''}`.trim())
  const json: any = await readBoundedJsonResponse(res, 'OpenRouter models')
  const data: any[] = json?.data ?? []
  const ids = data.map((m) => m?.id).filter(Boolean) as string[]
  return Array.from(new Set([...OPENROUTER_PRIORITY_MODELS, ...ids]))
}

export function resolveProviderProbe(provider: Provider): Provider {
  const stored = store.settings.providers.find((candidate) => candidate.id === provider.id)
  // A stored (masked) key must never be sent to a renderer-chosen URL: swapping baseUrl
  // while replaying the mask would exfiltrate the real key to an attacker host.
  const maskedEnvironment = (env?: Record<string, string>) => Object.values(env ?? {}).some(isSecretMask)
  const hasMaskedCredential = isSecretMask(provider.apiKey) || maskedEnvironment(provider.cliEnv) || (provider.accounts ?? []).some((account) => isSecretMask(account.apiKey) || maskedEnvironment(account.env))
  if (hasMaskedCredential && !stored) throw new Error('The saved provider secret is unavailable. Re-enter the key before testing.')
  const maskedConflict = maskedSecretUrlError(hasMaskedCredential, provider.baseUrl, stored?.baseUrl)
  if (maskedConflict) throw new Error(maskedConflict)
  if (stored && hasMaskedCredential) {
    if ((provider.transport ?? 'chat-completions') !== (stored.transport ?? 'chat-completions')) throw new Error('A saved secret can only be used with its saved transport; re-enter it for a different connection.')
    const launchBinding = (value: { cliPath?: string; cliProfile?: string; cliHome?: string }) => JSON.stringify([value.cliPath?.trim() ?? '', value.cliProfile?.trim() ?? '', value.cliHome?.trim() ?? ''])
    if ((maskedEnvironment(provider.cliEnv) || provider.transport?.endsWith('-cli')) && launchBinding(provider) !== launchBinding(stored)) throw new Error('A saved CLI secret can only be used with its saved executable and launch profile; re-enter it for a different launch.')
    for (const account of provider.accounts ?? []) {
      const existing = stored.accounts?.find((candidate) => candidate.id === account.id)
      if (maskedEnvironment(account.env) && (!existing || launchBinding(account) !== launchBinding(existing))) throw new Error('A saved CLI account secret can only be used with its saved launch profile; re-enter it for a different launch.')
    }
  }
  return providerWithAccount(stored ? {
    ...provider,
    apiKey: isSecretMask(provider.apiKey) ? stored.apiKey : provider.apiKey,
    cliEnv: Object.fromEntries(Object.entries(provider.cliEnv ?? {}).map(([key, value]) => [key, isSecretMask(value) ? (stored.cliEnv?.[key] ?? '') : value])),
    accounts: provider.accounts?.map((account) => {
      const existing = stored.accounts?.find((candidate) => candidate.id === account.id)
      return {
        ...account,
        apiKey: isSecretMask(account.apiKey) ? (existing?.apiKey ?? '') : account.apiKey,
        env: Object.fromEntries(Object.entries(account.env ?? {}).map(([key, value]) => [key, isSecretMask(value) ? (existing?.env?.[key] ?? '') : value])),
      }
    }),
  } : provider, provider.defaultAccountId)
}

export async function listProviderModels(provider: Provider): Promise<string[]> {
  if (isEngineProvider(provider)) return ((await engineService()?.installed(true)) ?? []).map((m) => m.repo)
  const effectiveProvider = resolveProviderProbe(provider)
  if (effectiveProvider.transport === 'codex-cli') {
    await checkCodexCli(effectiveProvider)
    return effectiveProvider.models?.length ? effectiveProvider.models : defaultCliModels('codex-cli')
  }
  if (effectiveProvider.transport === 'claude-cli') {
    await checkClaudeCli(effectiveProvider)
    return effectiveProvider.models?.length ? effectiveProvider.models : defaultCliModels('claude-cli')
  }
  if (effectiveProvider.transport === 'pi-cli' || effectiveProvider.transport === 'cursor-cli' || effectiveProvider.transport === 'gemini-cli') {
    const toolId = effectiveProvider.transport === 'pi-cli' ? 'pi' : effectiveProvider.transport === 'cursor-cli' ? 'cursor' : 'gemini'
    const tool = cliToolStatuses(builtinClis(), [effectiveProvider]).find((candidate) => candidate.id === toolId)
    if (!tool?.installed) throw new Error(`${effectiveProvider.name || toolId} CLI is not installed. Set a valid binary path in Settings or install it first.`)
    return effectiveProvider.models?.length ? effectiveProvider.models : defaultCliModels(effectiveProvider.transport)
  }
  const providerFetch = createProviderFetch(effectiveProvider)
  if (isOpenRouterProvider(effectiveProvider)) return fetchOpenRouterModels(effectiveProvider, providerFetch)
  if (isAiGatewayProvider(effectiveProvider)) return fetchAiGatewayModels(effectiveProvider, providerFetch)
  if (isOllamaProvider(effectiveProvider)) return fetchOllamaModels(effectiveProvider.baseUrl)
  if (isLmStudioProvider(effectiveProvider)) return fetchLmStudioModels(effectiveProvider.baseUrl, effectiveProvider.apiKey)
  const url = effectiveProvider.baseUrl.replace(/\/+$/, '') + '/models'
  const res = await providerFetch(url, {
    headers: effectiveProvider.apiKey ? { Authorization: `Bearer ${effectiveProvider.apiKey}` } : {},
    signal: AbortSignal.timeout(8000),
  })
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText || ''}`.trim())
  const json: any = await readBoundedJsonResponse(res, 'Provider models')
  const data: any[] = json?.data ?? json?.models ?? (Array.isArray(json) ? json : [])
  const ids = data.map((m) => (typeof m === 'string' ? m : m?.id ?? m?.name)).filter(Boolean)
  const hosted = contextsFromModelsPayload(json)
  for (const id of ids) if (typeof id === 'string') recordHostedContext(effectiveProvider.id, id, hosted[id.trim()])
  return Array.from(new Set(ids)).sort() as string[]
}

export function modelRouteAccountLabel(provider: Provider, accountId: string | null): string {
  if (accountId === null) return 'provider default'
  return selectProviderAccount(provider, accountId)?.name ?? accountId
}

export function modelRouteAutoFailoverEnabled(provider: Provider, accountId: string | null): boolean {
  if (accountId === null) return false
  return selectProviderAccount(provider, accountId)?.autoFailover === true
}

export function modelRouteResetPolicy(provider: Provider, accountId: string | null): ModelFailoverResetPolicy {
  if (accountId === null) return 'return_after_reset'
  return selectProviderAccount(provider, accountId)?.failoverResetPolicy ?? 'return_after_reset'
}

export function modelRouteMemoryForDecision(provider: Provider, decision: ModelRouteDecision) {
  const resetAt = decision.candidates.find((candidate) => candidate.accountId === decision.requestedAccountId)?.exhaustedUntil ?? null
  return {
    preferredProviderAccountId: decision.requestedAccountId,
    failoverProviderAccountId: decision.selectedAccountId,
    resetPolicy: modelRouteResetPolicy(provider, decision.requestedAccountId),
    failedOverAt: Date.now(),
    resetAt,
    reason: decision.audit.reason,
  }
}

export function routeHandoffPreview(value: unknown, max = 220): string {
  return redactText(auditPreview(value, max)).text
}

export function rememberModelRouteHistory(
  scope: ModelRouterHistoryEntry['scope'],
  audit: ModelRouteAudit,
  signal?: ModelExhaustionSignal | null,
  subject?: { id?: string | null; label?: string | null },
) {
  if (audit.action !== 'failover') return
  const provider = store.settings.providers.find((candidate) => candidate.id === audit.providerId)
  const providerName = provider?.name ?? audit.providerId
  const entry: ModelRouterHistoryEntry = {
    id: randomUUID(),
    scope,
    subjectId: subject?.id ?? null,
    subjectLabel: subject?.label ?? null,
    providerId: audit.providerId,
    providerName,
    fromAccountId: audit.requestedAccountId,
    fromAccountName: provider ? modelRouteAccountLabel(provider, audit.requestedAccountId) : audit.requestedAccountId ?? 'provider default',
    toAccountId: audit.selectedAccountId,
    toAccountName: provider ? modelRouteAccountLabel(provider, audit.selectedAccountId) : audit.selectedAccountId ?? 'provider default',
    model: audit.model,
    reason: routeHandoffPreview(audit.reason, 260),
    exhaustionKind: signal?.kind ?? null,
    resetAt: audit.resetAt ?? signal?.resetAt ?? null,
    createdAt: Date.now(),
  }
  store.modelRouteHistory = [entry, ...store.modelRouteHistory].slice(0, 50)
}

export function modelRouterCooldownSnapshot(now = Date.now()): ModelRouterSnapshot {
  const cooldown = (entry: ModelRouteState['exhausted'][number]): ModelRouterCooldownSnapshot | null => {
    const provider = store.settings.providers.find((candidate) => candidate.id === entry.providerId)
    if (!provider) return null
    const account = entry.accountId ? selectProviderAccount(provider, entry.accountId) : null
    return {
      providerId: provider.id,
      providerName: provider.name,
      providerKind: provider.kind ?? 'cloud',
      providerTransport: provider.transport ?? null,
      accountId: entry.accountId,
      accountName: account?.name ?? modelRouteAccountLabel(provider, entry.accountId),
      accountKind: account?.kind ?? 'provider-default',
      model: entry.model,
      kind: entry.kind,
      observedAt: entry.observedAt,
      resetAt: entry.resetAt,
      source: entry.source,
    }
  }
  const rows = (store.modelRouteState.exhausted ?? [])
    .map(cooldown)
    .filter((entry): entry is ModelRouterCooldownSnapshot => Boolean(entry))
  return {
    generatedAt: now,
    activeCooldowns: rows
      .filter((entry) => entry.resetAt > now)
      .sort((a, b) => a.resetAt - b.resetAt)
      .slice(0, 20),
    recentlyReset: rows
      .filter((entry) => entry.resetAt <= now)
      .sort((a, b) => b.resetAt - a.resetAt)
      .slice(0, 10),
    history: store.modelRouteHistory.slice(0, 20),
  }
}

export async function auditModelRoute(thread: Thread, audit: ModelRouteAudit, signal?: ModelExhaustionSignal | null) {
  rememberModelRouteHistory('thread', audit, signal, { id: thread.id, label: thread.title })
  await auditEvent(`model_router.${audit.action}`, {
    threadId: thread.id,
    providerId: audit.providerId,
    model: audit.model,
    requestedAccountId: audit.requestedAccountId,
    selectedAccountId: audit.selectedAccountId,
    exhaustedAccountIds: audit.exhaustedAccountIds,
    eligibleAccountIds: audit.eligibleAccountIds,
    failoverChainAccountIds: audit.failoverChainAccountIds,
    resetEligibleAccountIds: audit.resetEligibleAccountIds,
    resetAt: audit.resetAt ?? signal?.resetAt ?? null,
    source: audit.source ?? signal?.source ?? null,
    reason: auditPreview(audit.reason),
    exhaustionKind: signal?.kind ?? null,
  })
}

export async function auditScopedModelRoute(scope: string, audit: ModelRouteAudit, signal?: ModelExhaustionSignal | null, extra: Record<string, unknown> = {}) {
  const historyScope = (['ticket', 'automation', 'mission', 'room', 'local_pr'].includes(scope) ? scope : 'thread') as ModelRouterHistoryEntry['scope']
  rememberModelRouteHistory(historyScope, audit, signal, {
    id: typeof extra.threadId === 'string' ? extra.threadId : typeof extra.roomId === 'string' ? extra.roomId : typeof extra.missionId === 'string' ? extra.missionId : typeof extra.localPrId === 'string' ? extra.localPrId : null,
    label: typeof extra.seatName === 'string' ? extra.seatName : typeof extra.title === 'string' ? extra.title : null,
  })
  await auditEvent(`model_router.${scope}.${audit.action}`, {
    ...extra,
    providerId: audit.providerId,
    model: audit.model,
    requestedAccountId: audit.requestedAccountId,
    selectedAccountId: audit.selectedAccountId,
    exhaustedAccountIds: audit.exhaustedAccountIds,
    eligibleAccountIds: audit.eligibleAccountIds,
    failoverChainAccountIds: audit.failoverChainAccountIds,
    resetEligibleAccountIds: audit.resetEligibleAccountIds,
    resetAt: audit.resetAt ?? signal?.resetAt ?? null,
    source: audit.source ?? signal?.source ?? null,
    reason: auditPreview(audit.reason),
    exhaustionKind: signal?.kind ?? null,
  })
}

export async function auditModelRetry(thread: Thread, input: { providerId: string; providerAccountId: string | null; fromModel: string; toModel: string; reason: string; source: string; availableModels?: string[] }) {
  await auditEvent('model_router.model_retry', {
    threadId: thread.id,
    providerId: input.providerId,
    providerAccountId: input.providerAccountId,
    fromModel: input.fromModel,
    toModel: input.toModel,
    source: input.source,
    availableModels: (input.availableModels ?? []).slice(0, 20),
    reason: auditPreview(input.reason),
  })
}

export function modelRouteFailoverText(provider: Provider, audit: ModelRouteAudit, signal?: ModelExhaustionSignal | null): string {
  const from = modelRouteAccountLabel(provider, audit.requestedAccountId)
  const to = modelRouteAccountLabel(provider, audit.selectedAccountId)
  const resetAt = audit.resetAt ?? signal?.resetAt
  const window = resetAt ? ` Reset window: ${new Date(resetAt).toISOString()}.` : ''
  const cause = signal?.kind ? `${signal.kind.replace(/_/g, ' ')}` : 'cooldown'
  return `Model router failover: ${from} hit ${cause} for ${audit.model}. Retrying on ${to}.${window}`
}

type ProjectRoutePurpose = 'kanban_draft' | 'ticket_build' | 'ticket_review' | 'kanban_final_gate'

function projectKanbanPhaseSelection(project: Project | null | undefined, purpose: ProjectRoutePurpose): KanbanPhaseModel | null {
  const kanban = project?.kanban
  if (!kanban) return null
  const fallback = kanban.single ?? null
  if (kanban.modelMode !== 'phased') return fallback
  if (purpose === 'kanban_draft') return kanban.draft ?? fallback
  if (purpose === 'ticket_build') return kanban.build ?? fallback
  if (purpose === 'ticket_review') return kanban.review ?? fallback
  if (purpose === 'kanban_final_gate') return kanban.finalGate ?? kanban.review ?? fallback
  return fallback
}

export function ticketWorkOptionsWithProjectRoute(project: Project | null | undefined, opts: TicketWorkOptions | undefined, purpose: ProjectRoutePurpose): TicketWorkOptions | undefined {
  const phase = opts?.phase
  const hasExplicitRoute = !!opts?.providerId || !!opts?.model || Object.prototype.hasOwnProperty.call(opts ?? {}, 'providerAccountId')
  if (hasExplicitRoute) return opts
  const selection = projectKanbanPhaseSelection(project, purpose)
  if (!selection?.providerId || !selection.model) return opts
  return {
    ...opts,
    phase,
    providerId: selection.providerId,
    providerAccountId: selection.providerAccountId ?? null,
    model: selection.model,
  }
}

export function resolveProjectProvider(project?: Project | null, accountId?: string | null, modelOverride?: string | null, providerId?: string | null): { baseProvider: Provider; provider: Provider; providerAccountId: string | null; model: string } {
  const allowed = project?.localOnly ? store.settings.providers.filter((p) => p.kind === 'local') : store.settings.providers
  const baseProvider =
    allowed.find((p) => p.id === providerId) ??
    allowed.find((p) => p.id === project?.defaultProviderId) ??
    allowed.find((p) => p.id === store.settings.activeProviderId) ??
    allowed[0]
  if (!baseProvider) throw new Error(project?.localOnly ? 'Configure a local provider before running this local-only project.' : 'Configure a provider and model first.')
  const requestedAccountId = accountId !== undefined ? accountId : (project?.defaultProviderAccountId ?? store.settings.activeProviderAccountId ?? baseProvider.defaultAccountId ?? null)
  const providerAccountId = selectedProviderAccountId(baseProvider, requestedAccountId)
  const activeModel = store.settings.activeProviderId === baseProvider.id ? store.settings.activeModel : null
  const model = resolveProviderModel(baseProvider, providerAccountId, modelOverride, project?.defaultModel, activeModel)
  if (!model) throw new Error(`Choose a model for ${baseProvider.name} before running this project.`)
  return { baseProvider, provider: runtimeProvider(baseProvider, providerAccountId), providerAccountId, model }
}
