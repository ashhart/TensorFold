import type { ModelSpendBudget, Provider, ProviderAccountKind, ProviderAccountProfile } from './types'

export type ModelRouteAccountId = string | null
export type ModelExhaustionKind = 'rate_limit' | 'quota_exhausted'

export interface ModelRouteExhaustion {
  providerId: string
  accountId: ModelRouteAccountId
  model: string
  kind: ModelExhaustionKind
  observedAt: number
  resetAt: number
  message: string
  source: string
}

export interface ModelRouteState {
  exhausted: ModelRouteExhaustion[]
}

export interface ModelExhaustionSignal {
  kind: ModelExhaustionKind
  status?: number
  resetAt: number
  retryAfterMs?: number
  message: string
  source: string
}

export interface ModelRouteCandidate {
  providerId: string
  accountId: ModelRouteAccountId
  accountName: string
  accountKind: ProviderAccountKind | 'provider-default'
  model: string
  preferred: boolean
  exhaustedUntil: number | null
}

export interface ModelRouteAudit {
  action: 'classify_error' | 'record_exhaustion' | 'select_route' | 'failover' | 'no_route'
  providerId: string
  model: string
  requestedAccountId: ModelRouteAccountId
  selectedAccountId: ModelRouteAccountId
  exhaustedAccountIds: ModelRouteAccountId[]
  eligibleAccountIds: ModelRouteAccountId[]
  failoverChainAccountIds: ModelRouteAccountId[]
  resetEligibleAccountIds: ModelRouteAccountId[]
  reason: string
  resetAt?: number
  source?: string
}

export interface ModelRouteDecision {
  providerId: string
  model: string
  requestedAccountId: ModelRouteAccountId
  selectedAccountId: ModelRouteAccountId
  selected: ModelRouteCandidate | null
  candidates: ModelRouteCandidate[]
  failoverChain: ModelRouteCandidate[]
  audit: ModelRouteAudit
}

export type ModelRouterRole =
  | 'chat'
  | 'task'
  | 'mission'
  | 'automation'
  | 'room_seat'
  | 'ticket_build'
  | 'ticket_review'
  | 'local_pr'
  | 'assistant_action'
  | 'planner'
  | 'builder'
  | 'reviewer'
  | 'summarizer'
  | 'scout'
  | 'connector'

export type ModelRouterCacheStrategy =
  | 'provider_default'
  | 'stable_prefix'
  | 'summarize_next'
  | 'compact_now'
  | 'split_thread'

export type ModelRouterScoreRole = 'planner' | 'builder' | 'reviewer' | 'summarizer' | 'scout' | 'connector'

export interface ModelRouterScorecard {
  providerId: string
  providerAccountId?: string | null
  model: string
  scores: Partial<Record<ModelRouterScoreRole, number>>
  latencyMs?: number | null
  costPerMillionTokens?: number | null
  contextTokens?: number | null
  successRate?: number | null
  updatedAt?: number | null
}

export interface ModelRouterRouteInput {
  role: ModelRouterRole
  provider: Provider
  model: string
  requestedAccountId?: string | null
  allowFailover?: boolean
  localOnly?: boolean
  now?: number
  state?: ModelRouteState
  toolNeeds?: string[]
  contextPressure?: number | null
  budget?: string | number | null
  providerHealth?: string | null
  accountQuota?: string | null
  scorecard?: string | number | null
  latencyMs?: number | null
  costEstimate?: number | null
  userPreference?: string | null
  contextLimit?: number | null
  reasoningEffort?: string | null
  cacheStrategy?: ModelRouterCacheStrategy | null
  source?: string | null
}

export interface ModelRouterRouteOutput {
  role: ModelRouterRole
  providerId: string
  accountId: ModelRouteAccountId
  model: string
  fallbackOrder: ModelRouteAccountId[]
  contextLimit: number
  reasoningEffort: string | null
  cacheStrategy: ModelRouterCacheStrategy
  spend: ModelRouterSpendStatus
  decision: ModelRouteDecision
  inputs: {
    toolNeeds: string[]
    contextPressure: number | null
    budget: string | number | null
    providerHealth: string | null
    accountQuota: string | null
    scorecard: string | number | null
    latencyMs: number | null
    costEstimate: number | null
    userPreference: string | null
    contextLimit: number | null
    source: string | null
  }
}

export interface ModelRouterBudgetCheck {
  enabled: boolean
  period: ModelSpendBudget['period'] | null
  limitUsd: number | null
  spentUsd: number
  projectedUsd: number | null
  remainingUsd: number | null
  hardStop: boolean
  blocked: boolean
  warning: string | null
}

export interface ModelRouterSpendStatus {
  projectedCostUsd: number | null
  provider: ModelRouterBudgetCheck | null
  account: ModelRouterBudgetCheck | null
  blocked: boolean
  warning: string | null
}

export interface ModelRouterRoleRouteInput {
  role: ModelRouterScoreRole
  providers: Provider[]
  requestedProviderId?: string | null
  requestedAccountId?: string | null
  requestedModel?: string | null
  localOnly?: boolean
  allowFailover?: boolean
  scorecards?: ModelRouterScorecard[]
  toolNeeds?: string[]
  contextPressure?: number | null
  budget?: string | number | null
  providerHealth?: string | null
  accountQuota?: string | null
  userPreference?: string | null
  contextLimit?: number | null
  reasoningEffort?: string | null
  cacheStrategy?: ModelRouterCacheStrategy | null
  source?: string | null
  now?: number
}

export interface ModelRouterRoleRanking {
  providerId: string
  providerName: string
  accountId: ModelRouteAccountId
  accountName: string
  model: string
  score: number
  scorecardScore: number | null
  heuristicScore: number
  reason: string
  route: ModelRouterRouteOutput
}

export interface ModelRouterRoleRouteResult {
  selected: ModelRouterRoleRanking | null
  rankings: ModelRouterRoleRanking[]
}

export interface ModelRouterErrorInput extends RecordModelExhaustionInput {
  role: ModelRouterRole
  toolNeeds?: string[]
  contextPressure?: number | null
  budget?: string | number | null
  providerHealth?: string | null
  accountQuota?: string | null
  scorecard?: string | number | null
  latencyMs?: number | null
  costEstimate?: number | null
  userPreference?: string | null
  contextLimit?: number | null
  reasoningEffort?: string | null
  cacheStrategy?: ModelRouterCacheStrategy | null
  source?: string | null
}

export interface ModelRouterErrorResult {
  state: ModelRouteState
  signal: ModelExhaustionSignal | null
  route: ModelRouteDecision
  audit: ModelRouteAudit[]
  output: ModelRouterRouteOutput
}

export interface ModelRouterStore {
  getState: () => ModelRouteState
  setState: (state: ModelRouteState) => void
}

export interface RecordModelExhaustionInput {
  provider: Provider
  accountId?: string | null
  model: string
  error: unknown
  allowFailover?: boolean
  localOnly?: boolean
  now?: number
  defaultCooldownMs?: number
}

export interface RecordModelExhaustionResult {
  state: ModelRouteState
  signal: ModelExhaustionSignal | null
  audit: ModelRouteAudit
}

const DEFAULT_RATE_LIMIT_COOLDOWN_MS = 60_000
const DEFAULT_QUOTA_COOLDOWN_MS = 60 * 60_000
const DEFAULT_CLOUD_CONTEXT_LIMIT = 128_000
const DEFAULT_LOCAL_CONTEXT_LIMIT = 24_000

function accountLabel(accountId: ModelRouteAccountId): string {
  return accountId ?? '(provider-default)'
}

function enabledAccounts(provider: Provider): ProviderAccountProfile[] {
  return (provider.accounts ?? []).filter((account) => account.enabled !== false)
}

function normalizedPreferredAccountId(provider: Provider, accountId?: string | null): ModelRouteAccountId {
  if (accountId === null) return null
  const accounts = enabledAccounts(provider)
  if (accountId && accounts.some((account) => account.id === accountId)) return accountId
  if (accountId === undefined && provider.defaultAccountId && accounts.some((account) => account.id === provider.defaultAccountId)) {
    return provider.defaultAccountId
  }
  return null
}

function accountModels(provider: Provider, account: ProviderAccountProfile | null): string[] {
  return account?.models?.length ? account.models : provider.models
}

function supportsModel(provider: Provider, account: ProviderAccountProfile | null, model: string): boolean {
  const models = accountModels(provider, account)
  return !models.length || models.includes(model)
}

function isLocalProvider(provider: Provider): boolean {
  if (provider.kind === 'local') return true
  return /(^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::|\/|$))/.test(provider.baseUrl)
}

function inferredLocalContextLimit(model: string): number {
  const m = model.toLowerCase()
  if (/\b(4k|4096)\b/.test(m)) return 3_000
  if (/\b(8k|8192)\b/.test(m)) return 6_000
  if (/\b(16k|16384)\b/.test(m)) return 12_000
  if (/\b(32k|32768)\b/.test(m)) return 24_000
  if (/\b(64k|65536)\b/.test(m)) return 48_000
  return DEFAULT_LOCAL_CONTEXT_LIMIT
}

function contextLimitForRoute(input: ModelRouterRouteInput): number {
  if (typeof input.contextLimit === 'number' && Number.isFinite(input.contextLimit) && input.contextLimit > 0) {
    return Math.floor(input.contextLimit)
  }
  if (isLocalProvider(input.provider)) return inferredLocalContextLimit(input.model)
  return DEFAULT_CLOUD_CONTEXT_LIMIT
}

function defaultCacheStrategy(input: ModelRouterRouteInput, contextLimit: number): ModelRouterCacheStrategy {
  if (input.cacheStrategy) return input.cacheStrategy
  const pressure = input.contextPressure
  if (typeof pressure === 'number' && Number.isFinite(pressure)) {
    if (pressure >= 1) return 'compact_now'
    if (pressure >= 0.85) return 'summarize_next'
    if (pressure >= 0.65) return 'stable_prefix'
  }
  if (isLocalProvider(input.provider) || contextLimit <= DEFAULT_LOCAL_CONTEXT_LIMIT) return 'stable_prefix'
  return 'provider_default'
}

function defaultReasoningEffort(role: ModelRouterRole): string | null {
  switch (role) {
    case 'summarizer':
    case 'connector':
    case 'automation':
      return 'low'
    case 'reviewer':
    case 'ticket_review':
    case 'planner':
    case 'mission':
      return 'medium'
    default:
      return null
  }
}

function scoreRoleToRouteRole(role: ModelRouterScoreRole): ModelRouterRole {
  switch (role) {
    case 'builder': return 'builder'
    case 'reviewer': return 'reviewer'
    case 'summarizer': return 'summarizer'
    case 'scout': return 'scout'
    case 'connector': return 'connector'
    case 'planner':
    default:
      return 'planner'
  }
}

function clampScore(value: unknown): number | null {
  const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  if (!Number.isFinite(n)) return null
  return Math.max(0, Math.min(1, n))
}

function scorecardForCandidate(scorecards: ModelRouterScorecard[], providerId: string, accountId: ModelRouteAccountId, model: string): ModelRouterScorecard | null {
  const exact = scorecards.find((card) =>
    card.providerId === providerId &&
    (card.providerAccountId ?? null) === accountId &&
    card.model === model
  )
  if (exact) return exact
  return scorecards.find((card) =>
    card.providerId === providerId &&
    (card.providerAccountId === undefined || card.providerAccountId === null) &&
    card.model === model
  ) ?? null
}

function roleHeuristicScore(role: ModelRouterScoreRole, provider: Provider, model: string): number {
  const m = model.toLowerCase()
  const local = isLocalProvider(provider)
  const frontier = /(opus|gpt-5|gpt-4\.1|o3|o4|gemini.*pro|claude.*4|sonnet)/i.test(m)
  const coder = /(coder|code|qwen|dev|deepseek|glm|gpt-oss)/i.test(m)
  const smallFast = /(mini|haiku|flash|small|8b|7b|4b)/i.test(m)
  switch (role) {
    case 'planner':
      return frontier ? 0.72 : local ? 0.48 : 0.6
    case 'builder':
      return coder ? (local ? 0.78 : 0.72) : local ? 0.58 : 0.55
    case 'reviewer':
      return frontier ? 0.78 : coder ? 0.64 : 0.58
    case 'summarizer':
      return smallFast ? 0.76 : local ? 0.62 : 0.66
    case 'scout':
      return local ? 0.72 : smallFast ? 0.68 : 0.6
    case 'connector':
      return smallFast ? 0.72 : local ? 0.56 : 0.64
    default:
      return 0.5
  }
}

function scorecardAdjustment(card: ModelRouterScorecard | null): number {
  if (!card) return 0
  const success = clampScore(card.successRate)
  const latencyPenalty = typeof card.latencyMs === 'number' && Number.isFinite(card.latencyMs)
    ? Math.min(0.12, Math.max(0, card.latencyMs - 1_000) / 20_000)
    : 0
  const costPenalty = typeof card.costPerMillionTokens === 'number' && Number.isFinite(card.costPerMillionTokens)
    ? Math.min(0.12, Math.max(0, card.costPerMillionTokens) / 200)
    : 0
  const reliabilityBoost = success == null ? 0 : (success - 0.5) * 0.16
  return reliabilityBoost - latencyPenalty - costPenalty
}

function budgetLimit(value: unknown): number | null {
  const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  return Number.isFinite(n) && n >= 0 ? n : null
}

function budgetSpent(value: unknown): number {
  const n = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : NaN
  return Number.isFinite(n) && n > 0 ? n : 0
}

function budgetCheck(budget: ModelSpendBudget | null | undefined, projectedCostUsd: number | null): ModelRouterBudgetCheck | null {
  if (!budget || budget.enabled === false) return null
  const limitUsd = budgetLimit(budget.limitUsd)
  const spentUsd = budgetSpent(budget.spentUsd)
  const hardStop = budget.hardStop === true
  const projected = projectedCostUsd != null && Number.isFinite(projectedCostUsd) && projectedCostUsd > 0 ? projectedCostUsd : null
  const total = spentUsd + (projected ?? 0)
  const remainingUsd = limitUsd == null ? null : Math.max(0, limitUsd - total)
  const blocked = hardStop && limitUsd != null && total > limitUsd
  const warnAtRaw = budgetLimit(budget.warnAtPercent)
  const warnAt = warnAtRaw == null ? 0.8 : Math.max(0, Math.min(1, warnAtRaw > 1 ? warnAtRaw / 100 : warnAtRaw))
  const warning = limitUsd != null && total >= limitUsd * warnAt
    ? `Projected spend ${total.toFixed(2)} USD is near the ${limitUsd.toFixed(2)} USD ${budget.period ?? 'run'} budget.`
    : null
  return {
    enabled: true,
    period: budget.period ?? null,
    limitUsd,
    spentUsd,
    projectedUsd: projected,
    remainingUsd,
    hardStop,
    blocked,
    warning,
  }
}

function accountForCandidate(provider: Provider, accountId: ModelRouteAccountId): ProviderAccountProfile | null {
  return accountId ? enabledAccounts(provider).find((account) => account.id === accountId) ?? null : null
}

/**
 * Budget verdict for a concrete route (VST-015): provider + account spend budgets checked
 * together — `blocked` when a hard-stop budget is exceeded, `warning` near the cap. Solo
 * thread runs consult this at start; role/room routes get it inside route() output.
 */
export function routeSpendStatus(provider: Provider, accountId: ModelRouteAccountId, projectedCostUsd: number | null = null): ModelRouterSpendStatus {
  return spendStatusForCandidate(provider, accountId, projectedCostUsd)
}

function spendStatusForCandidate(provider: Provider, accountId: ModelRouteAccountId, projectedCostUsd: number | null): ModelRouterSpendStatus {
  const account = accountForCandidate(provider, accountId)
  const providerBudget = budgetCheck(provider.spendBudget, projectedCostUsd)
  const accountBudget = budgetCheck(account?.spendBudget, projectedCostUsd)
  const warnings = [providerBudget?.warning, accountBudget?.warning].filter((value): value is string => Boolean(value))
  return {
    projectedCostUsd,
    provider: providerBudget,
    account: accountBudget,
    blocked: providerBudget?.blocked === true || accountBudget?.blocked === true,
    warning: warnings[0] ?? null,
  }
}

function preferredAccountKind(provider: Provider, preferredAccountId: ModelRouteAccountId): ProviderAccountKind | 'provider-default' {
  if (preferredAccountId === null) return 'provider-default'
  return enabledAccounts(provider).find((account) => account.id === preferredAccountId)?.kind ?? 'provider-default'
}

function equivalentAccount(provider: Provider, preferredAccountId: ModelRouteAccountId, account: ProviderAccountProfile): boolean {
  return account.kind === preferredAccountKind(provider, preferredAccountId)
}

function allCandidates(provider: Provider, model: string, preferredAccountId: ModelRouteAccountId): ModelRouteCandidate[] {
  const preferredKind = preferredAccountKind(provider, preferredAccountId)
  const accounts = preferredKind === 'provider-default' ? [] : enabledAccounts(provider)
    .filter((account) => equivalentAccount(provider, preferredAccountId, account))
    .filter((account) => supportsModel(provider, account, model))
    .map((account) => ({
      providerId: provider.id,
      accountId: account.id,
      accountName: account.name,
      accountKind: account.kind,
      model,
      preferred: account.id === preferredAccountId,
      exhaustedUntil: null,
    } satisfies ModelRouteCandidate))

  const includeProviderDefault = preferredKind === 'provider-default'
  const providerDefault = includeProviderDefault && supportsModel(provider, null, model)
    ? [{
      providerId: provider.id,
      accountId: null,
      accountName: provider.name,
      accountKind: 'provider-default' as const,
      model,
      preferred: preferredAccountId === null,
      exhaustedUntil: null,
    } satisfies ModelRouteCandidate]
    : []

  return [...providerDefault, ...accounts].sort((a, b) => {
    if (a.accountId === preferredAccountId) return -1
    if (b.accountId === preferredAccountId) return 1
    return accountLabel(a.accountId).localeCompare(accountLabel(b.accountId))
  })
}

function activeExhaustion(state: ModelRouteState | undefined, providerId: string, model: string, accountId: ModelRouteAccountId, now: number): ModelRouteExhaustion | null {
  return (state?.exhausted ?? [])
    .filter((entry) => entry.providerId === providerId && entry.model === model && entry.accountId === accountId && entry.resetAt > now)
    .sort((a, b) => b.resetAt - a.resetAt)[0] ?? null
}

function pruneExpired(state: ModelRouteState | undefined, now: number): ModelRouteState {
  return { exhausted: (state?.exhausted ?? []).filter((entry) => entry.resetAt > now) }
}

function getHeader(headers: unknown, name: string): string | null {
  if (!headers) return null
  const lower = name.toLowerCase()
  if (typeof (headers as { get?: unknown }).get === 'function') {
    const value = (headers as { get: (key: string) => string | null }).get(name)
    return value == null ? null : String(value)
  }
  if (headers instanceof Map) {
    const value = headers.get(name) ?? headers.get(lower)
    return value == null ? null : String(value)
  }
  if (typeof headers === 'object') {
    const record = headers as Record<string, unknown>
    const key = Object.keys(record).find((k) => k.toLowerCase() === lower)
    const value = key ? record[key] : undefined
    return value == null ? null : String(value)
  }
  return null
}

function retryAfterMs(value: string | null, now: number): number | null {
  if (!value) return null
  const seconds = Number(value)
  if (Number.isFinite(seconds) && seconds >= 0) return Math.round(seconds * 1000)
  const at = Date.parse(value)
  return Number.isFinite(at) && at > now ? at - now : null
}

function resetAtFromHeaders(headers: unknown, now: number): { resetAt: number | null; retryAfterMs?: number; source?: string } {
  const retryMs = retryAfterMs(getHeader(headers, 'retry-after'), now)
  if (retryMs != null) return { resetAt: now + retryMs, retryAfterMs: retryMs, source: 'retry-after' }

  for (const key of ['x-ratelimit-reset', 'x-rate-limit-reset', 'ratelimit-reset']) {
    const raw = getHeader(headers, key)
    if (!raw) continue
    const numeric = Number(raw)
    if (Number.isFinite(numeric) && numeric > 0) {
      // Standard RateLimit-Reset is a delay in seconds; X-RateLimit-Reset is
      // conventionally an absolute Unix timestamp. Treating a small standard
      // value like `30` as 1970 made it look expired and discarded the server's
      // cooldown signal entirely.
      if (key === 'ratelimit-reset') {
        const delayMs = Math.round(numeric * 1000)
        return { resetAt: now + delayMs, retryAfterMs: delayMs, source: key }
      }
      const resetAt = numeric > 10_000_000_000 ? numeric : numeric * 1000
      if (resetAt > now) return { resetAt, source: key }
    }
    const parsed = Date.parse(raw)
    if (Number.isFinite(parsed) && parsed > now) return { resetAt: parsed, source: key }
  }

  return { resetAt: null }
}

function errorStatus(error: unknown): number | undefined {
  const record = typeof error === 'object' && error ? error as Record<string, unknown> : {}
  const status = record.status ?? record.statusCode ?? record.code
  if (typeof status === 'number') return status
  if (typeof status === 'string' && /^\d+$/.test(status)) return Number(status)
  const message = String(record.message ?? error ?? '')
  const http = message.match(/\bHTTP\s+(\d{3})\b/i)
  return http ? Number(http[1]) : undefined
}

function errorHeaders(error: unknown): unknown {
  const record = typeof error === 'object' && error ? error as Record<string, unknown> : {}
  return record.headers ?? (record.response && typeof record.response === 'object' ? (record.response as Record<string, unknown>).headers : undefined)
}

function errorMessage(error: unknown): string {
  if (error instanceof Error) return error.message
  if (typeof error === 'string') return error
  if (typeof error === 'object' && error) {
    const record = error as Record<string, unknown>
    return String(record.message ?? record.error ?? record.type ?? error)
  }
  return String(error)
}

function isContextOrPayloadLimit(message: string): boolean {
  return /\b(context_length_exceeded|context window|maximum context|context length|token limit|max_tokens|input too large|payload too large|request too large)\b/i.test(message)
}

export function classifyModelExhaustion(error: unknown, now = Date.now(), defaultCooldownMs?: number): ModelExhaustionSignal | null {
  const status = errorStatus(error)
  const message = errorMessage(error)
  const lower = message.toLowerCase()
  if (isContextOrPayloadLimit(lower)) return null

  const headers = errorHeaders(error)
  const reset = resetAtFromHeaders(headers, now)
  const isRate = status === 429 || /\b(rate[\s_-]?limit|too many requests|throttl(?:e|ed|ing)|requests per (?:minute|day)|rpm|tpm)\b/i.test(lower)
  const isQuota = /\b(insufficient_quota|quota|billing hard limit|usage limit|credit balance|credits exhausted|monthly limit|daily limit|limit exhausted)\b/i.test(lower)
  if (!isRate && !isQuota) return null

  const kind: ModelExhaustionKind = isQuota ? 'quota_exhausted' : 'rate_limit'
  const cooldown = defaultCooldownMs ?? (kind === 'quota_exhausted' ? DEFAULT_QUOTA_COOLDOWN_MS : DEFAULT_RATE_LIMIT_COOLDOWN_MS)
  return {
    kind,
    status,
    resetAt: reset.resetAt ?? now + cooldown,
    retryAfterMs: reset.retryAfterMs,
    message: message.slice(0, 500),
    source: reset.source ?? (status ? `http-${status}` : 'message'),
  }
}

export function selectModelRoute(input: {
  provider: Provider
  model: string
  requestedAccountId?: string | null
  state?: ModelRouteState
  allowFailover?: boolean
  localOnly?: boolean
  now?: number
}): ModelRouteDecision {
  const now = input.now ?? Date.now()
  const requestedAccountId = normalizedPreferredAccountId(input.provider, input.requestedAccountId)
  if (input.localOnly && input.provider.kind !== 'local') {
    const audit: ModelRouteAudit = {
      action: 'no_route',
      providerId: input.provider.id,
      model: input.model,
      requestedAccountId,
      selectedAccountId: null,
      exhaustedAccountIds: [],
      eligibleAccountIds: [],
      failoverChainAccountIds: [],
      resetEligibleAccountIds: [],
      reason: `Project is local-only, so ${input.provider.name} cannot serve ${input.model}.`,
    }
    return {
      providerId: input.provider.id,
      model: input.model,
      requestedAccountId,
      selectedAccountId: null,
      selected: null,
      candidates: [],
      failoverChain: [],
      audit,
    }
  }
  const rawCandidates = allCandidates(input.provider, input.model, requestedAccountId)
  const resetEligibleAccountIds = rawCandidates
    .filter((candidate) => (input.state?.exhausted ?? []).some((entry) =>
      entry.providerId === input.provider.id &&
      entry.model === input.model &&
      entry.accountId === candidate.accountId &&
      entry.resetAt <= now
    ))
    .map((candidate) => candidate.accountId)

  const candidates = rawCandidates.map((candidate) => {
    const exhausted = activeExhaustion(input.state, input.provider.id, input.model, candidate.accountId, now)
    return { ...candidate, exhaustedUntil: exhausted?.resetAt ?? null }
  })
  const allowFailover = input.allowFailover !== false
  const eligible = candidates.filter((candidate) => candidate.exhaustedUntil == null && (allowFailover || candidate.accountId === requestedAccountId))
  const selected = eligible[0] ?? null
  const failoverChain = eligible
  const exhaustedAccountIds = candidates.filter((candidate) => candidate.exhaustedUntil != null).map((candidate) => candidate.accountId)
  const eligibleAccountIds = eligible.map((candidate) => candidate.accountId)
  const failoverChainAccountIds = failoverChain.map((candidate) => candidate.accountId)
  const failover = selected && selected.accountId !== requestedAccountId

  return {
    providerId: input.provider.id,
    model: input.model,
    requestedAccountId,
    selectedAccountId: selected?.accountId ?? null,
    selected,
    candidates,
    failoverChain,
    audit: {
      action: selected ? (failover ? 'failover' : 'select_route') : 'no_route',
      providerId: input.provider.id,
      model: input.model,
      requestedAccountId,
      selectedAccountId: selected?.accountId ?? null,
      exhaustedAccountIds,
      eligibleAccountIds,
      failoverChainAccountIds,
      resetEligibleAccountIds,
      reason: selected
        ? failover
          ? `Selected ${accountLabel(selected.accountId)} because ${accountLabel(requestedAccountId)} is exhausted or unavailable for ${input.model}.`
          : `Selected requested account ${accountLabel(selected.accountId)} for ${input.model}.`
        : allowFailover
          ? `No enabled account for ${input.provider.id} can currently serve ${input.model}.`
          : `Automatic failover is disabled for ${accountLabel(requestedAccountId)} and it cannot currently serve ${input.model}.`,
    },
  }
}

export function recordModelExhaustion(state: ModelRouteState | undefined, input: RecordModelExhaustionInput): RecordModelExhaustionResult {
  const now = input.now ?? Date.now()
  const requestedAccountId = normalizedPreferredAccountId(input.provider, input.accountId)
  const signal = classifyModelExhaustion(input.error, now, input.defaultCooldownMs)
  const baseAudit: Omit<ModelRouteAudit, 'action' | 'reason'> = {
    providerId: input.provider.id,
    model: input.model,
    requestedAccountId,
    selectedAccountId: requestedAccountId,
    exhaustedAccountIds: [],
    eligibleAccountIds: [],
    failoverChainAccountIds: [],
    resetEligibleAccountIds: [],
  }

  if (!signal) {
    return {
      state: pruneExpired(state, now),
      signal: null,
      audit: { ...baseAudit, action: 'classify_error', reason: 'Error is not provider account exhaustion.' },
    }
  }

  const pruned = pruneExpired(state, now)
  const nextEntry: ModelRouteExhaustion = {
    providerId: input.provider.id,
    accountId: requestedAccountId,
    model: input.model,
    kind: signal.kind,
    observedAt: now,
    resetAt: signal.resetAt,
    message: signal.message,
    source: signal.source,
  }
  const exhausted = [
    ...pruned.exhausted.filter((entry) =>
      !(entry.providerId === nextEntry.providerId && entry.accountId === nextEntry.accountId && entry.model === nextEntry.model)
    ),
    nextEntry,
  ]

  return {
    state: { exhausted },
    signal,
    audit: {
      ...baseAudit,
      action: 'record_exhaustion',
      exhaustedAccountIds: [requestedAccountId],
      reason: `${signal.kind} recorded for ${accountLabel(requestedAccountId)} until ${new Date(signal.resetAt).toISOString()}.`,
      resetAt: signal.resetAt,
      source: signal.source,
    },
  }
}

export function routeAfterModelError(state: ModelRouteState | undefined, input: RecordModelExhaustionInput): {
  state: ModelRouteState
  signal: ModelExhaustionSignal | null
  route: ModelRouteDecision
  audit: ModelRouteAudit[]
} {
  const recorded = recordModelExhaustion(state, input)
  const route = selectModelRoute({
    provider: input.provider,
    model: input.model,
    requestedAccountId: input.accountId,
    state: recorded.state,
    allowFailover: input.allowFailover ?? input.provider.accounts?.find((account) => account.id === normalizedPreferredAccountId(input.provider, input.accountId))?.autoFailover === true,
    localOnly: input.localOnly,
    now: input.now,
  })
  return { state: recorded.state, signal: recorded.signal, route, audit: [recorded.audit, route.audit] }
}

export class ModelRouter {
  constructor(private readonly store: ModelRouterStore) {}

  state(): ModelRouteState {
    return this.store.getState()
  }

  select(input: ModelRouterRouteInput): ModelRouteDecision {
    return this.route(input).decision
  }

  route(input: ModelRouterRouteInput): ModelRouterRouteOutput {
    const decision = selectModelRoute({
      provider: input.provider,
      model: input.model,
      requestedAccountId: input.requestedAccountId,
      state: input.state ?? this.store.getState(),
      allowFailover: input.allowFailover,
      localOnly: input.localOnly,
      now: input.now,
    })
    return this.output(input, this.applySpendControls(input, decision))
  }

  recommendRoleRoute(input: ModelRouterRoleRouteInput): ModelRouterRoleRouteResult {
    const scorecards = input.scorecards ?? []
    const rankings: ModelRouterRoleRanking[] = []
    const providers = input.providers
      .filter((provider) => !input.requestedProviderId || provider.id === input.requestedProviderId)
      .filter((provider) => !(input.localOnly && !isLocalProvider(provider)))
    for (const provider of providers) {
      const modelSet = new Set<string>()
      for (const model of provider.models) modelSet.add(model)
      for (const account of enabledAccounts(provider)) {
        for (const model of account.models ?? []) modelSet.add(model)
      }
      if (input.requestedModel) modelSet.add(input.requestedModel)
      for (const model of modelSet) {
        if (input.requestedModel && model !== input.requestedModel) continue
        const route = this.route({
          role: scoreRoleToRouteRole(input.role),
          provider,
          model,
          requestedAccountId: input.requestedAccountId,
          allowFailover: input.allowFailover,
          localOnly: input.localOnly,
          now: input.now,
          toolNeeds: input.toolNeeds,
          contextPressure: input.contextPressure,
          budget: input.budget,
          providerHealth: input.providerHealth,
          accountQuota: input.accountQuota,
          userPreference: input.userPreference,
          contextLimit: input.contextLimit,
          reasoningEffort: input.reasoningEffort,
          cacheStrategy: input.cacheStrategy,
          source: input.source,
        })
        const selected = route.decision.selected
        if (!selected) continue
        const card = scorecardForCandidate(scorecards, provider.id, selected.accountId, model)
        const cardScore = clampScore(card?.scores[input.role])
        const heuristicScore = roleHeuristicScore(input.role, provider, model)
        const baseScore = cardScore == null ? heuristicScore : (cardScore * 0.72) + (heuristicScore * 0.28)
        const preferenceBoost =
          (input.requestedProviderId && provider.id === input.requestedProviderId ? 0.04 : 0) +
          (input.requestedAccountId !== undefined && selected.accountId === (input.requestedAccountId ?? null) ? 0.03 : 0) +
          (input.requestedModel && model === input.requestedModel ? 0.03 : 0)
        const score = Math.max(0, Math.min(1, baseScore + scorecardAdjustment(card) + preferenceBoost))
        rankings.push({
          providerId: provider.id,
          providerName: provider.name,
          accountId: selected.accountId,
          accountName: selected.accountName,
          model,
          score: Math.round(score * 1000) / 1000,
          scorecardScore: cardScore,
          heuristicScore: Math.round(heuristicScore * 1000) / 1000,
          reason: cardScore == null
            ? `${input.role} route ranked by built-in role heuristic.`
            : `${input.role} route ranked from scorecard ${cardScore.toFixed(2)} plus role heuristic.`,
          route,
        })
      }
    }
    rankings.sort((a, b) =>
      b.score - a.score ||
      a.providerName.localeCompare(b.providerName) ||
      a.model.localeCompare(b.model) ||
      accountLabel(a.accountId).localeCompare(accountLabel(b.accountId))
    )
    return { selected: rankings[0] ?? null, rankings }
  }

  routeAfterError(input: ModelRouterErrorInput): ModelRouterErrorResult {
    const routed = routeAfterModelError(this.store.getState(), input)
    this.store.setState(routed.state)
    const routeInput: ModelRouterRouteInput = {
      role: input.role,
      provider: input.provider,
      model: input.model,
      requestedAccountId: input.accountId,
      allowFailover: input.allowFailover,
      localOnly: input.localOnly,
      now: input.now,
      toolNeeds: input.toolNeeds,
      contextPressure: input.contextPressure,
      budget: input.budget,
      providerHealth: input.providerHealth,
      accountQuota: input.accountQuota,
      scorecard: input.scorecard,
      latencyMs: input.latencyMs,
      costEstimate: input.costEstimate,
      userPreference: input.userPreference,
      contextLimit: input.contextLimit,
      reasoningEffort: input.reasoningEffort,
      cacheStrategy: input.cacheStrategy,
      source: input.source,
    }
    const route = this.applySpendControls(routeInput, routed.route)
    return {
      ...routed,
      route,
      audit: [routed.audit[0], route.audit],
      output: this.output(routeInput, route),
    }
  }

  private output(input: ModelRouterRouteInput, decision: ModelRouteDecision): ModelRouterRouteOutput {
    const contextLimit = contextLimitForRoute(input)
    const spend = decision.selected
      ? spendStatusForCandidate(input.provider, decision.selected.accountId, typeof input.costEstimate === 'number' ? input.costEstimate : null)
      : spendStatusForCandidate(input.provider, decision.selectedAccountId, typeof input.costEstimate === 'number' ? input.costEstimate : null)
    return {
      role: input.role,
      providerId: decision.providerId,
      accountId: decision.selectedAccountId,
      model: decision.model,
      fallbackOrder: decision.failoverChain.map((candidate) => candidate.accountId),
      contextLimit,
      reasoningEffort: input.reasoningEffort ?? defaultReasoningEffort(input.role),
      cacheStrategy: defaultCacheStrategy(input, contextLimit),
      spend,
      decision,
      inputs: {
        toolNeeds: input.toolNeeds ?? [],
        contextPressure: input.contextPressure ?? null,
        budget: input.budget ?? null,
        providerHealth: input.providerHealth ?? null,
        accountQuota: input.accountQuota ?? null,
        scorecard: input.scorecard ?? null,
        latencyMs: input.latencyMs ?? null,
        costEstimate: input.costEstimate ?? null,
        userPreference: input.userPreference ?? null,
        contextLimit: input.contextLimit ?? null,
        source: input.source ?? null,
      },
    }
  }

  private applySpendControls(input: ModelRouterRouteInput, decision: ModelRouteDecision): ModelRouteDecision {
    if (!decision.selected) return decision
    const projected = typeof input.costEstimate === 'number' && Number.isFinite(input.costEstimate) && input.costEstimate > 0 ? input.costEstimate : null
    const unblocked = decision.failoverChain.filter((candidate) => !spendStatusForCandidate(input.provider, candidate.accountId, projected).blocked)
    const selected = unblocked[0] ?? null
    if (selected && selected.accountId === decision.selected.accountId) return decision
    const failoverChainAccountIds = unblocked.map((candidate) => candidate.accountId)
    if (!selected) {
      return {
        ...decision,
        selected: null,
        selectedAccountId: null,
        failoverChain: [],
        audit: {
          ...decision.audit,
          action: 'no_route',
          selectedAccountId: null,
          eligibleAccountIds: [],
          failoverChainAccountIds: [],
          reason: `No eligible account for ${decision.providerId} can serve ${decision.model} without exceeding a hard spend budget.`,
        },
      }
    }
    const failover = selected.accountId !== decision.requestedAccountId
    return {
      ...decision,
      selected,
      selectedAccountId: selected.accountId,
      failoverChain: unblocked,
      audit: {
        ...decision.audit,
        action: failover ? 'failover' : 'select_route',
        selectedAccountId: selected.accountId,
        eligibleAccountIds: failoverChainAccountIds,
        failoverChainAccountIds,
        reason: failover
          ? `Selected ${accountLabel(selected.accountId)} because ${accountLabel(decision.requestedAccountId)} is unavailable or over budget for ${decision.model}.`
          : `Selected requested account ${accountLabel(selected.accountId)} for ${decision.model}.`,
      },
    }
  }
}
