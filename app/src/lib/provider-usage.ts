import type { ModelRouterSnapshot, Provider, TokenUsageAccount, TokenUsageModel, TokenUsageSnapshot } from './types'

type UsageAccountId = TokenUsageSnapshot['windows']['today']['accounts'][number]['id']
type Tone = 'good' | 'normal' | 'warn' | 'muted'

export interface ProviderUsageSummary {
  usage: string
  usageDetail: string
  budget: string
  budgetDetail: string
  route: string
  routeDetail: string
  routeTone: Tone
}

// Short token counts: 950, 1.2K, 340K, 3.4M, 12M.
const TOKEN_UNITS = [{ size: 1_000_000, suffix: 'M', wholeFrom: 10 }, { size: 1_000, suffix: 'K', wholeFrom: 100 }] as const

export function formatUsageTokens(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '0'
  const unit = TOKEN_UNITS.find((u) => value >= u.size)
  if (!unit) return String(Math.round(value))
  const scaled = value / unit.size
  return `${scaled.toFixed(scaled >= unit.wholeFrom ? 0 : 1)}${unit.suffix}`
}

export function normalizeUsageModel(model: string | null | undefined) {
  return (model ?? '').trim().toLowerCase()
}

export function usageAccountIdForProvider(provider: Provider | null | undefined): UsageAccountId | null {
  if (!provider) return null
  if (provider.transport === 'codex-cli') return 'codex'
  if (provider.transport === 'claude-cli') return 'claude'
  const hint = `${provider.id} ${provider.name} ${provider.source ?? ''}`.toLowerCase()
  if (hint.includes('opencode')) return 'opencode'
  if (hint.includes('codex')) return 'codex'
  if (hint.includes('claude')) return 'claude'
  return null
}

export function providerUsageAccount(provider: Provider | null | undefined, snapshot: TokenUsageSnapshot | null | undefined): TokenUsageAccount | null {
  const id = usageAccountIdForProvider(provider)
  if (!id || !snapshot) return null
  return snapshot.windows.today.accounts.find((account) => account.id === id) ?? null
}

export function usageForModel(model: string | null | undefined, snapshot: TokenUsageSnapshot | null | undefined): TokenUsageModel | null {
  if (!model || !snapshot) return null
  const normalized = normalizeUsageModel(model)
  return snapshot.windows.today.models.find((candidate) => normalizeUsageModel(candidate.model) === normalized) ?? null
}

export function modelUsageLabel(model: string | null | undefined, snapshot: TokenUsageSnapshot | null | undefined): string | null {
  const usage = usageForModel(model, snapshot)
  return usage?.total ? `${formatUsageTokens(usage.total)} today` : null
}

function money(value: number | null | undefined) {
  const n = typeof value === 'number' ? value : NaN
  if (!Number.isFinite(n) || n <= 0) return '$0'
  return `$${n.toFixed(n >= 10 ? 0 : 2)}`
}

function budgetFor(provider: Provider, accountId: string | null | undefined) {
  const account = accountId ? provider.accounts?.find((candidate) => candidate.id === accountId) ?? null : null
  return account?.spendBudget ?? provider.spendBudget ?? null
}

function budgetSummary(provider: Provider, accountId: string | null | undefined) {
  const budget = budgetFor(provider, accountId)
  if (!budget || budget.enabled === false) return { label: 'No budget', detail: 'No spend guard configured' }
  const spent = budget.spentUsd ?? 0
  const limit = budget.limitUsd ?? null
  const period = budget.period ?? 'run'
  if (limit != null) {
    const pct = limit > 0 ? Math.min(999, Math.round((spent / limit) * 100)) : 0
    return { label: `${money(spent)} / ${money(limit)}`, detail: `${pct}% of ${period} budget${budget.hardStop ? ' · hard stop' : ''}` }
  }
  return { label: 'Budget on', detail: `${money(spent)} tracked this ${period}` }
}

function routeSummary(provider: Provider, accountId: string | null | undefined, model: string | null | undefined, snapshot: ModelRouterSnapshot | null | undefined) {
  const cooldown = snapshot?.activeCooldowns.find((entry) => (
    entry.providerId === provider.id &&
    (accountId == null || entry.accountId === accountId) &&
    (!model || normalizeUsageModel(entry.model) === normalizeUsageModel(model))
  ))
  if (!cooldown) return { label: 'Ready', detail: 'No active failover cooldown', tone: 'good' as Tone }
  const reset = cooldown.resetAt > Date.now() ? `resets ${relativeUsageTime(cooldown.resetAt)}` : 'reset due'
  return { label: cooldown.kind === 'quota_exhausted' ? 'Quota' : 'Cooling', detail: `${cooldown.accountName} · ${reset}`, tone: 'warn' as Tone }
}

export function relativeUsageTime(ts: number, now = Date.now()) {
  const delta = ts - now
  const abs = Math.abs(delta)
  const suffix = delta >= 0 ? 'from now' : 'ago'
  const minute = 60_000
  const hour = 60 * minute
  const day = 24 * hour
  if (abs < minute) return delta >= 0 ? 'soon' : 'just now'
  if (abs < hour) return `${Math.round(abs / minute)}m ${suffix}`
  if (abs < day) return `${Math.round(abs / hour)}h ${suffix}`
  return `${Math.round(abs / day)}d ${suffix}`
}

export function providerUsageSummary(provider: Provider, accountId: string | null | undefined, model: string | null | undefined, usageSnapshot: TokenUsageSnapshot | null | undefined, routerSnapshot: ModelRouterSnapshot | null | undefined): ProviderUsageSummary {
  const account = providerUsageAccount(provider, usageSnapshot)
  const modelUsage = usageForModel(model, usageSnapshot)
  const tokens = modelUsage?.total ?? account?.total ?? 0
  const usage = tokens ? `${formatUsageTokens(tokens)} today` : provider.kind === 'local' ? 'Local' : 'No usage'
  const usageDetail = modelUsage
    ? `${formatUsageTokens(modelUsage.input)} in · ${formatUsageTokens(modelUsage.output)} out`
    : account
      ? `${account.name} account usage`
      : 'No local telemetry for this provider yet'
  const budget = budgetSummary(provider, accountId)
  const route = routeSummary(provider, accountId, model, routerSnapshot)
  return {
    usage,
    usageDetail,
    budget: budget.label,
    budgetDetail: budget.detail,
    route: route.label,
    routeDetail: route.detail,
    routeTone: route.tone,
  }
}
