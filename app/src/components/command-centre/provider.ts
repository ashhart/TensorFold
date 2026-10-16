import { formatDuration, relativeTime } from '../../lib/format'
import { formatUsageTokens } from '../../lib/provider-usage'
import { accountLabel, modelDisplayName, resolveProviderModel } from '../../lib/provider-accounts'
import type { ContextPressureSnapshot, ModelRouterSnapshot, Provider, TokenUsageSnapshot } from '../../lib/types'
import { MODEL_PRICE_MAP, RUNNING_ROOM_STATES, RUNNING_THREAD_STATES, type ProviderSignal, type ProviderVisibilityDraft, type ProviderVisibilityInput, type ProviderVisibilityRow, type Tone } from './model'
import { missionIsActive } from './attention'

export function buildProviderVisibilityRows(input: ProviderVisibilityInput): ProviderVisibilityRow[] {
  const { settings, threads, threadStatus, rooms, roomStatus, automations, missions, contextSnapshot, modelRouterSnapshot, usageSnapshot } = input
  if (!settings) return []

  const drafts = new Map<string, ProviderVisibilityDraft>()
  const addRoute = (route: {
    providerId: string | null
    providerAccountId?: string | null
    model?: string | null
    label: string
    active?: boolean
    updatedAt?: number
  }) => {
    if (!route.providerId) return
    const provider = settings.providers.find((candidate) => candidate.id === route.providerId)
    if (!provider) return
    const accountId = route.providerAccountId ?? provider.defaultAccountId ?? null
    const model = resolveProviderModel(provider, accountId, route.model, provider.models[0]) ?? route.model ?? provider.models[0] ?? null
    const key = providerVisibilityKey(provider.id, accountId, model)
    const existing = drafts.get(key)
    if (existing) {
      existing.labels.add(route.label)
      existing.latestAt = Math.max(existing.latestAt, route.updatedAt ?? 0)
      existing.activeCount += route.active ? 1 : 0
      existing.isDefault = existing.isDefault || route.label === 'Default'
      return
    }
    drafts.set(key, {
      provider,
      accountId,
      model,
      labels: new Set([route.label]),
      activeCount: route.active ? 1 : 0,
      latestAt: route.updatedAt ?? 0,
      isDefault: route.label === 'Default',
    })
  }

  addRoute({
    providerId: settings.activeProviderId,
    providerAccountId: settings.activeProviderAccountId,
    model: settings.activeModel,
    label: 'Default',
  })

  for (const thread of threads) {
    const state = threadStatus[thread.id] ?? thread.status
    if (!RUNNING_THREAD_STATES.has(state)) continue
    addRoute({
      providerId: thread.providerId,
      providerAccountId: thread.providerAccountId,
      model: thread.model,
      label: thread.kind === 'chat' ? 'Chat' : 'Task',
      active: true,
      updatedAt: thread.updatedAt,
    })
  }

  for (const room of rooms) {
    const state = roomStatus[room.id] ?? room.state
    if (!RUNNING_ROOM_STATES.has(state)) continue
    for (const seat of room.seats) {
      addRoute({
        providerId: seat.providerId,
        providerAccountId: seat.providerAccountId,
        model: seat.model,
        label: 'Room',
        active: true,
        updatedAt: room.updatedAt,
      })
    }
  }

  for (const automation of automations) {
    if (automation.runs[0]?.status !== 'running') continue
    addRoute({
      providerId: automation.providerId,
      providerAccountId: automation.providerAccountId,
      model: automation.model,
      label: 'Automation',
      active: true,
      updatedAt: automation.runs[0].startedAt,
    })
  }

  for (const mission of missions) {
    if (!missionIsActive(mission.status)) continue
    addRoute({
      providerId: mission.providerId,
      providerAccountId: mission.providerAccountId,
      model: mission.model,
      label: 'Mission',
      active: true,
      updatedAt: mission.startedAt ?? mission.updatedAt,
    })
  }

  return [...drafts.values()]
    .sort((a, b) => Number(b.isDefault) - Number(a.isDefault) || b.activeCount - a.activeCount || b.latestAt - a.latestAt || a.provider.name.localeCompare(b.provider.name))
    .slice(0, 5)
    .map((draft) => {
      const model = draft.model ? modelDisplayName(draft.provider, draft.model) : 'No model selected'
      const activeLabels = [...draft.labels].filter((label) => label !== 'Default')
      const activity = activeLabels.length
        ? `${draft.activeCount} active ${draft.activeCount === 1 ? 'route' : 'routes'}: ${activeLabels.slice(0, 3).join(', ')}`
        : 'Default for new runs'
      return {
        id: providerVisibilityKey(draft.provider.id, draft.accountId, draft.model),
        providerName: draft.provider.name,
        providerKind: providerKindLabel(draft.provider),
        label: draft.isDefault ? (draft.activeCount ? 'Default + active' : 'Default') : 'Active',
        model,
        account: providerAccountText(draft.provider, draft.accountId),
        activity,
        activeCount: draft.activeCount,
        health: providerHealthSignal(draft.provider, draft.model, draft.accountId),
        route: providerRouteSignal(draft.provider, draft.model, draft.accountId, modelRouterSnapshot),
        context: providerContextSignal(draft.provider, draft.model, contextSnapshot),
        latency: providerTimingSignal(draft.latestAt, draft.activeCount),
        spend: providerSpendSignal(draft.provider, draft.model, usageSnapshot),
      }
    })
}

export function providerVisibilityKey(providerId: string, accountId: string | null, model: string | null) {
  return `${providerId}:${accountId ?? 'default'}:${model ?? 'no-model'}`
}

export function providerKindLabel(provider: Provider) {
  if (provider.kind === 'local') return 'local'
  if (provider.transport === 'codex-cli') return 'codex'
  if (provider.transport === 'claude-cli') return 'claude'
  return provider.kind ?? 'cloud'
}

export function providerAccountText(provider: Provider, accountId: string | null) {
  const label = accountLabel(provider, accountId)
  if (label) return label
  if (accountId) return 'Unknown account'
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') return 'Default login'
  if ((provider.accounts ?? []).some((account) => account.enabled !== false)) return 'Provider default'
  return provider.kind === 'local' ? 'Default endpoint' : 'Provider key'
}

export function providerHealthSignal(provider: Provider, model: string | null, accountId: string | null): ProviderSignal {
  const account = accountId ? provider.accounts?.find((candidate) => candidate.id === accountId) : null
  if (account && account.enabled === false) return { label: 'Health', value: 'Disabled', detail: 'Account is disabled', tone: 'critical' }
  if (!model) return { label: 'Health', value: 'No model', detail: 'Model list is empty', tone: 'high' }
  if (provider.kind !== 'local' && provider.transport !== 'codex-cli' && provider.transport !== 'claude-cli') {
    const hasKey = !!(account?.apiKey || provider.apiKey)
    if (!hasKey) return { label: 'Health', value: 'Needs key', detail: 'No API key configured', tone: 'critical' }
  }
  if (provider.kind === 'local') {
    return { label: 'Health', value: provider.models.length ? 'Configured' : 'No models', detail: 'Endpoint not pinged', tone: provider.models.length ? 'normal' : 'high' }
  }
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') {
    return { label: 'Health', value: 'Configured', detail: 'CLI not probed here', tone: 'normal' }
  }
  return { label: 'Health', value: 'Configured', detail: 'Key present, no live probe', tone: 'normal' }
}

export function providerRouteSignal(provider: Provider, model: string | null, accountId: string | null, snapshot: ModelRouterSnapshot | null): ProviderSignal {
  if (!model) return { label: 'Failover', value: 'No model', detail: 'Choose a model first', tone: 'high' }
  const active = snapshot?.activeCooldowns.find((entry) =>
    entry.providerId === provider.id &&
    entry.accountId === accountId &&
    normalizeModel(entry.model) === normalizeModel(model)
  )
  if (active) {
    const remaining = formatDuration(Math.max(0, active.resetAt - Date.now()))
    return {
      label: 'Failover',
      value: active.kind === 'quota_exhausted' ? 'Quota' : 'Cooling',
      detail: `${active.accountName} resets in ${remaining}`,
      tone: active.kind === 'quota_exhausted' ? 'critical' : 'high',
    }
  }
  const account = accountId ? provider.accounts?.find((candidate) => candidate.id === accountId) : null
  if (provider.kind === 'local') return { label: 'Failover', value: 'Local', detail: 'Local route has no subscription failover', tone: 'good' }
  if (!account) return { label: 'Failover', value: 'Default', detail: 'Provider-default route only', tone: 'low' }
  const equivalent = (provider.accounts ?? [])
    .filter((candidate) => candidate.enabled !== false)
    .filter((candidate) => candidate.id !== account.id)
    .filter((candidate) => candidate.kind === account.kind)
    .filter((candidate) => accountSupportsModel(provider, candidate, model))
    .filter((candidate) => !snapshot?.activeCooldowns.some((entry) =>
      entry.providerId === provider.id &&
      entry.accountId === candidate.id &&
      normalizeModel(entry.model) === normalizeModel(model)
    ))
  if (account.autoFailover === true && equivalent.length > 0) {
    return {
      label: 'Failover',
      value: 'Armed',
      detail: `${equivalent.length} equivalent ${equivalent.length === 1 ? 'account' : 'accounts'} ready`,
      tone: 'good',
    }
  }
  if (equivalent.length > 0) {
    return { label: 'Failover', value: 'Manual', detail: 'Equivalent account exists; enable automatic failover', tone: 'high' }
  }
  return { label: 'Failover', value: 'Single', detail: 'No equivalent account for this model', tone: 'low' }
}

export function accountSupportsModel(provider: Provider, account: NonNullable<Provider['accounts']>[number], model: string) {
  const models = account.models?.length ? account.models : provider.models
  return models.length === 0 || models.some((candidate) => normalizeModel(candidate) === normalizeModel(model))
}

export function providerContextSignal(provider: Provider, model: string | null, snapshot: ContextPressureSnapshot | null): ProviderSignal {
  if (!snapshot) return { label: 'Context', value: 'Unavailable', detail: 'Snapshot not loaded', tone: 'low' }
  const matches = snapshot.threads.filter((thread) => (
    thread.providerId === provider.id
    && (!model || normalizeModel(thread.model) === normalizeModel(model))
  ))
  const primary = matches[0]
  if (primary) {
    return {
      label: 'Context',
      value: formatPercent(primary.pressureRatio),
      detail: `${primary.status} / ${formatTokens(primary.estimatedTokens)} of ${formatTokens(primary.workingLimitTokens)}`,
      tone: pressureTone(primary.status),
    }
  }
  const profile = snapshot.registry.find((candidate) => candidate.providerId === provider.id && (!model || normalizeModel(candidate.model) === normalizeModel(model)))
    ?? (snapshot.activeProfile?.providerId === provider.id ? snapshot.activeProfile : null)
  if (profile) return { label: 'Context', value: 'No runs', detail: `${formatTokens(profile.recommendedWorkingTokens)} working window`, tone: 'low' }
  return { label: 'Context', value: 'No runs', detail: provider.kind === 'local' ? 'No local pressure' : 'No thread pressure', tone: 'low' }
}

export function providerTimingSignal(latestAt: number, activeCount: number): ProviderSignal {
  if (activeCount > 0 && latestAt > 0) {
    return {
      label: 'Timing',
      value: formatDuration(Math.max(0, Date.now() - latestAt)),
      detail: `${activeCount} active ${activeCount === 1 ? 'route' : 'routes'}`,
      tone: 'normal',
    }
  }
  if (latestAt > 0) {
    return {
      label: 'Timing',
      value: relativeTime(latestAt),
      detail: 'Last route activity',
      tone: 'low',
    }
  }
  return { label: 'Timing', value: 'No runs', detail: 'No route timing yet', tone: 'low' }
}

export function providerSpendSignal(provider: Provider, model: string | null, snapshot: TokenUsageSnapshot | null): ProviderSignal {
  if (!snapshot) return { label: 'Spend', value: 'No usage', detail: 'Usage snapshot unavailable', tone: 'low' }
  const today = snapshot.windows.today
  const modelUsage = model ? today.models.find((candidate) => normalizeModel(candidate.model) === normalizeModel(model)) : null
  const tokens = modelUsage?.total ?? 0
  if (provider.kind === 'local') {
    return { label: 'Spend', value: 'Local', detail: tokens ? `${formatTokens(tokens)} tokens today` : 'No tokens today', tone: 'good' }
  }
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') {
    return { label: 'Spend', value: 'Subscription', detail: tokens ? `${formatTokens(tokens)} tokens today` : 'No tokens today', tone: tokens ? 'normal' : 'low' }
  }
  const price = model ? priceForModel(model) : null
  if (modelUsage && price) {
    const estimate = estimateSpendUsd(modelUsage, price)
    return {
      label: 'Spend',
      value: formatUsd(estimate),
      detail: `${formatTokens(tokens)} tokens today, estimate`,
      tone: estimate > 10 ? 'high' : estimate > 0 ? 'normal' : 'low',
    }
  }
  return {
    label: 'Spend',
    value: tokens ? 'Needs map' : 'No tokens',
    detail: tokens ? `${formatTokens(tokens)} tokens today, no price map` : 'No tokens today',
    tone: 'low',
  }
}

export function priceForModel(model: string) {
  const normalized = normalizeModel(model)
  return MODEL_PRICE_MAP.find((entry) => entry.match.test(normalized)) ?? null
}

export function estimateSpendUsd(usage: TokenUsageSnapshot['windows']['today']['models'][number], price: (typeof MODEL_PRICE_MAP)[number]) {
  const inputCost = usage.input * price.inputPerMillion
  const outputCost = usage.output * price.outputPerMillion
  const cacheWriteCost = usage.cacheCreation * (price.cacheWritePerMillion ?? price.inputPerMillion)
  const cacheReadCost = (usage.cacheRead + usage.cached) * (price.cacheReadPerMillion ?? price.inputPerMillion)
  const reasoningCost = usage.reasoning * (price.reasoningPerMillion ?? price.outputPerMillion)
  return (inputCost + outputCost + cacheWriteCost + cacheReadCost + reasoningCost) / 1_000_000
}

export function formatUsd(value: number) {
  if (!Number.isFinite(value) || value <= 0) return '$0.00'
  if (value < 0.01) return '<$0.01'
  return `$${value.toFixed(value >= 10 ? 0 : 2)}`
}

export function normalizeModel(model: string | null | undefined) {
  return (model ?? '').trim().toLowerCase()
}

export function pressureTone(status: ContextPressureSnapshot['threads'][number]['status']): Tone {
  if (status === 'critical') return 'critical'
  if (status === 'pressured') return 'high'
  if (status === 'healthy') return 'good'
  return status === 'idle' ? 'low' : 'normal'
}

export function formatPercent(value: number) {
  return `${Math.round(value * 100)}%`
}

export const formatTokens = formatUsageTokens
