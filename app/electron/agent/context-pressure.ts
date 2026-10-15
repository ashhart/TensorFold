import { estimateTokens, planCompaction } from './compaction'
import { getModelContextWindow, recommendedCompactionThreshold } from './model-context-windows'
import { engineContextWindow } from './engine/context'
import type {
  ApiMessage,
  ContextPressureAction,
  ContextPressureGuidance,
  ContextPressureSnapshot,
  ContextPressureStatus,
  ContextPressureThread,
  ModelContextProfile,
  Provider,
  Settings,
  Thread,
} from './types'

const KEEP_RECENT = 6
const DEFAULT_CLOUD_CONTEXT = 128_000
const DEFAULT_LOCAL_CONTEXT = 24_000

interface KnownLocalModel {
  match: RegExp
  nominalContextTokens: number
  recommendedWorkingTokens: number
  kvCacheTokens: number
  note: string
}

const KNOWN_LOCAL_MODELS: KnownLocalModel[] = [
  {
    match: /qwen[-_. :]?3\.?6|qwen3\.?6/i,
    nominalContextTokens: 205_000,
    recommendedWorkingTokens: 174_250,
    kvCacheTokens: 40_000,
    note: 'Qwen 3.6 long-context local profile; keep tool output summarized before sustained coding loops.',
  },
  {
    match: /qwen2\.5[-_: ]?coder|qwen2\.5/i,
    nominalContextTokens: 32_768,
    recommendedWorkingTokens: 24_000,
    kvCacheTokens: 6_000,
    note: 'Qwen-style local coding model; keep long tool output in summaries before the working set gets hot.',
  },
  {
    match: /llama[-_: ]?3\.1|llama3\.1/i,
    nominalContextTokens: 128_000,
    recommendedWorkingTokens: 32_000,
    kvCacheTokens: 8_000,
    note: 'Long nominal window, but local KV memory pressure usually appears well before the full context.',
  },
  {
    match: /deepseek[-_: ]?(r1|coder)/i,
    nominalContextTokens: 32_768,
    recommendedWorkingTokens: 24_000,
    kvCacheTokens: 6_000,
    note: 'Reasoning/coding local model; keep the prompt prefix stable and fold older tool logs early.',
  },
  {
    match: /mistral[-_: ]?nemo|mistral/i,
    nominalContextTokens: 128_000,
    recommendedWorkingTokens: 32_000,
    kvCacheTokens: 8_000,
    note: 'Long-context local family; prefer a narrow working set for repeat agent turns.',
  },
  {
    match: /gemma/i,
    nominalContextTokens: 8_192,
    recommendedWorkingTokens: 6_000,
    kvCacheTokens: 2_000,
    note: 'Small local context profile; compact before multi-file tool output accumulates.',
  },
  {
    match: /phi[-_: ]?4|phi4/i,
    nominalContextTokens: 16_000,
    recommendedWorkingTokens: 12_000,
    kvCacheTokens: 3_000,
    note: 'Mid-sized local context profile; preserve the goal and latest evidence verbatim.',
  },
]

export function buildContextPressureSnapshot(settings: Settings, threads: Thread[], now = Date.now()): ContextPressureSnapshot {
  const registry = buildModelContextRegistry(settings)
  const activeProvider = settings.providers.find((p) => p.id === settings.activeProviderId) ?? null
  const activeProfile = activeProvider && isLocalProvider(activeProvider)
    ? modelContextProfile(settings, activeProvider, settings.activeModel || activeProvider.models[0] || 'local-model')
    : null

  const rows = threads
    .map((thread) => threadPressure(settings, thread))
    .filter((row) => row.kind === 'local' || row.estimatedTokens > 0)
    .sort((a, b) => b.pressureRatio - a.pressureRatio || b.updatedAt - a.updatedAt)
    .slice(0, 16)

  const localRows = rows.filter((row) => row.kind === 'local')
  return {
    generatedAt: now,
    activeProviderId: settings.activeProviderId,
    activeModel: settings.activeModel,
    activeProfile,
    registry,
    threads: rows,
    totals: {
      localThreads: localRows.length,
      estimatedTokens: sum(localRows, (row) => row.estimatedTokens),
      pinnedTokens: sum(localRows, (row) => row.guidance.pinnedTokens + row.guidance.recentTokens),
      summarizableTokens: sum(localRows, (row) => row.guidance.summarizableTokens),
      maxPressureRatio: localRows.length ? Math.max(...localRows.map((row) => row.pressureRatio)) : 0,
    },
  }
}

export function buildModelContextRegistry(settings: Settings): ModelContextProfile[] {
  const out: ModelContextProfile[] = []
  const seen = new Set<string>()
  for (const provider of settings.providers) {
    if (!isLocalProvider(provider)) continue
    const models = new Set(provider.models.length ? provider.models : ['local-model'])
    if (provider.id === settings.activeProviderId && settings.activeModel) models.add(settings.activeModel)
    for (const model of models) {
      const key = `${provider.id}:${model}`
      if (seen.has(key)) continue
      seen.add(key)
      out.push(modelContextProfile(settings, provider, model))
    }
  }
  return out.sort((a, b) => a.providerName.localeCompare(b.providerName) || a.model.localeCompare(b.model))
}

function threadPressure(settings: Settings, thread: Thread): ContextPressureThread {
  const provider = providerForThread(settings, thread)
  const model = thread.model || (provider.id === settings.activeProviderId ? settings.activeModel : null) || provider.models[0] || null
  const kind: 'local' | 'cloud' = isLocalProvider(provider) ? 'local' : 'cloud'
  const profile = kind === 'local'
    ? modelContextProfile(settings, provider, model || 'local-model')
    : cloudContextProfile(settings, provider, model)
  const estimatedTokens = estimateTokens(thread.apiMessages)
  const guidance = buildGuidance(thread)
  const workingLimit = Math.max(1, profile.recommendedWorkingTokens)
  const pressureRatio = roundRatio(estimatedTokens / workingLimit)

  return {
    id: thread.id,
    title: thread.title,
    cwd: thread.cwd,
    providerId: thread.providerId,
    providerName: provider.name,
    model,
    kind,
    estimatedTokens,
    contextWindowTokens: profile.nominalContextTokens,
    workingLimitTokens: workingLimit,
    pressureRatio,
    status: statusFor(estimatedTokens, pressureRatio),
    guidance: {
      ...guidance,
      ...recommendAction(pressureRatio, workingLimit, guidance),
    },
    updatedAt: thread.updatedAt,
  }
}

function buildGuidance(thread: Thread): Omit<ContextPressureGuidance, 'recommendedAction' | 'reason' | 'cacheHint'> {
  const messages = thread.apiMessages
  const plan = planCompaction(messages, KEEP_RECENT)
  const sysCount = leadingSystemMessages(messages)
  const cut = plan?.cut ?? sysCount
  const pinnedMessages = messages.slice(0, sysCount)
  const summarizableMessages = plan ? messages.slice(sysCount, cut) : []
  const recentMessages = messages.slice(cut)
  const goalTokens = roughTextTokens(thread.goal ?? '')
  const planTokens = roughTextTokens((thread.plan ?? []).map((p) => p.text).join('\n'))
  const pinnedTokens = estimateTokens(pinnedMessages) + goalTokens + planTokens
  const summarizableTokens = estimateTokens(summarizableMessages)
  const recentTokens = estimateTokens(recentMessages)
  const toolTokens = toolPressureTokens(messages)
  const pinned: string[] = []
  const summarizable: string[] = []

  if (pinnedMessages.length) pinned.push('System and project instructions')
  if (thread.goal) pinned.push('Active goal')
  if (thread.plan?.length) pinned.push('Plan checklist')
  if (recentMessages.length) pinned.push(`Latest ${recentMessages.length} messages`)
  if (summarizableMessages.length) summarizable.push(`${summarizableMessages.length} older messages`)
  if (summarizableMessages.some((m) => m.role === 'tool' || m.tool_calls?.length)) summarizable.push('Older tool calls and outputs')
  if (!summarizable.length && messages.length > sysCount + KEEP_RECENT) summarizable.push('No safe fold point yet')

  return {
    pinnedTokens,
    recentTokens,
    summarizableTokens,
    toolTokens,
    pinned,
    summarizable,
  }
}

function recommendAction(
  pressureRatio: number,
  workingLimit: number,
  guidance: Omit<ContextPressureGuidance, 'recommendedAction' | 'reason' | 'cacheHint'>,
): Pick<ContextPressureGuidance, 'recommendedAction' | 'reason' | 'cacheHint'> {
  const pinnedTotal = guidance.pinnedTokens + guidance.recentTokens
  let recommendedAction: ContextPressureAction = 'continue'
  let reason = 'Working set is inside the local context budget.'

  if (pressureRatio >= 1 && guidance.summarizableTokens > 500) {
    recommendedAction = 'compact_now'
    reason = 'Estimated context is over the local working limit and older messages can be folded.'
  } else if (pressureRatio >= 1 && pinnedTotal > workingLimit * 0.55) {
    recommendedAction = 'reduce_pins'
    reason = 'The kept prefix and recent tail are taking most of the local working set.'
  } else if (pressureRatio >= 1) {
    recommendedAction = 'split_thread'
    reason = 'There is little safe material to summarize, so a narrower follow-up thread is cleaner.'
  } else if (pressureRatio >= 0.85 && guidance.summarizableTokens > 500) {
    recommendedAction = 'summarize_next'
    reason = 'Pressure is near the working limit; summarize the older middle before the next long turn.'
  } else if (pinnedTotal > workingLimit * 0.5) {
    recommendedAction = 'reduce_pins'
    reason = 'Pinned and recent context dominate the working set; keep only stable instructions pinned.'
  }

  return {
    recommendedAction,
    reason,
    cacheHint: cacheHintFor(recommendedAction),
  }
}

function cacheHintFor(action: ContextPressureAction): string {
  switch (action) {
    case 'compact_now':
      return 'Fold the older middle now, leaving system instructions, goal, and latest turns verbatim.'
    case 'summarize_next':
      return 'Summarize older turns before the next large tool result so the KV prefix stays stable.'
    case 'reduce_pins':
      return 'Move bulky file dumps out of the pinned prefix; keep paths and short evidence notes instead.'
    case 'split_thread':
      return 'Start a narrow follow-up with only the goal, touched files, and latest result.'
    case 'continue':
    default:
      return 'Keep appending turns without rewriting the stable prompt prefix.'
  }
}

function modelContextProfile(settings: Settings, provider: Provider, model: string): ModelContextProfile {
  const engineWindow = provider.source === 'tensorfold' ? engineContextWindow(model) : null
  if (engineWindow != null) {
    const working = recommendedCompactionThreshold(engineWindow)
    return {
      id: `${provider.id}:${model}`,
      providerId: provider.id,
      providerName: provider.name,
      model,
      kind: 'local',
      nominalContextTokens: engineWindow,
      recommendedWorkingTokens: working,
      kvCacheTokens: Math.max(1_000, Math.floor(working * 0.25)),
      source: 'engine',
      note: 'Context window reported by the running TensorFold engine.',
    }
  }
  const known = KNOWN_LOCAL_MODELS.find((entry) => entry.match.test(model))
  const namedContext = getModelContextWindow(model) ?? inferContextFromModelName(model)
  const localLimit = settings.localContextLimit || DEFAULT_LOCAL_CONTEXT
  const cloudLimit = settings.contextLimit || DEFAULT_CLOUD_CONTEXT
  const nominalContextTokens = known?.nominalContextTokens ?? namedContext ?? localLimit
  const recommendedWorkingTokens = Math.min(
    cloudLimit,
    localLimit,
    known?.recommendedWorkingTokens ?? Math.max(4_000, Math.floor(nominalContextTokens * 0.75)),
  )
  const source = known ? 'known' : namedContext ? 'model_name' : provider.models.includes(model) ? 'settings' : 'fallback'
  return {
    id: `${provider.id}:${model}`,
    providerId: provider.id,
    providerName: provider.name,
    model,
    kind: 'local',
    nominalContextTokens,
    recommendedWorkingTokens,
    kvCacheTokens: known?.kvCacheTokens ?? Math.max(1_000, Math.floor(recommendedWorkingTokens * 0.25)),
    source,
    note: known?.note ?? (namedContext ? 'Context window inferred from the model name.' : 'Unknown local model; using the configured local threshold.'),
  }
}

function cloudContextProfile(settings: Settings, provider: Provider, model: string | null): ModelContextProfile {
  const configuredLimit = settings.contextLimit || DEFAULT_CLOUD_CONTEXT
  const realWindow = getModelContextWindow(model)
  const nominalContextTokens = realWindow ?? configuredLimit
  const recommendedWorkingTokens = realWindow
    ? Math.min(configuredLimit, recommendedCompactionThreshold(realWindow))
    : configuredLimit
  return {
    id: `${provider.id}:${model || 'default'}`,
    providerId: provider.id,
    providerName: provider.name,
    model: model || provider.models[0] || 'default',
    kind: 'cloud',
    nominalContextTokens,
    recommendedWorkingTokens,
    kvCacheTokens: 0,
    source: realWindow ? 'known' : 'settings',
    note: realWindow ? 'Cloud model context window from the built-in registry.' : 'Cloud provider; pressure follows the configured cloud compaction threshold.',
  }
}

function providerForThread(settings: Settings, thread: Thread): Provider {
  const provider = settings.providers.find((p) => p.id === thread.providerId)
    ?? settings.providers.find((p) => p.id === settings.activeProviderId)
    ?? settings.providers[0]
  return provider ?? {
    id: 'unknown',
    name: 'Unknown provider',
    baseUrl: '',
    apiKey: '',
    models: [],
    kind: 'cloud',
  }
}

function isLocalProvider(provider: Provider): boolean {
  if (provider.kind === 'local') return true
  return /(^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(?::|\/|$))/.test(provider.baseUrl)
}

function inferContextFromModelName(model: string): number | null {
  const match = model.toLowerCase().match(/(?:^|[-_: ])(\d{1,3})k(?:[-_: ]|$)/)
  if (!match) return null
  const size = Number(match[1])
  if (!Number.isFinite(size) || size < 4) return null
  return size * 1_000
}

function leadingSystemMessages(messages: ApiMessage[]): number {
  let count = 0
  while (messages[count]?.role === 'system') count++
  return count
}

function toolPressureTokens(messages: ApiMessage[]): number {
  let chars = 0
  for (const message of messages) {
    if (message.role === 'tool') chars += message.content?.length ?? 0
    for (const call of message.tool_calls ?? []) chars += call.function.name.length + call.function.arguments.length
  }
  return Math.ceil(chars / 4)
}

function roughTextTokens(text: string): number {
  return Math.ceil(text.length / 4)
}

function statusFor(estimatedTokens: number, pressureRatio: number): ContextPressureStatus {
  if (estimatedTokens <= 0) return 'idle'
  if (pressureRatio >= 1) return 'critical'
  if (pressureRatio >= 0.85) return 'pressured'
  if (pressureRatio >= 0.55) return 'warm'
  return 'healthy'
}

function roundRatio(value: number): number {
  return Math.round(value * 1000) / 1000
}

function sum<T>(items: T[], select: (item: T) => number): number {
  return items.reduce((total, item) => total + select(item), 0)
}
