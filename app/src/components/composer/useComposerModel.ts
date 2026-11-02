import { useEffect, useMemo, useState } from 'react'
import { useStore } from '../../lib/store'
import { api } from '../../lib/ipc'
import type { Settings, Thread, Provider, Readiness, TokenUsageSnapshot, PersistedProviderStatus, ModelRouterSnapshot, ContextPressureSnapshot } from '../../lib/types'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { allowedProviders, projectDefaultModel, violatesPolicy } from '../../lib/policy'
import { resolveProviderModel } from '../../lib/provider-accounts'
import { resolveSurfaceModel, settingsForSurfaceModelPick } from '../../lib/assistant-model'
import { isExternalProvider } from '../../../shared/provider-trust'
import { currentModelQualification, modelTaskFit, type ModelTaskRequirements } from '../../../shared/model-fit'
import { activeEngineModelSettings, engineProfileContextWindow } from '../../../shared/model-profiles'
import { quotedUserText } from '../../../shared/message-quotes'
import { providerStatusViewFor } from '../../lib/provider-status'
import { providerUsageSummary } from '../../lib/provider-usage'
import { modelContextWindowTokens, providerSupportsFastMode, type ContextUsageSummary } from '../../lib/provider-model-metadata'
import { useEngineContextTokens } from '../../lib/engine'
import { boundedComposerEstimate } from '../../lib/composer-estimate'
import { composerBudget } from '../../lib/composer-budget'
import { loadRecentModels, pushRecentModel, type RecentModel } from '../../lib/recent-models'
import { useModelQualifications } from '../models/qualification/useModelQualifications'
import type { useConversationDraft } from '../../lib/use-conversation-draft'
import type { ComposerWorkspace } from './useComposerWorkspace'

interface ComposerModelOptions {
  mode: 'home' | 'thread'
  thread?: Thread
  settings: Settings
  draft: ReturnType<typeof useConversationDraft>
  work: ComposerWorkspace
}

export function useComposerModel({ mode, thread, settings, draft, work }: ComposerModelOptions) {
  const tx = useAuthoredText()
  const { text, attachments } = draft
  const quotes = draft.quotes ?? []
  const { policyProject, effectiveHostId, workCwd } = work
  const saveSettings = useStore((s) => s.saveSettings)
  const setThreadProvider = useStore((s) => s.setThreadProvider)
  const noteModelChange = useStore((s) => s.noteModelChange)
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [authBusy, setAuthBusy] = useState(false)
  const [usageSnapshot, setUsageSnapshot] = useState<TokenUsageSnapshot | null>(null)
  const [providerStatus, setProviderStatus] = useState<PersistedProviderStatus | null>(null)
  const [modelRouterSnapshot, setModelRouterSnapshot] = useState<ModelRouterSnapshot | null>(null)
  const [contextPressureSnapshot, setContextPressureSnapshot] = useState<ContextPressureSnapshot | null>(null)
  const localOnly = !!policyProject?.localOnly || !!thread?.phoneLocalOnly || !!thread?.googleLocalOnly
  const pickerProviders = allowedProviders(settings.providers, policyProject)
  // Per-surface default (VST-015): the home composer launches CODING tasks, so it seeds from
  // the code-surface route (Settings → Models → Coding tasks model), which folds back to the
  // globally-active route when no override is configured. Existing threads keep their own route.
  const codeSurface = resolveSurfaceModel(settings, 'code')
  const rawProviderId = thread?.bot?.providerId ?? (mode === 'thread' ? thread?.providerId ?? settings.activeProviderId : codeSurface.providerId)
  let provider = settings.providers.find((p) => p.id === rawProviderId) ?? (thread?.bot ? undefined : settings.providers[0])
  let providerAccountId = thread?.bot ? thread.bot.providerAccountId : mode === 'thread'
    ? (thread && Object.prototype.hasOwnProperty.call(thread, 'providerAccountId') ? (thread.providerAccountId ?? null) : (settings.activeProviderAccountId ?? provider?.defaultAccountId ?? null))
    : (codeSurface.providerAccountId ?? provider?.defaultAccountId ?? null)
  let model = thread?.bot?.model ?? (mode === 'thread' ? thread?.model ?? settings.activeModel : codeSurface.model)
  // Enforce the project policy: under local-only, fall back to the project's
  // default (or first local) instead of an out-of-policy cloud model.
  if (violatesPolicy(policyProject, provider)) {
    const d = projectDefaultModel(policyProject, settings.providers)
    provider = settings.providers.find((p) => p.id === d.providerId) || pickerProviders[0] || provider
    providerAccountId = d.providerAccountId ?? provider?.defaultAccountId ?? null
    model = d.model ?? provider?.models?.[0] ?? model
  }

  // Point-of-use readiness: is there a usable model? Surfaces "connect a model" /
  // "authorize" guidance instead of letting a send fail with a cryptic error.
  model = provider && !thread?.bot ? (resolveProviderModel(provider, providerAccountId, model) ?? model) : model
  const providerKey = provider?.id ?? ''
  useEffect(() => {
    let live = true
    api.getModelReadiness({ providerId: provider?.id ?? null, accountId: providerAccountId, model })
      .then((r) => { if (live) setReadiness(r) })
      .catch(() => { if (live) setReadiness(null) })
    return () => { live = false }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [providerKey, providerAccountId, model, settings.providers])

  const authorizeCli = async () => {
    if (!provider) return
    setAuthBusy(true)
    try {
      await api.loginCliProvider({ providerId: provider.id, accountId: providerAccountId })
      setReadiness(await api.getModelReadiness({ providerId: provider.id, accountId: providerAccountId, model }))
    } finally {
      setAuthBusy(false)
    }
  }
  const usageSummary = provider ? providerUsageSummary(provider, providerAccountId, model, usageSnapshot, modelRouterSnapshot) : null
  // Real rate-limit windows for the active provider/account (VST-037): session/weekly pace + ETA.
  const rateLimitStatus = provider ? providerStatusViewFor(providerStatus, provider.id, providerAccountId) : null
  const contextThread = thread ? contextPressureSnapshot?.threads.find((row) => row.id === thread.id && row.providerId === provider?.id && row.model === model) ?? null : null
  // The built-in engine's served window (max_model_len) beats a cached or name-based guess.
  const engineContextTokens = useEngineContextTokens(provider?.source === 'tensorfold', model)
  const selectedModelSettings = provider?.source === 'tensorfold' && model ? activeEngineModelSettings(settings, model) : null
  const contextWindowTokens = selectedModelSettings ? engineProfileContextWindow(selectedModelSettings, engineContextTokens) : engineContextTokens
    ?? contextThread?.contextWindowTokens
    ?? modelContextWindowTokens(provider, model)
    ?? (provider?.kind === 'local' ? settings.localContextLimit : settings.contextLimit)
    ?? 128_000
  const composerEstimate = useMemo(() => boundedComposerEstimate(text, attachments), [text, attachments])
  const { reports: qualifications } = useModelQualifications(api, effectiveHostId ?? null)
  const contextUsedTokens = contextThread?.estimatedTokens ?? composerEstimate.tokens
  const contextUsage: ContextUsageSummary = {
    usedTokens: contextUsedTokens,
    windowTokens: contextWindowTokens,
    workingLimitTokens: contextThread?.workingLimitTokens ?? null,
    percent: contextWindowTokens > 0 ? contextUsedTokens / contextWindowTokens : 0,
    status: contextThread?.status ?? (composerEstimate.partial ? 'Partial draft estimate; some attachment text exceeds the preview limit' : 'Approximate draft estimate'),
  }
  const canConfigureFastMode = provider?.transport === 'codex-cli' && providerSupportsFastMode(provider, model)
  const preSendBudget = useMemo(() => composerBudget({
    text, attachments, quotedText: quotes.length ? quotedUserText('', quotes) : '', messages: thread?.apiMessages ?? [], windowTokens: contextWindowTokens,
    windowVerified: !!(selectedModelSettings || engineContextTokens || contextThread || modelContextWindowTokens(provider, model)),
    outputReserve: selectedModelSettings?.maxOutputTokens ?? null,
    snapshotAt: contextPressureSnapshot?.generatedAt, updatedAt: thread?.updatedAt, running: thread?.status === 'running',
  }), [text, attachments, quotes, thread?.apiMessages, contextWindowTokens, selectedModelSettings, engineContextTokens, contextThread, provider, model, contextPressureSnapshot?.generatedAt, thread?.updatedAt, thread?.status])
  const taskRequirements: ModelTaskRequirements = {
    images: attachments.some(attachment => attachment.kind === 'image' && !attachment.text),
    tools: mode === 'thread' && thread?.kind === 'task' && !!workCwd && !!provider && !isExternalProvider(provider),
    localOnly,
    ...(thread?.bot ? { privateContext: 'bot' as const } : thread?.phoneLocalOnly ? { privateContext: 'phone' as const } : thread?.googleLocalOnly ? { privateContext: 'google' as const } : thread?.incognito ? { privateContext: 'incognito' as const } : {}),
  }
  const currentModelFit = modelTaskFit({ requirements: taskRequirements, local: !!provider && !isExternalProvider(provider), report: provider && model ? currentModelQualification(qualifications, { providerId: provider.id, accountId: providerAccountId, model }) : null, toolPolicyAllowed: !!provider && !isExternalProvider(provider) })
  useEffect(() => {
    let live = true
    Promise.all([
      api.getTokenUsage().catch(() => null),
      api.getModelRouterSnapshot().catch(() => null),
      api.getContextPressure().catch(() => null),
      api.getProviderStatus().catch(() => null),
    ]).then(([usage, routes, pressure, status]) => {
      if (!live) return
      setUsageSnapshot(usage)
      setModelRouterSnapshot(routes)
      setContextPressureSnapshot(pressure)
      setProviderStatus(status)
    })
    // Live rate-limit refresh (VST-037): a codex turn pushes updated windows.
    const off = api.onProviderStatus?.((status) => { if (live) setProviderStatus(status) })
    const offUsage = api.onUsageCleared?.(() => { if (live) setUsageSnapshot(null) })
    return () => { live = false; off?.(); offUsage?.() }
  }, [])
  // Re-bind a thread when policy or provider model availability makes its stored model stale.
  useEffect(() => {
    if (mode === 'thread' && thread && !thread.bot && provider && model && (
      (localOnly && violatesPolicy(policyProject, settings.providers.find((p) => p.id === thread.providerId))) ||
      thread.providerId !== provider.id ||
      (thread.providerAccountId ?? null) !== (providerAccountId ?? null) ||
      thread.model !== model
    )) {
      void setThreadProvider(thread.id, provider.id, model || provider.models[0], providerAccountId).catch(() => {})
    }
  }, [thread?.id, provider?.id, providerAccountId, model, localOnly]) // eslint-disable-line
  const pickModel = async (p: Provider, m: string, accountId: string | null = null) => {
    if (mode === 'thread' && thread) {
      // Inline "Model changed to X" notice — only on an existing thread, and only when
      // the pick is an actual change. Home / first selection stays notice-free.
      const changed = !(p.id === provider?.id && m === model && (accountId ?? null) === (providerAccountId ?? null))
      await setThreadProvider(thread.id, p.id, m, accountId)
      if (changed) await noteModelChange(thread.id, p.id, m, accountId).catch(() => {})
    } else await saveSettings(settingsForSurfaceModelPick(useStore.getState().settings ?? settings, 'code', { providerId: p.id, providerAccountId: accountId, model: m }), { silent: true })
    setRecentModels(pushRecentModel({ providerId: p.id, accountId, model: m }))
  }
  // One-click recent-model chips. Refresh from storage on mount so other threads'
  // picks propagate.
  const [recentModels, setRecentModels] = useState<RecentModel[]>(() => loadRecentModels())
  useEffect(() => { setRecentModels(loadRecentModels()) }, [thread?.id])

  return { provider, providerAccountId, model, localOnly, pickerProviders, readiness, authBusy, authorizeCli,
    usageSnapshot, usageSummary, rateLimitStatus, contextUsage, canConfigureFastMode, qualifications,
    taskRequirements, currentModelFit, preSendBudget, pickModel, recentModels }
}

export type ComposerModel = ReturnType<typeof useComposerModel>
