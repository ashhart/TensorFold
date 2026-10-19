import { useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { buildDiscoveredProvider, mergeDiscoveredProvider } from '../../lib/provider-discovery'
import { persistProviderSetup } from '../../lib/provider-setup-save'
import type { ModelSpendBudget, Provider, ProviderAccountProfile, ProviderDiscoveryKind, Settings as TSettings } from '../../lib/types'
import {
  CLOUD_PROVIDER_PRESETS,
  DEFAULT_PROVIDER_SETUP,
  LOCAL_RUNTIME_PRESETS,
  newProviderId,
  patchSpendBudget,
  providerPresetNeedsApiKey,
  type LocalRuntimeKey,
  type ProviderSeed,
  type ProviderSetupDraft,
} from './shared'
import type { SettingsHookDeps } from './types'

// All provider / model add/connect/discover logic. Owns the add-flow wizard
// state (providerSetup), the connection probes (conn), and the discovery busy
// flags. Receives the draft + commit primitives from Settings.tsx and returns
// an object spread into sectionCtx.
export function useProviderSettings({ draft, editDraft, commit, patch, save }: SettingsHookDeps) {
  const [providerSetup, setProviderSetup] = useState<ProviderSetupDraft>(DEFAULT_PROVIDER_SETUP)
  const [setupBusy, setSetupBusy] = useState(false)
  const [setupStatus, setSetupStatus] = useState<{ state: 'ok' | 'error'; msg: string } | null>(null)
  const [discoverBusy, setDiscoverBusy] = useState<string | null>(null)
  const [copiedLoginAccount, setCopiedLoginAccount] = useState<string | null>(null)
  const [conn, setConn] = useState<Record<string, { state: 'loading' | 'ok' | 'error'; msg?: string }>>({})
  const pendingSetupId = useRef<string | null>(null)

  const updateProvider = (id: string, p: Partial<Provider>) =>
    editDraft((d) => ({ ...d, providers: d.providers.map((x) => (x.id === id ? { ...x, ...p } : x)) }))
  const commitProviders = (providers: Provider[]) => commit({ ...draft, providers })
  const patchProviderAccount = (providers: Provider[], providerId: string, accountId: string, patch: Partial<ProviderAccountProfile>) =>
    providers.map((provider) => provider.id === providerId
      ? { ...provider, accounts: (provider.accounts ?? []).map((account) => account.id === accountId ? { ...account, ...patch, updatedAt: Date.now() } : account) }
      : provider)
  const updateProviderAccount = (providerId: string, accountId: string, patch: Partial<ProviderAccountProfile>) =>
    editDraft((d) => ({ ...d, providers: patchProviderAccount(d.providers, providerId, accountId, patch) }))
  const commitProviderAccount = (providerId: string, accountId: string, patch: Partial<ProviderAccountProfile>) =>
    commitProviders(patchProviderAccount(draft.providers, providerId, accountId, patch))
  const commitProviderSpendBudget = (providerId: string, patch: Partial<ModelSpendBudget>) =>
    commitProviders(draft.providers.map((provider) => provider.id === providerId
      ? { ...provider, spendBudget: patchSpendBudget(provider.spendBudget, patch) }
      : provider))
  const commitProviderAccountSpendBudget = (providerId: string, accountId: string, patch: Partial<ModelSpendBudget>) =>
    commitProviders(draft.providers.map((provider) => provider.id === providerId
      ? {
          ...provider,
          accounts: (provider.accounts ?? []).map((account) => account.id === accountId
            ? { ...account, spendBudget: patchSpendBudget(account.spendBudget, patch), updatedAt: Date.now() }
            : account),
        }
      : provider))
  const cliSubscriptionAccountSeeds = (provider: Provider): Partial<ProviderAccountProfile>[] => {
    if (provider.transport === 'codex-cli') return [
      { name: 'Personal Codex', cliProfile: 'personal', cliHome: '~/.codex-personal' },
      { name: 'Work Codex', cliProfile: 'work', cliHome: '~/.codex-work' },
    ]
    if (provider.transport === 'claude-cli') return [
      { name: 'Personal Claude', cliProfile: '~/.claude-personal/settings.json', cliHome: '~/.claude-personal' },
      { name: 'Work Claude', cliProfile: '~/.claude-work/settings.json', cliHome: '~/.claude-work' },
    ]
    return []
  }
  const buildProviderAccount = (provider: Provider, seed: Partial<ProviderAccountProfile> = {}): ProviderAccountProfile => {
    const now = Date.now()
    return {
      id: 'acct_' + Math.random().toString(36).slice(2, 8),
      name: seed.name || (provider.transport === 'codex-cli' ? 'Codex account' : provider.transport === 'claude-cli' ? 'Claude account' : 'API account'),
      kind: provider.transport === 'codex-cli' || provider.transport === 'claude-cli' ? 'subscription' : 'api-key',
      enabled: true,
      apiKey: '',
      models: seed.models ?? [],
      cliProfile: seed.cliProfile || '',
      cliHome: seed.cliHome || '',
      env: {},
      autoFailover: seed.autoFailover ?? false,
      failoverResetPolicy: seed.failoverResetPolicy ?? 'return_after_reset',
      createdAt: now,
      updatedAt: now,
    }
  }
  const addProviderAccount = (provider: Provider, seed: Partial<ProviderAccountProfile> = {}) => {
    const account = buildProviderAccount(provider, seed)
    commitProviders(draft.providers.map((p) => p.id === provider.id ? { ...p, accounts: [...(p.accounts ?? []), account], defaultAccountId: p.defaultAccountId ?? account.id } : p))
  }
  const addCliSubscriptionPair = (provider: Provider) => {
    const seeds = cliSubscriptionAccountSeeds(provider)
    if (!seeds.length) return
    const existing = provider.accounts ?? []
    const accounts = seeds
      .filter((seed) => !existing.some((account) => account.name === seed.name || (seed.cliHome && account.cliHome === seed.cliHome)))
      .map((seed) => buildProviderAccount(provider, seed))
    if (!accounts.length) return
    commitProviders(draft.providers.map((p) => p.id === provider.id ? { ...p, accounts: [...existing, ...accounts], defaultAccountId: p.defaultAccountId ?? accounts[0]?.id ?? null } : p))
  }
  const removeProviderAccount = (providerId: string, accountId: string) =>
    commitProviders(draft.providers.map((p) => p.id === providerId ? { ...p, accounts: (p.accounts ?? []).filter((account) => account.id !== accountId), defaultAccountId: p.defaultAccountId === accountId ? null : p.defaultAccountId } : p))
  const accountProfileLabel = (p: Provider) => p.transport === 'codex-cli' ? 'Codex profile' : p.transport === 'claude-cli' ? 'Settings file' : 'Profile'
  const accountProfileHint = (p: Provider) => p.transport === 'codex-cli' ? 'passed as --profile' : p.transport === 'claude-cli' ? 'passed as --settings' : undefined
  const accountHomeHint = (p: Provider) => p.transport === 'codex-cli' ? 'sets CODEX_HOME' : p.transport === 'claude-cli' ? 'sets Claude config env' : undefined
  const shellArg = (value: string) => /^[~A-Za-z0-9_@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, `'\\''`)}'`
  const loginCommand = (provider: Provider, account: ProviderAccountProfile): string | null => {
    const bin = shellArg(provider.cliPath?.trim() || (provider.transport === 'claude-cli' ? 'claude' : 'codex'))
    if (provider.transport === 'codex-cli') {
      const home = shellArg(account.cliHome?.trim() || '~/.codex')
      return `mkdir -p ${home} && CODEX_HOME=${home} ${bin} login`
    }
    if (provider.transport === 'claude-cli') {
      const home = shellArg(account.cliHome?.trim() || '~/.claude')
      return `mkdir -p ${home} && CLAUDE_CONFIG_DIR=${home} CLAUDE_HOME=${home} ${bin} login`
    }
    return null
  }
  const patchProviderAccountEnv = (providerId: string, account: ProviderAccountProfile, key: string, value: string, commitNow = false) => {
    const env = { ...(account.env ?? {}) }
    if (value) env[key] = value
    else delete env[key]
    if (commitNow) commitProviderAccount(providerId, account.id, { env })
    else updateProviderAccount(providerId, account.id, { env })
  }
  const copyLoginCommand = async (provider: Provider, account: ProviderAccountProfile) => {
    const command = loginCommand(provider, account)
    if (!command) return
    await api.writeClipboard(command)
    setCopiedLoginAccount(account.id)
    window.setTimeout(() => setCopiedLoginAccount((id) => id === account.id ? null : id), 1400)
  }

  // Provider type is chosen at add-time (#44): HTTP uses the cloud preset
  // dropdown; the CLI types resolve to their dedicated preset.
  const cliPresetKey = providerSetup.transportType === 'codex-cli' ? 'codex-cli' : providerSetup.transportType === 'claude-cli' ? 'claude-cli' : null
  const selectedCloudPreset = providerSetup.transportType === 'http'
    ? (CLOUD_PROVIDER_PRESETS.find((preset) => preset.key === providerSetup.cloudKey && preset.auth !== 'cli-oauth') ?? CLOUD_PROVIDER_PRESETS[0])
    : (CLOUD_PROVIDER_PRESETS.find((preset) => preset.key === cliPresetKey) ?? CLOUD_PROVIDER_PRESETS[0])
  const selectedLocalRuntime = LOCAL_RUNTIME_PRESETS.find((preset) => preset.key === providerSetup.localRuntime) ?? LOCAL_RUNTIME_PRESETS[0]
  const patchProviderSetup = (patch: Partial<ProviderSetupDraft>) => {
    setSetupStatus(null)
    setProviderSetup((current) => ({ ...current, ...patch }))
  }
  const selectLocalRuntime = (runtimeKey: LocalRuntimeKey) => {
    const currentRuntime = LOCAL_RUNTIME_PRESETS.find((preset) => preset.key === providerSetup.localRuntime) ?? LOCAL_RUNTIME_PRESETS[0]
    const nextRuntime = LOCAL_RUNTIME_PRESETS.find((preset) => preset.key === runtimeKey) ?? currentRuntime
    const shouldReplacePort = !providerSetup.port || providerSetup.port === currentRuntime.defaultPort
    const shouldReplaceKey = !providerSetup.apiKey || providerSetup.apiKey === currentRuntime.defaultApiKey
    patchProviderSetup({
      localRuntime: nextRuntime.key,
      port: shouldReplacePort ? nextRuntime.defaultPort : providerSetup.port,
      apiKey: shouldReplaceKey ? nextRuntime.defaultApiKey : providerSetup.apiKey,
    })
  }
  const normalizeBasePath = (value: string) => {
    const text = value.trim() || '/v1'
    return text.startsWith('/') ? text : `/${text}`
  }
  const localBaseUrl = () => {
    const host = providerSetup.localLocation === 'this-mac' ? '127.0.0.1' : (providerSetup.host.trim() || '127.0.0.1')
    const base = /^https?:\/\//i.test(host) ? host : `http://${host}`
    try {
      const url = new URL(base)
      if (providerSetup.port.trim()) url.port = providerSetup.port.trim()
      url.pathname = normalizeBasePath(providerSetup.basePath)
      url.search = ''
      url.hash = ''
      return url.toString().replace(/\/+$/, '')
    } catch {
      return `http://${host}:${providerSetup.port.trim() || selectedLocalRuntime.defaultPort}${normalizeBasePath(providerSetup.basePath)}`
    }
  }
  const localNodeName = () => {
    if (providerSetup.providerName.trim()) return providerSetup.providerName.trim()
    const node = providerSetup.localLocation === 'this-mac' ? 'This Mac' : (providerSetup.nodeName.trim() || providerSetup.host.trim() || 'Remote node')
    return `${node} · ${selectedLocalRuntime.label}`
  }
  const localRuntimeDescription = () => {
    if (providerSetup.localRuntime === 'omlx' && providerSetup.localLocation === 'network') return 'Remote oMLX server. Enter the node host, port, and API token from that machine.'
    if (providerSetup.localRuntime === 'omlx') return 'MLX local server. TensorFold can read ~/.omlx/settings.json on this Mac.'
    if (providerSetup.localLocation === 'network') return `${selectedLocalRuntime.label} running on another machine on your network.`
    return selectedLocalRuntime.desc
  }
  const providerExists = (provider: ProviderSeed) =>
    draft.providers.some((p) => provider.source ? p.source === provider.source || p.id === provider.source : provider.transport && provider.transport !== 'chat-completions' ? p.transport === provider.transport : p.baseUrl === provider.baseUrl)
  const addProviderToSettings = (provider: Provider, models?: string[]) =>
    persistProviderSetup(draft, provider, models, async (next) => { if (save) await save(next); else commit(next) })
  const buildCloudProvider = (): Provider => {
    const seed = selectedCloudPreset.provider
    // Transport is locked to the add-time provider-type choice (#44).
    const transport: Provider['transport'] = providerSetup.transportType === 'codex-cli' ? 'codex-cli' : providerSetup.transportType === 'claude-cli' ? 'claude-cli' : (seed.transport ?? 'chat-completions')
    const isCli = transport === 'codex-cli' || transport === 'claude-cli'
    const needsApiKey = providerPresetNeedsApiKey(selectedCloudPreset)
    const provider: Provider = {
      id: newProviderId(),
      name: providerSetup.providerName.trim() || seed.name,
      transport,
      auth: isCli ? (transport === 'codex-cli' ? 'codex-oauth' : 'claude-oauth') : needsApiKey ? (seed.auth ?? 'api-key') : seed.auth,
      baseUrl: isCli ? '' : seed.baseUrl,
      apiKey: isCli || !needsApiKey ? '' : providerSetup.apiKey,
      models: seed.models ?? [],
      kind: 'cloud',
      source: seed.source,
      cliPath: providerSetup.cliPath.trim() || seed.cliPath,
      cliProfile: providerSetup.cliProfile.trim() || seed.cliProfile,
      cliSandbox: seed.cliSandbox,
      cliApprovalPolicy: seed.cliApprovalPolicy,
      cliPermissionMode: seed.cliPermissionMode,
      // OpenCode transport carries its server endpoint (VST-033), not a /chat/completions
      // baseUrl. serverUrl defaults from the preset; serverPassword is reused from the
      // apiKey field so the add flow needs no bespoke input.
      ...(transport === 'opencode' || transport === 'kilo' ? { serverUrl: seed.serverUrl || 'http://127.0.0.1:4096', serverPassword: providerSetup.apiKey.trim() || undefined } : {}),
    }
    if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli') {
      const accounts = cliSubscriptionAccountSeeds(provider).map((accountSeed) => buildProviderAccount(provider, accountSeed))
      return { ...provider, accounts, defaultAccountId: accounts[0]?.id ?? null }
    }
    return provider
  }
  const buildLocalProvider = (): Provider => ({
    id: newProviderId(),
    name: localNodeName(),
    transport: 'chat-completions',
    auth: 'api-key',
    baseUrl: localBaseUrl(),
    apiKey: providerSetup.apiKey || selectedLocalRuntime.defaultApiKey,
    models: selectedLocalRuntime.defaultModels,
    kind: 'local',
    source: selectedLocalRuntime.source,
    ...(selectedLocalRuntime.trustedNetwork ? { trustedNetwork: true as const } : {}),
  })
  // Dedicated add path for Codex/Claude CLI (the simplified single/multi flow).
  // single  → no extra accounts (uses the default CLI login).
  // multi   → the wizard's chosen subscription accounts.
  // Awaits the persist so the caller can immediately drive sign-in (cliAuth:login
  // looks the provider up by id in the saved store).
  const addCliProvider = async (transport: 'codex-cli' | 'claude-cli', accountSeeds: Partial<ProviderAccountProfile>[] | null): Promise<Provider> => {
    const seed = (CLOUD_PROVIDER_PRESETS.find((p) => p.key === transport) ?? CLOUD_PROVIDER_PRESETS[0]).provider
    const base: Provider = {
      id: newProviderId(),
      name: providerSetup.providerName.trim() || seed.name,
      transport,
      auth: transport === 'codex-cli' ? 'codex-oauth' : 'claude-oauth',
      baseUrl: '',
      apiKey: '',
      models: seed.models ?? [],
      kind: 'cloud',
      source: seed.source,
      cliPath: providerSetup.cliPath.trim() || seed.cliPath,
      cliProfile: providerSetup.cliProfile.trim() || seed.cliProfile,
      cliSandbox: seed.cliSandbox,
      cliApprovalPolicy: seed.cliApprovalPolicy,
      cliPermissionMode: seed.cliPermissionMode,
    }
    const provider = accountSeeds && accountSeeds.length
      ? (() => { const accounts = accountSeeds.map((s) => buildProviderAccount(base, s)); return { ...base, accounts, defaultAccountId: accounts[0]?.id ?? null } })()
      : base
    const next: TSettings = {
      ...draft,
      providers: [...draft.providers, provider],
      activeProviderId: draft.activeProviderId ?? provider.id,
      activeModel: draft.activeProviderId ? draft.activeModel : (provider.models[0] ?? draft.activeModel),
    }
    // Sign-in starts only after the shared draft transaction confirms persistence.
    // The same transaction retains ownership and the form on a failed save.
    if (save) await save(next)
    else commit(next)
    setProviderSetup((current) => ({ ...current, providerName: '', cliProfile: '' }))
    return provider
  }
  const addConfiguredProvider = async (connectNow = true) => {
    const candidate = providerSetup.mode === 'cloud' ? buildCloudProvider() : buildLocalProvider()
    // The shared Retry save banner can confirm the failed add independently of
    // this hook; a subsequent Add then owns a new provider rather than that retry.
    if (pendingSetupId.current && useStore.getState().settings?.providers.some((item) => item.id === pendingSetupId.current)) pendingSetupId.current = null
    const provider = { ...candidate, id: pendingSetupId.current ?? candidate.id }
    pendingSetupId.current = provider.id
    setSetupBusy(true)
    setSetupStatus(null)
    try {
      let models: string[] | undefined
      let refreshError: string | null = null
      if (connectNow) try {
        models = await api.fetchModels(provider)
        if (!models.length) throw new Error('No models returned')
      } catch (e: any) {
        refreshError = String(e?.message || e || 'Could not refresh models').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, '')
      }
      const saved = await addProviderToSettings(provider, models)
      setConn((c) => ({ ...c, [saved.id]: refreshError ? { state: 'error', msg: refreshError } : { state: 'ok', msg: connectNow ? `${saved.models.length} model${saved.models.length === 1 ? '' : 's'}` : 'Added' } }))
      setSetupStatus(refreshError ? { state: 'error', msg: `Added ${provider.name}, but model refresh failed: ${refreshError}` } : { state: 'ok', msg: connectNow ? `Added ${saved.name} with ${saved.models.length} model${saved.models.length === 1 ? '' : 's'}.` : `Added ${saved.name}.` })
      pendingSetupId.current = null
      setProviderSetup((current) => ({ ...current, providerName: '', apiKey: '', cliProfile: '' }))
    } catch (e: any) {
      const msg = String(e?.message || e || 'Could not connect').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, '')
      setSetupStatus({ state: 'error', msg: `Provider was not saved: ${msg}` })
    } finally {
      setSetupBusy(false)
    }
  }

  // Probe the endpoint WITHOUT adding a provider, so you can verify reachability/auth before committing.
  const testConfiguredProvider = async () => {
    const provider = providerSetup.mode === 'cloud' ? buildCloudProvider() : buildLocalProvider()
    setSetupBusy(true)
    setSetupStatus(null)
    try {
      const models = await api.fetchModels(provider)
      const model = provider.models.find((candidate) => models.includes(candidate)) ?? models[0]
      if (!model) throw new Error('Connected, but the endpoint returned no models.')
      const probe = await api.testProviderResponse({ provider, model })
      setSetupStatus({ state: 'ok', msg: `Response received from ${probe.model} in ${(probe.firstTokenMs / 1000).toFixed(1)}s; ${models.length} model${models.length === 1 ? '' : 's'} available. Provider has not been added.` })
    } catch (e: any) {
      const msg = String(e?.message || e || 'Could not connect').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, '')
      setSetupStatus({ state: 'error', msg })
    } finally {
      setSetupBusy(false)
    }
  }
  const addProvider = () => {
    const np: Provider = { id: newProviderId(), name: 'Custom local endpoint', transport: 'chat-completions', auth: 'api-key', baseUrl: 'http://127.0.0.1:8000/v1', apiKey: '', models: [], kind: 'local', source: 'custom' }
    commitProviders([...draft.providers, np])
  }
  const removeProvider = (id: string) => {
    if (draft.providers.length <= 1) {
      setSetupStatus({ state: 'error', msg: 'Keep at least one provider configured — add another before removing the last one.' })
      return
    }
    const providers = draft.providers.filter((p) => p.id !== id)
    // The Assistant surface can point at a *dedicated* provider/model (Models settings) that is
    // independent of the active provider. resolveAssistantModel honours the override whenever the
    // id is non-null, so an orphaned id would silently route the Assistant to a deleted provider —
    // clear the override here when its provider is the one being removed.
    const assistantReset = draft.assistantProviderId === id
      ? { assistantProviderId: null, assistantProviderAccountId: null, assistantModel: null }
      : {}
    // Same orphan guard for the coding-tasks surface default (VST-015).
    const codeReset = draft.codeProviderId === id
      ? { codeProviderId: null, codeProviderAccountId: null, codeModel: null }
      : {}
    if (draft.activeProviderId !== id) {
      const active = providers.find((p) => p.id === draft.activeProviderId)
      const activeAccountValid = !draft.activeProviderAccountId || active?.accounts?.some((account) => account.id === draft.activeProviderAccountId)
      commit({ ...draft, providers, activeProviderAccountId: activeAccountValid ? draft.activeProviderAccountId : null, ...assistantReset, ...codeReset })
      return
    }
    const fallback = providers[0] ?? null
    commit({
      ...draft,
      providers,
      activeProviderId: fallback?.id ?? null,
      activeProviderAccountId: fallback?.defaultAccountId ?? null,
      activeModel: fallback?.models?.[0] ?? null,
      ...assistantReset, ...codeReset,
    })
  }
  const localDiscoveryKind = (p: Partial<Provider>): ProviderDiscoveryKind | null => {
    if (p.source === 'omlx' || p.id === 'omlx' || /\bomlx\b/i.test(p.name || '')) return 'omlx'
    if (p.source === 'ollama' || p.id === 'ollama' || /ollama/i.test(p.name || '') || /:11434(?:\/|$)/.test(p.baseUrl || '')) return 'ollama'
    if (p.source === 'lmstudio' || p.id === 'lmstudio' || /lm\s*studio/i.test(p.name || '') || /:1234(?:\/|$)/.test(p.baseUrl || '')) return 'lmstudio'
    return null
  }
  const isOpenRouterProvider = (p: Provider) => p.source === 'openrouter' || /openrouter\.ai/i.test(p.baseUrl)
  const isAiGatewayProvider = (p: Provider) => p.source === 'ai-gateway' || /ai-gateway\.vercel\.sh/i.test(p.baseUrl)
  const discoveryLabel = (kind: ProviderDiscoveryKind) => kind === 'omlx' ? 'oMLX' : kind === 'lmstudio' ? 'LM Studio' : 'Ollama'
  const applyLocalDiscovery = async (kind: ProviderDiscoveryKind, providerId?: string) => {
    const busyKey = providerId || 'new'
    setDiscoverBusy(busyKey)
    if (!providerId) setSetupStatus(null)
    try {
      // Pass the provider's own base URL + key so a non-loopback (LAN) host is probed
      // directly instead of falling back to this machine's local runtime config.
      const existing = providerId ? draft.providers.find((p) => p.id === providerId) : null
      const discovered = await api.discoverLocalProvider(kind, existing?.baseUrl, existing?.apiKey)
      if (!discovered.found) {
        if (providerId) setConn((c) => ({ ...c, [providerId]: { state: 'error', msg: discovered.message } }))
        else {
          const id = newProviderId()
          const provider: Provider = {
            id,
            name: discovered.name,
            transport: 'chat-completions',
            auth: 'api-key',
            kind: 'local',
            source: discovered.source,
            baseUrl: discovered.baseUrl,
            apiKey: discovered.apiKey,
            models: discovered.models.length ? discovered.models : ['local-model'],
          }
          commitProviders([...draft.providers, provider])
          setConn((c) => ({ ...c, [id]: { state: 'error', msg: discovered.message } }))
          setSetupStatus({ state: 'error', msg: discovered.message })
        }
        return
      }
      // Build the discovered provider, then dedup against existing endpoints so
      // rapid blurs + discovery retries can't spawn duplicate nodes (#45).
      const seedId = providerId || newProviderId()
      const seed = draft.providers.find((p) => p.id === providerId) ?? null
      const built = buildDiscoveredProvider(discovered, seedId, seed)
      const merged = mergeDiscoveredProvider(draft.providers, built, providerId)
      const nextProvider = merged.target
      const providers = merged.providers
      commit({
        ...draft,
        providers,
        activeProviderId: draft.activeProviderId ?? nextProvider.id,
        activeModel: (draft.activeProviderId === nextProvider.id || !draft.activeProviderId)
          ? (discovered.defaultModel || nextProvider.models[0] || draft.activeModel)
          : draft.activeModel,
      })
      setConn((c) => ({ ...c, [nextProvider.id]: { state: discovered.models.length ? 'ok' : 'error', msg: discovered.message } }))
      if (!providerId) setSetupStatus({ state: discovered.models.length ? 'ok' : 'error', msg: discovered.message })
    } catch (e: any) {
      const msg = String(e?.message || e || `${discoveryLabel(kind)} discovery failed`)
      if (providerId) setConn((c) => ({ ...c, [providerId]: { state: 'error', msg } }))
      else setSetupStatus({ state: 'error', msg })
    } finally {
      setDiscoverBusy(null)
    }
  }

  const browseCwd = async () => { const d = await api.pickDirectory(); if (d) patch({ defaultCwd: d }) }

  const connect = async (p: Provider) => {
    setConn((c) => ({ ...c, [p.id]: { state: 'loading' } }))
    try {
      const models = await api.fetchModels(p)
      if (!models.length) { setConn((c) => ({ ...c, [p.id]: { state: 'error', msg: 'No models returned' } })); return }
      // Remember the full discovered catalogue so the "Edit models" checklist can offer everything
      // the endpoint exposes (#46) — but PRESERVE the user's enabled/disabled selection across a
      // refresh. Keep previously-selected models that still exist, and auto-enable only genuinely-new
      // models (not ones the user had explicitly disabled).
      commitProviders(draft.providers.map((x) => {
        if (x.id !== p.id) return x
        const prevSelected = x.models ?? []
        const prevCatalogue = x.discoveredModels ?? []
        const kept = prevSelected.filter((m) => models.includes(m))
        const brandNew = models.filter((m) => !prevCatalogue.includes(m) && !prevSelected.includes(m))
        const nextModels = prevSelected.length ? Array.from(new Set([...kept, ...brandNew])) : models
        return { ...x, models: nextModels, discoveredModels: models }
      }))
      setConn((c) => ({ ...c, [p.id]: { state: 'ok', msg: `${models.length} model${models.length > 1 ? 's' : ''}` } }))
    } catch (e: any) {
      const msg = String(e?.message || '').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, '') || 'Connection failed'
      setConn((c) => ({ ...c, [p.id]: { state: 'error', msg } }))
    }
  }

  return {
    updateProvider,
    commitProviders,
    updateProviderAccount,
    commitProviderAccount,
    commitProviderSpendBudget,
    commitProviderAccountSpendBudget,
    addProviderAccount,
    addCliSubscriptionPair,
    removeProviderAccount,
    accountProfileLabel,
    accountProfileHint,
    accountHomeHint,
    loginCommand,
    patchProviderAccountEnv,
    copyLoginCommand,
    copiedLoginAccount,
    selectedCloudPreset,
    selectedLocalRuntime,
    providerSetup,
    patchProviderSetup,
    selectLocalRuntime,
    localBaseUrl,
    localNodeName,
    localRuntimeDescription,
    providerExists,
    addConfiguredProvider,
    addCliProvider,
    testConfiguredProvider,
    setupBusy,
    setupStatus,
    addProvider,
    removeProvider,
    localDiscoveryKind,
    isOpenRouterProvider,
    isAiGatewayProvider,
    discoveryLabel,
    applyLocalDiscovery,
    discoverBusy,
    browseCwd,
    conn,
    connect,
  }
}
