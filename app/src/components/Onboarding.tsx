import { LanguageQuickSelect } from './preferences/LanguageQuickSelect'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useFlowText } from '../lib/preferences/useFlowText'
import { useEffect, useRef, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { requestSettingsSection, useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { Icon, type IconName } from '../lib/icons'
import { BrandIcon, providerBrand } from '../lib/brand'
import { Button, Input } from './ui'
import {
  CLOUD_PROVIDER_PRESETS,
  LOCAL_RUNTIME_PRESETS,
  newProviderId,
  providerPresetNeedsApiKey,
  type CloudProviderPreset,
  type LocalRuntimePreset,
} from './settings/shared'
import {
  onboardingPreviouslyCompleted,
  readOnboardingDraft,
  saveOnboardingDraft,
  ONBOARDING_STEPS,
  nextOnboardingStep,
  onboardingStepIndex,
  previousOnboardingStep,
  onboardingProviderVisible,
  type OnboardingStepId,
  type OnboardingDesktopMode,
} from './onboarding-model'
export { ONBOARDED_KEY, ONBOARDING_STEPS, nextOnboardingStep, onboardingStepIndex, previousOnboardingStep, shouldShowOnboarding } from './onboarding-model'
import type { Provider, ProviderDiscoveryKind, Readiness } from '../lib/types'
import { EngineChoice } from './models/EngineChoice'
import { BrandMark } from './BrandMark'
import { ENGINE_PROVIDER_ID, formatBytes, useEngine } from '../lib/engine'
import { fullLayoutPreferenceReadable, savedFullWorkspaceLayout } from '../lib/full-workspace-layout'
import { finishOnboardingLanding } from '../lib/onboarding-landing'
import { FirstAnswerStep } from './onboarding/first-answer/FirstAnswerStep'
import { MinimalSetupGuide } from './onboarding/first-answer/MinimalSetupGuide'
import { firstAnswerNextAction, readFirstAnswerStarter, saveFirstAnswerStarter, type FirstAnswerStarterId, type FirstAnswerReadiness } from './onboarding/first-answer/model'
import { openFirstAnswerDraft } from './onboarding/first-answer/transition'
import { conversationDraftsFor } from '../lib/conversation-drafts'
import { desktopDraftPort } from '../lib/desktop-drafts'

type ProviderSetupStep = 'choice' | 'cloud' | 'local' | 'configure'
type Mode = 'cloud' | 'local'

const DISCOVERY_SOURCES: ProviderDiscoveryKind[] = ['omlx', 'ollama', 'lmstudio', 'custom']

export function Onboarding({ onDone }: { onDone: () => void }) {
  const tx = useAuthoredText()
  const { t, authored, known, locale } = useFlowText()
  const settings = useStore((s) => s.settings)
  const saveSettings = useStore((s) => s.saveSettings)
  const setView = useStore((s) => s.setView)
  const initialDraft = useRef(readOnboardingDraft(settings?.desktopMode ?? 'full')).current
  const previouslyCompleted = useRef(onboardingPreviouslyCompleted(settings?.desktopOnboardingComplete)).current
  const layoutBeforeSetup = useRef(savedFullWorkspaceLayout()).current
  const preferenceReadable = useRef(fullLayoutPreferenceReadable()).current
  const [desktopMode, setDesktopMode] = useState<OnboardingDesktopMode>(initialDraft.mode)
  const [tourStep, setTourStep] = useState<OnboardingStepId>(initialDraft.step)
  const [finishing, setFinishing] = useState(false)
  const finishLock = useRef(false)
  const heading = useRef<HTMLHeadingElement>(null)
  const { snapshot: engine } = useEngine()
  const [starter, setStarter] = useState<FirstAnswerStarterId>(() => readFirstAnswerStarter(previouslyCompleted))

  useEffect(() => { saveOnboardingDraft({ mode: desktopMode, step: tourStep }) }, [desktopMode, tourStep])
  useEffect(() => { heading.current?.focus() }, [tourStep])
  const [providerStep, setProviderStep] = useState<ProviderSetupStep>('choice')
  const [mode, setMode] = useState<Mode>('cloud')
  const [preset, setPreset] = useState<CloudProviderPreset | LocalRuntimePreset | null>(null)
  const [apiKey, setApiKey] = useState('')
  const [localPath, setLocalPath] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [foundModels, setFoundModels] = useState<string[] | null>(null)
  const [connectedProviderName, setConnectedProviderName] = useState<string | null>(null)
  const [cliAuth, setCliAuth] = useState<{ loggedIn: boolean; binaryPresent: boolean; detail?: string; message?: string } | null>(null)

  const providers = settings?.providers ?? []
  const activeProvider = providers.find((provider) => provider.id === settings?.activeProviderId) ?? null
  const activeModel = settings?.activeModel ?? activeProvider?.models?.[0] ?? null
  // "Connected" only once the active provider passes the same readiness check the composer uses.
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  useEffect(() => {
    let live = true
    setReadiness(null)
    if (!activeProvider) return
    api.getModelReadiness({ providerId: activeProvider.id })
      .then((r) => { if (live) setReadiness(r) })
      .catch((e) => { if (live) setReadiness({ ready: false, reason: 'no-provider', message: String(e?.message ?? e) }) })
    return () => { live = false }
  }, [activeProvider?.id, activeModel, activeProvider?.models?.join('\n'), connectedProviderName])
  const connectedLabel = activeProvider && readiness?.ready && onboardingProviderVisible(desktopMode, activeProvider.id)
    ? `${activeProvider.name}${activeModel ? ` / ${engine?.installed.find((model) => model.repo === activeModel)?.title ?? (/^local\//.test(activeModel) ? t("setup.localCheckpoint") : activeModel)}` : ''}`
    : null
  const statusText = connectedLabel
    ?? (!activeProvider ? t("setup.noModel") : !readiness ? t("setup.checking", { provider: activeProvider.name }) : `${activeProvider.name}: ${readiness.message ? `${known(readiness.message) ? '' : `${t("common.diagnostic")}: `}${authored(readiness.message)}` : t("common.notReady")}`)
  const currentStep = ONBOARDING_STEPS[onboardingStepIndex(tourStep)]
  const currentStepIndex = onboardingStepIndex(tourStep)
  const selectedModel = desktopMode === 'minimal' && activeProvider?.id !== ENGINE_PROVIDER_ID ? engine?.status.model ?? null : activeModel
  const selectedModelLabel = selectedModel && (desktopMode === 'minimal' || activeProvider?.id === ENGINE_PROVIDER_ID)
    ? engine?.installed.find((model) => model.repo === selectedModel)?.title || selectedModel
    : selectedModel
  const cachedModel = engine?.installed.find((model) => model.repo === selectedModel)
  const firstAnswerReadiness: FirstAnswerReadiness = {
    providerKind: activeProvider?.kind ?? null, builtIn: activeProvider?.id === ENGINE_PROVIDER_ID || !activeProvider,
    selected: !!selectedModel, cached: !!cachedModel || activeProvider?.id === ENGINE_PROVIDER_ID && engine?.status.model === selectedModel && ['ready', 'starting'].includes(engine.status.state), fitsMemory: cachedModel?.fitsMemory ?? null, configured: !!readiness?.ready,
    loading: activeProvider?.id === ENGINE_PROVIDER_ID && engine?.status.model === selectedModel && engine.status.state === 'starting',
    loaded: activeProvider?.id === ENGINE_PROVIDER_ID && engine?.status.model === selectedModel && engine.status.state === 'ready',
  }
  const firstAnswerAction = firstAnswerNextAction(firstAnswerReadiness)
  const openStarterDraft = async () => {
    const result = await openFirstAnswerDraft({ starter, locale: settings?.uiPreferences?.locale, drafts: conversationDraftsFor(desktopDraftPort), persistent: !!desktopDraftPort,
      threadIds: () => useStore.getState().threads.map((thread) => thread.id),
      currentThread: () => { const state = useStore.getState(); return state.view.name === 'thread' ? state.threads.find((thread) => thread.id === (state.view as { id: string }).id) ?? null : null },
      newChat: () => useStore.getState().newChat(),
    })
    if (result.preservedExistingDraft) useStore.getState().setNotice({ kind: 'info', message: t("setup.draftKept") })
  }

  const complete = async () => {
    if (finishLock.current || !settings) return
    finishLock.current = true
    setFinishing(true)
    setError('')
    try {
      await finishOnboardingLanding({
        mode: desktopMode, previouslyCompleted, savedLayout: layoutBeforeSetup, preferenceReadable,
        persist: api.desktopModeSet, applySaved: (saved) => useStore.setState({ settings: saved }),
        applyLayout: (layout) => useStore.getState().setFullWorkspaceLayout(layout),
        newChat: openStarterDraft, home: () => setView({ name: 'home' }),
        chatFailed: () => useStore.getState().setNotice({ kind: 'error', message: t("setup.draftFailed") }),
      })
      if (desktopMode === 'full' && previouslyCompleted && starter !== 'blank') await openStarterDraft()
      onDone()
    } catch (e: any) {
      setError(e?.message ?? t("setup.saveFailed"))
    } finally {
      finishLock.current = false
      setFinishing(false)
    }
  }
  // Opening a setup page or selecting a model never completes the first-run flow.
  const openSettings = (section: string) => {
    if (busy || finishing) return
    saveOnboardingDraft({ mode: desktopMode, step: 'connect' })
    requestSettingsSection(section)
    setView({ name: 'settings', section })
  }
  const openModels = () => {
    if (busy || finishing) return
    saveOnboardingDraft({ mode: desktopMode, step: 'connect' })
    setView({ name: 'models' })
  }
  // Selecting a downloaded model does not start the server or allocate its model memory.
  const chooseEngineModel = async (repo: string) => {
    if (!settings || busy) return
    setBusy(true)
    setError('')
    try {
      await saveSettings({ ...settings, activeProviderId: ENGINE_PROVIDER_ID, activeProviderAccountId: null, activeModel: repo }, { silent: true })
      setConnectedProviderName('TensorFold (built in)')
      setTourStep('guide')
    } catch (e: any) { setError(e?.message ?? t("setup.selectFailed")) }
    finally { setBusy(false) }
  }
  const goToStep = (step: OnboardingStepId) => {
    if (busy || finishing) return
    setTourStep(step)
    setError('')
  }
  const goNext = () => goToStep(nextOnboardingStep(tourStep))
  const goBack = () => {
    if (tourStep === 'connect' && providerStep === 'configure') setProviderStep(mode === 'cloud' ? 'cloud' : 'local')
    else if (tourStep === 'connect' && providerStep !== 'choice') setProviderStep('choice')
    else goToStep(previousOnboardingStep(tourStep))
  }

  const pickCloud = (p: CloudProviderPreset) => {
    setPreset(p)
    setMode('cloud')
    setFoundModels(null)
    setError('')
    setCliAuth(null)
    setProviderStep('configure')
    if (p.auth === 'cli-oauth') {
      void detectCli(p.provider.transport === 'codex-cli' ? 'codex-cli' : 'claude-cli', p.provider.cliPath)
    } else {
      setApiKey('')
    }
  }

  const pickLocal = (p: LocalRuntimePreset) => {
    setPreset(p)
    setMode('local')
    setFoundModels(null)
    setError('')
    setProviderStep('configure')
    if (p.source && DISCOVERY_SOURCES.includes(p.source as ProviderDiscoveryKind)) void detect(p.source as ProviderDiscoveryKind)
  }

  const detect = async (kind: ProviderDiscoveryKind) => {
    setBusy(true)
    setError('')
    setFoundModels(null)
    try {
      // Pass the typed address so a provider on another machine is probed, not just localhost.
      const result = await api.discoverLocalProvider(kind, localPath.trim() || undefined, apiKey.trim() || undefined)
      if (result.found && result.models.length) setFoundModels(result.models)
      else setError(result.message || t("setup.noServer"))
    } catch (e: any) {
      setError(e?.message ?? t("setup.discoveryFailed"))
    } finally {
      setBusy(false)
    }
  }

  const detectCli = async (transport: 'codex-cli' | 'claude-cli', cliPath?: string) => {
    setBusy(true)
    setCliAuth(null)
    try {
      const result = await api.checkCliTransport({ transport, cliPath })
      setCliAuth(result)
    } catch (e: any) {
      setCliAuth({ loggedIn: false, binaryPresent: false, message: e?.message ?? t("setup.cliFailed") })
    } finally {
      setBusy(false)
    }
  }

  const connect = async () => {
    if (!settings || !preset) return
    setBusy(true)
    setError('')
    try {
      let provider: Provider
      if (mode === 'cloud') {
        const p = preset as CloudProviderPreset
        const isCli = p.auth === 'cli-oauth'
        const needsApiKey = providerPresetNeedsApiKey(p)
        const transport = p.provider.transport ?? (isCli ? 'codex-cli' : 'chat-completions')
        provider = {
          id: newProviderId(),
          name: p.label,
          transport,
          auth: isCli ? (transport === 'codex-cli' ? 'codex-oauth' : 'claude-oauth') : needsApiKey ? 'api-key' : p.provider.auth,
          baseUrl: isCli ? '' : p.provider.baseUrl,
          apiKey: isCli || !needsApiKey ? '' : apiKey.trim(),
          models: p.provider.models ?? [],
          kind: 'cloud',
          source: p.provider.source,
          cliPath: p.provider.cliPath || (transport === 'codex-cli' ? 'codex' : 'claude'),
          cliSandbox: p.provider.cliSandbox,
          cliApprovalPolicy: p.provider.cliApprovalPolicy,
          cliPermissionMode: p.provider.cliPermissionMode,
        }
      } else {
        const p = preset as LocalRuntimePreset
        const host = localPath.trim() || `http://127.0.0.1:${p.defaultPort}`
        const base = host.replace(/\/+$/, '') + (host.includes('/v1') ? '' : '/v1')
        provider = {
          id: newProviderId(),
          name: p.label,
          transport: 'chat-completions',
          auth: 'api-key',
          baseUrl: base,
          apiKey: apiKey.trim() || p.defaultApiKey,
          models: foundModels ?? p.defaultModels,
          kind: 'local',
          source: p.source as Provider['source'],
          ...(p.trustedNetwork ? { trustedNetwork: true as const } : {}),
        }
      }

      if (provider.transport !== 'codex-cli' && provider.transport !== 'claude-cli') {
        try {
          const models = await api.fetchModels(provider)
          if (models.length) provider.models = models
        } catch {
          // Readiness will surface fetch/auth failures; keep the preset models for now.
        }
      }

      await saveSettings({
        ...settings,
        providers: [...settings.providers, provider],
        activeProviderId: provider.id,
        activeModel: provider.models[0] ?? null,
      })
      setConnectedProviderName(provider.name)
      setProviderStep('choice')
      setTourStep('guide')
    } catch (e: any) {
      setError(e?.message ?? t("setup.providerFailed"))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div data-testid="onboarding" className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto grid min-h-full w-full max-w-[1080px] content-start gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[210px_minmax(0,1fr)] lg:gap-10 lg:py-10">
        <aside className="min-w-0 lg:sticky lg:top-10 lg:self-start">
          <div className="mb-5 flex items-center gap-2.5">
            <BrandMark size={36} tile />
            <div><div className="text-[15px] font-semibold text-ink">{"TensorFold"}</div><div className="text-[13px] text-ink-2">{t("setup.welcome")}</div></div>
          </div>
          <div className="mb-5"><LanguageQuickSelect blocked={busy || finishing} /></div>
          <nav aria-label={t("setup.progress")} className="grid grid-cols-3 gap-1 lg:grid-cols-1 lg:gap-2">
            {ONBOARDING_STEPS.map((step, index) => {
              const active = step.id === tourStep
              return <button type="button" key={step.id} data-testid={`onboarding-step-${step.id}`} aria-current={active ? 'step' : undefined} disabled={busy || finishing} onClick={() => goToStep(step.id)} className={clsx('flex min-w-0 items-center gap-2 rounded-lg px-2 py-2.5 text-left text-[12px] focus-visible:outline-2 focus-visible:outline-accent sm:text-[13px]', active ? 'bg-surface-hover text-ink' : 'text-ink-2 hover:bg-surface')}>
                <span className={clsx('flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[12px]', active ? 'bg-accent/15 text-accent' : 'bg-surface text-ink-2')}>{index + 1}</span>
                <span>{authored(step.label)}</span>
              </button>
            })}
          </nav>
          <p className="mt-5 hidden text-[12px] leading-relaxed text-ink-2 lg:block">{t("setup.noAccount")}</p>
        </aside>

        <section className="flex min-w-0 flex-col">
          <div className="mb-5">
            <p className="mb-2 text-[12px] text-ink-2">{t("setup.step", { current: currentStepIndex + 1, total: ONBOARDING_STEPS.length })}</p>
            <h1 ref={heading} tabIndex={-1} className="text-[24px] font-semibold tracking-tight text-ink outline-none sm:text-[28px]">{authored(currentStep.title)}</h1>
            <p className="mt-2 max-w-[620px] text-[14px] leading-relaxed text-ink-2">{authored(currentStep.desc)}</p>
            <p className="mt-2 text-[12px] text-ink-3">{t("setup.coverage")}</p>
          </div>

          {tourStep === 'mode' && <WorkspaceChoice value={desktopMode} onChange={(choice) => { setDesktopMode(choice); setProviderStep('choice') }} />}

          {tourStep === 'connect' && <>
            {engine?.status.hardware && <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-surface px-3 py-2 text-[12px] text-ink-2"><Icon name="monitor" size={14} /><span>{engine.status.hardware.chip || t("setup.thisMac")}</span><span>{t("setup.memory", { size: formatBytes(engine.status.totalMemoryBytes) })}</span><span>{t("setup.modelChecks")}</span></div>}
            {desktopMode === 'full' && <p role="status" data-testid="onboarding-model-status" className="mb-3 break-words text-[12px] text-ink-2">{statusText}</p>}
            <fieldset disabled={busy} className="min-w-0"><ConnectStep providerStep={providerStep} setProviderStep={setProviderStep} mode={mode} preset={preset} apiKey={apiKey} localPath={localPath} busy={busy} error={error} foundModels={foundModels} cliAuth={cliAuth} connectedLabel={connectedLabel} desktopMode={desktopMode} onPickCloud={pickCloud} onPickLocal={pickLocal} onDetect={detect} onConnect={() => void connect()} onApiKeyChange={setApiKey} onLocalPathChange={setLocalPath} onOpenModels={() => openSettings('models')} onOpenEngineModels={openModels} onUseEngineModel={(repo) => void chooseEngineModel(repo)} onContinue={goNext} /></fieldset>
            <div className="mt-4 rounded-xl border border-line px-4 py-3 text-[13px] leading-relaxed text-ink-2"><span className="font-medium text-ink">{t("setup.downloaded")}</span>{' '}{t("setup.scanFolders")}{' '}<button type="button" disabled={busy} className="text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40" onClick={() => openSettings('library')}>{t("setup.modelFolders")}</button>. {t("setup.originalFiles")}</div>
          </>}

          {tourStep === 'guide' && (desktopMode === 'minimal' ? <MinimalSetupGuide model={selectedModelLabel} /> : <FirstAnswerStep starter={starter} onStarter={(id) => { setStarter(id); saveFirstAnswerStarter(id) }} model={selectedModelLabel} readiness={firstAnswerReadiness} disabled={finishing} />)}
          {error && !(tourStep === 'connect' && providerStep === 'configure') && <div role="alert" className="mt-4 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger"><span>{t("setup.error")} </span>{!known(error) && <span>{t("common.diagnostic")}: </span>}{authored(error)}</div>}

          <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
            <Button data-testid="onboarding-back" icon="back" onClick={goBack} disabled={tourStep === 'mode' || busy || finishing}>{t("common.back")}</Button>
            {tourStep !== 'guide' ? <Button data-testid="onboarding-next" variant="primary" iconRight="arrowRight" onClick={goNext} disabled={busy}>{tourStep === 'mode' ? t("setup.continueMode", { mode: desktopMode === 'minimal' ? t("setup.minimal") : t("setup.full") }) : selectedModel ? t("common.continue") : t("setup.later")}</Button> : <div className="flex flex-wrap items-center gap-3">
              {desktopMode === 'full' && firstAnswerAction.action !== 'draft' && <button type="button" onClick={() => void complete()} disabled={finishing || !settings} className="rounded text-[13px] text-ink-2 underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-accent">{t("setup.draftLater")}</button>}
              <Button data-testid="onboarding-finish" variant="primary" iconRight="arrowRight" onClick={() => { if (desktopMode === 'full' && firstAnswerAction.action !== 'draft') { if (firstAnswerAction.action === 'models') openModels(); else openSettings('models') } else void complete() }} disabled={finishing || !settings}>{finishing ? t("common.saving") : desktopMode === 'minimal' ? t("setup.server") : firstAnswerAction.action !== 'draft' ? authored(firstAnswerAction.label) : previouslyCompleted && starter === 'blank' ? t("setup.openApp") : starter === 'blank' ? t("setup.emptyChat") : t("first.openDraft")}</Button>
            </div>}
          </div>
        </section>
      </div>
    </div>
  )
}

function WorkspaceChoice({ value, onChange }: { value: OnboardingDesktopMode; onChange: (value: OnboardingDesktopMode) => void }) {
  const { t, authored, known, locale } = useFlowText()
  const choices = [
    { value: 'minimal', title: t("setup.minimal"), icon: 'cpu', desc: t("setup.minimal.desc"), features: [t("setup.minimal.feature1"), t("setup.minimal.feature2"), t("setup.minimal.feature3")], note: t("setup.minimal.note") },
    { value: 'full', title: t("setup.full"), icon: 'monitor', desc: t("setup.full.desc"), features: [t("setup.full.feature1"), t("setup.full.feature2"), t("setup.full.feature3")], note: t("setup.full.note") },
  ] as const
  return <fieldset><legend className="sr-only">{t("setup.workspace")}</legend><div className="grid gap-3 sm:grid-cols-2">{choices.map((choice) => <label key={choice.value} className="min-w-0 cursor-pointer">
    <input type="radio" name="onboarding-workspace" value={choice.value} checked={value === choice.value} onChange={() => onChange(choice.value)} className="peer sr-only" data-testid={`onboarding-mode-${choice.value}`} />
    <span className={clsx('flex h-full flex-col rounded-xl border bg-card p-4 shadow-card transition-colors peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent sm:p-5', value === choice.value ? 'border-accent' : 'border-line hover:border-line-strong')}>
      <span className="flex items-center justify-between gap-2"><Icon name={choice.icon} size={22} className="text-ink-2" /><span aria-hidden="true" className={clsx('flex h-5 w-5 items-center justify-center rounded-full border', value === choice.value ? 'border-accent bg-accent text-accent-ink' : 'border-line-strong')}>{value === choice.value && <Icon name="check" size={12} />}</span></span>
      <span className="mt-4 text-[18px] font-semibold text-ink">{choice.title}</span><span className="mt-1 text-[13px] leading-relaxed text-ink-2">{choice.desc}</span>
      <span className="my-4 grid gap-2">{choice.features.map((feature) => <span key={feature} className="flex items-start gap-2 text-[13px] leading-snug text-ink-2"><Icon name="check" size={14} className="mt-0.5 shrink-0 text-accent" />{feature}</span>)}</span>
      <span className="mt-auto border-t border-line pt-3 text-[12px] leading-relaxed text-ink-2">{choice.note}</span>
    </span>
  </label>)}</div><p className="mt-4 text-[13px] leading-relaxed text-ink-2">{t("setup.bothModes")}</p></fieldset>
}

function ConnectStep({
  providerStep,
  setProviderStep,
  mode,
  preset,
  apiKey,
  localPath,
  busy,
  error,
  foundModels,
  cliAuth,
  connectedLabel,
  desktopMode,
  onPickCloud,
  onPickLocal,
  onDetect,
  onConnect,
  onApiKeyChange,
  onLocalPathChange,
  onOpenModels,
  onOpenEngineModels,
  onUseEngineModel,
  onContinue,
}: {
  providerStep: ProviderSetupStep
  setProviderStep: (step: ProviderSetupStep) => void
  mode: Mode
  preset: CloudProviderPreset | LocalRuntimePreset | null
  apiKey: string
  localPath: string
  busy: boolean
  error: string
  foundModels: string[] | null
  cliAuth: { loggedIn: boolean; binaryPresent: boolean; detail?: string; message?: string } | null
  connectedLabel: string | null
  desktopMode: OnboardingDesktopMode
  onPickCloud: (preset: CloudProviderPreset) => void
  onPickLocal: (preset: LocalRuntimePreset) => void
  onDetect: (kind: ProviderDiscoveryKind) => void
  onConnect: () => void
  onApiKeyChange: (value: string) => void
  onLocalPathChange: (value: string) => void
  onOpenModels: () => void
  onOpenEngineModels: () => void
  onUseEngineModel: (repo: string) => void
  onContinue: () => void
}) {
  const tx = useAuthoredText()
  const { t, authored, known, locale } = useFlowText()
  if (providerStep === 'cloud') {
    return (
      <div className="grid gap-2.5">
        <StepHeader title={t("setup.pickCloud")} onBack={() => setProviderStep('choice')} />
        {CLOUD_PROVIDER_PRESETS.map((p) => (
          <PresetRow key={p.key} testId={`onboarding-cloud-${p.key}`} brand={providerBrand(p.provider)} title={p.label} desc={authored(p.desc)} badge={p.auth === 'cli-oauth' ? 'OAuth' : p.auth === 'cli-local' ? t("setup.cliAuth") : t("setup.apiKey")} onClick={() => onPickCloud(p)} />
        ))}
      </div>
    )
  }

  if (providerStep === 'local') {
    return (
      <div className="grid gap-2.5">
        <StepHeader title={t("setup.pickLocal")} onBack={() => setProviderStep('choice')} />
        {LOCAL_RUNTIME_PRESETS.map((p) => (
          <PresetRow key={p.key} testId={`onboarding-local-${p.key}`} icon="cpu" title={p.label} desc={authored(p.desc)} badge={`:${p.defaultPort}`} onClick={() => onPickLocal(p)} />
        ))}
      </div>
    )
  }

  if (providerStep === 'configure' && preset) {
    return (
      <div className="max-w-[720px]">
        <StepHeader title={t("setup.configure", { provider: preset.label })} onBack={() => setProviderStep(mode === 'cloud' ? 'cloud' : 'local')} />
        <div className="rounded-xl border border-line bg-card p-4 shadow-card">
          {mode === 'cloud' && (preset as CloudProviderPreset).auth === 'cli-oauth' ? (
            <div className="space-y-3">
              {busy && !cliAuth && <div className="flex items-center gap-2 text-[13px] text-ink-3"><Icon name="refresh" size={13} className="animate-spin" /> {t("setup.checkLogin", { cli: preset.label.includes('Codex') ? 'codex' : 'claude' })}</div>}
              {cliAuth && (
                <div className={clsx('rounded-lg border px-3 py-2.5 text-[13px]', cliAuth.loggedIn ? 'border-positive/30 bg-positive/10 text-positive' : cliAuth.binaryPresent ? 'border-warn/30 bg-warn/10 text-warn' : 'border-danger/30 bg-danger/10 text-danger')}>
                  <div className="flex items-center gap-2">
                    <Icon name={cliAuth.loggedIn ? 'check' : 'warning'} size={14} />
                    <span className="font-medium">{cliAuth.loggedIn ? `${t("setup.loggedIn")}${cliAuth.detail ? ` - ${cliAuth.detail}` : ''}` : cliAuth.binaryPresent ? t("setup.notLoggedIn") : t("setup.cliMissing")}</span>
                  </div>
                  {!cliAuth.loggedIn && cliAuth.binaryPresent && <div className="mt-1 text-ink-3">{t("setup.runCli")}{' '}<code className="rounded bg-card-2 px-1 py-0.5 font-mono text-[12px]">{preset.label.includes('Codex') ? 'codex login' : 'claude auth login'}</code>{' '}{t("setup.cliTerminal")}</div>}
                  {!cliAuth.binaryPresent && <div className="mt-1 text-ink-3">{t("common.diagnostic")}: {cliAuth.message}</div>}
                </div>
              )}
              <p className="text-[13px] leading-relaxed text-ink-3">{authored((preset as CloudProviderPreset).desc)}</p>
            </div>
          ) : mode === 'cloud' && (preset as CloudProviderPreset).auth === 'cli-local' ? (
            <div className="space-y-3">
              <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] leading-relaxed text-ink-3">{t("setup.piStart")}{' '}<code className="rounded bg-card px-1 py-0.5 font-mono text-[12px] text-ink-2">pi</code>{' '}{t("setup.piConfig")}{' '}<code className="rounded bg-card px-1 py-0.5 font-mono text-[12px] text-ink-2">OPENAI_API_KEY</code>{' '}{t("setup.piEnd")}</div>
              <p className="text-[13px] leading-relaxed text-ink-3">{(preset as CloudProviderPreset).desc}</p>
            </div>
          ) : mode === 'cloud' ? (
            <Field label={t("setup.apiKey")}>
              <Input type="password" value={apiKey} onChange={(e) => onApiKeyChange(e.target.value)} placeholder={"sk-..."} autoFocus spellCheck={false} />
            </Field>
          ) : (
            <div className="space-y-3">
              <Field label={t("setup.address")} hint={t("setup.override")}>
                <Input value={localPath} onChange={(e) => onLocalPathChange(e.target.value)} placeholder={`http://127.0.0.1:${(preset as LocalRuntimePreset).defaultPort}`} spellCheck={false} />
              </Field>
              {(() => {
                const src = (preset as LocalRuntimePreset).source
                if (!src || !DISCOVERY_SOURCES.includes(src as ProviderDiscoveryKind)) return null
                return (
                  <div className="flex items-center justify-between gap-3 rounded-lg border border-line bg-card-2 px-3 py-2">
                    <div className="min-w-0">
                      <div className="text-[13px] font-medium text-ink-2">{foundModels ? t(foundModels.length === 1 ? 'setup.foundModel' : 'setup.foundModels', { count: foundModels.length }) : t("setup.detectServer")}</div>
                      {foundModels && <div className="mt-0.5 truncate text-[13px] text-ink-3">{foundModels.slice(0, 4).join(', ')}{foundModels.length > 4 ? '...' : ''}</div>}
                    </div>
                    <Button variant="default" size="sm" icon="refresh" onClick={() => onDetect(src as ProviderDiscoveryKind)} disabled={busy}>{busy ? t("setup.detecting") : t("setup.detect")}</Button>
                  </div>
                )
              })()}
            </div>
          )}
          {error && <div className="mt-3 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger"><span>{t("setup.error")} </span>{!known(error) && <span>{t("common.diagnostic")}: </span>}{authored(error)}</div>}
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
          <button type="button" onClick={onContinue} className="text-[13px] text-ink-3 transition-colors hover:text-ink">{t("setup.withoutConnection")}</button>
          <Button variant="primary" icon="check" onClick={onConnect} disabled={busy || (mode === 'cloud' && providerPresetNeedsApiKey(preset as CloudProviderPreset) && !apiKey.trim())}>
            {busy ? t("setup.connecting") : t("setup.connectModel")}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="grid gap-4">
      {connectedLabel && (
        <div className="rounded-xl border border-positive/25 bg-positive/10 px-4 py-3 text-[13px] text-positive">
          <div className="flex items-center gap-2 font-medium"><Icon name="check" size={15} />{t("setup.connected")}</div>
          <div className="mt-1 text-ink-3">{connectedLabel}</div>
        </div>
      )}
      <EngineChoice minimal={desktopMode === 'minimal'} onOpenModels={onOpenEngineModels} onUse={onUseEngineModel} />
      {desktopMode === 'full' && <details className="rounded-xl border border-line p-4"><summary className="cursor-pointer text-[13px] font-medium text-ink focus-visible:outline-2 focus-visible:outline-accent">{t("setup.otherProvider")}</summary><div className="mt-3 grid gap-3 sm:grid-cols-2">
        <ChoiceCard
          testId="onboarding-provider-cloud"
          icon={<Icon name="globe" size={20} />}
          title={t("setup.cloud")}
          desc={t("setup.cloudDesc")}
          onClick={() => setProviderStep('cloud')}
        />
        <ChoiceCard
          testId="onboarding-provider-local"
          icon={<Icon name="cpu" size={20} />}
          title={t("setup.local")}
          desc={t("setup.localDesc")}
          onClick={() => setProviderStep('local')}
        />
      </div></details>}
      <div className="flex flex-wrap items-center gap-3">
        {desktopMode === 'full' && <Button variant="default" icon="settings" onClick={onOpenModels}>{t("setup.providerSettings")}</Button>}
        <button type="button" onClick={onContinue} className="text-[13px] text-ink-3 transition-colors hover:text-ink">
          {connectedLabel ? t("common.continue") : t("setup.withoutModel")}
        </button>
      </div>
    </div>
  )
}

function StepHeader({ title, onBack }: { title: string; onBack: () => void }) {
  const { t, authored, known, locale } = useFlowText()
  return (
    <div className="mb-3 flex items-center gap-2">
      <button type="button" onClick={onBack} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink" aria-label={t("common.back")}>
        <Icon name="back" size={16} />
      </button>
      <h2 className="text-[17px] font-medium tracking-tight text-ink">{title}</h2>
    </div>
  )
}

function ChoiceCard({ icon, title, desc, onClick, testId }: { icon: ReactNode; title: string; desc: string; onClick: () => void; testId?: string }) {
  return (
    <button type="button" data-testid={testId} onClick={onClick} className="group flex min-h-[150px] flex-col items-start gap-2 rounded-xl border border-line bg-card p-5 text-left shadow-card transition-colors hover:border-line-strong">
      <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-surface text-ink-2">{icon}</span>
      <span className="text-[15px] font-medium text-ink">{title}</span>
      <span className="text-[13px] leading-snug text-ink-3">{desc}</span>
    </button>
  )
}

function PresetRow({
  icon,
  brand,
  title,
  desc,
  badge,
  onClick,
  testId,
}: {
  icon?: IconName
  brand?: Parameters<typeof BrandIcon>[0]['brand']
  title: string
  desc: string
  badge?: string
  onClick: () => void
  testId?: string
}) {
  return (
    <button type="button" data-testid={testId} onClick={onClick} className="group flex w-full items-center gap-3 rounded-xl border border-line bg-card px-3.5 py-3 text-left shadow-card transition-colors hover:border-line-strong">
      {brand ? <BrandIcon brand={brand} size={32} /> : <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-surface text-ink-3"><Icon name={icon!} size={16} /></span>}
      <span className="min-w-0 flex-1">
        <span className="block text-[14px] font-medium text-ink">{title}</span>
        <span className="mt-0.5 block truncate text-[13px] text-ink-3">{desc}</span>
      </span>
      {badge && <span className="shrink-0 rounded-md bg-surface px-2 py-0.5 text-[12px] text-ink-3">{badge}</span>}
      <Icon name="chevronRight" size={14} className="shrink-0 text-ink-4 transition-transform group-hover:translate-x-0.5" />
    </button>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-[13px] font-medium text-ink-2">{label}</span>
        {hint && <span className="text-[12px] text-ink-4">{hint}</span>}
      </div>
      {children}
    </label>
  )
}
