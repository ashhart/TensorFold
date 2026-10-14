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
  commitOnboarding,
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

type ProviderSetupStep = 'choice' | 'cloud' | 'local' | 'configure'
type Mode = 'cloud' | 'local'

const DISCOVERY_SOURCES: ProviderDiscoveryKind[] = ['omlx', 'ollama', 'lmstudio']

export function Onboarding({ onDone }: { onDone: () => void }) {
  const settings = useStore((s) => s.settings)
  const saveSettings = useStore((s) => s.saveSettings)
  const setView = useStore((s) => s.setView)
  const initialDraft = useRef(readOnboardingDraft(settings?.desktopMode ?? 'full')).current
  const [desktopMode, setDesktopMode] = useState<OnboardingDesktopMode>(initialDraft.mode)
  const [tourStep, setTourStep] = useState<OnboardingStepId>(initialDraft.step)
  const [finishing, setFinishing] = useState(false)
  const finishLock = useRef(false)
  const heading = useRef<HTMLHeadingElement>(null)
  const { snapshot: engine } = useEngine()

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
    ? `${activeProvider.name}${activeModel ? ` / ${engine?.installed.find((model) => model.repo === activeModel)?.title ?? (/^local\//.test(activeModel) ? 'Local checkpoint' : activeModel)}` : ''}`
    : null
  const statusText = connectedLabel
    ?? (!activeProvider ? 'No model connected yet' : !readiness ? `Checking ${activeProvider.name}…` : `${activeProvider.name}: ${readiness.message || 'not ready'}`)
  const currentStep = ONBOARDING_STEPS[onboardingStepIndex(tourStep)]
  const currentStepIndex = onboardingStepIndex(tourStep)
  const selectedModel = desktopMode === 'minimal' && activeProvider?.id !== ENGINE_PROVIDER_ID ? engine?.status.model ?? null : activeModel
  const selectedModelLabel = selectedModel && (desktopMode === 'minimal' || activeProvider?.id === ENGINE_PROVIDER_ID)
    ? engine?.installed.find((model) => model.repo === selectedModel)?.title || selectedModel
    : selectedModel

  const complete = async () => {
    if (finishLock.current || !settings) return
    finishLock.current = true
    setFinishing(true)
    setError('')
    try {
      await commitOnboarding(desktopMode, api.desktopModeSet, (saved) => useStore.setState({ settings: saved }))
      setView({ name: 'home' })
      onDone()
    } catch (e: any) {
      setError(e?.message ?? 'Could not save your workspace choice. Try again.')
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
    } catch (e: any) { setError(e?.message ?? 'Could not select this model. Try again.') }
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
      const result = await api.discoverLocalProvider(kind)
      if (result.found && result.models.length) setFoundModels(result.models)
      else setError(result.message || 'No local server found. Start it, then retry.')
    } catch (e: any) {
      setError(e?.message ?? 'Discovery failed')
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
      setCliAuth({ loggedIn: false, binaryPresent: false, message: e?.message ?? 'Could not check CLI auth.' })
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
      setError(e?.message ?? 'Could not save the provider.')
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
            <div><div className="text-[15px] font-semibold text-ink">TensorFold</div><div className="text-[13px] text-ink-2">Let's get you set up</div></div>
          </div>
          <nav aria-label="Setup progress" className="grid grid-cols-3 gap-1 lg:grid-cols-1 lg:gap-2">
            {ONBOARDING_STEPS.map((step, index) => {
              const active = step.id === tourStep
              return <button type="button" key={step.id} data-testid={`onboarding-step-${step.id}`} aria-current={active ? 'step' : undefined} disabled={busy || finishing} onClick={() => goToStep(step.id)} className={clsx('flex min-w-0 items-center gap-2 rounded-lg px-2 py-2.5 text-left text-[12px] focus-visible:outline-2 focus-visible:outline-accent sm:text-[13px]', active ? 'bg-surface-hover text-ink' : 'text-ink-2 hover:bg-surface')}>
                <span className={clsx('flex h-6 w-6 shrink-0 items-center justify-center rounded-md text-[12px]', active ? 'bg-accent/15 text-accent' : 'bg-surface text-ink-2')}>{index + 1}</span>
                <span>{step.label}</span>
              </button>
            })}
          </nav>
          <p className="mt-5 hidden text-[12px] leading-relaxed text-ink-2 lg:block">No account is needed for the built-in local engine.</p>
        </aside>

        <section className="flex min-w-0 flex-col">
          <div className="mb-5">
            <p className="mb-2 text-[12px] text-ink-2">Step {currentStepIndex + 1} of {ONBOARDING_STEPS.length}</p>
            <h1 ref={heading} tabIndex={-1} className="text-[24px] font-semibold tracking-tight text-ink outline-none sm:text-[28px]">{currentStep.title}</h1>
            <p className="mt-2 max-w-[620px] text-[14px] leading-relaxed text-ink-2">{currentStep.desc}</p>
          </div>

          {tourStep === 'mode' && <WorkspaceChoice value={desktopMode} onChange={(choice) => { setDesktopMode(choice); setProviderStep('choice') }} />}

          {tourStep === 'connect' && <>
            {engine?.status.hardware && <div className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-1 rounded-lg bg-surface px-3 py-2 text-[12px] text-ink-2"><Icon name="monitor" size={14} /><span>{engine.status.hardware.chip || 'This Mac'}</span><span>{formatBytes(engine.status.totalMemoryBytes)} total memory</span><span>Models shows compatibility, memory and download space checks.</span></div>}
            {desktopMode === 'full' && <p role="status" data-testid="onboarding-model-status" className="mb-3 break-words text-[12px] text-ink-2">{statusText}</p>}
            <fieldset disabled={busy} className="min-w-0"><ConnectStep providerStep={providerStep} setProviderStep={setProviderStep} mode={mode} preset={preset} apiKey={apiKey} localPath={localPath} busy={busy} error={error} foundModels={foundModels} cliAuth={cliAuth} connectedLabel={connectedLabel} desktopMode={desktopMode} onPickCloud={pickCloud} onPickLocal={pickLocal} onDetect={detect} onConnect={() => void connect()} onApiKeyChange={setApiKey} onLocalPathChange={setLocalPath} onOpenModels={() => openSettings('models')} onOpenEngineModels={openModels} onUseEngineModel={(repo) => void chooseEngineModel(repo)} onContinue={goNext} /></fieldset>
            <div className="mt-4 rounded-xl border border-line px-4 py-3 text-[13px] leading-relaxed text-ink-2"><span className="font-medium text-ink">Already downloaded models?</span> Scan your Hugging Face cache or add model folders in <button type="button" disabled={busy} className="text-accent underline-offset-2 hover:underline focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40" onClick={() => openSettings('library')}>Settings → Model folders</button>. Supported models appear with a Load action. Your original files stay where they are.</div>
          </>}

          {tourStep === 'guide' && <QuickGuide mode={desktopMode} model={selectedModelLabel} />}
          {error && !(tourStep === 'connect' && providerStep === 'configure') && <div role="alert" className="mt-4 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger">{error}</div>}

          <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
            <Button data-testid="onboarding-back" icon="back" onClick={goBack} disabled={tourStep === 'mode' || busy || finishing}>Back</Button>
            {tourStep !== 'guide' ? <Button data-testid="onboarding-next" variant="primary" iconRight="arrowRight" onClick={goNext} disabled={busy}>{tourStep === 'mode' ? `Continue with ${desktopMode === 'minimal' ? 'Minimal' : 'Full'}` : selectedModel ? 'Continue' : 'Set up a model later'}</Button> : <Button data-testid="onboarding-finish" variant="primary" iconRight="arrowRight" onClick={() => void complete()} disabled={finishing || !settings}>{finishing ? 'Saving…' : desktopMode === 'minimal' ? 'Open server setup' : 'Open TensorFold'}</Button>}
          </div>
        </section>
      </div>
    </div>
  )
}

function WorkspaceChoice({ value, onChange }: { value: OnboardingDesktopMode; onChange: (value: OnboardingDesktopMode) => void }) {
  const choices = [
    { value: 'minimal', title: 'Minimal', icon: 'cpu', desc: 'A local model server, ready for your other apps.', features: ['Download and load models', 'Copy the local API URL and key', 'Run TensorFold in the menu bar'], note: 'Open the small server window whenever you need it.' },
    { value: 'full', title: 'Full', icon: 'monitor', desc: 'Your complete workspace for local AI.', features: ['Chat and compare model responses', 'Develop with projects and workers', 'Use Assistant, voice and connected tools'], note: 'Includes the same local model server and API.' },
  ] as const
  return <fieldset><legend className="sr-only">Choose your TensorFold workspace</legend><div className="grid gap-3 sm:grid-cols-2">{choices.map((choice) => <label key={choice.value} className="min-w-0 cursor-pointer">
    <input type="radio" name="onboarding-workspace" value={choice.value} checked={value === choice.value} onChange={() => onChange(choice.value)} className="peer sr-only" data-testid={`onboarding-mode-${choice.value}`} />
    <span className={clsx('flex h-full flex-col rounded-xl border bg-card p-4 shadow-card transition-colors peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-accent sm:p-5', value === choice.value ? 'border-accent' : 'border-line hover:border-line-strong')}>
      <span className="flex items-center justify-between gap-2"><Icon name={choice.icon} size={22} className="text-ink-2" /><span aria-hidden="true" className={clsx('flex h-5 w-5 items-center justify-center rounded-full border', value === choice.value ? 'border-accent bg-accent text-accent-ink' : 'border-line-strong')}>{value === choice.value && <Icon name="check" size={12} />}</span></span>
      <span className="mt-4 text-[18px] font-semibold text-ink">{choice.title}</span><span className="mt-1 text-[13px] leading-relaxed text-ink-2">{choice.desc}</span>
      <span className="my-4 grid gap-2">{choice.features.map((feature) => <span key={feature} className="flex items-start gap-2 text-[13px] leading-snug text-ink-2"><Icon name="check" size={14} className="mt-0.5 shrink-0 text-accent" />{feature}</span>)}</span>
      <span className="mt-auto border-t border-line pt-3 text-[12px] leading-relaxed text-ink-2">{choice.note}</span>
    </span>
  </label>)}</div><p className="mt-4 text-[13px] leading-relaxed text-ink-2">Both modes use the same models and settings. Choosing a mode does not download anything or start a model.</p></fieldset>
}

function QuickGuide({ mode, model }: { mode: OnboardingDesktopMode; model: string | null }) {
  const pointers = mode === 'minimal' ? [
    { icon: 'cpu', title: 'Load a model', body: 'Use Models to download a recommendation or load a supported model already on this Mac. Loading uses memory until you unload it.' },
    { icon: 'plug', title: 'Connect your apps', body: 'Starting a model creates a key for that server session. In API connection, use Copy URL and Copy API key to connect your client. Copy fresh details after restarting the model.' },
    { icon: 'coffee', title: 'Run in the menu bar', body: 'Use Run in menu bar after setup. Click the TensorFold icon to check the server, change models or open the full workspace. Quit stops the server.' },
  ] : [
    { icon: 'message', title: 'Chat', body: 'Start a conversation, choose a model beside the composer, and use Stop if an answer goes off track. Models has downloads, hardware checks and the local API.' },
    { icon: 'code', title: 'Code', body: 'Open a project to work with files, terminals and coding workers. Review requested actions before approving them; approval settings control what can run.' },
    { icon: 'spark', title: 'Assistant', body: 'Use voice, skills and connected tools. Add only the connections you need in Settings, then check Readiness before starting automation.' },
  ]
  return <div className="grid gap-4">
    <div className="rounded-xl border border-line bg-card px-4 py-3 text-[13px] text-ink-2"><span className="font-medium text-ink">{mode === 'minimal' ? 'Minimal mode' : 'Full workspace'}</span><div className="mt-1 break-words">{model ? `Selected model: ${model}` : 'No model selected yet. You can download or load one after setup.'}</div></div>
    <div className="rounded-xl border border-line bg-card shadow-card">{pointers.map((pointer, index) => <div key={pointer.title} className={clsx('flex gap-3 p-4', index > 0 && 'border-t border-line')}><Icon name={pointer.icon} size={18} className="mt-0.5 shrink-0 text-accent" /><div><h2 className="text-[14px] font-medium text-ink">{pointer.title}</h2><p className="mt-1 text-[13px] leading-relaxed text-ink-2">{pointer.body}</p></div></div>)}</div>
    <p className="text-[13px] leading-relaxed text-ink-2">Downloads and updates need an internet connection. Built-in local inference runs on this Mac once its model is installed. The authenticated local API stays available while TensorFold and its model are running. Cloud providers send prompts to their service.</p>
    <details className="rounded-xl border border-line bg-card p-4"><summary className="cursor-pointer text-[13px] font-medium text-ink focus-visible:outline-2 focus-visible:outline-accent">Optional AI tools and your data</summary><div className="mt-3 grid gap-2 text-[13px] leading-relaxed text-ink-2"><p>Cortheon can send search queries based on your chat to a web search service. Control it in AI settings.</p><p>Living Weights is experimental learning. Review the facts you choose to teach; learned recall can be wrong.</p><p>Share local API keys only with apps you trust. Unload a model to free its memory, or quit TensorFold to stop the server.</p></div></details>
    <p className="text-[12px] text-ink-2">You can reopen this guide in Settings.</p>
  </div>
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
  if (providerStep === 'cloud') {
    return (
      <div className="grid gap-2.5">
        <StepHeader title="Pick a cloud provider" onBack={() => setProviderStep('choice')} />
        {CLOUD_PROVIDER_PRESETS.map((p) => (
          <PresetRow key={p.key} testId={`onboarding-cloud-${p.key}`} brand={providerBrand(p.provider)} title={p.label} desc={p.desc} badge={p.auth === 'cli-oauth' ? 'OAuth' : p.auth === 'cli-local' ? 'CLI auth' : 'API key'} onClick={() => onPickCloud(p)} />
        ))}
      </div>
    )
  }

  if (providerStep === 'local') {
    return (
      <div className="grid gap-2.5">
        <StepHeader title="Pick a local runtime" onBack={() => setProviderStep('choice')} />
        {LOCAL_RUNTIME_PRESETS.map((p) => (
          <PresetRow key={p.key} testId={`onboarding-local-${p.key}`} icon="cpu" title={p.label} desc={p.desc} badge={`:${p.defaultPort}`} onClick={() => onPickLocal(p)} />
        ))}
      </div>
    )
  }

  if (providerStep === 'configure' && preset) {
    return (
      <div className="max-w-[720px]">
        <StepHeader title={`Configure ${preset.label}`} onBack={() => setProviderStep(mode === 'cloud' ? 'cloud' : 'local')} />
        <div className="rounded-xl border border-line bg-card p-4 shadow-card">
          {mode === 'cloud' && (preset as CloudProviderPreset).auth === 'cli-oauth' ? (
            <div className="space-y-3">
              {busy && !cliAuth && <div className="flex items-center gap-2 text-[13px] text-ink-3"><Icon name="refresh" size={13} className="animate-spin" /> Checking {preset.label.includes('Codex') ? 'codex' : 'claude'} login...</div>}
              {cliAuth && (
                <div className={clsx('rounded-lg border px-3 py-2.5 text-[13px]', cliAuth.loggedIn ? 'border-positive/30 bg-positive/10 text-positive' : cliAuth.binaryPresent ? 'border-warn/30 bg-warn/10 text-warn' : 'border-danger/30 bg-danger/10 text-danger')}>
                  <div className="flex items-center gap-2">
                    <Icon name={cliAuth.loggedIn ? 'check' : 'warning'} size={14} />
                    <span className="font-medium">{cliAuth.loggedIn ? `Already logged in${cliAuth.detail ? ` - ${cliAuth.detail}` : ''}` : cliAuth.binaryPresent ? 'Not logged in' : 'CLI not found'}</span>
                  </div>
                  {!cliAuth.loggedIn && cliAuth.binaryPresent && <div className="mt-1 text-ink-3">Run <code className="rounded bg-card-2 px-1 py-0.5 font-mono text-[12px]">{preset.label.includes('Codex') ? 'codex login' : 'claude auth login'}</code> in Terminal, or connect now and fix it in Readiness.</div>}
                  {!cliAuth.binaryPresent && <div className="mt-1 text-ink-3">{cliAuth.message}</div>}
                </div>
              )}
              <p className="text-[13px] leading-relaxed text-ink-3">{(preset as CloudProviderPreset).desc}</p>
            </div>
          ) : mode === 'cloud' && (preset as CloudProviderPreset).auth === 'cli-local' ? (
            <div className="space-y-3">
              <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] leading-relaxed text-ink-3">
                Launch <code className="rounded bg-card px-1 py-0.5 font-mono text-[12px] text-ink-2">pi</code> once and configure its provider/auth, or make a documented provider environment key such as <code className="rounded bg-card px-1 py-0.5 font-mono text-[12px] text-ink-2">OPENAI_API_KEY</code> available to TensorFold. A generic key is not requested because safely selecting its provider environment variable cannot be inferred.
              </div>
              <p className="text-[13px] leading-relaxed text-ink-3">{(preset as CloudProviderPreset).desc}</p>
            </div>
          ) : mode === 'cloud' ? (
            <Field label="API key">
              <Input type="password" value={apiKey} onChange={(e) => onApiKeyChange(e.target.value)} placeholder="sk-..." autoFocus spellCheck={false} />
            </Field>
          ) : (
            <div className="space-y-3">
              <Field label="Server address" hint="optional override">
                <Input value={localPath} onChange={(e) => onLocalPathChange(e.target.value)} placeholder={`http://127.0.0.1:${(preset as LocalRuntimePreset).defaultPort}`} spellCheck={false} />
              </Field>
              {(() => {
                const src = (preset as LocalRuntimePreset).source
                if (!src || !DISCOVERY_SOURCES.includes(src as ProviderDiscoveryKind)) return null
                return (
                  <div className="flex items-center justify-between gap-3 rounded-lg border border-line bg-card-2 px-3 py-2">
                    <div className="min-w-0">
                      <div className="text-[13px] font-medium text-ink-2">{foundModels ? `${foundModels.length} model${foundModels.length === 1 ? '' : 's'} found` : 'Detect running server'}</div>
                      {foundModels && <div className="mt-0.5 truncate text-[13px] text-ink-3">{foundModels.slice(0, 4).join(', ')}{foundModels.length > 4 ? '...' : ''}</div>}
                    </div>
                    <Button variant="default" size="sm" icon="refresh" onClick={() => onDetect(src as ProviderDiscoveryKind)} disabled={busy}>{busy ? 'Detecting' : 'Detect'}</Button>
                  </div>
                )
              })()}
            </div>
          )}
          {error && <div className="mt-3 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger">{error}</div>}
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-3">
          <button type="button" onClick={onContinue} className="text-[13px] text-ink-3 transition-colors hover:text-ink">Continue without connecting</button>
          <Button variant="primary" icon="check" onClick={onConnect} disabled={busy || (mode === 'cloud' && providerPresetNeedsApiKey(preset as CloudProviderPreset) && !apiKey.trim())}>
            {busy ? 'Connecting...' : 'Connect model'}
          </Button>
        </div>
      </div>
    )
  }

  return (
    <div className="grid gap-4">
      {connectedLabel && (
        <div className="rounded-xl border border-positive/25 bg-positive/10 px-4 py-3 text-[13px] text-positive">
          <div className="flex items-center gap-2 font-medium"><Icon name="check" size={15} /> Connected</div>
          <div className="mt-1 text-ink-3">{connectedLabel}</div>
        </div>
      )}
      <EngineChoice minimal={desktopMode === 'minimal'} onOpenModels={onOpenEngineModels} onUse={onUseEngineModel} />
      {desktopMode === 'full' && <details className="rounded-xl border border-line p-4"><summary className="cursor-pointer text-[13px] font-medium text-ink focus-visible:outline-2 focus-visible:outline-accent">Use a cloud provider or another local runtime</summary><div className="mt-3 grid gap-3 sm:grid-cols-2">
        <ChoiceCard
          testId="onboarding-provider-cloud"
          icon={<Icon name="globe" size={20} />}
          title="Cloud"
          desc="OpenAI, Anthropic, OpenRouter, z.ai, Groq, Codex CLI, or Claude CLI."
          onClick={() => setProviderStep('cloud')}
        />
        <ChoiceCard
          testId="onboarding-provider-local"
          icon={<Icon name="cpu" size={20} />}
          title="Local"
          desc="oMLX, Ollama, LM Studio, or another OpenAI-compatible runtime."
          onClick={() => setProviderStep('local')}
        />
      </div></details>}
      <div className="flex flex-wrap items-center gap-3">
        {desktopMode === 'full' && <Button variant="default" icon="settings" onClick={onOpenModels}>Provider settings</Button>}
        <button type="button" onClick={onContinue} className="text-[13px] text-ink-3 transition-colors hover:text-ink">
          {connectedLabel ? 'Continue' : 'Continue without a model'}
        </button>
      </div>
    </div>
  )
}

function StepHeader({ title, onBack }: { title: string; onBack: () => void }) {
  return (
    <div className="mb-3 flex items-center gap-2">
      <button type="button" onClick={onBack} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-surface-hover hover:text-ink" aria-label="Back">
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
