import { useEffect, useState, type ReactNode } from 'react'
import clsx from 'clsx'
import { requestSettingsSection, useStore, type View } from '../lib/store'
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
  ONBOARDED_KEY,
  ONBOARDING_STEPS,
  nextOnboardingStep,
  onboardingStepIndex,
  previousOnboardingStep,
  type OnboardingStepId,
} from './onboarding-model'
export { ONBOARDED_KEY, ONBOARDING_STEPS, nextOnboardingStep, onboardingStepIndex, previousOnboardingStep, shouldShowOnboarding } from './onboarding-model'
import type { Provider, ProviderDiscoveryKind, Readiness } from '../lib/types'
import { EngineChoice } from './models/EngineChoice'
import { BrandMark } from './BrandMark'
import { ENGINE_PROVIDER_ID } from '../lib/engine'

type ProviderSetupStep = 'choice' | 'cloud' | 'local' | 'configure'
type Mode = 'cloud' | 'local'

const DISCOVERY_SOURCES: ProviderDiscoveryKind[] = ['omlx', 'ollama', 'lmstudio']

function markOnboarded() {
  try { localStorage.setItem(ONBOARDED_KEY, '1') } catch { /* ignore */ }
}

export function Onboarding({ onDone }: { onDone: () => void }) {
  const settings = useStore((s) => s.settings)
  const saveSettings = useStore((s) => s.saveSettings)
  const setView = useStore((s) => s.setView)
  const newChat = useStore((s) => s.newChat)
  const openNewProject = useStore((s) => s.openNewProject)

  const [tourStep, setTourStep] = useState<OnboardingStepId>('connect')
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
  const connectedLabel = activeProvider && readiness?.ready
    ? `${activeProvider.name}${activeModel ? ` / ${activeModel}` : ''}`
    : null
  const statusText = connectedLabel
    ?? (!activeProvider ? 'No model connected yet' : !readiness ? `Checking ${activeProvider.name}…` : `${activeProvider.name}: ${readiness.message || 'not ready'}`)
  const currentStep = ONBOARDING_STEPS[onboardingStepIndex(tourStep)]
  const currentStepIndex = onboardingStepIndex(tourStep)

  const complete = (after?: () => void) => {
    markOnboarded()
    onDone()
    after?.()
  }
  const openView = (view: View) => complete(() => setView(view))
  const openSettings = (section: string) => complete(() => {
    requestSettingsSection(section)
    setView({ name: 'settings', section })
  })
  const openChat = () => complete(() => { void newChat() })
  // Choosing a downloaded engine model only makes it active; the engine starts on the first message.
  const chooseEngineModel = async (repo: string) => {
    if (!settings) return
    await saveSettings({ ...settings, activeProviderId: ENGINE_PROVIDER_ID, activeProviderAccountId: null, activeModel: repo }, { silent: true })
    setConnectedProviderName('TensorFold (built in)')
    goToStep('chat')
  }
  const openProjectModal = () => complete(() => {
    setView({ name: 'projects' })
    openNewProject()
  })

  const goToStep = (step: OnboardingStepId) => {
    setTourStep(step)
    setError('')
  }
  const goNext = () => goToStep(nextOnboardingStep(tourStep))
  const goBack = () => {
    if (tourStep !== 'connect') goToStep(previousOnboardingStep(tourStep))
    else if (providerStep === 'configure') setProviderStep(mode === 'cloud' ? 'cloud' : 'local')
    else if (providerStep !== 'choice') setProviderStep('choice')
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
      goToStep('chat')
    } catch (e: any) {
      setError(e?.message ?? 'Could not save the provider.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div data-testid="onboarding" className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto grid min-h-full w-full max-w-[1160px] gap-6 px-4 py-6 sm:px-6 lg:grid-cols-[280px_minmax(0,1fr)] lg:py-8">
        <aside className="flex min-w-0 flex-col gap-5 lg:sticky lg:top-8 lg:h-[calc(100vh-64px)]">
          <div className="flex items-center gap-2.5">
            <BrandMark size={36} tile />
            <div className="min-w-0">
              <div className="text-[15px] font-semibold text-ink">TensorFold</div>
              <div className="text-[13px] text-ink-3">First start</div>
            </div>
          </div>

          <div className="flex flex-col gap-0.5">
            {ONBOARDING_STEPS.map((step, index) => {
              const active = step.id === tourStep
              const done = index < currentStepIndex
              return (
                <button
                  type="button"
                  key={step.id}
                  data-testid={`onboarding-step-${step.id}`}
                  onClick={() => goToStep(step.id)}
                  className={clsx(
                    'flex w-full items-center gap-2.5 rounded-[10px] px-2.5 py-2 text-left transition-colors',
                    active ? 'bg-surface-hover text-ink' : 'text-ink-2 hover:bg-surface hover:text-ink',
                  )}
                >
                  <span className={clsx('flex h-7 w-7 shrink-0 items-center justify-center rounded-md', active ? 'bg-accent/15 text-accent' : done ? 'bg-positive/10 text-positive' : 'bg-surface text-ink-3')}>
                    <Icon name={done ? 'check' : step.icon} size={14} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[13px] font-medium">{step.label}</span>
                    <span className="block truncate text-[13px] text-ink-3">{step.title}</span>
                  </span>
                </button>
              )
            })}
          </div>

          <div className="rounded-xl border border-line bg-card p-3 shadow-card">
            <div className="flex items-center gap-2 text-[13px] font-medium text-ink-2">
              <Icon name={connectedLabel ? 'check' : 'warning'} size={14} className={connectedLabel ? 'text-positive' : 'text-warn'} />
              Model status
            </div>
            <div data-testid="onboarding-model-status" className="mt-1.5 line-clamp-2 text-[13px] text-ink-3">{statusText}</div>
            <div className="mt-3 flex flex-wrap gap-2">
              <Button size="xs" variant="default" icon="cpu" onClick={() => setView({ name: 'models' })}>Models</Button>
              <Button size="xs" variant="default" icon="settings" onClick={() => openSettings('models')}>Providers</Button>
              <Button size="xs" variant="default" icon="check" onClick={() => openSettings('readiness')}>Doctor</Button>
            </div>
          </div>

          <button type="button" onClick={() => complete()} className="self-start text-[13px] text-ink-3 transition-colors hover:text-ink">
            Skip onboarding
          </button>
        </aside>

        <section className="flex min-w-0 flex-col">
          <div className="mb-5 flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div className="min-w-0">
              <div className="mb-2 flex items-center gap-2 text-[13px] text-ink-3">
                Step {currentStepIndex + 1} of {ONBOARDING_STEPS.length}
              </div>
              <h1 className="text-[28px] font-semibold tracking-normal text-ink">{currentStep.title}</h1>
              <p className="mt-2 max-w-[620px] text-[14px] leading-relaxed text-ink-3">{currentStep.desc}</p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button data-testid="onboarding-back" variant="default" icon="back" onClick={goBack} disabled={tourStep === 'connect' && providerStep === 'choice'}>Back</Button>
              {tourStep !== 'settings' && <Button data-testid="onboarding-next" variant="primary" iconRight="arrowRight" onClick={goNext}>Next</Button>}
              {tourStep === 'settings' && <Button data-testid="onboarding-finish" variant="primary" icon="check" onClick={() => complete()}>Finish</Button>}
            </div>
          </div>

          {tourStep === 'connect' && (
            <ConnectStep
              providerStep={providerStep}
              setProviderStep={setProviderStep}
              mode={mode}
              preset={preset}
              apiKey={apiKey}
              localPath={localPath}
              busy={busy}
              error={error}
              foundModels={foundModels}
              cliAuth={cliAuth}
              connectedLabel={connectedLabel}
              onPickCloud={pickCloud}
              onPickLocal={pickLocal}
              onDetect={detect}
              onConnect={connect}
              onApiKeyChange={setApiKey}
              onLocalPathChange={setLocalPath}
              onOpenModels={() => openSettings('models')}
              onOpenEngineModels={() => setView({ name: 'models' })}
              onUseEngineModel={(repo) => void chooseEngineModel(repo)}
              onContinue={goNext}
            />
          )}

          {tourStep === 'chat' && (
            <ModeIntro
              icon="message"
              title="Start clean conversations without dragging a repo along."
              body="Chat creates regular threads, keeps them separate from Code projects, and is the right default for questions, drafting, comparison, and model exploration."
              highlights={['No project required', 'Fast new-thread flow', 'Model and account stay visible']}
              primary={{ label: 'Start chat', icon: 'message', onClick: openChat }}
              secondary={{ label: 'Keep going', icon: 'arrowRight', onClick: goNext }}
            />
          )}

          {tourStep === 'assistant' && (
            <ModeIntro
              icon="spark"
              title="Assistant is where TensorFold becomes a connected operator."
              body="This surface is for connectors, memory review, skills, voice, automations, artifacts, and cross-source work. It should feel different from a plain chat thread."
              highlights={['Connector sources', 'Reviewed memory and skills', 'Approvals for outbound actions']}
              primary={{ label: 'Open Assistant', icon: 'spark', onClick: () => openView({ name: 'assistant' }) }}
              secondary={{ label: 'Configure connectors', icon: 'plug', onClick: () => openSettings('connectors') }}
            />
          )}

          {tourStep === 'code' && (
            <ModeIntro
              icon="code"
              title="Code revolves around work, evidence, and handoff."
              body="Projects, rooms, tickets, terminals, diffs, tests, remote hosts, and run receipts live here. This is the mode to use when the model is expected to change a repo."
              highlights={['Projects and worktrees', 'Rooms and executable tickets', 'Diffs, tests, terminals, and receipts']}
              primary={{ label: 'Create project', icon: 'plus', onClick: openProjectModal }}
              secondary={{ label: 'Open projects', icon: 'folder', onClick: () => openView({ name: 'projects' }) }}
            />
          )}

          {tourStep === 'settings' && (
            <ModeIntro
              icon="check"
              title="Use Readiness as the preflight before trusting automation."
              body="The doctor is the place to prove each provider, CLI, connector, terminal backend, orchestrator host, and permission posture. Broken connectors should be visible here, not discovered mid-run."
              highlights={['Provider and CLI health', 'Connector capability proof', 'Remote, terminal, and security checks']}
              primary={{ label: 'Run Readiness', icon: 'check', onClick: () => openSettings('readiness') }}
              secondary={{ label: 'Finish', icon: 'arrowRight', onClick: () => complete() }}
            />
          )}
        </section>
      </div>
    </div>
  )
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
      <EngineChoice onOpenModels={onOpenEngineModels} onUse={onUseEngineModel} />
      <div className="-mb-1 mt-1 text-[13px] font-medium text-ink-3">Other providers</div>
      <div className="grid gap-3 sm:grid-cols-2">
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
      </div>
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="default" icon="settings" onClick={onOpenModels}>Open model settings</Button>
        <button type="button" onClick={onContinue} className="text-[13px] text-ink-3 transition-colors hover:text-ink">
          {connectedLabel ? 'Continue' : 'Continue without a model'}
        </button>
      </div>
    </div>
  )
}

function ModeIntro({
  icon,
  title,
  body,
  highlights,
  primary,
  secondary,
}: {
  icon: IconName
  title: string
  body: string
  highlights: string[]
  primary: { label: string; icon: IconName; onClick: () => void }
  secondary: { label: string; icon: IconName; onClick: () => void }
}) {
  return (
    <div className="grid gap-4">
      <div className="rounded-xl border border-line bg-card p-5 shadow-card">
        <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-surface text-ink-2"><Icon name={icon} size={20} /></div>
        <h2 className="mt-4 max-w-[620px] text-[19px] font-semibold text-ink">{title}</h2>
        <p className="mt-2 max-w-[680px] text-[14px] leading-relaxed text-ink-3">{body}</p>
        <div className="mt-5 grid gap-2.5 sm:grid-cols-3">
          {highlights.map((highlight) => (
            <div key={highlight} className="rounded-lg border border-line bg-card-2 px-3 py-2.5 text-[13px] text-ink-2">
              <Icon name="check" size={13} className="mb-1 text-positive" />
              {highlight}
            </div>
          ))}
        </div>
      </div>
      <div className="flex flex-wrap gap-2.5">
        <Button variant="primary" icon={primary.icon} onClick={primary.onClick}>{primary.label}</Button>
        <Button variant="default" icon={secondary.icon} onClick={secondary.onClick}>{secondary.label}</Button>
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
