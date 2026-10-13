import { useState, type Dispatch, type ReactNode, type SetStateAction } from 'react'
import clsx from 'clsx'
import { Icon, type IconName } from '../../lib/icons'
import { Button, Field, Input, Toggle } from '../ui'
import { CODEX_CLI_MODELS, CLAUDE_CLI_MODELS, OPENCODE_MODELS, KILO_MODELS, PI_CLI_MODELS, CURSOR_CLI_MODELS, GEMINI_CLI_MODELS } from '../../lib/cli-models'
import { modelDisplayName, providerModelGroups } from '../../lib/provider-accounts'
import { agentBuilderModelSourceCount } from '../../lib/agent-builder'
import type { AgentPreset, ModelSpendBudget, ModelSpendBudgetPeriod, Provider, ProviderDiscoveryKind, RoomSeatLane, Settings as TSettings } from '../../lib/types'

export interface SettingsSectionContext extends Record<string, any> {
  draft: TSettings
  setDraft: Dispatch<SetStateAction<TSettings>>
  voiceNames: string[]
}

export function Spinner({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" className="animate-spin">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  )
}

export type ProviderSetupMode = 'cloud' | 'local'
export type LocalSetupLocation = 'this-mac' | 'network'
export type LocalRuntimeKey = ProviderDiscoveryKind | 'custom'

export type ProviderSeed = Pick<Provider, 'name' | 'baseUrl' | 'kind'> & Partial<Provider>

export interface CloudProviderPreset {
  key: string
  label: string
  desc: string
  auth: 'api-key' | 'cli-oauth' | 'cli-local'
  provider: ProviderSeed
}

export interface LocalRuntimePreset {
  key: LocalRuntimeKey
  label: string
  desc: string
  defaultPort: string
  defaultApiKey: string
  defaultModels: string[]
  source: Provider['source']
}

export type CloudTransportType = 'http' | 'codex-cli' | 'claude-cli'

export interface ProviderSetupDraft {
  mode: ProviderSetupMode
  // Cloud transport chosen once at add-time (#44). Drives whether the HTTP
  // provider presets / API-key fields or the CLI path fields are shown.
  transportType: CloudTransportType
  // For Codex/Claude CLI: 'single' detects the one installed login and connects;
  // 'multi' walks a wizard to set up several subscription accounts.
  cliSubscriptionMode: 'single' | 'multi'
  cloudKey: string
  localLocation: LocalSetupLocation
  localRuntime: LocalRuntimeKey
  providerName: string
  nodeName: string
  host: string
  port: string
  basePath: string
  apiKey: string
  cliPath: string
  cliProfile: string
}

export const CLOUD_PROVIDER_PRESETS: CloudProviderPreset[] = [
  { key: 'openai', label: 'OpenAI', desc: 'Frontier API with GPT and reasoning models.', auth: 'api-key', provider: { name: 'OpenAI', baseUrl: 'https://api.openai.com/v1', kind: 'cloud', source: 'openai', models: ['gpt-4o', 'gpt-4o-mini', 'o4-mini'] } },
  { key: 'openrouter', label: 'OpenRouter', desc: 'Router, free models, GLM, Nemotron, Qwen, and model fan-out experiments.', auth: 'api-key', provider: { name: 'OpenRouter', baseUrl: 'https://openrouter.ai/api/v1', kind: 'cloud', source: 'openrouter', models: ['openrouter/free', 'openrouter/auto', 'openrouter/fusion', 'z-ai/glm-5.2', 'nvidia/nemotron-3-ultra-550b-a55b:free', 'qwen/qwen3-coder:free', 'openai/gpt-oss-120b:free'] } },
  { key: 'ai-gateway', label: 'Vercel AI Gateway', desc: 'OpenAI-compatible gateway path for one key across routed providers and gateway model ids.', auth: 'api-key', provider: { name: 'Vercel AI Gateway', baseUrl: 'https://ai-gateway.vercel.sh/v1', kind: 'cloud', source: 'ai-gateway', models: ['openai/gpt-5.4', 'openai/gpt-5.5', 'anthropic/claude-sonnet-4.6', 'anthropic/claude-opus-4.1', 'xai/grok-4.1-fast-reasoning'] } },
  { key: 'codex-cli', label: 'Codex CLI OAuth', desc: 'Use existing Codex CLI subscription logins, including GPT 5.4/5.5 with reasoning effort.', auth: 'cli-oauth', provider: { name: 'Codex CLI OAuth', transport: 'codex-cli', auth: 'codex-oauth', baseUrl: '', apiKey: '', kind: 'cloud', models: CODEX_CLI_MODELS, cliPath: 'codex', cliSandbox: 'workspace-write', cliApprovalPolicy: 'on-request' } },
  { key: 'claude-cli', label: 'Claude CLI OAuth', desc: 'Use Claude Code subscriptions with supported Opus and Sonnet model aliases.', auth: 'cli-oauth', provider: { name: 'Claude CLI OAuth', transport: 'claude-cli', auth: 'claude-oauth', baseUrl: '', apiKey: '', kind: 'cloud', models: CLAUDE_CLI_MODELS, cliPath: 'claude', cliPermissionMode: 'default' } },
  { key: 'opencode', label: 'OpenCode server', desc: 'Drive an `opencode serve` HTTP session. Defaults to http://127.0.0.1:4096; models are providerID/modelID.', auth: 'api-key', provider: { name: 'OpenCode', transport: 'opencode', baseUrl: '', apiKey: '', kind: 'cloud', models: OPENCODE_MODELS, serverUrl: 'http://127.0.0.1:4096' } },
  { key: 'kilo', label: 'Kilo server', desc: 'Drive a `kilo serve` HTTP session (OpenCode-protocol fork). Set the serverUrl to your Kilo server; models are providerID/modelID.', auth: 'api-key', provider: { name: 'Kilo', transport: 'kilo', baseUrl: '', apiKey: '', kind: 'cloud', models: KILO_MODELS, serverUrl: 'http://127.0.0.1:4096' } },
  { key: 'pi-cli', label: 'Pi CLI', desc: 'Drive the installed `pi` coding agent (RPC mode, resumable sessions). Authenticate in Pi itself or expose a provider-specific key environment such as OPENAI_API_KEY; TensorFold never passes a generic key in argv.', auth: 'cli-local', provider: { name: 'Pi', transport: 'pi-cli', baseUrl: '', apiKey: '', kind: 'cloud', models: PI_CLI_MODELS, cliPath: 'pi' } },
  { key: 'cursor-cli', label: 'Cursor Agent CLI', desc: 'Drive `cursor-agent` headless (stream-json, resumable chats). API key optional (CURSOR_API_KEY / --api-key).', auth: 'api-key', provider: { name: 'Cursor Agent', transport: 'cursor-cli', baseUrl: '', apiKey: '', kind: 'cloud', models: CURSOR_CLI_MODELS, cliPath: 'cursor-agent' } },
  { key: 'gemini-cli', label: 'Gemini CLI', desc: 'Drive `gemini` headless (json mode, one-shot turns). Auth via its own login or env in CLI env settings.', auth: 'api-key', provider: { name: 'Gemini CLI', transport: 'gemini-cli', baseUrl: '', apiKey: '', kind: 'cloud', models: GEMINI_CLI_MODELS, cliPath: 'gemini' } },
  { key: 'anthropic', label: 'Anthropic', desc: 'Claude via API key for cloud seats and assistant work.', auth: 'api-key', provider: { name: 'Anthropic', baseUrl: 'https://api.anthropic.com/v1', kind: 'cloud', models: ['claude-sonnet-4', 'claude-opus-4'] } },
  { key: 'groq', label: 'Groq', desc: 'Fast hosted OpenAI-compatible inference.', auth: 'api-key', provider: { name: 'Groq', baseUrl: 'https://api.groq.com/openai/v1', kind: 'cloud' } },
  { key: 'together', label: 'Together', desc: 'Hosted open models behind an OpenAI-compatible API.', auth: 'api-key', provider: { name: 'Together', baseUrl: 'https://api.together.xyz/v1', kind: 'cloud' } },
  { key: 'zai', label: 'z.ai GLM', desc: 'GLM models as first-class cloud seats.', auth: 'api-key', provider: { name: 'z.ai GLM', baseUrl: 'https://api.z.ai/api/paas/v4', kind: 'cloud', models: ['glm-5.2', 'glm-5.1', 'glm-4.6', 'glm-4-flash'] } },
  { key: 'deepseek', label: 'DeepSeek', desc: 'DeepSeek API endpoint.', auth: 'api-key', provider: { name: 'DeepSeek', baseUrl: 'https://api.deepseek.com/v1', kind: 'cloud' } },
  { key: 'xai', label: 'xAI Grok', desc: 'Grok API endpoint.', auth: 'api-key', provider: { name: 'xAI Grok', baseUrl: 'https://api.x.ai/v1', kind: 'cloud' } },
]

export const LOCAL_RUNTIME_PRESETS: LocalRuntimePreset[] = [
  { key: 'omlx', label: 'oMLX', desc: 'MLX local server. This Mac can read ~/.omlx/settings.json.', defaultPort: '9000', defaultApiKey: '', defaultModels: ['local-model'], source: 'omlx' },
  { key: 'lmstudio', label: 'LM Studio', desc: 'LM Studio local server or LAN node.', defaultPort: '1234', defaultApiKey: 'lm-studio', defaultModels: ['local-model'], source: 'lmstudio' },
  { key: 'ollama', label: 'Ollama', desc: 'Ollama OpenAI-compatible endpoint.', defaultPort: '11434', defaultApiKey: 'ollama', defaultModels: ['qwen2.5-coder:14b', 'llama3.1:8b'], source: 'ollama' },
  { key: 'custom', label: 'vLLM / llama.cpp', desc: 'Any OpenAI-compatible local or network endpoint.', defaultPort: '8000', defaultApiKey: '', defaultModels: ['local-model'], source: 'custom' },
]

// HTTP-endpoint cloud presets only (CLI providers are picked via the
// transport-type radio at add-time, not this dropdown — #44).
export const HTTP_CLOUD_PROVIDER_PRESETS = CLOUD_PROVIDER_PRESETS.filter((preset) => preset.auth !== 'cli-oauth')

export function providerPresetNeedsApiKey(preset: Pick<CloudProviderPreset, 'auth'>): boolean {
  return preset.auth === 'api-key'
}

export const DEFAULT_PROVIDER_SETUP: ProviderSetupDraft = {
  mode: 'cloud',
  transportType: 'http',
  cliSubscriptionMode: 'single',
  cloudKey: 'openrouter',
  localLocation: 'this-mac',
  localRuntime: 'omlx',
  providerName: '',
  nodeName: '',
  host: '127.0.0.1',
  port: '9000',
  basePath: '/v1',
  apiKey: '',
  cliPath: '',
  cliProfile: '',
}

export const SELECT_CLASS = 'no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-3 text-[13px] text-ink outline-none transition-colors focus:border-accent focus:ring-2 focus:ring-accent/15'
export const SPEND_BUDGET_PERIODS: Array<{ value: ModelSpendBudgetPeriod; label: string }> = [
  { value: 'run', label: 'Per run' },
  { value: 'day', label: 'Daily' },
  { value: 'week', label: 'Weekly' },
  { value: 'month', label: 'Monthly' },
]

export function newProviderId() {
  return 'p_' + Math.random().toString(36).slice(2, 8)
}

export function numberOrNull(value: string): number | null {
  if (!value.trim()) return null
  const n = Number(value)
  return Number.isFinite(n) && n >= 0 ? n : null
}

export function spendBudgetDefaults(budget: ModelSpendBudget | null | undefined): ModelSpendBudget {
  return {
    enabled: budget?.enabled ?? false,
    period: budget?.period ?? 'month',
    limitUsd: budget?.limitUsd ?? null,
    spentUsd: budget?.spentUsd ?? 0,
    warnAtPercent: budget?.warnAtPercent ?? 80,
    hardStop: budget?.hardStop ?? true,
    updatedAt: budget?.updatedAt ?? null,
  }
}

export function patchSpendBudget(current: ModelSpendBudget | null | undefined, patch: Partial<ModelSpendBudget>): ModelSpendBudget {
  return { ...spendBudgetDefaults(current), ...patch, updatedAt: Date.now() }
}

export type SettingsSection = 'readiness' | 'models' | 'agents' | 'agent' | 'orchestrator' | 'connectors' | 'persona' | 'security' | 'voice' | 'appearance' | 'skills'

export interface SettingsSectionMeta {
  id: SettingsSection
  label: string
  desc: string
  icon: IconName
}

export interface SettingsGroup {
  id: string
  label: string
  sections: SettingsSectionMeta[]
}

// Four grouped sections replace the flat nine-item list. Group IDs are stable
// keys for persisting the expanded-state in localStorage; section IDs are
// unchanged so deep-links (requestSettingsSection) and the render dispatch in
// Settings.tsx keep working untouched.
export const SETTINGS_GROUPS: SettingsGroup[] = [
  {
    id: 'ai',
    label: 'AI & Models',
    sections: [
      { id: 'readiness', label: 'Readiness', desc: 'Providers, CLIs, connectors', icon: 'check' },
      { id: 'models', label: 'Models & Accounts', desc: 'Sources for chats and agents', icon: 'cpu' },
      { id: 'agents', label: 'Agent Builder', desc: 'Named model seats for rooms', icon: 'message' },
    ],
  },
  {
    id: 'assistant',
    label: 'Assistant',
    sections: [
      { id: 'connectors', label: 'Connectors', desc: 'Email, documents, calendar, bots', icon: 'plug' },
      { id: 'persona', label: 'Persona', desc: 'The assistant’s voice + your profile', icon: 'spark' },
      { id: 'voice', label: 'Voice', desc: 'Speech in and readback', icon: 'speaker' },
    ],
  },
  {
    id: 'behavior',
    label: 'Behavior',
    sections: [
      { id: 'agent', label: 'Runtime', desc: 'Approvals, folders, context', icon: 'spark' },
      { id: 'security', label: 'Security', desc: 'Confidential terms, controller and permissions', icon: 'shield' },
      { id: 'orchestrator', label: 'Orchestrator', desc: 'Host and remote control', icon: 'monitor' },
    ],
  },
  {
    id: 'preferences',
    label: 'Preferences',
    sections: [
      { id: 'appearance', label: 'Appearance', desc: 'Theme and display', icon: 'sun' },
      { id: 'skills', label: 'Skills', desc: '$ skills and / commands', icon: 'list' },
    ],
  },
]

// Flattened view for lookups (active-section header meta, deep-link resolution).
export const ALL_SETTINGS_SECTIONS: SettingsSectionMeta[] = SETTINGS_GROUPS.flatMap((g) => g.sections)

/** @deprecated use ALL_SETTINGS_SECTIONS — kept for any external readers. */
export const SETTINGS_SECTIONS = ALL_SETTINGS_SECTIONS

/** Find the group a section belongs to (for auto-expansion on deep-link). */
export function groupOfSection(section: SettingsSection): SettingsGroup | undefined {
  return SETTINGS_GROUPS.find((g) => g.sections.some((s) => s.id === section))
}

export function Section({ title, desc, children }: { title: string; desc?: string; children: ReactNode }) {
  return (
    <section className="mb-8 last:mb-0">
      <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-ink">{title}</h2>
      {desc && <p className="mt-0.5 text-[13px] text-ink-3">{desc}</p>}
      <div className="mt-3.5">{children}</div>
    </section>
  )
}

export function SettingsPane({ children }: { children: ReactNode }) {
  return <div className="mx-auto w-full max-w-[760px]">{children}</div>
}

/**
 * Collapsible "Advanced" disclosure. Wraps the rare/power-user controls so each
 * panel leads with the common stuff (API keys, toggles, sync buttons) and hides
 * the long tail (failover policies, capture sizes, sandbox modes) behind one
 * click. Extracted from RuntimeSettings' original inline <details>; the styling
 * matches so the refactor there is behavior-neutral.
 */
export function AdvancedSection({
  title = 'Advanced',
  icon = 'sliders',
  children,
}: {
  title?: string
  icon?: IconName
  children: ReactNode
}) {
  return (
    <details className="group rounded-xl border border-line bg-card shadow-card">
      <summary className="flex cursor-pointer list-none items-center justify-between gap-2 px-3.5 py-3 text-[13px] font-medium text-ink-2 [&::-webkit-details-marker]:hidden">
        <span className="flex items-center gap-2"><Icon name={icon} size={15} className="text-ink-3" />{title}</span>
        <Icon name="chevronDown" size={15} className="text-ink-3 transition-transform group-open:rotate-180" />
      </summary>
      <div className="space-y-4 border-t border-line px-3.5 py-3.5">{children}</div>
    </details>
  )
}

export function SpendGuardControls({
  budget,
  onChange,
  title,
  desc,
  scope,
}: {
  budget: ModelSpendBudget | null | undefined
  onChange: (patch: Partial<ModelSpendBudget>) => void
  title: string
  desc: string
  scope: string
}) {
  const current = spendBudgetDefaults(budget)
  return (
    <div className="rounded-lg border border-line bg-card-2 px-3 py-2.5">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[13px] font-medium text-ink-2">{title}</div>
          <div className="truncate text-[12px] text-ink-3">{desc}</div>
        </div>
        <Toggle label={`${title} for ${scope}`} checked={current.enabled === true} onChange={(enabled) => onChange({ enabled })} />
      </div>
      {current.enabled && (
        <div className="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-5">
          <Field label="Period">
            <select
              value={current.period ?? 'month'}
              onChange={(e) => onChange({ period: e.target.value as ModelSpendBudgetPeriod })}
              className={SELECT_CLASS}
            >
              {SPEND_BUDGET_PERIODS.map((period) => <option key={period.value} value={period.value}>{period.label}</option>)}
            </select>
          </Field>
          <Field label="Limit" hint="USD">
            <Input
              type="number"
              min="0"
              step="0.01"
              value={current.limitUsd ?? ''}
              onChange={(e) => onChange({ limitUsd: numberOrNull(e.target.value) })}
              placeholder="25"
            />
          </Field>
          <Field label="Spent" hint="USD">
            <Input
              type="number"
              min="0"
              step="0.01"
              value={current.spentUsd ?? ''}
              onChange={(e) => onChange({ spentUsd: numberOrNull(e.target.value) })}
              placeholder="0"
            />
          </Field>
          <Field label="Warn" hint="%">
            <Input
              type="number"
              min="0"
              max="100"
              step="1"
              value={current.warnAtPercent ?? ''}
              onChange={(e) => onChange({ warnAtPercent: numberOrNull(e.target.value) })}
              placeholder="80"
            />
          </Field>
          <Field label="Hard stop">
            <div className="flex h-9 items-center justify-between rounded-lg border border-line bg-card px-2.5">
              <span className="text-[13px] text-ink-3">Block</span>
              <Toggle label={`${title} hard stop for ${scope}`} checked={current.hardStop === true} onChange={(hardStop) => onChange({ hardStop })} />
            </div>
          </Field>
        </div>
      )}
    </div>
  )
}

const AGENT_LANES: Array<{ id: RoomSeatLane; label: string; name: string; role: string; desc: string; icon: IconName; target: 'cloud' | 'local' }> = [
  { id: 'planning', label: 'Planner', name: 'Planner', role: 'Planner', desc: 'Breaks goals into tickets, checks, and handoffs.', icon: 'list', target: 'cloud' },
  { id: 'development', label: 'Builder', name: 'Builder', role: 'Engineer', desc: 'Implements changes with repo and terminal context.', icon: 'code', target: 'local' },
  { id: 'review', label: 'Reviewer', name: 'Reviewer', role: 'Reviewer', desc: 'Looks for risks, gaps, regressions, and test evidence.', icon: 'shield', target: 'cloud' },
  { id: 'research', label: 'Researcher', name: 'Researcher', role: 'Researcher', desc: 'Investigates docs, APIs, and outside constraints.', icon: 'search', target: 'cloud' },
  { id: 'operations', label: 'Operator', name: 'Operator', role: 'Operator', desc: 'Runs commands, checks services, and handles environment work.', icon: 'terminal', target: 'local' },
]

function agentLaneMeta(lane: RoomSeatLane) {
  return AGENT_LANES.find((option) => option.id === lane) ?? AGENT_LANES[1]
}

function agentInitials(name: string): string {
  return name.replace(/[^a-zA-Z0-9]/g, ' ').trim().split(/\s+/).slice(0, 2).map((part) => part[0]?.toUpperCase()).join('') || 'A'
}

function uniqueAgentName(base: string, presets: AgentPreset[]): string {
  const used = new Set(presets.map((preset) => preset.name.trim().toLowerCase()).filter(Boolean))
  if (!used.has(base.toLowerCase())) return base
  for (let i = 2; i < 100; i++) {
    const candidate = `${base} ${i}`
    if (!used.has(candidate.toLowerCase())) return candidate
  }
  return `${base} ${presets.length + 1}`
}

function agentModelChoiceValue(providerId: string, accountId: string | null | undefined, model: string) {
  return JSON.stringify([providerId, accountId ?? null, model])
}

function parseAgentModelChoice(value: string): Pick<AgentPreset, 'providerId' | 'providerAccountId' | 'model'> | null {
  try {
    const [providerId, providerAccountId, model] = JSON.parse(value)
    if (typeof providerId === 'string' && (typeof providerAccountId === 'string' || providerAccountId === null) && typeof model === 'string') {
      return { providerId, providerAccountId, model }
    }
  } catch {
    // Ignore malformed select values.
  }
  return null
}

function firstAgentModelChoice(providers: Provider[], lane: RoomSeatLane = 'development') {
  const target = agentLaneMeta(lane).target
  const ordered = [
    ...providers.filter((provider) => provider.kind === target),
    ...providers.filter((provider) => provider.kind !== target),
  ]
  for (const provider of ordered) {
    const group = providerModelGroups(provider).find((candidate) => candidate.models.length)
    const model = group?.models[0]
    if (model) return { providerId: provider.id, providerAccountId: group.accountId, model }
  }
  return null
}

export function AgentBuilderPanel({ settings, onSave }: { settings: TSettings; onSave: (next: TSettings) => void }) {
  const presets = settings.agentPresets ?? []
  const providers = settings.providers
  const modelGroups = providers.flatMap((provider) => providerModelGroups(provider).filter((group) => group.models.length).map((group) => ({ provider, group })))
  const [selectedId, setSelectedId] = useState<string | null>(presets[0]?.id ?? null)
  const selectedPreset = presets.find((preset) => preset.id === selectedId) ?? presets[0] ?? null
  const selectedProvider = selectedPreset ? providers.find((provider) => provider.id === selectedPreset.providerId) ?? null : null
  const updatePreset = (id: string, patch: Partial<AgentPreset>) => {
    const updatedAt = Date.now()
    onSave({ ...settings, agentPresets: presets.map((preset) => preset.id === id ? { ...preset, ...patch, updatedAt } : preset) })
  }
  const retunePresetLane = (preset: AgentPreset, lane: RoomSeatLane) => {
    const currentMeta = agentLaneMeta(preset.lane)
    const nextMeta = agentLaneMeta(lane)
    const provider = providers.find((item) => item.id === preset.providerId)
    const patch: Partial<AgentPreset> = { lane }
    if (!preset.role.trim() || preset.role === currentMeta.role || preset.role === currentMeta.label) patch.role = nextMeta.role
    if (provider?.kind !== nextMeta.target) {
      const pick = firstAgentModelChoice(providers, lane)
      if (pick) Object.assign(patch, pick)
    }
    updatePreset(preset.id, patch)
  }
  const deletePreset = (id: string) => {
    const nextPresets = presets.filter((preset) => preset.id !== id)
    if (selectedPreset?.id === id) setSelectedId(nextPresets[0]?.id ?? null)
    onSave({ ...settings, agentPresets: nextPresets })
  }
  const addPreset = (lane: RoomSeatLane = 'development') => {
    const pick = firstAgentModelChoice(providers, lane)
    if (!pick) return
    const now = Date.now()
    const meta = agentLaneMeta(lane)
    const preset: AgentPreset = {
      id: 'agent_' + Math.random().toString(36).slice(2, 8),
      name: uniqueAgentName(meta.name, presets),
      role: meta.role,
      lane,
      providerId: pick.providerId,
      providerAccountId: pick.providerAccountId,
      model: pick.model,
      createdAt: now,
      updatedAt: now,
    }
    onSave({ ...settings, agentPresets: [preset, ...presets] })
    setSelectedId(preset.id)
  }
  const addStarterTeam = () => {
    const now = Date.now()
    const next: AgentPreset[] = []
    for (const lane of ['planning', 'development', 'review'] as RoomSeatLane[]) {
      const pick = firstAgentModelChoice(providers, lane)
      if (!pick) continue
      const meta = agentLaneMeta(lane)
      next.push({
        id: 'agent_' + Math.random().toString(36).slice(2, 8),
        name: uniqueAgentName(meta.name, [...next, ...presets]),
        role: meta.role,
        lane,
        providerId: pick.providerId,
        providerAccountId: pick.providerAccountId,
        model: pick.model,
        createdAt: now,
        updatedAt: now,
      })
    }
    if (!next.length) return
    onSave({ ...settings, agentPresets: [...next, ...presets] })
    setSelectedId(next[0].id)
  }
  const duplicatePreset = (preset: AgentPreset) => {
    const now = Date.now()
    const copy: AgentPreset = {
      ...preset,
      id: 'agent_' + Math.random().toString(36).slice(2, 8),
      name: uniqueAgentName(`${preset.name || agentLaneMeta(preset.lane).name} copy`, presets),
      createdAt: now,
      updatedAt: now,
    }
    onSave({ ...settings, agentPresets: [copy, ...presets] })
    setSelectedId(copy.id)
  }
  const modelValue = (preset: AgentPreset) => agentModelChoiceValue(preset.providerId, preset.providerAccountId ?? null, preset.model)
  const selectedModelMissing = !!selectedPreset && !modelGroups.some(({ provider, group }) => provider.id === selectedPreset.providerId && group.accountId === (selectedPreset.providerAccountId ?? null) && group.models.includes(selectedPreset.model))
  const laneCounts = AGENT_LANES.map((lane) => ({ ...lane, count: presets.filter((preset) => preset.lane === lane.id).length }))
  const providerCount = agentBuilderModelSourceCount(providers)

  return (
    <Section title="Agent builder" desc="Create named seats from your model suite. These are light room roles, not deep persona prompts.">
      <div className="mb-4 overflow-hidden rounded-xl border border-line bg-card shadow-card">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-card-2/70 px-4 py-3.5">
          <div>
            <div className="flex items-center gap-2 text-[14px] font-semibold text-ink"><Icon name="layers" size={15} className="text-ink-3" />Room team canvas</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{presets.length} saved agent{presets.length === 1 ? '' : 's'} · {providerCount || 0} model source{providerCount === 1 ? '' : 's'}</div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button variant="default" size="sm" icon="spark" onClick={addStarterTeam} disabled={modelGroups.length === 0}>Starter trio</Button>
            <Button variant="primary" size="sm" icon="plus" onClick={() => addPreset()} disabled={modelGroups.length === 0}>Add agent</Button>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-px bg-line/70 sm:grid-cols-5">
          {laneCounts.map((lane) => (
            <button
              key={lane.id}
              onClick={() => addPreset(lane.id)}
              disabled={modelGroups.length === 0}
              className="group flex min-h-[88px] flex-col items-start justify-between bg-card px-3 py-3 text-left transition-colors hover:bg-card-2 disabled:pointer-events-none disabled:opacity-45"
            >
              <span className="flex items-center gap-2 text-[13px] font-medium text-ink"><Icon name={lane.icon} size={14} className="text-ink-3" />{lane.label}</span>
              <span className="max-h-[30px] overflow-hidden text-[12px] leading-snug text-ink-3">{lane.desc}</span>
              <span className="mt-2 inline-flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-[12px] text-ink-3 group-hover:text-ink">
                {lane.count ? `${lane.count} saved` : `Add ${lane.target}`}
              </span>
            </button>
          ))}
        </div>
      </div>

      {modelGroups.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-10 text-center">
          <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-ink-5 text-ink-3"><Icon name="cpu" size={22} /></div>
          <div className="mt-3 text-[13px] font-medium text-ink">Connect a model source first</div>
          <div className="mx-auto mt-1 max-w-[360px] text-[13px] leading-relaxed text-ink-3">Agent seats are named wrappers around your available models. Add a local node or cloud account, then come back and build the team.</div>
        </div>
      ) : presets.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line px-4 py-10 text-center">
          <div className="mx-auto flex h-11 w-11 items-center justify-center rounded-xl bg-ink-5 text-ink-3"><Icon name="message" size={22} /></div>
          <div className="mt-3 text-[13px] font-medium text-ink">No room agents yet</div>
          <div className="mx-auto mt-1 max-w-[360px] text-[13px] leading-relaxed text-ink-3">Planner, builder, and reviewer make a balanced default room team.</div>
          <div className="mt-4 flex justify-center gap-2">
            <Button variant="default" size="sm" icon="spark" onClick={addStarterTeam}>Starter trio</Button>
            <Button variant="primary" size="sm" icon="plus" onClick={() => addPreset()}>Add agent</Button>
          </div>
        </div>
      ) : (
        <div className="grid gap-3 md:grid-cols-[250px_minmax(0,1fr)]">
          <div className="max-h-[520px] space-y-2 overflow-y-auto pr-1 scroll-thin">
            <div className="px-1 text-[12px] font-semibold uppercase tracking-[0.06em] text-ink-3">Roster</div>
            {presets.map((preset) => {
              const provider = providers.find((item) => item.id === preset.providerId)
              const meta = agentLaneMeta(preset.lane)
              const active = selectedPreset?.id === preset.id
              return (
                <button
                  key={preset.id}
                  onClick={() => setSelectedId(preset.id)}
                  className={clsx('group flex w-full items-start gap-3 rounded-xl border p-3 text-left transition-colors', active ? 'border-line-strong bg-card text-ink shadow-card' : 'border-line bg-card/70 text-ink-2 hover:bg-card hover:text-ink')}
                >
                  <span className={clsx('flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-[13px] font-semibold', active ? 'bg-accent text-accent-ink' : 'bg-ink-5 text-ink-2')}>
                    {agentInitials(preset.name)}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5">
                      <span className="truncate text-[13px] font-semibold">{preset.name || meta.name}</span>
                      <Icon name={meta.icon} size={12} className="shrink-0 text-ink-3" />
                    </span>
                    <span className="mt-0.5 block truncate text-[13px] text-ink-3">{preset.role || meta.role}</span>
                    <span className="mt-2 flex items-center gap-1.5 text-[12px] text-ink-4">
                      <Icon name={provider?.kind === 'local' ? 'cpu' : 'globe'} size={11} />
                      <span className="truncate">{provider?.name ?? 'Missing provider'}</span>
                    </span>
                  </span>
                </button>
              )
            })}
          </div>

          {selectedPreset && (
            <div className="rounded-xl border border-line bg-card shadow-card">
              <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-3">
                <div className="flex min-w-0 items-center gap-3">
                  <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-accent text-[14px] font-semibold text-accent-ink">{agentInitials(selectedPreset.name)}</span>
                  <div className="min-w-0">
                    <div className="truncate text-[14px] font-semibold text-ink">{selectedPreset.name || agentLaneMeta(selectedPreset.lane).name}</div>
                    <div className="mt-0.5 flex items-center gap-1.5 text-[13px] text-ink-3">
                      <Icon name={agentLaneMeta(selectedPreset.lane).icon} size={12} />
                      {agentLaneMeta(selectedPreset.lane).label} · {selectedProvider?.name ?? 'Provider missing'}
                    </div>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-1">
                  <Button variant="ghost" size="xs" icon="copy" onClick={() => duplicatePreset(selectedPreset)}>Duplicate</Button>
                  <Button variant="danger" size="xs" icon="trash" onClick={() => deletePreset(selectedPreset.id)}>Delete</Button>
                </div>
              </div>

              <div className="space-y-4 p-4">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  <Field label="Name" hint="Shown in rooms">
                    <Input value={selectedPreset.name} onChange={(e) => updatePreset(selectedPreset.id, { name: e.target.value })} placeholder="Builder" />
                  </Field>
                  <Field label="Room role" hint="How the room uses it">
                    <Input value={selectedPreset.role} onChange={(e) => updatePreset(selectedPreset.id, { role: e.target.value })} placeholder="Engineer" />
                  </Field>
                </div>

                <div>
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <div className="text-[13px] font-medium text-ink-2">Role lane</div>
                    <div className="text-[12px] text-ink-3">Planner · Builder · Reviewer</div>
                  </div>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-5">
                    {AGENT_LANES.map((lane) => (
                    <button
                      key={lane.id}
                      onClick={() => retunePresetLane(selectedPreset, lane.id)}
                      className={clsx('flex min-h-[76px] flex-col items-start gap-1.5 rounded-lg border px-2.5 py-2 text-left transition-colors', selectedPreset.lane === lane.id ? 'border-line-strong bg-ink-5 text-ink' : 'border-line bg-card-2/50 text-ink-3 hover:text-ink')}
                    >
                      <Icon name={lane.icon} size={14} className={selectedPreset.lane === lane.id ? 'text-accent' : 'text-ink-3'} />
                      <span className="text-[13px] font-medium">{lane.label}</span>
                      <span className="text-[11px] text-ink-4">{lane.target}</span>
                    </button>
                    ))}
                  </div>
                </div>

                <Field label="Model" hint={selectedProvider?.kind === 'local' ? 'local seat' : selectedProvider ? 'cloud seat' : 'missing'}>
                  <select
                    value={modelValue(selectedPreset)}
                    onChange={(e) => {
                      const picked = parseAgentModelChoice(e.target.value)
                      if (picked) updatePreset(selectedPreset.id, picked)
                    }}
                    className={SELECT_CLASS}
                  >
                    {selectedModelMissing && <option value={modelValue(selectedPreset)}>Missing · {selectedPreset.model}</option>}
                    {modelGroups.map(({ provider, group }) => (
                      <optgroup key={group.id} label={group.detail ? `${group.label} · ${group.detail}` : group.label}>
                        {group.models.map((model) => <option key={`${group.id}:${model}`} value={agentModelChoiceValue(provider.id, group.accountId, model)}>{modelDisplayName(provider, model)}</option>)}
                      </optgroup>
                    ))}
                  </select>
                </Field>

                <div className="grid gap-2 sm:grid-cols-3">
                  <div className="rounded-lg border border-line bg-card-2/60 px-3 py-2">
                    <div className="text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">Provider</div>
                    <div className="mt-1 flex items-center gap-1.5 truncate text-[13px] font-medium text-ink"><Icon name={selectedProvider?.kind === 'local' ? 'cpu' : 'globe'} size={13} className="text-ink-3" />{selectedProvider?.name ?? 'Missing'}</div>
                  </div>
                  <div className="rounded-lg border border-line bg-card-2/60 px-3 py-2">
                    <div className="text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">Lane</div>
                    <div className="mt-1 truncate text-[13px] font-medium text-ink">{agentLaneMeta(selectedPreset.lane).label}</div>
                  </div>
                  <div className="rounded-lg border border-line bg-card-2/60 px-3 py-2">
                    <div className="text-[12px] font-medium uppercase tracking-[0.05em] text-ink-4">Model</div>
                    <div className="mt-1 truncate text-[13px] font-medium text-ink">{modelDisplayName(selectedProvider ?? undefined, selectedPreset.model)}</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </Section>
  )
}
