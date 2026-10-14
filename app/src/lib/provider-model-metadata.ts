import type { Attachment, Provider, ReasoningEffortSetting } from './types'
import type { IconName } from './icons'

export interface ProviderEffortOption {
  value: ReasoningEffortSetting
  label: string
  isDefault?: boolean
  runtime?: 'native' | 'compatible' | 'metadata'
  description?: string
}

export interface ProviderContextWindowOption {
  value: '200k' | '1m' | string
  label: string
  tokens: number
  isDefault?: boolean
}

export interface ProviderModelCapabilities {
  family: ProviderFamily
  effortOptions: ProviderEffortOption[]
  supportsFastMode: boolean
  supportsThinkingToggle: boolean
  contextWindowOptions: ProviderContextWindowOption[]
  promptInjectedEfforts: ReasoningEffortSetting[]
  // Whether a Guide cue lands inside the CURRENT turn. True for the native loop (cues
  // apply between its steps) and for delegated engines whose protocols take mid-turn input
  // (codex turn/steer, claude stream-json stdin, pi rpc steer). False for delegated
  // engines that only pick cues up after the running engine turn finishes.
  liveSteering: boolean
}

export function providerLiveSteering(provider: Provider | null | undefined): boolean {
  const transport = provider?.transport
  return !(transport === 'cursor-cli' || transport === 'gemini-cli' || transport === 'opencode' || transport === 'kilo')
}

export interface ContextUsageSummary {
  usedTokens: number
  windowTokens: number
  workingLimitTokens?: number | null
  percent: number
  status?: string | null
}

const BASE_COMPOSER_CONTEXT_TOKENS = 2_100

export type ProviderFamily = 'openai' | 'claude' | 'gemini' | 'grok' | 'cursor' | 'opencode' | 'kilo' | 'pi' | 'local' | 'generic'

const GENERIC_EFFORTS: ProviderEffortOption[] = [
  { value: 'minimal', label: 'Minimal', runtime: 'native' },
  { value: 'low', label: 'Low', runtime: 'native' },
  { value: 'medium', label: 'Medium', isDefault: true, runtime: 'native' },
  { value: 'high', label: 'High', runtime: 'native' },
]

const CODEX_GPT_5_EFFORTS: ProviderEffortOption[] = [
  { value: 'low', label: 'Low', runtime: 'native' },
  { value: 'medium', label: 'Medium', runtime: 'native' },
  { value: 'high', label: 'High', isDefault: true, runtime: 'native' },
  { value: 'xhigh', label: 'Extra High', runtime: 'native' },
]

const CODEX_GPT_5_5_EFFORTS: ProviderEffortOption[] = [
  { value: 'low', label: 'Low', runtime: 'native' },
  { value: 'medium', label: 'Medium', isDefault: true, runtime: 'native' },
  { value: 'high', label: 'High', runtime: 'native' },
  { value: 'xhigh', label: 'Extra High', runtime: 'native' },
]

const OPENAI_API_EFFORTS: ProviderEffortOption[] = [
  { value: 'minimal', label: 'Minimal', runtime: 'native' },
  { value: 'low', label: 'Low', runtime: 'native' },
  { value: 'medium', label: 'Medium', isDefault: true, runtime: 'native' },
  { value: 'high', label: 'High', runtime: 'native' },
]

const CLAUDE_EFFORTS: ProviderEffortOption[] = [
  { value: 'low', label: 'Low', runtime: 'native' },
  { value: 'medium', label: 'Medium', runtime: 'native' },
  { value: 'high', label: 'High', isDefault: true, runtime: 'native' },
  { value: 'xhigh', label: 'Extra High', runtime: 'native' },
  { value: 'max', label: 'Max', runtime: 'native' },
  { value: 'ultrathink', label: 'Ultrathink', runtime: 'compatible', description: 'Adds the Ultrathink prompt prefix and runs Claude at extra-high effort.' },
  { value: 'ultracode', label: 'Ultracode', runtime: 'metadata', description: 'Claude workflow mode; TensorFold runs it as extra-high until that mode is wired.' },
]

const GEMINI_3_EFFORTS: ProviderEffortOption[] = [
  { value: 'high', label: 'High', isDefault: true, runtime: 'metadata' },
  { value: 'low', label: 'Low', runtime: 'metadata' },
]

const GEMINI_25_EFFORTS: ProviderEffortOption[] = [
  { value: 'dynamic', label: 'Dynamic', isDefault: true, runtime: 'metadata' },
  { value: '512', label: '512 Tokens', runtime: 'metadata' },
]

const GROK_BUILD_EFFORTS: ProviderEffortOption[] = [
  { value: 'none', label: 'None', runtime: 'native' },
  { value: 'low', label: 'Low', isDefault: true, runtime: 'native' },
  { value: 'medium', label: 'Medium', runtime: 'native' },
  { value: 'high', label: 'High', runtime: 'native' },
]

const CLAUDE_DUAL_CONTEXT: ProviderContextWindowOption[] = [
  { value: '200k', label: '200k', tokens: 200_000, isDefault: true },
  { value: '1m', label: '1M', tokens: 1_000_000 },
]

export function providerFamily(provider: Provider | null | undefined, model?: string | null): ProviderFamily {
  const providerText = `${provider?.id ?? ''} ${provider?.name ?? ''} ${provider?.transport ?? ''} ${provider?.source ?? ''} ${provider?.baseUrl ?? ''}`.toLowerCase()
  const combined = `${providerText} ${model ?? ''}`.toLowerCase()
  if (provider?.transport === 'codex-cli') return 'openai'
  if (provider?.transport === 'claude-cli') return 'claude'
  if (/cursor/.test(providerText)) return 'cursor'
  if (/opencode/.test(providerText)) return 'opencode'
  if (/kilo/.test(providerText)) return 'kilo'
  if (/\bpi\b|(^|[-_\s])pi([-_\s]|$)/.test(providerText)) return 'pi'
  if (/gemini|google/.test(providerText)) return 'gemini'
  if (/grok|xai|x\.ai/.test(providerText)) return 'grok'
  if (provider?.kind === 'local') return 'local'
  if (/\bcodex\b|gpt-5|openai\/gpt|gpt-4|o[134]/.test(combined)) return 'openai'
  if (/claude|anthropic|opus|sonnet|haiku/.test(combined)) return 'claude'
  if (/gemini|google/.test(combined)) return 'gemini'
  if (/grok|xai|x\.ai/.test(combined)) return 'grok'
  return 'generic'
}

function isGemini25(model?: string | null): boolean {
  return /(?:^|[-_/])2\.5(?:[-_/]|$)|auto-gemini-2\.5/i.test(model ?? '')
}

function isCodex55(model?: string | null): boolean {
  return /gpt-5\.5/i.test(model ?? '')
}

function stripCursorParameterizedSuffix(model?: string | null): string {
  return (model ?? '').trim().replace(/\[[\s\S]*\]$/u, '')
}

function cursorModelParams(model?: string | null): Map<string, string> {
  const raw = (model ?? '').match(/\[([\s\S]*)\]$/u)?.[1]
  const params = new Map<string, string>()
  if (!raw) return params
  for (const part of raw.split(',')) {
    const index = part.indexOf('=')
    if (index <= 0) continue
    const key = part.slice(0, index).trim().toLowerCase()
    const value = part.slice(index + 1).trim()
    if (key && value) params.set(key, value)
  }
  return params
}

function normalizeCursorReasoningValue(value?: string | null): ReasoningEffortSetting | null {
  const normalized = value?.trim().toLowerCase()
  if (!normalized) return null
  if (normalized === 'extra-high' || normalized === 'extra high') return 'xhigh'
  if (['none', 'low', 'medium', 'high', 'xhigh', 'max'].includes(normalized)) return normalized as ReasoningEffortSetting
  return null
}

function cursorReasoningFromModel(model?: string | null): ReasoningEffortSetting | null {
  const params = cursorModelParams(model)
  const fromParams = normalizeCursorReasoningValue(params.get('reasoning') ?? params.get('effort'))
  if (fromParams) return fromParams
  const tokens = stripCursorParameterizedSuffix(model).toLowerCase().split('-')
  for (let index = tokens.length - 1; index >= 0; index -= 1) {
    const token = tokens[index]
    if (!token) continue
    if (token === 'high' && tokens[index - 1] === 'extra') return 'xhigh'
    const effort = normalizeCursorReasoningValue(token)
    if (effort) return effort
  }
  return null
}

function cursorEffortsForModel(model?: string | null): ProviderEffortOption[] {
  const base = stripCursorParameterizedSuffix(model).toLowerCase()
  const selected = cursorReasoningFromModel(model)
  let options: ProviderEffortOption[] = []
  if (/gpt-5|codex/.test(base)) options = isCodex55(base) ? CODEX_GPT_5_5_EFFORTS : CODEX_GPT_5_EFFORTS
  else if (/claude|opus|sonnet|haiku|fable/.test(base)) {
    options = [
      { value: 'low', label: 'Low', runtime: 'metadata' },
      { value: 'medium', label: 'Medium', runtime: 'metadata' },
      { value: 'high', label: 'High', isDefault: true, runtime: 'metadata' },
      { value: 'max', label: 'Max', runtime: 'metadata' },
    ]
  }
  if (selected && !options.some((option) => option.value === selected)) {
    const label = selected === 'xhigh' ? 'Extra High' : selected.charAt(0).toUpperCase() + selected.slice(1)
    options = [...options, { value: selected, label, runtime: 'metadata' }]
  }
  return options
}

function isCursorOneMillionContextModel(model?: string | null): boolean {
  const normalized = stripCursorParameterizedSuffix(model).toLowerCase()
  if (cursorModelParams(model).get('context')?.toLowerCase() === '1m') return true
  if (normalized.startsWith('gpt-5.5-')) return true
  if (/^gpt-5\.4-(?:low|medium|high|xhigh|extra-high)$/u.test(normalized)) return true
  if (/^claude-4\.6-(?:opus|sonnet)(?:-|$)/u.test(normalized)) return true
  if (/^claude-(?:fable-5|opus-4-(?:7|8))-/u.test(normalized)) return true
  return false
}

export function providerModelCapabilities(provider: Provider | null | undefined, model?: string | null): ProviderModelCapabilities {
  const family = providerFamily(provider, model)
  const liveSteering = providerLiveSteering(provider)
  switch (family) {
    case 'openai':
      return {
        family,
        effortOptions: provider?.transport === 'codex-cli' ? (isCodex55(model) ? CODEX_GPT_5_5_EFFORTS : CODEX_GPT_5_EFFORTS) : OPENAI_API_EFFORTS,
        supportsFastMode: provider?.transport === 'codex-cli',
        supportsThinkingToggle: false,
        contextWindowOptions: [],
        promptInjectedEfforts: [],
        liveSteering,
      }
    case 'claude':
      return {
        family,
        effortOptions: CLAUDE_EFFORTS,
        supportsFastMode: /opus|fable/i.test(model ?? ''),
        supportsThinkingToggle: /haiku/i.test(model ?? ''),
        contextWindowOptions: CLAUDE_DUAL_CONTEXT,
        promptInjectedEfforts: ['ultrathink'],
        liveSteering,
      }
    case 'gemini':
      return {
        family,
        effortOptions: isGemini25(model) ? GEMINI_25_EFFORTS : GEMINI_3_EFFORTS,
        supportsFastMode: false,
        supportsThinkingToggle: false,
        contextWindowOptions: [],
        promptInjectedEfforts: [],
        liveSteering,
      }
    case 'grok':
      return {
        family,
        effortOptions: GROK_BUILD_EFFORTS,
        supportsFastMode: false,
        supportsThinkingToggle: false,
        contextWindowOptions: [],
        promptInjectedEfforts: [],
        liveSteering,
      }
    case 'cursor': {
      const cursorEfforts = cursorEffortsForModel(model)
      return {
        family,
        effortOptions: cursorEfforts,
        supportsFastMode: /gpt-5|codex|fast/i.test(model ?? ''),
        supportsThinkingToggle: /thinking|\[.*thinking=/i.test(model ?? ''),
        contextWindowOptions: isCursorOneMillionContextModel(model) ? [{ value: '1m', label: '1M', tokens: 1_000_000, isDefault: true }] : [],
        promptInjectedEfforts: [],
        liveSteering,
      }
    }
    case 'opencode':
    case 'kilo':
    case 'pi':
      return {
        family,
        effortOptions: [],
        supportsFastMode: false,
        supportsThinkingToggle: false,
        contextWindowOptions: [],
        promptInjectedEfforts: [],
        liveSteering,
      }
    default:
      return {
        family,
        effortOptions: GENERIC_EFFORTS,
        supportsFastMode: false,
        supportsThinkingToggle: false,
        contextWindowOptions: [],
        promptInjectedEfforts: [],
        liveSteering,
      }
  }
}

export function providerIconName(provider: Provider | null | undefined, model?: string | null): IconName {
  switch (providerFamily(provider, model)) {
    case 'openai': return 'openai'
    case 'claude': return 'spark'
    case 'gemini': return 'spark'
    case 'grok': return 'bolt'
    case 'cursor': return 'code'
    case 'opencode': return 'terminal'
    case 'kilo': return 'layers'
    case 'pi': return 'cpu'
    case 'local': return 'cpu'
    default: return provider?.kind === 'local' ? 'cpu' : 'globe'
  }
}

export function providerEffortOptions(provider: Provider | null | undefined, model?: string | null): ProviderEffortOption[] {
  return providerModelCapabilities(provider, model).effortOptions
}

export function effortDisplayLabel(provider: Provider | null | undefined, model: string | null | undefined, effort: string | null | undefined): string {
  const value = (effort ?? '').trim()
  if (!value) return ''
  return providerEffortOptions(provider, model).find((option) => option.value === value)?.label
    ?? (value === 'xhigh' ? 'Extra High' : value === '512' ? '512 Tokens' : value.charAt(0).toUpperCase() + value.slice(1))
}

export function providerSupportsFastMode(provider: Provider | null | undefined, model?: string | null): boolean {
  return providerModelCapabilities(provider, model).supportsFastMode
}

export function providerContextWindowOptions(provider: Provider | null | undefined, model?: string | null): ProviderContextWindowOption[] {
  return providerModelCapabilities(provider, model).contextWindowOptions
}

export function modelContextWindowTokens(provider: Provider | null | undefined, model?: string | null): number | null {
  const text = `${provider?.id ?? ''} ${provider?.name ?? ''} ${provider?.transport ?? ''} ${provider?.source ?? ''} ${model ?? ''}`.toLowerCase()
  if (!text.trim()) return null
  if (/\b1m\b|\[1m\]|context=1m/.test(text)) return 1_000_000
  if (/qwen[-_.\s:]?3\.?6|qwen3\.?6/.test(text)) return 205_000
  if (/gpt-5(?:\.5|\.4|\.3|\.2)?|openai\/gpt-5/.test(text)) return 205_000
  if (/claude|opus|sonnet|haiku|anthropic/.test(text)) return 200_000
  if (/gemini/.test(text)) return 1_000_000
  if (/grok/.test(text)) return 256_000
  if (/gpt-4o|gpt-4\.1|gpt-4-turbo|o1|o3|o4/.test(text)) return 128_000
  const named = text.match(/(?:^|[-_.\s:])(\d{1,4})k(?:[-_.\s:]|$)/)
  if (named) {
    const size = Number(named[1])
    if (Number.isFinite(size) && size >= 4) return size * 1_000
  }
  return null
}

export function formatContextTokens(tokens: number): string {
  if (!Number.isFinite(tokens) || tokens <= 0) return '0'
  if (tokens >= 1_000_000) return Number.isInteger(tokens / 1_000_000) ? `${tokens / 1_000_000}M` : `${(tokens / 1_000_000).toFixed(1)}M`
  if (tokens >= 1000) {
    const value = tokens / 1000
    return Number.isInteger(value) ? `${value}k` : `${value.toFixed(value >= 100 ? 0 : 1)}k`
  }
  return String(Math.round(tokens))
}

export function estimateComposerTokens(text: string, attachments: Attachment[]): number {
  const attachmentChars = attachments.reduce((sum, attachment) => sum + (attachment.text?.length ?? attachment.name.length), 0)
  return BASE_COMPOSER_CONTEXT_TOKENS + Math.max(0, Math.ceil((text.length + attachmentChars) / 4))
}
