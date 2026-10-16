import type { Provider, ProviderAccountProfile } from './types'
import { cliModelCatalog } from './cli-models'

export interface ProviderAccountChoice {
  id: string | null
  name: string
  models: string[]
  account: ProviderAccountProfile | null
}

export interface ProviderModelGroup {
  id: string
  label: string
  detail: string | null
  accountId: string | null
  models: string[]
  account: ProviderAccountProfile | null
}

const LOCAL_SOURCE_LABELS: Partial<Record<NonNullable<Provider['source']>, string>> = {
  omlx: 'oMLX',
  ollama: 'Ollama',
  lmstudio: 'LM Studio',
}

function titleWord(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1).toLowerCase()
}

function cliFamily(provider: Provider): string | null {
  if (provider.transport === 'codex-cli') return 'Codex'
  if (provider.transport === 'claude-cli') return 'Claude'
  return null
}

function accountDisplayName(provider: Provider, account: ProviderAccountProfile | null): string {
  const family = cliFamily(provider)
  const name = account?.name?.trim()
  if (!name) return family ? `${family} account` : 'Account'
  if (!family) return name

  const familyLower = family.toLowerCase()
  const lower = name.toLowerCase().replace(/\s+/g, ' ').trim()
  const tier = lower.match(/^(?:personal|work)\b/)?.[0] ?? lower.match(/\b(personal|work)$/)?.[1]
  if (tier && (lower === `${familyLower} ${tier}` || lower === `${tier} ${familyLower}`)) return `${titleWord(tier)} ${family}`
  return name
}

function localGroupLabel(provider: Provider): string {
  const sourceLabel = provider.source ? LOCAL_SOURCE_LABELS[provider.source] : null
  if (sourceLabel) return `${sourceLabel} (local)`
  const name = provider.name.trim() || 'Local provider'
  return /\blocal\b/i.test(name) ? name : `${name} (local)`
}

// Utility/non-chat models (e.g. oMLX's MarkItDown, embedding/rerank/whisper models)
// aren't usable for chat/coding — hide them from selectable model lists.
const NON_CHAT_MODEL = /^markitdown$|(^|[-_/ ])(embed(ding)?s?|rerank(er)?|whisper|moderation)([-_/ ]|$)/i
function providerModels(provider: Provider, models: string[] | undefined): string[] {
  return cliModelCatalog(provider.transport, models?.length ? models : provider.models).filter((m) => !NON_CHAT_MODEL.test(m))
}

export function providerAccountChoices(provider: Provider): ProviderAccountChoice[] {
  const accounts = (provider.accounts ?? []).filter((account) => account.enabled !== false)
  const choices: ProviderAccountChoice[] = [{
    id: null,
    name: provider.transport === 'codex-cli' || provider.transport === 'claude-cli' ? 'Default login' : 'Provider key',
    models: providerModels(provider, provider.models),
    account: null,
  }]
  for (const account of accounts) choices.push({
    id: account.id,
    name: accountDisplayName(provider, account),
    models: providerModels(provider, account.models),
    account,
  })
  return choices
}

export function providerModelGroups(provider: Provider): ProviderModelGroup[] {
  const choices = providerAccountChoices(provider)
  const namedAccounts = choices.filter((choice) => choice.id !== null)
  const groups = namedAccounts.length ? namedAccounts : choices.slice(0, 1)
  return groups.map((choice) => {
    const label = namedAccounts.length ? choice.name : (provider.kind === 'local' ? localGroupLabel(provider) : provider.name)
    const detail = namedAccounts.length
      ? provider.name
      : provider.kind === 'local'
        ? (label !== provider.name ? provider.name : null)
        : choice.name
    return {
      id: `${provider.id}:${choice.id ?? 'default'}`,
      label,
      detail,
      accountId: choice.id,
      models: choice.models.length ? choice.models : providerModels(provider, provider.models),
      account: choice.account,
    }
  })
}

export function accountLabel(provider: Provider | null | undefined, accountId?: string | null): string | null {
  if (!provider) return null
  const choice = providerAccountChoices(provider).find((account) => account.id === (accountId ?? null))
  if (!choice || (!accountId && !(provider.accounts?.length))) return null
  return choice.name
}

export function modelChoicesForProvider(provider: Provider, accountId?: string | null): string[] {
  return providerAccountChoices(provider).find((choice) => choice.id === (accountId ?? null))?.models ?? provider.models
}

function modelAlias(model: string): string {
  const lower = model.toLowerCase()
  if (lower === 'opus-4.8' || lower === 'opus 4.8' || lower === 'claude-opus-4.8') return 'opus'
  if (lower === 'ultracode') return 'opus'
  return model
}

export function resolveProviderModel(provider: Provider, accountId?: string | null, ...requestedModels: Array<string | null | undefined>): string | null {
  const requested = requestedModels
    .map((model) => model?.trim())
    .filter((model): model is string => !!model)
    .flatMap((model) => {
      if (provider.transport !== 'claude-cli') return [model]
      const alias = modelAlias(model)
      return alias === model ? [model] : [model, alias]
    })
  const choices = modelChoicesForProvider(provider, accountId)
  if (!choices.length) return requested[0] ?? null
  return requested.find((model) => choices.includes(model)) ?? choices[0] ?? null
}

export function modelDisplayName(provider: Provider | null | undefined, model: string | null | undefined): string {
  const value = model?.trim()
  if (!value) return 'Select model'
  if (provider?.transport === 'claude-cli') {
    if (value === 'claude-code-default') return 'Claude default'
    if (value === 'opus' || value === 'opus-4.8' || value === 'claude-opus-4.8') return 'Claude Opus'
    if (value === 'ultracode') return 'Claude UltraCode'
    if (value === 'sonnet') return 'Sonnet'
  }
  if (provider?.transport === 'codex-cli') {
    if (value === 'codex-cli-default') return 'Codex default'
    if (value === 'gpt-5.4') return 'GPT-5.4'
    if (value === 'gpt-5.5') return 'GPT-5.5'
    if (value === 'gpt-5.5-codex-high') return 'GPT-5.5 Codex High'
    if (value === 'gpt-5.5-extra-high') return 'GPT-5.5 Extra High'
  }
  if (/^gpt-\d/i.test(value)) return value.replace(/^gpt-/i, 'GPT-')
  return value
}
