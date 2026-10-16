import os from 'node:os'
import path from 'node:path'
import type { Provider, ProviderAccountProfile, Settings } from './types'

function nonEmpty(value?: string | null): string | undefined {
  const trimmed = value?.trim()
  return trimmed ? trimmed : undefined
}

function expandHome(value: string): string {
  if (value === '~') return os.homedir()
  if (value.startsWith('~/')) return path.join(os.homedir(), value.slice(2))
  return value
}

export function enabledProviderAccounts(provider: Provider): ProviderAccountProfile[] {
  return (provider.accounts ?? []).filter((account) => account.enabled !== false)
}

export function selectProviderAccount(provider: Provider, accountId?: string | null): ProviderAccountProfile | null {
  const accounts = enabledProviderAccounts(provider)
  if (!accounts.length) return null
  if (accountId === null) return null
  if (accountId) {
    const selected = accounts.find((account) => account.id === accountId)
    if (selected) return selected
  }
  return null
}

export function providerWithAccount(provider: Provider, accountId?: string | null): Provider {
  const account = selectProviderAccount(provider, accountId)
  if (!account) return provider

  const cliEnv: Record<string, string> = { ...(provider.cliEnv ?? {}), ...(account.env ?? {}) }
  const cliHome = nonEmpty(account.cliHome)
  if (cliHome) {
    const home = expandHome(cliHome)
    if (provider.transport === 'codex-cli') cliEnv.CODEX_HOME = home
    if (provider.transport === 'claude-cli') {
      cliEnv.CLAUDE_CONFIG_DIR = home
      cliEnv.CLAUDE_HOME = home
    }
  }

  const cliProfile = nonEmpty(account.cliProfile)
  return {
    ...provider,
    apiKey: nonEmpty(account.apiKey) ?? provider.apiKey,
    models: account.models?.length ? account.models : provider.models,
    cliProfile: cliProfile && provider.transport === 'claude-cli' ? expandHome(cliProfile) : (cliProfile ?? provider.cliProfile),
    cliEnv,
  }
}

export function modelChoicesForProvider(provider: Provider, accountId?: string | null): string[] {
  const account = selectProviderAccount(provider, accountId)
  return account?.models?.length ? account.models : provider.models
}

function modelAlias(model: string): string {
  const lower = model.toLowerCase()
  if (lower === 'opus-4.8' || lower === 'opus 4.8' || lower === 'claude-opus-4.8') return 'opus'
  if (lower === 'ultracode') return 'opus'
  return model
}

export function resolveProviderModel(provider: Provider, accountId?: string | null, ...requestedModels: Array<string | null | undefined>): string | null {
  const requested = requestedModels.map((model) => model?.trim()).filter((model): model is string => !!model).map((model) => provider.transport === 'claude-cli' ? modelAlias(model) : model)
  const choices = modelChoicesForProvider(provider, accountId)
  if (!choices.length) return requested[0] ?? null
  return requested.find((model) => choices.includes(model)) ?? choices[0] ?? null
}

// Per-surface default route (VST-015), main-process side. Returns the surface's dedicated
// provider/account/model ONLY when a complete override (provider + model) is configured;
// null means "no override — use the active/global route". Mirrors the renderer's
// resolveSurfaceModel (src/lib/assistant-model.ts); the parity test in
// assistant-model.test.ts pins both to identical precedence.
export type ModelSurface = 'chat' | 'code'

export function surfaceModelOverride(
  settings: Pick<Settings, 'assistantProviderId' | 'assistantProviderAccountId' | 'assistantModel' | 'codeProviderId' | 'codeProviderAccountId' | 'codeModel'>,
  surface: ModelSurface,
): { providerId: string; providerAccountId: string | null; model: string } | null {
  const providerId = surface === 'chat' ? settings.assistantProviderId : settings.codeProviderId
  const model = surface === 'chat' ? settings.assistantModel : settings.codeModel
  if (!providerId || !model) return null
  return {
    providerId,
    providerAccountId: (surface === 'chat' ? settings.assistantProviderAccountId : settings.codeProviderAccountId) ?? null,
    model,
  }
}
