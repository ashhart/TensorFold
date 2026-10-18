import { currentModelQualification, modelTaskFit, type ModelQualificationReport, type ModelTaskRequirements } from '../../shared/model-fit'
import { isExternalProvider } from '../../shared/provider-trust'
import type { EngineSnapshot, Provider } from './types'
import type { RecentModel } from './recent-models'
import { providerModelGroups } from './provider-accounts'
import { statsModelLabel } from './turn-stats-line'
import { formatBytes } from './engine-format'
import { formatContextTokens } from './provider-model-metadata'

export interface ModelPickerChoice {
  key: string
  provider: Provider
  model: string
  accountId: string | null
  groupId: string
  groupLabel: string
  label: string
  detail: string
  state: string | null
  local: boolean
  disabledReason: string | null
  capabilityLabels?: string[]
  capabilityCheckedAt?: number
}

export function modelChoiceKey(providerId: string, model: string, accountId: string | null): string {
  return JSON.stringify([providerId, accountId, model])
}

/** The caller supplies policy-allowed providers, including for the recent list. */
export function modelPickerChoices(providers: readonly Provider[], engine: EngineSnapshot | null, fitOptions: { requirements?: ModelTaskRequirements; qualifications?: readonly ModelQualificationReport[] } = {}): ModelPickerChoice[] {
  return [...providers].sort((a, b) => Number(b.source === 'tensorfold') - Number(a.source === 'tensorfold'))
    .flatMap((provider) => providerModelGroups(provider).flatMap((group) => group.models.map((model): ModelPickerChoice => {
      const local = !isExternalProvider(provider)
      const builtIn = provider.source === 'tensorfold'
      const installed = builtIn ? engine?.installed.find((entry) => entry.repo === model) : null
      const status = engine?.status
      const ready = builtIn && status?.state === 'ready' && status.model === model
      const loading = builtIn && status?.state === 'starting' && status.model === model
      const tooBig = installed?.fitsMemory === false
        || installed?.memoryFloorBytes != null && status != null && status.memoryLimitBytes > 0 && installed.memoryFloorBytes > status.memoryLimitBytes
      const disabledReason = builtIn && status?.available === false ? 'The built-in engine is unavailable.'
        : builtIn && engine && !installed ? 'This model is no longer installed. Open Models to download it.'
        : tooBig ? 'This model needs more memory than this Mac allows.'
        : installed?.drafter && installed.drafterInstalled === false ? 'Download the companion drafter in Models before using this model.'
        : null
      const qualification = currentModelQualification(fitOptions.qualifications ?? [], { providerId: provider.id, accountId: group.accountId, model })
      // External routes remain deliberate planners; they never acquire local tools.
      const requirements = !local && fitOptions.requirements?.tools ? { ...fitOptions.requirements, tools: false } : fitOptions.requirements
      const fit = modelTaskFit({ local, report: qualification, requirements, toolPolicyAllowed: local })
      // The catalog maximum is not the window this engine will actually serve.
      const context = ready ? status?.contextTokens : null
      const facts = builtIn ? [
        installed?.format,
        context && context > 0 ? `${formatContextTokens(context)} context` : null,
        installed?.memoryFloorBytes != null ? `${formatBytes(installed.memoryFloorBytes)} memory` : null,
      ].filter(Boolean).join(' · ') : ''
      const state = !builtIn ? null : disabledReason ? 'Unavailable' : loading ? 'Loading' : ready ? 'Ready' : engine ? 'Loads on first message' : 'Checking engine'
      return {
        key: modelChoiceKey(provider.id, model, group.accountId), provider, model, accountId: group.accountId,
        groupId: group.id, groupLabel: group.label,
        label: installed?.title ?? statsModelLabel(model, provider, engine?.suggestions),
        detail: facts || (builtIn ? 'Runs with the built-in TensorFold engine' : `${local ? 'On this Mac' : 'Planning only'}${group.accountId ? ` · ${group.label}` : ` · ${provider.name}`}`),
        state, local, disabledReason: disabledReason ?? fit.disabledReason, capabilityLabels: fit.labels, capabilityCheckedAt: qualification?.checkedAt,
      }
    })))
}

export function filterModelChoices(choices: readonly ModelPickerChoice[], query: string, scope: 'all' | 'local' | 'external'): ModelPickerChoice[] {
  const words = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  return choices.filter((choice) => (scope === 'all' || choice.local === (scope === 'local')) && words.every((word) =>
    `${choice.label} ${choice.model} ${choice.provider.name} ${choice.groupLabel} ${choice.detail}`.toLocaleLowerCase().includes(word)))
}

/** Resolve against the exact selectable account/model routes, not a provider's default list. */
export function recentModelChoices(choices: readonly ModelPickerChoice[], recent: readonly RecentModel[], activeKey: string): ModelPickerChoice[] {
  const byKey = new Map(choices.map((choice) => [choice.key, choice]))
  const seen = new Set<string>([activeKey])
  return recent.flatMap((entry) => {
    const key = modelChoiceKey(entry.providerId, entry.model, entry.accountId ?? null)
    const choice = byKey.get(key)
    if (!choice || seen.has(key) || choice.disabledReason) return []
    seen.add(key)
    return [choice]
  }).slice(0, 3)
}
