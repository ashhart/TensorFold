import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { isExternalProvider } from '../../../shared/provider-trust'
import { Menu, MenuItem } from '../ui'
import type { Provider, Project, TokenUsageSnapshot } from '../../lib/types'
import type { RecentModel } from '../../lib/recent-models'
import { accountLabel, modelDisplayName, providerModelGroups } from '../../lib/provider-accounts'
import { modelUsageLabel } from '../../lib/provider-usage'
import { providerIconName } from '../../lib/provider-model-metadata'
import type { ProviderUsageSummary } from '../../lib/provider-usage'
import type { ProviderStatusView } from '../../lib/provider-status'
import { UsagePill } from './UsagePill'
import { RateLimitBar } from './RateLimitBar'
import { EngineDot, EngineStatusRow } from '../models/EngineStatus'
import { useStore } from '../../lib/store'
import { useEngine } from '../../lib/engine'
import { statsModelLabel } from '../../lib/turn-stats-line'

// Built-in engine models read by their catalog title ("Qwen3.5-2B"), not the repo id.
function EngineModelName({ provider, model }: { provider: Provider; model: string | null }) {
  const { snapshot } = useEngine()
  return <>{model ? statsModelLabel(model, provider, snapshot?.suggestions) : modelDisplayName(provider, model)}</>
}

export interface ModelPickerMenuProps {
  /** Active provider (already resolved against policy). */
  provider: Provider
  /** Active model id on the active provider. */
  model: string | null
  /** Active account id on the active provider, if any. */
  providerAccountId: string | null
  /** Local-only policy hides cloud providers and shows the pinning reason. */
  localOnly: boolean
  /** Project the local-only policy comes from (for the banner label). */
  policyProject: Project | null
  /** Providers allowed under the active project policy (the pickable list). */
  pickerProviders: Provider[]
  /** Recently used models (cross-thread), shown as one-click chips at the top. */
  recentModels: RecentModel[]
  /** All configured providers — used to resolve recent/group provider lookups. */
  providers: Provider[]
  /** Usage snapshot for per-model "used N tokens ago" hints. */
  usageSnapshot: TokenUsageSnapshot | null
  /** Rolled-up usage pills (usage / budget / route), null when no usage data. */
  usageSummary: ProviderUsageSummary | null
  /** Real rate-limit windows for the active provider/account, null when none. */
  rateLimitStatus: ProviderStatusView | null
  /** Apply a model pick (provider, model, account) — persists to thread/settings. */
  onPickModel: (provider: Provider, model: string, accountId: string | null) => void
  /** Run settings (effort, speed, context) shown above the model list. */
  settings?: React.ReactNode
  /** Short qualifier after the model name in the pill, e.g. the reasoning effort. */
  detail?: string
}

const PLANNING_ONLY_HINT = 'Planning only: this provider gets a redacted transcript, has no tools and never sees or touches your files.'

function PlanningOnlyBadge() {
  return <span title={PLANNING_ONLY_HINT} className="shrink-0 rounded-full border border-line px-1.5 py-px text-[12px] font-normal text-ink-3">Planning only</span>
}

export function ModelPickerMenu({
  provider,
  model,
  providerAccountId,
  localOnly,
  policyProject,
  pickerProviders,
  recentModels,
  providers,
  usageSnapshot,
  usageSummary,
  rateLimitStatus,
  onPickModel,
  settings,
  detail,
}: ModelPickerMenuProps) {
  const activeAccountText = accountLabel(provider, providerAccountId)
  const activeModelText = modelDisplayName(provider, model)
  const setView = useStore((s) => s.setView)
  // The built-in engine's models come first.
  const orderedProviders = [...pickerProviders].sort((a, b) => Number(b.source === 'tensorfold') - Number(a.source === 'tensorfold'))
  return (
    <Menu
      width={356}
      align="end"
      placement="top"
      trigger={(open) => (
        <button title="Model and run settings" className={clsx('flex h-8 min-w-0 items-center gap-1.5 rounded-full px-2.5 text-[13px] transition-colors', open ? 'bg-ink-5 text-ink' : 'text-ink-2 hover:bg-ink-5 hover:text-ink')}>
          {provider.source === 'tensorfold' ? <EngineDot /> : <Icon name={providerIconName(provider, model)} size={16} className="shrink-0 text-ink-3" />}
          <span className="max-w-[220px] truncate">{provider.source === 'tensorfold' ? <EngineModelName provider={provider} model={model} /> : activeModelText}{activeAccountText ? ` · ${activeAccountText}` : ''}</span>
          {detail && <span className="shrink-0 text-ink-3 @max-md:hidden">{detail}</span>}
          {isExternalProvider(provider) && <PlanningOnlyBadge />}
          <Icon name="chevronDown" size={14} className="shrink-0 text-ink-3" />
        </button>
      )}
    >
      {(close) => (
        <div className="max-h-[420px] overflow-y-auto scroll-thin">
          {orderedProviders.some((p) => p.source === 'tensorfold') && (
            <EngineStatusRow onOpenModels={() => { close(); setView({ name: 'models' }) }} />
          )}
          {settings}
          {localOnly && (
            <div className="mb-1 flex items-center gap-1.5 px-2.5 py-1.5 text-[12px] text-ink-3"><Icon name="cpu" size={11} className="text-accent" />Local models only · {policyProject?.name}</div>
          )}
          {usageSummary && (
            <div className="mb-1 grid grid-cols-3 gap-1.5 px-1 pb-1">
              <UsagePill label="Usage" value={usageSummary.usage} detail={usageSummary.usageDetail} tone={usageSummary.usage === 'No usage' ? 'muted' : 'normal'} />
              <UsagePill label="Budget" value={usageSummary.budget} detail={usageSummary.budgetDetail} tone={usageSummary.budget === 'No budget' ? 'muted' : 'normal'} />
              <UsagePill label="Route" value={usageSummary.route} detail={usageSummary.routeDetail} tone={usageSummary.routeTone} />
            </div>
          )}
          {rateLimitStatus && (
            <div className="mb-1 space-y-1 px-1 pb-1">
              {rateLimitStatus.windows.map((w) => <RateLimitBar key={w.id} view={w} />)}
            </div>
          )}
          {/* Recently used models, one click to switch — folded in from the old
              quick-switch chips so the picker is the single place to change model. */}
          {(() => {
            const recents = recentModels.filter((rm) => {
              const p = providers.find((x) => x.id === rm.providerId)
              return p && p.models.includes(rm.model) && !(rm.providerId === provider.id && rm.model === model && (providerAccountId ?? null) === rm.accountId)
            }).slice(0, 4)
            if (!recents.length) return null
            return (
              <div className="mb-1">
                <div className="px-2.5 pb-1 pt-2 text-[13px] font-medium text-ink-3">Recent</div>
                {recents.map((rm) => {
                  const p = providers.find((x) => x.id === rm.providerId)!
                  return (
                    <MenuItem key={`recent:${rm.providerId}:${rm.accountId ?? ''}:${rm.model}`} icon={providerIconName(p, rm.model)} onClick={() => { onPickModel(p, rm.model, rm.accountId); close() }}>
                      <span className="truncate">{modelDisplayName(p, rm.model)}</span>
                    </MenuItem>
                  )
                })}
              </div>
            )
          })()}
          {orderedProviders.flatMap((p) => providerModelGroups(p).map((group) => (
            <div key={group.id} className="mb-1">
              <div className="flex items-center gap-1.5 px-2.5 pb-1 pt-2 text-[13px] font-medium text-ink-3">
                <Icon name={providerIconName(p, group.models[0])} size={14} />
                <span className="truncate">{group.label}</span>
                {group.detail && <span className="truncate font-normal text-ink-4">· {group.detail}</span>}
                {isExternalProvider(p) && <span className="ml-auto"><PlanningOnlyBadge /></span>}
              </div>
              {!group.models.length && <div className="px-2.5 py-1 text-[13px] text-ink-3">{p.source === 'tensorfold' ? 'No models yet — download one in Models' : 'No models — Connect in Settings'}</div>}
              {group.models.map((m) => {
                const usage = modelUsageLabel(m, usageSnapshot)
                return (
                  <MenuItem key={`${group.id}:${m}`} active={p.id === provider.id && m === model && (providerAccountId ?? null) === group.accountId} onClick={() => { onPickModel(p, m, group.accountId); close() }}>
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="min-w-0 flex-1 truncate">{p.source === 'tensorfold' ? <EngineModelName provider={p} model={m} /> : modelDisplayName(p, m)}</span>
                      {usage && <span className="shrink-0 text-[12px] text-ink-4">{usage}</span>}
                    </span>
                  </MenuItem>
                )
              })}
            </div>
          )))}
        </div>
      )}
    </Menu>
  )
}
