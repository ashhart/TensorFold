import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { modelFitReason, modelCapabilityLabel, modelChoiceDetail } from '../../lib/preferences/display'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { ModelQualificationReport, ModelTaskRequirements } from '../../../shared/model-fit'
import { useEffect, useId, useRef, useState } from 'react'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import { isExternalProvider } from '../../../shared/provider-trust'
import { api } from '../../lib/ipc'
import { Menu } from '../ui'
import type { Provider, Project, TokenUsageSnapshot } from '../../lib/types'
import type { RecentModel } from '../../lib/recent-models'
import { accountLabel, modelDisplayName } from '../../lib/provider-accounts'
import { modelUsageLabel } from '../../lib/provider-usage'
import { providerIconName } from '../../lib/provider-model-metadata'
import type { ProviderUsageSummary } from '../../lib/provider-usage'
import type { ProviderStatusView } from '../../lib/provider-status'
import { UsagePill } from './UsagePill'
import { RateLimitBar } from './RateLimitBar'
import { EngineDot } from '../models/EngineStatus'
import { useStore } from '../../lib/store'
import { useLearning, learningBusy } from '../../lib/learning'
import { useEngine } from '../../lib/engine'
import { statsModelLabel } from '../../lib/turn-stats-line'
import { filterModelChoices, modelChoiceKey, modelPickerChoices, recentModelChoices, type ModelPickerChoice } from '../../lib/model-picker'
import { favoriteModelChoices, orderFavoritesFirst, useFavoriteModels } from '../../lib/favorite-models'

export interface ModelPickerMenuProps {
  provider: Provider
  model: string | null
  providerAccountId: string | null
  localOnly: boolean
  policyProject: Project | null
  pickerProviders: Provider[]
  recentModels: RecentModel[]
  usageSnapshot: TokenUsageSnapshot | null
  usageSummary: ProviderUsageSummary | null
  rateLimitStatus: ProviderStatusView | null
  onPickModel: (provider: Provider, model: string, accountId: string | null) => void | Promise<void>
  settings?: React.ReactNode
  detail?: string
  disabled?: boolean
  ownerId?: string | null
  requirements?: ModelTaskRequirements
  qualifications?: ModelQualificationReport[]
}

const PLANNING_ONLY_HINT = 'Planning only: this provider gets a redacted transcript, has no tools and never sees or touches your files.'
function PlanningOnlyBadge() {
  const tx = useAuthoredText()
  return <span title={tx("Planning only: this provider gets a redacted transcript, has no tools and never sees or touches your files.")} className="shrink-0 rounded-full border border-line px-1.5 py-px text-[12px] font-normal text-ink-2">{tx("Planning only")}</span>
}

function ModelPickerBody({ close, ...props }: ModelPickerMenuProps & { close: () => void }) {
  const tx = useAuthoredText()
  const { snapshot } = useEngine()
  const { snapshot: learning } = useLearning()
  const setView = useStore((s) => s.setView)
  const [query, setQuery] = useState('')
  const [scope, setScope] = useState<'all' | 'local' | 'external'>('all')
  const [showSettings, setShowSettings] = useState(false)
  const [pending, setPending] = useState<string | null>(null)
  const [error, setError] = useState('')
  const pickLock = useRef(false)
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLDivElement>(null)
  const id = useId()
  const favorites = useFavoriteModels(props.ownerId ?? null)
  const route = (choice: ModelPickerChoice) => ({ providerId: choice.provider.id, accountId: choice.accountId, model: choice.model })
  const choices = modelPickerChoices(props.pickerProviders, snapshot, { requirements: props.requirements, qualifications: props.qualifications }).filter((choice) => !props.localOnly || choice.local).map((choice) => learningBusy(learning) && (choice.provider.source === 'tensorfold' || choice.provider.id === 'tensorfold') && choice.model !== learning?.model
    ? { ...choice, disabledReason: 'Finish the current lesson before changing its local model.' } : choice)
  const shown = orderFavoritesFirst(filterModelChoices(choices, query, scope), favorites.favorites, route)
  const activeKey = modelChoiceKey(props.provider.id, props.model ?? '', props.providerAccountId)
  const starred = !query.trim() ? favoriteModelChoices(shown, favorites.favorites, route) : []
  const starredKeys = new Set(starred.map((choice) => choice.key))
  const recents = !query.trim() && scope === 'all' ? recentModelChoices(choices, props.recentModels, activeKey).filter((choice) => !starredKeys.has(choice.key)) : []
  const grouped = shown.filter((choice) => !starredKeys.has(choice.key))
  const groups = [...new Set(grouped.map((choice) => choice.groupId))]
  const builtIn = props.pickerProviders.some((p) => p.source === 'tensorfold')
  useEffect(() => { input.current?.focus() }, [])
  const pick = async (choice: ModelPickerChoice) => {
    if (choice.disabledReason || pickLock.current) return
    pickLock.current = true; setPending(choice.key); setError('')
    try {
      await props.onPickModel(choice.provider, choice.model, choice.accountId)
      close()
    } catch (failure) {
      setError(String((failure as Error)?.message ?? failure).replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, ''))
    } finally { pickLock.current = false; setPending(null) }
  }
  const navigate = (event: React.KeyboardEvent) => {
    if (event.nativeEvent.isComposing) return
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); return }
    if (event.key === 'Enter' && event.target === input.current && !event.nativeEvent.isComposing) {
      event.preventDefault()
      const first = (!query.trim() ? shown.find((choice) => choice.key === activeKey && !choice.disabledReason) : undefined)
        ?? shown.find((choice) => !choice.disabledReason)
      if (first) pick(first)
      return
    }
    if (!['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) return
    const fromSearch = event.target === input.current
    if (fromSearch && (event.key === 'Home' || event.key === 'End')) return
    if (!fromSearch && !(event.target instanceof HTMLElement && event.target.hasAttribute('data-model-choice'))) return
    const items = [...list.current?.querySelectorAll<HTMLButtonElement>('button[data-model-choice]:not(:disabled)') ?? []]
    if (!items.length) return
    event.preventDefault()
    const current = items.indexOf(event.target as HTMLButtonElement)
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1
      : current < 0 ? event.key === 'ArrowUp' ? items.length - 1 : 0
      : (current + (event.key === 'ArrowUp' ? -1 : 1) + items.length) % items.length
    items[next].focus()
  }
  const row = (choice: ModelPickerChoice, recent = false) => {
    const active = choice.key === activeKey
    const usage = modelUsageLabel(choice.model, props.usageSnapshot)
    const favorite = favorites.isFavorite(route(choice))
    return <div key={`${recent ? 'recent:' : ''}${choice.key}`} className={clsx('flex min-w-0 items-center rounded-lg', active && 'bg-accent/8')}><button type="button" data-model-choice={choice.key} aria-pressed={active} disabled={!!choice.disabledReason || pending != null} title={choice.disabledReason ? modelFitReason(choice.disabledReason, tx) : choice.model} onClick={() => void pick(choice)}
      className="group flex min-w-0 flex-1 items-start gap-3 rounded-lg px-3 py-2.5 text-left transition-colors hover:bg-ink-5 focus-visible:outline-2 focus-visible:-outline-offset-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-55">
      <span className={clsx('mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-line', choice.local ? 'bg-accent/8 text-accent' : 'bg-card-2 text-ink-2')}><Icon name={providerIconName(choice.provider, choice.model)} size={15} /></span>
      <span className="min-w-0 flex-1">
        <span className="flex min-w-0 items-center gap-2"><span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{choice.label}</span>{active && <Icon name="check" size={14} className="shrink-0 text-accent" />}</span>
        <span className="mt-0.5 block truncate text-[12px] leading-5 text-ink-2">{choice.disabledReason ? modelFitReason(choice.disabledReason, tx) : recent ? `${choice.groupLabel} · ${modelChoiceDetail(choice.detail, tx)}` : modelChoiceDetail(choice.detail, tx)}</span>
        <span className="mt-1 flex flex-wrap gap-x-2 text-[12px] leading-5 text-ink-3">{choice.capabilityLabels?.map(label => <span key={label}>{modelCapabilityLabel(label, tx)}</span>)}</span>
        {choice.capabilityCheckedAt && <span className="mt-1 block text-[12px] leading-5 text-ink-3">{tx("Checked ")}{new Date(choice.capabilityCheckedAt).toLocaleString(currentIntlLocale())} {tx("for this account · probe 1")}</span>}
        {(choice.state || usage || pending === choice.key) && <span className="mt-1 flex flex-wrap gap-x-2 text-[12px] leading-4"><span className={choice.disabledReason ? 'text-warn' : 'text-ink-2'}>{pending === choice.key ? tx("Selecting…") : choice.state ? tx(choice.state) : null}</span>{usage && <span className="text-ink-3">{usage}</span>}</span>}
      </span>
    </button><button type="button" aria-label={favorite ? tx("Remove {model} from favourites", { model: choice.label }) : tx("Add {model} to favourites", { model: choice.label })} aria-pressed={favorite} disabled={!favorites.ready || pending != null} title={favorite ? tx("Remove favourite") : tx("Add favourite")} onClick={() => favorites.toggle(route(choice))} className={clsx('mr-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg transition-colors hover:bg-ink-5 focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-50', favorite ? 'text-accent' : 'text-ink-3 hover:text-ink')}><Icon name={favorite ? 'starFilled' : 'star'} size={15} /></button></div>
  }
  return <div role="dialog" aria-label={tx("Choose a model")} onKeyDown={navigate} className="flex max-h-[min(560px,70vh)] min-h-0 flex-col">
    <div className="shrink-0 px-3 pb-2 pt-2">
      <div className="mb-3 flex items-center justify-between gap-3"><h2 className="text-[14px] font-semibold text-ink">{tx("Choose a model")}</h2><span className="text-[12px] text-ink-2">{props.localOnly ? tx("Local only") : tx("{value0} models", { value0: choices.length })}</span></div>
      <div className="flex items-center gap-2 rounded-lg border border-line bg-surface px-2.5 focus-within:border-accent focus-within:ring-2 focus-within:ring-accent/10"><Icon name="search" size={15} className="shrink-0 text-ink-3" /><input ref={input} value={query} onChange={(e) => setQuery(e.target.value)} aria-label={tx("Search models")} aria-controls={`${id}-models`} placeholder={tx("Search models or providers")} className="h-9 min-w-0 flex-1 bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-3" />{query && <button type="button" aria-label={tx("Clear model search")} onClick={() => { setQuery(''); input.current?.focus() }} className="rounded p-1 text-ink-2 hover:bg-ink-5"><Icon name="x" size={12} /></button>}</div>
      {!props.localOnly && <div className="mt-2 flex gap-1" aria-label={tx("Model location")}>{([['all', tx('All')], ['local', api.platform === 'darwin' ? tx('On this Mac') : tx('Local')], ['external', tx('External')]] as const).map(([value, label]) => <button key={value} type="button" aria-pressed={scope === value} onClick={() => setScope(value)} className={clsx('rounded-md px-2.5 py-1.5 text-[12px] transition-colors', scope === value ? 'bg-ink-5 font-medium text-ink' : 'text-ink-2 hover:bg-ink-5')}>{label}</button>)}</div>}
      {(props.requirements?.images || props.requirements?.tools || props.requirements?.reasoning) && <p className="mt-2 text-[12px] text-ink-2">{tx("This task requires ")}{[props.requirements.images && 'images', props.requirements.tools && 'tools', props.requirements.reasoning && 'reasoning'].filter(Boolean).join(', ')}{tx("; unknown capabilities have not been qualified.")}</p>}
      {props.localOnly && <p className="mt-2 text-[12px] text-ink-2">{props.policyProject?.name ?? tx("This conversation")} {tx("keeps work on this Mac.")}</p>}
    </div>
    <div ref={list} id={`${id}-models`} className="scroll-thin min-h-0 flex-1 overflow-y-auto border-t border-line px-1 py-1">
      {starred.length > 0 && <div className="mb-1"><p className="px-3 pb-1 pt-2 text-[12px] font-medium text-ink-2">{tx("Favourites")}</p>{starred.map((choice) => row(choice))}</div>}
      {recents.length > 0 && <div className="mb-1"><p className="px-3 pb-1 pt-2 text-[12px] font-medium text-ink-2">{tx("Recently used")}</p>{recents.map((choice) => row(choice, true))}</div>}
      {groups.map((groupId) => {
        const entries = grouped.filter((choice) => choice.groupId === groupId)
        const first = entries[0]
        return <div key={groupId} className="mb-1"><div className="flex items-center justify-between gap-2 px-3 pb-1 pt-2"><span className="truncate text-[12px] font-medium text-ink-2">{first.groupLabel}</span>{!first.local && <PlanningOnlyBadge />}</div>{entries.map((choice) => row(choice))}</div>
      })}
      {!shown.length && <div role="status" className="px-4 py-7 text-center"><p className="text-[13px] font-medium text-ink">{query ? tx("No matching models") : scope === 'local' ? tx("No local models yet") : tx("No models available")}</p><p className="mt-1 text-[12px] leading-5 text-ink-2">{query ? tx("Try a model name, provider or account.") : props.localOnly || scope === 'local' ? (api.platform === 'darwin' ? tx("Download a model that fits this Mac to get started.") : tx("Download a local model or connect a provider to get started.")) : tx("Download a local model or connect a provider to get started.")}</p>{query && <button type="button" onClick={() => { setQuery(''); setScope('all'); input.current?.focus() }} className="mt-3 rounded-md px-3 py-1.5 text-[12px] font-medium text-accent hover:bg-accent/8">{tx("Clear search")}</button>}</div>}
    </div>
    <div className="shrink-0 border-t border-line px-2 pb-1 pt-1">
      {error && <p role="alert" className="px-2 py-2 text-[12px] leading-5 text-danger">{tx("Could not switch models: ")}{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
      {favorites.error && <p role="alert" className="px-2 py-2 text-[12px] leading-5 text-danger">{<OriginalDiagnostic>{favorites.error}</OriginalDiagnostic>}</p>}
      {showSettings && <div className="max-h-[220px] overflow-y-auto px-1 py-2">{props.settings}{props.usageSummary && <div className="grid grid-cols-3 gap-1.5 pb-1"><UsagePill label={tx("Usage")} value={props.usageSummary.usage} detail={props.usageSummary.usageDetail} /><UsagePill label={tx("Budget")} value={props.usageSummary.budget} detail={props.usageSummary.budgetDetail} /><UsagePill label={tx("Route")} value={props.usageSummary.route} detail={props.usageSummary.routeDetail} tone={props.usageSummary.routeTone} /></div>}{props.rateLimitStatus?.windows.map((window) => <RateLimitBar key={window.id} view={window} />)}</div>}
      <div className="flex items-center justify-between gap-2">{props.settings ? <button type="button" aria-expanded={showSettings} onClick={() => setShowSettings((value) => !value)} className="flex items-center gap-1.5 rounded-md px-2 py-2 text-[12px] text-ink-2 hover:bg-ink-5 hover:text-ink"><Icon name="sliders" size={14} />{tx("Run settings")}<Icon name={showSettings ? 'chevronUp' : 'chevronDown'} size={12} /></button> : <span />}
        <button type="button" onClick={() => { close(); setView(builtIn && scope !== 'external' ? { name: 'models' } : { name: 'settings' }) }} className="flex items-center gap-1.5 rounded-md px-2 py-2 text-[12px] font-medium text-accent hover:bg-accent/8"><Icon name={builtIn && scope !== 'external' ? 'cpu' : 'plus'} size={14} />{builtIn && scope !== 'external' ? tx("Manage models") : tx("Connect provider")}<Icon name="arrowRight" size={12} /></button></div>
    </div>
  </div>
}

export function ModelPickerMenu(props: ModelPickerMenuProps) {
  const tx = useAuthoredText()
  const { provider, model, providerAccountId, detail, disabled } = props
  const { snapshot } = useEngine()
  const { snapshot: learning } = useLearning()
  const trigger = useRef<HTMLButtonElement>(null)
  const label = !model?.trim() ? tx('Choose a model') : provider.source === 'tensorfold' && model ? snapshot?.installed.find((entry) => entry.repo === model)?.title ?? statsModelLabel(model, provider, snapshot?.suggestions) : modelDisplayName(provider, model)
  const account = accountLabel(provider, providerAccountId)
  return <Menu width={400} align="end" placement="top" trigger={(open) => <button ref={trigger} type="button" disabled={disabled} aria-label={tx("Choose model, current {value0}", { value0: label })} aria-haspopup="dialog" aria-expanded={open} title={disabled ? tx("Stop the current run before changing its model") : tx("Choose a model · {value0}{value1}", { value0: label, value1: account ? ` · ${account}` : '' })} className={clsx('flex h-8 w-full min-w-0 items-center gap-1.5 rounded-full px-2.5 text-[13px] transition-colors focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-55', open ? 'bg-ink-5 text-ink' : 'text-ink-2 hover:bg-ink-5 hover:text-ink')}>
    {provider.source === 'tensorfold' ? <EngineDot /> : <Icon name={providerIconName(provider, model)} size={16} className="shrink-0 text-ink-3" />}<span className="min-w-0 flex-1 truncate text-left">{label}{account ? ` · ${account}` : ''}</span>{detail && <span className="shrink-0 text-ink-3 @max-md:hidden">{detail}</span>}{isExternalProvider(provider) && <PlanningOnlyBadge />}<Icon name="chevronDown" size={14} className="shrink-0 text-ink-3" />
  </button>}>
    {(close) => <ModelPickerBody {...props} close={() => { close(); requestAnimationFrame(() => trigger.current?.focus()) }} />}
  </Menu>
}
