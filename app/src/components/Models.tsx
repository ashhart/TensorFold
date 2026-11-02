import { aggregateHubFailures, decodeHubFailure } from '../../shared/hub-failure'
import { ClusterWorkspace } from './clusters/ClusterWorkspace'
import { OriginalDiagnostic } from './OriginalDiagnostic'
import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useCallback, useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { useStore } from '../lib/store'
import { ENGINE_PROVIDER_ID, loadEngineModel, refreshEngine, useEngine } from '../lib/engine'
import { openModelChat } from '../lib/model-chat'
import { useFavoriteModels, orderFavoritesFirst } from '../lib/favorite-models'
import type { EngineInstalledModel, HubModelSummary } from '../lib/types'
import { Button, Input } from './ui'
import { Icon } from '../lib/icons'
import { HubRow, InstalledRow, type PreflightState } from './models/ModelRows'
import { EngineRuntimePanel } from './models/EngineRuntimePanel'
import { EngineApiPanel } from './models/EngineApiPanel'
import { useLearning, learningBusy } from '../lib/learning'
import { LearningPanel } from './models/LearningPanel'
import { WorkersPanel } from './models/WorkersPanel'
import { isLocalWorker } from './models/workers'
import { recommendEngineModel } from '../lib/model-recommendation'
import { ModelRecommendationPanel } from './models/ModelRecommendationPanel'
import { readOnboardingDraft, saveOnboardingDraft, shouldShowOnboarding } from './onboarding-model'
import { ModelDetailsDrawer } from './models/ModelDetailsDrawer'
import { ModelComparisonPanel } from './models/ModelComparisonPanel'
import { ModelsDrawer } from './models/ModelsDrawer'
import { DownloadCentre } from './models/DownloadCentre'
import { DownloadActivity } from './models/DownloadActivity'
import { isDownloadPending } from './models/download-display'
import { ModelFoldersSettings } from './settings/ModelFoldersSettings'
import { ModelBenchmarkDrawer } from './models/benchmark/ModelBenchmarkDrawer'

const AUTO_CHECK = 10
const TABS = ['Models', 'Discover', 'Workers', 'Learning', 'API'] as const
type WorkspaceTab = typeof TABS[number]
type Utility = 'downloads' | 'folders' | 'runtime' | 'compare' | 'benchmark' | 'clusters' | null
const TAB_LABEL: Record<WorkspaceTab, string> = { get Models() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "My models") }, get Discover() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Discover") }, get Workers() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Workers") }, get Learning() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Learning") }, API: 'API' }
const route = (model: EngineInstalledModel) => ({ providerId: ENGINE_PROVIDER_ID, accountId: null, model: model.repo })
function errorText(error: unknown): string { return decodeHubFailure(error).message }

// Builds without an engine keep provider setup available.
function NoLocalEngineModels() {
  const tx = useAuthoredText()
  return <div data-testid="models-page" className="scroll-thin h-full overflow-y-auto">
    <div className="mx-auto w-full max-w-[1040px] px-4 py-6 sm:px-6">
      <header><h1 className="text-[21px] font-semibold tracking-[-0.02em] text-ink">{tx("Models")}</h1></header>
      <section className="mt-5 rounded-xl border border-line bg-card p-5">
        <h2 className="text-[16px] font-medium text-ink">{tx('Connect a model provider')}</h2>
        <p className="mb-4 mt-2 max-w-xl text-[14px] leading-relaxed text-ink-2">{tx('This build does not include the local engine. Connect a provider or another local runtime to start a conversation.')}</p>
        <Button size="sm" variant="primary" icon="plus" onClick={() => useStore.getState().setView({ name: 'settings', section: 'models' })}>{tx("Connect provider")}</Button>
      </section>
    </div>
  </div>
}

export function Models({ minimal = false, noLocalEngine = false }: { minimal?: boolean; noLocalEngine?: boolean } = {}) {
  // The gate sits in a component with no hooks, so the workspace below always runs the same hooks in the same order.
  if (noLocalEngine) return <NoLocalEngineModels />
  return <ModelsWorkspace minimal={minimal} />
}

function ModelsWorkspace({ minimal = false }: { minimal?: boolean }) {
  const tx = useAuthoredText()

  const { snapshot, error: engineError } = useEngine()
  const { snapshot: learning } = useLearning()
  const learningLocked = learningBusy(learning)
  const settings = useStore((state) => state.settings)
  const setupPending = shouldShowOnboarding(undefined, settings?.desktopOnboardingComplete)
  const setNotice = useStore((state) => state.setNotice)
  const threads = useStore((state) => state.threads)
  const runStatus = useStore((state) => state.status)
  const requestedTab = useStore((state) => state.view.name === 'models' ? state.view.tab : undefined)
  const [tab, setTab] = useState<WorkspaceTab>(minimal && requestedTab !== 'API' ? 'Models' : requestedTab ?? 'Models')
  useEffect(() => { if (minimal) setTab(requestedTab === 'API' ? 'API' : 'Models'); else if (requestedTab) setTab(requestedTab) }, [requestedTab, minimal])
  const activeThreads = threads.filter((thread) => thread.providerId === ENGINE_PROVIDER_ID && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && ['thinking', 'streaming', 'tool', 'awaiting'].includes(runStatus[thread.id] ?? (thread.status === 'running' ? 'thinking' : 'idle')))
  const activeRuns = activeThreads.length
  const [checks, setChecks] = useState<Record<string, PreflightState>>({})
  const [editingModel, setEditingModel] = useState<EngineInstalledModel | null>(null)
  const [utility, setUtility] = useState<Utility>(null)
  const [clusterCloseBlocked, setClusterCloseBlocked] = useState(false)
  const [benchmarkModel, setBenchmarkModel] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [libraryQuery, setLibraryQuery] = useState('')
  const [favoritesOnly, setFavoritesOnly] = useState(false)
  const [results, setResults] = useState<HubModelSummary[] | null>(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const checked = useRef(new Set<string>())
  const [startingModel, setStartingModel] = useState<string | null>(null)
  const [stopping, setStopping] = useState(false)
  const startLock = useRef(false)
  const stopLock = useRef(false)
  const cancelLoad = useRef(false)
  const searchLock = useRef(false)
  const mounted = useRef(true)
  const favorites = useFavoriteModels()
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])
  useEffect(() => { void refreshEngine(true) }, [])

  const check = useCallback(async (repo: string, revision: string | null = null, force = false) => {
    if (checked.current.has(repo) && !force) return
    checked.current.add(repo)
    setChecks((current) => ({ ...current, [repo]: { loading: true, result: current[repo]?.result ?? null, error: null } }))
    try {
      const result = await api.enginePreflight(repo, revision)
      if (mounted.current) setChecks((current) => ({ ...current, [repo]: { loading: false, result, error: null } }))
    } catch (error) {
      if (mounted.current) setChecks((current) => ({ ...current, [repo]: { loading: false, result: null, error: errorText(error), errorKind: decodeHubFailure(error).category } }))
    }
  }, [])
  const suggestions = snapshot?.suggestions ?? []
  useEffect(() => { for (const suggestion of suggestions) void check(suggestion.repo, suggestion.revision) }, [suggestions, check])
  const finished = (snapshot?.pulls ?? []).filter((pull) => pull.state === 'done').map((pull) => `${pull.repo}@${pull.finishedAt}`).join(',')
  useEffect(() => {
    for (const pull of snapshot?.pulls ?? []) if (pull.state === 'done' && checked.current.has(pull.repo)) void check(pull.repo, pull.revision, true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished])
  const revisionFor = (repo: string) => suggestions.find((suggestion) => suggestion.repo === repo)?.revision ?? null
  const search = async () => {
    const text = query.trim()
    if (!text || searchLock.current) return
    searchLock.current = true; setSearching(true); setSearchError(null); setSearchTerm(text)
    try {
      const found = await api.engineSearch(text)
      if (!mounted.current) return
      setResults(found)
      for (const hit of found.slice(0, AUTO_CHECK)) void check(hit.repo, revisionFor(hit.repo))
    } catch (error) { if (mounted.current) { setSearchError(errorText(error)); setResults(null) } }
    finally { searchLock.current = false; if (mounted.current) setSearching(false) }
  }
  const download = async (repo: string, revision: string | null = null) => {
    try { await api.enginePull(repo, revision) } catch (error) { if (mounted.current) setNotice({ kind: 'error', message: errorText(error) }) }
  }
  const useModel = async (repo: string) => {
    if (startLock.current || stopLock.current) return
    if (engineError || !snapshot?.status.available) { setNotice({ kind: 'error', message: tx('The local engine is unavailable. Open Engine details for help.') }); return }
    if (learningLocked) { setNotice({ kind: 'error', message: tx("Wait for the learning checkpoint before using a model.") }); return }
    if (activeRuns > 0 && (snapshot?.status.state !== 'ready' || snapshot.status.model !== repo)) { setNotice({ kind: 'error', message: tx("Stop active local runs before switching their model.") }); return }
    startLock.current = true; cancelLoad.current = false; setStartingModel(repo)
    try {
      await loadEngineModel(repo)
      if (!mounted.current || cancelLoad.current) return
      if (shouldShowOnboarding(undefined, useStore.getState().settings?.desktopOnboardingComplete)) {
        saveOnboardingDraft({ ...readOnboardingDraft(minimal ? 'minimal' : 'full'), step: 'guide' })
        useStore.getState().setView({ name: 'home' })
      } else if (!minimal) await openModelChat({ providerId: ENGINE_PROVIDER_ID, model: repo }, () => mounted.current && !cancelLoad.current)
    } catch (error) { if (mounted.current && !cancelLoad.current) setNotice({ kind: 'error', message: errorText(error) }) }
    finally { startLock.current = false; if (mounted.current) setStartingModel(null) }
  }
  const unload = async () => {
    if (stopLock.current) return
    if (activeRuns > 0 || learningLocked) { setNotice({ kind: 'error', message: tx("Finish active local work before unloading its model.") }); return }
    stopLock.current = true; setStopping(true)
    cancelLoad.current = startLock.current || snapshot?.status.state === 'starting'
    try { await api.engineUnload() }
    catch (error) { if (mounted.current) setNotice({ kind: 'error', message: errorText(error) }) }
    finally { stopLock.current = false; if (mounted.current) setStopping(false) }
  }
  const status = snapshot ? engineError ? { ...snapshot.status, available: false } : snapshot.status : null
  const installed = snapshot?.installed ?? []
  const pulls = snapshot?.pulls ?? []
  const pendingPulls = pulls.filter(isDownloadPending)
  const failedPulls = pulls.filter((pull) => pull.state === 'failed')
  const pullFor = (repo: string) => pulls.find((pull) => pull.repo === repo)
  const recommendation = recommendEngineModel({ status, installed, suggestions, preflights: Object.fromEntries(Object.entries(checks).map(([repo, state]) => [repo, state.result])) })
  const checkFailed = suggestions.length > 0 && suggestions.every((suggestion) => checks[suggestion.repo] && !checks[suggestion.repo].loading) && suggestions.some((suggestion) => !!checks[suggestion.repo]?.error)
  const failureKind = aggregateHubFailures(suggestions.flatMap((suggestion) => checks[suggestion.repo]?.error ? [checks[suggestion.repo].errorKind ?? 'check'] : []))
  const lowered = libraryQuery.trim().toLocaleLowerCase()
  const shown = orderFavoritesFirst(installed.filter((model) => (!favoritesOnly || favorites.isFavorite(route(model))) && (!lowered || [model.title, model.repo, model.family, model.format].some((value) => value?.toLocaleLowerCase().includes(lowered)))), favorites.favorites, route)
  const favoriteCount = installed.filter((model) => favorites.isFavorite(route(model))).length
  const locked = activeRuns > 0 || learningLocked || !status?.available || status.state === 'starting' || startingModel != null || stopping
  const canCompare = installed.filter((model) => model.fitsMemory === true).length >= 2
  const benchmarkBlockedReason = activeRuns > 0 ? 'Finish active local chats and workers before benchmarking.' : learningLocked ? 'Wait for the Living Weights checkpoint and turn learning off before benchmarking.' : status?.state === 'starting' || startingModel != null ? 'Wait for the current model to finish loading before benchmarking.' : !status?.available ? 'The built-in TensorFold engine is unavailable; check Engine details before benchmarking.' : null
  const showApi = () => { setUtility(null); setTab('API') }
  const showWorkers = () => {
    setUtility(null)
    if (activeThreads.length && !activeThreads.some(isLocalWorker)) useStore.getState().openThread(activeThreads[0].id)
    else setTab('Workers')
  }
  const recommendationPanel = <ModelRecommendationPanel minimal={minimal} recommendation={recommendation} pull={recommendation.selected ? pullFor(recommendation.selected.repo) : undefined} starting={startingModel != null} locked={locked && recommendation.selected?.repo !== status?.model || learningLocked || status?.state === 'starting'} checkFailed={checkFailed} failureKind={failureKind} onStart={(repo) => void useModel(repo)} onDownload={(repo) => void download(repo, revisionFor(repo))} onRetry={() => { for (const suggestion of suggestions) void check(suggestion.repo, suggestion.revision, true) }} />
  const tabs = minimal ? ['Models', 'Discover', 'API'] as const : TABS

  return <div data-testid="models-page" className="scroll-thin h-full overflow-y-auto">
    <div className="mx-auto w-full max-w-[1040px] px-4 py-6 sm:px-6">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div><h1 className="text-[21px] font-semibold tracking-[-0.02em] text-ink">{tx("Models")}</h1><p className="mt-1 max-w-xl text-[13px] leading-relaxed text-ink-2">{setupPending ? tx('Download or choose a model. Once it loads, setup continues with your next step.') : minimal ? tx("Choose a model for your local API.") : tx("Choose a model and get started.")}</p></div>
        <div className="flex flex-wrap gap-1"><Button size="sm" variant="ghost" onClick={() => setUtility('clusters')}>{tx("Compute and clusters")}</Button><Button size="sm" variant="ghost" icon="folder" onClick={() => setUtility('folders')}>{tx("Model folders")}</Button><Button size="sm" variant="ghost" icon="arrowDown" onClick={() => setUtility('downloads')}>{tx("Downloads")}{pendingPulls.length ? ` (${pendingPulls.length})` : failedPulls.length ? tx(" ({value0} need attention)", { value0: failedPulls.length }) : ''}</Button><Button size="sm" variant="ghost" icon="refresh" aria-label={tx("Refresh models")} onClick={() => void refreshEngine(true)} /></div>
      </header>
      <EngineRuntimePanel compact minimal={minimal} activeRuns={activeRuns} stopping={stopping} onDetails={() => setUtility('runtime')} onChat={status?.model ? () => void useModel(status.model!) : undefined} onRetry={status?.model && !startingModel ? () => void useModel(status.model!) : undefined} onApi={showApi} onWorkers={showWorkers} onUnload={() => void unload()} />
      <div role="tablist" aria-label={tx("Local AI workspace")} className="mt-5 flex flex-wrap gap-x-5 gap-y-1 border-b border-line">
        {tabs.map((name, index) => <button key={name} id={`local-tab-${name}`} role="tab" aria-label={name === 'Models' && installed.length ? tx("My models ({value0})", { value0: installed.length }) : TAB_LABEL[name]} aria-selected={tab === name} aria-controls={`local-panel-${name}`} tabIndex={tab === name ? 0 : -1} onClick={() => setTab(name)} onKeyDown={(event) => {
          const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null
          if (next === null) return
          event.preventDefault(); setTab(tabs[next]); event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus()
        }} className={clsx('relative pb-3 pt-1 text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-accent', tab === name ? 'text-ink after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:rounded-full after:bg-accent' : 'text-ink-2 hover:text-ink')}>{TAB_LABEL[name]}{name === 'Models' && installed.length > 0 && <span className="ml-1.5 text-ink-3">{installed.length}</span>}</button>)}
      </div>
      <DownloadActivity pulls={pulls} installed={installed} showCompleted={tab === 'Discover'} minimal={minimal} locked={locked} onAll={() => setUtility('downloads')} onCancel={(repo) => api.engineCancelPull(repo)} onRetry={(repo, revision) => api.enginePull(repo, revision)} onLoad={(repo) => void useModel(repo)} />
      <div id="local-panel-Models" role="tabpanel" aria-labelledby="local-tab-Models" hidden={tab !== 'Models'}>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-3"><div className="flex min-w-0 flex-1 basis-56 items-center gap-2"><Input aria-label={tx("Search your models")} value={libraryQuery} onChange={(event) => setLibraryQuery(event.target.value)} placeholder={tx("Search your models")} maxLength={200} /><button type="button" disabled={!favorites.ready} aria-pressed={favoritesOnly} aria-label={tx("Show favourite models only")} onClick={() => setFavoritesOnly((value) => !value)} className={clsx('flex shrink-0 items-center gap-1.5 rounded-lg border px-2.5 py-2 text-[13px] focus-visible:outline-2 focus-visible:outline-accent disabled:opacity-40', favoritesOnly ? 'border-accent/40 bg-accent/10 text-accent' : 'border-line text-ink-2 hover:text-ink')}><Icon name={favoritesOnly ? 'starFilled' : 'star'} size={14} />{tx("Favourites")}{favoriteCount > 0 ? ` (${favoriteCount})` : ''}</button></div><div className="flex flex-wrap gap-2"><Button size="sm" variant="default" onClick={() => { setBenchmarkModel(null); setUtility('benchmark') }}>{tx("Benchmarks")}</Button><Button size="sm" variant="default" icon="layers" disabled={locked || !canCompare} title={!canCompare ? tx("Install two compatible models to compare their replies") : locked ? tx("Finish active local work before comparing models") : undefined} onClick={() => setUtility('compare')}>{tx("Compare models")}</Button></div></div>
        {favorites.error && <p role="alert" className="mt-3 text-[13px] text-warn">{<OriginalDiagnostic>{favorites.error}</OriginalDiagnostic>}</p>}
        {!snapshot ? <p role="status" className="mt-6 text-[14px] text-ink-2">{tx("Checking installed models…")}</p> : installed.length === 0 ? <div className="mt-6"><div className="rounded-xl border border-dashed border-line px-5 py-7"><h2 className="text-[16px] font-medium text-ink">{tx("Your models will appear here")}</h2><p className="mt-2 max-w-xl text-[14px] leading-relaxed text-ink-2">{tx("Download a compatible model or add a folder with models you already have.")}</p><div className="mt-4 flex flex-wrap gap-2"><Button variant="primary" icon="search" onClick={() => setTab('Discover')}>{tx("Discover models")}</Button><Button icon="folder" onClick={() => setUtility('folders')}>{tx("Add model folder")}</Button></div></div>{recommendationPanel}</div> : <>
          <p className="mt-4 text-[12px] text-ink-3">{shown.length} {tx("of ")}{installed.length} {installed.length === 1 ? tx("model") : tx("models")} {tx("on this Mac")}{favoritesOnly ? tx(" · favourites") : ''}</p>
          <div className="mt-1" aria-label={tx("Installed models")}>{shown.map((model) => <InstalledRow key={model.repo} model={model} minimal={minimal} status={status} hubCache={snapshot.hubCache ?? 'the Hugging Face cache'} onSettings={() => setEditingModel(model)} favorite={favorites.isFavorite(route(model))} favoriteDisabled={!favorites.ready} onFavorite={() => { favorites.toggle(route(model)) }} using={startingModel === model.repo} chatBlocked={learningLocked} onLoad={() => void useModel(model.repo)} onUnload={() => void unload()} onTask={showWorkers} onChat={() => void useModel(model.repo)} locked={locked} onDelete={async () => {
            try { await api.engineDelete(model.repo) } catch (error) { if (mounted.current) setNotice({ kind: 'error', message: errorText(error) }); throw error }
          }} />)}</div>
          {shown.length === 0 && <div className="mt-5 rounded-xl border border-dashed border-line px-4 py-6 text-[14px] text-ink-2">{favoritesOnly && !libraryQuery.trim() ? tx("Star a model to keep it in your favourites.") : tx("No models match this search.")}</div>}
        </>}
      </div>
      <div id="local-panel-Discover" role="tabpanel" aria-labelledby="local-tab-Discover" hidden={tab !== 'Discover'}>
        <form className="mt-5 flex gap-2" onSubmit={(event) => { event.preventDefault(); void search() }}><Input data-testid="models-search" className="min-w-0" aria-label={tx("Search Hugging Face")} value={query} onChange={(event) => setQuery(event.target.value)} placeholder={tx("Search Hugging Face or enter owner/model")} spellCheck={false} maxLength={200} /><Button className="shrink-0 whitespace-nowrap" type="submit" icon="search" disabled={searching || !query.trim()}>{searching ? tx("Searching…") : tx("Search")}</Button></form>
        <p className="mt-2 text-[12px] text-ink-2">{tx("Models are checked for engine support, memory and download space before use.")}</p>
        {searchError && <p role="alert" className="mt-4 rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger">{searchError}</p>}
        {results ? <Section title={tx("Results for “{value0}”", { value0: searchTerm })} action={<Button size="xs" variant="ghost" onClick={() => { setResults(null); setQuery(''); setSearchError(null) }}>{tx("Clear search")}</Button>}>
          {results.length === 0 && <p className="py-3 text-[14px] text-ink-2">{tx("No repositories match; try a model name or its Hugging Face owner.")}</p>}
          {results.map((hit) => <HubRow key={hit.repo} testId="models-search-row" repo={hit.repo} summary={hit} preflight={checks[hit.repo]} pull={pullFor(hit.repo)} onCheck={() => void check(hit.repo, revisionFor(hit.repo), true)} onDownload={() => void download(hit.repo, revisionFor(hit.repo))} />)}
        </Section> : <>
          {(!recommendation.selected || recommendation.selected.action === 'download') && recommendationPanel}
          <Section title={tx("Supported starting points")}><div className="space-y-2">{suggestions.filter((suggestion) => !installed.some((model) => model.repo === suggestion.repo) && !(recommendation.selected?.action === 'download' && recommendation.selected.repo === suggestion.repo)).map((suggestion) => <HubRow key={suggestion.repo} testId="models-suggestion-row" title={suggestion.title} repo={suggestion.repo} note={suggestion.note} preflight={checks[suggestion.repo]} pull={pullFor(suggestion.repo)} onCheck={() => void check(suggestion.repo, suggestion.revision, true)} onDownload={() => void download(suggestion.repo, suggestion.revision)} />)}{suggestions.length > 0 && suggestions.every((suggestion) => installed.some((model) => model.repo === suggestion.repo)) && <p className="text-[14px] text-ink-2">{tx("Your suggested models are already in My models; search above to explore more.")}</p>}</div></Section>
        </>}
        <details className="mt-6 border-t border-line pt-3 text-[12px] text-ink-2"><summary className="cursor-pointer focus-visible:outline-2 focus-visible:outline-accent">{tx("About downloads and compatibility")}</summary><p className="mt-2 max-w-2xl leading-relaxed">{tx("Downloads use the shared Hugging Face cache and resume where they stopped. The built-in downloader does not use a Hugging Face token, so gated repositories need to be downloaded separately and added through Model folders; finding a repository does not guarantee engine support.")}</p>{snapshot?.hubCache && <p className="mt-2 break-all font-mono">{snapshot.hubCache}</p>}</details>
      </div>
      {!minimal && <div id="local-panel-Workers" role="tabpanel" aria-labelledby="local-tab-Workers" hidden={tab !== 'Workers'}>{tab === 'Workers' && <WorkersPanel onChooseModel={() => setTab('Models')} />}</div>}
      {!minimal && <div id="local-panel-Learning" role="tabpanel" aria-labelledby="local-tab-Learning" hidden={tab !== 'Learning'}>{tab === 'Learning' && <LearningPanel activeRuns={activeRuns} />}</div>}
      <div id="local-panel-API" role="tabpanel" aria-labelledby="local-tab-API" hidden={tab !== 'API'}>{tab === 'API' && <EngineApiPanel minimal={minimal} onModels={() => setTab('Models')} />}</div>
    </div>
    {editingModel && <ModelDetailsDrawer key={editingModel.repo} model={editingModel} status={status} blocked={locked} onClose={() => setEditingModel(null)} onBenchmark={() => { setBenchmarkModel(editingModel.repo); setEditingModel(null); setUtility('benchmark') }} />}
    {utility === 'clusters' && <ModelsDrawer title={tx('Compute and clusters')} labelId="models-clusters-title" onClose={() => setUtility(null)} closeDisabled={clusterCloseBlocked}><ClusterWorkspace onCloseBlockedChange={setClusterCloseBlocked} /></ModelsDrawer>}
    {utility === 'benchmark' && <ModelBenchmarkDrawer models={installed} activeModel={status?.model ?? null} preferredModel={benchmarkModel} blockedReason={benchmarkBlockedReason} onClose={() => setUtility(null)} />}
    {utility === 'compare' && <ModelComparisonPanel models={installed} activeModel={status?.model ?? null} locked={locked} onClose={() => setUtility(null)} />}
    {utility === 'downloads' && <ModelsDrawer title={tx("Downloads")} labelId="models-downloads-title" onClose={() => setUtility(null)} description={tx("Follow queued downloads, resume failures and check completed files.")}>{pulls.length ? <DownloadCentre pulls={pulls} onCancel={(repo) => api.engineCancelPull(repo)} onRetry={(repo, revision) => api.enginePull(repo, revision)} /> : <p className="py-4 text-[14px] text-ink-2">{tx("No downloads yet; find a model in Discover to get started.")}</p>}</ModelsDrawer>}
    {utility === 'folders' && <ModelsDrawer title={tx("Model folders")} labelId="models-folders-title" onClose={() => setUtility(null)} description={tx("Use models already on this Mac or an external drive.")}><ModelFoldersSettings compact /></ModelsDrawer>}
    {utility === 'runtime' && <ModelsDrawer title={tx("Engine details")} labelId="models-runtime-title" onClose={() => setUtility(null)}><EngineRuntimePanel minimal={minimal} activeRuns={activeRuns} stopping={stopping} onRetry={status?.model && !startingModel ? () => void useModel(status.model!) : undefined} onApi={showApi} onWorkers={showWorkers} onUnload={() => void unload()} /></ModelsDrawer>}
  </div>
}

function Section({ title, action, children }: { title: string; action?: React.ReactNode; children: React.ReactNode }) {
  const tx = useAuthoredText()
  return <section className="mt-6"><div className="mb-3 flex flex-wrap items-center justify-between gap-2"><h2 className="break-words text-[14px] font-medium text-ink">{title}</h2>{action}</div>{children}</section>
}
