import { useCallback, useEffect, useRef, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { useStore } from '../lib/store'
import { ENGINE_PROVIDER_ID, loadEngineModel, refreshEngine, useEngine } from '../lib/engine'
import type { HubModelSummary } from '../lib/types'
import { Button, Input } from './ui'
import { HubRow, InstalledRow, PullRow, type PreflightState } from './models/ModelRows'
import { EngineRuntimePanel } from './models/EngineRuntimePanel'
import { EngineApiPanel } from './models/EngineApiPanel'
import { useLearning, learningBusy } from '../lib/learning'
import { LearningPanel } from './models/LearningPanel'
import { WorkersPanel } from './models/WorkersPanel'
import { isLocalWorker } from './models/workers'
import { recommendEngineModel } from '../lib/model-recommendation'
import { ModelRecommendationPanel } from './models/ModelRecommendationPanel'
import { shouldShowOnboarding } from './onboarding-model'
import { ModelFoldersSettings } from './settings/ModelFoldersSettings'

// The built-in engine's home: what is on this Mac, what is downloading, what fits, and any Hub repo.

const AUTO_CHECK = 10
const TABS = ['Models', 'Workers', 'Learning', 'API'] as const
type WorkspaceTab = typeof TABS[number]

function errorText(e: unknown): string {
  return String((e as Error)?.message ?? e).replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '')
}

export function Models({ minimal = false }: { minimal?: boolean } = {}) {
  const { snapshot } = useEngine()
  const { snapshot: learning } = useLearning()
  const learningLocked = learningBusy(learning)
  const newChat = useStore((s) => s.newChat)
  const setNotice = useStore((s) => s.setNotice)
  const threads = useStore((s) => s.threads)
  const runStatus = useStore((s) => s.status)
  const requestedTab = useStore((s) => s.view.name === 'models' ? s.view.tab : undefined)
  const [tab, setTab] = useState<WorkspaceTab>(minimal && requestedTab !== 'API' ? 'Models' : requestedTab ?? 'Models')
  useEffect(() => { if (minimal) setTab(requestedTab === 'API' ? 'API' : 'Models'); else if (requestedTab) setTab(requestedTab) }, [requestedTab, minimal])
  const activeThreads = threads.filter((thread) => thread.providerId === ENGINE_PROVIDER_ID && !thread.orchestratorHostId && ['thinking', 'streaming', 'tool', 'awaiting'].includes(runStatus[thread.id] ?? (thread.status === 'running' ? 'thinking' : 'idle')))
  const activeRuns = activeThreads.length
  const [checks, setChecks] = useState<Record<string, PreflightState>>({})
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<HubModelSummary[] | null>(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const checked = useRef(new Set<string>())
  const [startingRecommendation, setStartingRecommendation] = useState<string | null>(null)
  const startLock = useRef(false)
  const mounted = useRef(true)
  useEffect(() => { mounted.current = true; return () => { mounted.current = false } }, [])

  useEffect(() => { void refreshEngine(true) }, [])

  const check = useCallback(async (repo: string, revision: string | null = null, force = false) => {
    if (checked.current.has(repo) && !force) return
    checked.current.add(repo)
    setChecks((c) => ({ ...c, [repo]: { loading: true, result: c[repo]?.result ?? null, error: null } }))
    try {
      const result = await api.enginePreflight(repo, revision)
      setChecks((c) => ({ ...c, [repo]: { loading: false, result, error: null } }))
    } catch (e) {
      setChecks((c) => ({ ...c, [repo]: { loading: false, result: null, error: errorText(e) } }))
    }
  }, [])

  // Suggested checkpoints are checked as soon as the page knows them.
  const suggestions = snapshot?.suggestions ?? []
  useEffect(() => {
    for (const s of suggestions) void check(s.repo, s.revision)
  }, [suggestions, check])

  // A finished download changes the verdict (installed, cached bytes), so check that repo again.
  const finished = (snapshot?.pulls ?? []).filter((p) => p.state === 'done').map((p) => `${p.repo}@${p.finishedAt}`).join(',')
  useEffect(() => {
    for (const pull of snapshot?.pulls ?? []) if (pull.state === 'done' && checked.current.has(pull.repo)) void check(pull.repo, pull.revision, true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [finished])

  const search = async () => {
    const text = query.trim()
    if (!text) return
    setSearching(true)
    setSearchError(null)
    try {
      const found = await api.engineSearch(text)
      setResults(found)
      for (const hit of found.slice(0, AUTO_CHECK)) void check(hit.repo)
    } catch (e) {
      setSearchError(errorText(e))
      setResults(null)
    } finally {
      setSearching(false)
    }
  }

  const download = async (repo: string, revision: string | null = null) => {
    try {
      await api.enginePull(repo, revision)
    } catch (e) {
      setNotice({ kind: 'error', message: errorText(e) })
    }
  }

  const load = async (repo: string) => {
    if (activeRuns > 0) { setNotice({ kind: 'error', message: 'Stop active local runs before switching their model.' }); return }
    try { await loadEngineModel(repo) } catch (e) { setNotice({ kind: 'error', message: errorText(e) }) }
  }

  const unload = async () => {
    if (activeRuns > 0) { setNotice({ kind: 'error', message: 'Stop active local runs before unloading their model.' }); return }
    try { await api.engineUnload() } catch (e) { setNotice({ kind: 'error', message: errorText(e) }) }
  }
  const chat = async () => {
    try {
      const { settings, saveSettings } = useStore.getState()
      const model = snapshot?.status.model
      if (!settings || !model || snapshot?.status.state !== 'ready') return
      await saveSettings({ ...settings, activeProviderId: ENGINE_PROVIDER_ID, activeProviderAccountId: null, activeModel: model }, { silent: true })
      await newChat()
    } catch (e) { setNotice({ kind: 'error', message: errorText(e) }) }
  }
  const startRecommended = async (repo: string) => {
    if (startLock.current) return
    if (activeRuns > 0 && (snapshot?.status.state !== 'ready' || snapshot.status.model !== repo)) {
      setNotice({ kind: 'error', message: 'Stop active local runs before switching their model.' }); return
    }
    startLock.current = true; setStartingRecommendation(repo)
    try {
      await loadEngineModel(repo)
      if (!mounted.current) return
      if (!minimal && !shouldShowOnboarding(undefined, useStore.getState().settings?.desktopOnboardingComplete)) await newChat()
    } catch (e) { if (mounted.current) setNotice({ kind: 'error', message: errorText(e) }) }
    finally { startLock.current = false; if (mounted.current) setStartingRecommendation(null) }
  }

  const status = snapshot?.status ?? null
  const installed = snapshot?.installed ?? []
  const pulls = (snapshot?.pulls ?? []).filter((p) => p.state !== 'done' || Date.now() - (p.finishedAt ?? 0) < 60_000)
  const pullFor = (repo: string) => snapshot?.pulls.find((p) => p.repo === repo)
  const revisionFor = (repo: string) => suggestions.find((s) => s.repo === repo)?.revision ?? null
  const recommendation = recommendEngineModel({ status, installed, suggestions, preflights: Object.fromEntries(Object.entries(checks).map(([repo, state]) => [repo, state.result])) })
  const checkFailed = suggestions.length > 0 && suggestions.every((s) => checks[s.repo] && !checks[s.repo].loading) && suggestions.some((s) => !!checks[s.repo]?.error)

  return (
    <div data-testid="models-page" className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[980px] px-4 py-6 sm:px-6">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-[20px] font-semibold tracking-[-0.02em] text-ink">Local AI</h1>
            <p className="mt-1 text-[13px] leading-relaxed text-ink-2">{minimal ? 'Download a model or load an existing checkpoint for your server.' : 'Models, workers, learning, and your local API.'}</p>
          </div>
          <Button size="sm" variant="ghost" icon="refresh" onClick={() => void refreshEngine(true)}>Refresh</Button>
        </div>

        <EngineRuntimePanel minimal={minimal} activeRuns={activeRuns} onApi={() => setTab('API')} onWorkers={() => {
          if (activeThreads.length && !activeThreads.some(isLocalWorker)) useStore.getState().openThread(activeThreads[0].id)
          else setTab('Workers')
        }} onUnload={() => void unload()} />

        <div role="tablist" aria-label="Local AI workspace" className="mt-6 flex gap-5 border-b border-line">
          {(minimal ? ['Models', 'API'] as const : TABS).map((name, index, tabs) => <button key={name} id={`local-tab-${name}`} role="tab" aria-selected={tab === name} aria-controls={`local-panel-${name}`} tabIndex={tab === name ? 0 : -1} onClick={() => setTab(name)} onKeyDown={(e) => {
            const next = e.key === 'ArrowRight' ? (index + 1) % tabs.length : e.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : e.key === 'Home' ? 0 : e.key === 'End' ? tabs.length - 1 : null
            if (next === null) return
            e.preventDefault(); setTab(tabs[next]); (e.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next])?.focus()
          }} className={clsx('relative pb-3 text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-accent', tab === name ? 'text-ink after:absolute after:inset-x-0 after:bottom-0 after:h-0.5 after:rounded-full after:bg-accent' : 'text-ink-2 hover:text-ink')}>{name}{name === 'Models' && installed.length > 0 && <span className="ml-1.5 text-ink-2">{installed.length}</span>}</button>)}
        </div>

        {!minimal && <div id="local-panel-Workers" role="tabpanel" aria-labelledby="local-tab-Workers" hidden={tab !== 'Workers'}><WorkersPanel onChooseModel={() => setTab('Models')} /></div>}
        {!minimal && <div id="local-panel-Learning" role="tabpanel" aria-labelledby="local-tab-Learning" hidden={tab !== 'Learning'}>{tab === 'Learning' && <LearningPanel activeRuns={activeRuns} />}</div>}
        <div id="local-panel-API" role="tabpanel" aria-labelledby="local-tab-API" hidden={tab !== 'API'}><EngineApiPanel minimal={minimal} onModels={() => setTab('Models')} /></div>
        <div id="local-panel-Models" role="tabpanel" aria-labelledby="local-tab-Models" hidden={tab !== 'Models'}>
        <ModelRecommendationPanel minimal={minimal} recommendation={recommendation} pull={recommendation.selected ? pullFor(recommendation.selected.repo) : undefined} starting={startingRecommendation != null} locked={learningLocked || activeRuns > 0 && recommendation.selected?.repo !== status?.model || status?.state === 'starting'} checkFailed={checkFailed} onStart={(repo) => void startRecommended(repo)} onDownload={(repo) => void download(repo, revisionFor(repo))} onRetry={() => { for (const suggestion of suggestions) void check(suggestion.repo, suggestion.revision, true) }} />

        {pulls.length > 0 && (
          <Section title="Downloads">
            {pulls.map((pull) => (
              <PullRow key={pull.repo} pull={pull} onCancel={() => void api.engineCancelPull(pull.repo)} onRetry={() => void download(pull.repo, pull.revision)} />
            ))}
          </Section>
        )}

        <Section title="On this Mac" detail={installed.length ? `${installed.length} model${installed.length === 1 ? '' : 's'}` : undefined}>
          {installed.length === 0 && (
            <div className="rounded-xl border border-dashed border-line px-4 py-5 text-[13px] text-ink-3">
              No models yet. Download a suggested model below, or search Hugging Face for any repo.
            </div>
          )}
          {installed.map((model) => (
            <InstalledRow
              key={model.repo}
              model={model}
              minimal={minimal}
              status={status}
              hubCache={snapshot?.hubCache ?? 'the Hugging Face cache'}
              onLoad={() => void load(model.repo)}
              onUnload={() => void unload()}
              onTask={() => setTab('Workers')}
              onChat={() => void chat()}
              locked={activeRuns > 0 || learningLocked}
              onDelete={async () => {
                try { await api.engineDelete(model.repo) } catch (e) { setNotice({ kind: 'error', message: errorText(e) }) }
              }}
            />
          ))}
        </Section>

        <ModelFoldersSettings compact />

        <Section title="Suggested for the built-in engine">
          {suggestions.map((s) => (
            <HubRow key={s.repo} testId="models-suggestion-row" title={s.title} repo={s.repo} note={s.note} preflight={checks[s.repo]} pull={pullFor(s.repo)} onCheck={() => void check(s.repo, s.revision, true)} onDownload={() => void download(s.repo, s.revision)} />
          ))}
        </Section>

        <Section title="Search Hugging Face">
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); void search() }}>
            <Input data-testid="models-search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Any repo, for example Qwen3.5 MLX 4bit" spellCheck={false} maxLength={200} />
            <Button type="submit" variant="default" icon="search" disabled={searching || !query.trim()}>{searching ? 'Searching' : 'Search'}</Button>
          </form>
          {searchError && <div className="rounded-lg border border-danger/25 bg-danger/10 px-3 py-2 text-[13px] text-danger">{searchError}</div>}
          {results && results.length === 0 && <div className="text-[13px] text-ink-3">No repos match.</div>}
          {results?.map((hit) => (
            <HubRow key={hit.repo} testId="models-search-row" repo={hit.repo} summary={hit} preflight={checks[hit.repo]} pull={pullFor(hit.repo)} onCheck={() => void check(hit.repo, null, true)} onDownload={() => void download(hit.repo, revisionFor(hit.repo))} />
          ))}
          <p className="text-[13px] leading-relaxed text-ink-2">
            Downloads go to the shared Hugging Face cache and resume where they stopped. The engine downloads without a Hugging Face token, so gated repos cannot be pulled here.
          </p>
        </Section>
        {snapshot?.hubCache && <details className="mt-6 border-t border-line pt-3 text-[12px] text-ink-2"><summary className="cursor-pointer">Storage location</summary><p className="mt-2 break-all font-mono">{snapshot.hubCache}</p></details>}
        </div>
      </div>
    </div>
  )
}

function Section({ title, detail, children }: { title: string; detail?: string; children: React.ReactNode }) {
  return (
    <section className="mt-7">
      <div className="mb-2.5 flex items-baseline justify-between">
        <h2 className="text-[13px] font-semibold text-ink-2">{title}</h2>
        {detail && <span className="text-[13px] text-ink-2">{detail}</span>}
      </div>
      <div className="grid min-w-0 grid-cols-[minmax(0,1fr)] gap-2">{children}</div>
    </section>
  )
}
