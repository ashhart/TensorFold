import { OriginalDiagnostic } from './components/OriginalDiagnostic'
import { useAuthoredText } from './lib/preferences/useAuthoredText'
import { useUiPreferences } from './lib/preferences/hooks'
import { ConversationSearchHost } from './components/search/ConversationSearchHost'
import { lazy, Suspense, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { useStore } from './lib/store'
import { applyTheme } from './lib/theme'
import { applicationBootstrapState } from './lib/bootstrap-state'
import { Sidebar } from './components/Sidebar'
import { TitleBar } from './components/TitleBar'
import { ONBOARDING_COMPLETED_EVENT, ONBOARDING_RESTART_EVENT, readOnboardingDraft, shouldShowOnboarding } from './components/onboarding-model'
import { api } from './lib/ipc'
import type { MinimalTarget } from './components/MinimalWorkspace'
import { NewProjectModal } from './components/NewProjectModal'
import { CommandPalette } from './components/CommandPalette'
import { ExportPanel } from './components/ExportPanel'
import { ReplayPanel } from './components/ReplayPanel'
import { IncidentCenter } from './components/IncidentCenter'
import { SshPrompts } from './components/SshPrompts'
import { UpdateModal } from './components/UpdateModal'
import { Button } from './components/ui'
import type { Provider } from './lib/types'
import { ALL_SETTINGS_SECTIONS, type SettingsSection } from './components/settings/shared'
import { fullLayoutStartupChat, simpleFullLayoutActive } from './lib/full-workspace-layout'
import { useConversationDraftLifecycle } from './lib/use-conversation-draft'
import { desktopDraftPort } from './lib/desktop-drafts'
import { forgetComposerFileSources, makeComposerFileSourcesIncognito } from './lib/composer-file-sources'
import { composerDraftKey } from '../shared/composer-drafts'

// Stable empty-array reference. Returning a fresh `[]` from a zustand selector
// makes useSyncExternalStore think the snapshot changed every render → infinite
// loop ("getSnapshot should be cached" → React #185 → blank screen) whenever
// settings is briefly null during init. Must be a single shared reference.
const EMPTY_PROVIDERS: Provider[] = []

// Route-level views carry editor, terminal, operations-dashboard, and settings dependencies.
// Load only the surface the user opens so the production entry chunk stays small without
// changing navigation or state ownership.
const Assistant = lazy(() => import('./components/Assistant').then((module) => ({ default: module.Assistant })))
const Home = lazy(() => import('./components/Home').then((module) => ({ default: module.Home })))
const Onboarding = lazy(() => import('./components/Onboarding').then((module) => ({ default: module.Onboarding })))
const CommandCentre = lazy(() => import('./components/CommandCentre').then((module) => ({ default: module.CommandCentre })))
const ThreadView = lazy(() => import('./components/Thread').then((module) => ({ default: module.ThreadView })))
const Settings = lazy(() => import('./components/Settings').then((module) => ({ default: module.Settings })))
const Plugins = lazy(() => import('./components/Plugins').then((module) => ({ default: module.Plugins })))
const Automations = lazy(() => import('./components/Automations').then((module) => ({ default: module.Automations })))
const ProjectsIndex = lazy(() => import('./components/Rooms').then((module) => ({ default: module.ProjectsIndex })))
const ProjectView = lazy(() => import('./components/Rooms').then((module) => ({ default: module.ProjectView })))
const RoomView = lazy(() => import('./components/Rooms').then((module) => ({ default: module.RoomView })))
import { useEngine } from './lib/engine'
import { engineManagementVisible } from './lib/engine-visibility'
const Models = lazy(() => import('./components/Models').then((module) => ({ default: module.Models })))
const Bots = lazy(() => import('./components/Bots').then((module) => ({ default: module.Bots })))
const WorkspaceDock = lazy(() => import('./components/WorkspaceDock').then((module) => ({ default: module.WorkspaceDock })))
const MinimalWorkspace = lazy(() => import('./components/MinimalWorkspace').then((module) => ({ default: module.MinimalWorkspace })))

export default function App() {
  const tx = useAuthoredText()
  useConversationDraftLifecycle(desktopDraftPort, desktopDraftPort ? api : null)
  useEffect(() => desktopDraftPort ? api.onComposerDraftReset((scope) => {
    if (scope?.incognito) makeComposerFileSourcesIncognito(composerDraftKey({ ...scope, incognito: false }), composerDraftKey(scope))
    else forgetComposerFileSources(scope ? composerDraftKey(scope) : undefined)
  }) : undefined, [])
  const ready = useStore((s) => s.ready)
  const view = useStore((s) => s.view)
  useEffect(() => {
    if (api.platform === 'web-remote') return
    const id = view.name === 'thread' ? view.id : null
    const local = id && useStore.getState().threads.find(thread => thread.id === id && !thread.orchestratorHostId)
    void api.conversationSearchSetActiveThread(local ? id : null).catch(() => {})
    return () => { void api.conversationSearchSetActiveThread(null).catch(() => {}) }
  }, [view])
  const preferences = useStore(state => state.settings?.uiPreferences)
  useUiPreferences(preferences)
  const theme = useStore((s) => s.resolvedTheme)
  const [settingsDraftDirty, setSettingsDraftDirty] = useState(false)
  const init = useStore((s) => s.init)
  const providers = useStore((s) => s.settings?.providers ?? EMPTY_PROVIDERS)
  // Onboarding is a first-run sequence, not only a provider picker. Keep it open
  // after the first model is saved so the Chat/Assistant/Code/Readiness steps can
  // complete in the same flow.
  const [onboardingDone, setOnboardingDone] = useState(() => !shouldShowOnboarding(providers))
  const replayingOnboarding = useRef(false)
  const [minimalTarget, setMinimalTarget] = useState<MinimalTarget>('overview')
  const showOnboarding = ready && !onboardingDone
  useEffect(() => {
    const complete = () => { replayingOnboarding.current = false; setOnboardingDone(true) }
    const restart = () => { replayingOnboarding.current = true; setOnboardingDone(false); useStore.getState().setView({ name: 'home' }) }
    window.addEventListener(ONBOARDING_COMPLETED_EVENT, complete)
    window.addEventListener(ONBOARDING_RESTART_EVENT, restart)
    return () => { window.removeEventListener(ONBOARDING_COMPLETED_EVENT, complete); window.removeEventListener(ONBOARDING_RESTART_EVENT, restart) }
  }, [])
  useEffect(() => {
    if (api.platform === 'web-remote') return
    const unsubscribeMode = api.onDesktopModeChanged((settings) => useStore.setState({ settings }))
    const unsubscribeNavigation = api.onDesktopNavigate(({ target }) => {
      setMinimalTarget(target)
      const state = useStore.getState()
      if (state.settings?.desktopMode === 'minimal') state.setView({ name: 'home' })
      else state.setView(target === 'overview' ? { name: 'home' } : { name: 'models', tab: target === 'api' ? 'API' : 'Models' })
    })
    return () => { unsubscribeMode(); unsubscribeNavigation() }
  }, [])

  const newTask = useStore((s) => s.newTask)
  const newChat = useStore((s) => s.newChat)
  const setView = useStore((s) => s.setView)
  const sidebarCollapsed = useStore((s) => s.sidebarCollapsed)
  const fullWorkspaceLayout = useStore((s) => s.fullWorkspaceLayout)
  const toggleSidebar = useStore((s) => s.toggleSidebar)
  const toggleRightPanel = useStore((s) => s.toggleRightPanel)
  const cycleApprovalMode = useStore((s) => s.cycleApprovalMode)

  useEffect(() => { init() }, [init])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.metaKey || e.ctrlKey
      const k = e.key.toLowerCase()
      if (useStore.getState().settings?.desktopMode === 'minimal' || showOnboarding) {
        if (mod && e.key === ',') { e.preventDefault(); setView({ name: 'settings', section: 'app' }) }
        else if (mod && (k === 'n' || k === 'b' || e.key === '[' || e.key === ']')) e.preventDefault()
        return
      }
      if (mod && k === 'n') {
        e.preventDefault()
        if (e.shiftKey) void newChat().catch(() => useStore.getState().setNotice({ kind: 'error', message: tx("Could not open a chat. Your current work is still here; try New chat again.") }))
        else newTask()
      }
      else if (mod && e.key === ',') { e.preventDefault(); setView({ name: 'settings' }) }
      else if (mod && e.altKey && k === 'b') { e.preventDefault(); toggleRightPanel() }
      else if (mod && k === 'b') { e.preventDefault(); toggleSidebar() }
      else if (mod && (e.key === '[' || e.key === ']')) {
        e.preventDefault()
        const { threads, view: v, openThread } = useStore.getState()
        const sorted = [...threads].sort((a, b) => b.updatedAt - a.updatedAt)
        if (!sorted.length) return
        let i = v.name === 'thread' ? sorted.findIndex((t) => t.id === v.id) : -1
        const n = sorted.length
        i = ((i + (e.key === ']' ? 1 : -1)) % n + n) % n
        openThread(sorted[i].id)
      }
      else if (e.shiftKey && e.key === 'Tab' && (document.activeElement as HTMLElement | null)?.closest?.('[data-vontra-composer="true"]')) {
        e.preventDefault(); cycleApprovalMode()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [newTask, newChat, setView, toggleSidebar, toggleRightPanel, cycleApprovalMode, showOnboarding])

  // Surface any user-action IPC failure as a notice toast (see src/lib/ipc.ts).
  useEffect(() => {
    const onIpcError = (e: Event) => useStore.getState().setNotice({ kind: 'error', message: (e as CustomEvent).detail || 'Something went wrong.' })
    window.addEventListener('vontra:ipc-error', onIpcError)
    return () => window.removeEventListener('vontra:ipc-error', onIpcError)
  }, [])

  const settings = useStore((s) => s.settings)
  useEffect(() => {
    if (settings?.desktopOnboardingComplete && !replayingOnboarding.current) setOnboardingDone(true)
  }, [settings?.desktopOnboardingComplete])
  const minimal = settings?.desktopMode === 'minimal' && api.platform !== 'web-remote'
  const simpleFull = simpleFullLayoutActive({ layout: fullWorkspaceLayout, desktopMode: settings?.desktopMode, platform: api.platform, onboarding: showOnboarding })
  const initialFullNavigation = useRef(false)
  useEffect(() => {
    if (!ready || !settings || minimal || showOnboarding || initialFullNavigation.current) return
    initialFullNavigation.current = true
    const current = useStore.getState()
    const recent = fullLayoutStartupChat(current.fullWorkspaceLayout, current.view.name, current.threads)
    if (recent) current.openThread(recent.id)
  }, [ready, settings, minimal, showOnboarding])
  const lastContentView = useRef(view)
  useEffect(() => {
    if (view.name !== 'settings') lastContentView.current = view
  }, [view])
  useLayoutEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    if (settings) applyTheme(theme === 'dark' ? settings.dark : settings.light, theme === 'dark', settings.pointerCursors)
  }, [theme, settings])

  useEffect(() => {
    if (ready) {
      const t = setTimeout(() => document.body.classList.add('ready'), 60)
      return () => clearTimeout(t)
    }
  }, [ready])

  // Every hook must run before the bootstrap early return below, or React reports a hook-count
  // change when the workspace becomes ready (error #310).
  const { snapshot: engineSnapshot } = useEngine()
  const engineStatus = engineSnapshot?.status ?? null
  const bootstrap = applicationBootstrapState(ready, settings)
  if (bootstrap !== 'ready') return <BootstrapRecovery unavailable={bootstrap === 'settings-unavailable'} onRetry={() => init({ retry: true })} />

  const contentView = view.name === 'settings' ? lastContentView.current : view
  const closeSettings = () => setView(lastContentView.current)
  const settingsInitialSection: SettingsSection | undefined = view.name === 'settings' && typeof view.section === 'string' && ALL_SETTINGS_SECTIONS.some((section) => section.id === view.section)
    ? view.section as SettingsSection
    : undefined

  return (
    <div className={`app-shell frosted flex h-full w-full overflow-hidden text-ink ${sidebarCollapsed || simpleFull ? 'sidebar-collapsed' : ''}`}>
      {/* Settings is a full-page takeover: its nav replaces the
          sidebar and its sections take the content card. */}
      {(view.name === 'settings' || settingsDraftDirty) && (
        <div className={view.name === 'settings' ? 'contents' : 'hidden'}>
          <Suspense fallback={<ViewFallback />}>
            <Settings onClose={closeSettings} initialSection={settingsInitialSection} onDraftDirtyChange={setSettingsDraftDirty} />
          </Suspense>
        </div>
      )}
      {view.name !== 'settings' && (
        <div className="flex h-full w-full overflow-hidden">
          {!sidebarCollapsed && !showOnboarding && !minimal && <div className={simpleFull ? 'hidden' : 'contents'} aria-hidden={simpleFull || undefined} data-full-workspace-sidebar><Sidebar /></div>}
          <div className="content-card relative flex min-w-0 flex-1 flex-col">
            {showOnboarding ? <header className="drag h-11 shrink-0" /> : !minimal && <TitleBar simple={simpleFull} />}
            <main className="min-h-0 flex-1 overflow-clip">
              <Suspense fallback={<ViewFallback />}>
                {showOnboarding && contentView.name === 'home' ? <Onboarding onDone={() => setOnboardingDone(true)} /> : minimal && !showOnboarding ? <MinimalWorkspace target={minimalTarget} onTarget={setMinimalTarget} /> : <div className="flex h-full min-h-0 flex-col">
                {showOnboarding && <div className="shrink-0 px-4 py-3"><Button size="sm" icon="back" onClick={() => setView({ name: 'home' })}>{tx("Return to setup")}</Button></div>}
                <div className="min-h-0 flex-1">
                {contentView.name === 'home' && <Home />}
                {contentView.name === 'thread' && <ThreadView key={contentView.id} threadId={contentView.id} simpleLayout={simpleFull} />}
                {contentView.name === 'commandCentre' && <CommandCentre />}
                {contentView.name === 'assistant' && <Assistant />}
                {contentView.name === 'plugins' && <Plugins />}
                {contentView.name === 'automations' && <Automations />}
                {contentView.name === 'projects' && <ProjectsIndex />}
                {contentView.name === 'project' && <ProjectView key={`${contentView.id}:${contentView.tab ?? 'overview'}`} id={contentView.id} initialTab={contentView.tab} />}
                {contentView.name === 'room' && <RoomView key={contentView.id} id={contentView.id} />}
                {contentView.name === 'models' && <Models noLocalEngine={!engineManagementVisible(api.platform, engineStatus)} minimal={showOnboarding && readOnboardingDraft().mode === 'minimal'} />}
                {contentView.name === 'bots' && <Bots />}
                </div></div>}
              </Suspense>
            </main>
            {!minimal && !showOnboarding && contentView.name !== 'home' && contentView.name !== 'thread' && (
              <div className={simpleFull ? 'hidden' : 'contents'} aria-hidden={simpleFull || undefined}><Suspense fallback={null}><WorkspaceDock /></Suspense></div>
            )}
          </div>
        </div>
      )}
      {!minimal && !showOnboarding && <><NewProjectModal /><CommandPalette /><ConversationSearchHost navigationBlocked={settingsDraftDirty} /></>}
      <ExportPanel />
      <ReplayPanel />
      <IncidentCenter />
      <SshPrompts />
      <UpdateModal />
      <NoticeToast />
    </div>
  )
}

function ViewFallback() {
  const tx = useAuthoredText()
  return (
    <div className="flex h-full w-full items-center justify-center bg-canvas text-[13px] text-ink-3">
      <span className="h-2 w-2 animate-pulse rounded-full bg-accent" />
      <span className="ml-2">{tx("Loading view…")}</span>
    </div>
  )
}

function BootstrapRecovery({ unavailable, onRetry }: { unavailable: boolean; onRetry: () => Promise<void> }) {
  const tx = useAuthoredText()
  const [retrying, setRetrying] = useState(false)
  return <div className="flex h-full w-full items-center justify-center bg-canvas px-6 text-ink">
    <div role={unavailable ? 'alert' : 'status'} className="w-full max-w-md rounded-xl border border-line bg-card p-6">
      <h1 className="text-[18px] font-semibold">{unavailable ? tx("Could not load your settings") : tx("Loading your workspace…")}</h1>
      {unavailable && <>
        <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{tx("Your saved configuration is unavailable, so setup is paused until it can be read.")}</p>
        <Button variant="primary" disabled={retrying} onClick={() => {
          setRetrying(true)
          void onRetry().finally(() => setRetrying(false)).catch(() => {})
        }} className="no-drag mt-5">{retrying ? tx("Retrying…") : tx("Retry loading")}</Button>
      </>}
    </div>
  </div>
}

// Surfaces persistence/save failures (and other notices) so they never fail silently.
function NoticeToast() {
  const tx = useAuthoredText()
  const notice = useStore((s) => s.notice)
  const dismiss = useStore((s) => s.dismissNotice)
  useEffect(() => {
    if (!notice) return
    // Errors linger; routine "saved" confirmations clear quickly (#48).
    const t = setTimeout(dismiss, notice.kind === 'info' ? 2500 : 9000)
    return () => clearTimeout(t)
  }, [notice, dismiss])
  if (!notice) return null
  return (
    <div className="fixed bottom-4 right-4 z-[100] flex max-w-sm items-start gap-2.5 rounded-xl border border-line bg-card px-4 py-3 text-[13px] shadow-pop fade-in">
      <span className={`mt-1 h-2 w-2 shrink-0 rounded-full ${notice.kind === 'error' ? 'bg-danger' : 'bg-accent'}`} />
      <span className="min-w-0 flex-1 text-ink">{notice.authored ? tx(notice.message) : notice.kind === 'error' ? <><span className="block">{tx("This action could not be completed.")}</span><OriginalDiagnostic>{notice.message}</OriginalDiagnostic></> : notice.message}</span>
      <button onClick={dismiss} aria-label={tx("Dismiss")} className="shrink-0 text-ink-3 transition-colors hover:text-ink">×</button>
    </div>
  )
}
