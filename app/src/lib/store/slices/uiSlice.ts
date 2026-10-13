import type { GetState, Slice } from '../types'
import { effectiveThreadCwd } from '../../thread-workspace'

const PINNED_KEY = 'vontra.pinnedThreads'

function loadPinned(): string[] {
  try {
    const raw = JSON.parse(localStorage.getItem(PINNED_KEY) ?? '[]')
    return Array.isArray(raw) ? raw.filter((id): id is string => typeof id === 'string') : []
  } catch { return [] }
}

// The thread on screen and whether it has a folder (folder threads share the global pane toggle).
function viewedThread(get: GetState): { id: string; hasFolder: boolean } | null {
  const { view, threads, projects } = get()
  if (view.name !== 'thread') return null
  const thread = threads.find((t) => t.id === view.id)
  if (!thread) return null
  const project = thread.projectId ? projects.find((p) => p.id === thread.projectId) ?? null : null
  return { id: thread.id, hasFolder: thread.kind === 'task' && !!effectiveThreadCwd(thread, project) }
}

export const uiSlice: Slice = (set, get) => ({
  ready: false,
  notice: null as { kind: 'error' | 'info'; message: string } | null,
  resolvedTheme: 'light' as 'light' | 'dark',
  newProjectOpen: false,
  newProjectIntent: 0,
  pendingMention: null as { path: string; n: number } | null,
  composerSeed: null as { text: string; n: number } | null,
  threadDockRequest: null as { threadId: string; dock: 'terminal' | 'preview'; nonce: number } | null,
  sidebarCollapsed: false,
  rightPanelHidden: false,
  chatPaneOpenId: null,
  workPaneTabs: {},
  workPaneFocus: null,
  pinnedThreadIds: loadPinned(),
  workspaceDockOpen: false,
  assistantSource: 'all',
  projectTagFilter: (() => { try { return localStorage.getItem('vontra.projectTag') || 'all' } catch { return 'all' } })(),
  collapseReasoningDefault: true,
  collapseToolRunsDefault: true,

  toggleSidebar() { set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })) },
  toggleRightPanel() {
    const current = viewedThread(get)
    if (current && !current.hasFolder) set((s) => ({ chatPaneOpenId: s.chatPaneOpenId === current.id ? null : current.id }))
    else set((s) => ({ rightPanelHidden: !s.rightPanelHidden }))
  },
  openWorkPane(tab, opts) {
    const current = viewedThread(get)
    if (!current) return
    set((s) => ({
      ...(current.hasFolder ? { rightPanelHidden: false } : { chatPaneOpenId: current.id }),
      workPaneTabs: { ...s.workPaneTabs, [current.id]: tab },
      workPaneFocus: { threadId: current.id, tab, path: opts?.path ?? null, nonce: (s.workPaneFocus?.nonce ?? 0) + 1 },
    }))
  },
  setWorkPaneTab(threadId, tab) { set((s) => ({ workPaneTabs: { ...s.workPaneTabs, [threadId]: tab } })) },
  closeWorkPane() {
    const current = viewedThread(get)
    if (current && !current.hasFolder) set({ chatPaneOpenId: null })
    else set({ rightPanelHidden: true })
  },
  togglePinnedThread(id) {
    const next = get().pinnedThreadIds.includes(id) ? get().pinnedThreadIds.filter((x) => x !== id) : [id, ...get().pinnedThreadIds]
    try { localStorage.setItem(PINNED_KEY, JSON.stringify(next)) } catch { /* ignore */ }
    set({ pinnedThreadIds: next })
  },
  toggleWorkspaceDock() { set((s) => ({ workspaceDockOpen: !s.workspaceDockOpen })) },
  setWorkspaceDockOpen(open) { set({ workspaceDockOpen: open }) },
  setAssistantSource(source: string) { set({ assistantSource: source }) },
  setProjectTagFilter(tag: string) { try { localStorage.setItem('vontra.projectTag', tag) } catch { /* ignore */ } set({ projectTagFilter: tag }) },
  setCollapseReasoningDefault(value) { set({ collapseReasoningDefault: value }) },
  setCollapseToolRunsDefault(value) { set({ collapseToolRunsDefault: value }) },
  dismissNotice() { set({ notice: null }) },
  setNotice(notice) { set({ notice }) },
  openNewProject() { set((state) => ({ newProjectOpen: true, newProjectIntent: (state.newProjectIntent ?? 0) + 1 })) },
  closeNewProject() { set((state) => ({ newProjectOpen: false, newProjectIntent: (state.newProjectIntent ?? 0) + (state.newProjectOpen ? 1 : 0) })) },
  addFileMention(path) { set((st) => ({ pendingMention: { path, n: (st.pendingMention?.n ?? 0) + 1 } })) },
  seedComposer(text) { set((st) => ({ composerSeed: { text, n: (st.composerSeed?.n ?? 0) + 1 } })) },
})
