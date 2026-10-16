// Pure rules for the thread's right-hand work pane: which tabs exist, which one opens, and
// whether the pane is open at all.

export type WorkPaneTab = 'changes' | 'terminal' | 'files' | 'preview' | 'details'

export const WORK_PANE_TABS: { id: WorkPaneTab; label: string; needsFolder: boolean }[] = [
  { id: 'changes', label: 'Changes', needsFolder: true },
  { id: 'terminal', label: 'Terminal', needsFolder: true },
  { id: 'files', label: 'Files', needsFolder: true },
  { id: 'preview', label: 'Preview', needsFolder: false },
  { id: 'details', label: 'Details', needsFolder: false },
]

export function workPaneTabsFor(hasFolder: boolean): WorkPaneTab[] {
  return WORK_PANE_TABS.filter((tab) => hasFolder || !tab.needsFolder).map((tab) => tab.id)
}

export function defaultWorkPaneTab({ hasFolder, hasChanges }: { hasFolder: boolean; hasChanges: boolean }): WorkPaneTab {
  if (!hasFolder) return 'details'
  return hasChanges ? 'changes' : 'files'
}

// The tab the user picked for this thread when it is still available, else the default.
export function resolveWorkPaneTab(chosen: WorkPaneTab | null | undefined, ctx: { hasFolder: boolean; hasChanges: boolean }): WorkPaneTab {
  if (chosen && workPaneTabsFor(ctx.hasFolder).includes(chosen)) return chosen
  return defaultWorkPaneTab(ctx)
}

// Threads with a folder follow the global pane toggle. A chat without a folder starts closed
// and opens only when the user opens the pane on that chat.
export function workPaneOpen({ threadId, hasFolder, rightPanelHidden, chatPaneOpenId }: {
  threadId: string
  hasFolder: boolean
  rightPanelHidden: boolean
  chatPaneOpenId: string | null
}): boolean {
  return hasFolder ? !rightPanelHidden : chatPaneOpenId === threadId
}
