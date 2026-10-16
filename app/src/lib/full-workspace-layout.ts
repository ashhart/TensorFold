import type { Thread } from './types'

export type FullWorkspaceLayout = 'chat' | 'workspace'
export const FULL_WORKSPACE_LAYOUT_KEY = 'tensorfold.full-workspace-layout.v1'

function browserStorage(): Pick<Storage, 'getItem' | 'setItem'> | null {
  try { return typeof localStorage === 'undefined' ? null : localStorage } catch { return null }
}
export function savedFullWorkspaceLayout(storage = browserStorage()): FullWorkspaceLayout | null {
  try { const value = storage?.getItem(FULL_WORKSPACE_LAYOUT_KEY); return value === 'chat' || value === 'workspace' ? value : null } catch { return null }
}
/** Existing installations without this new preference keep their complete workspace. */
export function readFullWorkspaceLayout(): FullWorkspaceLayout { return savedFullWorkspaceLayout() ?? 'workspace' }
export function fullLayoutPreferenceReadable(storage = browserStorage()): boolean {
  try { if (!storage) return false; storage.getItem(FULL_WORKSPACE_LAYOUT_KEY); return true } catch { return false }
}
export function saveFullWorkspaceLayout(layout: FullWorkspaceLayout, storage = browserStorage()): boolean {
  if (layout !== 'chat' && layout !== 'workspace') throw new Error('Choose Chat layout or Complete workspace.')
  try { if (!storage) return false; storage.setItem(FULL_WORKSPACE_LAYOUT_KEY, layout); return true } catch { return false }
}
export function firstFullOnboardingLayout(previouslyCompleted: boolean, savedChoice: FullWorkspaceLayout | null, preferenceReadable = true): FullWorkspaceLayout | null {
  return preferenceReadable && !previouslyCompleted && !savedChoice ? 'chat' : null
}
export function simpleFullLayoutActive(input: { layout: FullWorkspaceLayout; desktopMode?: 'minimal' | 'full'; platform: string; onboarding: boolean }): boolean {
  return input.layout === 'chat' && !input.onboarding && (input.desktopMode !== 'minimal' || input.platform === 'web-remote')
}
export function fullLayoutStartupChat(layout: FullWorkspaceLayout, viewName: string, threads: Thread[]): Thread | null {
  if (layout !== 'chat' || viewName !== 'home') return null
  return threads.filter((thread) => thread.kind === 'chat' && !thread.archivedAt).sort((a, b) => b.updatedAt - a.updatedAt)[0] ?? null
}
