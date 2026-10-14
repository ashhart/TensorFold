import { useSyncExternalStore } from 'react'
import { api } from './ipc'
import { useStore } from './store'
import type { EngineEvent, EngineInstalledModel, EngineSnapshot, EngineStatus } from './types'
export { VERDICT_BADGES, canDownload, engineStateLabel, formatBytes, pullPercent, type VerdictBadge } from './engine-format'

// Renderer mirror of the built-in engine: one snapshot shared by the Models page, the composer picker
// and onboarding, kept current by `engine:event` pushes.

export const ENGINE_PROVIDER_ID = 'tensorfold'

export interface EngineView {
  snapshot: EngineSnapshot | null
  error: string | null
}

let view: EngineView = { snapshot: null, error: null }
const subscribers = new Set<() => void>()
let started = false

function publish(next: Partial<EngineView>): void {
  view = { ...view, ...next }
  for (const notify of subscribers) notify()
}

// Keep the built-in provider's model list equal to what is installed, without a settings save.
function mirrorInstalled(installed: EngineInstalledModel[]): void {
  const repos = installed.map((m) => m.repo)
  useStore.setState((state) => state.settings ? {
    settings: { ...state.settings, providers: state.settings.providers.map((p) => (p.id === ENGINE_PROVIDER_ID ? { ...p, models: repos } : p)) },
  } : {})
}

function apply(event: EngineEvent): void {
  const snapshot = view.snapshot
  if (!snapshot) return
  if (event.kind === 'status') publish({ snapshot: { ...snapshot, status: event.status } })
  else if (event.kind === 'performance') publish({ snapshot: { ...snapshot, performance: event.performance } })
  else if (event.kind === 'installed') {
    publish({ snapshot: { ...snapshot, installed: event.installed } })
    mirrorInstalled(event.installed)
  } else if (event.kind === 'pull') {
    const pulls = snapshot.pulls.filter((p) => p.repo !== event.pull.repo)
    publish({ snapshot: { ...snapshot, pulls: [event.pull, ...pulls] } })
  }
}

export async function refreshEngine(hard = false): Promise<void> {
  try {
    const snapshot = hard ? await api.engineRefresh() : await api.engineSnapshot()
    publish({ snapshot, error: null })
    mirrorInstalled(snapshot.installed)
  } catch (e) {
    publish({ error: String((e as Error)?.message ?? e) })
  }
}

function start(): void {
  if (started) return
  started = true
  api.onEngineEvent(apply)
  void refreshEngine()
}

function subscribe(notify: () => void): () => void {
  subscribers.add(notify)
  return () => { subscribers.delete(notify) }
}

export function useEngine(): EngineView {
  start()
  return useSyncExternalStore(subscribe, () => view)
}
/** A gated subscriber for loading feedback; other API/CLI providers do not start this mirror. */
export function useEngineStatus(active: boolean): EngineStatus | null {
  if (active) start()
  const current = useSyncExternalStore(subscribe, () => view)
  return active ? current.snapshot?.status ?? null : null
}

// The running engine's window for `model`, for the context meter. Only the built-in provider starts
// the engine mirror; other providers just read null.
export function useEngineContextTokens(active: boolean, model: string | null | undefined): number | null {
  if (active) start()
  const current = useSyncExternalStore(subscribe, () => view)
  const status = current.snapshot?.status
  return active && status?.state === 'ready' && status.model === model ? status.contextTokens : null
}

// Load = make it the active model and start the engine on it.
export async function loadEngineModel(repo: string): Promise<EngineStatus> {
  const { settings, saveSettings } = useStore.getState()
  if (settings && (settings.activeProviderId !== ENGINE_PROVIDER_ID || settings.activeModel !== repo)) {
    await saveSettings({ ...settings, activeProviderId: ENGINE_PROVIDER_ID, activeProviderAccountId: null, activeModel: repo }, { silent: true })
  }
  return api.engineLoad(repo)
}
