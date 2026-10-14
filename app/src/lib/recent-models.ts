// One-click model switching: the models you've used recently, persisted to
// localStorage, surfaced as chips beside the model picker so a common switch is a
// single click instead of opening the grouped menu each time.
export const RECENT_MODELS_KEY = 'vontra.recentModels'
export const MAX_RECENT_MODELS = 6
export interface RecentModel { providerId: string; accountId: string | null; model: string }
export function loadRecentModels(): RecentModel[] {
  try { const raw = JSON.parse(localStorage.getItem(RECENT_MODELS_KEY) || '[]'); return Array.isArray(raw) ? raw.slice(0, MAX_RECENT_MODELS) : [] } catch { return [] }
}
export function pushRecentModel(entry: RecentModel): RecentModel[] {
  const next = [entry, ...loadRecentModels().filter((m) => !(m.providerId === entry.providerId && m.model === entry.model && m.accountId === entry.accountId))].slice(0, MAX_RECENT_MODELS)
  try { localStorage.setItem(RECENT_MODELS_KEY, JSON.stringify(next)) } catch { /* quota */ }
  return next
}
