import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react'
import { pairedToken } from './remote-pairing'

export interface FavoriteModel { providerId: string; accountId: string | null; model: string }
export const MAX_FAVORITE_MODELS = 128
const PREFIX = 'tensorfold.favoriteModels.v1:'
const EMPTY: readonly FavoriteModel[] = Object.freeze([])

export function favoriteModelKey(route: FavoriteModel): string {
  return JSON.stringify([route.providerId, route.accountId ?? null, route.model])
}

function boundedString(value: unknown, max: number): value is string {
  return typeof value === 'string' && !!value.trim() && value.length <= max && !/[\u0000-\u001f]/.test(value)
}

export function parseFavoriteModels(raw: string | null): readonly FavoriteModel[] {
  if (!raw || raw.length > 200_000) return EMPTY
  try {
    const parsed: unknown = JSON.parse(raw)
    if (!Array.isArray(parsed)) return EMPTY
    const seen = new Set<string>()
    return parsed.slice(0, MAX_FAVORITE_MODELS).flatMap((value: unknown) => {
      if (!value || typeof value !== 'object' || Array.isArray(value)) return []
      const item = value as Record<string, unknown>
      if (!boundedString(item.providerId, 320) || !boundedString(item.model, 1_024) || !(item.accountId === null || boundedString(item.accountId, 320))) return []
      const route = { providerId: item.providerId, accountId: item.accountId, model: item.model }
      const key = favoriteModelKey(route)
      if (seen.has(key)) return []
      seen.add(key)
      return [Object.freeze(route)]
    })
  } catch { return EMPTY }
}

export function orderFavoritesFirst<T>(items: readonly T[], favorites: readonly FavoriteModel[], route: (item: T) => FavoriteModel): T[] {
  const keys = new Set(favorites.map(favoriteModelKey))
  return [...items.filter((item) => keys.has(favoriteModelKey(route(item)))), ...items.filter((item) => !keys.has(favoriteModelKey(route(item))))]
}

/** Resolve only against the caller's currently allowed routes; removed accounts never reappear. */
export function favoriteModelChoices<T>(items: readonly T[], favorites: readonly FavoriteModel[], route: (item: T) => FavoriteModel): T[] {
  const available = new Map(items.map((item) => [favoriteModelKey(route(item)), item]))
  return favorites.flatMap((entry) => { const item = available.get(favoriteModelKey(entry)); return item ? [item] : [] })
}

export interface FavoriteModelsEnvironment { platform: string; origin: string; pairedCredential: string | null; ownerId: string | null }

/** Never put a pairing credential, token prefix or its unkeyed short hash in a preference key. */
export async function favoriteModelsScope(environment: FavoriteModelsEnvironment, digest: SubtleCrypto['digest'] | null = globalThis.crypto?.subtle?.digest.bind(globalThis.crypto.subtle) ?? null): Promise<string | null> {
  const owner = environment.ownerId ?? null
  if (owner !== null && !boundedString(owner, 320)) return null
  if (['darwin', 'win32', 'linux'].includes(environment.platform)) return PREFIX + JSON.stringify(['desktop', owner])
  if (environment.platform !== 'web-remote') return PREFIX + JSON.stringify(['demo', owner])
  if (!environment.pairedCredential || !environment.origin || !digest) return null
  // Origin alone is insufficient: re-pairing the same host as a different device/account
  // must not inherit the previous authenticated session's routes.
  const bytes = new TextEncoder().encode(JSON.stringify([environment.origin, environment.pairedCredential, owner]))
  const result = await digest('SHA-256', bytes)
  const hash = [...new Uint8Array(result)].map((byte) => byte.toString(16).padStart(2, '0')).join('')
  return PREFIX + `paired:${hash}`
}

export class FavoriteModelsPreferences {
  private raw: string | null | undefined
  private view: readonly FavoriteModel[] = EMPTY
  private listeners = new Set<() => void>()
  constructor(readonly key: string, private storage: () => Pick<Storage, 'getItem' | 'setItem'>, private events?: Pick<Window, 'addEventListener' | 'removeEventListener'>) {}
  get = (): readonly FavoriteModel[] => {
    let raw: string | null
    try { raw = this.storage().getItem(this.key) } catch { return this.view }
    if (this.raw !== raw) { this.raw = raw; this.view = parseFavoriteModels(raw) }
    return this.view
  }
  private changed = (event?: Event): void => {
    if (event && 'key' in event && (event as StorageEvent).key !== null && (event as StorageEvent).key !== this.key) return
    for (const listener of this.listeners) listener()
  }
  subscribe = (listener: () => void): (() => void) => {
    if (!this.listeners.size) this.events?.addEventListener('storage', this.changed)
    this.listeners.add(listener)
    return () => { this.listeners.delete(listener); if (!this.listeners.size) this.events?.removeEventListener('storage', this.changed) }
  }
  toggle(route: FavoriteModel): void {
    const valid = parseFavoriteModels(JSON.stringify([route]))
    if (valid.length !== 1) throw new Error('This model cannot be saved as a favourite.')
    // Writes require a successful fresh read. An unreadable saved list is unknown,
    // never an empty list that may be safely replaced by this click.
    let storage: Pick<Storage, 'getItem' | 'setItem'>, raw: string | null
    try { storage = this.storage(); raw = storage.getItem(this.key) }
    catch { throw new Error('Favourites could not be read on this device. Your saved favourites are unchanged; try again.') }
    const current = parseFavoriteModels(raw), key = favoriteModelKey(route)
    const exists = current.some((item) => favoriteModelKey(item) === key)
    if (!exists && current.length >= MAX_FAVORITE_MODELS) throw new Error('Remove a favourite before adding another model.')
    const next = exists ? current.filter((item) => favoriteModelKey(item) !== key) : [valid[0], ...current]
    const serialized = JSON.stringify(next)
    try { storage.setItem(this.key, serialized) } catch { throw new Error('Favourites could not be saved on this device. Check available storage and try again.') }
    this.raw = serialized; this.view = parseFavoriteModels(serialized)
    this.changed()
  }
}

const stores = new Map<string, FavoriteModelsPreferences>()
function storeFor(key: string): FavoriteModelsPreferences {
  let known = stores.get(key)
  if (!known) { known = new FavoriteModelsPreferences(key, () => localStorage, typeof window === 'undefined' ? undefined : window); stores.set(key, known) }
  return known
}
function environmentFor(ownerId: string | null, pairedSession = false): FavoriteModelsEnvironment {
  const bridge = typeof window === 'undefined' ? null : window.tensorfoldStudio ?? window.vontra
  const credential = bridge ? null : pairedToken()
  return { platform: bridge?.platform ?? (credential || pairedSession ? 'web-remote' : 'web'), origin: typeof location === 'undefined' ? '' : location.origin, pairedCredential: credential, ownerId }
}

/** Electron's existing profile partition owns local preferences; host/session scopes own paired ones. */
export function useFavoriteModels(ownerId: string | null = null) {
  const [authEpoch, setAuthEpoch] = useState(0)
  const pairedSession = useRef(!!pairedToken())
  const revokedCredential = useRef<string | null>(null)
  const environment = environmentFor(ownerId, pairedSession.current)
  if (environment.pairedCredential) pairedSession.current = true
  const revoked = environment.platform === 'web-remote' && !!revokedCredential.current && environment.pairedCredential === revokedCredential.current
  const environmentKey = JSON.stringify(environment)
  const currentEnvironment = useRef(environmentKey)
  currentEnvironment.current = environmentKey
  const [scope, setScope] = useState<{ environmentKey: string; key: string | null } | null>(null)
  const [failure, setFailure] = useState<{ environmentKey: string; message: string } | null>(null)
  useEffect(() => {
    let alive = true
    void (revoked ? Promise.resolve(null) : favoriteModelsScope(environment)).then((key) => { if (alive) setScope({ environmentKey, key }) }).catch(() => { if (alive) setScope({ environmentKey, key: null }) })
    return () => { alive = false }
  }, [environmentKey, authEpoch, revoked])
  useEffect(() => {
    const invalidate = (event: Event) => {
      if (event.type === 'storage' && (event as StorageEvent).key !== null && (event as StorageEvent).key !== 'vontra.paired.token') return
      if (event.type === 'vontra:paired-auth-failed') revokedCredential.current = pairedToken()
      setScope(null); setAuthEpoch((value) => value + 1)
    }
    window.addEventListener('storage', invalidate)
    window.addEventListener('vontra:paired-auth-failed', invalidate)
    return () => { window.removeEventListener('storage', invalidate); window.removeEventListener('vontra:paired-auth-failed', invalidate) }
  }, [])
  const key = scope?.environmentKey === environmentKey ? scope.key : null
  const store = useMemo(() => key ? storeFor(key) : null, [key])
  const subscribe = useCallback((listener: () => void) => store?.subscribe(listener) ?? (() => {}), [store])
  const snapshot = useCallback(() => store?.get() ?? EMPTY, [store])
  const favorites = useSyncExternalStore(subscribe, snapshot, () => EMPTY)
  const toggle = useCallback((route: FavoriteModel): boolean => {
    if (!store || revokedCredential.current && pairedToken() === revokedCredential.current || currentEnvironment.current !== environmentKey || JSON.stringify(environmentFor(ownerId, pairedSession.current)) !== environmentKey) return false
    try { store.toggle(route); setFailure(null); return true }
    catch (error) { setFailure({ environmentKey, message: error instanceof Error ? error.message : 'Favourites could not be saved.' }); return false }
  }, [store, environmentKey, ownerId])
  const keys = useMemo(() => new Set(favorites.map(favoriteModelKey)), [favorites])
  return {
    favorites, ready: !!store,
    error: failure?.environmentKey === environmentKey ? failure.message : scope?.environmentKey === environmentKey && !scope.key ? 'Favourites are unavailable in this session.' : null,
    isFavorite: (route: FavoriteModel) => keys.has(favoriteModelKey(route)), toggle,
  }
}
