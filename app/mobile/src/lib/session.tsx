import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AppState, Linking, Platform } from 'react-native'
import * as SecureStore from 'expo-secure-store'
import { HostError, StudioClient, validateHost, type DeviceSession, type MobileHost } from './client'
import { fetchForPin } from './pinned-fetch'

const KEY = 'tensorfold.mobile.device.v1'
interface SessionState {
  session: DeviceSession | null; host: MobileHost | null; ready: boolean; syncing: boolean; error: string | null;
  lastSynced: number | null; pairingLink: string; setPairingLink(value: string): void;
  pair(baseUrl: string, credential: string, certificatePin?: string): Promise<void>; forget(): Promise<void>; refresh(): Promise<void>; client: StudioClient | null
}
const Context = createContext<SessionState | null>(null)

export function SessionProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<DeviceSession | null>(null)
  const [host, setHost] = useState<MobileHost | null>(null)
  const [ready, setReady] = useState(false)
  const [syncing, setSyncing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [lastSynced, setLastSynced] = useState<number | null>(null)
  const [pairingLink, setPairingLink] = useState('')
  const [foreground, setForeground] = useState(AppState.currentState === 'active')
  const failures = useRef(0)
  const inflight = useRef<Promise<void> | null>(null)
  const generation = useRef(0)
  const controller = useRef<AbortController | null>(null)
  const client = useMemo(() => session ? new StudioClient(session, fetchForPin(session.certificatePin)) : null, [session])
  const invalidate = useCallback(() => { generation.current++; controller.current?.abort(); inflight.current = null }, [])

  const forget = useCallback(async () => {
    invalidate()
    setSession(null); setHost(null); setLastSynced(null); setError(null); failures.current = 0
    if (Platform.OS !== 'web') {
      try { await SecureStore.deleteItemAsync(KEY) }
      catch { setError('This device could not clear its saved token; revoke the device in TensorFold.') }
    }
  }, [invalidate])

  useEffect(() => {
    let alive = true
    const receive = (url: string | null) => {
      if (!url) return
      try {
        const parsed = new URL(url)
        const candidate = parsed.protocol === 'tensorfold-mobile:' ? parsed.searchParams.get('link') : url
        if (candidate && /[#?&]pair=/.test(candidate)) setPairingLink(candidate)
      } catch { /* A malformed external link never changes the saved host. */ }
    }
    void Linking.getInitialURL().then(receive)
    const linkSub = Linking.addEventListener('url', ({ url }) => receive(url))
    const appSub = AppState.addEventListener('change', (state) => { setForeground(state === 'active'); if (state !== 'active') controller.current?.abort() })
    void (async () => {
      try {
        const raw = Platform.OS === 'web' ? null : await SecureStore.getItemAsync(KEY)
        if (raw && alive) {
          const value = JSON.parse(raw) as DeviceSession
          if (typeof value.token !== 'string' || !value.token || typeof value.sessionId !== 'string' || !value.sessionId) throw new Error('Invalid device session')
          value.baseUrl = validateHost(value.baseUrl, Platform.OS === 'android' && __DEV__)
          if (value.certificatePin && !/^[a-fA-F0-9]{64}$/.test(value.certificatePin)) throw new Error('Invalid certificate pin')
          setSession(value)
        }
      } catch { if (alive) { setError('The saved connection could not be restored; pair this device again.'); if (Platform.OS !== 'web') { try { await SecureStore.deleteItemAsync(KEY) } catch { /* Remain signed out when secure storage is unavailable. */ } } } }
      finally { if (alive) setReady(true) }
    })()
    return () => { alive = false; invalidate(); linkSub.remove(); appSub.remove() }
  }, [invalidate])

  const refresh = useCallback(async () => {
    if (!session || !foreground) return
    if (inflight.current) return inflight.current
    const current = generation.current
    const aborter = new AbortController(); controller.current = aborter
    const request = (async () => {
      setSyncing(true)
      try {
        const next = await new StudioClient(session, fetchForPin(session.certificatePin)).host(aborter.signal)
        if (generation.current !== current) return
        setHost(next); setLastSynced(Date.now()); setError(null); failures.current = 0
      } catch (failure) {
        if (generation.current !== current || aborter.signal.aborted) return
        if (failure instanceof HostError && failure.status === 401) { await forget(); setError(failure.message) }
        else { failures.current++; setError((failure as Error).message || 'Could not reach TensorFold.') }
      } finally { if (generation.current === current) { setSyncing(false); inflight.current = null } }
    })()
    inflight.current = request
    return request
  }, [session, foreground, forget])

  useEffect(() => {
    if (!session || !foreground) return
    let active = true
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => { await refresh(); if (active) timer = setTimeout(tick, Math.min(30_000, 2_000 * 2 ** Math.min(failures.current, 4))) }
    void tick()
    return () => { active = false; clearTimeout(timer); invalidate() }
  }, [session, foreground, refresh, invalidate])

  const pair = useCallback(async (baseUrl: string, credential: string, certificatePin?: string) => {
    const next = await StudioClient.pair(baseUrl, credential, `TensorFold mobile ${Platform.OS}`, fetchForPin(certificatePin), certificatePin)
    // Persist only the revocable device session, never the one-time pairing code.
    if (Platform.OS !== 'web') {
      try { await SecureStore.setItemAsync(KEY, JSON.stringify(next), { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY }) }
      catch { throw new Error('TensorFold paired this device, but secure storage is unavailable; revoke this device in TensorFold and generate a new pairing code.') }
    }
    generation.current++; setHost(null); setSession(next); setPairingLink(''); setError(null); failures.current = 0
  }, [])

  return <Context.Provider value={{ session, host, ready, syncing, error, lastSynced, pairingLink, setPairingLink, pair, forget, refresh, client }}>{children}</Context.Provider>
}

export function useSession(): SessionState { const value = useContext(Context); if (!value) throw new Error('SessionProvider is missing'); return value }
