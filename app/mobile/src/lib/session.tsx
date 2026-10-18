import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { AppState, Linking, Platform } from 'react-native'
import * as SecureStore from 'expo-secure-store'
import { HostError, StudioClient, type DeviceSession, type MobileHost } from './client'
import { fetchForPin } from './pinned-fetch'
import { readSavedSession } from './session-restore'
import { disableBackgroundWatch } from './watch-native'

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
  // Identity changes are separate from foreground polling cancellations.
  const identityGeneration = useRef(0)
  const restoreDeferred = useRef(false)
  const secureWrites = useRef<Promise<unknown>>(Promise.resolve())
  const mutateSecure = useCallback((operation: () => Promise<unknown>) => {
    const next = secureWrites.current.catch(() => {}).then(operation)
    secureWrites.current = next
    return next
  }, [])
  const controller = useRef<AbortController | null>(null)
  const client = useMemo(() => session ? new StudioClient(session, fetchForPin(session.certificatePin)) : null, [session])
  const invalidate = useCallback(() => { generation.current++; controller.current?.abort(); inflight.current = null }, [])
  const cancelIdentity = useCallback(() => { identityGeneration.current++ }, [])

  const forget = useCallback(async () => {
    const current = ++identityGeneration.current
    restoreDeferred.current = false; invalidate()
    setSession(null); setHost(null); setLastSynced(null); setSyncing(false); setReady(true); setError(null); failures.current = 0
    // Always attempt both revocations. A native Keychain failure must not prevent
    // clearing the ordinary phone token, or disappear behind a polling error.
    const results = await Promise.allSettled([
      disableBackgroundWatch(),
      Platform.OS === 'web' ? Promise.resolve() : mutateSecure(() => SecureStore.deleteItemAsync(KEY)),
    ])
    if (identityGeneration.current !== current) return
    if (results[0].status === 'rejected') setError('Watch access could not be cleared on this phone; revoke this device in TensorFold on your Mac.')
    else if (results[1].status === 'rejected') setError('This device could not clear its saved token; revoke the device in TensorFold.')
  }, [invalidate, mutateSecure])

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
    let restoring = false
    const restore = async () => {
      if (restoring) return
      restoring = true
      const current = identityGeneration.current
      const result = await readSavedSession(() => Platform.OS === 'web' ? Promise.resolve(null) : SecureStore.getItemAsync(KEY), Platform.OS === 'android' && __DEV__)
      restoring = false
      if (!alive || identityGeneration.current !== current) return
      if (result.kind === 'deferred') {
        restoreDeferred.current = true
        setError('The saved connection is temporarily unavailable; unlock your phone and open TensorFold to restore it.')
      } else {
        restoreDeferred.current = false
        if (result.kind === 'restored') { setSession(result.session); setError(null) }
        else {
          // Successful absent/invalid reads fail closed, but retain invalid bytes
          // until the user explicitly pairs or forgets this device.
          setSession(null)
          if (result.kind === 'invalid') setError('The saved connection could not be restored; pair this device again.')
          try { await disableBackgroundWatch() }
          catch { if (alive && identityGeneration.current === current) setError('Watch access could not be cleared on this phone; revoke this device in TensorFold on your Mac.') }
        }
      }
      if (alive && identityGeneration.current === current) setReady(true)
    }
    const appSub = AppState.addEventListener('change', (state) => {
      setForeground(state === 'active')
      if (state !== 'active') controller.current?.abort()
      else if (restoreDeferred.current) void restore()
    })
    void restore()
    return () => { alive = false; cancelIdentity(); invalidate(); linkSub.remove(); appSub.remove() }
  }, [invalidate, cancelIdentity])

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
        if (failure instanceof HostError && failure.status === 401) {
          const revokedGeneration = identityGeneration.current + 1
          await forget()
          if (identityGeneration.current === revokedGeneration) setError(previous => previous ?? failure.message)
        }
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
    const current = ++identityGeneration.current
    restoreDeferred.current = false; invalidate()
    const assertCurrent = () => { if (identityGeneration.current !== current) throw new Error('The device connection changed; pairing was cancelled.') }
    try { await disableBackgroundWatch() }
    catch {
      const message = 'Watch access could not be cleared on this phone; revoke this device in TensorFold on your Mac before pairing again.'
      if (identityGeneration.current === current) setError(message)
      throw new Error(message)
    }
    assertCurrent()
    setSession(null); setHost(null); setLastSynced(null); setSyncing(false)
    const next = await StudioClient.pair(baseUrl, credential, `TensorFold mobile ${Platform.OS}`, fetchForPin(certificatePin), certificatePin)
    assertCurrent()
    // Persist only the revocable device session, never the one-time pairing code.
    // The native watch relay owns its separate AfterFirstUnlock consent store.
    if (Platform.OS !== 'web') {
      try { await mutateSecure(async () => {
        assertCurrent()
        await SecureStore.setItemAsync(KEY, JSON.stringify(next), { keychainAccessible: SecureStore.WHEN_UNLOCKED_THIS_DEVICE_ONLY })
      }) }
      catch {
        assertCurrent()
        throw new Error('TensorFold paired this device, but secure storage is unavailable; revoke this device in TensorFold and generate a new pairing code.')
      }
    }
    assertCurrent()
    setHost(null); setSession(next); setReady(true); setPairingLink(''); setError(null); failures.current = 0
  }, [invalidate, mutateSecure])

  return <Context.Provider value={{ session, host, ready, syncing, error, lastSynced, pairingLink, setPairingLink, pair, forget, refresh, client }}>{children}</Context.Provider>
}

export function useSession(): SessionState { const value = useContext(Context); if (!value) throw new Error('SessionProvider is missing'); return value }
