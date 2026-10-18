import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react'
import { AppState } from 'react-native'
import { useSession } from './session'
import { watchSnapshot } from './watch-relay'
import { backgroundWatch } from './watch-native'

const Context = createContext({ available: false, enabled: false, busy: false, error: null as string | null, setEnabled: async (_: boolean) => {} })
export function WatchBridgeProvider({ children }: { children: ReactNode }) {
  const session = useSession(); const current = useRef(session); current.current = session
  const [foreground, setForeground] = useState(AppState.currentState === 'active')
  const [enabled, setEnabledState] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const generation = useRef(0)
  const identity = session.session?.sessionId
  useEffect(() => backgroundWatch.onRevoke(() => { generation.current++; setEnabledState(false); setBusy(false); setError(null) }), [])
  useEffect(() => { const subscription = AppState.addEventListener('change', state => setForeground(state === 'active')); return () => subscription.remove() }, [])
  useEffect(() => {
    let alive = true
    const mine = ++generation.current
    setEnabledState(false); setBusy(false); setError(null)
    // Native saved consent survives React bootstrap and protected-data waits.
    if (!session.ready || !identity || !backgroundWatch.available) return
    void backgroundWatch.restore(identity).then((status) => {
      if (!alive || mine !== generation.current || current.current.session?.sessionId !== identity) return
      setEnabledState(status.enabled); setError(status.reason ?? null)
    }).catch(() => { if (alive && mine === generation.current) setError('Could not inspect background watch access; unlock your phone and try again.') })
    return () => { alive = false }
  }, [identity, session.ready, foreground])
  const setEnabled = async (next: boolean) => {
    const captured = current.current.session
    if (!captured) return
    const mine = ++generation.current
    if (!next) setEnabledState(false)
    setBusy(true); setError(null)
    try {
      const accepted = await backgroundWatch.set(captured, next)
      if (mine === generation.current && current.current.session?.sessionId === captured.sessionId) setEnabledState(accepted)
    } catch (failure) { if (mine === generation.current) { setEnabledState(false); setError((failure as Error).message || 'Background watch access could not be changed.') } }
    finally { if (mine === generation.current) setBusy(false) }
  }
  useEffect(() => {
    if (!foreground || !enabled || !session.session || !session.host || session.error) return
    const snapshot = watchSnapshot(session.host, session.session.sessionId, session.lastSynced ?? 0, null, Date.now())
    void backgroundWatch.publish(session.session.sessionId, JSON.stringify(snapshot)).catch(() => setError('Could not sync watch status; refresh from your watch.'))
  }, [foreground, enabled, session.host, session.error, session.lastSynced, session.session])
  // Native AppDelegate/Watch Connectivity owns commands, including cold background wakes.
  // Unmounting a React component never revokes explicitly enabled native access.
  return <Context.Provider value={{available: backgroundWatch.available, enabled, busy, error, setEnabled}}>{children}</Context.Provider>
}
export function useWatchBridge() { return useContext(Context) }
