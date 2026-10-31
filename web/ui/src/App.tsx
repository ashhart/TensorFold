import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type EngineInfo, type Machine, type Settings, ApiError, askTokenAgain } from './api'
import { setCustomBrands } from './brand'
import Palette from './Palette'
import { screenOf } from './route'
import Sidebar from './Sidebar'
import ErrorBoundary from './ErrorBoundary'
import Bench from './screens/Bench'
import Chat from './screens/Chat'
import Connect from './screens/Connect'
import Fleet from './screens/Fleet'
import Teach from './screens/Teach'
import Models from './screens/Models'
import Monitor from './screens/Monitor'
import SettingsScreen from './screens/Settings'
import Home from './screens/Home'

export type Route = 'home' | 'chat' | 'teach' | 'models' | 'connect' | 'bench' | 'monitor' | 'fleet' | 'settings'
// in the order a person uses them: what they came for first, the machine's own screens after
// the sidebar's order and the keys 1 to 9: Models right after Home, since choosing the AI comes first
export const ROUTES: Route[] = ['home', 'models', 'chat', 'teach', 'connect', 'bench', 'monitor', 'fleet', 'settings']
const TITLES: Record<Route, string> = { home: 'Home', chat: 'Chat', teach: 'Teach', models: 'Models', connect: 'Connect', bench: 'Bench', monitor: 'Monitor', fleet: 'Fleet', settings: 'Settings' }
const readRoute = (): Route => {
  const h = screenOf() as Route                 // `#bench/<run>` is the Bench, showing that run
  return ROUTES.includes(h) ? h : 'home'
}
type Toast = { id: number; text: string; kind: 'ok' | 'bad' }
const typing = (t: EventTarget | null) => t instanceof HTMLElement && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)

export default function App() {
  const [route, setRoute] = useState<Route>(readRoute)
  const [machine, setMachine] = useState<Machine | null>(null)
  const [engine, setEngine] = useState<EngineInfo | null>(null)
  const [settings, setSettings] = useState<Settings | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [studioDown, setStudioDown] = useState(false)
  const [missed, setMissed] = useState(0)                                // failed polls in a row, for the page
  const [now, setNow] = useState(() => Date.now())
  const [lastSeen, setLastSeen] = useState<Date | null>(null)          // when TensorFold Web last answered the engine poll
  const [decodeHistory, setDecodeHistory] = useState<number[]>([])     // kept here so it survives a visit to another screen
  const [toasts, setToasts] = useState<Toast[]>([])
  const [palette, setPalette] = useState(false)
  // the menu folded to its icons, remembered in this browser
  const [folded, setFolded] = useState(() => { try { return localStorage.getItem('tfs-sidebar') === 'folded' } catch { return false } })
  const fold = useCallback(() => setFolded((f) => {
    try { localStorage.setItem('tfs-sidebar', f ? 'full' : 'folded') } catch { /* no storage */ }
    return !f
  }), [])
  // the look applies at once and is remembered in this browser, then saved with the other settings
  // `?theme=light|dark|system` in the address shows that look for this visit only, for screenshots and links
  const pinned = (['light', 'dark', 'system'] as const).find((t) => t === new URLSearchParams(location.search).get('theme')) ?? null
  const [theme, setTheme] = useState<Settings['theme']>(() => { if (pinned) return pinned; try { return (localStorage.getItem('tfs-theme') as Settings['theme']) || 'system' } catch { return 'system' } })

  useEffect(() => {
    const on = () => setRoute(readRoute())
    addEventListener('hashchange', on)
    return () => removeEventListener('hashchange', on)
  }, [])
  // each screen names the tab, and a screen reader is taken to the new screen's content, not left in the menu
  const mainRef = useRef<HTMLElement>(null)
  const firstRoute = useRef(true)
  useEffect(() => {
    document.title = `${TITLES[route]} · TensorFold Web`
    if (firstRoute.current) { firstRoute.current = false; return }
    mainRef.current?.focus({ preventScroll: true })
  }, [route])
  // a refused token is not "the server is not answering": the banner below says what to do
  const explain = useCallback((what: string) => (e: unknown) => { if (!(e instanceof ApiError && e.status === 401)) setError(`${what}: ${e}`) }, [])
  const loadMachine = useCallback(() => api.machine().then((m) => { setMachine(m); setError(null) }).catch(explain('TensorFold Web server is not answering')), [explain])
  const loadSettings = useCallback(() => api.settings().then((s) => { setSettings(s); setError(null); if (!pinned) setTheme(s.theme) }).catch(explain('TensorFold Web could not read its settings')), [explain, pinned])
  // the makers kept in Settings come before the page's own list; the count re-renders the logos when they change
  const [, setBrandsRead] = useState(0)
  const refreshBrands = useCallback(() => api.brands().then((r) => { setCustomBrands(r.brands); setBrandsRead((n) => n + 1) }).catch(() => {}), [])
  useEffect(() => { loadMachine(); loadSettings(); refreshBrands() }, [loadMachine, loadSettings, refreshBrands])
  const machineRef = useRef<Machine | null>(null)
  const settingsRef = useRef<Settings | null>(null)
  useEffect(() => { machineRef.current = machine }, [machine])
  useEffect(() => { settingsRef.current = settings }, [settings])
  const [needsToken, setNeedsToken] = useState(false)
  const toast = useCallback((text: string, kind: 'ok' | 'bad' = 'ok') => {
    const id = Date.now() + Math.random()
    setToasts((t) => [...t.slice(-3), { id, text, kind }])
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === 'bad' ? 5000 : 2200)
  }, [])
  const down = useRef(0)                                                 // failed polls in a row
  const refreshEngine = useCallback(() => api.engine().then((e) => {
    setEngine(e); setStudioDown(false); setLastSeen(new Date()); down.current = 0; setMissed(0); setNeedsToken(false)
    setDecodeHistory((d) => [...d.slice(-119), e.decode_tps ?? 0])
    if (!machineRef.current) loadMachine()                              // refused at first (the token came later)
    if (!settingsRef.current) loadSettings()
  }).catch((e) => {
    if (e instanceof ApiError && e.status === 401) { setNeedsToken(true); return }   // declined: not a silent TensorFold Web
    setStudioDown(true); down.current += 1; setMissed(down.current)
  }), [loadMachine, loadSettings])
  const refreshSettings = useCallback(() => api.settings().then(setSettings).catch((e) => toast(`Settings not reloaded: ${e}`, 'bad')), [toast])
  // every 2 s while TensorFold Web answers; while it does not, the polls space out up to 10 s
  useEffect(() => {
    let t: ReturnType<typeof setTimeout>
    let live = true
    const loop = async () => {
      await refreshEngine()
      if (live) t = setTimeout(loop, Math.min(10000, 2000 * 2 ** Math.min(3, down.current)))
    }
    loop()
    return () => { live = false; clearTimeout(t) }
  }, [refreshEngine])
  useEffect(() => {
    const t = theme
    const mq = matchMedia('(prefers-color-scheme: dark)')
    const apply = () => {
      const dark = t === 'dark' || (t === 'system' && mq.matches)
      dark ? document.documentElement.setAttribute('data-theme', 'dark') : document.documentElement.removeAttribute('data-theme')
    }
    apply()
    mq.addEventListener('change', apply)
    return () => mq.removeEventListener('change', apply)
  }, [theme])
  // keys: ⌘K or Ctrl+K opens the palette, 1 to 9 switch screens, / goes to the chat composer, [ folds the menu
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); setPalette((p) => !p); return }
      if (e.metaKey || e.ctrlKey || e.altKey || typing(e.target)) return
      const n = Number(e.key)
      if (n >= 1 && n <= ROUTES.length) { location.hash = `#${ROUTES[n - 1]}` }
      else if (e.key === '[') fold()
      else if (e.key === '/') { e.preventDefault(); location.hash = '#chat'; setTimeout(() => dispatchEvent(new CustomEvent('tfs:focus-composer')), 0) }
    }
    addEventListener('keydown', on)
    return () => removeEventListener('keydown', on)
  }, [fold])

  // the look, applied and remembered for the next load (index.html reads it before the first paint)
  const rememberTheme = useCallback((t: Settings['theme']) => {
    setTheme(t)
    try { localStorage.setItem('tfs-theme', t) } catch { /* no storage */ }
  }, [])
  const saveSettings = async (patch: Partial<Settings>) => {
    const s = await api.saveSettings(patch)
    setSettings(s)
    if (patch.theme) rememberTheme(s.theme)
    return s
  }
  const onTheme = useCallback((t: Settings['theme']) => {
    rememberTheme(t)                                   // at once; the save follows
    api.saveSettings({ theme: t }).then(setSettings).catch((e) => toast(`The look applies here but was not saved: ${e}`, 'bad'))
  }, [toast, rememberTheme])
  const openPalette = useCallback(() => setPalette(true), [])
  // stale once TensorFold Web has been silent for 10 s and three polls: one missed poll is not worth greying the page
  useEffect(() => { if (!studioDown) return; const t = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(t) }, [studioDown])
  const age = lastSeen ? Math.round((now - lastSeen.getTime()) / 1000) : null
  const stale = studioDown && missed >= 3 && (age == null || age >= 10)
  const ctx = { machine, engine, settings, refreshEngine, refreshSettings, refreshBrands, saveSettings, toast, openPalette, stale, decodeHistory }
  return (
    <div className={folded ? 'app folded' : 'app'}>
      <a className="skip" href="#main" onClick={(e) => { e.preventDefault(); mainRef.current?.focus() }}>Skip to content</a>
      <Sidebar route={route} engine={engine} machine={machine} onPalette={openPalette} folded={folded} onFold={fold} />
      <main id="main" ref={mainRef} tabIndex={-1} className={stale ? 'stale' : undefined}>
        {needsToken && <div className="notice bad">This TensorFold Web needs its access token. <button className="btn sm" onClick={() => askTokenAgain().then((t) => { if (t) refreshEngine() })}>Enter the token</button></div>}
        {error && <div className="notice bad">{error}</div>}
        {stale && <div className="notice stale-banner" role="status">TensorFold Web server is not answering{lastSeen ? <> since {lastSeen.toLocaleTimeString()}; the figures below are {age != null && age < 120 ? `${age} s` : `${Math.round((age ?? 0) / 60)} min`} old</> : null}. Engine actions wait until it answers again.</div>}
        <ErrorBoundary name={TITLES[route]}>
          {route === 'home' && <Home {...ctx} />}
          {route === 'models' && <Models {...ctx} />}
          {route === 'monitor' && <Monitor {...ctx} />}
          {/* Chat stays mounted so a conversation, and a reply still streaming, survive a visit to another screen */}
          <div hidden={route !== 'chat'}><ErrorBoundary name="Chat"><Chat {...ctx} /></ErrorBoundary></div>
          {route === 'bench' && <Bench {...ctx} />}
          {route === 'teach' && <Teach {...ctx} />}
          {route === 'connect' && <Connect {...ctx} />}
          {route === 'settings' && <SettingsScreen {...ctx} />}
          {route === 'fleet' && <Fleet {...ctx} />}
        </ErrorBoundary>
      </main>
      <Palette open={palette} onClose={() => setPalette(false)} stale={stale} engine={engine} settings={settings} onTheme={onTheme} toast={toast} refreshEngine={refreshEngine} refreshSettings={refreshSettings} />
      <div className="toasts" aria-live="polite">{toasts.map((t) => <div key={t.id} className={`toast ${t.kind}`}>{t.text}</div>)}</div>
    </div>
  )
}
