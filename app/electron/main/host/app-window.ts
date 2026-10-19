import type { BrowserWindow as BrowserWindowType, session as sessionType } from 'electron'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import { buildSessionCsp } from '../../agent/security'
import { store } from '../state'
import { clearTrustedRenderer, configureTrustedRenderer, isTrustedRendererUrl, validateSender } from './ipc'

interface AppWindowRuntimeDeps {
  BrowserWindow: typeof BrowserWindowType
  baseDir: string
  openExternalSafely(url: string, opts?: { allowHttpLocalhost?: boolean }): Promise<void>
  session: typeof sessionType
  onWindowClosed?(): void
  onWindowCreated?(win: BrowserWindowType): void
  shouldShowWindow?(): boolean
  installTextEditing?(win: BrowserWindowType): void
  beforeWindowClose?(win: BrowserWindowType): Promise<void>
  shouldGuardWindowClose?(): boolean
  onWindowCloseFailure?(error: unknown): void
  /** Report a renderer that never loaded or died, so a blank window is not mistaken for a hang. */
  onRendererFailure?(detail: string): void
}

interface MediaPermissionDetails {
  requestingUrl?: string
  isMainFrame?: boolean
  mediaTypes?: string[]
  mediaType?: string
}

// Pure policy seam for tests: only the exact trusted entry's main frame may request audio.
// Camera/video, child frames, query-modified/file-neighbour URLs, and every other Electron
// permission remain denied.
export function trustedRendererMediaAllowed(permission: string, details: MediaPermissionDetails, entryUrl: string): boolean {
  if (permission !== 'media' || details.isMainFrame !== true) return false
  if (!details.requestingUrl || !isTrustedRendererUrl(details.requestingUrl, entryUrl)) return false
  if (details.mediaType && details.mediaType !== 'audio') return false
  if (details.mediaTypes?.length && details.mediaTypes.some((kind) => kind !== 'audio')) return false
  return true
}

export function createAppWindowRuntime(deps: AppWindowRuntimeDeps) {
  const { BrowserWindow, baseDir, openExternalSafely, session } = deps

  const rendererEntryUrl = () => {
    const devUrl = process.env.VITE_DEV_SERVER_URL
    return devUrl
      ? new URL(devUrl).href
      : pathToFileURL(path.join(baseDir, '../dist/index.html')).href
  }

  function installSessionHardening() {
    const devUrl = process.env.VITE_DEV_SERVER_URL
    const connectSources: string[] = []
    if (devUrl) {
      const origin = new URL(devUrl)
      connectSources.push(origin.origin)
      const socket = new URL(origin.origin)
      socket.protocol = socket.protocol === 'https:' ? 'wss:' : 'ws:'
      connectSources.push(socket.origin)
    }
    const csp = buildSessionCsp({ dev: !!devUrl, connectSources })
    session.defaultSession.webRequest.onHeadersReceived((details, callback) => {
      callback({ responseHeaders: { ...details.responseHeaders, 'Content-Security-Policy': [csp], 'Referrer-Policy': ['no-referrer'] } })
    })
    const entryUrl = rendererEntryUrl()
    session.defaultSession.setPermissionRequestHandler((contents, permission, callback, details) => {
      const senderTrusted = validateSender(contents.mainFrame, contents)
      callback(senderTrusted && trustedRendererMediaAllowed(permission, details, entryUrl))
    })
    session.defaultSession.setPermissionCheckHandler((contents, permission, requestingOrigin, details) => {
      if (!contents || !validateSender(contents.mainFrame, contents)) return false
      return trustedRendererMediaAllowed(permission, {
        ...details,
        requestingUrl: details.requestingUrl || requestingOrigin,
        mediaTypes: details.mediaType ? [details.mediaType] : undefined,
      }, entryUrl)
    })
  }

  function createWindow() {
    const appWindow = new BrowserWindow({
      width: 1180,
      height: 800,
      minWidth: 720,
      minHeight: 520,
      show: false,
      // Linux uses the window manager's native title bar and controls; no renderer IPC needed.
      titleBarStyle: process.platform === 'linux' ? 'default' : 'hiddenInset',
      trafficLightPosition: process.platform === 'linux' ? undefined : { x: 18, y: 22 },
      // macOS glass: native vibrancy blurs the desktop wallpaper through the window
      // (the dark tint comes from the translucent renderer surfaces on top). A clear
      // backgroundColor lets the material show; other platforms keep a solid dark.
      backgroundColor: process.platform === 'darwin' ? '#00000000' : '#111216',
      vibrancy: process.platform === 'darwin' ? 'under-window' : undefined,
      visualEffectState: 'active',
      webPreferences: {
        preload: path.join(baseDir, 'preload.js'),
        contextIsolation: true,
        nodeIntegration: false,
        sandbox: true,
      },
    })
    store.win = appWindow
    deps.onWindowCreated?.(appWindow)

    appWindow.once('ready-to-show', () => {
      if (store.win === appWindow && (deps.shouldShowWindow?.() ?? true)) appWindow.show()
    })

    const devUrl = process.env.VITE_DEV_SERVER_URL
    const appEntryUrl = rendererEntryUrl()
    const appWebContents = appWindow.webContents
    configureTrustedRenderer(appWebContents, appEntryUrl)
    deps.installTextEditing?.(appWindow)
    if (deps.beforeWindowClose) {
      let closing = false
      let saved = false
      appWindow.on('close', (event) => {
        if (deps.shouldGuardWindowClose?.() === false) return
        if (saved) { saved = false; return }
        event.preventDefault()
        if (closing) return
        closing = true
        appWindow.setEnabled(false)
        void Promise.resolve().then(() => deps.beforeWindowClose!(appWindow)).then(() => {
          closing = false
          if (!appWindow.isDestroyed()) { saved = true; appWindow.close() }
          if (!appWindow.isDestroyed()) appWindow.setEnabled(true)
        }, (error) => { closing = false; if (!appWindow.isDestroyed()) appWindow.setEnabled(true); deps.onWindowCloseFailure?.(error) })
      })
    }
    appWindow.once('closed', () => {
      clearTrustedRenderer(appWebContents)
      // Only clear the shared reference when this is still the current window. A delayed `closed`
      // from an older window must not erase a replacement created by macOS activation or cancel
      // prompts already owned by that replacement.
      if (store.win === appWindow) {
        store.win = null
        try { deps.onWindowClosed?.() } catch { /* teardown hooks are best-effort */ }
      }
    })

    // A failed load or a dead renderer otherwise leaves the static boot splash on screen forever.
    appWebContents.on('did-fail-load', (_event, errorCode, errorDescription, validatedURL, isMainFrame) => {
      if (!isMainFrame) return
      deps.onRendererFailure?.(`The interface failed to load (${errorCode} ${errorDescription}) for ${validatedURL}.`)
    })
    appWebContents.on('render-process-gone', (_event, details) => {
      deps.onRendererFailure?.(`The interface process stopped (${details.reason}${details.exitCode != null ? `, exit ${details.exitCode}` : ''}).`)
    })
    appWindow.webContents.setWindowOpenHandler(({ url }) => {
      openExternalSafely(url, { allowHttpLocalhost: true }).catch(() => {})
      return { action: 'deny' }
    })

    const guardNavigation = (event: { preventDefault(): void }, url: string) => {
      if (!isTrustedRendererUrl(url, appEntryUrl)) event.preventDefault()
    }
    appWindow.webContents.on('will-navigate', guardNavigation)
    // Server redirects are a separate Electron event; pin those to the same exact entry too.
    appWindow.webContents.on('will-redirect', guardNavigation)

    if (devUrl) {
      appWindow.loadURL(appEntryUrl)
    } else {
      appWindow.loadFile(path.join(baseDir, '../dist/index.html'))
    }
  }

  return { createWindow, installSessionHardening }
}
