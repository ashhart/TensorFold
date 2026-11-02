import { BrowserWindow, session, screen } from 'electron'
import path from 'node:path'
import { pathToFileURL } from 'node:url'
import type { IpcMainInvokeEvent } from 'electron'
import { isTrustedRendererUrl } from '../ipc'
import { buildSessionCsp } from '../../../agent/security'
import type { QuickAskEvent } from '../../../../shared/quick-ask'

export function askWindowBounds(area: { x: number; y: number; width: number; height: number }, saved?: { x: number; y: number; width: number; height: number }, requestedHeight = 104) {
  const width = Math.min(area.width, 620)
  const height = Math.min(area.height, Math.max(88, Math.min(680, requestedHeight)))
  return { width, height, x: Math.max(area.x, Math.min(saved?.x ?? area.x + (area.width - width) / 2, area.x + area.width - width)), y: Math.max(area.y, Math.min(saved?.y ?? area.y + (area.height - height) / 3, area.y + area.height - height)) }
}
export function createAskWindow(baseDir: string, onError: (message: string) => void) {
  let win: BrowserWindow | null = null
  let quitting = false, savedBounds: ReturnType<typeof askWindowBounds> | undefined
  const entry = process.env.VITE_DEV_SERVER_URL ? new URL(process.env.VITE_DEV_SERVER_URL).href : pathToFileURL(path.join(baseDir, '../dist/index.html')).href
  const ownSession = () => {
    const isolated = session.fromPartition('tensorfold-quick-ask')
    const dev = process.env.VITE_DEV_SERVER_URL
    const sources = dev ? [new URL(dev).origin, new URL(dev).origin.replace(/^http/, 'ws')] : []
    isolated.webRequest.onHeadersReceived((details, callback) => callback({ responseHeaders: { ...details.responseHeaders, 'Content-Security-Policy': [buildSessionCsp({ dev: !!dev, connectSources: sources })], 'Referrer-Policy': ['no-referrer'] } }))
    const trusted = (contents: Electron.WebContents | null) => !!win && contents === win.webContents && !contents.isDestroyed() && isTrustedRendererUrl(contents.mainFrame.url, entry)
    isolated.setPermissionRequestHandler((contents, permission, callback, details) => callback(trusted(contents) && permission === 'media' && details.isMainFrame === true && 'mediaTypes' in details && details.mediaTypes?.length === 1 && details.mediaTypes[0] === 'audio'))
    isolated.setPermissionCheckHandler((contents, permission, _origin, details) => trusted(contents) && permission === 'media' && details.isMainFrame === true && details.mediaType === 'audio')
    return isolated
  }
  const send = (event: QuickAskEvent) => { if (win && !win.isDestroyed() && !win.webContents.isDestroyed()) win.webContents.send('quick-ask:event', event) }
  function open(): void {
    if (quitting) return
    if (win && !win.isDestroyed()) {
      const display = screen.getDisplayMatching(win.getBounds())
      win.setBounds(askWindowBounds(display.workArea, win.getBounds()))
      if (win.isMinimized()) win.restore()
      win.show(); win.focus(); send({ kind: 'focus' }); return
    }
    const display = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
    win = new BrowserWindow({ ...askWindowBounds(display.workArea, savedBounds), minWidth: 320, minHeight: 88, title: 'Ask TensorFold', show: false, frame: false, resizable: false, maximizable: false, fullscreenable: false, alwaysOnTop: true, backgroundColor: '#111216', autoHideMenuBar: true,
      webPreferences: { preload: path.join(baseDir, 'preload.js'), additionalArguments: ['--tensorfold-quick-ask'], contextIsolation: true, sandbox: true, nodeIntegration: false, session: ownSession() } })
    const current = win
    current.once('ready-to-show', () => { if (!current.isDestroyed() && !quitting) { current.show(); current.focus() } })
    current.on('close', event => { savedBounds = current.getBounds(); if (!quitting) { event.preventDefault(); current.hide() } })
    current.once('closed', () => { if (win === current) win = null })
    current.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
    const guard = (event: { preventDefault(): void }, url: string) => { if (!isTrustedRendererUrl(url, entry)) event.preventDefault() }
    current.webContents.on('will-navigate', guard); current.webContents.on('will-redirect', guard)
    current.webContents.on('render-process-gone', () => { current.destroy(); onError('Ask TensorFold stopped. Reopen it to recover the saved draft.') })
    current.webContents.on('did-fail-load', (_event, code, _description, _url, main) => { if (main && code !== -3) { current.destroy(); onError('Ask TensorFold could not load. Your saved draft is preserved.') } })
    void current.loadURL(entry).catch(() => {})
  }
  return {
    open, send,
    resize: (height: number) => {
      if (!Number.isFinite(height) || height < 88 || height > 680) throw new Error('Choose a supported Ask window height.')
      if (!win || win.isDestroyed()) return
      const bounds = win.getBounds(), display = screen.getDisplayMatching(bounds)
      win.setBounds(askWindowBounds(display.workArea, bounds, Math.ceil(height)))
    },
    hide: () => win?.hide(),
    visible: () => !!win && !win.isDestroyed() && win.isVisible(),
    destroy: () => { if (win && !win.isDestroyed()) win.destroy() },
    available: () => !!win && !win.isDestroyed() && !win.webContents.isDestroyed(),
    setEnabled: (enabled: boolean) => { if (win && !win.isDestroyed()) win.setEnabled(enabled) },
    authorize: (event: IpcMainInvokeEvent) => {
      try { return !!win && !win.isDestroyed() && event.sender === win.webContents && event.senderFrame === win.webContents.mainFrame && isTrustedRendererUrl(event.senderFrame.url, entry) } catch { return false }
    },
    setQuitting: (value: boolean) => { quitting = value },
  }
}
