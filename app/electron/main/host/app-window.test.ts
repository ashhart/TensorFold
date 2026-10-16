import assert from 'node:assert/strict'
import { pathToFileURL } from 'node:url'
import path from 'node:path'
import { createAppWindowRuntime, trustedRendererMediaAllowed } from './app-window'
import { store } from '../state'

class FakeWebContents {
  id = 7
  mainFrame = { url: '' }
  handlers = new Map<string, Array<(...args: any[]) => void>>()
  openHandler: ((details: { url: string }) => { action: string }) | null = null

  setWindowOpenHandler(handler: (details: { url: string }) => { action: string }) { this.openHandler = handler }
  on(name: string, handler: (...args: any[]) => void) {
    this.handlers.set(name, [...(this.handlers.get(name) ?? []), handler])
  }
}

class FakeBrowserWindow {
  static latest: FakeBrowserWindow | null = null
  static all: FakeBrowserWindow[] = []
  webContents = new FakeWebContents()
  onceHandlers = new Map<string, () => void>()
  loadedFile: string | null = null
  loadedUrl: string | null = null

  constructor(_options: unknown) { FakeBrowserWindow.latest = this; FakeBrowserWindow.all.push(this) }
  once(name: string, handler: () => void) { this.onceHandlers.set(name, handler) }
  show() {}
  loadFile(file: string) {
    this.loadedFile = file
    this.webContents.mainFrame.url = pathToFileURL(file).href
  }
  loadURL(url: string) {
    this.loadedUrl = url
    this.webContents.mainFrame.url = url
  }
}

const previousDevUrl = process.env.VITE_DEV_SERVER_URL
delete process.env.VITE_DEV_SERVER_URL
try {
  const baseDir = '/Applications/TensorFold-Studio.app/Contents/Resources/app.asar/dist-electron'
  let closed = 0
  const runtime = createAppWindowRuntime({
    BrowserWindow: FakeBrowserWindow as any,
    baseDir,
    openExternalSafely: async () => {},
    session: {} as any,
    onWindowClosed: () => { closed++ },
  })
  runtime.createWindow()
  const win = FakeBrowserWindow.latest!
  const entry = path.join(baseDir, '../dist/index.html')
  const entryUrl = pathToFileURL(entry).href
  assert.equal(win.loadedFile, entry)

  for (const eventName of ['will-navigate', 'will-redirect']) {
    const guard = win.webContents.handlers.get(eventName)?.[0]
    assert.ok(guard, `${eventName} guard is installed`)
    let prevented = false
    guard!({ preventDefault: () => { prevented = true } }, entryUrl)
    assert.equal(prevented, false, `${eventName} permits only the exact app entry`)
    guard!({ preventDefault: () => { prevented = true } }, 'file:///tmp/rogue.html')
    assert.equal(prevented, true, `${eventName} blocks another file URL`)
    prevented = false
    guard!({ preventDefault: () => { prevented = true } }, `${entryUrl}?unexpected=1`)
    assert.equal(prevented, true, `${eventName} blocks a modified entry URL`)
  }

  const firstWindow = win
  runtime.createWindow()
  const replacement = FakeBrowserWindow.latest!
  assert.notEqual(replacement, firstWindow)
  firstWindow.onceHandlers.get('closed')?.()
  assert.equal(store.win, replacement as any, 'a late close from an older window cannot clear its replacement')
  assert.equal(closed, 0, 'an older window cannot tear down lifecycle state owned by its replacement')
  replacement.onceHandlers.get('closed')?.()
  assert.equal(store.win, null, 'closing the current window clears the shared BrowserWindow reference')
  assert.equal(closed, 1, 'current window close runs the lifecycle teardown hook')
} finally {
  store.win = null
  if (previousDevUrl == null) delete process.env.VITE_DEV_SERVER_URL
  else process.env.VITE_DEV_SERVER_URL = previousDevUrl
}

// Permission policy: getUserMedia audio is available to the exact trusted renderer and
// nothing else. Exercise both Electron handlers because Chromium consults both.
{
  const baseDir = '/Applications/TensorFold-Studio.app/Contents/Resources/app.asar/dist-electron'
  let requestHandler: ((contents: any, permission: string, callback: (allowed: boolean) => void, details: any) => void) | null = null
  let checkHandler: ((contents: any, permission: string, origin: string, details: any) => boolean) | null = null
  let headersHandler: ((details: any, callback: (result: any) => void) => void) | null = null
  const fakeSession = {
    defaultSession: {
      webRequest: { onHeadersReceived: (handler: typeof headersHandler) => { headersHandler = handler } },
      setPermissionRequestHandler: (handler: typeof requestHandler) => { requestHandler = handler },
      setPermissionCheckHandler: (handler: typeof checkHandler) => { checkHandler = handler },
    },
  }
  const runtime = createAppWindowRuntime({
    BrowserWindow: FakeBrowserWindow as any,
    baseDir,
    openExternalSafely: async () => {},
    session: fakeSession as any,
  })
  runtime.createWindow()
  runtime.installSessionHardening()
  let hardenedHeaders: any
  headersHandler!({ responseHeaders: {} }, (result) => { hardenedHeaders = result.responseHeaders })
  assert.deepEqual(hardenedHeaders['Referrer-Policy'], ['no-referrer'])
  assert.match(hardenedHeaders['Content-Security-Policy'][0], /require-trusted-types-for 'script'/)
  const contents = FakeBrowserWindow.latest!.webContents
  const entryUrl = pathToFileURL(path.join(baseDir, '../dist/index.html')).href
  const request = (target: any, permission: string, details: any) => {
    let allowed: boolean | null = null
    requestHandler!(target, permission, (next) => { allowed = next }, details)
    return allowed
  }

  assert.equal(request(contents, 'media', { requestingUrl: entryUrl, isMainFrame: true, mediaTypes: ['audio'] }), true, 'trusted main-frame microphone request is granted')
  assert.equal(request(contents, 'media', { requestingUrl: 'file:///tmp/rogue.html', isMainFrame: true, mediaTypes: ['audio'] }), false, 'another file URL is denied')
  assert.equal(request(contents, 'media', { requestingUrl: entryUrl, isMainFrame: false, mediaTypes: ['audio'] }), false, 'child-frame microphone request is denied')
  assert.equal(request(contents, 'media', { requestingUrl: entryUrl, isMainFrame: true, mediaTypes: ['video'] }), false, 'camera access remains denied')
  assert.equal(request(contents, 'notifications', { requestingUrl: entryUrl, isMainFrame: true }), false, 'unrelated permissions remain denied')
  assert.equal(request(new FakeWebContents(), 'media', { requestingUrl: entryUrl, isMainFrame: true, mediaTypes: ['audio'] }), false, 'unregistered WebContents is denied')

  assert.equal(checkHandler!(contents, 'media', 'file://', { requestingUrl: entryUrl, isMainFrame: true, mediaType: 'audio' }), true, 'permission check accepts trusted audio')
  assert.equal(checkHandler!(contents, 'media', 'file://', { requestingUrl: `${entryUrl}?changed=1`, isMainFrame: true, mediaType: 'audio' }), false, 'permission check denies modified entry URL')
  assert.equal(checkHandler!(contents, 'media', 'file://', { requestingUrl: entryUrl, isMainFrame: true, mediaType: 'video' }), false, 'permission check denies video')
}

assert.equal(trustedRendererMediaAllowed('media', { requestingUrl: 'http://127.0.0.1:5173/', isMainFrame: true, mediaTypes: ['audio'] }, 'http://127.0.0.1:5173/'), true)
assert.equal(trustedRendererMediaAllowed('media', { requestingUrl: 'http://127.0.0.1:5173/other', isMainFrame: true, mediaTypes: ['audio'] }, 'http://127.0.0.1:5173/'), false)

console.log('app window navigation + media permission trust tests passed')
