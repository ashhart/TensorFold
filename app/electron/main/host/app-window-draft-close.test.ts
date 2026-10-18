import assert from 'node:assert/strict'
import { pathToFileURL } from 'node:url'
import { createAppWindowRuntime } from './app-window'
import { store } from '../state'

class FakeContents {
  mainFrame = { url: '' }
  on() {}
  setWindowOpenHandler() {}
}
class FakeWindow {
  static latest: FakeWindow
  webContents = new FakeContents()
  handlers = new Map<string, Array<(...args: any[]) => void>>()
  oneShot = new Map<string, (...args: any[]) => void>()
  destroyed = false
  enabled = true
  cancelOtherClose = false
  closes = 0
  constructor(_options: unknown) { FakeWindow.latest = this }
  once(name: string, callback: (...args: any[]) => void) { this.oneShot.set(name, callback) }
  on(name: string, callback: (...args: any[]) => void) { this.handlers.set(name, [...(this.handlers.get(name) ?? []), callback]) }
  show() {}
  loadFile(file: string) { this.webContents.mainFrame.url = pathToFileURL(file).href }
  loadURL(url: string) { this.webContents.mainFrame.url = url }
  setEnabled(value: boolean) { this.enabled = value }
  isDestroyed() { return this.destroyed }
  close() {
    this.closes++
    let prevented = false
    const event = { preventDefault() { prevented = true } }
    for (const callback of this.handlers.get('close') ?? []) callback(event)
    if (this.cancelOtherClose) event.preventDefault()
    if (!prevented) { this.destroyed = true; this.oneShot.get('closed')?.() }
  }
}
function pending() { let resolve!: () => void, reject!: (error: Error) => void; const promise = new Promise<void>((ok, no) => { resolve = ok; reject = no }); return { promise, resolve, reject } }
async function settle() { for (let i = 0; i < 15; i++) await Promise.resolve() }
const originalWindow = store.win
try {
  let gate = pending(), calls = 0, failures = 0, guard = true
  const runtime = createAppWindowRuntime({ BrowserWindow: FakeWindow as any, baseDir: '/private/qa/Contents/Resources/app.asar/dist-electron', openExternalSafely: async () => {}, session: {} as any,
    shouldGuardWindowClose: () => guard,
    beforeWindowClose: async () => { calls++; await gate.promise },
    onWindowCloseFailure() { failures++ },
  })
  runtime.createWindow(); let win = FakeWindow.latest
  win.close(); win.close(); await settle()
  assert.equal(calls, 1, 'repeated close attempts share one pending renderer flush')
  assert.equal(win.destroyed, false)
  assert.equal(win.enabled, false, 'keyboard input cannot arrive after the close-flush snapshot')
  gate.resolve(); await settle()
  assert.equal(win.destroyed, true, 'renderer is destroyed only after successful flush')
  assert.equal(failures, 0)

  gate = pending(); runtime.createWindow(); win = FakeWindow.latest
  win.close(); await settle(); gate.reject(new Error('Synthetic draft write refused')); await settle()
  assert.equal(win.destroyed, false)
  assert.equal(win.enabled, true, 'a failed close restores editing and keeps the renderer alive')
  assert.equal(failures, 1)
  gate = pending(); win.close(); await settle(); assert.equal(win.enabled, false)
  gate.resolve(); await settle(); assert.equal(win.destroyed, true)

  gate = pending(); runtime.createWindow(); win = FakeWindow.latest
  win.cancelOtherClose = true
  win.close(); await settle(); gate.resolve(); await settle()
  assert.equal(win.destroyed, false, 'another close listener may still cancel the resumed close')
  assert.equal(win.enabled, true, 'a separately cancelled close cannot leave the app disabled')
  const afterCancelled = calls
  win.cancelOtherClose = false; gate = pending(); win.close(); await settle()
  assert.equal(calls, afterCancelled + 1, 'retry after cancelled close must flush any newly typed draft again')
  gate.resolve(); await settle(); assert.equal(win.destroyed, true)

  runtime.createWindow(); win = FakeWindow.latest; guard = false
  const beforeSettledQuit = calls
  win.close(); await settle()
  assert.equal(win.destroyed, true, 'a final already-settled Quit closes without another fence')
  assert.equal(calls, beforeSettledQuit)

  let syncFailures = 0
  createAppWindowRuntime({ BrowserWindow: FakeWindow as any, baseDir: '/private/qa/Contents/Resources/app.asar/dist-electron', openExternalSafely: async () => {}, session: {} as any,
    beforeWindowClose() { throw new Error('Synthetic synchronous bridge failure') }, onWindowCloseFailure() { syncFailures++ },
  }).createWindow()
  win = FakeWindow.latest; win.close(); await settle()
  assert.equal(syncFailures, 1)
  assert.equal(win.destroyed, false); assert.equal(win.enabled, true)
  console.log('Production window close: await draft flush, freeze input, duplicate guard, failed and independently cancelled close retries, settled Quit and synchronous failure recovery passed')
} finally { store.win = originalWindow }
