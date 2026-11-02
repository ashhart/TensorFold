import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import ts from 'typescript'

const nodeRequire = createRequire(import.meta.url)
let latest: FakeWindow, requestPermission: any, checkPermission: any
class FakeWindow extends EventEmitter {
  destroyed = false; visible = false; enabled = true
  webContents = Object.assign(new EventEmitter(), { mainFrame: { url: '' }, isDestroyed: () => this.destroyed, send: () => {}, setWindowOpenHandler: (handler: any) => { (this as any).openHandler = handler } })
  constructor(public options: any) { super(); latest = this }
  loadURL(url: string) { this.webContents.mainFrame.url = url; this.emit('ready-to-show'); return Promise.resolve() }
  isDestroyed() { return this.destroyed }
  isVisible() { return this.visible }
  isMinimized() { return false }
  getBounds() { return this.bounds }
  bounds = { x: 30, y: 40, width: 620, height: 104 }
  setBounds(bounds: any) { this.bounds = bounds }
  show() { this.visible = true }
  hide() { this.visible = false }
  focus() {}
  setEnabled(value: boolean) { this.enabled = value }
  destroy() { this.destroyed = true; this.emit('closed') }
}
const source = readFileSync(path.join(import.meta.dirname, 'window.ts'), 'utf8')
const javascript = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText
const output = { exports: {} as any }
new Function('require', 'module', 'exports', javascript)((name: string) => {
  if (name === 'electron') return { BrowserWindow: FakeWindow, screen: { getCursorScreenPoint: () => ({}), getDisplayNearestPoint: () => ({ workArea: { x: 0, y: 0, width: 1200, height: 900 } }), getDisplayMatching: () => ({ workArea: { x: 0, y: 0, width: 1200, height: 900 } }) }, session: { fromPartition: () => ({ webRequest: { onHeadersReceived() {} }, setPermissionRequestHandler: (handler: any) => { requestPermission = handler }, setPermissionCheckHandler: (handler: any) => { checkPermission = handler } }) } }
  if (name === '../ipc') return { isTrustedRendererUrl: (raw: string, expected: string) => { const actual = new URL(raw), entry = new URL(expected); actual.hash = ''; entry.hash = ''; return actual.href === entry.href } }
  if (name === '../../../agent/security') return { buildSessionCsp: () => "default-src 'self'; object-src 'none'" }
  return nodeRequire(name)
}, output, output.exports)
const { createAskWindow, askWindowBounds } = output.exports
const bounds = askWindowBounds({ x: -1200, y: 0, width: 1200, height: 800 }, { x: 9000, y: -3000, width: 2000, height: 2000 })
assert.deepEqual(bounds, { x: -620, y: 0, width: 620, height: 104 })
const runtime = createAskWindow('/example/app/dist-electron', () => {})
runtime.open()
const win = latest!
assert.equal(win.options.webPreferences.sandbox, true); assert.equal(win.options.webPreferences.nodeIntegration, false); assert.equal(win.options.webPreferences.contextIsolation, true)
assert.deepEqual(win.options.webPreferences.additionalArguments, ['--tensorfold-quick-ask'])
assert.equal(win.options.height, 104)
assert.equal(win.options.resizable, false)
assert.equal(win.options.alwaysOnTop, true)
runtime.resize(450); assert.equal(win.bounds.height, 450)
for (const height of [NaN, Infinity, 0, 681]) assert.throws(() => runtime.resize(height), /supported/)
runtime.open(); assert.equal(win.bounds.height, 104, 'reopening always returns to the compact bar')
assert.deepEqual(askWindowBounds({ x: 0, y: 0, width: 400, height: 300 }, undefined, 600), { x: 0, y: 0, width: 400, height: 300 })
const sender = { sender: win.webContents, senderFrame: win.webContents.mainFrame }
assert.equal(runtime.authorize(sender), true)
assert.equal(runtime.authorize({ ...sender, senderFrame: { url: win.webContents.mainFrame.url } }), false, 'A child frame cannot reuse the entry URL')
assert.equal(runtime.authorize({ ...sender, sender: {} }), false, 'The full app window cannot impersonate the quick window')
const entry = win.webContents.mainFrame.url
win.webContents.mainFrame.url += '?changed=1'
assert.equal(runtime.authorize(sender), false)
win.webContents.mainFrame.url = entry
let allowed: boolean | null = null
requestPermission(win.webContents, 'media', (value: boolean) => { allowed = value }, { isMainFrame: true, requestingUrl: entry, mediaTypes: ['audio'] })
assert.equal(allowed, true)
requestPermission(win.webContents, 'media', (value: boolean) => { allowed = value }, { isMainFrame: true, requestingUrl: entry, mediaTypes: ['video'] })
assert.equal(allowed, false)
assert.equal(checkPermission(win.webContents, 'media', '', { isMainFrame: false, mediaType: 'audio' }), false)
let prevented = false
win.emit('close', { preventDefault: () => { prevented = true } })
assert.equal(prevented, true); assert.equal(win.visible, false)
runtime.open(); assert.equal(win.visible, true)
runtime.destroy(); assert.equal(runtime.authorize(sender), false)
console.log('Quick Ask window sandbox, sender identity, media permissions, hide/reopen and display bounds checks passed')
