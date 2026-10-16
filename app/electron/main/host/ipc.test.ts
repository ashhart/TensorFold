import assert from 'node:assert/strict'
import {
  assertTrustedSender,
  clearTrustedRenderer,
  configureTrustedRenderer,
  isTrustedRendererUrl,
  validateSender,
} from './ipc'

const entry = 'file:///Applications/TensorFold-Studio.app/Contents/Resources/app.asar/dist/index.html'
const mainFrame = { url: entry } as any
const webContents = { id: 41, mainFrame } as any

configureTrustedRenderer(webContents, entry)
assert.equal(validateSender(mainFrame, webContents), true, 'registered main frame at the exact entry is trusted')
assert.equal(isTrustedRendererUrl(`${entry}#settings`, entry), true, 'in-document hash state stays on the exact entry')

mainFrame.url = 'file:///etc/passwd'
assert.equal(validateSender(mainFrame, webContents), false, 'an arbitrary file URL is never an app renderer')
mainFrame.url = entry

const wrongWindow = { id: 99, mainFrame } as any
assert.equal(validateSender(mainFrame, wrongWindow), false, 'another BrowserWindow cannot invoke privileged IPC')
assert.equal(validateSender({ url: entry } as any, webContents), false, 'a child/stale frame with the right URL is not the registered main frame')
assert.equal(isTrustedRendererUrl('http://127.0.0.1:5173/other', 'http://127.0.0.1:5173/'), false, 'dev trust is exact-path, not same-origin')
assert.equal(isTrustedRendererUrl('file:///tmp/other.html', entry), false, 'file scheme alone grants no trust')

assert.doesNotThrow(() => assertTrustedSender({ sender: webContents, senderFrame: mainFrame } as any))
assert.throws(
  () => assertTrustedSender({ sender: wrongWindow, senderFrame: mainFrame } as any),
  /Untrusted renderer IPC sender/,
)

clearTrustedRenderer(webContents)
assert.equal(validateSender(mainFrame, webContents), false, 'closed renderer registration is cleared')

console.log('ipc sender trust tests passed')
