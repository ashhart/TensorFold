import { ipcMain } from 'electron'
import type { IpcMainInvokeEvent, WebContents, WebFrameMain } from 'electron'

// The IPC trust boundary for the Electron main process.
//
// Every renderer-invocable channel is registered through `handle()`, which
// validates the sender frame before dispatching. Only the exact application entry in the
// main frame of the BrowserWindow registered at creation time is trusted. A rogue file URL,
// child frame, stale window, or cross-origin frame
// therefore cannot reach privileged channels (workspace:exec, security:*,
// remote:*, …). Note this gates the *frame origin*, not application-layer XSS in
// the trusted renderer — channel-level role checks live alongside each handler.

interface TrustedRendererTarget {
  webContents: WebContents
  entryUrl: string
}

let trustedRenderer: TrustedRendererTarget | null = null

export function isTrustedRendererUrl(raw: string, expectedEntry: string): boolean {
  try {
    const actual = new URL(raw)
    const expected = new URL(expectedEntry)
    // Hash navigation stays inside the exact loaded document. Path/query/origin must match;
    // in particular, another file: URL is never trusted merely because of its scheme.
    actual.hash = ''
    expected.hash = ''
    return actual.href === expected.href
  } catch {
    return false
  }
}

export function configureTrustedRenderer(webContents: WebContents, entryUrl: string): void {
  if (!entryUrl || !isTrustedRendererUrl(entryUrl, entryUrl)) throw new Error('Trusted renderer entry URL is invalid.')
  trustedRenderer = { webContents, entryUrl: new URL(entryUrl).href }
}

export function clearTrustedRenderer(webContents: WebContents): void {
  if (trustedRenderer?.webContents === webContents) trustedRenderer = null
}

export function validateSender(frame: WebFrameMain | null, sender: WebContents | null = null): boolean {
  try {
    // Read `.url` inside the try: on a frame destroyed mid-invoke (e.g. HMR/window reload while a
    // call is in flight) the getter throws "Object has been destroyed" — treat that as untrusted,
    // not as an opaque error escaping the gate.
    const trusted = trustedRenderer
    if (!trusted || !frame || !sender || !frame.url) return false
    if (sender !== trusted.webContents) return false
    if (frame !== sender.mainFrame) return false
    return isTrustedRendererUrl(frame.url, trusted.entryUrl)
  } catch {
    return false
  }
}

export function assertTrustedSender(event: IpcMainInvokeEvent) {
  if (!validateSender(event.senderFrame, event.sender)) throw new Error('Untrusted renderer IPC sender.')
}

export function handle(channel: string, listener: (event: IpcMainInvokeEvent, ...args: any[]) => any) {
  ipcMain.handle(channel, async (event, ...args) => {
    assertTrustedSender(event)
    return listener(event, ...args)
  })
}
