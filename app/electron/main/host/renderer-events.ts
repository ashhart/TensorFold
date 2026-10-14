import { store } from '../state'

/**
 * A BrowserWindow can become unusable just before Electron emits `closed`. Main-process
 * background work must treat renderer delivery as best-effort during that interval: a dead
 * window must never abort durable work or suppress remote/webhook fan-out.
 */
function windowAvailable(win: typeof store.win): boolean {
  if (!win) return false
  try {
    return !win.isDestroyed() && !win.webContents.isDestroyed()
  } catch {
    return false
  }
}

export function rendererWindowAvailable(): boolean {
  return windowAvailable(store.win)
}

export function sendToRenderer(channel: string, payload?: unknown): boolean {
  const win = store.win
  if (!win || !windowAvailable(win)) return false
  try {
    win.webContents.send(channel, payload)
    return true
  } catch {
    // Electron may destroy the WebContents between the availability check and send().
    return false
  }
}

export interface HostEventSink {
  label: string
  deliver(): void
}

/** Deliver independent host projections without allowing one failed sink to suppress another. */
export function fanOutHostEvent(channel: string, payload: unknown, sinks: HostEventSink[]): {
  rendererDelivered: boolean
  failedSinks: string[]
} {
  const rendererDelivered = sendToRenderer(channel, payload)
  const failedSinks: string[] = []
  for (const sink of sinks) {
    try {
      sink.deliver()
    } catch {
      // Do not include payloads or exception text: either may contain user/tool data.
      failedSinks.push(sink.label)
      console.warn(`[vontra] host event delivery failed for ${sink.label}`)
    }
  }
  return { rendererDelivered, failedSinks }
}
