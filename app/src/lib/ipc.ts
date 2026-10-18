import type { TensorFoldStudioApi } from '../../shared/studio-api'
export type { TensorFoldStudioApi } from '../../shared/studio-api'

// In the desktop app the preload bridge provides `window.tensorfoldStudio`.
// In a plain browser (Vite dev URL) we fall back to a demo implementation.
import { preparedBrowserDemo } from './browser-demo-bootstrap'
import { createRemoteBrowserApi } from './remote-transport'
import { pairedToken } from './remote-pairing'

// Background reads/polls fail quietly (they drive UI state, not user actions).
// Every other method is a user-initiated action — surface its failure so nothing
// fails silently. App.tsx listens for `vontra:ipc-error` and shows a notice toast.
// Engine reads and Hub checks report their errors inline on the Models page.
const QUIET_IPC = /^(cluster|modelHelper|modelQualification|conversationSearch|documentLibrary|readableExport|workspaceBackup|get|list|fetch|discover|check|gitStatus|gitDiff|engine(?:Snapshot|Refresh|Search|Preflight))/

function reportIpcError(channel: string, e: unknown): void {
  const raw = String((e as { message?: string })?.message ?? e)
  const message = raw.replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '').slice(0, 240)
  try { window.dispatchEvent(new CustomEvent('vontra:ipc-error', { detail: message || `${channel} failed.` })) } catch { /* no window (tests) */ }
}

function surfaceIpcErrors(raw: TensorFoldStudioApi): TensorFoldStudioApi {
  const out: Record<string, unknown> = {}
  for (const key of Object.keys(raw) as string[]) {
    const value = (raw as unknown as Record<string, unknown>)[key]
    if (typeof value !== 'function' || QUIET_IPC.test(key) || key === 'sshStatus') {
      out[key] = value
      continue
    }
    out[key] = (...args: unknown[]) => {
      let result: unknown
      try {
        result = (value as (...a: unknown[]) => unknown).apply(raw, args)
      } catch (e) {
        reportIpcError(key, e)
        throw e
      }
      if (result && typeof (result as Promise<unknown>).then === 'function') {
        return (result as Promise<unknown>).catch((e: unknown) => { reportIpcError(key, e); throw e })
      }
      return result
    }
  }
  return out as unknown as TensorFoldStudioApi
}

// Transport pick (VST-035), in order: the Electron preload bridge; a paired /v1 session
// against the host that served this page (token established pre-boot in main.tsx); the
// browser demo mock. The paired branch reads the token synchronously — main.tsx guarantees
// it exists before this module loads.
function pickBrowserTransport(): TensorFoldStudioApi {
  const token = pairedToken()
  if (token) return createRemoteBrowserApi({ token })
  return preparedBrowserDemo()
}

export const api: TensorFoldStudioApi = surfaceIpcErrors(window.tensorfoldStudio ?? window.vontra ?? pickBrowserTransport())
