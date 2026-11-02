import type { TensorFoldStudioApi } from '../../shared/studio-api'
export type { TensorFoldStudioApi } from '../../shared/studio-api'

// The desktop preload provides the bridge; a plain browser uses the prepared demo API.
import { preparedBrowserDemo } from './browser-demo-bootstrap'
import { createRemoteBrowserApi } from './remote-transport'
import { pairedToken } from './remote-pairing'

// Background reads and engine checks report errors inline; other failures emit a global notice.
const QUIET_IPC = /^(cluster|modelHelper|modelQualification|conversationSearch|documentLibrary|readableExport|workspaceBackup|get|list|fetch|discover|check|gitStatus|gitDiff|googleWorkspaceStatus$|engine(?:Runtime|Snapshot|Refresh|Search|Preflight))/

function reportIpcError(channel: string, e: unknown): void {
  const raw = String((e as { message?: string })?.message ?? e)
  const message = raw.replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '').slice(0, 240)
  // Keep the caller's cancellation rejection, but suppress its duplicate global error notice.
  if (channel === 'engineLoad' && message === 'Engine start was cancelled.') return
  try { window.dispatchEvent(new CustomEvent('vontra:ipc-error', { detail: message || `${channel} failed.` })) } catch {}
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

// Prefer the desktop bridge, then a paired host, then the browser demo prepared before this module loads.
function pickBrowserTransport(): TensorFoldStudioApi {
  const token = pairedToken()
  if (token) return createRemoteBrowserApi({ token })
  return preparedBrowserDemo()
}

export const api: TensorFoldStudioApi = surfaceIpcErrors(window.tensorfoldStudio ?? window.vontra ?? pickBrowserTransport())
