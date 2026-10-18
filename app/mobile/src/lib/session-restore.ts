import { validateHost, type DeviceSession } from './client'

export type SessionRestoreResult =
  | { kind: 'deferred' }
  | { kind: 'missing' }
  | { kind: 'invalid' }
  | { kind: 'restored'; session: DeviceSession }

/** A protected-data read failure is not evidence that the pairing was removed. */
export async function readSavedSession(read: () => Promise<string | null>, allowAndroidEmulator: boolean): Promise<SessionRestoreResult> {
  let raw: string | null
  try { raw = await read() } catch { return { kind: 'deferred' } }
  if (raw === null) return { kind: 'missing' }
  try {
    const value: unknown = JSON.parse(raw)
    if (!value || typeof value !== 'object' || Array.isArray(value)) return { kind: 'invalid' }
    const candidate = value as Record<string, unknown>
    if (typeof candidate.token !== 'string' || !candidate.token || typeof candidate.sessionId !== 'string' || !candidate.sessionId || typeof candidate.baseUrl !== 'string') return { kind: 'invalid' }
    if (candidate.certificatePin !== undefined && (typeof candidate.certificatePin !== 'string' || !/^[a-fA-F0-9]{64}$/.test(candidate.certificatePin))) return { kind: 'invalid' }
    return { kind: 'restored', session: {
      baseUrl: validateHost(candidate.baseUrl, allowAndroidEmulator), token: candidate.token, sessionId: candidate.sessionId,
      label: typeof candidate.label === 'string' ? candidate.label : 'This phone',
      ...(typeof candidate.certificatePin === 'string' ? { certificatePin: candidate.certificatePin } : {}),
    } }
  } catch { return { kind: 'invalid' } }
}
