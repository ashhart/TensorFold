export interface DeviceSession { baseUrl: string; token: string; sessionId: string; label: string; certificatePin?: string }
export interface MobileItem { id: string; kind: string; tool?: string; state?: string; status?: string; title?: string; text?: string; command?: string; path?: string; output?: string; diff?: { type: string; text: string }[]; createdAt?: number }
export interface MobileThread { incognito?: boolean; phoneLocalOnly?: boolean; id: string; title: string; kind: 'task' | 'chat'; projectId?: string | null; providerId: string | null; model: string | null; status: string; updatedAt: number; timeline: MobileItem[]; goal?: string; lastRunReceipt?: { verdict: string; summary: string; changedFiles?: string[] } | null }
export interface MobileSnapshot {
  serverName: string; appVersion: string; generatedAt: number; activeThreadIds: string[];
  threads: MobileThread[]; projects: { id: string; name: string }[];
  providers?: { id: string; name: string; kind: string; models: string[] }[];
  work: { approvals: { pending: number }; jobs: { running: number; queued: number } }
}
export interface DeviceStatus {
  scope: 'paired' | 'owner'; role: 'viewer' | 'operator' | 'owner';
  capabilities: { taskStart: boolean; taskGuide: boolean; taskStop: boolean; approvalResolve: boolean; transcriptReview: boolean; changesReview: boolean; engineControl: boolean };
  engine: { available: boolean; state: string; model: string | null; drafter: string | null; contextTokens: number | null; totalMemoryBytes: number; memoryLimitBytes: number; hardware: { platform: string; architecture: string; chip: string | null } | null; performance: { model: string; measuredAt: number; decodeTokensPerSecond: number | null } | null } | null;
}
export const RESTRICTED_CAPABILITIES: DeviceStatus['capabilities'] = { taskStart: false, taskGuide: false, taskStop: false, approvalResolve: false, transcriptReview: false, changesReview: false, engineControl: false }
export interface MobileHost { snapshot: MobileSnapshot; status: DeviceStatus | null }

export class HostError extends Error {
  constructor(message: string, readonly status = 0, readonly code = 'unavailable') { super(message); this.name = 'HostError' }
}

export function validateHost(value: string, allowAndroidEmulator = false): string {
  let url: URL
  try { url = new URL(value.trim()) } catch { throw new HostError('Enter the host URL shown by TensorFold, including https://.') }
  if (url.username || url.password || url.search || url.hash) throw new HostError('Use a host URL without credentials, a query, or a fragment.')
  const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname) || allowAndroidEmulator && url.hostname === '10.0.2.2'
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) throw new HostError('Use your private HTTPS host connection; plain HTTP is allowed only for local simulator testing.')
  if (!url.hostname || !/^\/[a-zA-Z0-9._~!$&'()*+,;=:@%/-]*$/.test(url.pathname)) throw new HostError('This host URL is not supported.')
  return url.href.replace(/\/+$/, '')
}

/** Accept the existing #pair= QR/link without keeping a credential in the saved host URL. */
export function parsePairingInput(value: string, host = '', allowAndroidEmulator = false): { baseUrl: string; credential: string; certificatePin?: string } {
  const text = value.trim()
  if (!/^https?:\/\//i.test(text)) {
    if (!text || text.length > 512) throw new HostError('Paste the one-time pairing code from TensorFold.')
    let url: URL
    try { url = new URL(host.trim()) } catch { throw new HostError('Enter the private host URL shown by TensorFold.') }
    return { ...pairingHost(url, allowAndroidEmulator), credential: text }
  }
  let url: URL
  try { url = new URL(text) } catch { throw new HostError('This pairing link is invalid.') }
  const credential = new URLSearchParams(url.hash.replace(/^#/, '')).get('pair') || url.searchParams.get('pair')
  if (!credential || credential.length > 512) throw new HostError('This link contains no valid one-time pairing code.')
  return { ...pairingHost(url, allowAndroidEmulator), credential }
}

function pairingHost(url: URL, allowAndroidEmulator: boolean): { baseUrl: string; certificatePin?: string } {
  const fields = new URLSearchParams(url.hash.replace(/^#/, ''))
  const pin = fields.get('pin') || fields.get('cert')
  if (fields.get('pin') && fields.get('cert') && fields.get('pin')!.toLowerCase() !== fields.get('cert')!.toLowerCase()) throw new HostError('This pairing link contains conflicting certificate pins.')
  if (pin && (!/^[a-fA-F0-9]{64}$/.test(pin) || url.protocol !== 'https:')) throw new HostError('This pairing link contains an invalid HTTPS certificate pin.')
  if (fields.has('v') && fields.get('v') !== '1') throw new HostError('This pairing link version is not supported.')
  if (fields.has('expires') && (!Number.isFinite(Number(fields.get('expires'))) || Number(fields.get('expires')) < Date.now())) throw new HostError('This pairing link expired; create a new link on your Mac.')
  url.searchParams.delete('pair'); url.hash = ''
  return { baseUrl: validateHost(url.href, allowAndroidEmulator), ...(pin ? { certificatePin: pin.toLowerCase() } : {}) }
}

async function responseJson(response: Response): Promise<unknown> {
  const mime = response.headers.get('content-type') ?? ''
  const declared = Number(response.headers.get('content-length') ?? 0)
  const max = 8 * 1024 * 1024
  if (!/^application\/(?:json|[a-z0-9.+-]+\+json)\b/i.test(mime) || declared > max) throw new HostError('The host returned an unexpected response.')
  if (response.body?.getReader) {
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let text = ''; let bytes = 0
    try {
      while (true) {
        const result = await reader.read(); if (result.done) break
        bytes += result.value.byteLength
        if (bytes > max) { await reader.cancel(); throw new HostError('The host response is too large.') }
        text += decoder.decode(result.value, { stream: true })
      }
      return JSON.parse(text + decoder.decode())
    } catch (error) { if (error instanceof HostError) throw error; throw new HostError('The host response could not be read.') }
    finally { reader.releaseLock() }
  }
  const text = await response.text()
  if (text.length > max) throw new HostError('The host response is too large.')
  try { return JSON.parse(text) } catch { throw new HostError('The host response could not be read.') }
}

export class StudioClient {
  constructor(readonly session: DeviceSession, private fetchImpl: typeof fetch = fetch) {}
  private async request<T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
    const endpoint = `${this.session.baseUrl}/${path.replace(/^\//, '')}`
    const controller = new AbortController()
    const abort = () => controller.abort()
    signal?.addEventListener('abort', abort, { once: true })
    if (signal?.aborted) controller.abort()
    const timeout = setTimeout(abort, 15_000)
    try {
      const response = await this.fetchImpl(endpoint, {
        method: body === undefined ? 'GET' : 'POST', headers: { accept: 'application/json', Authorization: `Bearer ${this.session.token}`, ...(body === undefined ? {} : { 'content-type': 'application/json' }) },
        ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: controller.signal, credentials: 'omit', redirect: 'error', cache: 'no-store',
      })
      if (response.url && new URL(response.url).origin !== new URL(endpoint).origin) throw new HostError('The host tried to redirect the device connection.')
      const result = await responseJson(response) as { error?: string }
      if (!response.ok) throw new HostError(response.status === 401 ? 'This device pairing was revoked or expired; pair again in TensorFold.' : response.status === 403 ? 'The host does not permit this action for your session.' : response.status === 409 ? 'The task is busy; refresh before trying again.' : 'The host could not complete this action.', response.status, result?.error ?? 'host_error')
      return result as T
    } catch (error) {
      if (error instanceof HostError) throw error
      throw new HostError(controller.signal.aborted ? 'The host did not respond in time.' : 'Could not reach TensorFold; keep your Mac awake and check the private host connection.')
    } finally { clearTimeout(timeout); signal?.removeEventListener('abort', abort) }
  }
  static async pair(baseUrl: string, credential: string, clientInfo: string, fetchImpl: typeof fetch = fetch, certificatePin?: string): Promise<DeviceSession> {
    const clean = validateHost(baseUrl, true)
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 15_000)
    try {
      const response = await fetchImpl(`${clean}/v1/auth/pair`, { method: 'POST', headers: { accept: 'application/json', 'content-type': 'application/json' }, body: JSON.stringify({ credential, clientInfo: clientInfo.slice(0, 120) }), redirect: 'error', credentials: 'omit', signal: controller.signal })
      const result = await responseJson(response) as { token?: string; sessionId?: string; label?: string; error?: string }
      if (!response.ok || !result.token || !result.sessionId) throw new HostError(result.error === 'already_consumed' ? 'That code was already used; create a new pairing link on your Mac.' : result.error === 'expired' ? 'That pairing code expired; create a new link on your Mac.' : response.status === 429 ? 'Too many pairing attempts; wait before trying again.' : 'Pairing failed; check the host URL and create a fresh link in TensorFold.', response.status, result.error)
      return { baseUrl: clean, token: result.token, sessionId: result.sessionId, label: result.label || 'This phone', ...(certificatePin ? { certificatePin } : {}) }
    } catch (error) { if (error instanceof HostError) throw error; throw new HostError('Could not reach TensorFold; check the private host connection and try again.') }
    finally { clearTimeout(timeout) }
  }
  async host(signal?: AbortSignal): Promise<MobileHost> {
    const snapshot = await this.request<MobileSnapshot>('v1/snapshot', undefined, signal)
    if (!snapshot || !Array.isArray(snapshot.threads) || !Array.isArray(snapshot.activeThreadIds) || typeof snapshot.serverName !== 'string') throw new HostError('This host returned an unsupported workspace snapshot.')
    let status: DeviceStatus | null = null
    try { status = await this.request<DeviceStatus>('v1/device-status', undefined, signal) } catch (error) { if (!(error instanceof HostError) || ![404, 405, 501].includes(error.status)) throw error }
    return { snapshot, status }
  }
  async send(threadId: string, userText: string, active: boolean, capabilities: DeviceStatus['capabilities']): Promise<void> {
    if (active ? !capabilities.taskGuide : !capabilities.taskStart) throw new HostError('Sending is disabled by this host session.', 403)
    const text = userText.trim(); if (!text || text.length > 8_000) throw new HostError('Enter a message of up to 8,000 characters.')
    await this.request(`v1/threads/${encodeURIComponent(threadId)}/${active ? 'guide' : 'start'}`, { userText: text })
  }
  async watchSpeechStatus(): Promise<{available: boolean; provider: 'kokoro'; reason: string | null}> { return this.request('v1/watch/speech/status') }
  async startWatchSpeech(threadId: string, itemId: string): Promise<{speechId: string; provider: 'kokoro'; sampleRate: number}> {
    return this.request('v1/watch/speech/start', {threadId, itemId})
  }
  async readWatchSpeech(speechId: string, sequence: number): Promise<{speechId: string; sequence: number; state: 'pending' | 'audio' | 'done' | 'error'; audioBase64?: string; sampleRate?: number; error?: string}> {
    if (!/^[a-zA-Z0-9-]{8,80}$/.test(speechId) || !Number.isInteger(sequence) || sequence < 0 || sequence > 10_000) throw new HostError('Invalid speech request.')
    return this.request(`v1/watch/speech/${speechId}/chunks/${sequence}`)
  }
  async cancelWatchSpeech(speechId: string): Promise<{cancelled: boolean}> {
    if (!/^[a-zA-Z0-9-]{8,80}$/.test(speechId)) throw new HostError('Invalid speech request.')
    return this.request(`v1/watch/speech/${speechId}/cancel`, {})
  }
  async stop(threadId: string, capabilities: DeviceStatus['capabilities']): Promise<void> {
    if (!capabilities.taskStop) throw new HostError('Stopping is disabled by this host session.', 403)
    await this.request(`v1/threads/${encodeURIComponent(threadId)}/cancel`, {})
  }
  async approve(callId: string, decision: 'allow' | 'deny', capabilities: DeviceStatus['capabilities']): Promise<void> {
    if (!capabilities.approvalResolve || !capabilities.transcriptReview) throw new HostError('Review and approve this request on the desktop.', 403)
    await this.request(`v1/approvals/${encodeURIComponent(callId)}`, { decision })
  }
}

export function mobileThreadState(thread: MobileThread, activeIds: readonly string[]): 'running' | 'waiting' | 'failed' | 'finished' | 'idle' {
  if (activeIds.includes(thread.id) && thread.timeline.some((item) => item.state === 'awaiting' || item.state === 'pending' && item.kind === 'approval')) return 'waiting'
  if (activeIds.includes(thread.id)) return 'running'
  if (thread.status === 'error') return 'failed'
  if (thread.timeline.some((item) => item.kind === 'assistant')) return 'finished'
  return 'idle'
}
