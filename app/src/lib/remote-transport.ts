// Browser transport over the /v1 remote protocol (VST-035): the SAME renderer, loaded in a
// browser or phone from the host's loopback/LAN server, drives THAT host instead of falling
// back to the demo mock. Local becomes a special case of remote.
//
// Scope, stated honestly (per the working rules): the core work surfaces ride the wire —
// threads/projects/rooms/tickets/jobs/automations from the host snapshot, live updates over
// the /v1/events SSE (agent deltas, thread/room/ticket/job pushes), run start/guide/cancel,
// approvals, and creation flows. Host-only surfaces that would otherwise render the demo
// mock's fake data as "healthy" are explicitly empty or fail closed below, so nothing broken
// or unknown can look green from a paired device. Settings are synthesized read-only:
// providers arrive REDACTED from the host (identity + models, never credentials) and
// saveSettings refuses. No unimplemented API method inherits mock behavior; the UI carries
// a persistent paired-session banner (platform 'web-remote').

import { createMockApi } from './mock'
import type { TensorFoldStudioApi } from './ipc'
import type { AgentEvent, ArchiveSnapshot, Attachment, Automation, CliDefinition, CliSession, CueRunPayload, CueRunResult, Job, Project, Provider, Room, Settings, StartRunPayload, Thread, Ticket } from './types'

export interface RemoteSseEvent {
  id: string
  kind: string
  createdAt: number
  data: unknown
}

export interface EventStreamLike {
  onEvent(cb: (event: RemoteSseEvent) => void): void
  close(): void
}

export interface RemoteTransportConfig {
  // '' = same origin (the normal case: the page was served by the host it drives).
  baseUrl?: string
  token: string
  fetchImpl?: typeof fetch
  // Injectable SSE source for tests; defaults to an authenticated fetch stream so the
  // bearer credential never appears in a URL/query string.
  openEventStream?: (url: string, token: string, onEvent: (event: RemoteSseEvent) => void) => EventStreamLike
}

export class PairedSessionError extends Error {
  constructor(what: string) {
    super(`${what} isn't available from a paired session — do this on the host.`)
    this.name = 'PairedSessionError'
  }
}

interface RemoteState {
  threads: Thread[]
  projects: Project[]
  rooms: Room[]
  tickets: Ticket[]
  jobs: Job[]
  automations: Automation[]
  providers: Provider[]
  serverName: string
  appVersion: string
  generatedAt: number
}

const REMOTE_SSE_FRAME_MAX_CHARS = 512 * 1024
export const REMOTE_SSE_CONNECT_TIMEOUT_MS = 15_000
export const REMOTE_SSE_IDLE_TIMEOUT_MS = 45_000
const REMOTE_JSON_MAX_BYTES = 16 * 1024 * 1024
const REMOTE_ERROR_MAX_BYTES = 64 * 1024

export interface AuthenticatedEventStreamOptions {
  connectTimeoutMs?: number
  idleTimeoutMs?: number
  reconnectDelayMs?: number
  onError?: (error: Error) => void
}

function pairedAuthFailed(): void {
  try { window.dispatchEvent(new CustomEvent('vontra:paired-auth-failed')) } catch { /* tests */ }
}

function pairedStreamFailed(error: Error): void {
  try { window.dispatchEvent(new CustomEvent('vontra:paired-stream-error', { detail: error.message })) } catch { /* tests */ }
}

function browserStreamAbortError(): Error {
  const error = new Error('Host event stream was aborted.')
  error.name = 'AbortError'
  return error
}

function waitForBrowserStream<T>(
  pending: Promise<T>,
  controller: AbortController,
  timeoutMs: number,
  timeoutMessage: string,
): Promise<T> {
  const bounded = Math.max(1, timeoutMs)
  if (controller.signal.aborted) return Promise.reject(browserStreamAbortError())
  return new Promise((resolve, reject) => {
    let settled = false
    const finish = (fn: () => void) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      controller.signal.removeEventListener('abort', onAbort)
      fn()
    }
    const onAbort = () => finish(() => reject(browserStreamAbortError()))
    const timer = setTimeout(() => {
      finish(() => reject(new Error(timeoutMessage)))
      controller.abort()
    }, bounded)
    controller.signal.addEventListener('abort', onAbort, { once: true })
    pending.then(
      (value) => finish(() => resolve(value)),
      (error) => finish(() => reject(error)),
    )
  })
}

async function readBoundedRemoteJson<T>(response: Response, maxBytes = REMOTE_JSON_MAX_BYTES): Promise<T> {
  // Existing unit-test doubles expose only json(). Real browser Response objects take the
  // MIME-checked streaming path and can never allocate an attacker-sized body at once.
  if (!response.headers || !response.body) return await response.json() as T
  const contentType = String(response.headers.get('content-type') ?? '').toLowerCase()
  if (!/^(?:application\/json|application\/[a-z0-9.+-]+\+json)\b/.test(contentType)) {
    await response.body.cancel().catch(() => {})
    throw new Error('Host returned a non-JSON response.')
  }
  const declared = response.headers.get('content-length')
  if (declared) {
    const bytes = Number(declared)
    if (!Number.isSafeInteger(bytes) || bytes < 0 || bytes > maxBytes) {
      await response.body.cancel().catch(() => {})
      throw new Error('Host response exceeded the safety limit.')
    }
  }
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let text = ''
  let bytes = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      bytes += value.byteLength
      if (bytes > maxBytes) {
        await reader.cancel().catch(() => {})
        throw new Error('Host response exceeded the safety limit.')
      }
      text += decoder.decode(value, { stream: true })
    }
    text += decoder.decode()
  } finally {
    reader.releaseLock()
  }
  return JSON.parse(text) as T
}

/**
 * Fetch-backed SSE client. Native EventSource cannot attach an Authorization header and
 * would force a long-lived device credential into the URL, where browser history, proxy
 * access logs, crash reports, and copied links can expose it. This parser preserves SSE
 * replay through Last-Event-ID while keeping authentication header-only.
 */
export function openAuthenticatedEventStream(
  url: string,
  token: string,
  onEvent: (event: RemoteSseEvent) => void,
  fetchImpl: typeof fetch = fetch,
  options: AuthenticatedEventStreamOptions = {},
): EventStreamLike {
  let closed = false
  let controller: AbortController | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null
  let lastEventId = ''
  const connectTimeoutMs = Math.max(1, options.connectTimeoutMs ?? REMOTE_SSE_CONNECT_TIMEOUT_MS)
  const idleTimeoutMs = Math.max(1, options.idleTimeoutMs ?? REMOTE_SSE_IDLE_TIMEOUT_MS)
  const reconnectDelayMs = Math.max(0, options.reconnectDelayMs ?? 1_000)

  const reportError = (error: unknown) => {
    const reported = error instanceof Error ? error : new Error(String(error))
    options.onError?.(reported)
    pairedStreamFailed(reported)
  }

  const scheduleReconnect = () => {
    if (closed || reconnectTimer) return
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      void connect()
    }, reconnectDelayMs)
  }

  const dispatchFrame = (frame: string) => {
    let kind = 'message'
    let id = ''
    const data: string[] = []
    for (const rawLine of frame.split(/\r?\n/)) {
      if (!rawLine || rawLine.startsWith(':')) continue
      const split = rawLine.indexOf(':')
      const field = split < 0 ? rawLine : rawLine.slice(0, split)
      const value = split < 0 ? '' : rawLine.slice(split + 1).replace(/^ /, '')
      if (field === 'event') kind = value
      else if (field === 'id' && !value.includes('\0')) id = value
      else if (field === 'data') data.push(value)
    }
    if (!data.length) return
    try {
      const parsed = JSON.parse(data.join('\n')) as Record<string, unknown>
      const eventId = String(parsed.id ?? id)
      if (eventId) lastEventId = eventId
      onEvent({
        id: eventId,
        kind: String(parsed.kind ?? kind),
        createdAt: Number(parsed.createdAt ?? Date.now()),
        data: parsed.data,
      })
    } catch {
      // A malformed untrusted frame is ignored; a valid later frame still advances.
    }
  }

  const connect = async () => {
    if (closed) return
    const requestController = new AbortController()
    controller = requestController
    try {
      const response = await waitForBrowserStream(fetchImpl(url, {
        method: 'GET',
        headers: {
          accept: 'text/event-stream',
          'x-vontra-token': token,
          ...(lastEventId ? { 'last-event-id': lastEventId } : {}),
        },
        cache: 'no-store',
        credentials: 'omit',
        redirect: 'error',
        signal: requestController.signal,
      }), requestController, connectTimeoutMs, `Host event stream did not connect within ${connectTimeoutMs}ms.`)
      if (response.status === 401) {
        await response.body?.cancel().catch(() => {})
        pairedAuthFailed()
        closed = true
        return
      }
      if (!response.ok) {
        await response.body?.cancel().catch(() => {})
        throw new Error(`Host event stream failed (${response.status}).`)
      }
      const contentType = String(response.headers.get('content-type') ?? '').toLowerCase()
      if (!contentType.startsWith('text/event-stream')) {
        await response.body?.cancel().catch(() => {})
        throw new Error('Host event stream returned the wrong content type.')
      }
      if (!response.body) throw new Error('Host event stream returned no body.')

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let complete = false
      try {
        while (!closed) {
          const { done, value } = await waitForBrowserStream(
            reader.read(),
            requestController,
            idleTimeoutMs,
            `Host event stream was idle for ${idleTimeoutMs}ms; reconnecting.`,
          )
          if (done) { complete = true; break }
          buffer += decoder.decode(value, { stream: true })
          if (buffer.length > REMOTE_SSE_FRAME_MAX_CHARS && !/\r?\n\r?\n/.test(buffer)) {
            await reader.cancel().catch(() => {})
            throw new Error('Host event stream frame exceeded the safety limit.')
          }
          while (true) {
            const boundary = buffer.search(/\r?\n\r?\n/)
            if (boundary < 0) break
            const match = buffer.slice(boundary).match(/^\r?\n\r?\n/)
            const frame = buffer.slice(0, boundary)
            buffer = buffer.slice(boundary + (match?.[0].length ?? 2))
            if (frame.length > REMOTE_SSE_FRAME_MAX_CHARS) throw new Error('Host event stream frame exceeded the safety limit.')
            dispatchFrame(frame)
          }
        }
      } finally {
        if (!complete) await reader.cancel().catch(() => {})
        reader.releaseLock()
      }
    } catch (error) {
      if (!closed && !(error instanceof Error && error.name === 'AbortError')) {
        reportError(error)
        scheduleReconnect()
      }
      return
    } finally {
      if (controller === requestController) controller = null
    }
    if (!closed) reportError(new Error('Host event stream closed; reconnecting.'))
    scheduleReconnect()
  }

  void connect()
  return {
    onEvent: () => {},
    close: () => {
      closed = true
      controller?.abort()
      if (reconnectTimer) clearTimeout(reconnectTimer)
      reconnectTimer = null
    },
  }
}

// Redacted host providers → renderer Provider shapes (blank credentials, never guessed).
function providersFromSnapshot(raw: unknown): Provider[] {
  if (!Array.isArray(raw)) return []
  return raw
    .filter((p): p is Record<string, unknown> => !!p && typeof p === 'object' && typeof (p as { id?: unknown }).id === 'string')
    .map((p) => ({
      id: String(p.id),
      name: String(p.name ?? p.id),
      kind: (p.kind === 'local' ? 'local' : 'cloud') as Provider['kind'],
      transport: typeof p.transport === 'string' ? p.transport : undefined,
      baseUrl: '',
      apiKey: '',
      models: Array.isArray(p.models) ? p.models.map(String) : [],
      ...(typeof p.defaultModel === 'string' && p.defaultModel ? { defaultModel: p.defaultModel } : {}),
    }) as unknown as Provider)
}

export function createRemoteBrowserApi(config: RemoteTransportConfig): TensorFoldStudioApi {
  const base = createMockApi()
  const fetchImpl = config.fetchImpl ?? ((...args: Parameters<typeof fetch>) => fetch(...args))
  const origin = (config.baseUrl ?? '').replace(/\/$/, '')
  const openEventStream = config.openEventStream
    ?? ((url, token, onEvent) => openAuthenticatedEventStream(url, token, onEvent, fetchImpl))

  const state: RemoteState = { threads: [], projects: [], rooms: [], tickets: [], jobs: [], automations: [], providers: [], serverName: 'TensorFold host', appVersion: '', generatedAt: 0 }
  const listeners = {
    agent: new Set<(e: AgentEvent) => void>(),
    thread: new Set<(t: Thread) => void>(),
    project: new Set<(p: Project) => void>(),
    room: new Set<(r: Room) => void>(),
    ticket: new Set<(t: Ticket) => void>(),
    job: new Set<(j: Job) => void>(),
    automation: new Set<(a: Automation) => void>(),
    terminalData: new Set<(p: { sessionId: string; chunk: string; orchestratorHostId?: string | null }) => void>(),
    terminalStatus: new Set<(p: { sessionId: string; status: CliSession['status']; orchestratorHostId?: string | null }) => void>(),
    terminalSessions: new Set<(p: { projectId: string; sessions: CliSession[]; orchestratorHostId?: string | null }) => void>(),
  }

  async function req<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetchImpl(`${origin}${path}`, {
      ...init,
      headers: {
        'content-type': 'application/json',
        'x-vontra-token': config.token,
        ...(init?.headers ?? {}),
      },
      cache: 'no-store',
      credentials: 'omit',
      redirect: 'error',
      signal: init?.signal ?? (typeof AbortSignal.timeout === 'function' ? AbortSignal.timeout(30_000) : undefined),
    })
    if (res.status === 401) {
      await res.body?.cancel().catch(() => {})
      pairedAuthFailed()
      throw new Error('This paired session is no longer authorized — pair again from the host.')
    }
    if (!res.ok) {
      let detail = ''
      try { detail = String((await readBoundedRemoteJson<{ error?: unknown }>(res, REMOTE_ERROR_MAX_BYTES))?.error ?? '') } catch { /* opaque */ }
      throw new Error(`Host request failed (${res.status})${detail ? `: ${detail}` : ''}`)
    }
    return await readBoundedRemoteJson<T>(res)
  }

  // Live pushes that land BEFORE the first snapshot fetch resolves must survive it. They
  // are delivered to subscribers immediately, buffered for the cache, then replayed over
  // the baseline without a duplicate notification.
  let snapshotApplied = false
  let rebaseliningAfterGap = false
  type RecordKind = 'thread' | 'project' | 'room' | 'ticket' | 'job' | 'automation'
  type RemoteRecord = Thread | Project | Room | Ticket | Job | Automation
  const earlyPushes: Array<{ kind: RecordKind; item: RemoteRecord }> = []
  const earlyAgentEvents: Array<{ event: AgentEvent; createdAt: number }> = []

  function listFor(kind: RecordKind): RemoteRecord[] {
    if (kind === 'thread') return state.threads
    if (kind === 'project') return state.projects
    if (kind === 'room') return state.rooms
    if (kind === 'ticket') return state.tickets
    if (kind === 'job') return state.jobs
    return state.automations
  }

  function setList(kind: RecordKind, list: RemoteRecord[]): void {
    if (kind === 'thread') state.threads = list as Thread[]
    else if (kind === 'project') state.projects = list as Project[]
    else if (kind === 'room') state.rooms = list as Room[]
    else if (kind === 'ticket') state.tickets = list as Ticket[]
    else if (kind === 'job') state.jobs = list as Job[]
    else state.automations = list as Automation[]
  }

  function isTombstone(item: RemoteRecord): boolean {
    return !!(item as { archivedAt?: number | null }).archivedAt
  }

  function applyPushToCache(kind: RecordKind, item: RemoteRecord): void {
    setList(kind, isTombstone(item) ? listFor(kind).filter((entry) => entry.id !== item.id) : upsert(listFor(kind), item))
  }

  function notifyRecord(kind: RecordKind, item: RemoteRecord): void {
    const set = listeners[kind] as Set<(value: RemoteRecord) => void>
    for (const cb of set) cb(item)
  }

  function recordChanged(previous: RemoteRecord | undefined, next: RemoteRecord): boolean {
    if (!previous) return true
    if ((previous.updatedAt ?? 0) !== (next.updatedAt ?? 0)) return true
    // Some records (notably thread status) can change without a persisted timestamp. The
    // snapshots are bounded/JSON-safe, so this exact fallback keeps reconnect projection honest.
    try { return JSON.stringify(previous) !== JSON.stringify(next) } catch { return true }
  }

  function projectSnapshotList(kind: RecordKind, previous: RemoteRecord[], next: RemoteRecord[], tombstoneAt: number): void {
    const before = new Map(previous.map((item) => [item.id, item]))
    const after = new Set(next.map((item) => item.id))
    for (const item of next) if (recordChanged(before.get(item.id), item)) notifyRecord(kind, item)
    for (const item of previous) {
      if (!after.has(item.id)) notifyRecord(kind, { ...item, archivedAt: tombstoneAt, updatedAt: Math.max(item.updatedAt ?? 0, tombstoneAt) } as RemoteRecord)
    }
  }

  function applySnapshot(data: Record<string, unknown>): boolean {
    const generatedAt = Number(data.generatedAt ?? 0)
    // A GET and the SSE baseline race during boot. Never let the older response roll the
    // cache/projection backwards after a newer baseline has already landed.
    if (snapshotApplied && generatedAt > 0 && state.generatedAt > generatedAt) return false

    const previous = {
      thread: state.threads as RemoteRecord[], project: state.projects as RemoteRecord[],
      room: state.rooms as RemoteRecord[], ticket: state.tickets as RemoteRecord[],
      job: state.jobs as RemoteRecord[], automation: state.automations as RemoteRecord[],
    }
    const next = {
      thread: (Array.isArray(data.threads) ? data.threads : []) as RemoteRecord[],
      project: (Array.isArray(data.projects) ? data.projects : []) as RemoteRecord[],
      room: (Array.isArray(data.rooms) ? data.rooms : []) as RemoteRecord[],
      ticket: (Array.isArray(data.tickets) ? data.tickets : []) as RemoteRecord[],
      job: (Array.isArray(data.jobs) ? data.jobs : []) as RemoteRecord[],
      automation: (Array.isArray(data.automations) ? data.automations : []) as RemoteRecord[],
    }

    for (const kind of Object.keys(next) as RecordKind[]) setList(kind, next[kind])
    state.providers = providersFromSnapshot(data.providers)
    if (typeof data.serverName === 'string' && data.serverName) state.serverName = data.serverName
    if (typeof data.appVersion === 'string') state.appVersion = data.appVersion
    state.generatedAt = Math.max(state.generatedAt, generatedAt)

    const tombstoneAt = generatedAt || Date.now()
    for (const kind of Object.keys(next) as RecordKind[]) projectSnapshotList(kind, previous[kind], next[kind], tombstoneAt)

    if (!snapshotApplied) {
      snapshotApplied = true
      for (const push of earlyPushes.splice(0, earlyPushes.length)) applyPushToCache(push.kind, push.item)
      for (const queued of earlyAgentEvents.splice(0, earlyAgentEvents.length)) {
        // The snapshot already contains events created at or before its baseline time.
        if (queued.createdAt && generatedAt && queued.createdAt <= generatedAt) continue
        for (const cb of listeners.agent) cb(queued.event)
      }
      rebaseliningAfterGap = false
    }
    return true
  }

  function applyPush(kind: RecordKind, item: RemoteRecord): void {
    if (!snapshotApplied) {
      // Bounded boot/gap buffer: newer records for the same id supersede older ones, so
      // dropping the oldest entries on overflow is lossy only for tombstone ordering.
      if (earlyPushes.length >= 5000) earlyPushes.splice(0, earlyPushes.length - 5000 + 1)
      earlyPushes.push({ kind, item })
    }
    else applyPushToCache(kind, item)
  }

  let snapshotOnce: Promise<void> | null = null
  function ensureSnapshot(): Promise<void> {
    if (snapshotApplied) return Promise.resolve()
    snapshotOnce ??= req<Record<string, unknown>>('/v1/snapshot')
      .then((data) => { applySnapshot(data) })
      .catch((error) => {
        // A transient host/network failure must not poison every later read with the same
        // memoized rejected promise. Keep concurrent callers deduplicated for this attempt,
        // then let the next caller fetch a fresh snapshot.
        snapshotOnce = null
        throw error
      })
    return snapshotOnce
  }

  function upsert<T extends { id: string }>(list: T[], item: T): T[] {
    const at = list.findIndex((x) => x.id === item.id)
    if (at < 0) return [item, ...list]
    const next = [...list]
    next[at] = item
    return next
  }

  // One SSE stream for the whole session: snapshot baseline + typed live pushes.
  let sseBaselines = 0
  openEventStream(`${origin}/v1/events`, config.token, (event) => {
    if (event.kind === 'stream_gap') {
      // A slow client lost streamed delta chunks. Keep the encrypted host state canonical:
      // fetch a fresh projection, buffer later pushes while it is in flight, and also ask
      // the durable spine to explain the missed control-plane history.
      snapshotApplied = false
      rebaseliningAfterGap = true
      snapshotOnce = null
      earlyPushes.length = 0
      earlyAgentEvents.length = 0
      try { window.dispatchEvent(new CustomEvent('vontra:remote-stream-gap', { detail: event.data })) } catch { /* tests */ }
      try { window.dispatchEvent(new CustomEvent('vontra:spine-rebaseline')) } catch { /* tests */ }
      // A single failed snapshot fetch must not leave the client in permanent
      // rebaseline limbo: without a retry, every later agent event is buffered
      // undelivered and pushes pile onto earlyPushes until a view happens to call
      // a snapshot-backed read. Retry with capped backoff until the gap resolves.
      const scheduleGapSnapshotRetry = (attempt: number): void => {
        window.setTimeout(() => {
          if (snapshotApplied || !rebaseliningAfterGap) return
          void ensureSnapshot().catch(() => scheduleGapSnapshotRetry(attempt + 1))
        }, Math.min(30_000, 500 * 2 ** Math.min(attempt, 6)))
      }
      void ensureSnapshot().catch(() => scheduleGapSnapshotRetry(0))
      return
    }
    if (event.kind === 'snapshot' && event.data && typeof event.data === 'object') {
      applySnapshot(event.data as Record<string, unknown>)
      // A snapshot AFTER the first one means a reconnect gap outgrew the live ring — the
      // host re-baselined us. Signal the spine projector to gap-fill the missed history
      // from its durable cursor (VST-041 walk-away reconnect).
      if (++sseBaselines > 1) {
        try { window.dispatchEvent(new CustomEvent('vontra:spine-rebaseline')) } catch { /* tests */ }
      }
      return
    }
    if (event.kind === 'agent') {
      if (!snapshotApplied && rebaseliningAfterGap) {
        // Bounded: a wedged rebaseline must not accumulate deltas for a whole session.
        // Oldest entries are dropped — the retried snapshot supersedes their state.
        if (earlyAgentEvents.length >= 5000) earlyAgentEvents.splice(0, earlyAgentEvents.length - 5000 + 1)
        earlyAgentEvents.push({ event: event.data as AgentEvent, createdAt: event.createdAt })
      }
      else for (const cb of listeners.agent) cb(event.data as AgentEvent)
      return
    }
    if (event.kind === 'terminal' && event.data && typeof event.data === 'object') {
      const payload = event.data as { type?: unknown; sessionId?: unknown; chunk?: unknown; status?: unknown; projectId?: unknown; sessions?: unknown }
      if (payload.type === 'data' && typeof payload.sessionId === 'string' && typeof payload.chunk === 'string') {
        for (const cb of listeners.terminalData) cb({ sessionId: payload.sessionId, chunk: payload.chunk })
      } else if (payload.type === 'status' && typeof payload.sessionId === 'string' && typeof payload.status === 'string') {
        for (const cb of listeners.terminalStatus) cb({ sessionId: payload.sessionId, status: payload.status as CliSession['status'] })
      } else if (payload.type === 'sessions' && typeof payload.projectId === 'string' && Array.isArray(payload.sessions)) {
        for (const cb of listeners.terminalSessions) cb({ projectId: payload.projectId, sessions: payload.sessions as CliSession[] })
      }
      return
    }
    if (event.kind === 'thread' || event.kind === 'project' || event.kind === 'room' || event.kind === 'ticket' || event.kind === 'job' || event.kind === 'automation') {
      const item = event.data as { id?: unknown }
      if (!item || typeof item.id !== 'string') return
      applyPush(event.kind, event.data as RemoteRecord)
      notifyRecord(event.kind, event.data as RemoteRecord)
    }
  })

  // Read-only settings for a paired device: the mock's defaults for look-and-feel, the
  // HOST's redacted providers for labels/pickers, and every host-config surface blanked
  // so demo accounts can never masquerade as real ones.
  async function pairedSettings(): Promise<Settings> {
    await ensureSnapshot()
    const mockSettings = await base.getSettings()
    const activeProvider = state.providers.find((provider) => provider.id === mockSettings.activeProviderId) ?? state.providers[0] ?? null
    return {
      ...mockSettings,
      providers: state.providers,
      activeProviderId: activeProvider?.id ?? null,
      activeProviderAccountId: null,
      activeModel: activeProvider?.models?.[0] ?? null,
      sshHosts: [],
      trustedSshHostKeys: [],
      approvalRules: [],
      email: { ...mockSettings.email, sources: [] },
      slack: { ...mockSettings.slack, sources: [] },
      calendar: { ...mockSettings.calendar, sources: [] },
      documents: { ...mockSettings.documents, sources: [] },
      browser: { ...mockSettings.browser, sources: [] },
      whatsapp: { ...mockSettings.whatsapp, sources: [] },
      discord: { ...mockSettings.discord, sources: [] },
      mattermost: { ...mockSettings.mattermost, sources: [] },
      matrix: { ...mockSettings.matrix, sources: [] },
      googlechat: { ...mockSettings.googlechat, sources: [] },
      imessage: { ...mockSettings.imessage, enabled: false, lastSyncAt: null, lastError: null },
      botControl: {
        telegram: { ...mockSettings.botControl.telegram, enabled: false, botToken: '', allowedChatIds: '', allowedUserIds: '', lastError: null },
        whatsapp: { ...mockSettings.botControl.whatsapp, enabled: false, accessToken: '', verifyToken: '', appSecret: '', allowedChatIds: '', lastError: null },
      },
      remoteControl: {
        ...mockSettings.remoteControl,
        enabled: false,
        serverName: state.serverName,
        authToken: '',
        allowLan: false,
        tunnelEnabled: false,
        tunnelUrl: null,
        lastStartedAt: null,
        lastClientAt: null,
        lastError: null,
      },
      remoteClient: { ...mockSettings.remoteClient, enabled: false, serverUrl: '', authToken: '', lastConnectedAt: null, lastError: null },
      remoteOrchestrators: [],
      defaultExecutionTarget: 'local',
      defaultOrchestratorId: null,
    } as unknown as Settings
  }

  const unavailable = (what: string) => (..._args: unknown[]) => Promise.reject(new PairedSessionError(what))
  const noSubscription = () => () => {}

  const overrides: Partial<TensorFoldStudioApi> = {
    platform: 'web-remote',

    // --- core work surfaces, real over /v1 ---
    getThreads: async () => { await ensureSnapshot(); return state.threads },
    getProjects: async () => { await ensureSnapshot(); return state.projects },
    getRooms: async () => { await ensureSnapshot(); return state.rooms },
    getTickets: async () => { await ensureSnapshot(); return state.tickets },
    getJobs: async () => { await ensureSnapshot(); return state.jobs },
    getAutomations: async () => { await ensureSnapshot(); return state.automations },
    createThread: (cwd, kind = 'task', projectId = null, commandId, incognito = false) => {
      if (incognito) return Promise.reject(new Error('Incognito tasks are available in the local desktop app.'))
      return req<Thread>('/v1/threads', { method: 'POST', body: JSON.stringify({ cwd: cwd ?? '', kind, projectId: projectId ?? null, ...(commandId ? { commandId } : {}) }) })
    },
    startRun: (payload: StartRunPayload) => req<{ runId: string }>(`/v1/threads/${encodeURIComponent(payload.threadId)}/start`, {
      method: 'POST',
      body: JSON.stringify({ userText: payload.userText, attachments: payload.attachments, sourceThreadKey: payload.sourceThreadKey }),
    }),
    cueRun: (payload: CueRunPayload) => req<CueRunResult>(`/v1/threads/${encodeURIComponent(payload.threadId)}/guide`, {
      method: 'POST',
      body: JSON.stringify({ userText: payload.userText, attachments: payload.attachments }),
    }),
    cancelRun: (threadId: string) => req<{ ok: boolean }>(`/v1/threads/${encodeURIComponent(threadId)}/cancel`, { method: 'POST', body: '{}' }).then(() => {}),
    approve: (callId: string, decision: 'allow' | 'allow_always' | 'deny') =>
      req<{ ok: boolean }>(`/v1/approvals/${encodeURIComponent(callId)}`, { method: 'POST', body: JSON.stringify({ decision }) }).then(() => {}),
    createProject: (input) => req<Project>('/v1/projects', { method: 'POST', body: JSON.stringify(input ?? {}) }),
    updateProject: (id, patch) => req<Project>(`/v1/projects/${encodeURIComponent(id)}/update`, { method: 'POST', body: JSON.stringify(patch ?? {}) }),
    renameProject: (id, name) => req<Project>(`/v1/projects/${encodeURIComponent(id)}/update`, { method: 'POST', body: JSON.stringify({ name }) }).then(() => {}),
    deleteProject: (id) => req<Project | null>(`/v1/projects/${encodeURIComponent(id)}/delete`, { method: 'POST', body: '{}' }).then(() => {}),
    createRoom: (projectId, name) => req<Room>('/v1/rooms', { method: 'POST', body: JSON.stringify({ projectId, name }) }),
    updateRoom: (id, patch) => req<Room>(`/v1/rooms/${encodeURIComponent(id)}/update`, { method: 'POST', body: JSON.stringify(patch ?? {}) }),
    renameRoom: (id, name) => req<Room>(`/v1/rooms/${encodeURIComponent(id)}/update`, { method: 'POST', body: JSON.stringify({ name }) }).then(() => {}),
    deleteRoom: (id) => req<Room | null>(`/v1/rooms/${encodeURIComponent(id)}/delete`, { method: 'POST', body: '{}' }).then(() => {}),
    startRoom: (roomId, problem, attachments) => {
      if (attachments?.length) return Promise.reject(new PairedSessionError('Starting a room with attachments'))
      return req<{ ok: boolean; room?: Room }>(`/v1/rooms/${encodeURIComponent(roomId)}/start`, { method: 'POST', body: JSON.stringify({ problem }) })
    },
    cancelRoom: (roomId) => req<{ ok: boolean }>(`/v1/rooms/${encodeURIComponent(roomId)}/cancel`, { method: 'POST', body: '{}' }).then(() => {}),
    createTicket: (input) => req<Ticket>('/v1/tickets', { method: 'POST', body: JSON.stringify(input ?? {}) }),
    updateTicket: (id, patch) => req<Ticket>(`/v1/tickets/${encodeURIComponent(id)}/update`, { method: 'POST', body: JSON.stringify(patch ?? {}) }),
    deleteTicket: (id) => req<{ ok: boolean }>(`/v1/tickets/${encodeURIComponent(id)}/delete`, { method: 'POST', body: '{}' }).then(() => {}),
    listClis: () => req<CliDefinition[]>('/v1/clis'),
    listTerminals: (projectId) => req<CliSession[]>(`/v1/terminals?projectId=${encodeURIComponent(projectId)}`),
    startTerminal: (projectId, cliId, opts) => {
      if (opts?.command || cliId.startsWith('custom:')) return Promise.reject(new PairedSessionError('Starting custom terminal commands'))
      return req<CliSession>('/v1/terminals', { method: 'POST', body: JSON.stringify({ projectId, cliId, label: opts?.label }) })
    },
    writeTerminal: (sessionId, data) => req<{ ok: boolean }>(`/v1/terminals/${encodeURIComponent(sessionId)}/write`, { method: 'POST', body: JSON.stringify({ data }) }).then(() => {}),
    resizeTerminal: (sessionId, cols, rows) => req<{ ok: boolean }>(`/v1/terminals/${encodeURIComponent(sessionId)}/resize`, { method: 'POST', body: JSON.stringify({ cols, rows }) }).then(() => {}),
    renameTerminal: (sessionId, title) => req<{ ok: boolean }>(`/v1/terminals/${encodeURIComponent(sessionId)}/rename`, { method: 'POST', body: JSON.stringify({ title }) }).then(() => {}),
    killTerminal: (sessionId) => req<{ ok: boolean }>(`/v1/terminals/${encodeURIComponent(sessionId)}/kill`, { method: 'POST', body: '{}' }).then(() => {}),
    spineReadSince: (input) => req(`/v1/spine/events?since=${encodeURIComponent(String(input?.since ?? 0))}${input?.aggregate ? `&aggregate=${encodeURIComponent(input.aggregate)}` : ''}${input?.streamId ? `&streamId=${encodeURIComponent(input.streamId)}` : ''}${input?.limit ? `&limit=${input.limit}` : ''}`),
    spineReceipt: (commandId) => req<{ receipt: Awaited<ReturnType<TensorFoldStudioApi['spineReceipt']>> }>(`/v1/spine/receipt?commandId=${encodeURIComponent(commandId)}`).then((result) => result.receipt),

    // --- live pushes over the shared SSE stream ---
    onAgentEvent: (cb) => { listeners.agent.add(cb); return () => listeners.agent.delete(cb) },
    onThreadUpdate: (cb) => { listeners.thread.add(cb); return () => listeners.thread.delete(cb) },
    onProjectUpdate: (cb) => { listeners.project.add(cb); return () => listeners.project.delete(cb) },
    onRoomUpdate: (cb) => { listeners.room.add(cb); return () => listeners.room.delete(cb) },
    onTicketUpdate: (cb) => { listeners.ticket.add(cb); return () => listeners.ticket.delete(cb) },
    onJobUpdate: (cb) => { listeners.job.add(cb); return () => listeners.job.delete(cb) },
    onAutomationUpdate: (cb) => { listeners.automation.add(cb); return () => listeners.automation.delete(cb) },
    onTerminalData: (cb) => { listeners.terminalData.add(cb); return () => listeners.terminalData.delete(cb) },
    onTerminalStatus: (cb) => { listeners.terminalStatus.add(cb); return () => listeners.terminalStatus.delete(cb) },
    onTerminalSessions: (cb) => { listeners.terminalSessions.add(cb); return () => listeners.terminalSessions.delete(cb) },

    // --- settings: synthesized read-only ---
    getSettings: pairedSettings,
    saveSettings: unavailable('Changing settings') as TensorFoldStudioApi['saveSettings'],
    revokeApprovalRule: unavailable('Editing approval rules') as TensorFoldStudioApi['revokeApprovalRule'],

    // --- recoverable host archives; paired projection is redacted server-side ---
    getArchives: () => req<ArchiveSnapshot>('/v1/archives'),
    archiveItem: async (kind, id) => {
      const path = kind === 'project'
        ? `/v1/projects/${encodeURIComponent(id)}/delete`
        : kind === 'thread'
          ? `/v1/threads/${encodeURIComponent(id)}/archive`
          : kind === 'room'
            ? `/v1/rooms/${encodeURIComponent(id)}/delete`
            : `/v1/tickets/${encodeURIComponent(id)}/delete`
      const record = await req<RemoteRecord>(path, { method: 'POST', body: '{}' })
      if (record && typeof record.id === 'string') {
        applyPush(kind, record)
        notifyRecord(kind, record)
      }
      applySnapshot(await req<Record<string, unknown>>('/v1/snapshot'))
      return req<ArchiveSnapshot>('/v1/archives')
    },
    restoreItem: async (kind, id) => {
      const path = kind === 'project'
        ? `/v1/projects/${encodeURIComponent(id)}/restore`
        : kind === 'thread'
          ? `/v1/threads/${encodeURIComponent(id)}/restore`
          : kind === 'room'
            ? `/v1/rooms/${encodeURIComponent(id)}/restore`
            : `/v1/tickets/${encodeURIComponent(id)}/restore`
      const record = await req<RemoteRecord>(path, { method: 'POST', body: '{}' })
      if (record && typeof record.id === 'string') {
        applyPush(kind, record)
        notifyRecord(kind, record)
      }
      applySnapshot(await req<Record<string, unknown>>('/v1/snapshot'))
      return req<ArchiveSnapshot>('/v1/archives')
    },
    getUpdateStatus: async () => {
      await ensureSnapshot()
      return {
        state: 'not_configured', currentVersion: state.appVersion || 'unknown', latestVersion: null,
        releaseName: null, releaseNotes: null, downloadUrl: null, checksum: null, signature: null,
        signatureVerified: false, integrity: 'not_checked', feedUrl: null, checkedAt: null, error: null,
      }
    },
    listCliTools: async () => [],
    listSshHosts: async () => [],
    listImportableSessions: async () => [],
    importExternalSession: unavailable('Importing sessions') as TensorFoldStudioApi['importExternalSession'],
    purgeLocalMemory: unavailable('Purging host memory') as TensorFoldStudioApi['purgeLocalMemory'],
  }

  // Use the demo API only as a runtime key schema and for appearance defaults above.
  // Every unimplemented function is replaced with a paired-session refusal; subscriptions
  // become inert unsubscribable streams. This prevents a newly-added TensorFoldStudioApi method from
  // silently inheriting a convincing mock success or demo Assistant data.
  const failClosed: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(base as unknown as Record<string, unknown>)) {
    if (typeof value === 'function') failClosed[key] = key.startsWith('on') ? noSubscription : unavailable(key)
    else failClosed[key] = value
  }
  return Object.assign(failClosed, overrides) as unknown as TensorFoldStudioApi
}
