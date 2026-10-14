// Run: jiti src/lib/remote-transport.test.ts
//
// Browser /v1 transport (VST-035): snapshot-backed reads, wire shapes for run
// start/guide/cancel/approve, SSE fan-out, 401 handling, and the read-only paired
// settings (host providers redacted, demo host-config surfaces blanked).
import assert from 'node:assert/strict'
import { createRemoteBrowserApi, openAuthenticatedEventStream, PairedSessionError, type RemoteSseEvent } from './remote-transport'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('remote-transport tests:')

// Node test shim: the transport touches window only for auth-failure events. navigator is
// a getter-only global in modern node — defineProperty instead of assignment.
if (!(globalThis as { window?: unknown }).window) {
  Object.defineProperty(globalThis, 'window', { value: { dispatchEvent: () => true }, configurable: true, writable: true })
}
try {
  Object.defineProperty(globalThis, 'localStorage', { value: { getItem: () => null, setItem: () => {}, removeItem: () => {} }, configurable: true })
} catch { /* already present */ }

{
  let requestUrl = ''
  let requestInit: RequestInit | undefined
  let stream: ReturnType<typeof openAuthenticatedEventStream>
  const received = await new Promise<RemoteSseEvent>((resolve, reject) => {
    const fetchImpl = (async (url: RequestInfo | URL, init?: RequestInit) => {
      requestUrl = String(url)
      requestInit = init
      const frame = 'id: 7\nevent: thread\ndata: {"id":"7","kind":"thread","createdAt":9,"data":{"id":"t7"}}\n\n'
      return new Response(frame, { status: 200, headers: { 'content-type': 'text/event-stream' } })
    }) as typeof fetch
    stream = openAuthenticatedEventStream('/v1/events', 'device-secret', resolve, fetchImpl)
    setTimeout(() => reject(new Error('timed out waiting for authenticated SSE frame')), 1_000)
  })
  stream!.close()
  assert.equal(requestUrl, '/v1/events', 'device credential is absent from the SSE URL')
  assert.equal(new Headers(requestInit?.headers).get('x-vontra-token'), 'device-secret', 'SSE authenticates with a header')
  assert.equal(received.id, '7')
  assert.equal(received.kind, 'thread')
  assert.deepEqual(received.data, { id: 't7' })
  ok('authenticated SSE: token is header-only and frames stream through the bounded parser')
}

{
  let attempts = 0
  let halfOpenCancelled = false
  const errors: string[] = []
  let stream: ReturnType<typeof openAuthenticatedEventStream>
  const received = await new Promise<RemoteSseEvent>((resolve, reject) => {
    const fetchImpl = (async () => {
      attempts++
      if (attempts === 1) return await new Promise<Response>(() => {}) // hung connect
      if (attempts === 2) {
        return new Response(new ReadableStream<Uint8Array>({
          start() {},
          cancel() { halfOpenCancelled = true },
        }), { headers: { 'content-type': 'text/event-stream' } })
      }
      return new Response('data: {"id":"9","kind":"thread","createdAt":9,"data":{"id":"recovered"}}\n\n', {
        headers: { 'content-type': 'text/event-stream' },
      })
    }) as typeof fetch
    stream = openAuthenticatedEventStream('/v1/events', 'device-secret', resolve, fetchImpl, {
      connectTimeoutMs: 10,
      idleTimeoutMs: 10,
      reconnectDelayMs: 1,
      onError: (error) => errors.push(error.message),
    })
    setTimeout(() => reject(new Error('timed out waiting for SSE liveness recovery')), 1_000)
  })
  stream!.close()
  assert.equal(received.data && (received.data as { id?: string }).id, 'recovered')
  assert.equal(attempts, 3, 'hung connect and half-open read each trigger one reconnect')
  assert.equal(halfOpenCancelled, true, 'the half-open response body is cancelled before reconnect')
  assert.ok(errors.some((message) => /did not connect within 10ms/.test(message)), 'connect timeout is reported clearly')
  assert.ok(errors.some((message) => /idle for 10ms/.test(message)), 'idle timeout is reported clearly')
  ok('authenticated SSE: connect/read deadlines recover half-open streams')
}

interface Recorded { url: string; method: string; body: unknown; headers: Record<string, string> }

function harness(snapshot: Record<string, unknown> | (() => Record<string, unknown>), replies: Record<string, unknown> = {}) {
  const calls: Recorded[] = []
  let sseHandler: ((event: RemoteSseEvent) => void) | null = null
  const fetchImpl = (async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = String(url)
    calls.push({
      url: path,
      method: init?.method ?? 'GET',
      body: init?.body ? JSON.parse(String(init.body)) : null,
      headers: (init?.headers ?? {}) as Record<string, string>,
    })
    if (path.includes('401')) return { ok: false, status: 401, json: async () => ({ error: 'unauthorized' }) }
    if (path.startsWith('/v1/snapshot')) return { ok: true, status: 200, json: async () => typeof snapshot === 'function' ? snapshot() : snapshot }
    const match = Object.entries(replies).find(([key]) => path.startsWith(key))
    if (match) return { ok: true, status: 200, json: async () => match[1] }
    return { ok: true, status: 200, json: async () => ({}) }
  }) as unknown as typeof fetch
  const api = createRemoteBrowserApi({
    token: 'tok-1',
    fetchImpl,
    openEventStream: (_url, _token, onEvent) => {
      sseHandler = onEvent
      return { onEvent: () => {}, close: () => {} }
    },
  })
  return { api, calls, push: (event: RemoteSseEvent) => sseHandler?.(event) }
}

const SNAPSHOT = {
  generatedAt: 1,
  serverName: 'Mini',
  appVersion: '1.2.3',
  threads: [{ id: 't1', title: 'Host thread', kind: 'task', timeline: [], apiMessages: [] }],
  projects: [{ id: 'p1', name: 'Repo' }],
  rooms: [],
  tickets: [],
  jobs: [],
  automations: [],
  providers: [
    { id: 'codex', name: 'Codex CLI', kind: 'cloud', transport: 'codex-cli', models: ['gpt-5.5'], defaultModel: 'gpt-5.5' },
  ],
}

{
  const h = harness(SNAPSHOT, {
    '/v1/threads': { id: 'created-thread' },
    '/v1/projects': { id: 'created-project' },
    '/v1/rooms': { id: 'created-room' },
    '/v1/tickets': { id: 'created-ticket' },
    '/v1/terminals': { id: 'created-terminal' },
  })
  await h.api.createThread('/srv/repo', 'chat', 'p1', 'cmd-1')
  await h.api.createProject({ name: 'Created project', cwd: '/srv/repo' })
  await h.api.updateProject('p1', { name: 'Renamed project' })
  await h.api.renameProject('p1', 'Again')
  await h.api.deleteProject('p1')
  await h.api.createRoom('p1', 'Build room')
  await h.api.updateRoom('room 1', { problem: 'Ship it' })
  await h.api.renameRoom('room 1', 'Review room')
  await h.api.startRoom('room 1', 'Do the work')
  await h.api.cancelRoom('room 1')
  await h.api.deleteRoom('room 1')
  await h.api.createTicket({ projectId: 'p1', title: 'Fix it' })
  await h.api.updateTicket('ticket/1', { status: 'ready' })
  await h.api.deleteTicket('ticket/1')
  await h.api.startTerminal('p1', 'codex', { label: 'Agent' })
  await h.api.writeTerminal('term/1', 'hello\n')
  await h.api.resizeTerminal('term/1', 120, 40)
  await h.api.renameTerminal('term/1', 'Build')
  await h.api.killTerminal('term/1')

  const call = (url: string) => h.calls.find((entry) => entry.url === url)
  assert.deepEqual(call('/v1/threads')?.body, { cwd: '/srv/repo', kind: 'chat', projectId: 'p1', commandId: 'cmd-1' }, 'positional createThread args become the server request object')
  assert.deepEqual(call('/v1/rooms')?.body, { projectId: 'p1', name: 'Build room' }, 'positional createRoom args become the server request object')
  assert.deepEqual(call('/v1/projects/p1/update')?.body, { name: 'Renamed project' })
  assert.deepEqual(call('/v1/rooms/room%201/update')?.body, { problem: 'Ship it' })
  assert.deepEqual(call('/v1/rooms/room%201/start')?.body, { problem: 'Do the work' })
  assert.deepEqual(call('/v1/tickets/ticket%2F1/update')?.body, { status: 'ready' })
  assert.deepEqual(call('/v1/terminals/term%2F1/resize')?.body, { cols: 120, rows: 40 })
  assert.deepEqual(call('/v1/terminals/term%2F1/rename')?.body, { title: 'Build' })
  await assert.rejects(h.api.startRoom('room 1', 'with a file', [{ id: 'a1' } as never]), PairedSessionError)
  await assert.rejects(h.api.startTerminal('p1', 'custom:danger', { command: 'rm -rf nope' }), PairedSessionError)
  ok('core paired mutations: explicit /v1 routes and exact positional wire shapes')
}

{
  const h = harness(SNAPSHOT)
  const threads = await h.api.getThreads()
  assert.equal(threads.length, 1)
  assert.equal(threads[0].title, 'Host thread', 'threads come from the HOST snapshot, not demo fixtures')
  const projects = await h.api.getProjects()
  assert.equal(projects[0].name, 'Repo')
  const snapshotCalls = h.calls.filter((c) => c.url.startsWith('/v1/snapshot'))
  assert.equal(snapshotCalls.length, 1, 'snapshot is fetched once and cached')
  assert.equal(snapshotCalls[0].headers['x-vontra-token'], 'tok-1', 'requests carry the device token')
  ok('snapshot-backed reads: host data, cached, token attached')
}

{
  let active = true
  let generatedAt = 20
  const thread = { ...(SNAPSHOT.threads[0] as any), updatedAt: 1, archivedAt: null }
  const fetchImpl = (async (url: RequestInfo | URL, init?: RequestInit) => {
    const path = String(url)
    if (path === '/v1/threads/t1/archive' && init?.method === 'POST') {
      active = false
      generatedAt++
      return { ok: true, status: 202, json: async () => ({ ...thread, archivedAt: generatedAt, updatedAt: generatedAt }) }
    }
    if (path === '/v1/threads/t1/restore' && init?.method === 'POST') {
      active = true
      generatedAt++
      return { ok: true, status: 202, json: async () => ({ ...thread, archivedAt: null, updatedAt: generatedAt }) }
    }
    if (path === '/v1/snapshot') {
      return {
        ok: true,
        status: 200,
        json: async () => ({ ...SNAPSHOT, generatedAt, threads: active ? [{ ...thread, updatedAt: generatedAt }] : [] }),
      }
    }
    if (path === '/v1/archives') {
      return {
        ok: true,
        status: 200,
        json: async () => ({
          projects: [],
          threads: active ? [] : [{ ...thread, archivedAt: generatedAt, updatedAt: generatedAt }],
          rooms: [],
          tickets: [],
        }),
      }
    }
    throw new Error(`Unexpected remote transport request: ${path}`)
  }) as unknown as typeof fetch
  const api = createRemoteBrowserApi({
    token: 'tok-no-sse',
    fetchImpl,
    openEventStream: () => ({ onEvent: () => {}, close: () => {} }),
  })

  assert.deepEqual((await api.getThreads()).map((item) => item.id), ['t1'])
  const archived = await api.archiveItem('thread', 't1')
  assert.deepEqual(await api.getThreads(), [], 'archive refreshes the active cache even when SSE emits nothing')
  assert.deepEqual(archived.threads.map((item) => item.id), ['t1'])

  const restored = await api.restoreItem('thread', 't1')
  assert.deepEqual((await api.getThreads()).map((item) => item.id), ['t1'], 'restore repopulates the active cache without an SSE push')
  assert.deepEqual(restored.threads, [])
  ok('archive/restore without SSE: mutation response and snapshot refresh keep active/archive caches coherent')
}

{
  let snapshotAttempts = 0
  const fetchImpl = (async (url: RequestInfo | URL) => {
    assert.equal(String(url), '/v1/snapshot')
    snapshotAttempts++
    if (snapshotAttempts === 1) {
      return { ok: false, status: 503, json: async () => ({ error: 'temporarily unavailable' }) }
    }
    return { ok: true, status: 200, json: async () => SNAPSHOT }
  }) as unknown as typeof fetch
  const api = createRemoteBrowserApi({
    token: 'tok-retry',
    fetchImpl,
    openEventStream: () => ({ onEvent: () => {}, close: () => {} }),
  })

  await assert.rejects(api.getThreads(), /Host request failed \(503\): temporarily unavailable/)
  assert.equal((await api.getThreads())[0]?.title, 'Host thread', 'a later read retries and applies the recovered snapshot')
  assert.equal(snapshotAttempts, 2, 'rejected snapshot promises are cleared instead of memoized forever')
  ok('snapshot retry: a rejected memoized attempt is cleared for the next read')
}

{
  const wrongMime = createRemoteBrowserApi({
    token: 'tok-mime',
    fetchImpl: (async () => new Response('<html>not json</html>', { status: 200, headers: { 'content-type': 'text/html' } })) as typeof fetch,
    openEventStream: () => ({ onEvent: () => {}, close: () => {} }),
  })
  await assert.rejects(wrongMime.getThreads(), /non-JSON response/)

  const oversized = createRemoteBrowserApi({
    token: 'tok-size',
    fetchImpl: (async () => new Response('x', {
      status: 200,
      headers: { 'content-type': 'application/json', 'content-length': String(17 * 1024 * 1024) },
    })) as typeof fetch,
    openEventStream: () => ({ onEvent: () => {}, close: () => {} }),
  })
  await assert.rejects(oversized.getThreads(), /safety limit/)
  ok('remote JSON: MIME and response-size limits fail closed')
}

{
  const h = harness(SNAPSHOT, { '/v1/threads/t1/start': { runId: 'r1' } })
  const run = await h.api.startRun({ threadId: 't1', userText: 'go' })
  assert.equal(run.runId, 'r1')
  const call = h.calls.find((c) => c.url === '/v1/threads/t1/start')
  assert.equal(call?.method, 'POST')
  assert.equal((call?.body as { userText?: string }).userText, 'go')

  await h.api.cueRun({ threadId: 't1', userText: 'steer it' })
  assert.ok(h.calls.some((c) => c.url === '/v1/threads/t1/guide'), 'guide rides /v1')
  await h.api.cancelRun('t1')
  assert.ok(h.calls.some((c) => c.url === '/v1/threads/t1/cancel'))
  await h.api.approve('call-9', 'allow')
  const approval = h.calls.find((c) => c.url === '/v1/approvals/call-9')
  assert.deepEqual(approval?.body, { decision: 'allow' })
  ok('run start/guide/cancel/approve: correct /v1 wire shapes')
}

{
  const h = harness(SNAPSHOT)
  const agentEvents: unknown[] = []
  const threadPushes: unknown[] = []
  h.api.onAgentEvent((e) => agentEvents.push(e))
  h.api.onThreadUpdate((t) => threadPushes.push(t))
  h.push({ id: '1', kind: 'agent', createdAt: 1, data: { type: 'status', status: 'thinking', threadId: 't1', runId: 'r1' } })
  h.push({ id: '2', kind: 'thread', createdAt: 2, data: { id: 't2', title: 'New from host' } })
  assert.equal(agentEvents.length, 1)
  assert.equal((agentEvents[0] as { status?: string }).status, 'thinking')
  assert.equal(threadPushes.length, 1)
  const threads = await h.api.getThreads()
  assert.ok(threads.some((t) => t.id === 't2'), 'thread pushes upsert the cache')
  ok('SSE fan-out: agent events reach listeners, thread pushes update the cache')
}

{
  let snapshot = SNAPSHOT
  const h = harness(() => snapshot)
  const projected: any[] = []
  h.api.onThreadUpdate((thread) => projected.push(thread))
  assert.equal((await h.api.getThreads())[0].title, 'Host thread')

  snapshot = {
    ...SNAPSHOT,
    generatedAt: 50,
    threads: [{ ...(SNAPSHOT.threads[0] as any), title: 'Recovered from canonical snapshot', updatedAt: 50 }],
  }
  h.push({
    id: '49', kind: 'stream_gap', createdAt: 49,
    data: { reason: 'backpressure', droppedEvents: 12, droppedBytes: 4096, fromEventId: '30', toEventId: '49', recovery: 'snapshot' },
  })
  await new Promise((resolve) => setTimeout(resolve, 0))

  assert.equal((await h.api.getThreads())[0].title, 'Recovered from canonical snapshot')
  assert.equal(projected.at(-1)?.title, 'Recovered from canonical snapshot')
  assert.equal(h.calls.filter((call) => call.url.startsWith('/v1/snapshot')).length, 2, 'an explicit stream gap forces a fresh canonical snapshot')
  ok('SSE backpressure gap: the paired cache re-baselines instead of keeping partial streamed text')
}

{
  const h = harness(SNAPSHOT)
  const threads: any[] = []
  const projects: any[] = []
  const tickets: any[] = []
  h.api.onThreadUpdate((item) => threads.push(item))
  h.api.onProjectUpdate((item) => projects.push(item))
  h.api.onTicketUpdate((item) => tickets.push(item))
  await h.api.getThreads()
  threads.length = projects.length = tickets.length = 0

  h.push({
    id: '10', kind: 'snapshot', createdAt: 10,
    data: {
      ...SNAPSHOT,
      generatedAt: 10,
      threads: [{ ...(SNAPSHOT.threads[0] as any), title: 'Changed on reconnect', updatedAt: 10 }],
      projects: [],
      tickets: [{ id: 'tk1', projectId: 'p1', title: 'New ticket', updatedAt: 10 }],
    },
  })
  assert.equal(threads.at(-1)?.title, 'Changed on reconnect', 'changed records project through listeners')
  assert.ok(projects.some((project) => project.id === 'p1' && project.archivedAt), 'records absent from the new snapshot project as tombstones')
  assert.equal(tickets.at(-1)?.id, 'tk1', 'snapshot additions project through listeners')
  assert.deepEqual((await h.api.getProjects()).map((project) => project.id), [], 'cache removes snapshot deletions')
  assert.deepEqual((await h.api.getTickets()).map((ticket) => ticket.id), ['tk1'])

  h.push({ id: '11', kind: 'snapshot', createdAt: 11, data: { ...SNAPSHOT, generatedAt: 11, projects: [], tickets: [], threads: [] } })
  assert.ok(threads.some((thread) => thread.id === 't1' && thread.archivedAt), 'thread deletion is projected')
  assert.ok(tickets.some((ticket) => ticket.id === 'tk1' && ticket.archivedAt), 'ticket deletion is projected')
  assert.deepEqual(await h.api.getThreads(), [])
  ok('snapshot reconnect: changes, additions, and deletions reach listeners and cache')
}

{
  const h = harness(SNAPSHOT)
  const settings = await h.api.getSettings()
  assert.equal(settings.providers.length, 1)
  assert.equal(settings.providers[0].id, 'codex', 'providers come from the host, redacted')
  assert.equal((settings.providers[0] as { apiKey?: string }).apiKey, '', 'no credential ever appears')
  assert.deepEqual(settings.sshHosts, [], 'demo ssh hosts are blanked')
  assert.deepEqual((settings as { email?: { sources?: unknown[] } }).email?.sources, [], 'demo email accounts are blanked')
  assert.deepEqual(settings.calendar.sources, [])
  assert.deepEqual(settings.documents.sources, [])
  assert.deepEqual(settings.browser.sources, [])
  assert.deepEqual(settings.discord.sources, [])
  assert.deepEqual(settings.mattermost.sources, [])
  assert.deepEqual(settings.matrix.sources, [])
  assert.deepEqual(settings.googlechat.sources, [])
  assert.deepEqual(settings.whatsapp.sources, [])
  assert.equal(settings.remoteControl.enabled, false)
  assert.equal(settings.remoteControl.authToken, '', 'demo host token is never presented as paired-session configuration')
  await assert.rejects(h.api.saveSettings(settings), PairedSessionError, 'settings are read-only from a paired session')
  assert.deepEqual(await h.api.listCliTools(), [], 'host CLIs render unknown, never demo-green')
  assert.deepEqual(await h.api.listSshHosts(), [])
  assert.equal(h.api.platform, 'web-remote')
  ok('paired settings: host providers redacted, demo surfaces blanked, mutations refuse')
}

{
  const h = harness(SNAPSHOT)
  const before = h.calls.length
  for (const action of [
    () => h.api.getAssistantSnapshot(),
    () => h.api.deleteThread('t1'),
    () => h.api.createAutomation({ name: 'must not be demo-created' }),
    () => h.api.retryJob('job-1'),
    () => h.api.generateTickets('p1', 'fake plan'),
  ]) await assert.rejects(action(), PairedSessionError)
  assert.equal(h.calls.length, before, 'unsupported actions never hit a guessed endpoint')
  assert.equal(typeof h.api.onRoomEvent(() => {}), 'function', 'unsupported subscription surfaces are inert/unsubscribable, not rejected promises')
  ok('fail-closed API: unsupported and Assistant surfaces cannot fall through to mock success')
}

{
  const h = harness(SNAPSHOT, {})
  let authFailed = false
  ;(globalThis as unknown as { window: { dispatchEvent: (e: { type?: string }) => boolean } }).window.dispatchEvent = (e) => {
    if ((e as { type?: string }).type === 'vontra:paired-auth-failed') authFailed = true
    return true
  }
  await assert.rejects(
    h.api.startRun({ threadId: '401-revoked', userText: 'x' }),
    /no longer authorized/,
  )
  assert.equal(authFailed, true, 'a 401 dispatches the auth-failure event for the pairing gate')
  ok('revoked token: 401 → clear error + auth-failure signal')
}

console.log(`  ${passed} passed`)
