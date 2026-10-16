import assert from 'node:assert/strict'
import { createServer } from 'node:net'
import { createRemoteControlServer, PairingAttemptLimiter, type RemoteControlServerDeps } from './remote-control'
import { createEventSpine } from './event-spine'
import { createPairingRegistry } from './pairing-sessions'
import type { RemoteControlSettings } from './types'

async function freePort(): Promise<number> {
  return await new Promise((resolve, reject) => {
    const probe = createServer()
    probe.once('error', reject)
    probe.listen(0, '127.0.0.1', () => {
      const address = probe.address()
      const port = typeof address === 'object' && address ? address.port : 0
      probe.close(() => resolve(port))
    })
  })
}

async function withSecurityRole<T>(role: 'viewer' | 'operator' | 'owner', run: () => Promise<T>): Promise<T> {
  const before = process.env.VONTRA_SECURITY_ROLE
  process.env.VONTRA_SECURITY_ROLE = role
  try { return await run() } finally {
    if (before == null) delete process.env.VONTRA_SECURITY_ROLE
    else process.env.VONTRA_SECURITY_ROLE = before
  }
}

const settings: RemoteControlSettings = {
  enabled: true,
  serverName: 'TensorFold Studio Test',
  bindHost: '127.0.0.1',
  allowLan: false,
  port: await freePort(),
  authToken: 'test-token',
  lastStartedAt: null,
  lastClientAt: null,
  lastError: null,
}

const terminalRenames: Array<{ sessionId: string; title: string }> = []
const approvals: Array<{ callId: string; decision: 'allow' | 'allow_always' | 'deny' }> = []
const ticketDeletes: string[] = []

function remoteDeps(s: RemoteControlSettings, deps: Partial<RemoteControlServerDeps> = {}): RemoteControlServerDeps {
  return {
    getSettings: () => s,
    getSnapshot: async () => ({ secretRawThreadContent: 'do-not-return-from-doctor' }) as any,
    getArchives: async () => ({ projects: [], threads: [], rooms: [], tickets: [] }),
    getPairedArchives: async () => ({ projects: [], threads: [], rooms: [], tickets: [] }),
    getDoctor: async () => ({
      generatedAt: 123,
      serverName: s.serverName,
      appVersion: '1.0.0',
      headless: true,
      counts: {
        threads: 2,
        projects: 1,
        rooms: 0,
        tickets: 0,
        automations: 0,
        missions: 0,
        jobs: 0,
        connectorEvents: 0,
        activeThreads: 1,
        activeRooms: 0,
        activeMissions: 0,
      },
      work: {
        jobs: { total: 0, visible: 0, queued: 0, running: 0, waitingApproval: 0, paused: 0, stalled: 0, failed: 0, deadLetter: 0, retryable: 0 },
        leases: { active: 0, expired: 0, expiringSoon: 0, missingRunning: 0 },
        approvals: { pending: 0, tool: 0, connector: 0, mission: 0 },
        nextWakeAt: null,
        nextWakeJobId: null,
        generatedAt: 123,
      },
    }),
    createThread: async () => ({ id: 't1', title: 'Remote task', kind: 'task', projectId: 'p1', cwd: '/tmp', workspace: null, providerId: null, providerAccountId: null, model: null, createdAt: Date.now(), updatedAt: Date.now(), status: 'idle', apiMessages: [], timeline: [] }) as any,
    startThread: async () => ({ runId: 'run' }),
    guideThread: async () => ({ queued: true, queueDepth: 1 }),
    cancelThread: async () => {},
    deleteThread: async () => ({ id: 't1', archivedAt: Date.now() }) as any,
    restoreThread: async () => ({ id: 't1', archivedAt: null }) as any,
    approve: async (callId, decision) => { approvals.push({ callId, decision }) },
    updateProject: async () => ({ id: 'p1', name: 'Remote project', cwd: '/tmp', memory: '', team: '', createdAt: Date.now(), updatedAt: Date.now() }) as any,
    deleteProject: async () => ({ id: 'p1', name: 'Remote project', cwd: '/tmp', memory: '', team: '', archivedAt: Date.now(), createdAt: Date.now(), updatedAt: Date.now() }) as any,
    restoreProject: async () => ({ id: 'p1', name: 'Remote project', cwd: '/tmp', memory: '', team: '', archivedAt: null, createdAt: Date.now(), updatedAt: Date.now() }) as any,
    createRoom: async () => ({ id: 'r1', projectId: 'p1', name: 'Room' }) as any,
    updateRoom: async () => ({ id: 'r1', projectId: 'p1', name: 'Room' }) as any,
    renameRoom: async (_id, name) => ({ id: 'r1', projectId: 'p1', name }) as any,
    steerRoom: async (_id, text) => ({ id: 'steer', tick: 1, kind: 'user', seat: 'you', target: 'all', text, createdAt: Date.now() }),
    mutateRoomMemory: async () => ({ id: 'r1', projectId: 'p1', name: 'Room', memory: [] }) as any,
    deleteRoom: async () => ({ id: 'r1', projectId: 'p1', name: 'Room', archivedAt: Date.now() }) as any,
    restoreRoom: async () => ({ id: 'r1', projectId: 'p1', name: 'Room', archivedAt: null }) as any,
    startRoom: async () => ({ ok: true }),
    cancelRoom: async () => {},
    createTicket: async () => ({ id: 'tk1', projectId: 'p1', title: 'Ticket' }) as any,
    startTicketExecution: async () => ({ ticket: { id: 'tk1', projectId: 'p1', title: 'Ticket' }, thread: { id: 't1' }, runId: 'run' }) as any,
    updateTicket: async () => ({ id: 'tk1', projectId: 'p1', title: 'Ticket' }) as any,
    transitionTicket: async () => ({ id: 'tk1', projectId: 'p1', title: 'Ticket' }) as any,
    deleteTicket: async (id) => { ticketDeletes.push(id); return id === 'missing' ? null : { id, archivedAt: Date.now() } as any },
    restoreTicket: async (id) => ({ id, archivedAt: null }) as any,
    setMissionStatus: async () => ({ id: 'm1' }) as any,
    recordMissionHeartbeat: async () => ({ id: 'm1' }) as any,
    listDirectories: async () => ({ ok: true, path: '/tmp', parent: '/', entries: [] }),
    createProject: async () => ({ id: 'p1', name: 'Remote project', cwd: '/tmp', memory: '', team: '', createdAt: Date.now(), updatedAt: Date.now() }) as any,
    onClientSeen: () => {},
    onStarted: () => {},
    onError: () => {},
    audit: () => {},
    ...deps,
  }
}

const server = createRemoteControlServer(remoteDeps(settings, {
  heartbeatIntervalMs: 20,
  listClis: async () => [
    { id: 'shell', name: 'Shell', short: 'zsh', command: null, resumeCommand: null, color: '#8B8B97', icon: 'terminal', builtin: true },
    { id: 'codex', name: 'Codex', short: 'codex', command: 'codex', resumeCommand: 'codex resume --last', color: '#2FB9B0', icon: 'code', approvedFlags: '--full-auto', builtin: true },
  ],
  renameTerminal: (sessionId, title) => { terminalRenames.push({ sessionId, title }) },
}))

await server.syncFromSettings()
const base = server.status().url
assert.ok(base, 'server should bind to an ephemeral loopback port')

try {
  const unauth = await fetch(`${base}/v1/control-plane`)
  assert.equal(unauth.status, 401)

  const headers = { Authorization: 'Bearer test-token' }
  const eventAbort = new AbortController()
  const eventResponse = await fetch(`${base}/v1/events`, { headers, signal: eventAbort.signal })
  assert.equal(eventResponse.status, 200)
  const eventReader = eventResponse.body!.getReader()
  const eventDecoder = new TextDecoder()
  let eventWire = ''
  const heartbeatDeadline = setTimeout(() => eventAbort.abort(), 2_000)
  try {
    while (!eventWire.includes(': heartbeat\n\n')) {
      const { done, value } = await eventReader.read()
      if (done) break
      eventWire += eventDecoder.decode(value, { stream: true }).replace(/\r\n/g, '\n')
    }
    assert.match(eventWire, /(?:^|\n): heartbeat\n\n/, 'SSE clients receive credential-free comment heartbeats')
  } finally {
    clearTimeout(heartbeatDeadline)
    await eventReader.cancel().catch(() => {})
    eventAbort.abort()
  }

  const catalogRes = await fetch(`${base}/v1/control-plane`, { headers })
  assert.equal(catalogRes.status, 200)
  const catalog = await catalogRes.json() as any
  assert.ok(catalog.commands.some((command: any) => command.id === 'doctor' && command.path === '/v1/doctor'))
  assert.ok(catalog.commands.some((command: any) => command.id === 'thread.start' && command.effect === 'write'))
  assert.ok(catalog.commands.some((command: any) => command.id === 'clis.list' && command.path === '/v1/clis'))
  assert.ok(catalog.commands.some((command: any) => command.id === 'terminal.rename' && command.path === '/v1/terminals/:id/rename'))
  assert.ok(catalog.commands.some((command: any) => command.id === 'project.delete' && command.title === 'Archive project' && /Soft-archive/.test(command.description)), 'legacy /delete route is documented honestly as archive')
  const commandIds = new Set(catalog.commands.map((command: any) => command.id))
  for (const id of [
    'control-plane',
    'doctor',
    'archives',
    'snapshot',
    'files.list',
    'clis.list',
    'project.create',
    'project.update',
    'project.delete',
    'project.restore',
    'thread.create',
    'thread.start',
    'thread.guide',
    'thread.cancel',
    'thread.archive',
    'thread.restore',
    'events',
    'approval.resolve',
    'terminal.list',
    'terminal.start',
    'terminal.write',
    'terminal.resize',
    'terminal.rename',
    'terminal.kill',
    'room.create',
    'room.update',
    'room.start',
    'room.cancel',
    'room.delete',
    'room.restore',
    'mission.status',
    'mission.heartbeat',
    'ticket.create',
    'ticket.update',
    'ticket.delete',
    'ticket.restore',
  ]) assert.ok(commandIds.has(id), `catalog includes ${id}`)

  const clisRes = await fetch(`${base}/v1/clis`, { headers })
  assert.equal(clisRes.status, 200)
  const clis = await clisRes.json() as any[]
  assert.deepEqual(clis.map((cli) => cli.id), ['shell', 'codex'])

  const terminalListUnavailable = await fetch(`${base}/v1/terminals?projectId=p1`, { headers })
  assert.equal(terminalListUnavailable.status, 501)

  const terminalStartUnavailable = await withSecurityRole('owner', () => fetch(`${base}/v1/terminals`, {
    method: 'POST',
    headers: { ...headers, 'content-type': 'application/json' },
    body: JSON.stringify({ projectId: 'p1', cliId: 'shell' }),
  }))
  assert.equal(terminalStartUnavailable.status, 501)

  const renameRes = await withSecurityRole('owner', () => fetch(`${base}/v1/terminals/sess%201/rename`, {
      method: 'POST',
      headers: { ...headers, 'content-type': 'application/json' },
      body: JSON.stringify({ title: '  Remote shell  ' }),
    }))
  assert.equal(renameRes.status, 202)
  assert.deepEqual(terminalRenames, [{ sessionId: 'sess 1', title: 'Remote shell' }])

  const approvalRes = await fetch(`${base}/v1/approvals/call%201`, {
    method: 'POST',
    headers: { ...headers, 'content-type': 'application/json' },
    body: JSON.stringify({ decision: 'allow_always' }),
  })
  assert.equal(approvalRes.status, 202)
  assert.deepEqual(approvals.at(-1), { callId: 'call 1', decision: 'allow_always' })

  const invalidApprovalRes = await fetch(`${base}/v1/approvals/call%202`, {
    method: 'POST',
    headers: { ...headers, 'content-type': 'application/json' },
    body: JSON.stringify({ decision: 'later' }),
  })
  assert.equal(invalidApprovalRes.status, 400)

  const deleteTicketRes = await fetch(`${base}/v1/tickets/tk%201/delete`, { method: 'POST', headers })
  assert.equal(deleteTicketRes.status, 202)
  assert.ok((await deleteTicketRes.json() as any).archivedAt, 'ticket delete returns the retained archive tombstone')
  assert.deepEqual(ticketDeletes.at(-1), 'tk 1')
  const missingArchive = await fetch(`${base}/v1/tickets/missing/delete`, { method: 'POST', headers })
  assert.equal(missingArchive.status, 404, 'archive endpoints cannot report success for a missing record')
  assert.deepEqual(await missingArchive.json(), { error: 'not_found' })

  const doctorRes = await fetch(`${base}/v1/doctor`, { headers })
  assert.equal(doctorRes.status, 200)
  const doctor = await doctorRes.json() as any
  assert.equal(doctor.counts.threads, 2)
  assert.equal(doctor.counts.activeThreads, 1)
  assert.doesNotMatch(JSON.stringify(doctor), /do-not-return-from-doctor/)
  console.log('remote-control redacted control-plane tests passed')
} finally {
  await server.stop()
}

await (async () => {
  const s: RemoteControlSettings = { ...settings, port: await freePort(), lastStartedAt: null, lastClientAt: null, lastError: null }
  const terminalServer = createRemoteControlServer(remoteDeps(s, {
    startTerminal: (input) => ({ id: 'term1', projectId: input.projectId, cliId: input.cliId, title: input.cliId, status: 'idle', createdAt: Date.now(), managedBy: 'external' }) as any,
    listTerminals: () => [],
  }))
  await terminalServer.syncFromSettings()
  const terminalBase = terminalServer.status().url
  assert.ok(terminalBase)
  const headers = { Authorization: 'Bearer test-token', 'content-type': 'application/json' }
  try {
    await withSecurityRole('owner', async () => {
      const customCommand = await fetch(`${terminalBase}/v1/terminals`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ projectId: 'p1', cliId: 'shell', command: 'gemini --yolo' }),
      })
      assert.equal(customCommand.status, 403, 'remote terminal rejects inline custom commands')
      const customCli = await fetch(`${terminalBase}/v1/terminals`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ projectId: 'p1', cliId: 'custom:gemini' }),
      })
      assert.equal(customCli.status, 403, 'remote terminal rejects custom cli ids')
      const builtin = await fetch(`${terminalBase}/v1/terminals`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ projectId: 'p1', cliId: 'shell' }),
      })
      assert.equal(builtin.status, 201, 'remote terminal starts builtin cli ids')
    })
  } finally {
    await terminalServer.stop()
  }
})()

await (async () => {
  const starts: Array<{ at: number; error: string | null }> = []
  const errors: Array<string | null> = []
  const s: RemoteControlSettings = {
    ...settings,
    enabled: false,
    port: await freePort(),
    lastStartedAt: null,
    lastClientAt: null,
    lastError: null,
  }
  const modeServer = createRemoteControlServer(remoteDeps(s, {
    onStarted: (at, error) => { starts.push({ at, error }) },
    onError: (message) => { errors.push(message) },
  }))
  try {
    await modeServer.syncFromSettings()
    assert.equal(modeServer.status().listening, false, 'disabled remote control does not listen')

    s.enabled = true
    const roundedPort = await freePort()
    s.port = roundedPort + 0.4
    await modeServer.syncFromSettings()
    assert.equal(modeServer.status().listening, true, 'enabled remote control listens')
    assert.equal(modeServer.status().port, roundedPort, 'fractional ports are normalized before listening')
    assert.ok(starts.length > 0, 'starting records a started callback')

    s.allowLan = true
    s.bindHost = '0.0.0.0'
    await modeServer.syncFromSettings()
    assert.equal(modeServer.status().listening, true, 'stale LAN settings are normalized to the safe listener')
    assert.equal(modeServer.status().bindHost, '127.0.0.1', 'the built-in plaintext server is always loopback-only')
  } finally {
    await modeServer.stop()
  }
})()

// ---- process security role caps authenticated REST authority ----
await (async () => {
  let projectsCreated = 0
  let terminalsStarted = 0
  const s: RemoteControlSettings = { ...settings, port: await freePort() }
  const roleServer = createRemoteControlServer(remoteDeps(s, {
    createProject: async () => {
      projectsCreated++
      return { id: `p${projectsCreated}`, name: 'Role project', cwd: '/tmp', memory: '', team: '', createdAt: Date.now(), updatedAt: Date.now() } as any
    },
    startTerminal: (input) => {
      terminalsStarted++
      return { id: `term${terminalsStarted}`, projectId: input.projectId, cliId: input.cliId, title: input.cliId, status: 'idle', createdAt: Date.now(), managedBy: 'external' } as any
    },
  }))
  await roleServer.syncFromSettings()
  const base = roleServer.status().url!
  const readHeaders = { Authorization: 'Bearer test-token' }
  const write = (path: string, body: Record<string, unknown>) => fetch(`${base}${path}`, {
    method: 'POST',
    headers: { ...readHeaders, 'content-type': 'application/json' },
    body: JSON.stringify(body),
  })
  try {
    await withSecurityRole('viewer', async () => {
      assert.equal((await fetch(`${base}/v1/snapshot`, { headers: readHeaders })).status, 200, 'viewer keeps authenticated reads')
      const denied = await write('/v1/projects', { name: 'Denied' })
      assert.equal(denied.status, 403, 'viewer cannot mutate REST resources')
      assert.equal((await denied.json() as any).requiredRole, 'operator')
      assert.equal(projectsCreated, 0, 'denied viewer mutation never reaches its handler')
    })

    await withSecurityRole('operator', async () => {
      assert.equal((await write('/v1/projects', { name: 'Allowed' })).status, 201, 'operator may perform ordinary mutations')
      const terminal = await write('/v1/terminals', { projectId: 'p1', cliId: 'shell' })
      assert.equal(terminal.status, 403, 'operator cannot cross the owner-only terminal boundary')
      assert.equal(terminalsStarted, 0)
    })

    await withSecurityRole('owner', async () => {
      assert.equal((await write('/v1/terminals', { projectId: 'p1', cliId: 'shell' })).status, 201, 'owner may drive terminals')
      assert.equal(terminalsStarted, 1)
    })
  } finally {
    await roleServer.stop()
  }
})()

// The brute-force ledger itself is bounded and expires old source keys.
{
  let now = 0
  const limiter = new PairingAttemptLimiter({ now: () => now, maxFailures: 2, windowMs: 1_000, maxEntries: 2 })
  limiter.recordFailure('a')
  limiter.recordFailure('b')
  limiter.recordFailure('c')
  assert.equal(limiter.trackedSourceCount(), 2, 'source churn cannot grow the attempt ledger past its cap')
  assert.equal(limiter.check('a').blocked, false, 'oldest source is evicted when the cap is reached')
  limiter.recordFailure('c')
  assert.equal(limiter.check('c').blocked, true)
  now = 1_001
  assert.equal(limiter.trackedSourceCount(), 0, 'attempt state expires with its rate-limit window')
}

// ---- ordered event spine over the remote surface (VST-034) ----
// Durable cursor replay + receipt lookups for remote clients; 501 without a spine.
await (async () => {
  let ids = 0
  const spine = createEventSpine({ now: () => 123456, newEventId: () => `s${++ids}` })
  spine.append({ aggregate: 'thread', streamId: 'T1', type: 'agent.status', payload: { status: 'thinking' } })
  spine.append({ aggregate: 'job', streamId: 'J1', type: 'job.queued', payload: {} })
  spine.append({ aggregate: 'thread', streamId: 'T1', type: 'agent.run-done', payload: {} })
  spine.execute({ commandId: 'cmd-spine-1', aggregate: 'thread', streamId: 'T1', kind: 'thread.create' }, () => (
    { aggregate: 'thread', streamId: 'T1', type: 'thread.created', payload: {} }
  ))

  const s: RemoteControlSettings = { ...settings, port: await freePort() }
  const spineServer = createRemoteControlServer(remoteDeps(s, {
    spineReadSince: (since, options) => spine.readSince(since, options),
    spineReceipt: (commandId) => spine.receipt(commandId),
  }))
  const noSpineSettings: RemoteControlSettings = { ...settings, port: await freePort() }
  const noSpineServer = createRemoteControlServer(remoteDeps(noSpineSettings))
  try {
    await spineServer.syncFromSettings()
    await noSpineServer.syncFromSettings()
    const base = spineServer.status().url!
    const headers = { Authorization: 'Bearer test-token' }

    const unauth = await fetch(`${base}/v1/spine/events`)
    assert.equal(unauth.status, 401, 'spine replay sits behind the auth token')

    const catalog = await (await fetch(`${base}/v1/control-plane`, { headers })).json() as any
    assert.ok(catalog.commands.some((command: any) => command.id === 'spine.events' && command.path === '/v1/spine/events' && command.effect === 'read'))
    assert.ok(catalog.commands.some((command: any) => command.id === 'spine.receipt' && command.effect === 'read'))

    const all = await (await fetch(`${base}/v1/spine/events?since=0`, { headers })).json() as any
    assert.equal(all.events.length, 4)
    assert.deepEqual(all.events.map((event: any) => event.sequence), [1, 2, 3, 4], 'ordered replay')
    assert.equal(all.snapshotRequired, false)

    const paged = await (await fetch(`${base}/v1/spine/events?since=1&limit=1`, { headers })).json() as any
    assert.equal(paged.events.length, 1)
    assert.equal(paged.events[0].sequence, 2)
    assert.equal(paged.hasMore, true, 'paging reports the remainder')

    const filtered = await (await fetch(`${base}/v1/spine/events?since=0&aggregate=thread&streamId=T1`, { headers })).json() as any
    assert.deepEqual(filtered.events.map((event: any) => event.type), ['agent.status', 'agent.run-done', 'thread.created'], 'aggregate+stream filters')

    const receipt = await (await fetch(`${base}/v1/spine/receipt?commandId=cmd-spine-1`, { headers })).json() as any
    assert.equal(receipt.receipt.status, 'accepted')
    assert.equal(receipt.receipt.resultSequence, 4)
    const missing = await (await fetch(`${base}/v1/spine/receipt?commandId=nope`, { headers })).json() as any
    assert.equal(missing.receipt, null)
    const badRequest = await fetch(`${base}/v1/spine/receipt`, { headers })
    assert.equal(badRequest.status, 400, 'a receipt lookup requires a commandId')

    const noSpineBase = noSpineServer.status().url!
    const unavailable = await fetch(`${noSpineBase}/v1/spine/events`, { headers })
    assert.equal(unavailable.status, 501, 'a host without a spine serves 501, never a fake empty log')
  } finally {
    await spineServer.stop()
    await noSpineServer.stop()
  }
})()

// ---- per-device pairing + sessions (VST-035) ----
// One-time pairing credential → durable per-device session token; owner-only session
// management; a device token authorizes the data API but not pairing administration.
await (async () => {
  const s: RemoteControlSettings = { ...settings, port: await freePort() }
  const reg = createPairingRegistry({ linkTtlMs: 600_000 })
  const pairedThreadStarts: any[] = []
  const pairedThreadGuides: any[] = []
  const pairedRoomStarts: any[] = []
  const pairedArchiveCalls: string[] = []
  const pairedListDirCalls: Array<{ dir: string; options?: { confineToHome?: boolean } }> = []
  const server = createRemoteControlServer(remoteDeps(s, {
    getSnapshot: async () => ({ scope: 'owner', secret: 'owner-only' }) as any,
    getPairedSnapshot: async () => ({ scope: 'paired', secret: undefined }) as any,
    startThread: async (payload) => { pairedThreadStarts.push(payload); return { runId: 'paired-run' } },
    guideThread: async (payload) => { pairedThreadGuides.push(payload); return { queued: true, queueDepth: 1 } },
    startRoom: async (roomId, problem, attachments) => { pairedRoomStarts.push({ roomId, problem, attachments }); return { ok: true } },
    deleteProject: async (id) => { pairedArchiveCalls.push(`project:archive:${id}`); return { id, archivedAt: 1 } as any },
    restoreProject: async (id) => { pairedArchiveCalls.push(`project:restore:${id}`); return { id, archivedAt: null } as any },
    deleteThread: async (id) => { pairedArchiveCalls.push(`thread:archive:${id}`); return { id, archivedAt: 1 } as any },
    restoreThread: async (id) => { pairedArchiveCalls.push(`thread:restore:${id}`); return { id, archivedAt: null } as any },
    deleteRoom: async (id) => { pairedArchiveCalls.push(`room:archive:${id}`); return { id, archivedAt: 1 } as any },
    restoreRoom: async (id) => { pairedArchiveCalls.push(`room:restore:${id}`); return { id, archivedAt: null } as any },
    deleteTicket: async (id) => { pairedArchiveCalls.push(`ticket:archive:${id}`); return { id, archivedAt: 1 } as any },
    restoreTicket: async (id) => { pairedArchiveCalls.push(`ticket:restore:${id}`); return { id, archivedAt: null } as any },
    listDirectories: async (dir, options) => { pairedListDirCalls.push({ dir, options }); return { ok: true, path: '/tmp', parent: '/', entries: [] } },
    pairing: {
      issuePairingLink: (label: string) => reg.issuePairingLink(label),
      authorizeSessionToken: (token: string) => reg.authorizeSessionToken(token),
      redeemPairingLink: (credential: string, clientInfo?: string | null) => reg.redeemPairingLink(credential, clientInfo),
      listSessions: () => reg.listSessions(),
      revokeSession: (id: string) => reg.revokeSession(id),
    },
  }))
  try {
    await server.syncFromSettings()
    const base = server.status().url!
    const owner = { Authorization: 'Bearer test-token' }

    // Owner issues a one-time pairing credential (host-side), then a fresh device redeems it.
    const link = await server.issuePairingLink('Alex phone')
    assert.ok(link.credential, 'owner issues a pairing credential')
    const paired = await (await fetch(`${base}/v1/auth/pair`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ credential: link.credential, clientInfo: 'iOS' }) })).json() as any
    assert.ok(paired.token && paired.sessionId, 'redeeming yields a device session token + id')
    assert.notEqual(paired.token, 'test-token', 'the device token is NOT the shared owner token')

    // The device token authorizes the data API…
    const snap = await fetch(`${base}/v1/snapshot`, { headers: { Authorization: `Bearer ${paired.token}` } })
    assert.equal(snap.status, 200, 'a paired device token authorizes the data API')
    assert.deepEqual(await snap.json(), { scope: 'paired' }, 'paired tokens receive the minimized snapshot projection')
    const ownerSnapshot = await (await fetch(`${base}/v1/snapshot`, { headers: owner })).json() as any
    assert.equal(ownerSnapshot.secret, 'owner-only', 'the owner token retains the full local snapshot')
    const pairedArchives = await fetch(`${base}/v1/archives`, { headers: { Authorization: `Bearer ${paired.token}` } })
    assert.equal(pairedArchives.status, 200, 'paired tokens can enumerate the redacted recovery archive')

    // Directory listing follows the same least-privilege split: a paired device is confined
    // to the home subtree (enforced in listHostDirectories), the owner browses unrestricted.
    const filesAsDevice = await fetch(`${base}/v1/files?dir=${encodeURIComponent('~')}`, { headers: { Authorization: `Bearer ${paired.token}` } })
    assert.equal(filesAsDevice.status, 200, 'a paired device token can list directories')
    assert.equal(pairedListDirCalls.at(-1)?.options?.confineToHome, true, 'a paired device token is confined to the home subtree')
    const filesAsOwner = await fetch(`${base}/v1/files?dir=${encodeURIComponent('~')}`, { headers: owner })
    assert.equal(filesAsOwner.status, 200)
    assert.equal(pairedListDirCalls.at(-1)?.options?.confineToHome, false, 'the owner token keeps unrestricted directory listing')

    for (const role of ['owner', 'operator'] as const) {
      await withSecurityRole(role, async () => {
        for (const [route, body] of [
          ['/v1/terminals', { projectId: 'p1', cliId: 'shell' }],
          ['/v1/terminals/fixture/write', { data: 'text' }],
          ['/v1/terminals/fixture/resize', { cols: 80, rows: 24 }],
          ['/v1/terminals/fixture/rename', { title: 'name' }],
          ['/v1/terminals/fixture/kill', {}],
        ] as const) {
          const rejected = await fetch(`${base}${route}`, { method: 'POST', headers: { Authorization: `Bearer ${paired.token}`, 'content-type': 'application/json' }, body: JSON.stringify(body) })
          assert.equal(rejected.status, 403, `paired token cannot mutate terminals on ${role} host: ${route}`)
        }
      })
    }

    // Paired write routes apply the same host-owned attachment policy as local IPC before their
    // callbacks can persist or forward content. This covers thread start + guide and the room
    // path that previously discarded attachments at the REST bridge.
    await withSecurityRole('operator', async () => {
      const pairedHeaders = { Authorization: `Bearer ${paired.token}`, 'content-type': 'application/json' }
      for (const [path, body] of [
        ['/v1/projects/p1/update', { archivedAt: null }],
        ['/v1/rooms/r1/update', { id: 'replacement' }],
        ['/v1/tickets', { projectId: 'p1', archivedAt: 1 }],
      ] as const) {
        const rejected = await fetch(`${base}${path}`, { method: 'POST', headers: pairedHeaders, body: JSON.stringify(body) })
        assert.equal(rejected.status, 400, `paired REST cannot transition archive/id state through generic mutation ${path}`)
      }
      const document = { kind: 'document', name: '../../plan.md', mime: 'NOT A MIME', text: 'safe plan', bytes: 999_999 }
      const start = await fetch(`${base}/v1/threads/t1/start`, {
        method: 'POST', headers: pairedHeaders, body: JSON.stringify({ userText: 'start', attachments: [document] }),
      })
      assert.equal(start.status, 202, 'paired thread start accepts a bounded document')
      assert.equal(pairedThreadStarts[0].attachments[0].name, 'plan.md', 'paired start receives a normalized leaf filename')
      assert.equal(pairedThreadStarts[0].attachments[0].mime, 'text/plain', 'paired start receives normalized MIME')
      assert.equal(pairedThreadStarts[0].attachments[0].bytes, Buffer.byteLength('safe plan'), 'paired start receives recomputed bytes')

      const guide = await fetch(`${base}/v1/threads/t1/guide`, {
        method: 'POST', headers: pairedHeaders, body: JSON.stringify({ userText: 'guide', attachments: [document] }),
      })
      assert.equal(guide.status, 202, 'paired guide accepts normalized attachments')
      assert.equal(pairedThreadGuides.length, 1)
      assert.equal(pairedThreadGuides[0].attachments[0].name, 'plan.md')

      const external = { kind: 'image', name: 'internal.png', dataUrl: 'http://169.254.169.254/latest/meta-data' }
      const rejectedStart = await fetch(`${base}/v1/threads/t1/start`, {
        method: 'POST', headers: pairedHeaders, body: JSON.stringify({ userText: 'unsafe', attachments: [external] }),
      })
      assert.equal(rejectedStart.status, 400, 'paired start rejects external/private image URLs before its callback')
      assert.equal(pairedThreadStarts.length, 1, 'rejected paired start never reaches persistence/model callback')

      const rejectedGuide = await fetch(`${base}/v1/threads/t1/guide`, {
        method: 'POST', headers: pairedHeaders, body: JSON.stringify({ userText: 'unsafe', attachments: 'not-an-array' }),
      })
      assert.equal(rejectedGuide.status, 400, 'paired guide rejects non-array attachments instead of silently dropping them')
      assert.equal(pairedThreadGuides.length, 1, 'rejected paired guide never reaches its callback')

      const room = await fetch(`${base}/v1/rooms/r1/start`, {
        method: 'POST', headers: pairedHeaders, body: JSON.stringify({ problem: 'review', attachments: [document] }),
      })
      assert.equal(room.status, 202, 'paired room start forwards normalized attachments')
      assert.equal(pairedRoomStarts[0].attachments[0].name, 'plan.md')
      const rejectedRoom = await fetch(`${base}/v1/rooms/r1/start`, {
        method: 'POST', headers: pairedHeaders, body: JSON.stringify({ problem: 'review', attachments: [external] }),
      })
      assert.equal(rejectedRoom.status, 400, 'paired room start rejects external image URLs')
      assert.equal(pairedRoomStarts.length, 1, 'rejected paired room start never reaches its callback')

      for (const [path, expected] of [
        ['/v1/projects/pair-project/delete', 'project:archive:pair-project'],
        ['/v1/projects/pair-project/restore', 'project:restore:pair-project'],
        ['/v1/threads/pair-thread/archive', 'thread:archive:pair-thread'],
        ['/v1/threads/pair-thread/restore', 'thread:restore:pair-thread'],
        ['/v1/rooms/pair-room/delete', 'room:archive:pair-room'],
        ['/v1/rooms/pair-room/restore', 'room:restore:pair-room'],
        ['/v1/tickets/pair-ticket/delete', 'ticket:archive:pair-ticket'],
        ['/v1/tickets/pair-ticket/restore', 'ticket:restore:pair-ticket'],
      ] as const) {
        const response = await fetch(`${base}${path}`, { method: 'POST', headers: pairedHeaders, body: '{}' })
        assert.equal(response.status, 202, `paired direct REST ${path} is authenticated and available`)
        assert.equal(pairedArchiveCalls.at(-1), expected, `paired direct REST ${path} reaches only the soft archive/restore dependency`)
      }
    })
    // …but NOT owner-only session administration.
    const listAsDevice = await fetch(`${base}/v1/auth/sessions`, { headers: { Authorization: `Bearer ${paired.token}` } })
    assert.equal(listAsDevice.status, 403, 'a device cannot list sessions (owner-only)')

    // Owner lists sessions — never sees raw tokens.
    const listResponse = await withSecurityRole('owner', () => fetch(`${base}/v1/auth/sessions`, { headers: owner }))
    const list = await listResponse.json() as any
    assert.equal(list.sessions.length, 1)
    assert.equal(list.sessions[0].id, paired.sessionId)
    assert.ok(!('token' in list.sessions[0]), 'the sessions list never exposes raw device tokens')

    // A consumed credential cannot be redeemed again.
    const reuse = await fetch(`${base}/v1/auth/pair`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ credential: link.credential }) })
    assert.equal(reuse.status, 400, 'a one-time pairing credential is refused on reuse')

    // Owner revokes the device → its token stops authorizing.
    const revokeResponse = await withSecurityRole('owner', () => fetch(`${base}/v1/auth/sessions/revoke`, { method: 'POST', headers: { ...owner, 'content-type': 'application/json' }, body: JSON.stringify({ sessionId: paired.sessionId }) }))
    const revoke = await revokeResponse.json() as any
    assert.equal(revoke.revoked, true)
    const afterRevoke = await fetch(`${base}/v1/snapshot`, { headers: { Authorization: `Bearer ${paired.token}` } })
    assert.equal(afterRevoke.status, 401, 'a revoked device token no longer authorizes')

    // An unpaired bearer is still rejected outright.
    const stranger = await fetch(`${base}/v1/snapshot`, { headers: { Authorization: 'Bearer not-a-token' } })
    assert.equal(stranger.status, 401)

    // The consumed-link retry above is one failed attempt. Four more failures reach the
    // per-IP threshold; the next request is rejected before another credential comparison.
    for (let i = 0; i < 4; i++) {
      const miss = await fetch(`${base}/v1/auth/pair`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ credential: `wrong-${i}` }) })
      assert.equal(miss.status, 400)
    }
    const throttled = await fetch(`${base}/v1/auth/pair`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ credential: 'wrong-final' }) })
    assert.equal(throttled.status, 429, 'failed pairing attempts are rate limited by source IP')
    assert.ok(Number(throttled.headers.get('retry-after')) >= 1, 'rate limit tells the client when to retry')
  } finally {
    await server.stop()
  }
})()

// Revocation is not acknowledged if the host's durable write fails. The live registry
// remains fail-closed (token revoked), while the HTTP caller receives an error rather than a
// false success that could later resurrect on restart.
await (async () => {
  const reg = createPairingRegistry({ newId: () => 'dev-durable', newToken: (() => {
    let n = 0
    return () => `tok-durable-${++n}`
  })() })
  const link = reg.issuePairingLink('Durable device')
  const paired = reg.redeemPairingLink(link.credential).session!
  const s: RemoteControlSettings = { ...settings, port: await freePort() }
  const durableServer = createRemoteControlServer(remoteDeps(s, {
    pairing: {
      issuePairingLink: (label) => reg.issuePairingLink(label),
      authorizeSessionToken: (token) => reg.authorizeSessionToken(token),
      redeemPairingLink: (credential, clientInfo) => reg.redeemPairingLink(credential, clientInfo),
      listSessions: () => reg.listSessions(),
      revokeSession: async (id) => {
        const revoked = reg.revokeSession(id)
        if (revoked) throw new Error('protected state write failed')
        return false
      },
    },
  }))
  await durableServer.syncFromSettings()
  try {
    const response = await withSecurityRole('owner', () => fetch(`${durableServer.status().url}/v1/auth/sessions/revoke`, {
      method: 'POST',
      headers: { Authorization: 'Bearer test-token', 'content-type': 'application/json' },
      body: JSON.stringify({ sessionId: paired.id }),
    }))
    assert.equal(response.status, 500, 'failed durable revoke is never acknowledged as success')
    assert.equal(reg.authorizeSessionToken(paired.token).ok, false, 'live token remains revoked after persistence failure')
  } finally {
    await durableServer.stop()
  }
})()

// Pairing hooks may include protected persistence. Both the owner issue API and the unauthenticated
// redeem route await those hooks; neither can acknowledge or expose a token while the write is
// pending, and a rejected write becomes a generic 500 rather than leaking persistence details.
await (async () => {
  let resolveIssue!: (value: { credential: string; expiresAt: number }) => void
  const issuePersist = new Promise<{ credential: string; expiresAt: number }>((resolve) => { resolveIssue = resolve })
  const s: RemoteControlSettings = { ...settings, port: await freePort() }
  const asyncServer = createRemoteControlServer(remoteDeps(s, {
    pairing: {
      issuePairingLink: () => issuePersist,
      authorizeSessionToken: () => ({ ok: false }),
      redeemPairingLink: async () => { throw new Error('protected state write failed: secret-token') },
      listSessions: () => [],
      revokeSession: async () => false,
    },
  }))

  let issueSettled = false
  const issued = asyncServer.issuePairingLink('Async device').then((value) => { issueSettled = true; return value })
  await Promise.resolve()
  assert.equal(issueSettled, false, 'issue does not return a credential before its async host hook settles')
  resolveIssue({ credential: 'durable-credential', expiresAt: 123 })
  assert.equal((await issued).credential, 'durable-credential')

  await asyncServer.syncFromSettings()
  try {
    const response = await fetch(`${asyncServer.status().url}/v1/auth/pair`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ credential: 'durable-credential' }),
    })
    assert.equal(response.status, 500, 'redeem persistence rejection is never acknowledged')
    const body = await response.text()
    assert.match(body, /remote_control_error/)
    assert.doesNotMatch(body, /secret-token|protected state/i, 'remote response does not expose persistence details')
  } finally {
    await asyncServer.stop()
  }
})()

// --- static web client + query-token + provider redaction (VST-035) ---
await (async () => {
  const { mkdtemp, mkdir, writeFile, rm } = await import('node:fs/promises')
  const { tmpdir } = await import('node:os')
  const path = await import('node:path')
  const dist = await mkdtemp(path.join(tmpdir(), 'vontra-webclient-'))
  await writeFile(path.join(dist, 'index.html'), '<html>vontra-web</html>')
  await mkdir(path.join(dist, 'assets'), { recursive: true })
  await writeFile(path.join(dist, 'assets', 'app.abcdef12.js'), 'console.log(1)')

  const served = createRemoteControlServer(remoteDeps({ ...settings, port: await freePort() }, {
    staticRoot: () => dist,
  }))
  await served.syncFromSettings()
  const servedBase = served.status().url
  try {
    // The shell is public: no token needed for the app code (like any hosted web app).
    const index = await fetch(`${servedBase}/`)
    assert.equal(index.status, 200)
    assert.match(await index.text(), /vontra-web/)
    assert.match(index.headers.get('content-type') ?? '', /text\/html/)
    assert.match(index.headers.get('content-security-policy') ?? '', /frame-ancestors 'none'/)
    assert.equal(index.headers.get('x-content-type-options'), 'nosniff')
    assert.equal(index.headers.get('x-frame-options'), 'DENY')
    assert.equal(index.headers.get('referrer-policy'), 'no-referrer')

    const spa = await fetch(`${servedBase}/settings`)
    assert.equal(spa.status, 200, 'client routes deep-link via the SPA fallback')

    const asset = await fetch(`${servedBase}/assets/app.abcdef12.js`)
    assert.equal(asset.status, 200)
    assert.match(asset.headers.get('cache-control') ?? '', /immutable/)

    const missingAsset = await fetch(`${servedBase}/assets/gone.js`)
    assert.equal(missingAsset.status, 404, 'missing assets 404 — never HTML')

    // The DATA API stays gated even while the shell is public.
    const bare = await fetch(`${servedBase}/v1/snapshot`)
    assert.equal(bare.status, 401, 'static serving never weakens the data gate')

    // Credentials in query strings are never accepted, including on the SSE route.
    const viaQuery = await fetch(`${servedBase}/v1/control-plane?token=test-token`)
    assert.equal(viaQuery.status, 401, 'query token does not authorize')
    const badQuery = await fetch(`${servedBase}/v1/control-plane?token=wrong`)
    assert.equal(badQuery.status, 401)
  } finally {
    await served.stop()
    await rm(dist, { recursive: true, force: true })
  }

  // Without a static root, GETs outside /v1 explain themselves instead of a bare 404.
  const bareServer = createRemoteControlServer(remoteDeps({ ...settings, port: await freePort() }))
  await bareServer.syncFromSettings()
  try {
    const root = await fetch(`${bareServer.status().url}/`)
    assert.equal(root.status, 404)
    assert.equal((await root.json()).error, 'web_client_not_served')
  } finally {
    await bareServer.stop()
  }
})()

// Provider redaction for the paired-client snapshot: PICKED fields only — a new secret
// field on Provider can never leak by default.
{
  const { redactProvidersForRemote } = await import('../main/host/remote-snapshot')
  const redacted = redactProvidersForRemote([
    {
      id: 'codex', name: 'Codex CLI', kind: 'cloud', transport: 'codex-cli', models: ['gpt-5.5'],
      apiKey: 'sk-SECRET', baseUrl: 'https://internal', serverPassword: 'hunter2',
      cliEnv: { OPENAI_API_KEY: 'sk-LEAK' }, cliProfile: '/private/profile.json',
    } as any,
  ])
  assert.equal(redacted.length, 1)
  assert.deepEqual(Object.keys(redacted[0]).sort(), ['defaultModel', 'id', 'kind', 'models', 'name', 'transport'])
  const flat = JSON.stringify(redacted)
  assert.ok(!flat.includes('SECRET') && !flat.includes('hunter2') && !flat.includes('LEAK') && !flat.includes('internal'), 'no credential material survives redaction')
  ok2('redactProvidersForRemote: picked identity fields only, secrets cannot leak')
}

function ok2(name: string) { console.log(`  \u2713 ${name}`) }

console.log('remote-control pairing/session tests passed')
