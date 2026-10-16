// Run: jiti electron/agent/orchestrator-flow-smoke.test.ts
import assert from 'node:assert/strict'
import { createServer } from 'node:net'
import { createRemoteControlServer } from './remote-control'
import type {
  CliSession,
  Mission,
  Project,
  RemoteControlDoctorSnapshot,
  RemoteControlSettings,
  RemoteControlSnapshot,
  Room,
  Ticket,
  Thread,
} from './types'

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

const work = () => ({
  jobs: { total: 0, visible: 0, queued: 0, running: 0, waitingApproval: 0, paused: 0, stalled: 0, failed: 0, deadLetter: 0, retryable: 0 },
  leases: { active: 0, expired: 0, expiringSoon: 0, missingRunning: 0 },
  approvals: { pending: 0, tool: 0, connector: 0, mission: 0 },
  nextWakeAt: null,
  nextWakeJobId: null,
  generatedAt: Date.now(),
})

const settings: RemoteControlSettings = {
  enabled: true,
  serverName: 'TensorFold Studio Smoke Host',
  bindHost: '127.0.0.1',
  allowLan: false,
  port: await freePort(),
  authToken: 'smoke-token',
  lastStartedAt: null,
  lastClientAt: null,
  lastError: null,
}

const projects = new Map<string, Project>()
const rooms = new Map<string, Room>()
const threads = new Map<string, Thread>()
const tickets = new Map<string, Ticket>()
const missions = new Map<string, Mission>()
const terminals = new Map<string, CliSession>()
const terminalWrites: Array<{ sessionId: string; data: string }> = []
const terminalResizes: Array<{ sessionId: string; cols: number; rows: number }> = []
const threadLifecycle: string[] = []
const roomLifecycle: string[] = []
const ticketLifecycle: string[] = []
const missionLifecycle: string[] = []
const approvals: Array<{ callId: string; decision: 'allow' | 'allow_always' | 'deny' }> = []

let projectSeq = 0
let roomSeq = 0
let threadSeq = 0
let ticketSeq = 0
let missionSeq = 0
let terminalSeq = 0

function mustGet<T>(map: Map<string, T>, id: string, kind: string): T {
  const value = map.get(id)
  if (!value) throw Object.assign(new Error(`${kind} not found`), { code: 404 })
  return value
}

function makeMission(id: string, status: Mission['status'] = 'draft'): Mission {
  return {
    id,
    title: `Remote mission ${++missionSeq}`,
    objective: 'Smoke-test mission',
    status,
    checkpoints: [],
    journal: [],
    approvals: [],
    artifacts: [],
    createdAt: Date.now(),
    updatedAt: Date.now(),
  } as unknown as Mission
}

function makeSnapshot(): RemoteControlSnapshot {
  return {
    generatedAt: Date.now(),
    serverName: settings.serverName,
    appVersion: '1.0.0',
    headless: true,
    projects: Array.from(projects.values()).filter((item) => !item.archivedAt),
    rooms: Array.from(rooms.values()).filter((item) => !item.archivedAt),
    threads: Array.from(threads.values()).filter((item) => !item.archivedAt),
    tickets: Array.from(tickets.values()).filter((item) => !item.archivedAt),
    automations: [],
    missions: Array.from(missions.values()),
    jobs: [],
    connectorEvents: [],
    work: work(),
    activeThreadIds: Array.from(threads.values()).filter((thread) => thread.status === 'running').map((thread) => thread.id),
    activeRoomIds: Array.from(rooms.values()).filter((room) => room.state === 'running').map((room) => room.id),
    activeMissionIds: Array.from(missions.values()).filter((mission) => mission.status === 'running').map((mission) => mission.id),
  }
}

function makeDoctor(): RemoteControlDoctorSnapshot {
  const snapshot = makeSnapshot()
  return {
    generatedAt: snapshot.generatedAt,
    serverName: snapshot.serverName,
    appVersion: snapshot.appVersion,
    headless: snapshot.headless,
    counts: {
      threads: snapshot.threads.length,
      projects: snapshot.projects.length,
      rooms: snapshot.rooms.length,
      tickets: snapshot.tickets.length,
      automations: 0,
      missions: snapshot.missions.length,
      jobs: 0,
      connectorEvents: 0,
      activeThreads: snapshot.activeThreadIds.length,
      activeRooms: snapshot.activeRoomIds.length,
      activeMissions: snapshot.activeMissionIds.length,
    },
    work: snapshot.work,
  }
}

const server = createRemoteControlServer({
  getSettings: () => settings,
  getSnapshot: async () => makeSnapshot(),
  getArchives: async () => ({
    projects: Array.from(projects.values()).filter((item) => !!item.archivedAt),
    threads: Array.from(threads.values()).filter((item) => !!item.archivedAt),
    rooms: Array.from(rooms.values()).filter((item) => !!item.archivedAt),
    tickets: Array.from(tickets.values()).filter((item) => !!item.archivedAt),
  }),
  getPairedArchives: async () => ({
    projects: Array.from(projects.values()).filter((item) => !!item.archivedAt),
    threads: Array.from(threads.values()).filter((item) => !!item.archivedAt),
    rooms: Array.from(rooms.values()).filter((item) => !!item.archivedAt),
    tickets: Array.from(tickets.values()).filter((item) => !!item.archivedAt),
  }),
  getDoctor: async () => makeDoctor(),
  createProject: async (input) => {
    const id = `remote-project-${++projectSeq}`
    const project: Project = {
      id,
      name: String(input.name || `Remote Project ${projectSeq}`),
      cwd: typeof input.cwd === 'string' && input.cwd ? input.cwd : '/tmp/vontra-smoke',
      memory: typeof input.memory === 'string' ? input.memory : '',
      team: typeof input.team === 'string' ? input.team : '',
      executionTarget: 'orchestrator',
      orchestratorId: 'smoke-host',
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    projects.set(id, project)
    return project
  },
  updateProject: async (id, patch) => {
    const existing = mustGet(projects, id, 'project')
    const updated: Project = { ...existing, ...patch, id: existing.id, updatedAt: Date.now() }
    projects.set(id, updated)
    return updated
  },
  deleteProject: async (id) => {
    const existing = projects.get(id)
    if (!existing) return null
    const now = Date.now()
    const archived = { ...existing, archivedAt: now, updatedAt: now }
    projects.set(id, archived)
    for (const [roomId, room] of rooms) if (room.projectId === id) rooms.set(roomId, { ...room, archivedAt: now, updatedAt: now })
    for (const [threadId, thread] of threads) if (thread.projectId === id) threads.set(threadId, { ...thread, archivedAt: now, updatedAt: now })
    for (const [ticketId, ticket] of tickets) if (ticket.projectId === id) tickets.set(ticketId, { ...ticket, archivedAt: now, updatedAt: now })
    return archived
  },
  restoreProject: async (id) => {
    const existing = projects.get(id)
    if (!existing) return null
    const now = Date.now()
    const restored = { ...existing, archivedAt: null, updatedAt: now }
    projects.set(id, restored)
    for (const [roomId, room] of rooms) if (room.projectId === id) rooms.set(roomId, { ...room, archivedAt: null, updatedAt: now })
    for (const [threadId, thread] of threads) if (thread.projectId === id) threads.set(threadId, { ...thread, archivedAt: null, updatedAt: now })
    for (const [ticketId, ticket] of tickets) if (ticket.projectId === id) tickets.set(ticketId, { ...ticket, archivedAt: null, updatedAt: now })
    return restored
  },
  createRoom: async (input) => {
    mustGet(projects, input.projectId, 'project')
    const id = `remote-room-${++roomSeq}`
    const room: Room = {
      id,
      projectId: input.projectId,
      name: input.name || `Remote Room ${roomSeq}`,
      seats: [],
      problem: '',
      state: 'idle',
      events: [],
      finalText: '',
      createdAt: Date.now(),
      updatedAt: Date.now(),
    }
    rooms.set(id, room)
    roomLifecycle.push(`create:${id}`)
    return room
  },
  updateRoom: async (id, patch) => {
    const existing = mustGet(rooms, id, 'room')
    const updated: Room = { ...existing, ...patch, id: existing.id, projectId: existing.projectId, updatedAt: Date.now() }
    rooms.set(id, updated)
    roomLifecycle.push(`update:${id}:${updated.name}`)
    return updated
  },
  renameRoom: async (id, name) => {
    const room = mustGet(rooms, id, 'room')
    room.name = name
    room.updatedAt = Date.now()
    return room
  },
  steerRoom: async (id, text) => {
    const room = mustGet(rooms, id, 'room')
    const item = { id: `steer-${room.events.length + 1}`, tick: room.events.length + 1, kind: 'user', seat: 'you', target: 'all', text, createdAt: Date.now() }
    room.events.push(item)
    room.updatedAt = Date.now()
    return item
  },
  mutateRoomMemory: async (id, command) => {
    const room = mustGet(rooms, id, 'room')
    if (command.action === 'add') room.memory = [{ id: `memory-${Date.now()}`, kind: command.kind, text: command.text, source: 'manual', createdAt: Date.now(), updatedAt: Date.now() }, ...(room.memory ?? [])]
    else room.memory = (room.memory ?? []).filter((entry) => entry.id !== command.entryId)
    return room
  },
  deleteRoom: async (id) => {
    const existing = rooms.get(id)
    if (!existing) return null
    const now = Date.now()
    const archived = { ...existing, archivedAt: now, updatedAt: now }
    rooms.set(id, archived)
    for (const [ticketId, ticket] of tickets) if (ticket.roomId === id) tickets.set(ticketId, { ...ticket, archivedAt: now, updatedAt: now })
    roomLifecycle.push(`archive:${id}`)
    return archived
  },
  restoreRoom: async (id) => {
    const existing = rooms.get(id)
    if (!existing) return null
    const now = Date.now()
    const restored = { ...existing, archivedAt: null, updatedAt: now }
    rooms.set(id, restored)
    for (const [ticketId, ticket] of tickets) if (ticket.roomId === id) tickets.set(ticketId, { ...ticket, archivedAt: null, updatedAt: now })
    roomLifecycle.push(`restore:${id}`)
    return restored
  },
  startRoom: async (roomId, problem) => {
    const existing = mustGet(rooms, roomId, 'room')
    rooms.set(roomId, { ...existing, problem, state: 'running', updatedAt: Date.now() })
    roomLifecycle.push(`start:${roomId}:${problem}`)
    return { ok: true }
  },
  cancelRoom: async (roomId) => {
    const existing = mustGet(rooms, roomId, 'room')
    rooms.set(roomId, { ...existing, state: 'stopped', updatedAt: Date.now() })
    roomLifecycle.push(`cancel:${roomId}`)
  },
  createThread: async (input) => {
    if (input.projectId) mustGet(projects, input.projectId, 'project')
    const id = `remote-thread-${++threadSeq}`
    const thread: Thread = {
      id,
      title: input.kind === 'chat' ? 'Remote chat smoke' : 'Remote task smoke',
      kind: input.kind || 'task',
      projectId: input.projectId ?? null,
      cwd: input.cwd || (input.projectId ? projects.get(input.projectId)?.cwd : '') || '/tmp/vontra-smoke',
      workspace: null,
      providerId: null,
      providerAccountId: null,
      model: null,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      status: 'idle',
      apiMessages: [],
      timeline: [],
    }
    threads.set(id, thread)
    threadLifecycle.push(`create:${id}`)
    return thread
  },
  startThread: async (payload) => {
    const existing = mustGet(threads, payload.threadId, 'thread')
    threads.set(payload.threadId, { ...existing, status: 'running', updatedAt: Date.now() })
    threadLifecycle.push(`start:${payload.threadId}:${payload.userText}`)
    return { runId: `run-${payload.threadId}` }
  },
  guideThread: async (payload) => {
    mustGet(threads, payload.threadId, 'thread')
    threadLifecycle.push(`guide:${payload.threadId}:${payload.userText}`)
    return { queued: true, queueDepth: 1 }
  },
  cancelThread: async (threadId) => {
    const existing = mustGet(threads, threadId, 'thread')
    threads.set(threadId, { ...existing, status: 'idle', updatedAt: Date.now() })
    threadLifecycle.push(`cancel:${threadId}`)
  },
  deleteThread: async (id) => {
    const existing = threads.get(id)
    if (!existing) return null
    const archived = { ...existing, archivedAt: Date.now(), updatedAt: Date.now() }
    threads.set(id, archived)
    threadLifecycle.push(`archive:${id}`)
    return archived
  },
  restoreThread: async (id) => {
    const existing = threads.get(id)
    if (!existing) return null
    const restored = { ...existing, archivedAt: null, updatedAt: Date.now() }
    threads.set(id, restored)
    threadLifecycle.push(`restore:${id}`)
    return restored
  },
  approve: async (callId, decision) => {
    approvals.push({ callId, decision })
  },
  createTicket: async (input) => {
    mustGet(projects, input.projectId, 'project')
    const id = `remote-ticket-${++ticketSeq}`
    const ticket = {
      id,
      projectId: input.projectId,
      title: input.title || `Remote ticket ${ticketSeq}`,
      body: input.body || '',
      type: input.type || 'task',
      status: input.status || 'todo',
      priority: input.priority || 'medium',
      effort: input.effort || 'medium',
      tags: input.tags || [],
      parentId: input.parentId ?? null,
      roomId: input.roomId ?? null,
      createdAt: Date.now(),
      updatedAt: Date.now(),
    } as Ticket
    tickets.set(id, ticket)
    ticketLifecycle.push(`create:${id}`)
    return ticket
  },
  startTicketExecution: async (id, input) => {
    const ticket = mustGet(tickets, id, 'ticket')
    const thread = Array.from(threads.values()).find((candidate) => candidate.id === ticket.threadId) ?? {
      id: `ticket-thread-${id}`, title: ticket.title, kind: 'task' as const, projectId: ticket.projectId, cwd: '/tmp', workspace: null,
      providerId: null, providerAccountId: null, model: null, status: 'running' as const, apiMessages: [], timeline: [], createdAt: Date.now(), updatedAt: Date.now(),
    }
    threads.set(thread.id, thread)
    ticket.threadId = thread.id; ticket.status = 'in_progress'; ticket.runState = 'running'
    ticket.execution = { target: 'remote', hostId: input.hostId, threadId: thread.id, runId: 'ticket-run', timelineBaseline: 0, startedAt: Date.now(), finishedAt: null, checks: [], blockers: [], reviewVerdict: null }
    return { ticket, thread, runId: 'ticket-run' }
  },
  updateTicket: async (id, patch) => {
    const existing = mustGet(tickets, id, 'ticket')
    const updated = { ...existing, ...patch, id: existing.id, projectId: existing.projectId, updatedAt: Date.now() } as Ticket
    tickets.set(id, updated)
    ticketLifecycle.push(`update:${id}:${updated.status}`)
    return updated
  },
  transitionTicket: async (id, command) => {
    const ticket = mustGet(tickets, id, 'ticket')
    if (command.action === 'begin') {
      ticket.threadId = command.threadId
      ticket.status = 'in_progress'
      ticket.runState = 'queued'
    } else if (command.action === 'cancel') {
      ticket.status = 'needs_clarification'
      ticket.runState = 'cancelled'
    }
    ticket.updatedAt = Date.now()
    return ticket
  },
  deleteTicket: async (id) => {
    const existing = mustGet(tickets, id, 'ticket')
    const archived = { ...existing, archivedAt: Date.now(), updatedAt: Date.now() }
    tickets.set(id, archived)
    ticketLifecycle.push(`archive:${id}`)
    return archived
  },
  restoreTicket: async (id) => {
    const existing = mustGet(tickets, id, 'ticket')
    const restored = { ...existing, archivedAt: null, updatedAt: Date.now() }
    tickets.set(id, restored)
    ticketLifecycle.push(`restore:${id}`)
    return restored
  },
  setMissionStatus: async (id, status, message) => {
    const existing = missions.get(id) ?? makeMission(id)
    const updated = { ...existing, status, updatedAt: Date.now() } as Mission
    missions.set(id, updated)
    missionLifecycle.push(`status:${id}:${status}:${message ?? ''}`)
    return updated
  },
  recordMissionHeartbeat: async (id, message) => {
    const existing = missions.get(id) ?? makeMission(id, 'running')
    const updated = { ...existing, updatedAt: Date.now(), lastHeartbeatAt: Date.now() } as Mission
    missions.set(id, updated)
    missionLifecycle.push(`heartbeat:${id}:${message ?? ''}`)
    return updated
  },
  listDirectories: async (dir) => ({
    ok: true,
    path: dir || '/tmp',
    parent: dir && dir !== '/' ? '/tmp' : null,
    entries: [
      { name: 'Project A', path: `${dir || '/tmp'}/Project A`, isDirectory: true },
      { name: 'Project B', path: `${dir || '/tmp'}/Project B`, isDirectory: true },
    ],
  }),
  listClis: async () => [
    { id: 'shell', name: 'Shell', short: 'sh', command: null, color: '#888888', icon: 'terminal', builtin: true },
  ],
  startTerminal: (input) => {
    mustGet(projects, input.projectId, 'project')
    const id = `remote-terminal-${++terminalSeq}`
    const session: CliSession = {
      id,
      projectId: input.projectId,
      cliId: input.cliId,
      title: input.label || input.cliId,
      status: 'idle',
      createdAt: Date.now(),
      command: input.command,
      hostLocation: 'orchestrator',
      managedBy: 'external',
    }
    terminals.set(id, session)
    return session
  },
  writeTerminal: (sessionId, data) => {
    const existing = mustGet(terminals, sessionId, 'terminal')
    terminals.set(sessionId, { ...existing, status: 'busy' })
    terminalWrites.push({ sessionId, data })
  },
  resizeTerminal: (sessionId, cols, rows) => {
    mustGet(terminals, sessionId, 'terminal')
    terminalResizes.push({ sessionId, cols, rows })
  },
  renameTerminal: (sessionId, title) => {
    const existing = mustGet(terminals, sessionId, 'terminal')
    terminals.set(sessionId, { ...existing, title })
  },
  killTerminal: (sessionId) => {
    const existing = mustGet(terminals, sessionId, 'terminal')
    terminals.set(sessionId, { ...existing, status: 'done' })
  },
  listTerminals: (projectId) => Array.from(terminals.values()).filter((session) => session.projectId === projectId),
  onClientSeen: () => {},
  onStarted: () => {},
  onError: () => {},
  audit: () => {},
})

await server.syncFromSettings()
const base = server.status().url
assert.ok(base, 'smoke server should bind to an ephemeral loopback port')

const headers = {
  authorization: `Bearer ${settings.authToken}`,
  'content-type': 'application/json',
}
const previousSecurityRole = process.env.VONTRA_SECURITY_ROLE
process.env.VONTRA_SECURITY_ROLE = 'owner'

async function request<T>(method: 'GET' | 'POST', path: string, expectedStatus: number, body?: unknown): Promise<T> {
  const res = await fetch(`${base}${path}`, {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  const text = await res.text()
  const parsed = text ? JSON.parse(text) : null
  assert.equal(res.status, expectedStatus, `${method} ${path} returned ${res.status}: ${text}`)
  return parsed as T
}

try {
  const project = await request<Project>('POST', '/v1/projects', 201, {
    name: 'Remote smoke project',
    cwd: '/tmp/vontra-remote-smoke',
    memory: 'Smoke-test memory',
    team: 'Remote smoke team',
  })
  assert.equal(project.executionTarget, 'orchestrator')
  assert.equal(project.cwd, '/tmp/vontra-remote-smoke')

  const renamedProject = await request<Project>('POST', `/v1/projects/${project.id}/update`, 202, { name: 'Remote smoke project renamed' })
  assert.equal(renamedProject.name, 'Remote smoke project renamed')
  assert.equal(projects.get(project.id)?.name, 'Remote smoke project renamed')

  const files = await request<{ ok: true; path: string; entries: Array<{ name: string }> }>('GET', '/v1/files?dir=%2Ftmp%2Fvontra-remote-smoke', 200)
  assert.equal(files.path, '/tmp/vontra-remote-smoke')
  assert.deepEqual(files.entries.map((entry) => entry.name), ['Project A', 'Project B'])

  const eventController = new AbortController()
  const eventRes = await fetch(`${base}/v1/events`, { headers, signal: eventController.signal })
  assert.equal(eventRes.status, 200)
  const eventReader = eventRes.body!.getReader()
  // The SSE server writes the snapshot frame as three separate res.write() calls
  // (`id:`\n, `event:`\n, `data:`\n\n). Node 26's stream chunking can deliver the
  // `id:` line as its own first chunk before the rest is flushed, so accumulate
  // chunks until the complete snapshot frame (event line + JSON data line) arrives.
  let eventText = ''
  for (let i = 0; i < 20 && !(eventText.includes('event: snapshot') && eventText.includes('"serverName":"TensorFold Studio Smoke Host"')); i++) {
    const eventChunk = await eventReader.read()
    if (eventChunk.done) break
    eventText += Buffer.from(eventChunk.value ?? new Uint8Array()).toString('utf8')
  }
  assert.match(eventText, /event: snapshot/)
  assert.match(eventText, /"serverName":"TensorFold Studio Smoke Host"/)
  await eventReader.cancel().catch(() => {})
  eventController.abort()

  const clis = await request<Array<{ id: string }>>('GET', '/v1/clis', 200)
  assert.deepEqual(clis.map((cli) => cli.id), ['shell'])

  await request<{ error: string }>('POST', '/v1/terminals', 403, {
    projectId: project.id,
    cliId: 'shell',
    command: 'printf smoke',
  })

  const terminal = await request<CliSession>('POST', '/v1/terminals', 201, {
    projectId: project.id,
    cliId: 'shell',
    label: 'Smoke shell',
  })
  assert.equal(terminal.projectId, project.id)
  assert.equal(terminal.hostLocation, 'orchestrator')
  assert.equal(terminal.managedBy, 'external')
  await request<{ ok: true }>('POST', `/v1/terminals/${terminal.id}/write`, 202, { data: 'echo remote-client\n' })
  await request<{ ok: true }>('POST', `/v1/terminals/${terminal.id}/resize`, 202, { cols: 111.9, rows: 38.9 })
  await request<{ ok: true }>('POST', `/v1/terminals/${terminal.id}/rename`, 202, { title: 'Smoke shell renamed' })
  assert.deepEqual(terminalWrites, [{ sessionId: terminal.id, data: 'echo remote-client\n' }])
  assert.deepEqual(terminalResizes, [{ sessionId: terminal.id, cols: 111, rows: 38 }])
  assert.equal(terminals.get(terminal.id)?.title, 'Smoke shell renamed')
  const listedTerminals = await request<CliSession[]>('GET', `/v1/terminals?projectId=${project.id}`, 200)
  assert.equal(listedTerminals.length, 1)
  assert.equal(listedTerminals[0].status, 'busy')
  await request<{ ok: true }>('POST', `/v1/terminals/${terminal.id}/kill`, 202, {})
  assert.equal(terminals.get(terminal.id)?.status, 'done')

  const room = await request<Room>('POST', '/v1/rooms', 201, { projectId: project.id, name: 'Remote room smoke' })
  assert.equal(room.projectId, project.id)
  const renamedRoom = await request<Room>('POST', `/v1/rooms/${room.id}/update`, 202, { name: 'Remote room smoke renamed' })
  assert.equal(renamedRoom.name, 'Remote room smoke renamed')
  await request<{ ok: boolean }>('POST', `/v1/rooms/${room.id}/start`, 202, { problem: 'Exercise the remote room path.' })
  assert.equal(rooms.get(room.id)?.state, 'running')
  await request<{ ok: true }>('POST', `/v1/rooms/${room.id}/cancel`, 202, {})
  assert.equal(rooms.get(room.id)?.state, 'stopped')

  const ticket = await request<Ticket>('POST', '/v1/tickets', 201, { projectId: project.id, title: 'Remote ticket smoke', status: 'todo' })
  assert.equal(ticket.projectId, project.id)
  const updatedTicket = await request<Ticket>('POST', `/v1/tickets/${ticket.id}/update`, 202, { status: 'doing', roomId: room.id })
  assert.equal(updatedTicket.status, 'doing')
  assert.equal(updatedTicket.roomId, room.id)

  await request<{ ok: true }>('POST', '/v1/approvals/call-smoke', 202, { decision: 'allow' })
  assert.deepEqual(approvals, [{ callId: 'call-smoke', decision: 'allow' }])

  const mission = await request<Mission>('POST', '/v1/missions/mission-smoke/status', 202, { status: 'running', message: 'Smoke mission started.' })
  assert.equal(mission.status, 'running')
  await request<Mission>('POST', '/v1/missions/mission-smoke/heartbeat', 202, { message: 'Smoke heartbeat.' })
  const completedMission = await request<Mission>('POST', '/v1/missions/mission-smoke/status', 202, { status: 'completed', message: 'Smoke mission done.' })
  assert.equal(completedMission.status, 'completed')

  const thread = await request<Thread>('POST', '/v1/threads', 201, { projectId: project.id, cwd: project.cwd, kind: 'task' })
  assert.equal(thread.projectId, project.id)
  const started = await request<{ runId: string }>('POST', `/v1/threads/${thread.id}/start`, 202, { userText: 'Run the remote smoke task.' })
  assert.equal(started.runId, `run-${thread.id}`)
  await request<{ queued: true; queueDepth: number }>('POST', `/v1/threads/${thread.id}/guide`, 202, { userText: 'Capture terminal evidence.' })
  await request<{ ok: true }>('POST', `/v1/threads/${thread.id}/cancel`, 202, {})
  assert.deepEqual(threadLifecycle, [
    `create:${thread.id}`,
    `start:${thread.id}:Run the remote smoke task.`,
    `guide:${thread.id}:Capture terminal evidence.`,
    `cancel:${thread.id}`,
  ])

  const snapshot = await request<RemoteControlSnapshot>('GET', '/v1/snapshot', 200)
  assert.equal(snapshot.projects.length, 1)
  assert.equal(snapshot.rooms.length, 1)
  assert.equal(snapshot.threads.length, 1)
  assert.equal(snapshot.tickets.length, 1)
  assert.equal(snapshot.missions.length, 1)
  assert.deepEqual(snapshot.activeThreadIds, [])
  assert.deepEqual(snapshot.activeRoomIds, [])
  assert.deepEqual(snapshot.activeMissionIds, [])

  const archivedTicket = await request<Ticket>('POST', `/v1/tickets/${ticket.id}/delete`, 202, {})
  assert.ok(archivedTicket.archivedAt)
  assert.equal(tickets.has(ticket.id), true, 'ticket archive retains the record')
  assert.equal((await request<RemoteControlSnapshot>('GET', '/v1/snapshot', 200)).tickets.length, 0)
  await request<Ticket>('POST', `/v1/tickets/${ticket.id}/restore`, 202, {})
  assert.equal(tickets.get(ticket.id)?.archivedAt, null)

  const archivedRoom = await request<Room>('POST', `/v1/rooms/${room.id}/delete`, 202, {})
  assert.ok(archivedRoom.archivedAt)
  assert.equal(rooms.has(room.id), true, 'room archive retains the record')
  assert.ok(tickets.get(ticket.id)?.archivedAt, 'room archive cascades to room-scoped tickets')
  await request<Room>('POST', `/v1/rooms/${room.id}/restore`, 202, {})
  assert.equal(rooms.get(room.id)?.archivedAt, null)
  assert.equal(tickets.get(ticket.id)?.archivedAt, null)

  await request<Thread>('POST', `/v1/threads/${thread.id}/archive`, 202, {})
  assert.ok(threads.get(thread.id)?.archivedAt)
  assert.equal(threads.has(thread.id), true, 'thread archive retains the transcript-bearing record')
  await request<Thread>('POST', `/v1/threads/${thread.id}/restore`, 202, {})
  assert.equal(threads.get(thread.id)?.archivedAt, null)

  const archivedProject = await request<Project>('POST', `/v1/projects/${project.id}/delete`, 202, {})
  assert.ok(archivedProject.archivedAt)
  assert.equal(projects.has(project.id), true, 'project archive retains the project')
  assert.equal(threads.has(thread.id), true, 'project archive retains child threads/journals')
  assert.ok(threads.get(thread.id)?.archivedAt)
  assert.ok(rooms.get(room.id)?.archivedAt)
  assert.ok(tickets.get(ticket.id)?.archivedAt)
  const archivedSnapshot = await request<RemoteControlSnapshot>('GET', '/v1/snapshot', 200)
  assert.deepEqual([archivedSnapshot.projects.length, archivedSnapshot.threads.length, archivedSnapshot.rooms.length, archivedSnapshot.tickets.length], [0, 0, 0, 0])
  await request<Project>('POST', `/v1/projects/${project.id}/restore`, 202, {})
  const restoredSnapshot = await request<RemoteControlSnapshot>('GET', '/v1/snapshot', 200)
  assert.deepEqual([restoredSnapshot.projects.length, restoredSnapshot.threads.length, restoredSnapshot.rooms.length, restoredSnapshot.tickets.length], [1, 1, 1, 1], 'project archive round-trips the full ownership closure')

  assert.deepEqual(roomLifecycle, [
    `create:${room.id}`,
    `update:${room.id}:Remote room smoke renamed`,
    `start:${room.id}:Exercise the remote room path.`,
    `cancel:${room.id}`,
    `archive:${room.id}`,
    `restore:${room.id}`,
  ])
  assert.deepEqual(ticketLifecycle, [
    `create:${ticket.id}`,
    `update:${ticket.id}:doing`,
    `archive:${ticket.id}`,
    `restore:${ticket.id}`,
  ])
  assert.deepEqual(missionLifecycle, [
    'status:mission-smoke:running:Smoke mission started.',
    'heartbeat:mission-smoke:Smoke heartbeat.',
    'status:mission-smoke:completed:Smoke mission done.',
  ])

  const doctor = await request<RemoteControlDoctorSnapshot>('GET', '/v1/doctor', 200)
  assert.equal(doctor.counts.projects, 1)
  assert.equal(doctor.counts.rooms, 1)
  assert.equal(doctor.counts.threads, 1)
  assert.equal(doctor.counts.tickets, 1)
  assert.equal(doctor.counts.missions, 1)
  assert.equal(doctor.counts.activeMissions, 0)

  console.log('orchestrator client flow smoke test passed')
} finally {
  await server.stop()
  if (previousSecurityRole == null) delete process.env.VONTRA_SECURITY_ROLE
  else process.env.VONTRA_SECURITY_ROLE = previousSecurityRole
}
