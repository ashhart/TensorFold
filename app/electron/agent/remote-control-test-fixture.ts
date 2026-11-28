import { createServer } from 'node:net'
import type { RemoteControlServerDeps } from './remote-control'
import type { RemoteControlSettings } from './types'

export async function freePort(): Promise<number> {
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

export const settings: RemoteControlSettings = {
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

export const approvals: Array<{ callId: string; decision: 'allow' | 'allow_always' | 'deny' }> = []
export const ticketDeletes: string[] = []

export function remoteDeps(s: RemoteControlSettings, deps: Partial<RemoteControlServerDeps> = {}): RemoteControlServerDeps {
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

export function ok2(name: string) { console.log(`  \u2713 ${name}`) }
