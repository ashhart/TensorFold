// Behavioral tests for the room start_ticket_work launch path in
// workspace-runtime-ipc: when the worker launch throws, the pre-launch ticket
// mutation (workerStarts++, assignee, status, runState 'queued') is rolled back so
// the ticket stays retryable instead of wedging. Also covers the room delegate
// localOnly guard and the memory_write renderer notification.
import assert from 'node:assert/strict'
import os from 'node:os'
import { createWorkspaceRuntimeIpc } from './workspace-runtime-ipc'

interface HarnessOpts {
  startTicketWork: (input: any) => Promise<any>
  localOnly?: boolean
}

function buildHarness(opts: HarnessOpts) {
  const store: any = {
    projects: [{ id: 'p1', name: 'P', cwd: os.tmpdir(), workspace: null, localOnly: opts.localOnly === true, archivedAt: null }],
    rooms: [{
      id: 'room-1', projectId: 'p1', name: 'Room', problem: '', state: 'idle', archivedAt: null,
      seats: [{ id: 's1', name: 'Builder', role: 'builder', providerId: 'p-local', providerAccountId: null, model: 'm1' }],
      events: [], memory: [], finalText: '', createdAt: 1, updatedAt: 1,
    }],
    tickets: [{
      id: 't1', projectId: 'p1', roomId: 'room-1', title: 'Build the thing', body: '', type: 'task',
      status: 'ready', priority: 'medium', tags: [], risk: 'low', assignee: null, runState: null,
      evidence: [], order: 1, createdAt: 1, updatedAt: 1,
    }],
    settings: {
      providers: [
        { id: 'p-local', name: 'Local', kind: 'local', models: ['m1'] },
        { id: 'p-cloud', name: 'Cloud', kind: 'cloud', models: ['gpt-5'] },
      ],
      autoCompact: false, temperature: 0.2, reasoningEffort: 'medium',
    },
  }

  const emittedTickets: any[] = []
  const remote: any[] = []
  let persistTicketsCalls = 0
  let runDeps: any = null
  let runDone: Promise<void> | null = null

  const base: Record<string, any> = {
    store,
    handle: () => {},
    requireSecurityRole: () => {},
    auditEvent: async () => {},
    auditPreview: (s: string) => s,
    roomAborters: new Map(),
    workspaceMutationGuard: { reserveStart: () => ({ addWorkspaceKeys: () => {}, release: () => {} }) },
    ensureRoomWorktree: async () => os.tmpdir(),
    persistRooms: () => {},
    persistTickets: () => { persistTicketsCalls++ },
    emitRoom: () => {},
    emitTicket: (t: any) => emittedTickets.push({ ...t }),
    emitJob: () => {},
    publishRemote: (event: any) => remote.push(event),
    claimJob: () => ({ id: 'job-1', kind: 'room' }),
    patchJob: (job: any) => job,
    ensureRoomSupervisorMission: () => ({ id: 'm-1' }),
    buildRoomSupervisorJobMetadata: () => ({}),
    roomWorkerBudget: () => ({ maxActive: 2, maxStarts: 4 }),
    patchRoomSupervisorJob: (job: any) => job,
    finishRoomSupervisorMission: () => {},
    finishJob: () => {},
    ticketsForRoom: (room: any) => store.tickets.filter((t: any) => t.roomId === room.id),
    runtimeProvider: (p: any) => p,
    roomTicketConflictBlockers: () => [],
    ticketFileRefs: () => [],
    normalizeStringList: (v: any, d: any) => (Array.isArray(v) ? v.filter((x) => typeof x === 'string') : d),
    normalizeTicketCapabilities: (v: any, d: any) => (Array.isArray(v) && v.length ? v : d),
    describeTicketForRoom: () => '',
    summarizeRoomLeftToWork: () => '',
    activeJob: () => null,
    parkRoomIfWorkRemains: () => false,
    startTicketWork: opts.startTicketWork,
    runRoom: (deps: any) => {
      runDeps = deps
      runDone = Promise.resolve()
      return runDone
    },
    registerWebhookIpc: () => {},
  }
  const ctx = new Proxy(base, {
    get(target, prop: string) {
      if (prop in target) return target[prop]
      return () => undefined
    },
  })
  const api = createWorkspaceRuntimeIpc(ctx as any)
  return {
    api, store, emittedTickets, remote,
    get persistTicketsCalls() { return persistTicketsCalls },
    // Start the room (which captures the live room tools), then settle the run.
    run: async () => {
      const result = await api.startRoomRun('room-1', 'build the feature')
      await runDone
      await new Promise((resolve) => setImmediate(resolve))
      return result
    },
    tool: (name: string, args: any, seat: string) => {
      if (!runDeps?.tools?.[name]) throw new Error(`room tool ${name} unavailable; run() first`)
      return runDeps.tools[name](args, seat) as Promise<string>
    },
  }
}

async function main() {
  // 1) LAUNCH THROW — the ticket mutation is rolled back: runState cleared, status and
  //    assignee restored, ticket re-persisted, and the tool returns a clear error.
  {
    const h = buildHarness({
      startTicketWork: async () => { throw new Error('worker spawn exploded') },
    })
    const result = await h.run()
    assert.equal(result.ok, true, 'room run itself still starts')

    const out = await h.tool('start_ticket_work', { id: 't1', phase: 'build' }, 'Builder')
    assert.match(out, /launch failed/, 'tool error explains the launch failure')
    assert.match(out, /worker spawn exploded/)
    const ticket = h.store.tickets[0]
    assert.equal(ticket.runState, null, 'runState reset so retries are not blocked')
    assert.equal(ticket.status, 'ready', 'status restored to its pre-launch value')
    assert.equal(ticket.assignee, null, 'assignee restored to its pre-launch value')
    assert.ok(h.persistTicketsCalls >= 2, 'rollback re-persisted the ticket')
    const lastEmit = h.emittedTickets[h.emittedTickets.length - 1]
    assert.equal(lastEmit.runState, null, 'renderer sees the rolled-back ticket')
    console.log('ok - failed worker launch rolls the ticket back to a retryable state')
  }

  // 2) RETRY AFTER ROLLBACK — the same ticket starts cleanly on a later attempt:
  //    the worker budget was not burned and no stale 'queued' runState blocks the gate.
  {
    let attempts = 0
    const h = buildHarness({
      startTicketWork: async () => {
        attempts++
        if (attempts === 1) throw new Error('transient launch failure')
        return { id: 'thread-9' }
      },
    })
    await h.run()
    const first = await h.tool('start_ticket_work', { id: 't1', phase: 'build' }, 'Builder')
    assert.match(first, /launch failed/)
    assert.equal(h.store.tickets[0].runState, null)

    const second = await h.tool('start_ticket_work', { id: 't1', phase: 'build' }, 'Builder')
    assert.equal(attempts, 2, 'the retry reached the launcher')
    assert.match(second, /thread-9/, 'retry reports the new worker thread')
    assert.equal(h.store.tickets[0].runState, 'queued')
    assert.equal(h.store.tickets[0].assignee, 'Builder')
    console.log('ok - ticket is retryable after a rolled-back launch')
  }

  // 3) ROOM DELEGATE — a local-only project cannot delegate to a cloud provider.
  {
    const h = buildHarness({ startTicketWork: async () => ({ id: 'thread-1' }), localOnly: true })
    await h.run()
    const out = await h.tool('delegate', { task: 'scout the repo', providerId: 'p-cloud' }, 'Builder')
    assert.match(out, /Cannot delegate/, 'cloud delegation refused for a local-only project')
    assert.match(out, /must be a local provider/)
    console.log('ok - room delegate rejects a cloud route for a local-only project')
  }

  // 4) memory_write — a changed write publishes the room to remote subscribers, a
  //    redundant write does not, and oversized content is a clear tool error.
  {
    const h = buildHarness({ startTicketWork: async () => ({ id: 'thread-1' }) })
    await h.run()
    const out = await h.tool('memory_write', { name: 'decisions', content: 'use sqlite' }, 'Builder')
    assert.match(out, /updated \(rev 1\)/)
    assert.equal(h.remote.filter((e) => e.kind === 'room').length, 1, 'changed write publishes the room')

    const again = await h.tool('memory_write', { name: 'decisions', content: 'use sqlite' }, 'Builder')
    assert.match(again, /already says that/)
    assert.equal(h.remote.filter((e) => e.kind === 'room').length, 1, 'redundant write does not re-publish')

    const big = await h.tool('memory_write', { name: 'plan', content: 'x'.repeat(9000) }, 'Builder')
    assert.match(big, /too large/, 'oversized content is a clear tool error')
    console.log('ok - memory_write notifies the renderer on change and rejects oversize content')
  }

  console.log('room ticket work launch tests passed')
}

main().catch((e) => { console.error(e); process.exit(1) })
