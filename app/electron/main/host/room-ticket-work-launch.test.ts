// Behavioral tests for the room start_ticket_work launch path in
// workspace-runtime-ipc: when the worker launch throws, the pre-launch ticket
// mutation (workerStarts++, assignee, status, runState 'queued') is rolled back so
// the ticket stays retryable instead of wedging. Also covers the room delegate
// localOnly guard and the memory_write renderer notification.
import assert from 'node:assert/strict'
import os from 'node:os'
import { randomUUID } from 'node:crypto'
import { buildPrimer } from '../../agent/room-protocol'
import { createSubAgentFromLaunch, parentForRoomSeat, startSubAgent } from '../../agent/subagents'
import { createWorkspaceRuntimeIpc } from './workspace-runtime-ipc'

interface HarnessOpts {
  startTicketWork: (input: any) => Promise<any>
  localOnly?: boolean
  beforeWorkspace?: () => void
  isTicketStarting?: (id: string) => boolean
  onPersistTickets?: (tickets: unknown[]) => void
}

function buildHarness(opts: HarnessOpts) {
  const store: any = {
    subAgents: [],
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
  const delegated: any[] = []
  const remote: any[] = []
  const aborters = new Map()
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
    aborters,
    isTicketStarting: opts.isTicketStarting,
    workspaceMutationGuard: { reserveStart: () => ({ addWorkspaceKeys: () => {}, release: () => {} }) },
    ensureRoomWorktree: async () => { opts.beforeWorkspace?.(); return os.tmpdir() },
    randomUUID, createSubAgentFromLaunch, parentForRoomSeat, startSubAgent,
    runReadOnlySubAgent: async (agent: any, provider: any, model: string, input: any) => {
      delegated.push({ provider: provider.id, model, ...input })
      return { agent, output: 'Synthetic delegated result' }
    },
    persistRooms: () => {},
    persistTickets: () => { persistTicketsCalls++; opts.onPersistTickets?.(store.tickets) },
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
    api, store, emittedTickets, remote, delegated, aborters,
    get runDeps() { return runDeps },
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

  {
    const workerSources: any[] = []
    const h = buildHarness({ startTicketWork: async (input) => { workerSources.push(input.source); return { id: 'worker' } }, beforeWorkspace: () => {
      h.store.settings.reasoningEffort = 'medium'
      for (const seat of h.store.rooms[0].seats) seat.reasoningEffort = 'medium'
    } })
    h.store.settings.reasoningEffort = 'xhigh'
    h.store.rooms[0].seats[0].reasoningEffort = 'low'
    h.store.rooms[0].seats.push({ id: 'reviewer', name: 'Reviewer', role: 'Review', providerId: 'p-local', model: 'm1', reasoningEffort: 'high' },
      { id: 'legacy', name: 'Legacy', role: 'Review', providerId: 'p-local', model: 'm1' })
    await h.run()
    assert.deepEqual(h.store.rooms[0].seats.map((seat: any) => h.runDeps.reasoningEffortForSeat(seat)), ['low', 'high', 'xhigh'],
      'the actual host passes the same captured values into the Room turn engine')
    await h.tool('delegate', { task: 'Review synthetic input' }, 'Builder')
    await h.tool('delegate', { task: 'Review synthetic input', targetSeat: 'Reviewer' }, 'Builder')
    await h.tool('delegate', { task: 'Review synthetic input', targetSeat: 'Legacy' }, 'Builder')
    assert.deepEqual(h.delegated.map(call => call.reasoningEffort), ['low', 'high', 'xhigh'],
      'actual Room delegation captures origin, target and legacy app effort before workspace awaits')
    await h.tool('start_ticket_work', { id: 't1', phase: 'build' }, 'Builder')
    assert.equal(workerSources[0]?.reasoningEffort, 'low', 'ticket workers inherit the same Room run capture')
    console.log('ok - actual Room delegation forwards independent captured efforts')
  }
  {
    let starting = false
    let savedTickets = ''
    const first = buildHarness({
      startTicketWork: async () => { throw new Error('Release must not start a worker') },
      onPersistTickets: (tickets) => { savedTickets = JSON.stringify(tickets) },
    })
    const parked = first.store.tickets[0]
    parked.threadId = 'previous-thread'
    parked.evidence = [{ id: 'existing-evidence', summary: 'Prior evidence' }]
    parked.execution = { history: ['existing-record'] }
    await first.run()
    await first.tool('claim_ticket', { id: parked.id }, 'Builder')

    const h = buildHarness({
      startTicketWork: async () => { throw new Error('Release must not start a worker') },
      isTicketStarting: (id) => id === 't1' && starting,
    })
    const room = h.store.rooms[0]
    h.store.tickets = JSON.parse(savedTickets)
    room.seats.push({ ...room.seats[0], id: 's2', name: 'Reviewer', role: 'reviewer' })
    await h.run()
    const ticket = h.store.tickets[0]
    const selector = { id: ticket.id }
    assert.match(await h.tool('claim_ticket', selector, 'Reviewer'), /Already claimed by Builder/)

    async function rejectRelease(expected: RegExp, args: { id?: string; title?: string } = selector, seat = 'Builder') {
      const previous = structuredClone(ticket)
      const saves = h.persistTicketsCalls
      const updates = h.emittedTickets.length
      assert.match(await h.tool('release_ticket', args, seat), expected)
      assert.deepEqual(ticket, previous, 'a refused release leaves the stored ticket untouched')
      assert.equal(h.persistTicketsCalls, saves)
      assert.equal(h.emittedTickets.length, updates)
    }

    await rejectRelease(/Only Builder/, selector, 'Reviewer')
    await rejectRelease(/uniquely configured/, selector, 'Removed')
    await rejectRelease(/No matching/, { id: 'missing', title: ticket.title })
    h.store.tickets.push({ ...ticket, id: 'other', roomId: 'other-room' })
    await rejectRelease(/No matching/, { id: 'other' })
    assert.equal(h.store.tickets[1].assignee, 'Builder')
    room.seats.push({ ...room.seats[0], id: 'conflicting', name: 'builder' })
    await rejectRelease(/uniquely configured/)
    room.seats.pop()

    for (const state of ['queued', 'running']) {
      ticket.runState = state
      await rejectRelease(/active ticket run/)
    }
    ticket.runState = null
    h.aborters.set(ticket.threadId, {})
    await rejectRelease(/active ticket run/)
    h.aborters.clear()
    starting = true
    await rejectRelease(/active ticket run/)
    starting = false
    ticket.archivedAt = 1
    await rejectRelease(/No matching/)
    ticket.archivedAt = null
    ticket.status = 'completed'
    await rejectRelease(/completed/)
    ticket.status = 'in_progress'
    ticket.updatedAt = 1

    const expected = { ...structuredClone(ticket), assignee: null, status: 'ready' }
    const saves = h.persistTicketsCalls
    const updates = h.emittedTickets.length
    assert.match(await h.tool('release_ticket', { title: ticket.title }, 'Builder'), /Released/)
    assert.deepEqual(ticket, { ...expected, updatedAt: ticket.updatedAt })
    assert.ok(ticket.updatedAt > expected.updatedAt)
    assert.equal(h.persistTicketsCalls, saves + 1)
    assert.equal(h.emittedTickets.length, updates + 1)
    assert.deepEqual(h.emittedTickets.at(-1), ticket)
    await rejectRelease(/no claim to release/)
    assert.match(await h.tool('claim_ticket', selector, 'Reviewer'), /You claimed/)
    assert.equal(ticket.assignee, 'Reviewer')
    assert.match(buildPrimer(h.store.projects[0], room, room.seats[0], room.seats), /TOOL release_ticket/)
    console.log('ok - only the configured owner can release an idle Room claim without losing its history')
  }
  console.log('room ticket work launch tests passed')
}

main().catch((e) => { console.error(e); process.exit(1) })
