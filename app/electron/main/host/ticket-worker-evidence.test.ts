import assert from 'node:assert/strict'
import { ticketReviewGateState } from '../../agent/tickets'
import type { AgentEvent, Job, Thread, Ticket, TimelineItem } from '../../agent/types'
import { store } from '../state'
import { createTicketWorkerRuntime } from './ticket-worker-runtime'

const oldDiff = { id: 'old-diff', kind: 'tool', path: 'src/parser.ts', state: 'ok', diff: [{ type: 'add', text: 'const value = 1' }] } as TimelineItem
const oldTest = { id: 'old-test', kind: 'tool', command: 'npm test', output: '2 passed', state: 'ok' } as TimelineItem
const currentRead = { id: 'current-read', kind: 'tool', path: 'src/parser.ts', state: 'ok' } as TimelineItem
const currentTest = { id: 'current-test', kind: 'tool', command: 'npm test', output: '3 passed', state: 'ok' } as TimelineItem

interface Scenario {
  phase?: 'build' | 'review'
  rewrite?: 'trim' | 'replace' | 'reverse' | 'remove-current'
  interrupted?: boolean
}

function applyEvent(thread: Thread, event: AgentEvent): void {
  if (event.type === 'item_add') thread.timeline.push(structuredClone(event.item))
  else if (event.type === 'item_update') Object.assign(thread.timeline.find((item) => item.id === event.id) ?? {}, event.patch)
  else if (event.type === 'item_delta') {
    const item = thread.timeline.find((candidate) => candidate.id === event.id)
    if (item) item[event.field] = (item[event.field] ?? '') + event.chunk
  }
}

let passed = 0
async function scenario(previous: TimelineItem[], current: TimelineItem[], options: Scenario = {}) {
  const saved = { tickets: store.tickets, projects: store.projects, rooms: store.rooms, threads: store.threads, settings: store.settings }
  const phase = options.phase ?? 'review'
  const thread = {
    id: 'reused-thread', projectId: 'project', title: 'Inspect parser', kind: 'task', cwd: '/workspace/project',
    providerId: 'fixture', model: 'fixture',
    status: 'idle', timeline: structuredClone(previous), apiMessages: [], createdAt: 1, updatedAt: 1,
  } as Thread
  const ticket = {
    id: 'ticket', projectId: 'project', threadId: thread.id, title: 'Inspect parser', body: '', type: 'task',
    status: 'review', priority: 'medium', effort: 'small', tags: [], order: 0, createdAt: 1, updatedAt: 1,
    parentId: null, assignee: 'Builder', runState: null,
    risk: 'high', allowedCapabilities: ['read'],
    evidence: [{ id: 'builder', kind: 'file', summary: 'Inspected src/parser.ts.', createdBy: 'Builder', createdAt: 10 }],
  } as Ticket
  const provider = { id: 'fixture', name: 'Fixture', kind: 'local', models: ['fixture'], baseUrl: 'http://127.0.0.1:9897/v1', apiKey: '' } as const
  const aborters = new Map<string, AbortController>()
  let released = false, outcome: string | null = null, savedTimeline: TimelineItem[] = [], published = 0
  let settled!: () => void
  const completion = new Promise<void>((resolve) => { settled = resolve })
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    store.tickets = [ticket]
    store.projects = [{ id: 'project', name: 'Fixture project', cwd: thread.cwd, memory: '', team: '', createdAt: 1, updatedAt: 1 }]
    store.rooms = []
    store.threads = [thread]
    store.settings = { ...saved.settings, providers: [{ ...provider, models: [...provider.models] }], activeProviderId: provider.id, activeModel: 'fixture' }
    const runtime = createTicketWorkerRuntime({
      runAgent: async ({ emit, thread: active }) => {
        const base = { runId: 'fixture-run', threadId: active.id }
        for (const item of current) emit({ ...base, type: 'item_add', item: structuredClone(item) })
        if (options.rewrite === 'trim') active.timeline.splice(0, previous.length)
        if (options.rewrite === 'replace') active.timeline = []
        if (options.rewrite === 'reverse') active.timeline.reverse()
        if (options.rewrite === 'remove-current') {
          active.timeline = active.timeline.filter((item) => item.id !== currentTest.id)
          emit({ ...base, type: 'item_update', id: currentTest.id, patch: { state: 'ok', output: '3 ' } })
          emit({ ...base, type: 'item_delta', id: currentTest.id, field: 'output', chunk: 'passed' })
        }
        emit({ ...base, type: 'item_update', id: oldDiff.id, patch: { state: 'ok' } })
        if (options.interrupted) aborters.get(active.id)!.abort()
        savedTimeline = structuredClone(active.timeline)
      },
      resolveWorkspaceCwd: async () => '/workspace/project',
      inspectGitRepository: async () => ({ branch: 'fixture' }) as any,
      persistThreads() {}, persistTickets() {}, emitTicket() { published++ }, appendRoomSupervisorEvent() {},
      claimJob: () => ({ id: 'job' }) as Job, patchJob: (job) => job, fullToolContext: () => ({}),
      drainThreadCues: () => [], waitForThreadCue: async () => {}, approvalResolver: () => async () => 'deny',
      timelineErrorSince: (active, start) => active.timeline.slice(start).find((item) => item.kind === 'error') ?? null,
      timelineError: () => new Error('Synthetic agent error'), isHighRiskTool: () => false, emitToRenderer() {},
      addThreadNotice() {}, auditModelRoute: async () => {}, markRouteErrorRecovered() {}, injectThreadRouteHandoff: () => '',
      aborters, workspaceMutationGuard: {
        assertThreadStartAllowed() {},
        reserveStart: () => ({ addWorkspaceKeys() {}, addThreadId() {}, release() { released = true } }),
      },
      applyEventToThread: applyEvent, activeJob: () => null,
      finishJob: (_job, result) => { outcome = result; settled() }, markUndrainedCues() {}, effectiveContextLimit: () => 4096,
    })
    assert.equal(await runtime.startTicketWork({ ticketId: ticket.id, opts: { phase } }), thread)
    await Promise.race([completion, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('Worker settlement timed out')), 2000) })])
    assert.ok(released, 'the production finalizer releases its run reservation')
    assert.equal(aborters.size, 0, 'the production finalizer clears the active worker')
    assert.equal(published, 2, 'startup and settlement both publish the ticket')
    assert.equal(runtime.isTicketStarting(ticket.id), false)
    assert.deepEqual(thread.timeline, savedTimeline, 'settlement leaves live conversation history intact')
    passed++
    return { ticket, outcome }
  } finally {
    clearTimeout(timer)
    Object.assign(store, saved)
  }
}

const clean = await scenario([oldDiff, oldTest], [currentRead, currentTest])
assert.equal(clean.ticket.evidence!.filter((item) => item.kind === 'diff').length, 0, 'old diffs are not reviewer-made changes')
assert.equal(clean.ticket.status, 'completed')
assert.equal(clean.outcome, 'completed')
assert.equal(ticketReviewGateState(clean.ticket).hasReviewOrSkipAfterBuilder, true)
assert.equal(ticketReviewGateState(clean.ticket).builderEvidenceBy, 'Builder')

for (const rewrite of ['trim', 'replace', 'reverse', 'remove-current'] as const) {
  const result = await scenario([oldDiff, oldTest], [currentRead, currentTest], { rewrite })
  assert.deepEqual(result.ticket.execution!.checks, [{ command: 'npm test', passed: true }], `${rewrite} retains this run's verification`)
  assert.ok(!result.ticket.evidence!.some((item) => item.kind === 'diff'), `${rewrite} excludes historical diffs`)
}

const recovered = await scenario([{ id: 'old-error', kind: 'error', text: 'Previous run failed.' } as TimelineItem], [currentRead, currentTest])
assert.equal(recovered.ticket.status, 'completed', 'old errors do not fail this review')

const noCurrentTest = await scenario([oldTest], [currentRead])
assert.ok(!noCurrentTest.ticket.evidence!.some((item) => item.kind === 'test'))
assert.deepEqual(noCurrentTest.ticket.execution!.checks, [])

const changed = await scenario([], [{ ...oldDiff, id: 'current-diff' }, currentTest])
assert.ok(changed.ticket.evidence!.some((item) => item.kind === 'diff'))
assert.equal(ticketReviewGateState(changed.ticket).hasReviewOrSkipAfterBuilder, false, 'reviewer-made changes need independent review')

const failed = await scenario([], [{ id: 'current-error', kind: 'error', text: 'This review failed.' } as TimelineItem], { rewrite: 'replace' })
assert.equal(failed.ticket.status, 'needs_clarification')
assert.equal(failed.outcome, 'failed')
assert.ok(failed.ticket.evidence!.some((item) => item.kind === 'review' && item.summary.includes('blocked')))

const rebuilt = await scenario([oldDiff, oldTest], [currentRead], { phase: 'build' })
assert.equal(rebuilt.ticket.status, 'review')
assert.equal(rebuilt.outcome, 'completed')
assert.ok(!rebuilt.ticket.evidence!.some((item) => item.kind === 'test' || item.kind === 'diff'))

const interrupted = await scenario([oldTest], [currentRead], { interrupted: true })
assert.equal(interrupted.ticket.status, 'in_progress')
assert.equal(interrupted.ticket.runState, null)
assert.equal(interrupted.outcome, 'paused')
console.log(`ticket worker evidence: ${passed} exported-runtime scenarios passed`)
