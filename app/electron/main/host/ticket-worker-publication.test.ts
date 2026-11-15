import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import ts from 'typescript'
import { normalizeTicketCapabilities, startTicketExecution, settleTicketExecution, ticketStateAfterRun, ticketWorkPrompt } from '../../agent/tickets'

const file = path.join(__dirname, 'ticket-worker-runtime.ts')
const source = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.ES2022, true)
const names = new Set(['assertActiveTicketWorkOwnership', 'awaitTicketStartupStep', 'createTicketWorkerRuntime'])
const declarations = source.statements.filter((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && !!node.name && names.has(node.name.text))
assert.equal(declarations.length, names.size, 'execute the production runtime, not a copied startup sequence')
const printed = declarations.map((node) => ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, source).replace(/^export /, '')).join('\n')
const compiled = ts.transpileModule(printed, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText

async function scenario(roomScoped: boolean, rendererAvailable = true) {
  const provider = { id: 'synthetic', name: 'Synthetic', kind: 'local', models: ['fixture'], baseUrl: 'http://127.0.0.1:9897/v1' }
  const project = { id: 'project', cwd: '/workspace/project', updatedAt: 1 }
  const room = { id: 'room', projectId: project.id, workspaceCwd: '/workspace/room-code', updatedAt: 1 }
  const ticket: any = {
    id: 'ticket', projectId: project.id, roomId: roomScoped ? room.id : null, title: 'Synthetic ticket', body: '',
    type: 'task', status: 'ready', priority: 'medium', effort: 'small', tags: [], order: 0, createdAt: 1, updatedAt: 1,
  }
  const store: any = { tickets: [ticket], projects: [project], rooms: roomScoped ? [room] : [], threads: [], settings: { providers: [provider], temperature: 0, reasoningEffort: 'low', approvalMode: 'readonly' } }
  const order: string[] = [], snapshots: any[] = [], persisted: any[] = []
  let finishRun: (() => void) | undefined, settled = 0, rejectWorkspace = false
  const aborters = new Map<string, AbortController>()
  const bindings: Record<string, unknown> = {
    selectedResponseLocale: () => 'en', randomUUID, path, store,
    runAgent: () => new Promise<void>((resolve) => { finishRun = resolve }),
    assertProviderEndpoint() {}, auditPreview: String, auditEvent: async () => {}, requireSecurityRole() {},
    normalizeTicketCapabilities, startTicketExecution, settleTicketExecution, ticketStateAfterRun, ticketWorkPrompt,
    restrictTicketToolContext: (context: unknown) => context,
    ensureRoomWorktree: async () => room.workspaceCwd,
    requireTicketDependenciesReady() {}, inspectGitRepository: async () => ({ branch: 'fixture' }),
    resolveWorkspaceCwd: async (cwd: string) => { if (rejectWorkspace) throw new Error('Synthetic workspace rejected'); return cwd },
    lexicalLocalWorkspaceKey: (cwd: string) => cwd,
    resolveProjectProvider: () => ({ baseProvider: provider, provider, model: 'fixture', providerAccountId: null }),
    ticketWorkOptionsWithProjectRoute: (_project: unknown, opts: unknown) => opts,
    modelRouter: { select: () => ({ selected: true, selectedAccountId: null, candidates: [], audit: { action: 'none' } }) },
    modelRouteAutoFailoverEnabled: () => false,
    summarizeTicketWorkerOutcome: () => ({ signals: { files: [], diffs: [], commands: [], blockers: [] }, evidence: [], text: 'Synthetic result' }),
    sendToRenderer: (channel: string, thread: any) => {
      order.push(channel)
      if (rendererAvailable) snapshots.push(structuredClone(thread))
      return rendererAvailable
    },
  }
  const create = new Function(...Object.keys(bindings), `${compiled}; return createTicketWorkerRuntime;`)(...Object.values(bindings))
  const runtime = create({
    persistThreads: () => { order.push('persist'); persisted.push(structuredClone(store.threads)) },
    persistTickets() {}, emitTicket: () => order.push('ticket'), appendRoomSupervisorEvent() {},
    claimJob: () => ({ id: 'job' }), patchJob: (job: unknown) => job, fullToolContext: () => ({}),
    drainThreadCues: () => [], waitForThreadCue: async () => {}, approvalResolver: () => async () => 'deny',
    timelineErrorSince: () => null, isHighRiskTool: () => false,
    emitToRenderer: () => order.push('agent'), addThreadNotice() {}, auditModelRoute: async () => {},
    aborters, workspaceMutationGuard: { assertThreadStartAllowed() {}, reserveStart: () => ({ addWorkspaceKeys() {}, addThreadId() {}, release() {} }) },
    applyEventToThread() {}, activeJob: () => null, finishJob: () => settled++, markUndrainedCues() {}, effectiveContextLimit: () => 4096,
  })
  const input = { ticketId: ticket.id, ...(roomScoped ? { source: { roomId: room.id, seat: 'Builder' } } : {}) }
  const first = await runtime.startTicketWork(input)
  assert.equal(order.filter((item) => item === 'thread:update').length, 1, 'a new worker publishes its thread without waiting for restart')
  assert.ok(order.indexOf('persist') < order.indexOf('thread:update'), 'publish only after adding and persisting the new thread')
  assert.ok(order.indexOf('thread:update') < order.indexOf('agent'), 'the renderer must know the thread before receiving timeline events')
  assert.equal(persisted[0][0].id, first.id)
  if (rendererAvailable) {
    assert.equal(snapshots.length, 1)
    assert.equal(snapshots[0].id, first.id)
    assert.equal(snapshots[0].projectId, project.id)
    assert.equal(snapshots[0].cwd, roomScoped ? room.workspaceCwd : project.cwd)
    assert.equal(snapshots[0].title, ticket.title)
  }
  assert.ok(finishRun, 'the worker still starts when the owner window is unavailable')
  finishRun!()
  for (let i = 0; i < 20 && settled < 1; i++) await Promise.resolve()
  assert.equal(settled, 1)
  assert.equal(aborters.size, 0)
  const second = await runtime.startTicketWork(input)
  assert.equal(second.id, first.id, 'resuming work reuses its existing thread')
  assert.equal(order.filter((item) => item === 'thread:update').length, 1, 'resuming does not publish duplicate creation snapshots')
  finishRun!()
  for (let i = 0; i < 20 && settled < 2; i++) await Promise.resolve()
  assert.equal(settled, 2)
  rejectWorkspace = true
  await assert.rejects(runtime.startTicketWork(input), /workspace rejected/)
  assert.equal(order.filter((item) => item === 'thread:update').length, 1, 'a rejected startup must not publish a phantom thread')
}

await scenario(false)
await scenario(true)
await scenario(true, false)
console.log('Ticket worker creation publishes before agent events for solo/room workers; reuse, startup rejection and missing renderer remain safe')
