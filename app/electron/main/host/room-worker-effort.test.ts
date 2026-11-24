import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import { randomUUID } from 'node:crypto'
import ts from 'typescript'
import * as tickets from '../../agent/tickets'
import { restrictTicketToolContext } from '../../agent/ticket-tool-policy'
import { requireTicketDependenciesReady } from '../../../shared/ticket-dependencies'
import { lexicalLocalWorkspaceKey } from './workspace-mutation-guard'
import * as reasoning from '../../../shared/reasoning-effort'
import * as botPolicy from '../../../shared/bot-policy'

const code = ts.transpileModule(readFileSync(path.join(__dirname, 'ticket-worker-runtime.ts'), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true },
}).outputText
const observed: unknown[] = []
const expected: unknown[] = []
for (const [global, configured, sourceEffort] of [['high', 'low', undefined], ['low', 'high', undefined],
  ['high', undefined, undefined], ['high', 'medium', 'low']] as const) {
  for (const recover of [false, true]) {
    let releaseWorkspace!: (cwd: string) => void
    const workspace = new Promise<string>(resolve => { releaseWorkspace = resolve })
    let releaseAgent!: () => void
    const agent = new Promise<void>(resolve => { releaseAgent = resolve })
    let finish!: () => void
    const done = new Promise<void>(resolve => { finish = resolve })
    const provider = { id: 'fixture', name: 'Fixture', kind: 'local', baseUrl: 'http://127.0.0.1:1/v1', models: ['fixture'] }
    const seat: any = { id: 'builder', name: 'Builder', role: 'Review input', providerId: provider.id,
      model: 'fixture', reasoningEffort: configured }
    const ticket: any = { id: 'ticket', projectId: 'project', roomId: 'room', title: 'Review input', body: '',
      type: 'task', status: 'ready', priority: 'medium', effort: 'small', tags: [], parentId: null,
      assignee: null, threadId: null, allowedCapabilities: ['read'], requiredEvidence: [], acceptanceCriteria: [],
      dependsOn: [], order: 1, createdAt: 1, updatedAt: 1 }
    const store: any = {
      projects: [{ id: 'project', name: 'Synthetic project', cwd: '/workspace/project' }],
      rooms: [{ id: 'room', projectId: 'project', seats: [seat], state: 'running' }], tickets: [ticket], threads: [],
      settings: { providers: [provider], reasoningEffort: global, approvalMode: 'suggest', temperature: 0 },
    }
    const routeEfforts: unknown[] = []
    const requests: any[] = []
    const imports: Record<string, any> = {
      'node:crypto': { randomUUID }, 'node:path': path,
      '../../agent/response-language': { selectedResponseLocale: () => 'en' },
      '../../agent/loop': { runAgent: async (params: any) => {
        requests.push(params)
        if (recover && requests.length === 1) throw new Error('Synthetic provider limit')
        await agent
      } },
      '../../agent/security': { assertProviderEndpoint() {}, auditPreview: (value: string) => value },
      '../../agent/tickets': tickets, '../../agent/ticket-tool-policy': { restrictTicketToolContext },
      '../state': { store }, './audit': { requireSecurityRole() {}, auditEvent: async () => {} },
      './room-worktree': { ensureRoomWorktree: () => workspace },
      '../../../shared/ticket-dependencies': { requireTicketDependenciesReady },
      '../../../shared/bot-policy': botPolicy,
      '../../agent/bot-tool-scope': { scopeBotToolContext: (_bot: any, base: any) => base },
      './ticket-bot': { captureTicketBot: () => undefined, assertTicketBotReady: () => {},
        sameTicketBot: (previous: any, next: any) => (!previous || !next ? previous === next : true) },
      './workspace': { resolveWorkspaceCwd: async (cwd: string) => cwd, inspectGitRepository: async () => ({ branch: 'fixture' }) },
      './ticket-helpers': { summarizeTicketWorkerOutcome: () => ({ text: 'Synthetic result', evidence: [],
        codeChanged: false, testsPassed: false, signals: { commands: [], blockers: [], files: [], diffs: [] } }) },
      './workspace-mutation-guard': { lexicalLocalWorkspaceKey },
      './renderer-events': { sendToRenderer() {} },
      '../../../shared/reasoning-effort': reasoning,
      './provider-runtime': {
        ticketWorkOptionsWithProjectRoute: (_project: any, opts: any) => opts,
        resolveProjectProvider: () => ({ baseProvider: provider, provider, providerAccountId: null, model: 'fixture' }),
        runtimeProviderWithFreshModels: async () => provider,
        modelRouteAccountLabel: () => 'Fixture', modelRouteAutoFailoverEnabled: () => true,
        modelRouteFailoverText: () => 'Synthetic recovery', modelRouteResetPolicy: () => 'rotate_on_limit',
        modelRouter: {
          select: (params: any) => {
            routeEfforts.push(params.reasoningEffort)
            return { selected: true, selectedAccountId: null, audit: { action: 'keep', reason: 'Fixture' } }
          },
          routeAfterError: (params: any) => {
            routeEfforts.push(params.reasoningEffort)
            return { signal: { resetAt: 1, kind: 'rate_limit' }, audit: [],
              route: { selected: true, selectedAccountId: 'recovery', audit: { reason: 'Fixture recovery' } } }
          },
        },
      },
    }
    const module = { exports: {} as any }
    new Function('require', 'module', 'exports', code)((id: string) => {
      assert.ok(id in imports, `Reviewed worker import ${id}`)
      return imports[id]
    }, module, module.exports)
    const noop = () => {}
    const runtime = module.exports.createTicketWorkerRuntime({
      persistThreads: noop, persistTickets: noop, emitTicket: noop, appendRoomSupervisorEvent: noop,
      claimJob: () => ({ id: 'job', status: 'running' }), patchJob: (job: any) => job,
      fullToolContext: () => ({}), drainThreadCues: () => [], waitForThreadCue: async () => {},
      approvalResolver: () => async () => 'deny', timelineErrorSince: () => null,
      timelineError: () => new Error('Unexpected timeline error'), isHighRiskTool: () => false,
      emitToRenderer: noop, addThreadNotice: noop, auditModelRoute: async () => {},
      markRouteErrorRecovered: noop, injectThreadRouteHandoff: () => '', aborters: new Map(),
      workspaceMutationGuard: { assertThreadStartAllowed: noop,
        reserveStart: () => ({ addWorkspaceKeys: noop, addThreadId: noop, release: noop }) },
      applyEventToThread: noop, activeJob: () => null, finishJob: () => finish(),
      markUndrainedCues: noop, effectiveContextLimit: () => 16_384,
    })
    const pending = runtime.startTicketWork({ ticketId: ticket.id, source: { roomId: 'room', seat: 'Builder', reasoningEffort: sourceEffort } })
    assert.equal(requests.length, 0, 'workspace preparation genuinely pauses the production launch')
    seat.reasoningEffort = 'medium'
    store.settings.reasoningEffort = 'medium'
    releaseWorkspace('/workspace/room')
    await pending
    const captured = ticket.execution?.reasoningEffort
    releaseAgent()
    let timer: ReturnType<typeof setTimeout> | undefined
    try {
      await Promise.race([done, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error('Worker did not settle')), 2_000)
      })])
    } finally { clearTimeout(timer) }
    assert.equal(requests.length, recover ? 2 : 1, 'the actual worker executes initial and recovery routes')
    const effort = sourceEffort ?? configured ?? global
    observed.push({ global, configured, recover, routes: routeEfforts,
      requests: requests.map(request => request.reasoningEffort), captured, settled: ticket.execution?.reasoningEffort })
    expected.push({ global, configured, recover, routes: Array(recover ? 2 : 1).fill(effort),
      requests: Array(recover ? 2 : 1).fill(effort), captured: effort, settled: effort })
  }
}
assert.deepEqual(observed, expected, 'Room workers snapshot effort before awaits and retain it through recovery and settlement')
console.log('Room worker effort checks passed: opposite defaults, legacy inheritance, delayed startup, recovery and execution records.')
