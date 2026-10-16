import assert from 'node:assert/strict'
import { defaultSettings } from '../../store'
import { resolveJobApproval } from '../../agent/jobs'
import { createMission, requestMissionApproval } from '../../agent/missions'
import { store } from '../state'
import { createJobControlRuntime } from './job-control-runtime'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'

const saved = { settings: store.settings, jobs: store.jobs, threads: store.threads, projects: store.projects }
try {
  store.settings = { ...defaultSettings(), providers: [], activeProviderId: 'none', activeModel: 'none' }
  store.projects = []
  const thread = (id: string) => ({ id, cwd: '/tmp', timeline: [{ id: `item-${id}`, kind: 'tool', callId: 'legacy-shared', tool: 'write_file', args: { path: 'x', content: id } }], apiMessages: [{ role: 'assistant', content: null, tool_calls: [{ id: 'legacy-shared', type: 'function', function: { name: 'write_file', arguments: '{}' } }] }] } as any)
  const first = thread('first'), second = thread('second')
  store.threads = [first, second]
  store.jobs = [first, second].map((t) => ({ id: `job-${t.id}`, kind: 'thread', owner: t.id, threadId: t.id, currentThreadId: t.id, status: 'waiting_approval', pendingApprovals: [{ id: `pending-${t.id}`, callId: 'legacy-shared', threadId: t.id, itemId: `item-${t.id}`, tool: 'write_file', resolvedAt: null, policy: { key: 'write:file', label: 'Write file', explanation: 'One reviewed fixture write.' } }] } as any))
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  let mission = requestMissionApproval(createMission({ title: 'Legacy mission', objective: 'Review two synthetic workers' } as any), { callId: 'legacy-shared', title: 'Legacy approval', summary: 'Choose the intended worker' })
  let executions = 0
  let holdExecution = false
  let release!: () => void
  const held = new Promise<void>((resolve) => { release = resolve })
  let entered!: () => void
  const didEnter = new Promise<void>((resolve) => { entered = resolve })
  const pendingApprovals = new Map<string, (decision: string) => void>()
  const pendingApprovalMeta = new Map<string, any>()
  const runtime = createJobControlRuntime({
    handle: (name: string, fn: any) => handlers.set(name, fn), getMission: () => mission, replaceMission: (next: typeof mission) => { mission = next; return next }, missionApprovalResolvers: new Map(), aborters: new Map(), workspaceMutationGuard: new WorkspaceMutationGuard(),
    pendingApprovals, pendingApprovalMeta, fullToolContext: () => ({}), mcpToolContext: () => ({}),
    executeTool: async () => { executions++; if (holdExecution) { entered(); await held } return { ok: true, output: 'Synthetic reviewed write' } },
    isHighRiskTool: () => false, emitToRenderer: () => {}, persistThreads: () => {},
    resolvePersistedJobApproval: (id: string, decision: any) => { const result = resolveJobApproval(store.jobs, id, decision); store.jobs = result.jobs; return null },
  } as any)
  await assert.rejects(() => runtime.resolveDurableApprovalAfterRestart('legacy-shared', 'allow'), /ambiguous/i, 'An old globally shared provider ID cannot select one worker')
  assert.equal(executions, 0)
  assert.equal(first.apiMessages.length, 1)
  assert.equal(second.apiMessages.length, 1)
  assert.throws(() => resolveJobApproval(store.jobs, 'legacy-shared', 'deny'), /ambiguous/i, 'Pure durable settlement also refuses ambiguous old IDs')
  await assert.rejects(() => handlers.get('missions:approval:resolve')!(null, { id: mission.id, approvalId: mission.pendingApprovals[0].id, decision: 'deny' }), /ambiguous/i)
  assert.equal(mission.pendingApprovals.length, 1, 'An ambiguous restored tool decision cannot mark the mission approval resolved')
  await runtime.resolveToolApproval('item-second', 'deny')
  assert.equal(first.apiMessages.length, 1, 'Selecting second timeline item cannot settle first worker')
  assert.equal(second.apiMessages.at(-1).tool_call_id, 'legacy-shared', 'Model protocol response retains the old raw ID')
  assert.equal(store.jobs.find((j) => j.id === 'job-first')!.pendingApprovals[0].resolvedAt, null)
  assert.ok(store.jobs.find((j) => j.id === 'job-second')!.pendingApprovals[0].resolvedAt)
  holdExecution = true
  const allowing = runtime.resolveToolApproval('pending-first', 'allow')
  await didEnter
  await assert.rejects(() => runtime.resolveDurableApprovalAfterRestart('item-first', 'deny'), /already being resolved/, 'Two aliases for one restored approval share the same operation guard')
  release(); await allowing
  assert.equal(executions, 1)
  assert.equal(first.apiMessages.at(-1).tool_call_id, 'legacy-shared')
  assert.ok(store.jobs.find((j) => j.id === 'job-first')!.pendingApprovals[0].resolvedAt)
  const third = thread('third'), fourth = thread('fourth')
  store.threads.push(third, fourth)
  store.jobs.push(...[third, fourth].map((t) => ({ id: `job-${t.id}`, kind: 'thread', owner: t.id, threadId: t.id, currentThreadId: t.id, status: 'waiting_approval', pendingApprovals: [{ id: `pending-${t.id}`, callId: 'legacy-shared', threadId: t.id, itemId: `item-${t.id}`, tool: 'write_file', resolvedAt: null, policy: { key: 'write:file', label: 'Write file', explanation: 'One reviewed fixture write.' } }] } as any)))
  mission = requestMissionApproval(createMission({ title: 'Linked legacy mission', objective: 'Review the fourth worker', threadId: fourth.id } as any), { callId: 'legacy-shared', title: 'Legacy approval', summary: 'Choose linked worker' })
  await handlers.get('missions:approval:resolve')!(null, { id: mission.id, approvalId: mission.pendingApprovals[0].id, decision: 'deny' })
  assert.equal(mission.pendingApprovals.length, 0, 'A uniquely linked legacy mission remains resolvable')
  assert.equal(third.apiMessages.length, 1, 'Mission ownership excludes another worker reusing the raw ID')
  assert.equal(fourth.apiMessages.at(-1).tool_call_id, 'legacy-shared')
  assert.equal(store.jobs.find((j) => j.id === 'job-third')!.pendingApprovals[0].resolvedAt, null)
  let liveDecision: string | null = null
  pendingApprovals.set('external-live-request', (decision) => { liveDecision = decision })
  pendingApprovalMeta.set('external-live-request', { threadId: first.id, itemId: 'live-item', tool: 'write_file', policy: { key: 'write:file', label: 'Write' } })
  await runtime.resolveToolApproval('live-item', 'deny')
  assert.equal(liveDecision, 'deny', 'A stable timeline address also resolves its uniquely owned live external request')
  assert.equal(pendingApprovals.size, 0)
  console.log('Legacy restored approval ambiguity, stable addresses, wire IDs, alias race and live compatibility pass')
} finally { Object.assign(store, saved) }
