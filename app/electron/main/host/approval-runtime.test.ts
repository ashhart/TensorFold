import assert from 'node:assert/strict'
import { createApprovalRuntime } from './approval-runtime'
import { registerMissionApprovalResolver } from './mission-runtime'
import type { ApprovalPolicyScope } from '../../agent/approval-policy'

// Minimal mock deps. jobHandle is left undefined so recordJobApproval/resolveJobApproval
// short-circuit (job === null), keeping the test focused on the Map + abort-listener logic.
function makeDeps() {
  const pendingApprovals = new Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>()
  const pendingApprovalMeta = new Map<string, { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }>()
  return {
    pendingApprovals,
    pendingApprovalMeta,
    emitJob() {},
    emitToRenderer() {},
    findApprovalRule: async (..._args: any[]): Promise<{ decision: 'allow' | 'deny' } | null> => null, // no saved rule -> manual path
    patchJob: (job: any) => job,
    persistJobs() {},
    persistThreads() {},
  }
}

const policy: ApprovalPolicyScope = { scope: 'tool', key: 'run_command', label: 'l', explanation: 'e' } as ApprovalPolicyScope

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => { resolve = res; reject = rej })
  return { promise, resolve, reject }
}

async function testSavedRuleWaitsForDurableConsumption() {
  const deps = makeDeps()
  const disk = deferred<{ decision: 'allow' } | null>()
  deps.findApprovalRule = () => disk.promise
  const controller = new AbortController()
  const resolver = createApprovalRuntime(deps).approvalResolver(controller)
  deps.pendingApprovalMeta.set('durable-rule', { threadId: 't', itemId: 'i', tool: 'run_command', policy })

  let settled = false
  const result = resolver('durable-rule').then((decision) => { settled = true; return decision })
  await Promise.resolve()
  assert.equal(settled, false, 'saved allow is not applied while durable rule consumption is pending')
  assert.equal(deps.pendingApprovals.has('durable-rule'), false, 'manual resolver is not exposed during the durable lookup')
  disk.resolve({ decision: 'allow' })
  assert.equal(await result, 'allow')
  assert.equal(deps.pendingApprovalMeta.has('durable-rule'), false)
  console.log('ok - saved approval waits for durable bounded-rule consumption')
}

async function testAbortDuringSavedRuleLookupFailsClosed() {
  const deps = makeDeps()
  const disk = deferred<{ decision: 'allow' } | null>()
  deps.findApprovalRule = () => disk.promise
  const controller = new AbortController()
  const resolver = createApprovalRuntime(deps).approvalResolver(controller)
  deps.pendingApprovalMeta.set('abort-lookup', { threadId: 't', itemId: 'i', tool: 'run_command', policy })

  const result = resolver('abort-lookup')
  await Promise.resolve()
  controller.abort()
  disk.resolve({ decision: 'allow' })
  assert.equal(await result, 'deny', 'a durably consumed rule is not applied after cancellation')
  assert.equal(deps.pendingApprovals.has('abort-lookup'), false)
  assert.equal(deps.pendingApprovalMeta.has('abort-lookup'), false)
  console.log('ok - abort during asynchronous saved-rule lookup denies without parking')
}

async function testSavedRuleLookupFailureFallsBackToManualApproval() {
  const deps = makeDeps()
  deps.findApprovalRule = async () => { throw new Error('settings unavailable') }
  const controller = new AbortController()
  const resolver = createApprovalRuntime(deps).approvalResolver(controller)
  deps.pendingApprovalMeta.set('lookup-error', { threadId: 't', itemId: 'i', tool: 'run_command', policy })

  const result = resolver('lookup-error')
  await Promise.resolve()
  await Promise.resolve()
  assert.ok(deps.pendingApprovals.get('lookup-error'), 'lookup failure leaves a real manual gate')
  deps.pendingApprovals.get('lookup-error')!('deny')
  assert.equal(await result, 'deny')
  console.log('ok - saved-rule lookup failure degrades to manual approval')
}

async function testAbortCleansBothMaps() {
  const deps = makeDeps()
  const controller = new AbortController()
  const { approvalResolver } = createApprovalRuntime(deps)
  const resolver = approvalResolver(controller, undefined)

  deps.pendingApprovalMeta.set('call-1', { threadId: 't', itemId: 'i', tool: 'run_command', policy })
  const p = resolver('call-1')
  await Promise.resolve()
  assert.equal(deps.pendingApprovals.has('call-1'), true, 'pendingApprovals registered')

  controller.abort()
  const decision = await p
  assert.equal(decision, 'deny', 'abort resolves to deny')
  assert.equal(deps.pendingApprovals.has('call-1'), false, 'pendingApprovals cleared on abort')
  // The bug: pendingApprovalMeta was previously leaked on abort.
  assert.equal(deps.pendingApprovalMeta.has('call-1'), false, 'pendingApprovalMeta cleared on abort (no leak)')
  console.log('ok - abort cleans both pendingApprovals and pendingApprovalMeta')
}

async function testNormalDoneRemovesAbortListener() {
  const deps = makeDeps()
  const controller = new AbortController()
  const { approvalResolver } = createApprovalRuntime(deps)
  const resolver = approvalResolver(controller, undefined)

  // Request many approvals on the SAME controller, resolving each normally.
  const decisions: Array<'allow' | 'allow_always' | 'deny'> = []
  for (let n = 0; n < 5; n++) {
    const callId = `call-${n}`
    deps.pendingApprovalMeta.set(callId, { threadId: 't', itemId: 'i', tool: 'run_command', policy })
    const p = resolver(callId)
    await Promise.resolve()
    const done = deps.pendingApprovals.get(callId)
    assert.ok(done, `done handler registered for ${callId}`)
    done!('allow') // normal resolution -> should removeEventListener('abort', onAbort)
    decisions.push(await p)
    assert.equal(deps.pendingApprovalMeta.has(callId), false, `pendingApprovalMeta cleared for ${callId}`)
  }
  assert.deepEqual(decisions, ['allow', 'allow', 'allow', 'allow', 'allow'])
  assert.equal(deps.pendingApprovals.size, 0, 'all done handlers removed after normal resolution')

  // After all approvals resolved normally, every per-approval abort listener should have been
  // torn down. If a stale listener survived, aborting would try to delete already-gone callIds
  // (still harmless) — but more importantly the resolved promises must not re-resolve. We assert
  // the abort fires without re-touching the (empty) maps in a way that throws.
  let aborted = false
  controller.signal.addEventListener('abort', () => { aborted = true })
  controller.abort()
  assert.equal(aborted, true, 'controller aborts cleanly')
  assert.equal(deps.pendingApprovals.size, 0, 'maps stay clean after abort with no outstanding approvals')
  console.log('ok - normal resolution removes its abort listener; no accumulation')
}

async function testListenerCountDoesNotAccumulate() {
  // Directly observe listener accumulation via a spy on add/removeEventListener.
  const deps = makeDeps()
  const controller = new AbortController()
  const { approvalResolver } = createApprovalRuntime(deps)
  const resolver = approvalResolver(controller, undefined)

  let added = 0
  let removed = 0
  const realAdd = controller.signal.addEventListener.bind(controller.signal)
  const realRemove = controller.signal.removeEventListener.bind(controller.signal)
  ;(controller.signal as any).addEventListener = (...args: any[]) => { if (args[0] === 'abort') added++; return (realAdd as any)(...args) }
  ;(controller.signal as any).removeEventListener = (...args: any[]) => { if (args[0] === 'abort') removed++; return (realRemove as any)(...args) }

  for (let n = 0; n < 4; n++) {
    const callId = `c-${n}`
    deps.pendingApprovalMeta.set(callId, { threadId: 't', itemId: 'i', tool: 'run_command', policy })
    const p = resolver(callId)
    await Promise.resolve()
    deps.pendingApprovals.get(callId)!('allow')
    await p
  }
  assert.equal(added, 4, 'one abort listener added per approval')
  assert.equal(removed, 4, 'each abort listener removed on normal resolution (net live listeners = 0)')
  console.log('ok - abort listeners net to zero after normal resolutions')
}

async function testMissionApprovalBridgeCleansLikeThreadApprovals() {
  const controller = new AbortController()
  const resolvers = new Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>()
  const meta = new Map<string, unknown>([['mission-call', { threadId: 't', itemId: 'i' }]])
  const decisions: string[] = []

  let added = 0
  let removed = 0
  const realAdd = controller.signal.addEventListener.bind(controller.signal)
  const realRemove = controller.signal.removeEventListener.bind(controller.signal)
  ;(controller.signal as any).addEventListener = (...args: any[]) => { if (args[0] === 'abort') added++; return (realAdd as any)(...args) }
  ;(controller.signal as any).removeEventListener = (...args: any[]) => { if (args[0] === 'abort') removed++; return (realRemove as any)(...args) }

  registerMissionApprovalResolver('mission-call', controller, resolvers, meta, (decision) => { decisions.push(decision) })
  assert.equal(resolvers.has('mission-call'), true, 'mission resolver registered')
  resolvers.get('mission-call')!('allow')

  assert.deepEqual(decisions, ['allow'])
  assert.equal(resolvers.has('mission-call'), false, 'mission resolver cleared on normal resolution')
  assert.equal(meta.has('mission-call'), false, 'mission approval meta cleared on normal resolution')
  assert.equal(added, 1)
  assert.equal(removed, 1, 'mission abort listener removed on normal resolution')
  controller.abort()
  assert.deepEqual(decisions, ['allow'], 'later abort does not re-resolve a settled mission approval')

  const abortController = new AbortController()
  const abortResolvers = new Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>()
  const abortMeta = new Map<string, unknown>([['mission-abort', { threadId: 't', itemId: 'i' }]])
  const abortDecisions: string[] = []
  registerMissionApprovalResolver('mission-abort', abortController, abortResolvers, abortMeta, (decision) => { abortDecisions.push(decision) })
  abortController.abort()
  assert.deepEqual(abortDecisions, ['deny'])
  assert.equal(abortResolvers.has('mission-abort'), false, 'mission resolver cleared on abort')
  assert.equal(abortMeta.has('mission-abort'), false, 'mission approval meta cleared on abort')
  console.log('ok - mission approval bridge clears resolver/meta and removes abort listener')
}

async function main() {
  await testSavedRuleWaitsForDurableConsumption()
  await testAbortDuringSavedRuleLookupFailsClosed()
  await testSavedRuleLookupFailureFallsBackToManualApproval()
  await testAbortCleansBothMaps()
  await testNormalDoneRemovesAbortListener()
  await testListenerCountDoesNotAccumulate()
  await testMissionApprovalBridgeCleansLikeThreadApprovals()
  console.log('all approval-runtime tests passed')
}

main().catch((err) => { console.error(err); process.exit(1) })
