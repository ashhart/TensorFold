// Run: jiti src/lib/execution-target.test.ts
import assert from 'node:assert/strict'
import { resolveExecutionTarget, executionTargetLabel } from './execution-target'
import type { RemoteOrchestratorProfile } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('execution-target tests:')

const now = 1_700_000_000_000
const host = (id: string, extra: Partial<RemoteOrchestratorProfile> = {}): RemoteOrchestratorProfile => ({
  id, name: id === 'mini' ? 'Mac mini' : id, serverUrl: `http://127.0.0.1:9${id.length}00`, authToken: 'tok',
  enabled: true, lastConnectedAt: null, lastError: null, createdAt: now, updatedAt: now, ...extra,
} as RemoteOrchestratorProfile)

const settingsWith = (extra: Record<string, unknown> = {}) => ({
  remoteOrchestrators: [host('mini'), host('studio')],
  defaultExecutionTarget: undefined,
  defaultOrchestratorId: undefined,
  ...extra,
}) as any

// 1. Default: everything local when no mode is set.
{
  const decision = resolveExecutionTarget(settingsWith(), null)
  assert.deepEqual(decision, { hostId: null, source: 'local', unavailableReason: null })
  assert.equal(executionTargetLabel(decision, []), null, 'no chip label when local by default')
  ok('no mode, no project → local')
}

// 2. Global orchestrator mode routes project-less work (chats, quick tasks) to the default host.
{
  const decision = resolveExecutionTarget(settingsWith({ defaultExecutionTarget: 'orchestrator', defaultOrchestratorId: 'mini' }), null)
  assert.equal(decision.hostId, 'mini')
  assert.equal(decision.source, 'global')
  assert.equal(executionTargetLabel(decision, [host('mini')]), 'Mac mini · all new work', 'the chip says global mode drove it')
  ok('global mode, no project → default host, labeled as all-new-work')
}

// 3. Global mode without a chosen host falls back to the first enabled host.
{
  const decision = resolveExecutionTarget(settingsWith({ defaultExecutionTarget: 'orchestrator' }), null)
  assert.equal(decision.hostId, 'mini', 'first enabled host when none chosen')
  ok('global mode without an explicit host → first enabled host')
}

// 4. A project pinned to its own orchestrator keeps ITS host even under global mode.
{
  const decision = resolveExecutionTarget(
    settingsWith({ defaultExecutionTarget: 'orchestrator', defaultOrchestratorId: 'mini' }),
    { executionTarget: 'orchestrator', orchestratorId: 'studio', localOnly: false },
  )
  assert.equal(decision.hostId, 'studio', 'the project pin wins over the global default host')
  assert.equal(decision.source, 'project')
  ok('project orchestrator pin beats the global default host')
}

// 5. localOnly is a hard local pin — global mode never overrides it.
{
  const decision = resolveExecutionTarget(
    settingsWith({ defaultExecutionTarget: 'orchestrator', defaultOrchestratorId: 'mini' }),
    { executionTarget: 'local', orchestratorId: null, localOnly: true },
  )
  assert.equal(decision.hostId, null)
  assert.equal(decision.source, 'local_only')
  ok('localOnly project stays local under global mode')
}

// 6. A plain local project (the stored default) DOES follow global mode.
{
  const decision = resolveExecutionTarget(
    settingsWith({ defaultExecutionTarget: 'orchestrator', defaultOrchestratorId: 'mini' }),
    { executionTarget: 'local', orchestratorId: null, localOnly: false },
  )
  assert.equal(decision.hostId, 'mini', 'the stored-default local target follows the mode switch')
  assert.equal(decision.source, 'global')
  ok('non-pinned project follows global mode')
}

// 7. Unusable hosts surface a reason instead of silently running local.
{
  const disabled = settingsWith({ defaultExecutionTarget: 'orchestrator', remoteOrchestrators: [host('mini', { enabled: false })] })
  const globalMiss = resolveExecutionTarget(disabled, null)
  assert.equal(globalMiss.hostId, null)
  assert.ok(globalMiss.unavailableReason?.includes('no enabled host'), 'global mode without a usable host says so')
  assert.equal(executionTargetLabel(globalMiss, []), 'Orchestrator unavailable')

  const projectMiss = resolveExecutionTarget(settingsWith({ remoteOrchestrators: [] }), { executionTarget: 'orchestrator', orchestratorId: 'gone', localOnly: false })
  assert.equal(projectMiss.source, 'project')
  assert.ok(projectMiss.unavailableReason?.includes('no enabled host'), 'a dead project pin says so')
  ok('unusable hosts produce visible reasons, never a silent local fallback')
}

// 8. A disabled project host does not fall through to an unrelated enabled host silently…
//    it resolves to the first enabled host ONLY via the explicit orchestratorId-missing path.
{
  const decision = resolveExecutionTarget(
    settingsWith({ remoteOrchestrators: [host('mini'), host('studio', { enabled: false })] }),
    { executionTarget: 'orchestrator', orchestratorId: 'studio', localOnly: false },
  )
  assert.equal(decision.hostId, null, 'a specifically-pinned but disabled host is unavailable, not swapped')
  assert.ok(decision.unavailableReason)
  ok('a pinned-but-disabled host reports unavailable rather than swapping hosts')
}

console.log(`\n${passed} passed`)
