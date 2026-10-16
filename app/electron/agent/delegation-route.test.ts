// Run: npx jiti electron/agent/delegation-route.test.ts
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { resolveDelegationRoute } from './delegation-route.ts'
import { executeTool } from './tools.ts'
import type { Provider, Seat, SubAgent } from './types.ts'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

const providers = [
  { id: 'p-local', name: 'Local', baseUrl: 'http://localhost:11434/v1', apiKey: '', models: ['llama3'], kind: 'local' },
  {
    id: 'p-cloud', name: 'Cloud', baseUrl: 'https://api.example.com/v1', apiKey: 'k', models: ['gpt-5', 'gpt-5-mini'],
    accounts: [{ id: 'acc-2', name: 'Work', kind: 'api-key', enabled: true, models: ['gpt-5-mini'], createdAt: 0, updatedAt: 0 }],
  },
] as unknown as Provider[]

const seats = [
  { id: 's1', name: 'Planner', role: 'planner', providerId: 'p-local', model: 'llama3' },
  { id: 's2', name: 'Reviewer', role: 'reviewer', providerId: 'p-cloud', providerAccountId: 'acc-2', model: 'gpt-5-mini' },
] as Seat[]

const base = { providerId: 'p-local', providerAccountId: undefined, model: 'llama3' }

function stubAgent(input: any): SubAgent {
  return {
    id: 'sub-1',
    parent: { kind: 'thread', id: 'thread-1', label: 'Parent thread', threadId: 'thread-1' },
    name: input.name || 'Thread sub-agent',
    task: input.task,
    status: 'succeeded',
    projectId: null,
    roomId: null,
    threadId: 'thread-1',
    seatId: null,
    providerId: input.providerId ?? 'p-local',
    providerAccountId: input.accountId ?? null,
    model: input.model ?? 'llama3',
    runId: 'run-sub-1',
    result: 'done',
    error: null,
    events: [],
    messages: [],
    createdAt: 1,
    updatedAt: 2,
    startedAt: 1,
    finishedAt: 2,
  }
}

async function run() {
  console.log('delegation route tests:')

  // --- Inherit: no override fields returns the caller's route untouched ---------------
  {
    const routed = resolveDelegationRoute(providers, base, {}, seats)
    assert.equal(routed.error, undefined)
    assert.deepEqual(routed.route, base)
    assert.equal(routed.route!.providerAccountId, undefined, 'undefined account (active/default fallback) must survive inherit')
    const nullOnly = resolveDelegationRoute(providers, base, { targetSeat: null, providerId: null, accountId: null, model: null }, seats)
    assert.deepEqual(nullOnly.route, base, 'explicit null fields also inherit')
    ok('absent or null override fields inherit the caller route exactly')
  }

  // --- Cross-model: targetSeat adopts the target seat's provider/account/model --------
  {
    const routed = resolveDelegationRoute(providers, base, { targetSeat: 'Reviewer' }, seats)
    assert.equal(routed.error, undefined)
    assert.deepEqual(routed.route, { providerId: 'p-cloud', providerAccountId: 'acc-2', model: 'gpt-5-mini' })
    ok('targetSeat routes the sub-agent to the target seat provider/account/model')
  }

  // --- Explicit providerId/accountId/model overrides ----------------------------------
  {
    const routed = resolveDelegationRoute(providers, base, { providerId: 'p-cloud', model: 'gpt-5' }, seats)
    assert.deepEqual(routed.route, { providerId: 'p-cloud', model: 'gpt-5' }, 'provider switch without account resets the account')
    const scoped = resolveDelegationRoute(providers, base, { providerId: 'p-cloud', accountId: 'acc-2' }, seats)
    assert.deepEqual(scoped.route, { providerId: 'p-cloud', providerAccountId: 'acc-2', model: 'gpt-5-mini' }, 'model resolves against the account model list')
    const targetPlusModel = resolveDelegationRoute(providers, base, { targetSeat: 'Reviewer', model: 'gpt-5-mini' }, seats)
    assert.equal(targetPlusModel.route!.model, 'gpt-5-mini')
    ok('explicit providerId/accountId/model override the inherited or seat route')
  }

  // --- Model-only override keeps legacy fallback semantics ------------------------------
  {
    const cloudBase = { providerId: 'p-cloud', providerAccountId: undefined, model: 'gpt-5' }
    const valid = resolveDelegationRoute(providers, cloudBase, { model: 'gpt-5-mini' }, seats)
    assert.equal(valid.route!.model, 'gpt-5-mini')
    const invalid = resolveDelegationRoute(providers, cloudBase, { model: 'no-such-model' }, seats)
    assert.equal(invalid.route!.model, 'gpt-5', 'unknown model on the same provider falls back to the caller model')
    ok('model-only override adopts valid models and falls back otherwise')
  }

  // --- Clean errors for unknown seat / provider / account -----------------------------
  {
    const ghost = resolveDelegationRoute(providers, base, { targetSeat: 'Ghost' }, seats)
    assert.match(ghost.error!, /unknown seat "Ghost"/)
    const noProvider = resolveDelegationRoute(providers, base, { providerId: 'p-nope' }, seats)
    assert.match(noProvider.error!, /unknown provider "p-nope"/)
    const noAccount = resolveDelegationRoute(providers, base, { accountId: 'acc-nope' }, seats)
    assert.match(noAccount.error!, /unknown account "acc-nope"/)
    ok('unknown seat, provider, or account produce clean errors')
  }

  // --- localOnly: a routed cloud provider is rejected ----------------------------------
  {
    const blocked = resolveDelegationRoute(providers, base, { providerId: 'p-cloud' }, seats, true)
    assert.equal(blocked.route, undefined)
    assert.match(blocked.error!, /must be a local provider for this local-only project/)
    const blockedSeat = resolveDelegationRoute(providers, base, { targetSeat: 'Reviewer' }, seats, true)
    assert.match(blockedSeat.error!, /must be a local provider/, 'targetSeat cannot route a local-only project to a cloud seat')
    const allowed = resolveDelegationRoute(providers, base, { providerId: 'p-local', model: 'llama3' }, seats, true)
    assert.equal(allowed.error, undefined, 'a local provider route still resolves for a local-only project')
    const noFlag = resolveDelegationRoute(providers, base, { providerId: 'p-cloud' }, seats)
    assert.equal(noFlag.error, undefined, 'without the localOnly flag the same route resolves')
    ok('localOnly rejects any routed non-local provider on either override path')
  }

  // --- Same-provider override keeps the inherited account ------------------------------
  {
    const cloudBase = { providerId: 'p-cloud', providerAccountId: 'acc-2' as string | null, model: 'gpt-5-mini' }
    const same = resolveDelegationRoute(providers, cloudBase, { providerId: 'p-cloud' }, seats)
    assert.equal(same.route!.providerAccountId, 'acc-2', 'same-provider override must not reset the account')
    const cross = resolveDelegationRoute(providers, cloudBase, { providerId: 'p-local' }, seats)
    assert.equal(cross.route!.providerAccountId, undefined, 'cross-provider switch still resets the account')
    ok('same-provider override preserves the account; cross-provider reset is unchanged')
  }

  // --- Silent model fallback surfaces a notice -----------------------------------------
  {
    const downgraded = resolveDelegationRoute(providers, base, { providerId: 'p-cloud', model: 'no-such-model' }, seats)
    assert.equal(downgraded.route!.model, 'gpt-5', 'falls back to the provider first choice')
    assert.match(downgraded.notice!, /Model fallback/, 'the downgrade is surfaced as a notice')
    assert.match(downgraded.notice!, /gpt-5/)
    const exact = resolveDelegationRoute(providers, base, { providerId: 'p-cloud', model: 'gpt-5' }, seats)
    assert.equal(exact.notice, undefined, 'no notice when the requested model is honored')
    const inherited = resolveDelegationRoute(providers, base, { providerId: 'p-local', model: 'llama3' }, seats)
    assert.equal(inherited.notice, undefined, 'no notice when the inherited model is honored')
    ok('model fallback to choices[0] carries a notice; honored models do not')
  }

  // --- launch_subagent passes the override through ToolContext -------------------------
  {
    const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-delegation-'))
    const seen: any[] = []
    const ctx = {
      cwd,
      launchSubAgent: async (input: any) => {
        seen.push(input)
        return { output: `launched: ${input.task}`, agent: stubAgent(input) }
      },
    }
    const single = await executeTool('launch_subagent', { task: 'Inspect tests', providerId: 'p-cloud', accountId: 'acc-2', model: 'gpt-5' }, ctx)
    assert.equal(single.ok, true)
    assert.equal(seen[0].providerId, 'p-cloud')
    assert.equal(seen[0].accountId, 'acc-2')
    assert.equal(seen[0].model, 'gpt-5')
    const batch = await executeTool('launch_subagents', { tasks: [
      { task: 'Lane A', providerId: 'p-cloud', accountId: 'acc-2', model: 'gpt-5-mini' },
      { task: 'Lane B' },
    ] }, ctx)
    assert.equal(batch.ok, true)
    assert.equal(seen[1].providerId, 'p-cloud')
    assert.equal(seen[1].accountId, 'acc-2')
    assert.equal(seen[1].model, 'gpt-5-mini')
    assert.equal(seen[2].providerId, null, 'omitted override stays null (inherit)')
    assert.equal(seen[2].accountId, null)
    ok('launch_subagent and launch_subagents pass providerId/accountId/model through to the launcher')
  }

  // --- Source pins: both spawn paths route through the resolver ------------------------
  {
    const source = (rel: string) => readFileSync(fileURLToPath(new URL(rel, import.meta.url)), 'utf8')
    const roomIpc = source('../main/host/workspace-runtime-ipc.ts')
    const delegateStart = roomIpc.indexOf('delegate: async (args, seat)')
    const delegateEnd = roomIpc.indexOf('read_file: async', delegateStart)
    assert.ok(delegateStart >= 0 && delegateEnd > delegateStart, 'room delegate tool remains discoverable')
    const delegate = roomIpc.slice(delegateStart, delegateEnd)
    assert.match(delegate, /resolveDelegationRoute\(store\.settings\.providers/, 'room delegate resolves the cross-model route')
    assert.match(delegate, /!routed\.route\) return `Cannot delegate:/, 'route errors surface as tool errors, not crashes')
    assert.match(delegate, /runReadOnlySubAgent\(created\.agent, prov, routed\.route\.model/, 'room delegate runs the sub-agent on the routed model')
    assert.match(delegate, /parent: parentForRoomSeat\(room, s\)/, 'delegated sub-agent lineage stays under the delegating seat')

    const threadRuntime = source('../main/host/tool-context-runtime.ts')
    assert.match(threadRuntime, /resolveDelegationRoute\(store\.settings\.providers/, 'thread sub-agent launcher resolves the cross-model route')
    assert.match(threadRuntime, /runReadOnlySubAgent\(created\.agent, childProvider, childModel/, 'thread sub-agent runs on the routed provider/model')
    assert.match(threadRuntime, /approvalMode: 'readonly'/, 'sub-agents stay read-only')
    ok('room delegate and thread launcher both bind provider/account/model through the resolver')
  }

  console.log(`${passed} delegation route tests passed`)
}

run().catch((error) => {
  console.error(error)
  process.exit(1)
})
